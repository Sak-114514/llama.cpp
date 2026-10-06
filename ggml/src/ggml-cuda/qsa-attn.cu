// Block-sparse attention over a Q8_0 K/V cache (QSA): every query attends to its own list of
// ratio-cell blocks, read straight from the cache. No gathered K/V copy and no cache-sized mask,
// so the cost follows the selected budget instead of the context length; the same kernel serves
// decode (a few queries, split over many blocks) and prompt batches.
//
// Expressed as a GGML_OP_FLASH_ATTN_EXT node with src[5] = block ids (see ggml_cuda_qsa_attn_is_sparse):
//   src[0] q     F32  [D, n_head, n_tok]
//   src[1] k     Q8_0 [D, n_head_kv, n_cells]   (a cache view; cell = block*ratio + c)
//   src[2] v     Q8_0 [D, n_head_kv, n_cells]   (Q4_0 also, with LLAMA_QSA_Q4V=1)
//   src[3] mask  F16  [n_wblk*ratio, 1, 1, n_tok]  0 or -inf per selected cell
//   src[5] ids   I32  [n_wblk, n_tok]
//   dst          F32  [D, n_head, n_tok]
//   op_params: [0] scale, [6] ratio

#include "qsa-attn.cuh"
#include "common.cuh"

#include <cstdlib>

#define QSA_ATTN_D     256
#define QSA_ATTN_R     4

static constexpr int qsa_dpl = QSA_ATTN_D / WARP_SIZE; // dims per lane: 8

// raw Q8_0 data of one lane's 8 dims (dims 8*lane .. 8*lane+7) for the R cells of a block,
// loaded ahead of use so the next block's reads overlap this block's math
struct qsa_raw {
    uint32_t q[QSA_ATTN_R][2];
    float    d[QSA_ATTN_R];
};

static __device__ __forceinline__ void qsa_load_raw(const char * base, const size_t nb_cell, const int lane, qsa_raw & r) {
#pragma unroll
    for (int c = 0; c < QSA_ATTN_R; ++c) {
        const block_q8_0 * b = (const block_q8_0 *) (base + c*nb_cell) + lane/4;
        const uint16_t * q16 = (const uint16_t *) (b->qs + (lane % 4)*8);
        r.q[c][0] = (uint32_t) q16[0] | ((uint32_t) q16[1] << 16);
        r.q[c][1] = (uint32_t) q16[2] | ((uint32_t) q16[3] << 16);
        r.d[c]    = __half2float(b->d);
    }
}

static __device__ __forceinline__ void qsa_dequant(const qsa_raw & r, const int c, float (&x)[qsa_dpl]) {
#pragma unroll
    for (int w = 0; w < 2; ++w) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            x[4*w + b] = r.d[c] * (float) (int8_t) ((r.q[c][w] >> (8*b)) & 0xFF);
        }
    }
}

// Q4_0 V (LLAMA_QSA_Q4V=1): a q4_0 block packs 32 dims as nibbles - byte i holds dim i in its
// low half and dim i+16 in its high half - so a lane's 8 dims are 8 nibbles of 8 consecutive
// bytes (low half for dims 0..15, high half for dims 16..31). The dequant mirrors the gather
// fallback bit for bit: the fallback materializes F16 V via GET_ROWS (dequantize_q4_0 in
// dequantize.cuh: ((float)nib - 8.0f) * d in F32, then cast to F16), so both paths feed the
// attention the same V values.
struct qsa_raw_v4 {
    uint32_t q[QSA_ATTN_R];
    float    d[QSA_ATTN_R];
    uint32_t sh;   // nibble select: 0 = low (dims 0..15), 4 = high (dims 16..31)
};

static __device__ __forceinline__ void qsa_load_raw_v4(const char * base, const size_t nb_cell, const int lane, qsa_raw_v4 & r) {
    const int boff = 8*(lane % 2);                 // first of the 8 bytes this lane reads
    r.sh = (lane % 4) < 2 ? 0u : 4u;
#pragma unroll
    for (int c = 0; c < QSA_ATTN_R; ++c) {
        const block_q4_0 * b = (const block_q4_0 *) (base + c*nb_cell) + lane/4;
        uint32_t packed = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            packed |= (uint32_t) b->qs[boff + i] << (8*i);
        }
        r.q[c] = packed;
        r.d[c] = __half2float(b->d);
    }
}

static __device__ __forceinline__ void qsa_dequant_v4(const qsa_raw_v4 & r, const int c, float (&x)[qsa_dpl]) {
#pragma unroll
    for (int b = 0; b < qsa_dpl; ++b) {
        const float f = ((float) ((r.q[c] >> (8*b + r.sh)) & 0xF) - 8.0f) * r.d[c];
        x[b] = __half2float(__float2half_rn(f));   // the F16 round the gather path's output carries
    }
}

template <int HPW, int NGRP, bool VQ4>
__launch_bounds__(NGRP*WARP_SIZE)
static __global__ void qsa_attn_kernel(
        const char * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v,
        const half * __restrict__ mask, const int32_t * __restrict__ ids, float * __restrict__ dst,
        float * __restrict__ part_acc, float2 * __restrict__ part_ml,
        const int n_head, const int gqa, const int n_wblk, const int n_split,
        const size_t nbq1, const size_t nbq2, const size_t nbk1, const size_t nbk2,
        const size_t nbv1, const size_t nbv2, const size_t nbm_tok, const size_t nbi_tok,
        const float scale) {
    static_assert(HPW*QSA_ATTN_R <= WARP_SIZE, "one lane per (head, cell) score");
    // one CUDA block per (split, kv head, token); its warps split the kv head's query heads, so they
    // read the same K/V rows (shared through L1)
    const int lane  = threadIdx.x;
    const int split = blockIdx.x;
    const int h_kv  = blockIdx.y;
    const int tok   = blockIdx.z;
    const int h0    = h_kv*gqa + threadIdx.y*HPW;   // first query head of this warp

    __shared__ float red[NGRP][WARP_SIZE][HPW*QSA_ATTN_R + 1];

    // this block's share of the query's block list
    const int per = (n_wblk + n_split - 1) / n_split;
    const int j0  = split*per;
    const int j1  = min(n_wblk, j0 + per);

    // the share's block ids and cell masks, loaded once for all warps of the block
    extern __shared__ char qsa_smem[];
    int32_t * s_ids  = (int32_t *) qsa_smem;
    float   * s_mask = (float *) (s_ids + per);
    const int32_t * id_t = (const int32_t *) ((const char *) ids + tok*nbi_tok);
    const half    * m_t  = (const half *) ((const char *) mask + tok*nbm_tok);
    for (int i = threadIdx.y*WARP_SIZE + lane; i < (j1 - j0)*QSA_ATTN_R; i += NGRP*WARP_SIZE) {
        s_mask[i] = __half2float(m_t[j0*QSA_ATTN_R + i]);
        if (i < j1 - j0) {
            s_ids[i] = id_t[j0 + i];
        }
    }
    __syncthreads();

    float qr[HPW][qsa_dpl];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        const float * qh = (const float *) (q + tok*nbq2 + (h0 + h)*nbq1) + lane*qsa_dpl;
#pragma unroll
        for (int d = 0; d < qsa_dpl; ++d) {
            qr[h][d] = qh[d] * scale;
        }
    }

    float acc[HPW][qsa_dpl];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
#pragma unroll
        for (int d = 0; d < qsa_dpl; ++d) {
            acc[h][d] = 0.0f;
        }
    }
    // running max and sum of the head this lane scores (lanes h*R .. h*R + R-1 agree)
    float m_run = -INFINITY;
    float l_run = 0.0f;

    const int  my_c   = lane % QSA_ATTN_R;
    const bool scorer = lane < HPW*QSA_ATTN_R;

    // next block with a visible cell (padding blocks are off entirely)
    auto next_active = [&](int j) {
        while (j < j1) {
            const float * mj = s_mask + (j - j0)*QSA_ATTN_R;
            bool any = false;
#pragma unroll
            for (int c = 0; c < QSA_ATTN_R; ++c) {
                any |= mj[c] != -INFINITY;
            }
            if (any) {
                break;
            }
            ++j;
        }
        return j;
    };

    for (int j = next_active(j0); j < j1; j = next_active(j + 1)) {
        const float mv = s_mask[(j - j0)*QSA_ATTN_R + my_c];
        // K and V reads both issued before the math
        const int64_t cell0 = (int64_t) s_ids[j - j0] * QSA_ATTN_R;
        qsa_raw kr;
        qsa_raw vr;
        qsa_raw_v4 vr4;
        qsa_load_raw(k + cell0*nbk2 + h_kv*nbk1, nbk2, lane, kr);
        if (VQ4) {
            qsa_load_raw_v4(v + cell0*nbv2 + h_kv*nbv1, nbv2, lane, vr4);
        } else {
            qsa_load_raw(v + cell0*nbv2 + h_kv*nbv1, nbv2, lane, vr);
        }

        // q.k partial sums over this lane's dims, for every (head, cell)
        float part[HPW][QSA_ATTN_R];
#pragma unroll
        for (int c = 0; c < QSA_ATTN_R; ++c) {
            float kx[qsa_dpl];
            qsa_dequant(kr, c, kx);
#pragma unroll
            for (int h = 0; h < HPW; ++h) {
                float sacc = 0.0f;
#pragma unroll
                for (int d = 0; d < qsa_dpl; ++d) {
                    sacc += qr[h][d] * kx[d];
                }
                part[h][c] = sacc;
            }
        }
        // transpose through shared memory: lane h*R + c sums (h, c) over all lanes
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
#pragma unroll
            for (int c = 0; c < QSA_ATTN_R; ++c) {
                red[threadIdx.y][lane][h*QSA_ATTN_R + c] = part[h][c];
            }
        }
        __syncwarp();
        float s = -INFINITY;
        if (scorer) {
            float sum = 0.0f;
#pragma unroll 8
            for (int i = 0; i < WARP_SIZE; ++i) {
                sum += red[threadIdx.y][i][lane];
            }
            s = mv == -INFINITY ? -INFINITY : sum + mv;
        }
        __syncwarp();

        // online softmax per head over the block's R cells (groups of R lanes)
        float bmax = s;
#pragma unroll
        for (int off = 1; off < QSA_ATTN_R; off <<= 1) {
            bmax = fmaxf(bmax, __shfl_xor_sync(0xFFFFFFFF, bmax, off));
        }
        const float m_new = fmaxf(m_run, bmax);
        const float alpha = m_new == -INFINITY ? 1.0f : expf(m_run - m_new);
        const float p     = s == -INFINITY ? 0.0f : expf(s - m_new);
        float psum = p;
#pragma unroll
        for (int off = 1; off < QSA_ATTN_R; off <<= 1) {
            psum += __shfl_xor_sync(0xFFFFFFFF, psum, off);
        }
        l_run = l_run*alpha + psum;
        m_run = m_new;

        // acc = acc*alpha + sum_c p(h, c) * v_c
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
            const float al = __shfl_sync(0xFFFFFFFF, alpha, h*QSA_ATTN_R);
#pragma unroll
            for (int d = 0; d < qsa_dpl; ++d) {
                acc[h][d] *= al;
            }
        }
#pragma unroll
        for (int c = 0; c < QSA_ATTN_R; ++c) {
            float vx[qsa_dpl];
            if (VQ4) {
                qsa_dequant_v4(vr4, c, vx);
            } else {
                qsa_dequant(vr, c, vx);
            }
#pragma unroll
            for (int h = 0; h < HPW; ++h) {
                const float pc = __shfl_sync(0xFFFFFFFF, p, h*QSA_ATTN_R + c);
#pragma unroll
                for (int d = 0; d < qsa_dpl; ++d) {
                    acc[h][d] += pc * vx[d];
                }
            }
        }
    }

    // per head (m, l) from its scoring lanes
    float mh[HPW], lh[HPW];
#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        mh[h] = __shfl_sync(0xFFFFFFFF, m_run, h*QSA_ATTN_R);
        lh[h] = __shfl_sync(0xFFFFFFFF, l_run, h*QSA_ATTN_R);
    }

    if (n_split == 1) {
#pragma unroll
        for (int h = 0; h < HPW; ++h) {
            const float inv = lh[h] > 0.0f ? 1.0f/lh[h] : 0.0f;
            float * o = dst + ((int64_t) tok*n_head + h0 + h)*QSA_ATTN_D + lane*qsa_dpl;
#pragma unroll
            for (int d = 0; d < qsa_dpl; ++d) {
                o[d] = acc[h][d]*inv;
            }
        }
        return;
    }

#pragma unroll
    for (int h = 0; h < HPW; ++h) {
        const int64_t row = ((int64_t) tok*n_head + h0 + h)*n_split + split;
        float * o = part_acc + row*QSA_ATTN_D + lane*qsa_dpl;
#pragma unroll
        for (int d = 0; d < qsa_dpl; ++d) {
            o[d] = acc[h][d];
        }
        if (lane == 0) {
            part_ml[row] = make_float2(mh[h], lh[h]);
        }
    }
}

// one block per (head, token): merge the splits' running softmax states
static __global__ void qsa_attn_combine(const float * __restrict__ part_acc, const float2 * __restrict__ part_ml,
        float * __restrict__ dst, const int n_split) {
    const int64_t row = (int64_t) blockIdx.y*gridDim.x + blockIdx.x;   // tok*n_head + head
    const float2 * ml = part_ml + row*n_split;
    float m = -INFINITY;
    for (int s = 0; s < n_split; ++s) {
        m = fmaxf(m, ml[s].x);
    }
    float l = 0.0f;
    float o = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        const float w = ml[s].x == -INFINITY ? 0.0f : expf(ml[s].x - m);
        l += w*ml[s].y;
        o += w*part_acc[(row*n_split + s)*QSA_ATTN_D + threadIdx.x];
    }
    dst[row*QSA_ATTN_D + threadIdx.x] = l > 0.0f ? o/l : 0.0f;
}

bool ggml_cuda_qsa_attn_is_sparse(const ggml_tensor * dst) {
    return dst->op == GGML_OP_FLASH_ATTN_EXT && dst->src[5] != nullptr && dst->src[5]->type == GGML_TYPE_I32;
}

// LLAMA_QSA_Q4V=1: also take a Q4_0 V cache (the K side stays Q8_0). The kernel dequantizes V
// in place with the gather fallback's exact arithmetic, so 0 (the default) keeps today's
// F16 gather fallback and 1 removes its [D, n_head_kv, n_wblk*ratio, n_tok] materialization.
static bool qsa_q4v_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_QSA_Q4V");
        return env != nullptr && atoi(env) != 0;
    }();
    return enabled;
}

bool ggml_cuda_qsa_attn_supported(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * m = dst->src[3];
    const ggml_tensor * ids = dst->src[5];
    const bool v_ok = v->type == GGML_TYPE_Q8_0 || (v->type == GGML_TYPE_Q4_0 && qsa_q4v_enabled());
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_Q8_0 || !v_ok || !m || m->type != GGML_TYPE_F16) {
        return false;
    }
    if (q->ne[0] != QSA_ATTN_D || k->ne[0] != QSA_ATTN_D || v->ne[0] != QSA_ATTN_D || ggml_get_op_params_i32(dst, 6) != QSA_ATTN_R) {
        return false;
    }
    const int gqa = q->ne[1] / k->ne[1];
    // the head groupings instantiated below: 24/12/8/6/4/3/2/1 query heads per KV head (6: 4 GPUs on the KV-duplicated model)
    if (q->ne[1] % k->ne[1] != 0 || !(gqa == 24 || gqa == 16 || gqa == 12 || gqa == 8 || gqa == 6 || gqa == 4 || gqa == 3 ||
            gqa == 2 || gqa == 1)) {
        return false;
    }
    return q->nb[0] == sizeof(float) && ggml_is_contiguous(m) && ggml_is_contiguous(ids) &&
           ids->ne[1] == q->ne[2] && m->ne[0] == ids->ne[0]*QSA_ATTN_R && m->ne[3] == q->ne[2] &&
           dst->ne[0] == QSA_ATTN_D && dst->ne[1] == q->ne[1] && dst->ne[2] == q->ne[2] && ggml_is_contiguous(dst);
}

void ggml_cuda_qsa_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * m   = dst->src[3];
    const ggml_tensor * ids = dst->src[5];
    GGML_ASSERT(ggml_cuda_qsa_attn_supported(dst));

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int n_head = q->ne[1];
    const int n_tok  = q->ne[2];
    const int gqa    = n_head / k->ne[1];
    const int n_wblk = ids->ne[0];

    // 3 heads per warp for gqa 12 (4 warps per block); other group sizes pick a divisor
    const int hpw   = gqa % 3 == 0 && gqa/3 <= 8 ? 3 : gqa % 4 == 0 && gqa/4 <= 8 ? 4 : gqa % 2 == 0 && gqa/2 <= 8 ? 2 : 1;
    const int n_grp = gqa / hpw;
    GGML_ASSERT(n_grp <= 8);

    // enough blocks to fill the GPU: small batches split each query's block list
    const int n_sm    = ggml_cuda_info().devices[ctx.device].nsm;
    const int blocks0 = n_tok * (int) k->ne[1];
    int n_split = std::max(1, std::min(64, (n_sm*8 + blocks0 - 1) / blocks0));
    n_split = std::min(n_split, n_wblk);

    ggml_cuda_pool_alloc<float>  part_acc(ctx.pool());
    ggml_cuda_pool_alloc<float2> part_ml(ctx.pool());
    if (n_split > 1) {
        part_acc.alloc((size_t) n_tok*n_head*n_split*QSA_ATTN_D);
        part_ml.alloc((size_t) n_tok*n_head*n_split);
    }

    const dim3 grid(n_split, k->ne[1], n_tok);
    const dim3 block(WARP_SIZE, n_grp);
    cudaStream_t stream = ctx.stream();

    const int    per  = (n_wblk + n_split - 1) / n_split;
    const size_t smem = (size_t) per*(sizeof(int32_t) + QSA_ATTN_R*sizeof(float));
    const bool   v_q4 = v->type == GGML_TYPE_Q4_0;
#define QSA_LAUNCH(HPW, NGRP, VQ4) qsa_attn_kernel<HPW, NGRP, VQ4><<<grid, block, smem, stream>>>( \
        (const char *) q->data, (const char *) k->data, (const char *) v->data, (const half *) m->data, \
        (const int32_t *) ids->data, (float *) dst->data, part_acc.ptr, part_ml.ptr, \
        n_head, gqa, n_wblk, n_split, q->nb[1], q->nb[2], k->nb[1], k->nb[2], v->nb[1], v->nb[2], \
        m->nb[3], ids->nb[1], scale)
    if      (hpw == 3 && n_grp == 8) { if (v_q4) { QSA_LAUNCH(3, 8, true); } else { QSA_LAUNCH(3, 8, false); } }
    else if (hpw == 3 && n_grp == 4) { if (v_q4) { QSA_LAUNCH(3, 4, true); } else { QSA_LAUNCH(3, 4, false); } }
    else if (hpw == 3 && n_grp == 2) { if (v_q4) { QSA_LAUNCH(3, 2, true); } else { QSA_LAUNCH(3, 2, false); } }
    else if (hpw == 3 && n_grp == 1) { if (v_q4) { QSA_LAUNCH(3, 1, true); } else { QSA_LAUNCH(3, 1, false); } }
    else if (hpw == 4 && n_grp == 1) { if (v_q4) { QSA_LAUNCH(4, 1, true); } else { QSA_LAUNCH(4, 1, false); } }
    else if (hpw == 4 && n_grp == 2) { if (v_q4) { QSA_LAUNCH(4, 2, true); } else { QSA_LAUNCH(4, 2, false); } }
    else if (hpw == 4 && n_grp == 4) { if (v_q4) { QSA_LAUNCH(4, 4, true); } else { QSA_LAUNCH(4, 4, false); } }
    else if (hpw == 2 && n_grp == 4) { if (v_q4) { QSA_LAUNCH(2, 4, true); } else { QSA_LAUNCH(2, 4, false); } }
    else if (hpw == 2 && n_grp == 1) { if (v_q4) { QSA_LAUNCH(2, 1, true); } else { QSA_LAUNCH(2, 1, false); } }
    else if (hpw == 1 && n_grp == 1) { if (v_q4) { QSA_LAUNCH(1, 1, true); } else { QSA_LAUNCH(1, 1, false); } }
    else { GGML_ABORT("qsa attention: unsupported head grouping %d x %d", hpw, n_grp); }
#undef QSA_LAUNCH
    CUDA_CHECK(cudaGetLastError());

    if (n_split > 1) {
        qsa_attn_combine<<<dim3(n_head, n_tok), QSA_ATTN_D, 0, stream>>>(part_acc.ptr, part_ml.ptr, (float *) dst->data, n_split);
        CUDA_CHECK(cudaGetLastError());
    }
}
