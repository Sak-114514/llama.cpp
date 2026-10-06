#pragma once

#if defined(__linux__)
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

namespace ggml_ple_io {
struct source_info {
    const char * base = nullptr;
    size_t bytes = 0;
    uint64_t offset = 0;
    dev_t device = 0;
    ino_t inode = 0;
    off_t file_size = 0;
    timespec modified{};
    std::string path;
    bool operator==(const source_info & rhs) const {
        return base == rhs.base && bytes == rhs.bytes && offset == rhs.offset && device == rhs.device &&
            inode == rhs.inode && file_size == rhs.file_size && modified.tv_sec == rhs.modified.tv_sec &&
            modified.tv_nsec == rhs.modified.tv_nsec && path == rhs.path;
    }
};

inline bool mapped_source(const ggml_tensor * tensor, source_info & out) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(tensor->data);
    const size_t bytes = ggml_nbytes(tensor);
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        unsigned long begin = 0, end = 0;
        unsigned long long offset = 0, inode = 0;
        unsigned major_id = 0, minor_id = 0;
        char perms[5]{};
        int used = 0;
        if (std::sscanf(line.c_str(), "%lx-%lx %4s %llx %x:%x %llu %n", &begin, &end, perms,
                &offset, &major_id, &minor_id, &inode, &used) != 7) continue;
        if (address < begin || address >= end || bytes > end - address) continue;
        std::string path = line.substr(used);
        const size_t first = path.find_first_not_of(' ');
        if (first == std::string::npos) return false;
        path.erase(0, first);
        if (path.empty() || path[0] != '/' || path.find(" (deleted)") != std::string::npos || path.find('\\') != std::string::npos) return false;
        struct stat st{};
        if (stat(path.c_str(), &st) || !S_ISREG(st.st_mode) || st.st_ino != inode ||
                major(st.st_dev) != major_id || minor(st.st_dev) != minor_id) return false;
        const uint64_t file_offset = offset + address - begin;
        if (file_offset > uint64_t(st.st_size) || bytes > uint64_t(st.st_size) - file_offset) return false;
        out = {reinterpret_cast<const char *>(address), bytes, file_offset, st.st_dev, st.st_ino, st.st_size, st.st_mtim, std::move(path)};
        return true;
    }
    return false;
}

struct aligned_scratch {
    void * data = nullptr;
    aligned_scratch() { if (posix_memalign(&data, 65536, 8192)) data = nullptr; }
    ~aligned_scratch() { std::free(data); }
};

class row_cache {
    static constexpr size_t n_shards = 64, n_sets = 1024, ways = 4;
    struct entry {
        uint64_t row = UINT64_MAX, age = 0;
        std::array<uint8_t, 128> data{};
    };
    struct shard {
        std::mutex mutex;
        std::vector<entry> entries;
        uint64_t clock = 0;
    };
    std::array<shard, n_shards> shards;
    int fd = -1;
    size_t row_bytes = 0, page_size = 0;
    std::atomic<bool> direct_ok{false};
    std::atomic<uint64_t> hits{0}, resident{0}, direct{0}, fallback{0}, io_bytes{0};
public:
    const source_info info;
    row_cache(source_info source, size_t width) : row_bytes(width), page_size(sysconf(_SC_PAGESIZE)), info(std::move(source)) {
        fd = open(info.path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        if (fd >= 0) {
            struct stat st{};
            if (fstat(fd, &st) || st.st_ino != info.inode || st.st_dev != info.device ||
                    st.st_size != info.file_size || st.st_mtim.tv_sec != info.modified.tv_sec || st.st_mtim.tv_nsec != info.modified.tv_nsec) {
                close(fd); fd = -1;
            }
        }
        direct_ok.store(fd >= 0);
        GGML_LOG_INFO("PLE encoded-row cache active: row_bytes=%zu capacity=%zu direct=%d page=%zu\n", row_bytes, n_shards*n_sets*ways, int(fd >= 0), page_size);
    }
    ~row_cache() {
        GGML_LOG_INFO("PLE row-cache stats hits=%llu resident=%llu direct=%llu fallback=%llu io_bytes=%llu\n",
            (unsigned long long) hits.load(), (unsigned long long) resident.load(), (unsigned long long) direct.load(),
            (unsigned long long) fallback.load(), (unsigned long long) io_bytes.load());
        if (fd >= 0) close(fd);
    }
    void read(uint64_t row, uint8_t * result) {
        const uint64_t hash = row * UINT64_C(11400714819323198485);
        auto & part = shards[hash & (n_shards - 1)];
        const size_t start = ((hash >> 6) & (n_sets - 1)) * ways;
        {
            std::lock_guard<std::mutex> lock(part.mutex);
            if (part.entries.empty()) part.entries.resize(n_sets * ways);
            for (size_t i = start; i < start + ways; ++i) if (part.entries[i].row == row) {
                std::memcpy(result, part.entries[i].data.data(), row_bytes);
                part.entries[i].age = ++part.clock;
                hits.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        const char * source = info.base + row * row_bytes;
        const uintptr_t address = reinterpret_cast<uintptr_t>(source);
        const uintptr_t first = address & ~(page_size - 1);
        const uintptr_t last = (address + row_bytes + page_size - 1) & ~(page_size - 1);
        unsigned char residency[2]{};
        const bool ready = mincore(reinterpret_cast<void *>(first), last - first, residency) == 0 &&
            (residency[0] & 1) && (last - first == page_size || (residency[1] & 1));
        bool read_ok = false;
        if (ready) {
            std::memcpy(result, source, row_bytes);
            resident.fetch_add(1, std::memory_order_relaxed);
            read_ok = true;
        } else if (direct_ok.load(std::memory_order_relaxed)) {
            thread_local aligned_scratch scratch;
            const uint64_t offset = info.offset + row * row_bytes;
            const uint64_t block = offset & ~UINT64_C(4095);
            const size_t skip = offset - block;
            const size_t length = ((skip + row_bytes + 4095) / 4096) * 4096;
            ssize_t size = -1;
            if (scratch.data) {
                do { size = pread(fd, scratch.data, length, off_t(block)); } while (size < 0 && errno == EINTR);
            }
            if (size >= ssize_t(skip + row_bytes)) {
                std::memcpy(result, static_cast<const char *>(scratch.data) + skip, row_bytes);
                direct.fetch_add(1, std::memory_order_relaxed);
                io_bytes.fetch_add(size, std::memory_order_relaxed);
                read_ok = true;
            } else if (size < 0 && (errno == EINVAL || errno == EOPNOTSUPP)) {
                direct_ok.store(false, std::memory_order_relaxed);
            }
        }
        if (!read_ok) {
            std::memcpy(result, source, row_bytes);
            fallback.fetch_add(1, std::memory_order_relaxed);
        }
        {
            std::lock_guard<std::mutex> lock(part.mutex);
            size_t victim = start;
            for (size_t i = start; i < start + ways; ++i) {
                if (part.entries[i].row == row) { victim = i; break; }
                if (part.entries[i].age < part.entries[victim].age) victim = i;
            }
            auto & slot = part.entries[victim];
            slot.row = row; slot.age = ++part.clock;
            std::memcpy(slot.data.data(), result, row_bytes);
        }
    }
};

inline std::shared_ptr<row_cache> acquire(const ggml_tensor * table) {
    source_info source;
    if (!mapped_source(table, source)) return {};
    static std::mutex mutex;
    static std::shared_ptr<row_cache> cache;
    std::lock_guard<std::mutex> lock(mutex);
    if (!cache || !(cache->info == source)) cache = std::make_shared<row_cache>(std::move(source), table->nb[1]);
    return cache;
}
} // namespace ggml_ple_io
#endif

static bool ggml_cpu_ple_cached_rows(const ggml_compute_params * params, ggml_tensor * dst, ggml_to_float_t dequantize) {
#if defined(__linux__)
    static const bool enabled = [] { const char * value = getenv("GGML_CPU_PLE_ROW_CACHE"); return value && atoi(value) != 0; }();
    const auto * table = dst->src[0];
    const auto * indices = dst->src[1];
    if (!enabled || table->type != GGML_TYPE_IQ4_NL || table->ne[0] != 160 || table->ne[2] != 1 || table->ne[3] != 1 ||
            strcmp(table->name, "per_layer_token_embd.weight") || ggml_nbytes(table) <= 64*1024*1024 ||
            !ggml_is_contiguous(table) || !ggml_is_contiguous(indices) || !ggml_is_contiguous(dst)) return false;
    using shared_cache = std::shared_ptr<ggml_ple_io::row_cache>;
    static_assert(sizeof(shared_cache) <= 64, "CPU graph workspace must fit shared cache ownership");
    GGML_ASSERT(params->wdata && params->wsize >= sizeof(shared_cache));
    auto * shared = static_cast<shared_cache *>(params->wdata);
    if (params->ith == 0) new (shared) shared_cache(ggml_ple_io::acquire(table));
    ggml_barrier(params->threadpool);
    auto cache = *shared;
    if (cache) {
        const int64_t count = ggml_nelements(indices);
        const int64_t per_thread = (count + params->nth - 1) / params->nth;
        const int64_t end = std::min(count, per_thread * (params->ith + 1));
        alignas(16) uint8_t packed[128];
        for (int64_t i = per_thread * params->ith; i < end; ++i) {
            const int32_t row = static_cast<const int32_t *>(indices->data)[i];
            GGML_ASSERT(row >= 0 && row < table->ne[1]);
            cache->read(row, packed);
            dequantize(packed, reinterpret_cast<float *>(static_cast<char *>(dst->data) + i * dst->nb[1]), 160);
        }
    }
    ggml_barrier(params->threadpool);
    if (params->ith == 0) shared->~shared_cache();
    ggml_barrier(params->threadpool);
    return bool(cache);
#else
    GGML_UNUSED(params); GGML_UNUSED(dst); GGML_UNUSED(dequantize);
    return false;
#endif
}
