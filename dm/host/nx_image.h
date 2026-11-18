// An NX container (nxformat SPEC.md, versions (0,2) and (0,3)) as a RomFS image: one snapshot's romfs subtree is turned into the
// in-memory index a DMROMFS1 image has (romfs_image.h), so hle_fs.c's lookups, listings and sizes are the same code for both.
//
// Naming: "<file>.nx", "<file>.nx:<selector>" or "<file>.nx:<selector>/romfs", the selector being base, update/<decimal version> or
// addon/<16 hex digits>; with none, the highest-version update's RomFS, else the base's. Add-on mounts use "aoc/<index>": the DLC
// snapshot with ADDON_TITLE_ID whose title ID's low 12 bits equal index.
//
// Files: a file entry has IMG_ENT_NX set and points at its nx_file record (first block reference, block count, chunk shift). ReadFile maps
// (offset, size) to blocks: RAW blocks are pread straight into the guest buffer (with dm_guest_write_prepare / EFAULT retry); ZSTD blocks
// are decoded whole (ZSTD_decompress, guarded by DM_NUP_ZSTD) into a thread-safe LRU block cache (DM_NX_CACHE_MB, default 64) and copied.
// Every slice read is checked against its CRC-32C entry in the integrity table unless DM_NX_VERIFY=0.
#pragma once
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "romfs_image.h"
#include "plat/plat.h"

#ifdef __ARM_FEATURE_CRC32
#include <arm_acle.h>
#endif

#ifdef DM_NUP_ZSTD
#ifndef DM_ZSTD_DECLARED
#define DM_ZSTD_DECLARED
size_t ZSTD_decompress(void *dst, size_t dst_cap, const void *src, size_t src_size);
unsigned ZSTD_isError(size_t code);
const char *ZSTD_getErrorName(size_t code);
#endif
#endif

#define IMG_ENT_NX 8        // in memory only (an NX image): a = index in nx_aux.files
#define NX_NONE 0xFFFFFFFFu
#define NX_CHECK_SIZE 16384u
#define NX_MAX_CATALOG (16ull << 20)
#define NX_MAX_MANIFEST (4ull << 30)
#define NX_MAX_BLOCK_LOGICAL (1u << 20)
#define NX_MAX_BLOCK_STORED ((1u << 20) + (64u << 10))
#define NX_CACHE_BUCKETS 4096

static inline uint16_t nx_u16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t nx_u32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t nx_u64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }

static inline int nx_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a > UINT64_MAX - b) return 0;
    *out = a + b; return 1;
}
static inline int nx_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (a && b > UINT64_MAX / a) return 0;
    *out = a * b; return 1;
}
static inline uint64_t nx_align4(uint64_t v) { return (v + 3ull) & ~3ull; }
static inline uint64_t nx_align8(uint64_t v) { return (v + 7ull) & ~7ull; }
static inline uint64_t nx_align16(uint64_t v) { return (v + 15ull) & ~15ull; }

#if defined(__aarch64__) && !defined(__ARM_FEATURE_CRC32) && defined(__linux__)
// Android's arm64 baseline (armv8-a) leaves the CRC extension off, so __ARM_FEATURE_CRC32 is not defined there and the table path
// below ran at a few hundred MB/s: 2.3 s of extra host time over OoT's first 900 MB of reads on the S26. Every arm64 Android phone has
// the instructions (mandatory from ARMv8.1); HWCAP_CRC32 is still checked once so an old core falls back to the table.
#include <sys/auxv.h>
#define NX_CRC_RUNTIME 1
__attribute__((target("crc"))) static uint32_t nx_crc32c_hw(uint32_t c, const uint8_t *p, size_t len) {
    while (len && ((uintptr_t)p & 7)) { c = __builtin_arm_crc32cb(c, *p++); len--; }
    while (len >= 8) { uint64_t v; memcpy(&v, p, 8); c = __builtin_arm_crc32cd(c, v); p += 8; len -= 8; }
    while (len--) c = __builtin_arm_crc32cb(c, *p++);
    return c;
}
#endif

static uint32_t nx_crc32c(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint32_t c = 0xFFFFFFFFu;
#ifdef __ARM_FEATURE_CRC32
    while (len && ((uintptr_t)p & 7)) { c = __crc32cb(c, *p++); len--; }
    while (len >= 8) { uint64_t v; memcpy(&v, p, 8); c = __crc32cd(c, v); p += 8; len -= 8; }
    while (len--) c = __crc32cb(c, *p++);
#else
#ifdef NX_CRC_RUNTIME
    static _Atomic int hw = -1;   // -1 unknown, 0 table, 1 CRC instructions
    if (hw < 0) hw = (getauxval(AT_HWCAP) & (1ul << 7)) != 0;   // HWCAP_CRC32
    if (hw) return nx_crc32c_hw(c, p, len) ^ 0xFFFFFFFFu;
#endif
    static const uint32_t tab[256] = {
        0x00000000u, 0xf26b8303u, 0xe13b70f7u, 0x1350f3f4u, 0xc79a971fu, 0x35f1141cu, 0x26a1e7e8u, 0xd4ca64ebu,
        0x8ad958cfu, 0x78b2dbccu, 0x6be22838u, 0x9989ab3bu, 0x4d43cfd0u, 0xbf284cd3u, 0xac78bf27u, 0x5e133c24u,
        0x105ec76fu, 0xe235446cu, 0xf165b798u, 0x030e349bu, 0xd7c45070u, 0x25afd373u, 0x36ff2087u, 0xc494a384u,
        0x9a879fa0u, 0x68ec1ca3u, 0x7bbcef57u, 0x89d76c54u, 0x5d1d08bfu, 0xaf768bbcu, 0xbc267848u, 0x4e4dfb4bu,
        0x20bd8edeu, 0xd2d60dddu, 0xc186fe29u, 0x33ed7d2au, 0xe72719c1u, 0x154c9ac2u, 0x061c6936u, 0xf477ea35u,
        0xaa64d611u, 0x580f5512u, 0x4b5fa6e6u, 0xb93425e5u, 0x6dfe410eu, 0x9f95c20du, 0x8cc531f9u, 0x7eaeb2fau,
        0x30e349b1u, 0xc288cab2u, 0xd1d83946u, 0x23b3ba45u, 0xf779deaeu, 0x05125dadu, 0x1642ae59u, 0xe4292d5au,
        0xba3a117eu, 0x4851927du, 0x5b016189u, 0xa96ae28au, 0x7da08661u, 0x8fcb0562u, 0x9c9bf696u, 0x6ef07595u,
        0x417b1dbcu, 0xb3109ebfu, 0xa0406d4bu, 0x522bee48u, 0x86e18aa3u, 0x748a09a0u, 0x67dafa54u, 0x95b17957u,
        0xcba24573u, 0x39c9c670u, 0x2a993584u, 0xd8f2b687u, 0x0c38d26cu, 0xfe53516fu, 0xed03a29bu, 0x1f682198u,
        0x5125dad3u, 0xa34e59d0u, 0xb01eaa24u, 0x42752927u, 0x96bf4dccu, 0x64d4cecfu, 0x77843d3bu, 0x85efbe38u,
        0xdbfc821cu, 0x2997011fu, 0x3ac7f2ebu, 0xc8ac71e8u, 0x1c661503u, 0xee0d9600u, 0xfd5d65f4u, 0x0f36e6f7u,
        0x61c69362u, 0x93ad1061u, 0x80fde395u, 0x72966096u, 0xa65c047du, 0x5437877eu, 0x4767748au, 0xb50cf789u,
        0xeb1fcbadu, 0x197448aeu, 0x0a24bb5au, 0xf84f3859u, 0x2c855cb2u, 0xdeeedfb1u, 0xcdbe2c45u, 0x3fd5af46u,
        0x7198540du, 0x83f3d70eu, 0x90a324fau, 0x62c8a7f9u, 0xb602c312u, 0x44694011u, 0x5739b3e5u, 0xa55230e6u,
        0xfb410cc2u, 0x092a8fc1u, 0x1a7a7c35u, 0xe811ff36u, 0x3cdb9bddu, 0xceb018deu, 0xdde0eb2au, 0x2f8b6829u,
        0x82f63b78u, 0x709db87bu, 0x63cd4b8fu, 0x91a6c88cu, 0x456cac67u, 0xb7072f64u, 0xa457dc90u, 0x563c5f93u,
        0x082f63b7u, 0xfa44e0b4u, 0xe9141340u, 0x1b7f9043u, 0xcfb5f4a8u, 0x3dde77abu, 0x2e8e845fu, 0xdce5075cu,
        0x92a8fc17u, 0x60c37f14u, 0x73938ce0u, 0x81f80fe3u, 0x55326b08u, 0xa759e80bu, 0xb4091bffu, 0x466298fcu,
        0x1871a4d8u, 0xea1a27dbu, 0xf94ad42fu, 0x0b21572cu, 0xdfeb33c7u, 0x2d80b0c4u, 0x3ed04330u, 0xccbbc033u,
        0xa24bb5a6u, 0x502036a5u, 0x4370c551u, 0xb11b4652u, 0x65d122b9u, 0x97baa1bau, 0x84ea524eu, 0x7681d14du,
        0x2892ed69u, 0xdaf96e6au, 0xc9a99d9eu, 0x3bc21e9du, 0xef087a76u, 0x1d63f975u, 0x0e330a81u, 0xfc588982u,
        0xb21572c9u, 0x407ef1cau, 0x532e023eu, 0xa145813du, 0x758fe5d6u, 0x87e466d5u, 0x94b49521u, 0x66df1622u,
        0x38cc2a06u, 0xcaa7a905u, 0xd9f75af1u, 0x2b9cd9f2u, 0xff56bd19u, 0x0d3d3e1au, 0x1e6dcdeeu, 0xec064eedu,
        0xc38d26c4u, 0x31e6a5c7u, 0x22b65633u, 0xd0ddd530u, 0x0417b1dbu, 0xf67c32d8u, 0xe52cc12cu, 0x1747422fu,
        0x49547e0bu, 0xbb3ffd08u, 0xa86f0efcu, 0x5a048dffu, 0x8ecee914u, 0x7ca56a17u, 0x6ff599e3u, 0x9d9e1ae0u,
        0xd3d3e1abu, 0x21b862a8u, 0x32e8915cu, 0xc083125fu, 0x144976b4u, 0xe622f5b7u, 0xf5720643u, 0x07198540u,
        0x590ab964u, 0xab613a67u, 0xb831c993u, 0x4a5a4a90u, 0x9e902e7bu, 0x6cfbad78u, 0x7fab5e8cu, 0x8dc0dd8fu,
        0xe330a81au, 0x115b2b19u, 0x020bd8edu, 0xf0605beeu, 0x24aa3f05u, 0xd6c1bc06u, 0xc5914ff2u, 0x37faccf1u,
        0x69e9f0d5u, 0x9b8273d6u, 0x88d28022u, 0x7ab90321u, 0xae7367cau, 0x5c18e4c9u, 0x4f48173du, 0xbd23943eu,
        0xf36e6f75u, 0x0105ec76u, 0x12551f82u, 0xe03e9c81u, 0x34f4f86au, 0xc69f7b69u, 0xd5cf889du, 0x27a40b9eu,
        0x79b737bau, 0x8bdcb4b9u, 0x988c474du, 0x6ae7c44eu, 0xbe2da0a5u, 0x4c4623a6u, 0x5f16d052u, 0xad7d5351u
    };
    for (size_t i = 0; i < len; i++) c = (c >> 8) ^ tab[(c ^ p[i]) & 0xFFu];
#endif
    return c ^ 0xFFFFFFFFu;
}

typedef struct { uint64_t off; uint32_t stored, logical, first_check; uint8_t codec; } nx_block;
typedef struct { uint64_t size; uint32_t first_ref, nblocks, name; uint16_t nlen; uint8_t chunk_shift; } nx_file;

typedef struct nx_cache_ent {
    uint32_t block_id, logical_size, refs;
    uint8_t loading, bad;
    uint8_t *buf;
    struct nx_cache_ent *hnext, *prev, *next;
} nx_cache_ent;

typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t cv;
    uint64_t held, budget, ndec;
    uint32_t nfiles, nblocks, nraw, nzstd, nchecks, nrefs;
    uint8_t verify;
    nx_file *files;
    uint32_t *refs;
    nx_block *blocks;
    uint32_t *checks;
    nx_cache_ent *buckets[NX_CACHE_BUCKETS];
    nx_cache_ent *lru_head, *lru_tail;   // head = MRU, tail = LRU
    char what[160];
} nx_aux;

static void nx_aux_free(void *p) {
    nx_aux *a = p; if (!a) return;
    nx_cache_ent *e = a->lru_head;
    while (e) { nx_cache_ent *nx = e->next; free(e->buf); free(e); e = nx; }
    free(a->files); free(a->refs); free(a->blocks); free(a->checks);
    pthread_mutex_destroy(&a->lock); pthread_cond_destroy(&a->cv);
    free(a);
}

// "<file>.nx[:<selector>[/romfs]]" -> file path in file and selector in sel ("" for none, "/romfs" dropped). 0 if not .nx.
static int nx_split(const char *spec, char *file, size_t fn, char *sel, size_t sn) {
    if (!spec) return 0;
    const char *c = NULL;
    for (const char *p = spec; (p = strstr(p, ".nx:")); p++) c = p;
    size_t L = strlen(spec);
    if (!c) {
        if (L < 4 || strcmp(spec + L - 3, ".nx")) return 0;   // require at least one char before ".nx"
        if (L >= fn) return 0;
        memcpy(file, spec, L + 1); if (sn) sel[0] = 0; return 1;
    }
    size_t flen = (size_t)(c - spec) + 3;
    if (flen == 3 || flen >= fn || strlen(c + 4) >= sn) return 0;
    memcpy(file, spec, flen); file[flen] = 0;
    strlcpy(sel, c + 4, sn);
    size_t s = strlen(sel);
    while (s && sel[s - 1] == '/') sel[--s] = 0;
    if (s >= 6 && !strcmp(sel + s - 6, "/romfs")) sel[s - 6] = 0;
    return 1;
}
static int nx_is_spec(const char *spec) { char file[1100], sel[128]; return spec && nx_split(spec, file, sizeof file, sel, sizeof sel); }

typedef struct {
    uint16_t major, minor;
    uint64_t archive_size, catalog_offset, catalog_size, media_offset, media_size;
    uint64_t data_offset, data_size, manifest_offset, manifest_size;
    uint64_t block_index_offset, block_count, integrity_offset, integrity_size;
    uint8_t catalog_sha[32], manifest_sha[32], index_sha[32];
    uint64_t generation;
} nx_hdr;

static int nx_is_zero(const uint8_t *p, size_t n) {
    for (size_t i = 0; i < n; i++) if (p[i]) return 0;
    return 1;
}

static int nx_parse_hdr(const uint8_t h[256], uint64_t phys_size, nx_hdr *o) {
    if (memcmp(h, "NXAR", 4)) return 0;
    o->major = nx_u16(h + 4); o->minor = nx_u16(h + 6);
    if (o->major != 0 || (o->minor != 2 && o->minor != 3)) return 0;
    if (nx_u32(h + 8) != 256 || nx_u16(h + 12) != 1 || nx_u16(h + 14) != 0) return 0;
    if (nx_u32(h + 224) != NX_CHECK_SIZE || !nx_is_zero(h + 228, 24)) return 0;
    if (nx_u32(h + 252) != nx_crc32c(h, 252)) return 0;
    o->archive_size = nx_u64(h + 16);
    o->catalog_offset = nx_u64(h + 24); o->catalog_size = nx_u64(h + 32);
    o->media_offset = nx_u64(h + 40); o->media_size = nx_u64(h + 48);
    o->data_offset = nx_u64(h + 56); o->data_size = nx_u64(h + 64);
    o->manifest_offset = nx_u64(h + 72); o->manifest_size = nx_u64(h + 80);
    o->block_index_offset = nx_u64(h + 88); o->block_count = nx_u64(h + 96);
    o->integrity_offset = nx_u64(h + 104); o->integrity_size = nx_u64(h + 112);
    memcpy(o->catalog_sha, h + 120, 32);
    memcpy(o->manifest_sha, h + 152, 32);
    memcpy(o->index_sha, h + 184, 32);
    o->generation = nx_u64(h + 216);

    if (o->archive_size > phys_size || o->archive_size > (uint64_t)INT64_MAX) return 0;
    if (o->catalog_offset % 16 || o->catalog_size < 128 || o->catalog_size > NX_MAX_CATALOG) return 0;
    if (o->manifest_offset % 16 || o->manifest_size < 96 || o->manifest_size > NX_MAX_MANIFEST) return 0;
    if ((o->media_offset == 0) != (o->media_size == 0)) return 0;
    if (o->media_size && (o->media_offset % 16 || o->media_size < 32)) return 0;
    if (o->block_count > 0x100000000ull) return 0;
    if (o->block_count == 0) {
        if (o->data_offset || o->data_size || o->block_index_offset || o->integrity_offset || o->integrity_size) return 0;
    } else {
        if (o->data_offset != 8192 || !o->data_size) return 0;
        if (o->block_index_offset % 16 || o->integrity_offset % 16 || (o->integrity_size % 4)) return 0;
        if (o->integrity_size / 4 < o->block_count) return 0;
    }
    uint64_t cur = 8192, end = 0;
    if (o->block_count > 0) {
        if (!nx_add_u64(o->data_offset, o->data_size, &cur)) return 0;
    }
    if (o->media_size) {
        if (o->media_offset < nx_align16(cur)) return 0;
        if (!nx_add_u64(o->media_offset, o->media_size, &end)) return 0;
        if (o->catalog_offset != nx_align16(end)) return 0;
    } else if (o->catalog_offset < nx_align16(cur)) return 0;
    if (!nx_add_u64(o->catalog_offset, o->catalog_size, &end)) return 0;
    if (o->manifest_offset != nx_align16(end)) return 0;
    if (!nx_add_u64(o->manifest_offset, o->manifest_size, &end)) return 0;
    if (o->block_count == 0) return end == o->archive_size;
    if (o->block_index_offset != nx_align16(end)) return 0;
    uint64_t bi_sz = 0;
    if (!nx_mul_u64(o->block_count, 24, &bi_sz) || !nx_add_u64(o->block_index_offset, bi_sz, &end)) return 0;
    if (o->integrity_offset != nx_align16(end)) return 0;
    if (!nx_add_u64(o->integrity_offset, o->integrity_size, &end)) return 0;
    return end == o->archive_size;
}

static int nx_ext_known_required(uint32_t type_id) {
    return type_id == 0x00010004u || type_id == 0x00010005u || type_id == 0x00010008u;
}

static romfs_img *nx_mount_hdr(int fd, const nx_hdr *H, const char *file, const char *sel, char *why, size_t wn, int *out_err) {
    uint8_t *cat = NULL, *man = NULL, *idx_raw = NULL;
    uint32_t *queue = NULL; uint64_t *ent_of = NULL; uint8_t *seen_dir = NULL, *seen_file = NULL;
    romfs_img *m = NULL; nx_aux *aux = NULL;
    int e = EINVAL;
#define NX_BAD(...) do { snprintf(why, wn, __VA_ARGS__); goto bad; } while (0)

    if (!(cat = malloc((size_t)H->catalog_size))) { e = ENOMEM; NX_BAD("out of memory for Catalog"); }
    if (!romfs_image_read(fd, cat, (size_t)H->catalog_size, (off_t)H->catalog_offset)) { e = errno ? errno : EIO; NX_BAD("cannot read Catalog"); }
    { uint8_t h[32]; plat_sha256(cat, (size_t)H->catalog_size, h); if (memcmp(h, H->catalog_sha, 32)) NX_BAD("Catalog SHA-256 mismatch"); }
    if (memcmp(cat, "NXCT", 4) || nx_u16(cat + 4) != 1 || nx_u16(cat + 6) != 128 || nx_u64(cat + 8) != H->catalog_size ||
        nx_u32(cat + 16) != 0 || nx_u32(cat + 44) != 0 || !nx_is_zero(cat + 112, 16))
        NX_BAD("invalid Catalog header");
    uint32_t app_cnt = nx_u32(cat + 20), prog_cnt = nx_u32(cat + 24), snap_cnt = nx_u32(cat + 28);
    uint32_t loc_cnt = nx_u32(cat + 32), dep_cnt = nx_u32(cat + 36), ext_cnt = nx_u32(cat + 40);
    if (!app_cnt || app_cnt > 4096 || prog_cnt > 16384 || !snap_cnt || snap_cnt > 65536 || loc_cnt > 16384 || dep_cnt > 32768 || ext_cnt > 4096)
        NX_BAD("Catalog counts out of limits");
    uint64_t app_off = nx_u64(cat + 48), prog_off = nx_u64(cat + 56), snap_off = nx_u64(cat + 64);
    uint64_t loc_off = nx_u64(cat + 72), dep_off = nx_u64(cat + 80), ext_off = nx_u64(cat + 88);
    uint64_t cpool_off = nx_u64(cat + 96), cpool_sz = nx_u64(cat + 104);
    {   uint64_t cur = 128, end = 0;
        if (app_off != cur) NX_BAD("bad application table offset");
        cur = nx_align8(cur + (uint64_t)app_cnt * 80);
        if (prog_cnt ? prog_off != cur : prog_off != 0) NX_BAD("bad program table offset");
        if (prog_cnt) cur = nx_align8(cur + (uint64_t)prog_cnt * 64);
        if (snap_off != cur) NX_BAD("bad snapshot table offset");
        cur = nx_align8(cur + (uint64_t)snap_cnt * 120);
        if (loc_cnt ? loc_off != cur : loc_off != 0) NX_BAD("bad localized table offset");
        if (loc_cnt) cur = nx_align8(cur + (uint64_t)loc_cnt * 24);
        if (dep_cnt ? dep_off != cur : dep_off != 0) NX_BAD("bad dependency table offset");
        if (dep_cnt) cur = nx_align8(cur + (uint64_t)dep_cnt * 56);
        if (ext_cnt ? ext_off != cur : ext_off != 0) NX_BAD("bad extension table offset");
        if (ext_cnt) cur = nx_align8(cur + (uint64_t)ext_cnt * 32);
        if (cpool_off != cur || !nx_add_u64(cpool_off, cpool_sz, &end) || end > H->catalog_size) NX_BAD("Catalog string pool out of bounds");
        uint64_t ext_cur = nx_align8(end);
        for (uint32_t i = 0; i < ext_cnt; i++) {
            const uint8_t *xr = cat + ext_off + (uint64_t)i * 32;
            uint32_t tid = nx_u32(xr), fl = nx_u32(xr + 4); uint16_t okind = nx_u16(xr + 8), res0 = nx_u16(xr + 10);
            uint32_t oidx = nx_u32(xr + 12); uint64_t doff = nx_u64(xr + 16), dsz = nx_u64(xr + 24), dend = 0;
            if (res0 || (fl & ~1u) || okind > 3) NX_BAD("invalid Catalog extension %u", i);
            if ((okind == 0 && oidx != NX_NONE) || (okind == 1 && oidx >= app_cnt) || (okind == 2 && oidx >= prog_cnt) || (okind == 3 && oidx >= snap_cnt))
                NX_BAD("Catalog extension %u owner out of range", i);
            if (doff < ext_cur || (doff % 8) || !nx_add_u64(doff, dsz, &dend) || dend > H->catalog_size)
                NX_BAD("Catalog extension %u payload out of bounds", i);
            ext_cur = nx_align8(dend);
            if ((fl & 1u) && !nx_ext_known_required(tid)) NX_BAD("unknown required Catalog extension %#x", tid);
        }
    }

    // Select snapshot matching selector
    int want_kind = -1; uint32_t want_ver = 0; uint64_t want_tid = 0; int aoc = -1;
    if (!sel[0]) want_kind = -2;
    else if (!strcmp(sel, "base")) want_kind = 0;
    else if (!strncmp(sel, "update/", 7) && sel[7] && strspn(sel + 7, "0123456789") == strlen(sel + 7)) {
        char *ep = NULL; unsigned long long v = strtoull(sel + 7, &ep, 10);
        if (*ep || v > UINT32_MAX) NX_BAD("invalid update version \"%s\"", sel + 7);
        want_kind = 1; want_ver = (uint32_t)v;
    } else if (!strncmp(sel, "addon/", 6) && strlen(sel + 6) == 16 && strspn(sel + 6, "0123456789abcdefABCDEF") == 16) {
        want_kind = 2; want_tid = strtoull(sel + 6, NULL, 16);
    } else if (!strncmp(sel, "aoc/", 4) && sel[4] && strspn(sel + 4, "0123456789") == strlen(sel + 4)) {
        want_kind = 2; aoc = atoi(sel + 4);
    } else NX_BAD("unknown selector \"%s\" (base, update/<version> or addon/<title ID>)", sel);

    int64_t pick = -1; uint32_t best_ver = 0; uint8_t pick_type = 0;
    for (uint32_t i = 0; i < snap_cnt; i++) {
        const uint8_t *s = cat + snap_off + (uint64_t)i * 120;
        uint32_t pidx = nx_u32(s); uint8_t stype = s[4], sflags = s[5];
        if (nx_u16(s + 6) || stype > 4 || (sflags & ~3u)) NX_BAD("invalid snapshot %u", i);
        if (pidx != NX_NONE && pidx >= prog_cnt) NX_BAD("snapshot %u program index out of range", i);
        if ((sflags & 2u) && (stype != 2 || pidx != NX_NONE)) NX_BAD("invalid ADDON_TITLE_ID snapshot %u", i);
        uint32_t ver = nx_u32(s + 8);
        if (want_kind == -2) {
            if (stype == 1 || stype == 3) {
                if (pick < 0 || (pick_type != 1 && pick_type != 3) || ver > best_ver) { pick = i; best_ver = ver; pick_type = stype; }
            } else if (stype == 0 && pick < 0) { pick = i; pick_type = stype; }
        } else if (want_kind == 0 && stype == 0) { pick = i; break; }
        else if (want_kind == 1 && (stype == 1 || stype == 3) && ver == want_ver) { pick = i; break; }
        else if (want_kind == 2 && stype == 2 && (sflags & 2u) && nx_u64(s + 40) == 0) {
            uint64_t tid = nx_u64(s + 32);
            if (tid && (aoc >= 0 ? (int)(tid & 0xfff) == aoc : tid == want_tid)) { pick = i; break; }
        }
    }
    if (pick < 0) NX_BAD("no snapshot matching %s", sel[0] ? sel : "(update or base)");
    uint32_t snap_root = nx_u32(cat + snap_off + (uint64_t)pick * 120 + 12);

    // Read and verify Manifest + Hot Block Index + Integrity Table
    if (H->manifest_size > SIZE_MAX) { e = ENOMEM; NX_BAD("Manifest too large for host address space"); }
    if (!(man = malloc((size_t)H->manifest_size))) { e = ENOMEM; NX_BAD("out of memory for Manifest"); }
    if (!romfs_image_read(fd, man, (size_t)H->manifest_size, (off_t)H->manifest_offset)) { e = errno ? errno : EIO; NX_BAD("cannot read Manifest"); }
    { uint8_t h[32]; plat_sha256(man, (size_t)H->manifest_size, h); if (memcmp(h, H->manifest_sha, 32)) NX_BAD("Manifest SHA-256 mismatch"); }

    uint64_t bi_sz = H->block_count * 24ull, idx_tot = bi_sz + H->integrity_size;
    if (idx_tot > SIZE_MAX) { e = ENOMEM; NX_BAD("block index too large"); }
    if (idx_tot) {
        if (!(idx_raw = malloc((size_t)idx_tot))) { e = ENOMEM; NX_BAD("out of memory for block index"); }
        if (!romfs_image_read(fd, idx_raw, (size_t)bi_sz, (off_t)H->block_index_offset)) { e = errno ? errno : EIO; NX_BAD("cannot read block index"); }
        if (H->integrity_size && !romfs_image_read(fd, idx_raw + bi_sz, (size_t)H->integrity_size, (off_t)H->integrity_offset)) {
            e = errno ? errno : EIO; NX_BAD("cannot read integrity table");
        }
    }
    { uint8_t h[32]; plat_sha256(idx_raw ? idx_raw : (const uint8_t *)"", (size_t)idx_tot, h); if (memcmp(h, H->index_sha, 32)) NX_BAD("index SHA-256 mismatch"); }

    // Parse and validate Manifest header and tables
    if (memcmp(man, "NXMF", 4) || nx_u16(man + 4) != 1 || nx_u16(man + 6) != 96 || nx_u64(man + 8) != H->manifest_size || !nx_is_zero(man + 80, 16))
        NX_BAD("invalid Manifest header");
    uint32_t nd = nx_u32(man + 16), nf = nx_u32(man + 20), ne = nx_u32(man + 24), nref = nx_u32(man + 28);
    if (!nd || nd > 16777216u || nf > 33554432u || ne > 50331648u) NX_BAD("Manifest counts out of limits");
    uint64_t dir_off = nx_u64(man + 32), file_off = nx_u64(man + 40), ent_off = nx_u64(man + 48);
    uint64_t ref_off = nx_u64(man + 56), mpool_off = nx_u64(man + 64), mpool_sz = nx_u64(man + 72);
    if (mpool_sz > (2ull << 30)) NX_BAD("Manifest path pool too large");
    {   uint64_t cur = 96, end = 0;
        if (dir_off != cur) NX_BAD("bad Manifest directory table offset");
        cur = nx_align8(cur + (uint64_t)nd * 16);
        if (nf ? file_off != cur : file_off != 0) NX_BAD("bad Manifest file table offset");
        if (nf) cur = nx_align8(cur + (uint64_t)nf * 64);
        if (ne ? ent_off != cur : ent_off != 0) NX_BAD("bad Manifest entry table offset");
        if (ne) cur = nx_align8(cur + (uint64_t)ne * 24);
        if (nref ? ref_off != cur : ref_off != 0) NX_BAD("bad Manifest block reference table offset");
        if (nref) cur = nx_align4(cur + (uint64_t)nref * 4);
        if (mpool_off != nx_align8(cur) || !nx_add_u64(mpool_off, mpool_sz, &end) || end > H->manifest_size)
            NX_BAD("Manifest path pool out of bounds");
    }
    if (snap_root >= nd) NX_BAD("snapshot root directory out of range");

    // Validate block descriptors and integrity check counts
    if (!(aux = calloc(1, sizeof *aux))) { e = ENOMEM; NX_BAD("out of memory"); }
    pthread_mutex_init(&aux->lock, NULL); pthread_cond_init(&aux->cv, NULL);
    aux->nblocks = (uint32_t)H->block_count;
    aux->nchecks = (uint32_t)(H->integrity_size / 4);
    if (aux->nblocks) {
        if (!(aux->blocks = malloc((size_t)aux->nblocks * sizeof *aux->blocks)) || !(aux->checks = malloc((size_t)H->integrity_size))) {
            e = ENOMEM; NX_BAD("out of memory for block tables");
        }
        uint64_t prev_end = H->data_offset, data_end = H->data_offset + H->data_size, chk_sum = 0;
        for (uint32_t i = 0; i < aux->nblocks; i++) {
            const uint8_t *br = idx_raw + (uint64_t)i * 24;
            uint64_t boff = nx_u64(br), bend = 0;
            uint32_t bstored = nx_u32(br + 8), blogical = nx_u32(br + 12), fchk = nx_u32(br + 20);
            uint8_t codec = br[16], bflags = br[17]; uint16_t bres = nx_u16(br + 18);
            if (bflags || bres || codec > 1 || !bstored || bstored > NX_MAX_BLOCK_STORED || !blogical || blogical > NX_MAX_BLOCK_LOGICAL)
                NX_BAD("invalid block %u descriptor", i);
            if (!codec && bstored != blogical) NX_BAD("RAW block %u stored/logical size mismatch", i);
            if ((boff % 16) || boff < prev_end || !nx_add_u64(boff, bstored, &bend) || bend > data_end)
                NX_BAD("block %u data range out of bounds", i);
            if (fchk != chk_sum) NX_BAD("block %u first_check mismatch", i);
            uint32_t nchk = (blogical + NX_CHECK_SIZE - 1) / NX_CHECK_SIZE;
            if (!nx_add_u64(chk_sum, nchk, &chk_sum) || chk_sum > aux->nchecks) NX_BAD("block %u integrity check overflow", i);
            aux->blocks[i] = (nx_block){boff, bstored, blogical, fchk, codec};
            if (codec == 0) aux->nraw++; else aux->nzstd++;
            prev_end = nx_align16(bend);
        }
        if (chk_sum != aux->nchecks) NX_BAD("integrity table size mismatch");
        for (uint32_t i = 0; i < aux->nchecks; i++) aux->checks[i] = nx_u32(idx_raw + bi_sz + (uint64_t)i * 4);
    }
    free(idx_raw); idx_raw = NULL;

    // Locate "romfs" directory under snapshot's root directory
    uint64_t mpool_end = mpool_off + mpool_sz;
    uint32_t romfs_dir = NX_NONE;
    {   const uint8_t *rd = man + dir_off + (uint64_t)snap_root * 16;
        uint32_t rfirst = nx_u32(rd), rcnt = nx_u32(rd + 4);
        if (nx_u32(rd + 8) || nx_u32(rd + 12) || (uint64_t)rfirst + rcnt > ne) NX_BAD("invalid snapshot root directory");
        for (uint32_t j = 0; j < rcnt; j++) {
            const uint8_t *er = man + ent_off + (uint64_t)(rfirst + j) * 24;
            uint64_t no = nx_u64(er), nend = 0; uint32_t nl = nx_u32(er + 8), tidx = nx_u32(er + 12);
            if (no >= mpool_off && nx_add_u64(no, nl, &nend) && nend <= mpool_end && er[16] == 1 && nl == 5 && !memcmp(man + no, "romfs", 5)) {
                romfs_dir = tidx; break;
            }
        }
    }
    if (romfs_dir == NX_NONE || romfs_dir >= nd) NX_BAD("snapshot has no romfs directory");

    // Pass 1: BFS over romfs subtree to validate topology, names, files, block references, and count entries
    uint64_t n = 1, names = 0, nfiles = 0, tot_refs = 0;
    if (!(queue = malloc(sizeof *queue * (size_t)nd)) || !(seen_dir = calloc((size_t)nd, 1)) || !(seen_file = calloc(nf ? (size_t)nf : 1, 1))) {
        e = ENOMEM; NX_BAD("out of memory");
    }
    {   uint64_t qh = 0, qt = 0; queue[qt++] = romfs_dir; seen_dir[romfs_dir] = 1;
        while (qh < qt) {
            const uint8_t *d = man + dir_off + (uint64_t)queue[qh++] * 16;
            uint64_t first = nx_u32(d), cnt = nx_u32(d + 4);
            if (nx_u32(d + 8) || nx_u32(d + 12) || first > ne || cnt > ne - first) NX_BAD("Manifest directory entries out of range");
            const uint8_t *prev = NULL; uint32_t plen = 0;
            for (uint64_t j = 0; j < cnt; j++) {
                const uint8_t *er = man + ent_off + (first + j) * 24;
                uint64_t no = nx_u64(er), nend = 0; uint32_t nl = nx_u32(er + 8), tidx = nx_u32(er + 12);
                uint8_t kind = er[16], efl = er[17]; uint16_t eres0 = nx_u16(er + 18); uint32_t eres1 = nx_u32(er + 20);
                if (efl || eres0 || eres1 || (kind != 1 && kind != 2)) NX_BAD("invalid directory entry");
                if (!nl || nl > 0x300 || no < mpool_off || !nx_add_u64(no, nl, &nend) || nend > mpool_end) NX_BAD("entry name out of range");
                const uint8_t *nm = man + no;
                if (memchr(nm, 0, nl) || memchr(nm, '/', nl) || memchr(nm, '\\', nl) || (nl == 1 && nm[0] == '.') || (nl == 2 && !memcmp(nm, "..", 2)))
                    NX_BAD("invalid entry component name");
                if (prev) { int c = memcmp(prev, nm, plen < nl ? plen : nl); if (c > 0 || (!c && plen >= nl)) NX_BAD("directory entries not sorted"); }
                prev = nm; plen = nl;
                if (kind == 1) {
                    if (tidx >= nd || seen_dir[tidx]) NX_BAD("bad or repeated directory id %u", tidx);
                    seen_dir[tidx] = 1; queue[qt++] = tidx;
                } else {
                    if (tidx >= nf || seen_file[tidx]) NX_BAD("bad or repeated file id %u", tidx);
                    seen_file[tidx] = 1;
                    const uint8_t *fr = man + file_off + (uint64_t)tidx * 64;
                    uint64_t lsz = nx_u64(fr); uint32_t fref = nx_u32(fr + 8), bc = nx_u32(fr + 12);
                    uint8_t cshift = fr[16], fflags = fr[17]; uint16_t fres0 = nx_u16(fr + 18);
                    if (lsz > (uint64_t)INT64_MAX || fflags || fres0 || !nx_is_zero(fr + 52, 12) || cshift < 16 || cshift > 20)
                        NX_BAD("invalid file record %u", tidx);
                    uint64_t csz = 1ull << cshift;
                    uint64_t exp_bc = lsz ? (lsz + csz - 1) / csz : 0;
                    if (bc != exp_bc || (uint64_t)fref + bc > nref) NX_BAD("file %u block count/references mismatch", tidx);
                    for (uint32_t b = 0; b < bc; b++) {
                        uint32_t bid = nx_u32(man + ref_off + (uint64_t)(fref + b) * 4);
                        if (bid >= aux->nblocks) NX_BAD("file %u block id %u out of range", tidx, bid);
                        uint64_t exp_lsz = (b + 1 < bc || !(lsz % csz)) ? csz : (lsz % csz);
                        if (aux->blocks[bid].logical != exp_lsz) NX_BAD("file %u block %u logical size mismatch", tidx, bid);
                    }
                    nfiles++; tot_refs += bc;
                }
                n++; names += nl;
            }
        }
    }
    if (n > UINT32_MAX || nfiles > UINT32_MAX || tot_refs > UINT32_MAX) NX_BAD("too many entries");

    // Pass 2: build compact romfs_img and nx_aux file/ref tables
    size_t isz = (size_t)n * sizeof(img_ent) + (size_t)names;
    if (!(m = calloc(1, sizeof *m))) { e = ENOMEM; NX_BAD("out of memory"); }
    if (!(m->e = malloc(isz ? isz : 1))) { e = ENOMEM; NX_BAD("out of memory"); }
    if (nfiles && !(aux->files = calloc((size_t)nfiles, sizeof *aux->files))) { e = ENOMEM; NX_BAD("out of memory"); }
    if (tot_refs && !(aux->refs = malloc((size_t)tot_refs * sizeof *aux->refs))) { e = ENOMEM; NX_BAD("out of memory"); }
    m->n = n; m->names = (const char *)m->e + n * sizeof(img_ent);
    aux->nfiles = (uint32_t)nfiles; aux->nrefs = (uint32_t)tot_refs;
    { const char *mb = getenv("DM_NX_CACHE_MB"); long long v = mb && *mb ? atoll(mb) : 64; aux->budget = (uint64_t)(v > 0 ? v : 0) << 20; }
    { const char *vf = getenv("DM_NX_VERIFY"); aux->verify = !(vf && !strcmp(vf, "0")); }
    { const char *b = strrchr(file, '/'); snprintf(aux->what, sizeof aux->what, "%s snapshot %s", b ? b + 1 : file, sel[0] ? sel : "(newest)"); }

    if (!(ent_of = malloc(sizeof *ent_of * (size_t)nd))) { e = ENOMEM; NX_BAD("out of memory"); }
    {   char *np = (char *)m->names; uint64_t at = 0, k = 1, qh = 0, qt = 0; uint32_t fi = 0, ri = 0;
        memset(&m->e[0], 0, sizeof m->e[0]); queue[qt++] = romfs_dir; ent_of[0] = 0;
        while (qh < qt) {
            uint64_t me = ent_of[qh]; const uint8_t *d = man + dir_off + (uint64_t)queue[qh++] * 16;
            uint64_t first = nx_u32(d), cnt = nx_u32(d + 4);
            m->e[me].a = k; m->e[me].b = cnt;
            for (uint64_t j = 0; j < cnt; j++, k++) {
                const uint8_t *er = man + ent_off + (first + j) * 24;
                uint64_t no = nx_u64(er); uint32_t nl = nx_u32(er + 8), tidx = nx_u32(er + 12);
                img_ent *x = &m->e[k]; memset(x, 0, sizeof *x);
                x->name = (uint32_t)at; x->nlen = (uint16_t)nl; memcpy(np + at, man + no, nl); at += nl;
                if (er[16] == 1) { ent_of[qt] = k; queue[qt++] = tidx; }
                else {
                    const uint8_t *fr = man + file_off + (uint64_t)tidx * 64;
                    uint64_t lsz = nx_u64(fr); uint32_t fref = nx_u32(fr + 8), bc = nx_u32(fr + 12);
                    x->type = 1; x->b = lsz; x->a = fi; x->flags = IMG_ENT_NX;
                    aux->files[fi] = (nx_file){lsz, ri, bc, x->name, x->nlen, fr[16]};
                    for (uint32_t b = 0; b < bc; b++) aux->refs[ri++] = nx_u32(man + ref_off + (uint64_t)(fref + b) * 4);
                    fi++;
                }
            }
        }
    }
    m->fd = fd; m->aux = aux; m->aux_free = nx_aux_free;
    free(ent_of); free(queue); free(seen_dir); free(seen_file); free(man); free(cat);
    return m;

bad:
    if (m) { m->aux = NULL; m->fd = -1; romfs_image_close(m); }
    nx_aux_free(aux); free(ent_of); free(queue); free(seen_dir); free(seen_file);
    free(idx_raw); free(man); free(cat);
    *out_err = e; return NULL;
#undef NX_BAD
}

// Opens spec as an NX image; NULL with errno set and a reason in why.
static romfs_img *nx_image_open(const char *spec, char *why, size_t wn) {
    char file[1100], sel[128];
    if (!nx_split(spec, file, sizeof file, sel, sizeof sel)) { snprintf(why, wn, "not a .nx path"); errno = EINVAL; return NULL; }
    int fd = open(file, O_RDONLY);
    if (fd < 0) { int e = errno; snprintf(why, wn, "cannot open %s", file); errno = e; return NULL; }
    struct stat st;
    if (fstat(fd, &st) || st.st_size < 256) { int e = errno ? errno : EINVAL; close(fd); snprintf(why, wn, "cannot read header"); errno = e; return NULL; }
    uint64_t fsize = (uint64_t)st.st_size;
    uint8_t ha[256], hb[256]; nx_hdr a, b;
    int ok_a = romfs_image_read(fd, ha, 256, 0) && nx_parse_hdr(ha, fsize, &a);
    int ok_b = fsize >= 4096 + 256 && romfs_image_read(fd, hb, 256, 4096) && nx_parse_hdr(hb, fsize, &b);
    if (!ok_a && !ok_b) { close(fd); snprintf(why, wn, "neither header slot is valid"); errno = EINVAL; return NULL; }
    if (ok_a && ok_b && a.generation == b.generation && memcmp(ha, hb, 256) != 0) {
        close(fd); snprintf(why, wn, "header slots have identical generation but differ"); errno = EINVAL; return NULL;
    }
    nx_hdr order[2]; int ntry = 0;
    if (ok_a && ok_b) {
        if (a.generation == b.generation) order[ntry++] = a;
        else if (a.generation > b.generation) { order[ntry++] = a; order[ntry++] = b; }
        else { order[ntry++] = b; order[ntry++] = a; }
    } else order[ntry++] = ok_a ? a : b;

    int err = EINVAL;
    for (int i = 0; i < ntry; i++) {
        romfs_img *m = nx_mount_hdr(fd, &order[i], file, sel, why, wn, &err);
        if (m) return m;
    }
    close(fd); errno = err; return NULL;
}

// ---- Block cache & read path ----

static void nx_lru_remove(nx_aux *a, nx_cache_ent *e) {
    if (e->prev) e->prev->next = e->next; else if (a->lru_head == e) a->lru_head = e->next;
    if (e->next) e->next->prev = e->prev; else if (a->lru_tail == e) a->lru_tail = e->prev;
    e->prev = e->next = NULL;
}

static void nx_lru_push_front(nx_aux *a, nx_cache_ent *e) {
    e->prev = NULL; e->next = a->lru_head;
    if (a->lru_head) a->lru_head->prev = e; else a->lru_tail = e;
    a->lru_head = e;
}

// Caller holds a->lock. Frees unpinned LRU tail entries until held <= budget.
static void nx_cache_trim(nx_aux *a) {
    nx_cache_ent *e = a->lru_tail;
    while (a->held > a->budget && e) {
        nx_cache_ent *prev = e->prev;
        if (!e->refs && !e->loading && e->buf) {
            uint32_t bi = e->block_id % NX_CACHE_BUCKETS;
            nx_cache_ent **pp = &a->buckets[bi];
            while (*pp && *pp != e) pp = &(*pp)->hnext;
            if (*pp == e) *pp = e->hnext;
            nx_lru_remove(a, e);
            a->held -= e->logical_size;
            free(e->buf); free(e);
        }
        e = prev;
    }
}

static nx_cache_ent *nx_cache_find(nx_aux *a, uint32_t bid) {
    for (nx_cache_ent *e = a->buckets[bid % NX_CACHE_BUCKETS]; e; e = e->hnext)
        if (e->block_id == bid) return e;
    return NULL;
}

// Reads and verifies a ZSTD block into the cache (or returns cached copy), pinned with refs++.
static nx_cache_ent *nx_block_get(romfs_img *m, uint32_t bid, const char *fname, uint16_t flen) {
    nx_aux *a = m->aux;
    const nx_block *blk = &a->blocks[bid];
    pthread_mutex_lock(&a->lock);
    for (;;) {
        nx_cache_ent *e = nx_cache_find(a, bid);
        if (e) {
            e->refs++;
            while (e->loading) pthread_cond_wait(&a->cv, &a->lock);
            if (!e->bad && e->buf) {
                nx_lru_remove(a, e);
                nx_lru_push_front(a, e);
                pthread_mutex_unlock(&a->lock);
                return e;
            }
            int bad = e->bad;
            if (--e->refs == 0 && bad) free(e);
            if (bad) { pthread_mutex_unlock(&a->lock); errno = EIO; return NULL; }
            continue;
        }
        nx_cache_ent *ne = calloc(1, sizeof *ne);
        if (!ne) { pthread_mutex_unlock(&a->lock); errno = ENOMEM; return NULL; }
        ne->block_id = bid; ne->logical_size = blk->logical; ne->refs = 1; ne->loading = 1;
        uint32_t bi = bid % NX_CACHE_BUCKETS;
        ne->hnext = a->buckets[bi]; a->buckets[bi] = ne;
        pthread_mutex_unlock(&a->lock);

        uint8_t *out = NULL; int err = 0;
#ifndef DM_NUP_ZSTD
        LOG("[RED] fs NX %s: compressed block %u in %.*s requires DM_NUP_ZSTD", a->what, bid, (int)flen, fname);
        err = ENOTSUP;
#else
        uint8_t *src = NULL;
        if (!(out = malloc(blk->logical)) || !(src = malloc(blk->stored))) {
            LOG("[RED] fs NX %s: out of memory decompressing %.*s (block %u)", a->what, (int)flen, fname, bid);
            err = ENOMEM;
        } else if (!romfs_image_read(m->fd, src, blk->stored, (off_t)blk->off)) {
            err = errno ? errno : EIO;
            LOG("[RED] fs NX %s: I/O error reading %.*s (block %u, errno=%d)", a->what, (int)flen, fname, bid, err);
        } else {
            size_t r = ZSTD_decompress(out, blk->logical, src, blk->stored);
            if (ZSTD_isError(r) || r != blk->logical) {
                LOG("[RED] fs NX %s: zstd error in %.*s (block %u): %s", a->what, (int)flen, fname, bid,
                    ZSTD_isError(r) ? ZSTD_getErrorName(r) : "decoded size mismatch");
                err = EIO;
            } else if (a->verify) {
                uint32_t nchk = (blk->logical + NX_CHECK_SIZE - 1) / NX_CHECK_SIZE;
                for (uint32_t s = 0; s < nchk; s++) {
                    size_t off = (size_t)s * NX_CHECK_SIZE;
                    size_t len = off + NX_CHECK_SIZE <= blk->logical ? NX_CHECK_SIZE : blk->logical - off;
                    uint32_t got = nx_crc32c(out + off, len), want = a->checks[blk->first_check + s];
                    if (got != want) {
                        LOG("[RED] fs NX %s: CRC-32C mismatch in %.*s (block %u slice %u, got %#08x want %#08x)",
                            a->what, (int)flen, fname, bid, s, got, want);
                        err = EIO; break;
                    }
                }
            }
        }
        free(src);
#endif
        pthread_mutex_lock(&a->lock);
        ne->loading = 0;
        if (err) {
            free(out);
            nx_cache_ent **pp = &a->buckets[bi];
            while (*pp && *pp != ne) pp = &(*pp)->hnext;
            if (*pp == ne) *pp = ne->hnext;
            ne->bad = 1;
            pthread_cond_broadcast(&a->cv);
            if (--ne->refs == 0) free(ne);
            pthread_mutex_unlock(&a->lock);
            errno = err;
            return NULL;
        }
        ne->buf = out;
        nx_lru_push_front(a, ne);
        a->held += blk->logical;
        a->ndec++;
        nx_cache_trim(a);
        pthread_cond_broadcast(&a->cv);
        pthread_mutex_unlock(&a->lock);
        return ne;
    }
}

static void nx_block_put(nx_aux *a, nx_cache_ent *e) {
    pthread_mutex_lock(&a->lock);
    if (e->refs) e->refs--;
    nx_cache_trim(a);
    pthread_mutex_unlock(&a->lock);
}

static int nx_pread_guest(int fd, uint8_t *dst, size_t len, off_t off) {
    size_t got = 0; int reprep = 0;
    while (got < len) {
        ssize_t r = pread(fd, dst + got, len - got, off + (off_t)got);
        if (r < 0) {
            if (errno == EINTR) continue;
            if (errno == EFAULT && reprep++ < 8) {
                static _Atomic int once;
                if (!once++) LOG("[fs] ReadFile into %p+%#zx: the destination was write-protected again during the read, prepared again at +%#zx (first case)", dst, len, got);
                dm_guest_write_prepare(dst + got, len - got);
                continue;
            }
            if (errno == EFAULT) {
                int e = errno; uint64_t ra = (uint64_t)(uintptr_t)(dst + got), rs = 0; int prot = -1;
                int kr = plat_mem_region((uint64_t)(uintptr_t)(dst + got), &ra, &rs, &prot);
                LOG("[RED] fs ReadFile into %p+%#zx failed at +%#zx (EFAULT): VM region %#lx+%#lx protection %d%s", dst, len, got, (unsigned long)ra, (unsigned long)rs, !kr ? prot : -1, !kr ? "" : " (none)");
                errno = e;
            }
            return 0;
        }
        if (r == 0) { errno = EIO; return 0; }
        got += (size_t)r;
    }
    return 1;
}

static int nx_read_raw_block(romfs_img *m, uint32_t bid, size_t off0, uint8_t *dst, size_t take, const char *fname, uint16_t flen) {
    nx_aux *a = m->aux;
    const nx_block *blk = &a->blocks[bid];
    if (!a->verify) return nx_pread_guest(m->fd, dst, take, (off_t)(blk->off + off0));
    uint32_t s0 = (uint32_t)(off0 / NX_CHECK_SIZE), s1 = (uint32_t)((off0 + take - 1) / NX_CHECK_SIZE);
    uint8_t sbuf[NX_CHECK_SIZE];
    for (uint32_t s = s0; s <= s1;) {
        size_t soff = (size_t)s * NX_CHECK_SIZE;
        size_t send = soff + NX_CHECK_SIZE <= blk->logical ? soff + NX_CHECK_SIZE : blk->logical;
        if (soff >= off0 && send <= off0 + take) {
            // Contiguous run of full slices: pread straight into guest buffer, then verify each slice
            uint32_t s_end = s;
            size_t run_end = send;
            while (s_end + 1 <= s1) {
                size_t noff = (size_t)(s_end + 1) * NX_CHECK_SIZE;
                size_t nend = noff + NX_CHECK_SIZE <= blk->logical ? noff + NX_CHECK_SIZE : blk->logical;
                if (nend > off0 + take) break;
                s_end++; run_end = nend;
            }
            uint8_t *gptr = dst + (soff - off0);
            if (!nx_pread_guest(m->fd, gptr, run_end - soff, (off_t)(blk->off + soff))) return 0;
            for (uint32_t k = s; k <= s_end; k++) {
                size_t koff = (size_t)k * NX_CHECK_SIZE;
                size_t kend = koff + NX_CHECK_SIZE <= blk->logical ? koff + NX_CHECK_SIZE : blk->logical;
                uint32_t got = nx_crc32c(dst + (koff - off0), kend - koff), want = a->checks[blk->first_check + k];
                if (got != want) {
                    LOG("[RED] fs NX %s: CRC-32C mismatch in %.*s (block %u slice %u, got %#08x want %#08x)",
                        a->what, (int)flen, fname, bid, k, got, want);
                    errno = EIO; return 0;
                }
            }
            s = s_end + 1;
        } else {
            size_t slen = send - soff;
            if (!romfs_image_read(m->fd, sbuf, slen, (off_t)(blk->off + soff))) return 0;
            uint32_t got = nx_crc32c(sbuf, slen), want = a->checks[blk->first_check + s];
            if (got != want) {
                LOG("[RED] fs NX %s: CRC-32C mismatch in %.*s (block %u slice %u, got %#08x want %#08x)",
                    a->what, (int)flen, fname, bid, s, got, want);
                errno = EIO; return 0;
            }
            size_t c0 = soff > off0 ? soff : off0;
            size_t c1 = send < off0 + take ? send : off0 + take;
            memcpy(dst + (c0 - off0), sbuf + (c0 - soff), c1 - c0);
            s++;
        }
    }
    return 1;
}

// Reads sz bytes from NX file entry x at byte offset off into buf. Caller must call dm_guest_write_prepare(buf, sz) first.
static int nx_read(romfs_img *m, const img_ent *x, uint64_t off, void *buf, size_t sz, size_t *out) {
    *out = 0;
    if (!sz) return 1;
    nx_aux *a = m->aux;
    const nx_file *f = &a->files[x->a];
    const char *fname = m->names + f->name;
    size_t csz = (size_t)1 << f->chunk_shift;
    size_t done = 0;
    while (done < sz) {
        uint64_t pos = off + done;
        uint32_t bidx = (uint32_t)(pos >> f->chunk_shift);
        size_t boff = (size_t)(pos & (csz - 1));
        uint32_t bid = a->refs[f->first_ref + bidx];
        const nx_block *blk = &a->blocks[bid];
        size_t avail = blk->logical - boff;
        size_t take = sz - done < avail ? sz - done : avail;
        if (blk->codec == 0) {
            if (!nx_read_raw_block(m, bid, boff, (uint8_t *)buf + done, take, fname, f->nlen)) return 0;
        } else {
            nx_cache_ent *ce = nx_block_get(m, bid, fname, f->nlen);
            if (!ce) return 0;
            memcpy((uint8_t *)buf + done, ce->buf + boff, take);
            nx_block_put(a, ce);
        }
        done += take;
    }
    *out = done;
    return 1;
}
