// Synthetic NX container unit test (nx_image.h, hle_fs.c, SPEC.md (0,2) and (0,3)).
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../hle_fs.c"
#include "nup_fixture_zst.h"

struct dm_flags g_flag; uint64_t g_t0_ns; uint64_t now_ns(void) { return 0; }
void dm_guest_write_prepare(void *a, size_t n) { (void)a; (void)n; }

static int fails, checks;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; printf("  ✗ "); } else printf("  ✓ "); printf(__VA_ARGS__); printf("\n"); } while (0)

static void w16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }
static void w32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void w64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }

static void nup_fixture_text(char *out) { for (int i = 0; i < 3000; i++) out += sprintf(out, "line %05d of the compressed file\n", i); }

static uint8_t g_raw4096[4096];
static uint8_t g_zst64k[13] = {0x28, 0xB5, 0x2F, 0xFD, 0xA0, 0x00, 0x00, 0x01, 0x00, 0x03, 0x00, 0x08, 'Z'};

// Writes an NX (0,2) or (0,3) fixture into out (returns size; out must hold >= 256 KiB).
// Snapshots: base (v0), update/65536 (type 1), update/131072 (type 3), addon/0400486024dec001 (type 2).
static size_t nx_build_fixture(uint8_t *out, int minor_ver, uint32_t bad_req_ext) {
    memset(out, 0, 262144);
    for (int i = 0; i < 4096; i++) g_raw4096[i] = (uint8_t)(i * 7 + 3);
    static uint8_t multi_raw[65536 + 4096];
    memset(multi_raw, 'Z', 65536);
    memcpy(multi_raw + 65536, g_raw4096, 4096);
    static char ztext[102001];
    nup_fixture_text(ztext);

    // 8 blocks:
    // 0: RAW "hello" (5)
    // 1: ZSTD 65536 * 'Z' (13 stored, 65536 logical)
    // 2: RAW g_raw4096 (4096)
    // 3: ZSTD nup_zst (nup_zst_len stored, 102000 logical)
    // 4: RAW "base-top" (8)
    // 5: RAW "HELLO" (5)
    // 6: RAW "two" (3)
    // 7: RAW "dlc" (3)
    const uint8_t *blk_src[8] = {(const uint8_t *)"hello", g_zst64k, g_raw4096, nup_zst, (const uint8_t *)"base-top", (const uint8_t *)"HELLO", (const uint8_t *)"two", (const uint8_t *)"dlc"};
    const uint8_t *blk_log[8] = {(const uint8_t *)"hello", multi_raw, g_raw4096, (const uint8_t *)ztext, (const uint8_t *)"base-top", (const uint8_t *)"HELLO", (const uint8_t *)"two", (const uint8_t *)"dlc"};
    uint32_t blk_st[8] = {5, sizeof g_zst64k, 4096, nup_zst_len, 8, 5, 3, 3};
    uint32_t blk_ls[8] = {5, 65536, 4096, 102000, 8, 5, 3, 3};
    uint8_t blk_codec[8] = {0, 1, 0, 1, 0, 0, 0, 0};

    uint64_t at = 8192;
    uint64_t blk_off[8];
    uint32_t checks_arr[32];
    uint32_t blk_fchk[8];
    uint32_t nchecks = 0;
    for (int i = 0; i < 8; i++) {
        blk_off[i] = at;
        memcpy(out + at, blk_src[i], blk_st[i]);
        at = (at + blk_st[i] + 15) & ~15ull;
        blk_fchk[i] = nchecks;
        uint32_t nc = (blk_ls[i] + NX_CHECK_SIZE - 1) / NX_CHECK_SIZE;
        for (uint32_t s = 0; s < nc; s++) {
            size_t so = (size_t)s * NX_CHECK_SIZE;
            size_t sl = so + NX_CHECK_SIZE <= blk_ls[i] ? NX_CHECK_SIZE : blk_ls[i] - so;
            checks_arr[nchecks++] = nx_crc32c(blk_log[i] + so, sl);
        }
    }
    uint64_t data_end = blk_off[7] + blk_st[7];
    uint64_t data_size = data_end - 8192;

    // Manifest:
    // Dirs:
    //  0: snap0 root -> ["romfs"(1)]
    //  1: snap0 romfs -> ["data"(2), "top.txt"(f4)]
    //  2: snap0 data -> ["a"(f0), "e"(f1), "multi"(f2), "z"(f3)]
    //  3: snap1 root -> ["romfs"(4)]
    //  4: snap1 romfs -> ["data"(5)]
    //  5: snap1 data -> ["a"(f5), "multi"(f2)]
    //  6: snap2 root -> ["romfs"(7)]
    //  7: snap2 romfs -> ["v2.txt"(f6)]
    //  8: snap3 root -> ["romfs"(9)]
    //  9: snap3 romfs -> ["d.txt"(f7)]
    // (For minor_ver == 3, add dir 10 for source resource root -> ["src.bin"(f0)])
    struct dir_spec { int first, count; } D[11] = {
        {0, 1}, {1, 2}, {3, 4}, {7, 1}, {8, 1}, {9, 2}, {11, 1}, {12, 1}, {13, 1}, {14, 1}, {15, 1}
    };
    struct ent_spec { const char *name; uint32_t target; uint8_t kind; } E[16] = {
        {"romfs", 1, 1},
        {"data", 2, 1}, {"top.txt", 4, 2},
        {"a", 0, 2}, {"e", 1, 2}, {"multi", 2, 2}, {"z", 3, 2},
        {"romfs", 4, 1},
        {"data", 5, 1},
        {"a", 5, 2}, {"multi", 2, 2},
        {"romfs", 7, 1},
        {"v2.txt", 6, 2},
        {"romfs", 9, 1},
        {"d.txt", 7, 2},
        {"src.bin", 0, 2}
    };
    uint32_t nd = minor_ver == 3 ? 11 : 10, ne = minor_ver == 3 ? 16 : 15, nf = 8;
    struct file_spec { uint64_t size; uint32_t fref, bc; uint8_t cshift; const uint8_t *data; } F[8] = {
        {5, 0, 1, 16, blk_log[0]},
        {0, 1, 0, 16, (const uint8_t *)""},
        {65536 + 4096, 1, 2, 16, multi_raw},
        {102000, 3, 1, 17, blk_log[3]},
        {8, 4, 1, 16, blk_log[4]},
        {5, 5, 1, 16, blk_log[5]},
        {3, 6, 1, 16, blk_log[6]},
        {3, 7, 1, 16, blk_log[7]}
    };
    uint32_t refs[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint32_t nref = 8;

    uint8_t man[4096] = {0};
    uint64_t m_dir = 96;
    uint64_t m_file = (m_dir + (uint64_t)nd * 16 + 7) & ~7ull;
    uint64_t m_ent = (m_file + (uint64_t)nf * 64 + 7) & ~7ull;
    uint64_t m_ref = (m_ent + (uint64_t)ne * 24 + 7) & ~7ull;
    uint64_t m_pool = (m_ref + (uint64_t)nref * 4 + 7) & ~7ull;
    uint64_t m_pool_at = 0;
    for (uint32_t i = 0; i < nd; i++) {
        w32(man + m_dir + i * 16, (uint32_t)D[i].first);
        w32(man + m_dir + i * 16 + 4, (uint32_t)D[i].count);
    }
    for (uint32_t i = 0; i < nf; i++) {
        uint8_t *r = man + m_file + i * 64;
        w64(r, F[i].size); w32(r + 8, F[i].fref); w32(r + 12, F[i].bc); r[16] = F[i].cshift;
        plat_sha256(F[i].data, (size_t)F[i].size, r + 20);
    }
    for (uint32_t i = 0; i < ne; i++) {
        uint8_t *r = man + m_ent + i * 24;
        size_t nl = strlen(E[i].name);
        w64(r, m_pool + m_pool_at); w32(r + 8, (uint32_t)nl); w32(r + 12, E[i].target); r[16] = E[i].kind;
        memcpy(man + m_pool + m_pool_at, E[i].name, nl);
        m_pool_at += nl;
    }
    for (uint32_t i = 0; i < nref; i++) w32(man + m_ref + i * 4, refs[i]);
    uint64_t man_sz = m_pool + m_pool_at;
    memcpy(man, "NXMF", 4); w16(man + 4, 1); w16(man + 6, 96); w64(man + 8, man_sz);
    w32(man + 16, nd); w32(man + 20, nf); w32(man + 24, ne); w32(man + 28, nref);
    w64(man + 32, m_dir); w64(man + 40, m_file); w64(man + 48, m_ent); w64(man + 56, m_ref);
    w64(man + 64, m_pool); w64(man + 72, m_pool_at);

    // Catalog
    uint8_t cat[4096] = {0};
    uint32_t ext_cnt = 2 + (minor_ver == 3 ? 1 : 0) + (bad_req_ext ? 1 : 0);
    uint64_t c_app = 128;
    uint64_t c_prog = (c_app + 80 + 7) & ~7ull;
    uint64_t c_snap = (c_prog + 64 + 7) & ~7ull;
    uint64_t c_ext = (c_snap + 4 * 120 + 7) & ~7ull;
    uint64_t c_pool = (c_ext + ext_cnt * 32 + 7) & ~7ull;
    uint64_t c_extp = (c_pool + 7) & ~7ull;
    w64(cat + c_app, 0x0400486024DEC000ull); cat[c_app + 8] = 2; cat[c_app + 9] = 3;
    w32(cat + c_app + 12, 0); w32(cat + c_app + 16, 1);
    w32(cat + c_app + 20, 0); w32(cat + c_app + 24, 4);
    w32(cat + c_app + 28, NX_NONE); w32(cat + c_app + 36, NX_NONE); w32(cat + c_app + 44, NX_NONE);
    w32(cat + c_app + 56, NX_NONE); w32(cat + c_app + 60, NX_NONE); w32(cat + c_app + 68, NX_NONE);
    w64(cat + c_prog, 0x0400486024DEC000ull); w32(cat + c_prog + 8, 0);
    w32(cat + c_prog + 32, 0); w32(cat + c_prog + 36, 3); w32(cat + c_prog + 40, NX_NONE);

    // Snapshots: 0 base, 1 update/65536, 2 update/131072, 3 addon/0400486024dec001
    uint32_t s_pidx[4] = {0, 0, 0, NX_NONE};
    uint8_t s_type[4] = {0, 1, 3, 2};
    uint8_t s_flags[4] = {0, 0, 0, 2};
    uint32_t s_ver[4] = {0, 65536, 131072, 0};
    uint32_t s_root[4] = {0, 3, 6, 8};
    for (int i = 0; i < 4; i++) {
        uint8_t *s = cat + c_snap + i * 120;
        w32(s, s_pidx[i]); s[4] = s_type[i]; s[5] = s_flags[i]; w32(s + 8, s_ver[i]); w32(s + 12, s_root[i]);
        w32(s + 24, NX_NONE);
        if (i == 3) w64(s + 32, 0x0400486024DEC001ull);
    }
    uint32_t ei = 0;
    uint32_t etypes[4] = {0x00010004u, 0x00010005u, 0x00010008u, bad_req_ext};
    for (int k = 0; k < 4; k++) {
        if (k == 2 && minor_ver != 3) continue;
        if (k == 3 && !bad_req_ext) continue;
        uint8_t *er = cat + c_ext + ei * 32;
        w32(er, etypes[k]); w32(er + 4, 1); w16(er + 8, 1); w32(er + 12, 0);
        w64(er + 16, c_extp); w64(er + 24, 16);
        memcpy(cat + c_extp, "NXSM", 4);
        c_extp += 16;
        ei++;
    }
    uint64_t cat_sz = c_extp;
    memcpy(cat, "NXCT", 4); w16(cat + 4, 1); w16(cat + 6, 128); w64(cat + 8, cat_sz);
    w32(cat + 20, 1); w32(cat + 24, 1); w32(cat + 28, 4); w32(cat + 40, ext_cnt);
    w64(cat + 48, c_app); w64(cat + 56, c_prog); w64(cat + 64, c_snap); w64(cat + 88, c_ext);
    w64(cat + 96, c_pool); w64(cat + 104, 0);

    // Place tail: Catalog -> Manifest -> Block Index -> Integrity Table
    uint64_t cat_off = (data_end + 15) & ~15ull;
    memcpy(out + cat_off, cat, cat_sz);
    uint64_t man_off = (cat_off + cat_sz + 15) & ~15ull;
    memcpy(out + man_off, man, man_sz);
    uint64_t bi_off = (man_off + man_sz + 15) & ~15ull;
    uint8_t idx_buf[8 * 24 + 32 * 4];
    for (int i = 0; i < 8; i++) {
        uint8_t *b = idx_buf + i * 24; memset(b, 0, 24);
        w64(b, blk_off[i]); w32(b + 8, blk_st[i]); w32(b + 12, blk_ls[i]); b[16] = blk_codec[i]; w32(b + 20, blk_fchk[i]);
    }
    memcpy(out + bi_off, idx_buf, 8 * 24);
    uint64_t int_off = (bi_off + 8 * 24 + 15) & ~15ull;
    for (uint32_t i = 0; i < nchecks; i++) w32(idx_buf + 8 * 24 + i * 4, checks_arr[i]);
    memcpy(out + int_off, idx_buf + 8 * 24, nchecks * 4);
    uint64_t arch_sz = int_off + nchecks * 4;

    uint8_t hdr[256] = {0};
    memcpy(hdr, "NXAR", 4); w16(hdr + 4, 0); w16(hdr + 6, (uint16_t)minor_ver); w32(hdr + 8, 256); w16(hdr + 12, 1);
    w64(hdr + 16, arch_sz); w64(hdr + 24, cat_off); w64(hdr + 32, cat_sz);
    w64(hdr + 56, 8192); w64(hdr + 64, data_size);
    w64(hdr + 72, man_off); w64(hdr + 80, man_sz);
    w64(hdr + 88, bi_off); w64(hdr + 96, 8);
    w64(hdr + 104, int_off); w64(hdr + 112, nchecks * 4);
    plat_sha256(cat, (size_t)cat_sz, hdr + 120);
    plat_sha256(man, (size_t)man_sz, hdr + 152);
    plat_sha256(idx_buf, 8 * 24 + nchecks * 4, hdr + 184);
    w32(hdr + 224, NX_CHECK_SIZE);
    w32(hdr + 252, nx_crc32c(hdr, 252));
    memcpy(out, hdr, 256);
    memcpy(out + 4096, hdr, 256);
    return (size_t)arch_sz;
}

static void write_file(const char *p, const uint8_t *b, size_t n) { FILE *f = fopen(p, "wb"); fwrite(b, 1, n, f); fclose(f); }
static int read_fs(const char *path, uint8_t *out, size_t n, size_t *got) {
    uint64_t h; if (h_fs_OpenFile(&h, path, MODE_READ)) return 0;
    Result r = h_fs_ReadFile(got, h, 0, out, n); h_fs_CloseFile(h); return !r;
}

typedef struct { int tid; const uint8_t *want_multi; const uint8_t *want_z; int bad; } mt_arg;
static void *mt_reader(void *p) {
    mt_arg *a = p;
    uint8_t buf[8192];
    uint64_t hm, hz;
    if (h_fs_OpenFile(&hm, "rom:/data/multi", MODE_READ) || h_fs_OpenFile(&hz, "rom:/data/z", MODE_READ)) { a->bad = 1; return NULL; }
    uint32_t rng = 0x12345678u ^ (uint32_t)(a->tid * 2654435761u);
    for (int i = 0; i < 64; i++) {
        rng = rng * 1664525u + 1013904223u;
        uint64_t off = rng % 69000;
        size_t len = 1 + (rng >> 16) % 4096;
        if (off + len > 69632) len = 69632 - (size_t)off;
        size_t got = 0;
        if (h_fs_ReadFile(&got, hm, (int64_t)off, buf, len) != RES_OK || got != len || memcmp(buf, a->want_multi + off, len)) { a->bad = 1; break; }
        off = rng % 101000; len = 1 + (rng >> 12) % 4096;
        if (off + len > 102000) len = 102000 - (size_t)off;
        if (h_fs_ReadFile(&got, hz, (int64_t)off, buf, len) != RES_OK || got != len || memcmp(buf, a->want_z + off, len)) { a->bad = 1; break; }
    }
    h_fs_CloseFile(hm); h_fs_CloseFile(hz);
    return NULL;
}

int main(void) {
    char root[1024], np[1100], np3[1100], spec[1200];
    const char *sp = getenv("TMPDIR");
    snprintf(root, sizeof root, "%s/t_nx.XXXXXX", sp ? sp : "/tmp");
    if (!mkdtemp(root)) { perror("mkdtemp"); return 1; }

    static uint8_t buf[262144], got[110000], want_multi[69632];
    static char want_z[102001];
    nup_fixture_text(want_z);
    size_t n = nx_build_fixture(buf, 2, 0);
    memset(want_multi, 'Z', 65536); memcpy(want_multi + 65536, g_raw4096, 4096);
    snprintf(np, sizeof np, "%s/game.nx", root);
    write_file(np, buf, n);

    printf("NX container\n");
    CHECK(nx_crc32c("123456789", 9) == 0xe3069283u, "CRC-32C matches standard test vector 0xe3069283");
    unsetenv("DM_ROMFS_OVERLAY"); unsetenv("DM_NX_CACHE_MB"); unsetenv("DM_NX_VERIFY"); g_nmnt = 0;

    // 1. Default selector -> highest update (131072, standalone update)
    setenv("DM_ROMFS", np, 1);
    CHECK(h_fs_MountRom("rom", NULL, 0) == RES_OK && g_mnt[0].img, "a .nx with no selector mounts");
    size_t got_n = 0;
    CHECK(read_fs("rom:/v2.txt", got, sizeof got, &got_n) && got_n == 3 && !memcmp(got, "two", 3), "no selector picks highest-version update (131072)");
    { uint64_t h; CHECK(h_fs_OpenFile(&h, "rom:/data/a", MODE_READ) == RES_PATH_NOT_FOUND, "update/65536 files are absent in 131072"); }

    // 2. update/65536/romfs selector
    g_nmnt = 0; snprintf(spec, sizeof spec, "%s:update/65536/romfs", np); setenv("DM_ROMFS", spec, 1);
    CHECK(h_fs_MountRom("rom", NULL, 0) == RES_OK && read_fs("rom:/data/a", got, sizeof got, &got_n) && got_n == 5 && !memcmp(got, "HELLO", 5), "update/65536/romfs selects update 65536");

    // 3. :base, empty file, multi-block ZSTD+RAW file across boundary, Huffman ZSTD file, directory listing
    g_nmnt = 0; snprintf(spec, sizeof spec, "%s:base", np); setenv("DM_ROMFS", spec, 1);
    CHECK(h_fs_MountRom("rom", NULL, 0) == RES_OK, ":base mounts");
    CHECK(read_fs("rom:/top.txt", got, sizeof got, &got_n) && got_n == 8 && !memcmp(got, "base-top", 8), "reads raw file top.txt");
    CHECK(read_fs("rom:/data/e", got, sizeof got, &got_n) && got_n == 0, "empty file reads 0 bytes");
    {   uint64_t h; int64_t sz = -1; size_t r1 = 0, r2 = 0, r3 = 0;
        CHECK(h_fs_OpenFile(&h, "rom:/data/multi", MODE_READ) == RES_OK && h_fs_GetFileSize(&sz, h) == RES_OK && sz == 69632, "multi-block file size is 69632");
        CHECK(h_fs_ReadFile(&r1, h, 0, got, 60000) == RES_OK && h_fs_ReadFile(&r2, h, 60000, got + 60000, 20000) == RES_OK && r1 == 60000 && r2 == 9632 && !memcmp(got, want_multi, 69632),
              "multi-block ZSTD+RAW file reads across 64 KiB boundary (%zu + %zu)", r1, r2);
        CHECK(h_fs_ReadFile(&r3, h, 69632, got, 16) == RES_OK && r3 == 0, "read at EOF returns 0 bytes");
        h_fs_CloseFile(h);
    }
    {   uint64_t h; size_t r1 = 0, r2 = 0;
        CHECK(h_fs_OpenFile(&h, "rom:/data/z", MODE_READ) == RES_OK && h_fs_ReadFile(&r1, h, 1000, got, 50000) == RES_OK && h_fs_ReadFile(&r2, h, 51000, got + 50000, 60000) == RES_OK
              && r1 == 50000 && r2 == 51000 && !memcmp(got, want_z + 1000, 101000), "compressed file > 1 block (102000 bytes) reads at unaligned offset");
        h_fs_CloseFile(h);
    }
    {   uint64_t dh; DirectoryEntry de[8]; int64_t cnt = 0;
        CHECK(h_fs_OpenDirectory(&dh, "rom:/data", DIRMODE_DIR | DIRMODE_FILE) == RES_OK && h_fs_ReadDirectory(&cnt, de, dh, 8) == RES_OK
              && cnt == 4 && !strcmp(de[0].name, "a") && !strcmp(de[1].name, "e") && !strcmp(de[2].name, "multi") && !strcmp(de[3].name, "z"),
              "lists nested directory entries sorted (%lld)", (long long)cnt);
        h_fs_CloseDirectory(dh);
    }

    // 4. Multi-threaded concurrent reads
    {   pthread_t th[8]; mt_arg args[8]; int any_bad = 0;
        for (int i = 0; i < 8; i++) { args[i] = (mt_arg){i, want_multi, (const uint8_t *)want_z, 0}; pthread_create(&th[i], NULL, mt_reader, &args[i]); }
        for (int i = 0; i < 8; i++) { pthread_join(th[i], NULL); any_bad |= args[i].bad; }
        CHECK(!any_bad, "concurrent reads across 8 threads match all bytes");
    }

    // 5. DM_NX_CACHE_MB=0
    g_nmnt = 0; setenv("DM_NX_CACHE_MB", "0", 1); setenv("DM_ROMFS", spec, 1);
    CHECK(h_fs_MountRom("rom", NULL, 0) == RES_OK && read_fs("rom:/data/z", got, sizeof got, &got_n) && got_n == 102000, "reads with DM_NX_CACHE_MB=0");
    { nx_aux *a = g_mnt[0].img->aux; CHECK(a->held == 0 && a->lru_head == NULL, "DM_NX_CACHE_MB=0 evicts immediately after read (held=%llu)", (unsigned long long)a->held); }
    unsetenv("DM_NX_CACHE_MB");

    // 6. Add-on mount (DM_AOC_DIR/<index>.nx) and addon/<tid>
    {   char aocd[1100], ln[1200]; size_t cache = 0;
        snprintf(aocd, sizeof aocd, "%s/aocnx", root); mkdir(aocd, 0755);
        snprintf(ln, sizeof ln, "%s/1.nx", aocd); unlink(ln);
        CHECK(!symlink(np, ln), "link 1.nx to the .nx"); setenv("DM_AOC_DIR", aocd, 1);
        CHECK(h_fs_QueryAocCache(&cache, 1) == RES_OK && cache > 0x30, "NX add-on reports table size (%zu)", cache);
        CHECK(h_fs_MountAoc("aocx", 1, NULL, 0) == RES_OK && read_fs("aocx:/d.txt", got, sizeof got, &got_n) && got_n == 3 && !memcmp(got, "dlc", 3), "add-on 1 mounts from 1.nx");
        unsetenv("DM_AOC_DIR");
    }
    snprintf(spec, sizeof spec, "%s:addon/0400486024dec001", np);
    { romfs_img *m = img_open(spec); CHECK(m && m->n == 2, "addon/<title ID> selects the add-on"); romfs_image_close(m); }

    // 7. Version (0,3) with 0x00010008 Source Resources accepted
    size_t n3 = nx_build_fixture(buf, 3, 0);
    snprintf(np3, sizeof np3, "%s/game3.nx", root); write_file(np3, buf, n3);
    { romfs_img *m = img_open(np3); CHECK(m != NULL, "version (0,3) with required 0x00010004/0x00010005/0x00010008 opens"); romfs_image_close(m); }

    // 8. CRC-32C verification & DM_NX_VERIFY=0
    n = nx_build_fixture(buf, 2, 0);
    {   char bad[1100]; snprintf(bad, sizeof bad, "%s/corrupt_raw.nx", root);
        static uint8_t cb[262144]; memcpy(cb, buf, n);
        cb[8192] ^= 0x55;   // flip first byte of RAW block 0 ("hello")
        write_file(bad, cb, n);
        g_nmnt = 0; setenv("DM_ROMFS", bad, 1);
        char bspec[1200]; snprintf(bspec, sizeof bspec, "%s:base", bad); setenv("DM_ROMFS", bspec, 1);
        CHECK(h_fs_MountRom("rom", NULL, 0) == RES_OK, "corrupt RAW block mounts");
        CHECK(!read_fs("rom:/data/a", got, sizeof got, &got_n), "corrupt RAW slice fails CRC-32C check");
        g_nmnt = 0; setenv("DM_NX_VERIFY", "0", 1);
        CHECK(h_fs_MountRom("rom", NULL, 0) == RES_OK && read_fs("rom:/data/a", got, sizeof got, &got_n) && got_n == 5 && got[0] == ('h' ^ 0x55),
              "DM_NX_VERIFY=0 skips CRC-32C slice check");
        unsetenv("DM_NX_VERIFY");
    }

    // 9. Spec validation rejections + dual-generation fallback
    {   char bp[1100], upper[1100];
        snprintf(upper, sizeof upper, "%s/GAME.NX", root); write_file(upper, buf, n);
        CHECK(!nx_is_spec(upper) && !img_open(upper), "uppercase .NX extension is refused");

        snprintf(bp, sizeof bp, "%s/bad.nx", root);
        // Unknown required extension
        size_t nu = nx_build_fixture(buf, 2, 0x00019999u);
        write_file(bp, buf, nu); CHECK(!img_open(bp), "unknown required Catalog extension is refused");

        // Rebuild clean fixture for tamper checks
        n = nx_build_fixture(buf, 2, 0);
        static uint8_t tb[262144];
        // Identical generation slots differing (flip a byte of slot B's manifest_sha, keeping header structurally valid)
        memcpy(tb, buf, n); tb[4096 + 152] ^= 1; w32(tb + 4096 + 252, nx_crc32c(tb + 4096, 252));
        write_file(bp, tb, n); CHECK(!img_open(bp), "identical generation slots with differing bytes are refused");

        // Corrupt catalog SHA-256 in slot A & B
        memcpy(tb, buf, n); tb[120] ^= 1; w32(tb + 252, nx_crc32c(tb, 252));
        memcpy(tb + 4096, tb, 256);
        write_file(bp, tb, n); CHECK(!img_open(bp), "corrupt Catalog SHA-256 is refused");

        // Dual generation fallback: slot B has gen 1 with valid header CRC but corrupt Catalog SHA-256; slot A has gen 0 valid
        memcpy(tb, buf, n);
        w64(tb + 4096 + 216, 1); tb[4096 + 120] ^= 1; w32(tb + 4096 + 252, nx_crc32c(tb + 4096, 252));
        write_file(bp, tb, n);
        { romfs_img *m = img_open(bp); CHECK(m != NULL, "dual-generation fallback to gen 0 when gen 1 metadata is damaged"); romfs_image_close(m); }

        // Cut-off file
        write_file(bp, buf, n - 16); CHECK(!img_open(bp), "cut-off .nx is refused");
    }

    char cmd[1200]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", root); system(cmd);
    printf("== %d checks, %d failed ==\n", checks, fails);
    return fails != 0;
}
