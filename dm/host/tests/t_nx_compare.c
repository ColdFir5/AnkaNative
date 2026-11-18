// Real-file comparison of the host's NX and NUP readers through hle_fs.c:
//   build/t_nx_compare <a.nx:sel/romfs> <b.nup:sel/romfs>
// Mounts the NX snapshot as "nx:" and the NUP package as "nup:" through hle_fs.c, checks that every directory and file exists in both
// with identical size and bytes (sequential reads with varying sizes crossing block boundaries, plus seeded random offset/length reads
// crossing block boundaries), and prints counts, GiB, elapsed time and throughput.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../hle_fs.c"

struct dm_flags g_flag; uint64_t g_t0_ns; uint64_t now_ns(void) { return 0; }
void dm_guest_write_prepare(void *a, size_t n) { (void)a; (void)n; }

static uint64_t nfiles, n_only_nx, n_only_nup, nbytes, nrand, nbad;
static uint8_t g_a[1572864], g_b[1572864];
static uint32_t g_rng = 0x9E3779B9u;

static uint64_t ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static uint32_t next_u32(void) { g_rng = g_rng * 1664525u + 1013904223u; return g_rng; }

static void compare_file(const char *rel) {
    char pnx[2048], pnup[2048];
    snprintf(pnx, sizeof pnx, "nx:/%s", rel);
    snprintf(pnup, sizeof pnup, "nup:/%s", rel);
    uint64_t h1 = 0, h2 = 0; int64_t s1 = -1, s2 = -2;
    if (h_fs_OpenFile(&h2, pnup, MODE_READ) != RES_OK) { n_only_nx++; return; }
    if (h_fs_GetFileSize(&s2, h2) != RES_OK) { h_fs_CloseFile(h2); printf("  ✗ stat failed in .nup: %s\n", pnup); nbad++; return; }
    if (h_fs_OpenFile(&h1, pnx, MODE_READ) != RES_OK || h_fs_GetFileSize(&s1, h1) != RES_OK) {
        h_fs_CloseFile(h2); printf("  ✗ cannot open %s\n", pnx); nbad++; return;
    }
    if (s1 != s2) {
        printf("  ✗ size differs %s: nx=%lld nup=%lld\n", rel, (long long)s1, (long long)s2);
        h_fs_CloseFile(h1); h_fs_CloseFile(h2); nbad++; return;
    }
    static const size_t chunks[] = {262144 + 777, 1048576 + 123, 65536 + 19, 524288 + 4093};
    int64_t off = 0; int ok = 1, ci = 0;
    while (off < s1) {
        size_t want = chunks[ci++ & 3];
        if ((int64_t)want > s1 - off) want = (size_t)(s1 - off);
        size_t g1 = 0, g2 = 0;
        if (h_fs_ReadFile(&g1, h1, off, g_a, want) != RES_OK || h_fs_ReadFile(&g2, h2, off, g_b, want) != RES_OK ||
            g1 != want || g2 != want || memcmp(g_a, g_b, want)) {
            printf("  ✗ seq mismatch %s at %lld (%zu vs %zu)\n", rel, (long long)off, g1, g2);
            ok = 0; break;
        }
        off += (int64_t)want;
    }
    // Random offset/length reads crossing block boundaries (especially for files > 64 KiB)
    if (ok && s1 > 0) {
        int rounds = s1 > 65536 ? 6 : 2;
        for (int r = 0; r < rounds; r++) {
            uint64_t roff = 0;
            if (s1 > 65536 && r < 3) {
                uint64_t bnd = ((next_u32() % (uint64_t)((s1 - 1) / 65536 + 1)) ?: 1) * 65536ull;
                uint64_t back = 1 + (next_u32() % 4096);
                roff = bnd > back ? bnd - back : 0;
            } else roff = next_u32() % (uint64_t)s1;
            size_t rlen = 1 + (next_u32() % (s1 > 65536 ? 131072u : 4096u));
            if ((int64_t)rlen > s1 - (int64_t)roff) rlen = (size_t)(s1 - (int64_t)roff);
            size_t g1 = 0, g2 = 0;
            if (h_fs_ReadFile(&g1, h1, (int64_t)roff, g_a, rlen) != RES_OK || h_fs_ReadFile(&g2, h2, (int64_t)roff, g_b, rlen) != RES_OK ||
                g1 != rlen || g2 != rlen || memcmp(g_a, g_b, rlen)) {
                printf("  ✗ rand mismatch %s at %llu+%zu\n", rel, (unsigned long long)roff, rlen);
                ok = 0; break;
            }
            nrand++;
            nbytes += rlen;
        }
    }
    h_fs_CloseFile(h1); h_fs_CloseFile(h2);
    if (!ok) nbad++;
    nfiles++; nbytes += (uint64_t)s1;
}

static void walk_nx(const char *rel) {
    char pnx[2048];
    snprintf(pnx, sizeof pnx, "nx:/%s", rel);
    uint64_t dh = 0;
    if (h_fs_OpenDirectory(&dh, pnx, DIRMODE_DIR | DIRMODE_FILE) != RES_OK) { printf("  ✗ cannot list %s\n", pnx); nbad++; return; }
    DirectoryEntry de[64]; int64_t cnt = 0;
    while (h_fs_ReadDirectory(&cnt, de, dh, 64) == RES_OK && cnt > 0) {
        for (int64_t i = 0; i < cnt; i++) {
            char sub[2048];
            if (rel[0]) snprintf(sub, sizeof sub, "%s/%s", rel, de[i].name);
            else snprintf(sub, sizeof sub, "%s", de[i].name);
            if (de[i].type == ENT_DIR) walk_nx(sub);
            else compare_file(sub);
        }
    }
    h_fs_CloseDirectory(dh);
}

static void check_nup_extras(const char *rel) {
    char pnup[2048], pnx[2048];
    snprintf(pnup, sizeof pnup, "nup:/%s", rel);
    uint64_t dh = 0;
    if (h_fs_OpenDirectory(&dh, pnup, DIRMODE_DIR | DIRMODE_FILE) != RES_OK) return;
    DirectoryEntry de[64]; int64_t cnt = 0;
    while (h_fs_ReadDirectory(&cnt, de, dh, 64) == RES_OK && cnt > 0) {
        for (int64_t i = 0; i < cnt; i++) {
            char sub[2048];
            if (rel[0]) snprintf(sub, sizeof sub, "%s/%s", rel, de[i].name);
            else snprintf(sub, sizeof sub, "%s", de[i].name);
            snprintf(pnx, sizeof pnx, "nx:/%s", sub);
            int32_t t = -1;
            if (h_fs_GetEntryType(&t, pnx) != RES_OK) { if (de[i].type == ENT_FILE) n_only_nup++; }
            else if (de[i].type == ENT_DIR) check_nup_extras(sub);
        }
    }
    h_fs_CloseDirectory(dh);
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <a.nx:sel/romfs> <b.nup:sel/romfs>\n", argv[0]); return 2; }
    uint64_t t0 = ns();
    setenv("DM_ROMFS", argv[1], 1);
    if (h_fs_MountRom("nx", NULL, 0) != RES_OK || !g_mnt[0].img) { printf("mount nx failed\n"); return 1; }
    setenv("DM_ROMFS", argv[2], 1);
    if (h_fs_MountRom("nup", NULL, 0) != RES_OK || !g_mnt[1].img) { printf("mount nup failed\n"); return 1; }
    uint64_t t1 = ns();
    printf("mount: %.0f ms, nx %llu entries, nup %llu entries\n", (t1 - t0) / 1e6,
           (unsigned long long)g_mnt[0].img->n, (unsigned long long)g_mnt[1].img->n);
    walk_nx("");
    check_nup_extras("");
    double sec = (ns() - t1) / 1e9;
    double gib = nbytes / 1073741824.0;
    printf("%llu files compared (%llu only in nx, %llu only in nup), %llu random reads, %.2f GiB read in %.2f s (%.0f MiB/s)\n",
           (unsigned long long)nfiles, (unsigned long long)n_only_nx, (unsigned long long)n_only_nup, (unsigned long long)nrand, gib, sec, sec > 0 ? (nbytes / 1048576.0) / sec : 0);
    printf(nbad ? "== %llu problems ==\n" : "== every file in both matches ==\n", (unsigned long long)nbad);
    return nbad != 0;
}
