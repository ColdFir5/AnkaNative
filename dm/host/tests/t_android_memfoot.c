// The guest memory footprint line's pure parts (dm/host/mem_foot.h; hle_svc.c "Guest memory footprint", DM_MEM_FOOT): the pools' union and the remainders around it,
// what the host still reads of a pool, the pools by swapped / touched, the last line's values by pool id
#include "../mem_foot.h"
#include <stdio.h>

static int fails;
#define CHECK(c, what) do { if (!(c)) { printf("FAIL %s:%d %s\n", __FILE__, __LINE__, what); fails++; } } while (0)
typedef struct { int n; dmf_iv g[16]; } gaps;
static void got(uint64_t a, uint64_t e, void *ctx) { gaps *g = ctx; if (g->n < 16) g->g[g->n] = (dmf_iv){a, e}; g->n++; }
typedef struct { uint64_t per[8], read; int pieces_bad; } sweep_t;
static void piece(uint64_t x, uint64_t y, const int *act, int na, void *ctx) {   // read once: y - x; each query over it gets it
    sweep_t *s = ctx; s->read += y - x; if (y <= x || !na) s->pieces_bad++;
    for (int k = 0; k < na; k++) s->per[act[k]] += y - x;
}

int main(void) {
    // merge: unsorted, overlapping, touching, empty
    dmf_iv v[] = {{50, 60}, {10, 20}, {15, 30}, {30, 35}, {70, 70}, {80, 90}};
    int n = dmf_merge(v, 6);
    CHECK(n == 3 && v[0].lo == 10 && v[0].hi == 35 && v[1].lo == 50 && v[1].hi == 60 && v[2].lo == 80 && v[2].hi == 90, "merge sorts, joins overlapping and touching, drops empty");
    // gaps of [0, 100) around them
    gaps g = {0}; int c = dmf_gaps(0, 100, v, n, got, &g);
    CHECK(c == 4 && g.g[0].lo == 0 && g.g[0].hi == 10 && g.g[1].lo == 35 && g.g[1].hi == 50 && g.g[2].lo == 60 && g.g[2].hi == 80 && g.g[3].lo == 90 && g.g[3].hi == 100, "gaps of the whole range");
    g.n = 0; c = dmf_gaps(15, 55, v, n, got, &g);
    CHECK(c == 1 && g.g[0].lo == 35 && g.g[0].hi == 50, "a range starting and ending inside intervals");
    g.n = 0; c = dmf_gaps(52, 58, v, n, got, &g);
    CHECK(c == 0, "a range inside one interval: no gap");
    g.n = 0; c = dmf_gaps(0, 100, v, 0, got, &g);
    CHECK(c == 1 && g.g[0].lo == 0 && g.g[0].hi == 100, "no intervals: the range");
    g.n = 0; c = dmf_gaps(91, 95, v, n, got, &g);
    CHECK(c == 1 && g.g[0].lo == 91 && g.g[0].hi == 95, "past every interval");
    // what the host still reads: the strongest reason first
    dm_pool_foot p = {.size = 100};
    CHECK(dmf_use(&p) == DMF_NONE, "no image, nothing placed");
    p.placed = 10; CHECK(dmf_use(&p) == DMF_PARTS, "parts placed, no image");
    p.n_gpu = 2; CHECK(dmf_use(&p) == DMF_GPU, "render targets only");
    p.n_fill = 1; CHECK(dmf_use(&p) == DMF_FILL, "an image uploaded once wins over render targets");
    p.n_drop = 1; CHECK(dmf_use(&p) == DMF_DROP, "levels left out win over uploaded once");
    p.n_track = 1; CHECK(dmf_use(&p) == DMF_TRACK, "tracked wins");
    p.placed = 100; CHECK(dmf_use(&p) == DMF_PLACED, "placed whole wins");
    dm_pool_foot sp = {.size = 100, .flags = 0x064, .fl_known = 1};
    CHECK(dmf_use(&sp) == DMF_SHADER, "a shader-code pool not placed, no image: shader code");
    sp.placed = 16; CHECK(dmf_use(&sp) == DMF_SHADER, "a shader-code pool with parts placed: still shader code");
    sp.n_fill = 1; CHECK(dmf_use(&sp) == DMF_FILL, "an image over a shader-code pool wins");
    sp.n_fill = 0; sp.placed = 100; CHECK(dmf_use(&sp) == DMF_PLACED, "a shader-code pool placed whole: placed");
    sp.placed = 0; sp.fl_known = 0; CHECK(dmf_use(&sp) == DMF_NONE, "flags not known: not shader code");
    for (int u = 0; u < DMF_NUSE; u++) CHECK(dmf_use_name(u)[0] != '?', "every use has a name");
    char b[64]; dmf_flags_str(0x124, 1, b, sizeof b); CHECK(!strcmp(b, "0x124 cpu-cached gpu-cached phys"), "flags named");
    dmf_flags_str(0, 0, b, sizeof b); CHECK(!strcmp(b, "flags ?"), "unknown flags");
    // top pools by swapped and by touched
    dm_pool_foot P[6] = {{.swp = 5, .res = 1}, {.swp = 0, .res = 50}, {.swp = 9}, {.swp = 7, .shr = 1}, {.swp = 1}, {.swp = 9, .res = 2}};
    int ix[3]; int k = dmf_top(P, 6, 0, ix, 3);
    CHECK(k == 3 && P[ix[0]].swp == 9 && P[ix[1]].swp == 9 && P[ix[2]].swp == 7, "the 3 most swapped, largest first");
    k = dmf_top(P, 6, 1, ix, 3);
    CHECK(k == 3 && ix[0] == 1 && ix[1] == 5 && ix[2] == 2, "the 3 most touched (resident + shared + swapped)");
    dm_pool_foot Z[2] = {{0}, {0}}; k = dmf_top(Z, 2, 0, ix, 3); CHECK(k == 0, "pools with nothing swapped are left out");
    // the last line's values by id
    dmf_prev pv[] = {{9, 1, 0, 0}, {3, 2, 0, 0}, {5, 3, 0, 0}};
    qsort(pv, 3, sizeof *pv, dmf_prev_cmp);
    CHECK(dmf_prev_find(pv, 3, 5) && dmf_prev_find(pv, 3, 5)->tch == 3, "found by id");
    CHECK(!dmf_prev_find(pv, 3, 4) && !dmf_prev_find(pv, 3, 10) && !dmf_prev_find(pv, 0, 5), "a pool new since the last line has none");
    // one pass over overlapping ranges: each piece once, each range gets exactly its own length
    {   dmf_q q[] = {{0, 100, 0}, {20, 40, 1}, {30, 60, 2}, {100, 120, 3}, {200, 200, 4}, {50, 60, 5}, {110, 130, 6}};
        dmf_ev ev[14]; int act[7]; sweep_t sw = {{0}};
        int np = dmf_sweep(q, 7, ev, act, piece, &sw);
        CHECK(sw.read == 130, "the union read once (0-130)");
        CHECK(sw.per[0] == 100 && sw.per[1] == 20 && sw.per[2] == 30 && sw.per[3] == 20 && sw.per[4] == 0 && sw.per[5] == 10 && sw.per[6] == 20, "every range gets its own length");
        CHECK(!sw.pieces_bad && np == 9, "9 pieces: 0-20 20-30 30-40 40-50 50-60 60-100 100-110 110-120 120-130 (ranges touching at 100 do not overlap)");
        sweep_t z = {{0}}; CHECK(dmf_sweep(q, 0, ev, act, piece, &z) == 0 && z.read == 0, "no ranges: no pieces"); }
    printf("t_android_memfoot: %s\n", fails ? "FAILED" : "ok");
    return fails != 0;
}
