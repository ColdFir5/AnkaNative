// test_vka_pidx.py's run: the host's page index code (dm/gfx/vk_android/vka_pidx.h) against a brute-force model. Random sequences of table entries (segments placed,
// given back, re-placed with another delta, merged neighbours, windows that grow and move, entries off the page or past 2^44) go through vka_pidx_take and
// vka_pidx_update into one buffer kept across steps, as vka_draw.c keeps it; after every step each record of the window is checked against the entries (the page's
// device address | 1 inside an entry, else 0; the sentinel 0), and random addresses are looked up as gmem_pidx's code does (the high word clamped, the page below or
// past the window to the sentinel) and compared with the table's answer (vk_android_spv.py: the entry whose start <= address < end, its device address; else the sink).
// Prints "ok <steps> <full> <incremental> <records written>" or the first difference (exit 1)
#include "../vk_android/vka_pidx.h"
#include <stdio.h>
#include <stdlib.h>

static uint64_t rs = 88172645463325252ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
enum { MAXE = 4095, UNIV = 1 << 17 };   // entries; the pages segments are placed in (512 MiB from G0)
static const uint64_t G0 = 0x4123f00000ull;   // a guest address near the phone's (pidx/GATE.md)

typedef struct { uint64_t lo, hi, bda; } ent;
static ent E[MAXE]; static int NE;
static int cmp(const void *a, const void *b) { const ent *x = a, *y = b; return x->lo < y->lo ? -1 : x->lo > y->lo; }

// the table's answer for address a: 1 and the device address when an entry holds it
static int table_find(const uint64_t *lo, const uint64_t *hi, const uint64_t *dl, int n, uint64_t a, uint64_t *dev) {
    for (int i = 0; i < n; i++) if (lo[i] <= a && a < hi[i]) { *dev = a + dl[i]; return 1; }
    return 0;
}
// gmem_pidx's lookup (vk_android_spv.py _pidx_search's emitted code) on the buffer
static int pidx_find(const uint8_t *idx, uint64_t a, uint64_t *dev) {
    const uint32_t *w = (const uint32_t *)idx; const uint64_t *rec = (const uint64_t *)(idx + 16);
    uint32_t lo = (uint32_t)a, hi = (uint32_t)(a >> 32), h = hi < 0xfff ? hi : 0xfff;
    uint32_t page = (h << 20) | (lo >> 12), pg = page - w[0]; if (pg > w[1]) pg = w[1];
    uint64_t r = rec[pg];
    *dev = ((r & ~(uint64_t)0xfff) | (lo & 0xfff));
    return (int)(r & 1);
}

int main(int argc, char **argv) {
    int steps = argc > 1 ? atoi(argv[1]) : 400; if (argc > 2) rs ^= strtoull(argv[2], NULL, 0) * 0x9e3779b97f4a7c15ull;
    uint64_t gran = argc > 3 ? strtoull(argv[3], NULL, 0) : VKA_PIDX_GRAN;
    static uint64_t lo[MAXE], hi[MAXE], bda[MAXE], dl[MAXE], slo[MAXE + 1], shi[MAXE + 1], sdl[MAXE + 1];
    uint64_t cap = 4 * (UNIV + gran) + 4 * gran;   // records the buffer has room for (windows up to 4 universes: the moves below)
    uint8_t *idx = calloc(1, (size_t)vka_pidx_bytes(cap)); if (!idx) return 2;
    vka_pidx s = {.max = MAXE + 1, .lo = slo, .hi = shi, .dl = sdl}; vka_pidx_stats st = {0};
    uint64_t shift = 0;   // the universe's offset in pages: moved now and then, so the window moves (full builds)
    for (int t = 0; t < steps; t++) {
        int op = (int)(rnd() % 16);
        if (op < 6 && NE < 300) {   // place a segment in a free run
            uint64_t a = rnd() % UNIV, len = 1 + rnd() % (rnd() % 4 ? 16 : 600), alo = G0 + ((a + shift) << 12), ahi = alo + (len << 12);
            int clash = 0; for (int i = 0; i < NE; i++) if (alo < E[i].hi && E[i].lo < ahi) clash = 1;
            if (!clash) E[NE++] = (ent){alo, ahi, 0x400d000000ull + ((rnd() % 0x40000) << 12)};
        } else if (op < 9 && NE) { int i = (int)(rnd() % NE); E[i] = E[--NE]; }                                 // give one back
        else if (op < 11 && NE) { int i = (int)(rnd() % NE); E[i].bda = 0x5000000000ull + ((rnd() % 0x40000) << 12); }   // re-placed elsewhere: another delta
        else if (op == 11 && NE) { int i = (int)(rnd() % NE); E[i].hi -= (E[i].hi - E[i].lo > 4096) ? 4096 : 0; }    // shorter
        else if (op == 12 && rnd() % 8 == 0) {   // everything moves by a few GiB (the window moves: a full build), or back
            uint64_t d = (shift ? 0 : (uint64_t)(1 + rnd() % 3) * (UNIV + gran)) ; for (int i = 0; i < NE; i++) { E[i].lo = E[i].lo - (shift << 12) + (d << 12); E[i].hi = E[i].hi - (shift << 12) + (d << 12); } shift = d;
        }
        qsort(E, NE, sizeof *E, cmp);
        int k = 0;   // the table's entries: touching segments of the same delta merged (vka_mem_chunks), plus odd ones the index must leave out
        for (int i = 0; i < NE; i++) {
            if (k && hi[k - 1] == E[i].lo && bda[k - 1] - lo[k - 1] == E[i].bda - E[i].lo) { hi[k - 1] = E[i].hi; continue; }
            lo[k] = E[i].lo; hi[k] = E[i].hi; bda[k] = E[i].bda; k++;
        }
        int odd = rnd() % 5 == 0;
        if (odd) { lo[k] = 0xfff00000000ull + 0x1000; hi[k] = lo[k] + 0x2000; bda[k] = 0x6000000000ull; k++; }   // past 2^44 - 2^32 (page END): left out
        if (odd && k > 1) { bda[0] += 0x10; }                                                                 // a delta off the page: left out
        uint64_t tlo[MAXE + 1], thi[MAXE + 1], tdl[MAXE + 1]; for (int i = 0; i < k; i++) { tlo[i] = lo[i]; thi[i] = hi[i]; tdl[i] = bda[i] - lo[i]; }   // the table's view
        int un0 = st.unaligned, hi0 = st.high;
        int n = vka_pidx_take(lo, hi, bda, dl, k, gran, cap - 1 - gran, &st);
        if (odd && (st.high != hi0 + 1 || st.unaligned != un0 + (k > 1))) { printf("step %d: the odd entries not left out\n", t); return 1; }
        int r = vka_pidx_update(idx + 0, cap, &s, lo, hi, dl, n, gran, &st);
        if (!r) { printf("step %d: no room (window %llu pages)\n", t, (unsigned long long)s.pages); return 1; }
        const uint32_t *w = (const uint32_t *)idx; const uint64_t *rec = (const uint64_t *)(idx + 16);
        if (w[0] != (uint32_t)s.base || w[1] != (uint32_t)s.pages || w[2] || w[3]) { printf("step %d: header\n", t); return 1; }
        for (uint64_t p = 0; p <= s.pages; p++) {   // every record against the entries the index holds
            uint64_t a = (s.base + p) << 12, want = 0, dev;
            if (p < s.pages && table_find(lo, hi, dl, n, a, &dev)) want = dev | 1;
            if (rec[p] != want) { printf("step %d: record %llu (page %#llx) %#llx, want %#llx\n", t, (unsigned long long)p, (unsigned long long)(s.base + p), (unsigned long long)rec[p], (unsigned long long)want); return 1; }
        }
        for (int q = 0; q < 2000; q++) {   // lookups as the shader makes them, against the table's answer (entries left out of the index: the sink in both, as they are not in tlo)
            uint64_t a;
            switch (q % 8) {
                case 0: a = rnd(); break;
                case 1: a = rnd() & 0xfffffffffffull; break;
                case 2: a = ((uint64_t)0xfff << 32) | (rnd() & 0xffffffffull); break;
                case 3: a = G0 - 4096 + (rnd() % 8192); break;
                default: { int i = n ? (int)(rnd() % n) : 0; a = n ? lo[i] + rnd() % (hi[i] - lo[i] + 8192) - 4096 : rnd(); }
            }
            uint64_t d1 = 0, d2 = 0; int in1 = table_find(lo, hi, dl, n, a, &d1), in2 = pidx_find(idx, a, &d2);
            int in3 = 0; uint64_t d3 = 0; in3 = table_find(tlo, thi, tdl, k, a, &d3);
            if (in1 != in2 || (in1 && d1 != d2)) { printf("step %d: address %#llx: table %d %#llx, index %d %#llx\n", t, (unsigned long long)a, in1, (unsigned long long)d1, in2, (unsigned long long)d2); return 1; }
            if (in3 && !in1 && !odd) { printf("step %d: address %#llx in the table but left out of the index\n", t, (unsigned long long)a); return 1; }
        }
    }
    {   // no room: a window larger than the buffer's records leaves the state as it was
        vka_pidx s2 = {.max = MAXE + 1, .lo = slo, .hi = shi, .dl = sdl}; uint64_t l = G0, h = G0 + ((2 * gran + 8) << 12), d = 0;
        uint8_t small[16 + 8 * 4];
        if (vka_pidx_update(small, 4, &s2, &l, &h, &d, 1, gran, &st) != 0 || s2.valid) { printf("no room not refused\n"); return 1; }
    }
    printf("ok %d %llu %llu %llu\n", steps, (unsigned long long)st.full, (unsigned long long)st.inc, (unsigned long long)st.written);
    return 0;
}
