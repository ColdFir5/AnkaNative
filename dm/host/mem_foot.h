// Guest memory footprint by region (DM_MEM_FOOT, hle_svc.c "Guest memory footprint"): the pure parts, free of the host and of Vulkan so the unit test
// (dm/host/tests/t_android_memfoot.c) can check them. A memory pool's record as the replay fills it (vka_mem.c dm_gfx_pool_foot, vka_tex.c vka_tex_pool_use) and the
// scan counts it (pagemap), the interval arithmetic that keeps the heap and alias remainders apart from the pools, what the host still reads of a pool, and the last
// line's values for the change since then
#ifndef DM_MEM_FOOT_H
#define DM_MEM_FOOT_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// One memory pool. lo, size: its guest range; id: the replay's pool serial (unique; two pools may share a start); gen: its NVN Initialize serial (0: not in NVN's live
// table); flags: its NVN flags (fl_known 0: not found); placed: the bytes of it that are placed Vulkan memory (all of it, the parts placed so far, or none).
// The images over its storage (the live table's, by storage address; sparse virtual pools' images read through their page table and are left out), by content:
// fill = uploaded once (ST_FILL, or ST_TRACK without tracking), track = re-uploaded when the CPU writes its pages, gpu = GPU content (render targets, copy and storage
// destinations; ST_ZERO or written by the GPU since), drop = finer levels left out (mip drop, coarse-first: made whole again from the storage). n_* count them, s_* sum
// their storage bytes (overlapping storage counted in each), img their image memory (a shared allocation counted in each image over it).
// The scan: res = resident anonymous, shr = resident shared (placed Vulkan memory, the driver's), swp = swapped out (zram), all = the range's bytes
// The tables a footprint line was read from (vka_mem.c dm_gfx_pool_foot): live pools in the replay's table, and NVN's live pools with storage, of which nvn_listed were
// read (the rest: no flags for them, "flags ?")
typedef struct { int pools, nvn_live, nvn_listed; } dmf_counts;
typedef struct {
    uint64_t lo, size, id, gen, placed;
    uint32_t flags, fl_known, n_fill, n_track, n_gpu, n_drop;
    uint64_t s_fill, s_track, s_gpu, s_drop, img;
    uint64_t res, shr, swp, all;
} dm_pool_foot;
static inline uint64_t dmf_touched(const dm_pool_foot *p) { return p->res + p->shr + p->swp; }

// What the host still reads of a pool's bytes (the strongest reason first):
// PLACED: placed whole: the GPU reads it there (buffers, texture addresses), not a copy
// TRACK: tracked images: the CPU's writes are uploaded again
// DROP: images with finer levels left out: made whole again from the storage (a GPU write, a copy, promotion when the pressure ends)
// FILL: images uploaded once: read again only when the image is made again (eviction, feedback release) or a texture is registered over it anew
// GPU: only GPU-content images: the host does not read the bytes (the GPU's content lives in the images; a write-back into the storage writes them)
// SHADER: an NVN SHADER_CODE pool (flags 0x40) not placed whole, no image: the host reads its code when a program is set (the shader key's hash), the GPU never
// (DM_VK_PLACE_SHADER_POOLS=0 leaves such pools not placed; hle_nvn.c "Shader-code pools")
// NONE: no image and not placed whole: read only when a texture or buffer is made over it (or by a part placed: PARTS)
enum { DMF_PLACED, DMF_TRACK, DMF_DROP, DMF_FILL, DMF_GPU, DMF_PARTS, DMF_NONE, DMF_SHADER, DMF_NUSE };
static inline int dmf_use(const dm_pool_foot *p) {
    if (p->size && p->placed >= p->size) return DMF_PLACED;
    if (p->n_track) return DMF_TRACK;
    if (p->n_drop) return DMF_DROP;
    if (p->n_fill) return DMF_FILL;
    if (p->n_gpu) return DMF_GPU;
    if (p->fl_known && (p->flags & 0x40)) return DMF_SHADER;
    return p->placed ? DMF_PARTS : DMF_NONE;
}
static inline const char *dmf_use_name(int u) {
    static const char *n[DMF_NUSE] = {"placed: GPU reads it", "texture storage, tracked: re-read on CPU writes", "texture storage, levels left out: re-read to make them whole",
                                      "texture storage, uploaded once: re-read only to remake an image", "render-target storage: not read by the host",
                                      "parts placed, no image", "no image: not read by the host now", "shader code: read by the host when a program is set, not by the GPU"};
    return u >= 0 && u < DMF_NUSE ? n[u] : "?";
}
// NVN pool flags, short: cpu-{none,unc,cached} gpu-{none,unc,cached} and shader / compr / phys / virt
static inline void dmf_flags_str(uint32_t f, int known, char *b, size_t n) {
    if (!known) { snprintf(b, n, "flags ?"); return; }
    const char *c = f & 1 ? "none" : f & 2 ? "unc" : f & 4 ? "cached" : "-", *g = f & 8 ? "none" : f & 0x10 ? "unc" : f & 0x20 ? "cached" : "-";
    snprintf(b, n, "%#05x cpu-%s gpu-%s%s%s%s%s", f, c, g, f & 0x40 ? " shader" : "", f & 0x80 ? " compr" : "", f & 0x100 ? " phys" : "", f & 0x200 ? " virt" : "");
}

// Intervals [lo, hi): sorted by lo and merged (touching ones too); the count. In place
typedef struct { uint64_t lo, hi; } dmf_iv;
static inline int dmf_iv_cmp(const void *a, const void *b) { uint64_t x = ((const dmf_iv *)a)->lo, y = ((const dmf_iv *)b)->lo; return x < y ? -1 : x > y; }
static inline int dmf_merge(dmf_iv *v, int n) {
    int k = 0;
    qsort(v, (size_t)n, sizeof *v, dmf_iv_cmp);
    for (int i = 0; i < n; i++) {
        if (v[i].hi <= v[i].lo) continue;
        if (k && v[i].lo <= v[k - 1].hi) { if (v[i].hi > v[k - 1].hi) v[k - 1].hi = v[i].hi; }
        else v[k++] = v[i];
    }
    return k;
}
// The parts of [lo, hi) that no interval of v (merged) covers, each passed to f in address order; how many
static inline int dmf_gaps(uint64_t lo, uint64_t hi, const dmf_iv *v, int n, void (*f)(uint64_t a, uint64_t e, void *ctx), void *ctx) {
    int c = 0, i = 0;
    while (i < n && v[i].hi <= lo) i++;
    for (uint64_t a = lo; a < hi; i++) {
        uint64_t e = i < n && v[i].lo < hi ? v[i].lo : hi;
        if (e > a) { f(a, e, ctx); c++; }
        if (i >= n || v[i].lo >= hi) break;
        if (v[i].hi > a) a = v[i].hi;
    }
    return c;
}

// One pagemap pass for ranges that overlap (the touched line's heap, alias runs and pools not placed; the footprint's pools and regions: the same pages several times
// before). Each query is a range [lo, hi) and the counter it adds to (acc). The sweep cuts the queries' union at every query boundary and hands each piece [x, y) that
// some query covers to piece() once, with the queries over it (act, na): the caller reads the piece's pages once and adds the result to each of them. ev holds 2n
// events, act n indexes (the caller's). Returns the pieces handed. Queries with hi <= lo are left out
typedef struct { uint64_t lo, hi; int acc; } dmf_q;
typedef struct { uint64_t at; int q, start; } dmf_ev;
static inline int dmf_ev_cmp(const void *a, const void *b) {   // by address; at one address the ends before the starts (touching ranges do not overlap)
    const dmf_ev *x = a, *y = b;
    if (x->at != y->at) return x->at < y->at ? -1 : 1;
    return x->start - y->start;
}
static inline int dmf_sweep(const dmf_q *q, int n, dmf_ev *ev, int *act, void (*piece)(uint64_t x, uint64_t y, const int *act, int na, void *ctx), void *ctx) {
    int ne = 0, na = 0, np = 0;
    for (int i = 0; i < n; i++) if (q[i].hi > q[i].lo) { ev[ne++] = (dmf_ev){q[i].lo, i, 1}; ev[ne++] = (dmf_ev){q[i].hi, i, 0}; }
    qsort(ev, (size_t)ne, sizeof *ev, dmf_ev_cmp);
    for (int i = 0; i < ne; ) {
        uint64_t x = ev[i].at;
        for (; i < ne && ev[i].at == x; i++) {
            if (ev[i].start) act[na++] = ev[i].q;
            else for (int k = 0; k < na; k++) if (act[k] == ev[i].q) { act[k] = act[--na]; break; }
        }
        if (i < ne && na) { piece(x, ev[i].at, act, na, ctx); np++; }
    }
    return np;
}

// The last line's values of one pool (by id), for the change since then: kept sorted by id
typedef struct { uint64_t id, tch, res, swp; } dmf_prev;
static inline int dmf_prev_cmp(const void *a, const void *b) { uint64_t x = ((const dmf_prev *)a)->id, y = ((const dmf_prev *)b)->id; return x < y ? -1 : x > y; }
static inline const dmf_prev *dmf_prev_find(const dmf_prev *v, int n, uint64_t id) {
    int lo = 0, hi = n;
    while (lo < hi) { int m = (lo + hi) / 2; if (v[m].id < id) lo = m + 1; else hi = m; }
    return lo < n && v[lo].id == id ? &v[lo] : NULL;
}
// The k largest pools by key (0 swapped, 1 touched) into ix (indexes into p), largest first; how many (pools with a key of 0 left out)
static inline uint64_t dmf_key(const dm_pool_foot *p, int by) { return by ? dmf_touched(p) : p->swp; }
static inline int dmf_top(const dm_pool_foot *p, int n, int by, int *ix, int k) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        uint64_t x = dmf_key(&p[i], by); if (!x) continue;
        int j = m < k ? m++ : k; if (j == k) { if (x <= dmf_key(&p[ix[k - 1]], by)) continue; j = k - 1; }
        while (j > 0 && dmf_key(&p[ix[j - 1]], by) < x) { ix[j] = ix[j - 1]; j--; }
        ix[j] = i;
    }
    return m;
}
#endif
