// Raw global memory's page index (dmsh_abi_vk.h DMSH_VK_GMEM_PIDX_*, vk_android_spv.py gmem_pidx): the records after the pool table, built from the table's entries and kept
// up to date by entry differences. No Vulkan here: vka_draw.c (gmem_pidx) keeps one state per buffer, the tests (dm/gfx/tests/vka_pidx_test.c, vk_gmem_run.c) the same code.
//
// The window is the entries' pages rounded out to `gran` pages (a power of two; the host's VKA_PIDX_GRAN, 256 MiB), so a segment placed near the others falls in the
// window the index already has and only its pages are written. An update whose entries all lie in the window the buffer holds rewrites only the pages of the entries that
// came, went or changed (start, end or delta): first those of the entries gone are zeroed, then those of the entries now present are written, so a page an old and a new
// entry share ends with the new record. Otherwise the window is made again from the entries and every record rewritten (a full build), when the buffer has room for it.
#pragma once
#include <stdint.h>
#include <string.h>
#include "../nvn/dmsh_abi_vk.h"

#define VKA_PIDX_GRAN ((uint64_t)1 << 16)   // window rounding in pages: 256 MiB
#define VKA_PIDX_MASK (((uint64_t)1 << DMSH_VK_GMEM_PIDX_SHIFT) - 1)

typedef struct {
    int valid, n, max;                  // valid: the buffer holds this window and these n entries (max: the arrays' room)
    uint64_t base, pages;               // the window: pages [base, base + pages) by guest address >> DMSH_VK_GMEM_PIDX_SHIFT
    uint64_t *lo, *hi, *dl;             // the entries the records hold (start, end, delta = device - guest address), by address
} vka_pidx;

typedef struct { int unaligned, high, over; uint64_t full, inc, written; } vka_pidx_stats;   // entries left out (start, end or delta not on a page; past
// DMSH_VK_GMEM_PIDX_PAGE_END; past max_pages of window), full builds, incremental updates, records written

// The table's entries the index can hold (in place over lo / hi; dl gets the deltas): on pages, below DMSH_VK_GMEM_PIDX_PAGE_END, and within max_pages of the window that
// starts at the first one's page rounded down to gran. → the count kept; st counts the others
static inline int vka_pidx_take(uint64_t *lo, uint64_t *hi, const uint64_t *bda, uint64_t *dl, int k, uint64_t gran, uint64_t max_pages, vka_pidx_stats *st) {
    int n = 0; uint64_t base = 0;
    for (int i = 0; i < k; i++) {
        uint64_t l = lo[i], h = hi[i], d = bda[i] - lo[i];
        if ((l | h | d) & VKA_PIDX_MASK) { st->unaligned++; continue; }
        if ((h >> DMSH_VK_GMEM_PIDX_SHIFT) > DMSH_VK_GMEM_PIDX_PAGE_END) { st->high++; continue; }
        if (!n) base = (l >> DMSH_VK_GMEM_PIDX_SHIFT) & ~(gran - 1);
        if ((h >> DMSH_VK_GMEM_PIDX_SHIFT) - base > max_pages) { st->over++; continue; }
        lo[n] = l; hi[n] = h; dl[n] = d; n++;
    }
    return n;
}

// The window entries [lo, hi) need (rounded out to gran pages, the end at most DMSH_VK_GMEM_PIDX_PAGE_END); none: an empty window at 0
static inline void vka_pidx_window(const uint64_t *lo, const uint64_t *hi, int n, uint64_t gran, uint64_t *base, uint64_t *pages) {
    if (!n) { *base = 0; *pages = 0; return; }
    uint64_t b = (lo[0] >> DMSH_VK_GMEM_PIDX_SHIFT) & ~(gran - 1), e = ((hi[n - 1] >> DMSH_VK_GMEM_PIDX_SHIFT) + gran - 1) & ~(gran - 1);
    if (e > DMSH_VK_GMEM_PIDX_PAGE_END) e = DMSH_VK_GMEM_PIDX_PAGE_END;
    *base = b; *pages = e - b;
}

static inline void vka_pidx_put(uint64_t *rec, uint64_t base, uint64_t lo, uint64_t hi, uint64_t dl, int set, vka_pidx_stats *st) {
    uint64_t a = lo >> DMSH_VK_GMEM_PIDX_SHIFT, z = hi >> DMSH_VK_GMEM_PIDX_SHIFT;
    if (!set) { memset(rec + (a - base), 0, (size_t)(z - a) * 8); return; }
    for (uint64_t pg = a; pg < z; pg++) rec[pg - base] = ((pg << DMSH_VK_GMEM_PIDX_SHIFT) + dl) | 1;   // (dl a multiple of the page: bit 0 free)
    st->written += z - a;
}

static inline int vka_pidx_same(const vka_pidx *s, int i, uint64_t lo, uint64_t hi, uint64_t dl) { return s->lo[i] == lo && s->hi[i] == hi && s->dl[i] == dl; }

// Brings the index at idx (the buffer's byte DMSH_VK_GMEM_PIDX_WORD * 4: header, then records) with room for cap records to the n entries lo / hi / dl (vka_pidx_take's).
// → 1 incremental, 2 full build, 0 no room (the caller needs a buffer of vka_pidx_bytes(pages) after the table; s is then unchanged)
static inline int vka_pidx_update(uint8_t *idx, uint64_t cap, vka_pidx *s, const uint64_t *lo, const uint64_t *hi, const uint64_t *dl, int n, uint64_t gran, vka_pidx_stats *st) {
    uint64_t b, p; vka_pidx_window(lo, hi, n, gran, &b, &p);
    uint32_t *w = (uint32_t *)idx; uint64_t *rec = (uint64_t *)(idx + 16);
    int full = 0;
    if (n > s->max) return 0;
    if (s->valid && (!n || ((lo[0] >> DMSH_VK_GMEM_PIDX_SHIFT) >= s->base && (hi[n - 1] >> DMSH_VK_GMEM_PIDX_SHIFT) <= s->base + s->pages))) {
        for (int i = 0, j = 0; i < s->n; i++) {   // entries gone or changed: zeroed
            while (j < n && lo[j] < s->lo[i]) j++;
            if (!(j < n && vka_pidx_same(s, i, lo[j], hi[j], dl[j]))) vka_pidx_put(rec, s->base, s->lo[i], s->hi[i], 0, 0, st);
        }
        for (int i = 0, j = 0; i < n; i++) {   // entries come or changed: written
            while (j < s->n && s->lo[j] < lo[i]) j++;
            if (!(j < s->n && vka_pidx_same(s, j, lo[i], hi[i], dl[i]))) vka_pidx_put(rec, s->base, lo[i], hi[i], dl[i], 1, st);
        }
        st->inc++;
    } else {
        if (p + 1 > cap) return 0;
        w[0] = (uint32_t)b; w[1] = (uint32_t)p; w[2] = w[3] = 0;
        memset(rec, 0, (size_t)(p + 1) * 8);   // (record p: the sentinel, past the window)
        for (int i = 0; i < n; i++) vka_pidx_put(rec, b, lo[i], hi[i], dl[i], 1, st);
        s->base = b; s->pages = p; st->full++; full = 1;
    }
    memcpy(s->lo, lo, (size_t)n * 8); memcpy(s->hi, hi, (size_t)n * 8); memcpy(s->dl, dl, (size_t)n * 8); s->n = n; s->valid = 1;
    return full ? 2 : 1;
}

// Bytes of the index (header and records) for a window of pages
static inline uint64_t vka_pidx_bytes(uint64_t pages) { return 16 + (pages + 1) * DMSH_VK_GMEM_PIDX_REC; }
