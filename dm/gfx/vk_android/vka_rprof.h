// The recording thread's CPU split per frame (DM_VK_RPROF=<n>, live: the next submission; 0 = off): vka_exec.c times each submission's begin, every command
// record by kind and the commit; vka_draw.c the draw path's steps (its DM_VK_PROF phases, kept cumulative here) and the set writes' parts; vka_tex.c the texture work at a
// submission's start. One `[rprof]` line for one frame in n, when the next frame's first submission begins (vka_exec.c rprof_line). Only the recording thread touches
// these counters, and only between its submission's begin and commit (vka_rprof_on is set there), so no atomics. Times are the ARM generic timer (cntvct), a few ns a read
#pragma once
#include <stdint.h>
#include <time.h>

enum {   // timed inside vka_draw.c / vka_tex.c (inclusive: each sits inside one of vka_exec.c's command kinds or the submission's begin)
    VKA_RPX_TEXFB,      // vka_tex_prepare: the texture-use feedback readbacks that are done (tfb_consume)
    VKA_RPX_EVICT,      // vka_tex_prepare: images let go (tex_evict)
    VKA_RPX_MIPDROP,    // vka_tex_prepare: mip drop shrinks and coarse-first promotions (md_shrink, cf_promote)
    VKA_RPX_IMAGES,     // a draw's images first used: made, cleared or uploaded (make_images)
    VKA_RPX_POOLSCAN,   // a draw's or dispatch's texture pool scan and the lazy images it writes (pool_scan, vka_tex_pool_write)
    VKA_RPX_PIDX,       // set writes: the raw global memory page index (gmem_pidx: buffer pick, waits, updates, the table)
    VKA_RPX_GMAP,       // set writes: the raw global memory table alone (gmem_map)
    VKA_RPX_N
};
extern int vka_rprof_on;
extern uint64_t vka_rprof_x[VKA_RPX_N];
static inline uint64_t vka_rprof_t(void) {
#if defined(__aarch64__)
    uint64_t v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v)); return v;
#else
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
#endif
}
static inline double vka_rprof_ms(uint64_t ticks) {
#if defined(__aarch64__)
    static uint64_t f; if (!f) { __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f)); if (!f) f = 1; }
    return (double)ticks * 1e3 / (double)f;
#else
    return ticks / 1e6;
#endif
}
#define VKA_RP_T0() (vka_rprof_on ? vka_rprof_t() : 0)
#define VKA_RP_ADD(i, t0) do { if (vka_rprof_on) vka_rprof_x[i] += vka_rprof_t() - (t0); } while (0)
// vka_draw.c: the cumulative draw path phases (the DM_VK_PROF names) and set parts, in ticks, for vka_exec.c's per-frame differences
enum { VKA_RP_PT = 8, VKA_RP_PS = 5 };   // phases: resolve, layout, images+pass+key, vertex input, pipeline, sets, vertex buffers+dynamic, draw; sets: walk, writes, push, driver segments, images in writes
void vka_draw_rprof(uint64_t pt[VKA_RP_PT], uint64_t ps[VKA_RP_PS]);
