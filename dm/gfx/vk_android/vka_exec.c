// The command replay: hle_nvn.c records the game's NVN commands into the game's own command memory (nvn_obj.h records); a submission replays its handles here into one
// Vulkan command buffer (mtl_sub_begin → mtl_exec per handle → mtl_sub_commit), as nvn_mtl.m does into Metal. This milestone covers what a first frame needs: render targets,
// colour / depth-stencil / texture clears, buffer and texture copies, inline updates and fills, barriers, events, counters and fences. Draws and dispatches log once and are skipped
// until the shader side lands. Every transfer op is followed by a full memory barrier: correct, not fast
#if defined(__ANDROID__)
#include "vka.h"
#include "../nvn/nvn_mtl.h"
#include "../nvn/nvn_timestamp.h"
#include "vka_present_rot.h"
#include "vka_rprof.h"
#include <math.h>
#include <sys/resource.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint64_t serial; p_set_rt rt; int has_rt; p_scissor sc; int has_sc; int64_t copy_stride, copy_img_stride; int rc_skip; vka_dstate d; } qstate;   // what a queue keeps between submissions (d: the draw state, vka_draw.c)
static qstate S;
enum { NQS = 8 };
static qstate g_qs[NQS]; static int g_nqs;
enum { NSLOT = 8 };
enum { NCHUNK = 32 };
static struct { VkCommandBuffer cb; uint64_t fence; VkCommandBuffer x[NCHUNK]; } g_slot[NSLOT];   // x: the command buffers a submission goes on in once it hands its first part to the queue (chunk)
static unsigned g_slot_i;
static int g_chunk_draws = -1, g_nx; static uint64_t g_chunk_d0;   // chunk(): draws a part, parts this submission, draws at its part's start
static VkCommandPool g_pool;
static VkCommandBuffer g_cb;   // the submission being recorded (NULL without a device)
VkCommandBuffer vka_rp_cb(void) { return g_cb; }
typedef struct { int kind; uint32_t mode, v, type; uint64_t addr; } pop;   // kind 1 event, 2 counter report
static pop *g_post; static int g_npost, g_postcap;
static uint64_t g_frame, g_stat_ops, g_stat_draws, g_stat_skipped;
static int g_origin_ll = 1, g_queue_origin_ll = 1, g_depth_zero;
_Atomic int vka_synclog_n;   // DM_VK_SYNCLOG: lines left (mtl_wait)
#define g_synclog vka_synclog_n
static double ms_now(void) { return (now_ns() % 1000000000000ull) / 1e6; }
// Counter reports recorded but not written yet (profile render_cond). A report's counter is written when its submission completes (post_run), but a render condition is
// decided when it is recorded (OP_RENDER_COND), often before the report it reads is written: on the GPU the report comes first, so the condition must see the value the
// report will hold, not what memory holds now. Each pending report's address keeps the counter post_run will write (0 for a timestamp under render_cond, 1 for samples
// passed) and how many writes to it are still pending; rc_known reads the table before memory, and post_run takes an entry's count down after writing the memory. Read from
// memory, a condition over a report from the same or a recent submission read 0 (never written, or cleared) and skipped its draws: Ocarina of Time's first cutscene, whose
// occlusion conditions compare a samples-passed report with 0, skipped ground, tree and background draws for its first ~10 s (white polygonal holes, changing each frame)
#define RC_TAB 16384   // a power of two; linear probing; full: the report is not tracked (the condition reads memory, as before)
static struct { uint64_t addr, val; uint32_t pend; } g_rc_tab[RC_TAB];
static pthread_mutex_t g_rc_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t rc_counter(uint64_t t) { return t == NVN_COUNTER_TYPE_TIMESTAMP || t == NVN_COUNTER_TYPE_TIMESTAMP_TOP ? 0 : t == NVN_COUNTER_TYPE_SAMPLES_PASSED ? 1 : 0; }   // under render_cond (post_run)
static unsigned rc_slot(uint64_t a) { return (unsigned)((a >> 4) * 0x9E3779B97F4A7C15ull >> 40) & (RC_TAB - 1); }
static int g_rc_n, g_rc_full;   // live entries; reports not tracked because the table was full (under g_rc_mu)
static int rc_find(uint64_t a) {   // its slot, or -1 - the empty slot the probe ended at (-1 - RC_TAB: full); under g_rc_mu. An empty slot has pend 0
    unsigned i = rc_slot(a);
    for (int k = 0; k < RC_TAB; k++, i = (i + 1) & (RC_TAB - 1)) { if (!g_rc_tab[i].pend) return -1 - (int)i; if (g_rc_tab[i].addr == a) return (int)i; }
    return -1 - RC_TAB;
}
static void rc_pend_add(uint64_t a, uint64_t val) {   // the recording thread: a report at a was recorded
    pthread_mutex_lock(&g_rc_mu);
    int s = rc_find(a);
    if (s >= 0) { g_rc_tab[s].val = val; g_rc_tab[s].pend++; }
    else if (s != -1 - RC_TAB && g_rc_n < RC_TAB * 3 / 4) { s = -1 - s; g_rc_tab[s].addr = a; g_rc_tab[s].val = val; g_rc_tab[s].pend = 1; g_rc_n++; }
    else if (!g_rc_full++) VRED("more than %d counter reports pending: render conditions over the rest read memory, which may not hold them yet", RC_TAB * 3 / 4);
    pthread_mutex_unlock(&g_rc_mu);
}
static void rc_pend_done(uint64_t a) {   // the completion thread: the report at a was written to memory
    pthread_mutex_lock(&g_rc_mu);
    int s = rc_find(a);
    if (s >= 0 && !--g_rc_tab[s].pend) {   // empty now: move later entries of the probe run back over it (linear probing's deletion), so no probe stops early
        unsigned i = (unsigned)s, j = i; g_rc_n--;
        for (;;) {
            j = (j + 1) & (RC_TAB - 1); if (!g_rc_tab[j].pend) break;
            unsigned h = rc_slot(g_rc_tab[j].addr);
            if (((j - h) & (RC_TAB - 1)) >= ((j - i) & (RC_TAB - 1))) { g_rc_tab[i] = g_rc_tab[j]; g_rc_tab[j].pend = 0; i = j; }   // j's home is at or before i: it may move to i
        }
    }
    pthread_mutex_unlock(&g_rc_mu);
}
static int rc_pend_get(uint64_t a, uint64_t *v) {
    pthread_mutex_lock(&g_rc_mu);
    int s = rc_find(a); if (s >= 0) *v = g_rc_tab[s].val;
    pthread_mutex_unlock(&g_rc_mu); return s >= 0;
}

int vka_origin_ll(void) { return __atomic_load_n(&g_origin_ll, __ATOMIC_RELAXED); }
int vka_queue_origin_ll(void) { return g_queue_origin_ll; }
int vka_depth_zero(void) { return g_depth_zero; }
void mtl_set_origin_lower_left(int on) { if (__atomic_exchange_n(&g_origin_ll, on != 0, __ATOMIC_RELAXED) != (on != 0)) VLOG("window origin: %s", on ? "lower left (the present flips the picture)" : "upper left"); }
void vka_rp_set_queue_origin(int on) { g_queue_origin_ll = on != 0; }
void mtl_set_depth_zero(int on) { g_depth_zero = on != 0; }
// Replay statistics every 300 frames (DM_VK_STATS=0: none): what a frame costs the GPU in passes and barriers (each pass loads and stores its attachments on a tiler), and the
// CPU per draw (recording: mtl_sub_begin to vkEndCommandBuffer)
extern uint64_t vka_n_rp, vka_n_bar; static uint64_t g_n_ub, g_n_sub, g_rec_ns, g_n_pclear, g_n_chunks;
// g_slotw_ns: time vka_rp_sub_begin waited for the fence of the submission that last used its command buffer slot (NSLOT in flight: the GPU is that far behind). It is
// back-pressure, not recording: the recording time (g_rec_ns, mtl_sub_begin to vkEndCommandBuffer) starts after it, so a slower GPU no longer reads as more CPU per draw
static _Atomic uint64_t g_slotw_ns;
uint64_t vka_exec_slot_wait_ns(void) { return atomic_load_explicit(&g_slotw_ns, memory_order_relaxed); }
static _Atomic uint64_t g_fstart[64], g_fstart_f[64];   // when the replay began recording each frame's first submission (its number mod 64): the adaptive floor's spans
static uint64_t g_job_frame; static int g_job_frame_set;   // the frame of the submission the worker replays (vka_exec_sub_frame)
void vka_exec_sub_frame(uint64_t f) { g_job_frame = f; g_job_frame_set = 1; }
uint64_t vka_frame_start_ns(uint64_t f) { unsigned k = f % 64; return atomic_load(&g_fstart_f[k]) == f ? atomic_load(&g_fstart[k]) : 0; }   // now_ns; 0 = not known (vka_perf.c)
void mtl_set_frame(uint64_t f) {
    g_frame = f; vka_ctl_frame(f);
    { if (atomic_load_explicit(&vka_synclog_n, memory_order_relaxed) > 0) VLOG("sync frame %llu begins at %.2f ms", (unsigned long long)f, (now_ns() % 1000000000000ull) / 1e6); }
    static uint64_t f0, d0, rp0, b0, ub0, s0, r0, o0, pc0, ch0, sw0;
    int on = VKA_ENV("DM_VK_STATS", 1) != 0;
    if (on && f - f0 >= 300) {
        double n = (double)(f - f0);
        uint64_t sw = vka_exec_slot_wait_ns();
        if (f0) VLOG("replay up to frame %llu: per frame %.0f draws, %.0f render passes, %.0f clears / copies (%.0f in a pass), %.0f full barriers, %.0f inline uniform updates, %.1f submissions (%.1f parts); recording %.2f ms a frame (%.2f us a draw), waiting for a submission slot (the GPU behind) %.2f ms a frame",
                     (unsigned long long)f, (g_stat_draws - d0) / n, (vka_n_rp - rp0) / n, (g_stat_ops - o0) / n, (g_n_pclear - pc0) / n, (vka_n_bar - b0) / n, (g_n_ub - ub0) / n, (g_n_sub - s0) / n, (g_n_sub - s0 + g_n_chunks - ch0) / n, (g_rec_ns - r0) / n / 1e6,
                     g_stat_draws > d0 ? (g_rec_ns - r0) / 1e3 / (double)(g_stat_draws - d0) : 0.0, (sw - sw0) / n / 1e6);
        sw0 = sw;
        { char pb[400]; void vka_draw_prof(char *b, size_t n, double draws); vka_draw_prof(pb, sizeof pb, (double)(g_stat_draws - d0)); if (pb[0]) VLOG("draw path per draw (us): %s", pb);
          vka_enc_stats(pb, sizeof pb, n); if (pb[0]) VLOG("%s", pb);
          vka_floor_stats(pb, sizeof pb); if (pb[0]) VLOG("%s", pb);
          vka_rp_bar_stats(pb, sizeof pb, n); if (pb[0]) VLOG("%s", pb);
          { void vka_drv_memo_stats(char *b, size_t n); vka_drv_memo_stats(pb, sizeof pb); if (pb[0]) VLOG("%s", pb); } }
        f0 = f; d0 = g_stat_draws; rp0 = vka_n_rp; b0 = vka_n_bar; ub0 = g_n_ub; s0 = g_n_sub; r0 = g_rec_ns; o0 = g_stat_ops; pc0 = g_n_pclear; ch0 = g_n_chunks;
    }
}
uint64_t vka_frame(void) { return g_frame; }
void vka_exec_counts(uint64_t *ops, uint64_t *subs, uint64_t *rec_ns) { *ops = g_stat_ops; *subs = g_n_sub; *rec_ns = g_rec_ns; }
void mtl_stats(uint64_t *passes, uint64_t *draws) { if (passes) *passes = g_stat_ops; if (draws) *draws = g_stat_draws; }
void mtl_prepare(void) { vka_init(); }
void mtl_sub_index(const uint64_t *h, int n) { (void)h; (void)n; }
void mtl_exec_rest(const uint64_t *handles, int n) { (void)handles; (void)n; }

void vka_rp_queue_activate(uint64_t serial) {
    if (S.serial == serial) return;
    int i = 0; while (i < g_nqs && g_qs[i].serial != S.serial) i++;
    if (S.serial) { if (i == g_nqs && g_nqs < NQS) g_nqs++; if (i < NQS) g_qs[i] = S; }
    for (i = 0; i < g_nqs; i++) if (g_qs[i].serial == serial) { S = g_qs[i]; return; }
    memset(&S, 0, sizeof S); S.serial = serial;
}
void vka_rp_queue_forget(uint64_t serial) {
    for (int i = 0; i < g_nqs; i++) if (g_qs[i].serial == serial) { g_qs[i] = g_qs[--g_nqs]; break; }
    if (S.serial == serial) memset(&S, 0, sizeof S);
}

static void post(pop o) {
    if (g_npost == g_postcap) { g_postcap = g_postcap ? g_postcap * 2 : 64; g_post = realloc(g_post, (size_t)g_postcap * sizeof *g_post); }
    g_post[g_npost++] = o;
}
uint64_t guest_ns(void);
// Counter timestamps by segment (profile gpu_timestamps segment; DM_NVN_GPU_TS=segment / submission overrides it), as nvn_mtl.m's: a timestamp report also writes a GPU
// timestamp where it is recorded (TOP_OF_PIPE for TIMESTAMP_TOP, else BOTTOM_OF_PIPE), and once the submission is done each report gets the completion time less the GPU time
// from it to the submission's last report, so two reports of one submission bracket the GPU work between them (a game's dynamic resolution and frame pacing read that).
// Without it every report of a submission reads the same time (the completion)
enum { SEGQ = 128 };   // queries per submission slot
static VkQueryPool g_sqp; static int g_sq_n, g_sq_base, g_seg = -1;
static PFN_vkGetQueryPoolResults g_sq_get; static double g_sq_ns;
static int seg_ts(void) {
    if (g_seg < 0) { const char *e = getenv("DM_NVN_GPU_TS"); g_seg = e && *e ? !strcmp(e, "segment") : DM_GAME_GPU_TS_SEGMENT;
                     VLOG("counter timestamps: %s (profile key gpu_timestamps; DM_NVN_GPU_TS=segment / submission)", g_seg ? "GPU time of their place in the submission" : "one time for the whole submission"); }
    return g_seg;
}
static void post_run(void *ops_, int n) {   // the completion thread, in submission order, before the fence reads as done
    pop *ops = ops_;
    uint64_t g = guest_ns(); if (!g) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); g = (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
    uint64_t ticks = nvn_timestamp_from_ns(g);
    uint64_t gq[SEGQ], tmax = 0; int nq = 0;   // segment timestamps: each report's GPU time, the latest of them
    for (int i = 0; i < n && g_sqp; i++) if (ops[i].kind == 2 && ops[i].v) {
        uint64_t t = 0; if (g_sq_get(V.dev, g_sqp, ops[i].v - 1, 1, 8, &t, 8, VK_QUERY_RESULT_64_BIT) || !t) { ops[i].v = 0; continue; }
        if (nq < SEGQ) { gq[nq++] = t; if (t > tmax) tmax = t; } else ops[i].v = 0;
    }
    if (atomic_load_explicit(&g_synclog, memory_order_relaxed) > 0) for (int i = 0; i < n; i++) if (ops[i].kind == 2 && ops[i].type == NVN_COUNTER_TYPE_TIMESTAMP) {
        VLOG("sync done: %d reports written at %.2f ms (the submission's GPU work done)", n, ms_now()); break; }
    for (int i = 0, q = 0; i < n; i++) {
        if (ops[i].kind == 3) {   // the frame's span to here (the adaptive GPU clock floor)
            uint64_t f = ops[i].addr; unsigned k = f % 64;
            vka_floor_done(ops[i].v);
            if (atomic_load(&g_fstart_f[k]) == f) { uint64_t t0 = atomic_load(&g_fstart[k]), t1 = now_ns(); if (t1 > t0) vka_floor_span(f, t1 - t0); }
            continue;
        }
        if (ops[i].kind == 2 && ops[i].v) {   // a segment timestamp: the completion time less the GPU time from this report to the last
            uint64_t back = (uint64_t)((double)(tmax - gq[q++]) * g_sq_ns); uint64_t tk = nvn_timestamp_from_ns(g > back ? g - back : 0), tt = ops[i].type;
            uint64_t rec[2] = {DM_GAME_RENDER_COND ? 0 : tk, tk}; (void)tt;
            memcpy((void *)(uintptr_t)ops[i].addr, rec, sizeof rec); if (DM_GAME_RENDER_COND) rc_pend_done(ops[i].addr); continue;
        }
        if (ops[i].kind == 1) { volatile uint32_t *v = (volatile uint32_t *)(uintptr_t)ops[i].addr; if (ops[i].mode == NVN_EVENT_SIGNAL_MODE_ADD) *v += ops[i].v; else *v = ops[i].v; }
        else {
            // A timestamp report's counter: the time, or 0 under render_cond (as on the GPU, nvn_mtl.m rc_value). Samples passed: 1, so occlusion culling keeps everything visible until draws count
            uint64_t t = ops[i].type, rec[2] = {t == NVN_COUNTER_TYPE_TIMESTAMP || t == NVN_COUNTER_TYPE_TIMESTAMP_TOP ? (DM_GAME_RENDER_COND ? 0 : ticks) : t == NVN_COUNTER_TYPE_SAMPLES_PASSED ? 1 : 0, ticks};
            memcpy((void *)(uintptr_t)ops[i].addr, rec, sizeof rec);
            if (DM_GAME_RENDER_COND) rc_pend_done(ops[i].addr);
        }
    }
    free(ops);
}

// ---------- GPU time ----------
// A timestamp at the top and the bottom of each part of a submission (chunk: a submission goes to the queue in up to NCHUNK + 1 parts, two queries per part in the slot's
// range); once the slot's fence is done the parts' busy times go to the frame the submission was recorded in, which hle_nvn.c reads 8 frames later (mtl_gpu_ms: the
// [longframe] lines, the frame statistics). The queue runs one part after another, so the sum is the frame's GPU busy time. Busy is counted per part: with one pair for the
// whole submission (before 2026-11-03) the GPU's idle time between parts, while the replay thread was still recording the next one (a pipeline compiled, a texture
// deswizzled, the encoder held up), counted as GPU time, and frames stalled on the CPU looked GPU-bound in the [longframe] lines (200-980 ms "GPU" spikes with no new
// pipeline). The span (first part's top to the last part's bottom) is kept beside it: span minus busy is the time the GPU waited for the recording (vka_gpu_span_ms).
// Without timestamps on the queue the frame reads -1 (no GPU time) instead of 0
static struct { PFN_vkCreateQueryPool create; PFN_vkCmdResetQueryPool reset; PFN_vkCmdWriteTimestamp write; PFN_vkGetQueryPoolResults get; } qf;
static VkQueryPool g_qp; static int g_ts_ok = -1; static double g_ts_ns;   // nanoseconds per tick
enum { GRING = 64 };
static struct { _Atomic uint64_t frame, ns, t0, t1, span; _Atomic int pending; } g_gpu[GRING];
static uint64_t g_slot_frame[8]; static uint8_t g_slot_ts[8]; static uint32_t g_slot_parts[8];
enum { QPS = 2 * (NCHUNK + 1) };   // queries a slot: a top and a bottom per part
static int ts_init(void) {
    if (g_ts_ok >= 0) return g_ts_ok;
    g_ts_ok = 0;
    uint32_t nq = 0; vk.vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &nq, NULL);
    VkQueueFamilyProperties qp[16]; if (nq > 16) nq = 16; vk.vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &nq, qp);
    if (V.qf >= nq || !qp[V.qf].timestampValidBits || V.props.limits.timestampPeriod <= 0) { VRED("GPU time: the queue has no timestamps"); return 0; }
    qf.create = (PFN_vkCreateQueryPool)vk.vkGetDeviceProcAddr(V.dev, "vkCreateQueryPool"); qf.reset = (PFN_vkCmdResetQueryPool)vk.vkGetDeviceProcAddr(V.dev, "vkCmdResetQueryPool");
    qf.write = (PFN_vkCmdWriteTimestamp)vk.vkGetDeviceProcAddr(V.dev, "vkCmdWriteTimestamp"); qf.get = (PFN_vkGetQueryPoolResults)vk.vkGetDeviceProcAddr(V.dev, "vkGetQueryPoolResults");
    VkQueryPoolCreateInfo ci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = NSLOT * QPS};
    if (!qf.create || !qf.reset || !qf.write || !qf.get || qf.create(V.dev, &ci, NULL, &g_qp)) { VRED("GPU time: no timestamp query pool"); return 0; }
    g_ts_ns = V.props.limits.timestampPeriod; g_ts_ok = 1;
    VLOG("GPU time: timestamps at %.2f ns a tick", g_ts_ns);
    return 1;
}
static void gpu_tag_begin(uint64_t f) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f) { atomic_store(&g_gpu[k].frame, f); atomic_store(&g_gpu[k].ns, 0); atomic_store(&g_gpu[k].pending, 0); atomic_store(&g_gpu[k].t0, UINT64_MAX); atomic_store(&g_gpu[k].t1, 0); atomic_store(&g_gpu[k].span, 0); }
    atomic_fetch_add(&g_gpu[k].pending, 1);
}
static void gpu_slot_done(unsigned slot) {   // the slot's fence is done: its two timestamps are there
    if (!g_slot_ts[slot]) return;
    g_slot_ts[slot] = 0;
    uint32_t np = g_slot_parts[slot] ? g_slot_parts[slot] : 1; if (np > NCHUNK + 1) np = NCHUNK + 1;
    uint64_t f = g_slot_frame[slot], t[QPS]; unsigned k = f % GRING;
    int ok = !qf.get(V.dev, g_qp, slot * QPS, 2 * np, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT);
    if (atomic_load(&g_gpu[k].frame) != f) return;
    if (ok && t[2 * np - 1] > t[0]) {
        // Busy: each part from its top (or the previous part's bottom, when the part's top was written while the previous part still ran) to its bottom
        uint64_t busy = 0, prev = 0;
        for (uint32_t i = 0; i < np; i++) { uint64_t a = t[2 * i] > prev ? t[2 * i] : prev, b = t[2 * i + 1]; if (b > a) busy += b - a; if (b > prev) prev = b; }
        uint64_t s0 = (uint64_t)(t[0] * g_ts_ns), s1 = (uint64_t)(t[2 * np - 1] * g_ts_ns);
        atomic_fetch_add(&g_gpu[k].ns, (uint64_t)(busy * g_ts_ns)); atomic_fetch_add(&g_gpu[k].span, s1 - s0);
        if (s0 < atomic_load(&g_gpu[k].t0)) atomic_store(&g_gpu[k].t0, s0);
        if (s1 > atomic_load(&g_gpu[k].t1)) atomic_store(&g_gpu[k].t1, s1);
    }
    atomic_fetch_sub(&g_gpu[k].pending, 1);
}
double mtl_gpu_ms(uint64_t f) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f || atomic_load(&g_gpu[k].pending)) return -1;
    return atomic_load(&g_gpu[k].ns) / 1e6;
}
// The submissions' spans of frame f (first part's start to last part's end, summed over its submissions): above mtl_gpu_ms by the time the GPU waited for the recording; -1 = not in yet
double vka_gpu_span_ms(uint64_t f) {
    unsigned k = f % GRING;
    if (atomic_load(&g_gpu[k].frame) != f || atomic_load(&g_gpu[k].pending)) return -1;
    return atomic_load(&g_gpu[k].span) / 1e6;
}
int mtl_gpu_span(uint64_t f, uint64_t *t0, uint64_t *t1) { (void)f; if (t0) *t0 = 0; if (t1) *t1 = 0; return 0; }   // GPU ticks are not on the host clock (no calibrated timestamps yet)

void vka_perf_thread(int nice);   // vka_perf.c
// What each recent submission held, by fence, for the [RED] line of a GPU hang (vka_dev.c hang_declare)
enum { NHREC = 16 };
static struct { uint64_t fence, frame, queue, draws, ops, skipped, t_ns; } g_hrec[NHREC];
static uint64_t g_sub_draws0, g_sub_ops0, g_sub_skip0, g_sub_t0;
void vka_exec_hang_report(uint64_t fence) {
    for (int i = 0; i < NHREC; i++) if (g_hrec[i].fence == fence || g_hrec[i].fence == fence + 1) {
        VRED("GPU hang: submission %llu (frame %llu, queue serial %llu): %llu draws, %llu clears / copies, %llu skipped, recorded in %.2f ms", (unsigned long long)g_hrec[i].fence,
             (unsigned long long)g_hrec[i].frame, (unsigned long long)g_hrec[i].queue, (unsigned long long)g_hrec[i].draws, (unsigned long long)g_hrec[i].ops,
             (unsigned long long)g_hrec[i].skipped, g_hrec[i].t_ns / 1e6); }
}

// ---------- DM_VK_RPROF: the recording thread's CPU split per frame (vka_rprof.h) ----------
// Each submission's wall time from vka_rp_sub_begin to the end of vka_rp_sub_commit is cut into: its begin (the command buffer slot's wait for the GPU, vka_tex_prepare,
// vka_draw_sub_begin), its command records by kind (draws, with the part ends before them; dispatches; clears; copies; buffer fills, copies and updates; state, barriers,
// reports and pass ends) and its commit (pass end, linear texture writeback, the texture-use feedback copy, the encoder's drain, upload flush, submit). Inside the draws, the
// DM_VK_PROF phases and vka_rprof.h's parts. The thread's CPU time and major page faults (getrusage RUSAGE_THREAD) over the same windows tell a blocked or faulting
// thread from a busy one. Summed over the submissions of one game frame (the worker's job frame) and logged as one `[rprof]` line for one frame in DM_VK_RPROF
enum { RK_SLOTW, RK_TEXPREP, RK_DSB, RK_BEGIN, RK_PARTS, RK_STATE, RK_DRAW, RK_DISPATCH, RK_CLEAR, RK_COPY, RK_BUF, RK_C_PASS, RK_C_WB, RK_C_TEXFB, RK_C_ENC, RK_C_UP, RK_C_SUBMIT, RK_COMMIT, RK_N };
int vka_rprof_on; uint64_t vka_rprof_x[VKA_RPX_N];
static struct { int have; uint64_t f, nf, subs, draws0, draws, wall, cpu, majflt, k[RK_N], pt[VKA_RP_PT], ps[VKA_RP_PS], x[VKA_RPX_N], enc, t0, c0, m0; } g_rp;
uint64_t vka_enc_waits(void);
static uint64_t rp_cpu_ns(void) { struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
static uint64_t rp_majflt(void) { struct rusage u; return getrusage(RUSAGE_THREAD, &u) ? 0 : (uint64_t)u.ru_majflt; }
static int rp_kind(uint32_t op) {
    switch (op) {
    case OP_DRAW: case OP_DRAW_IDX: return RK_DRAW;
    case OP_DISPATCH: return RK_DISPATCH;
    case OP_CLEAR_COLOR: case OP_CLEAR_DS: case OP_CLEAR_TEX: return RK_CLEAR;
    case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: return RK_COPY;
    case OP_CLEAR_BUF: case OP_COPY_B2B: case OP_UPDATE_UB: return RK_BUF;
    default: return RK_STATE;
    }
}
static void rp_line(void) {   // the frame g_rp holds, as differences from its start
    uint64_t pt[VKA_RP_PT], ps[VKA_RP_PS]; vka_draw_rprof(pt, ps);
    for (int i = 0; i < VKA_RP_PT; i++) pt[i] -= g_rp.pt[i];
    for (int i = 0; i < VKA_RP_PS; i++) ps[i] -= g_rp.ps[i];
    uint64_t x[VKA_RPX_N]; for (int i = 0; i < VKA_RPX_N; i++) x[i] = vka_rprof_x[i] - g_rp.x[i];
    const uint64_t *k = g_rp.k;
    uint64_t begin_parts = k[RK_SLOTW] + k[RK_TEXPREP] + k[RK_DSB], commit_parts = k[RK_C_PASS] + k[RK_C_WB] + k[RK_C_TEXFB] + k[RK_C_ENC] + k[RK_C_UP] + k[RK_C_SUBMIT];
    uint64_t ops = k[RK_PARTS] + k[RK_STATE] + k[RK_DRAW] + k[RK_DISPATCH] + k[RK_CLEAR] + k[RK_COPY] + k[RK_BUF], known = k[RK_BEGIN] + ops + k[RK_COMMIT];
    uint64_t pass_key = pt[2] > x[VKA_RPX_IMAGES] ? pt[2] - x[VKA_RPX_IMAGES] : 0;   // (the rest of the phase: the walk, the draw's part of the pool scan, the pass and the key)
#define M(t) vka_rprof_ms(t)
    VLOG("[rprof] frame %llu: %llu submissions, %llu draws, recording %.1f ms (thread CPU %.1f, %llu major faults on it); "
         "begin %.1f (slot wait %.1f, texture prepare %.1f: feedback readbacks %.1f, evictions %.1f, mip drop + coarse %.1f; draw state + set pools %.1f, other %.1f); "
         "draws %.1f (part ends %.1f; resolve %.1f, layout %.1f, images %.1f, walk + pool scan + pass + key %.1f, vertex input %.1f, pipeline %.1f, sets %.1f, vertex buffers + dynamic %.1f, draw call %.1f); "
         "sets with dispatches: walk %.1f, writes %.1f (page index %.1f, table %.1f, driver segments %.1f, images %.1f), push %.1f; pool scans %.1f; "
         "dispatches %.1f, clears %.1f, copies %.1f, buffer ops %.1f, state + barriers + pass ends %.1f; "
         "commit %.1f (pass end %.1f, writeback %.1f, feedback copy %.1f, encoder drain %.1f, upload flush %.1f, submit %.1f, other %.1f); waited for the encoder %.1f; unattributed %.1f",
         (unsigned long long)g_rp.f, (unsigned long long)g_rp.subs, (unsigned long long)g_rp.draws, g_rp.wall / 1e6, g_rp.cpu / 1e6, (unsigned long long)g_rp.majflt,
         M(k[RK_BEGIN]), M(k[RK_SLOTW]), M(k[RK_TEXPREP]), M(x[VKA_RPX_TEXFB]), M(x[VKA_RPX_EVICT]), M(x[VKA_RPX_MIPDROP]), M(k[RK_DSB]), M(k[RK_BEGIN] > begin_parts ? k[RK_BEGIN] - begin_parts : 0),
         M(k[RK_DRAW] + k[RK_PARTS]), M(k[RK_PARTS]), M(pt[0]), M(pt[1]), M(x[VKA_RPX_IMAGES]), M(pass_key), M(pt[3]), M(pt[4]), M(pt[5]), M(pt[6]), M(pt[7]),
         M(ps[0]), M(ps[1]), M(x[VKA_RPX_PIDX]), M(x[VKA_RPX_GMAP]), M(ps[3]), M(ps[4]), M(ps[2]), M(x[VKA_RPX_POOLSCAN]),
         M(k[RK_DISPATCH]), M(k[RK_CLEAR]), M(k[RK_COPY]), M(k[RK_BUF]), M(k[RK_STATE]),
         M(k[RK_COMMIT]), M(k[RK_C_PASS]), M(k[RK_C_WB]), M(k[RK_C_TEXFB]), M(k[RK_C_ENC]), M(k[RK_C_UP]), M(k[RK_C_SUBMIT]), M(k[RK_COMMIT] > commit_parts ? k[RK_COMMIT] - commit_parts : 0),
         (vka_enc_waits() - g_rp.enc) / 1e6, g_rp.wall / 1e6 > M(known) ? g_rp.wall / 1e6 - M(known) : 0.0);
#undef M
}
static void rp_begin(uint64_t f) {   // vka_rp_sub_begin's first step: a new frame logs the last one (one in n) and starts its own sums
    int n = VKA_ENV("DM_VK_RPROF", 0);
    if (n <= 0) { vka_rprof_on = 0; g_rp.have = 0; return; }
    if (!g_rp.have || f != g_rp.f) {
        if (g_rp.have && g_rp.subs && g_rp.nf % (uint64_t)n == 0) rp_line();
        uint64_t nf = g_rp.have ? g_rp.nf + 1 : 0;
        memset(&g_rp, 0, sizeof g_rp); g_rp.have = 1; g_rp.f = f; g_rp.nf = nf;
        vka_draw_rprof(g_rp.pt, g_rp.ps); memcpy(g_rp.x, vka_rprof_x, sizeof g_rp.x); g_rp.enc = vka_enc_waits(); g_rp.draws0 = g_stat_draws;
    }
    vka_rprof_on = 1; g_rp.subs++;
    g_rp.t0 = now_ns(); g_rp.c0 = rp_cpu_ns(); g_rp.m0 = rp_majflt();
}
static void rp_end(void) {   // the end of vka_rp_sub_commit
    if (!vka_rprof_on) return;
    g_rp.wall += now_ns() - g_rp.t0; g_rp.cpu += rp_cpu_ns() - g_rp.c0; g_rp.majflt += rp_majflt() - g_rp.m0; g_rp.draws = g_stat_draws - g_rp.draws0;
    vka_rprof_on = 0;
}
#define RP_T() (vka_rprof_on ? vka_rprof_t() : 0)
#define RP_K(i, t0) do { if (vka_rprof_on) g_rp.k[i] += vka_rprof_t() - (t0); } while (0)

void vka_rp_sub_begin(void) {
    rp_begin(g_job_frame_set ? g_job_frame : g_frame);
    uint64_t rb_ = RP_T();
    vka_perf_thread(0); vka_ctl_sub();
    { uint64_t f = g_job_frame_set ? g_job_frame : g_frame; if (atomic_load(&g_fstart_f[f % 64]) != f) { atomic_store(&g_fstart[f % 64], now_ns()); atomic_store(&g_fstart_f[f % 64], f); } }
    g_npost = 0; g_cb = VK_NULL_HANDLE;
    if (!vka_init()) { RP_K(RK_BEGIN, rb_); return; }
    vka_gpu_floor();
    if (vka_hung()) { VRED_ONCE("GPU hang: submissions from now on run their effects without the GPU"); RP_K(RK_BEGIN, rb_); return; }   // no command buffer: mtl_sub_commit takes the CPU path
    g_sub_draws0 = g_stat_draws; g_sub_ops0 = g_stat_ops; g_sub_skip0 = g_stat_skipped; g_n_sub++;
    if (!g_pool) {
        VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = V.qf};
        VkCommandBuffer cbs[NSLOT];
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = NSLOT};
        if (vk.vkCreateCommandPool(V.dev, &pi, NULL, &g_pool) || ((ai.commandPool = g_pool), vk.vkAllocateCommandBuffers(V.dev, &ai, cbs))) { VRED("submission command buffers not made"); g_pool = VK_NULL_HANDLE; RP_K(RK_BEGIN, rb_); return; }
        for (int i = 0; i < NSLOT; i++) g_slot[i].cb = cbs[i];
    }
    unsigned k = g_slot_i++ % NSLOT;
    if (g_slot[k].fence) { uint64_t w0 = now_ns(), rw_ = RP_T(); vka_wait_done(g_slot[k].fence, UINT64_MAX); atomic_fetch_add_explicit(&g_slotw_ns, now_ns() - w0, memory_order_relaxed); RP_K(RK_SLOTW, rw_); }
    g_sub_t0 = now_ns();   // (the recording time from here: the slot wait above is g_slotw_ns)
    uint64_t done = vka_done();
    for (unsigned i = 0; i < NSLOT; i++) if (g_slot_ts[i] && g_slot[i].fence <= done) gpu_slot_done(i);
    g_cb = g_slot[k].cb;
    vk.vkResetCommandBuffer(g_cb, 0);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vk.vkBeginCommandBuffer(g_cb, &bi);
    if (ts_init()) {
        qf.reset(g_cb, g_qp, k * QPS, QPS); qf.write(g_cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_qp, k * QPS);
        g_slot_frame[k] = g_frame; g_slot_ts[k] = 1; g_slot_parts[k] = 1; gpu_tag_begin(g_frame);
        g_sq_n = 0; g_sq_base = (int)k * SEGQ;
        if (seg_ts() && !g_sqp) {
            VkQueryPoolCreateInfo ci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = NSLOT * SEGQ};
            if (qf.create(V.dev, &ci, NULL, &g_sqp)) { g_sqp = VK_NULL_HANDLE; g_seg = 0; VRED("segment timestamps: no query pool: one time for the whole submission"); }
            g_sq_get = qf.get; g_sq_ns = g_ts_ns;
        }
        if (g_sqp) qf.reset(g_cb, g_sqp, (uint32_t)g_sq_base, SEGQ);
    }
    vka_gp_sub_begin(g_cb, g_frame);
    vka_enc_start(g_cb);
    vka_barrier(g_cb);   // the host's writes to guest memory before this submission (made available by the submit), and earlier submissions' writes
    { uint64_t rt_ = RP_T(); vka_tex_prepare(g_cb); RP_K(RK_TEXPREP, rt_); }
    vka_barrier(g_cb);   // the re-uploads of CPU-written textures before the submission's draws read them
    { uint64_t rd_ = RP_T(); vka_draw_sub_begin(g_cb, k); RP_K(RK_DSB, rd_); }
    { void vka_pso_pause(int on); vka_pso_pause(0); }   // the game submits: resumed (mtl_drain paused the pipeline threads)
    g_chunk_draws = VKA_ENV("DM_VK_CHUNK", 600);   // live: from the next submission
    g_nx = 0; g_chunk_d0 = g_stat_draws;
    RP_K(RK_BEGIN, rb_);
}

// A long submission goes to the queue in parts, so the GPU starts on its first passes while the rest is recorded: before the first pass after DM_VK_CHUNK draws (default
// 600; 0 = one part) of the part in progress, the part is ended and submitted without a signal, and the submission goes on in the slot's next command buffer. The parts run
// in order on the one queue, and the barriers recorded in a part cover what later parts do (submission order); only the last part signals the submission's fence.
// On the Adreno 840 phone, Donkey Kong Bananza's level recorded ~45 ms a frame before its GPU work (~30 ms) began, and the game waited for that frame's fence (@sync_ms ~64)
typedef struct { VkCommandBuffer cb; uint32_t q; } part_arg;   // q: the part's bottom (part_submit) or top (part_begin) timestamp query, UINT32_MAX = none
static void part_submit(const void *a) {   // the part ends and goes to the queue (on the encoder thread when it is on)
    part_arg pa; memcpy(&pa, a, sizeof pa); VkCommandBuffer cb = pa.cb;
    if (pa.q != UINT32_MAX) qf.write(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_qp, pa.q);
    VkResult r = vk.vkEndCommandBuffer(cb);
    if (!r) r = vka_submit(cb, 0, 0, NULL, NULL, NULL, VK_NULL_HANDLE);
    if (r) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 4) VRED("part of a submission: %d", r); }
}
static void part_begin(const void *a) {
    part_arg pa; memcpy(&pa, a, sizeof pa); VkCommandBuffer cb = pa.cb;
    vk.vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vk.vkBeginCommandBuffer(cb, &bi);
    if (pa.q != UINT32_MAX) qf.write(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, g_qp, pa.q);
}
static void chunk(void) {
    unsigned k = (g_slot_i - 1) % NSLOT;
    if (g_nx >= NCHUNK) return;
    if (!g_slot[k].x[g_nx]) {
        vka_enc_sync();   // the pool: not while the encoder thread records into one of its command buffers
        VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        if (vk.vkAllocateCommandBuffers(V.dev, &ai, &g_slot[k].x[g_nx])) { g_slot[k].x[g_nx] = VK_NULL_HANDLE; g_nx = NCHUNK; return; }
    }
    VkCommandBuffer next = g_slot[k].x[g_nx++];
    // The part's bottom and the next part's top timestamps (the slot's queries were reset at the submission's start, in its first part)
    uint32_t part = g_slot_parts[k]; int ts = g_slot_ts[k] && part < NCHUNK + 1;
    part_arg pe = {g_cb, ts ? k * QPS + 2 * (part - 1) + 1 : UINT32_MAX}, pb = {next, ts ? k * QPS + 2 * part : UINT32_MAX};
    if (ts) g_slot_parts[k] = part + 1;
    vka_up_flush(); vka_up_throttle();   // the upload batch goes first: the part may use what it made (vka_dev.c "Upload batches")
    // With the encoder the part's end, its submit and the next part's reset and begin are made on its thread, in order with the calls before: the replay thread goes on
    if (vka_enc_call(part_submit, &pe, sizeof pe)) { vka_enc_call(part_begin, &pb, sizeof pb); vka_enc_switch(next); }
    else { part_submit(&pe); part_begin(&pb); }
    g_cb = next;
    vka_draw_cb_switch(g_cb); vka_gp_part(g_cb);
    g_chunk_d0 = g_stat_draws; g_n_chunks++;
}
// DM_VK_CHUNK_PASS=n (live; default 600, 0 = off): a pass that has had n draws in the part in progress is also ended there and the part submitted; the next draw opens it again over the
// same targets (loadOp LOAD, after the pass end's barrier), so its contents and the draws' order are kept. A pass of 1,700 draws whose GPU work (14-18 ms) only began after
// all of it was recorded then starts on the GPU while the rest is recorded. Donkey Kong Bananza's level (Adreno 840, DM_VK_GPUPROF): the extra splits cost no measurable GPU
// time (20.2-20.5 ms a frame with them, 20.8-21.0 without); with the GPU clock floor the GPU finished 6-10 ms after recording instead of 10-33
static inline void chunk_check(void) {
    if (!g_cb) return;
    uint64_t in = g_stat_draws - g_chunk_d0;
    if (g_chunk_draws > 0 && in >= (uint64_t)g_chunk_draws && vka_rp_split_ok(&S.rt, S.has_rt)) { vka_rp_end(); chunk(); return; }
    int cp = VKA_ENV("DM_VK_CHUNK_PASS", 600);
    if (cp > 0 && in >= (uint64_t)cp && vka_rp_open_for(&S.rt, S.has_rt)) { vka_rp_end(); chunk(); }
}
static uint64_t g_cur_fence;   // the fence of the submission being replayed, reserved before it (vka_worker.c); 0 = reserved at its commit
static uint64_t sub_commit(uint64_t reserved);
uint64_t vka_rp_sub_commit(uint64_t reserved) { uint64_t rc_ = RP_T(), f = sub_commit(reserved); RP_K(RK_COMMIT, rc_); rp_end(); return f; }
static uint64_t sub_commit(uint64_t reserved) {
    uint64_t fence = reserved ? reserved : vka_fence_next(); g_cur_fence = 0;
    int tail = 0;   // counted for the tail floor (DM_VK_GPU_FLOOR_AUTO=2): given back by the span op when it completes
    if (g_cb && g_stat_draws > g_sub_draws0) { int vka_floor_tail_add(void); tail = vka_floor_tail_add(); post((pop){3, 0, (uint32_t)tail, 0, g_job_frame_set ? g_job_frame : g_frame}); }   // the frame's span at the end of this submission with draws (post_run)
    pop *ops = g_npost ? malloc((size_t)g_npost * sizeof *ops) : NULL; if (ops) memcpy(ops, g_post, (size_t)g_npost * sizeof *ops);
    int n = g_npost; g_npost = 0;
    if (!g_cb) { vka_fence_cpu(fence, ops, n, post_run); return fence; }
    { uint64_t r_ = RP_T(); vka_rp_end(); RP_K(RK_C_PASS, r_); }
    { uint64_t r_ = RP_T(); vka_tex_writeback(g_cb); RP_K(RK_C_WB, r_); }   // linear textures the GPU wrote, back into their storage for the CPU
    { uint64_t r_ = RP_T(); vka_tex_fb_commit(g_cb, fence); RP_K(RK_C_TEXFB, r_); }   // the texture IDs its draws indexed (DM_VK_TEXFB): copied out for the host (the barrier below) and cleared
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT};
    vk.vkCmdPipelineBarrier(g_cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    { uint64_t r_ = RP_T(); vka_enc_stop(); RP_K(RK_C_ENC, r_); }   // every call made: the command buffer is complete
    unsigned k = (g_slot_i - 1) % NSLOT;
    if (g_slot_ts[k]) qf.write(g_cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_qp, k * QPS + 2 * (g_slot_parts[k] - 1) + 1);
    vka_gp_commit(g_cb, fence);
    VkResult r = vk.vkEndCommandBuffer(g_cb);
    g_rec_ns += now_ns() - g_sub_t0;
    g_hrec[fence % NHREC] = (typeof(g_hrec[0])){fence, g_frame, S.serial, g_stat_draws - g_sub_draws0, g_stat_ops - g_sub_ops0, g_stat_skipped - g_sub_skip0, now_ns() - g_sub_t0};
    { static int tl = -1; static uint64_t tn; if (tl < 0) { const char *e_ = getenv("DM_TMP_FRAMEDRAWS"); tl = e_ ? atoi(e_) : 0; }   // TEMP: per submission: frame, draws, skipped, ops
      if (tl > 0 && g_stat_draws - g_sub_draws0 >= 50 && tn++ < (uint64_t)tl) VLOG("TMPFD %llu %llu %llu %llu", (unsigned long long)(g_job_frame_set ? g_job_frame : g_frame), (unsigned long long)(g_stat_draws - g_sub_draws0), (unsigned long long)(g_stat_skipped - g_sub_skip0), (unsigned long long)(g_stat_ops - g_sub_ops0)); }
    vka_tex_gw_commit(fence);
    vka_floor_commit(tail);
    { uint64_t r_ = RP_T(); vka_up_flush(); vka_up_throttle(); RP_K(RK_C_UP, r_); }   // before the submission that may use what it made
    { uint64_t r_ = RP_T(); if (!r) r = vka_submit(g_cb, fence, 0, NULL, NULL, NULL, VK_NULL_HANDLE); RP_K(RK_C_SUBMIT, r_); }
    g_slot[(g_slot_i - 1) % NSLOT].fence = fence;
    if (vka_nan_on()) vka_nan_commit(fence);
    vka_scratch_retire(fence); vka_gmem_retire(fence);
    if (r) {
        static _Atomic int nred; if (atomic_fetch_add(&nred, 1) < 8) VRED("submission %llu: %d (its effects run without the GPU)", (unsigned long long)fence, r);
        VkSemaphoreSignalInfo si = {VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = V.tl, .value = fence};
        pthread_mutex_lock(&V.qmu);   // signalled by the host: later submissions may wait for it
        VkResult sr = vka_hung() ? VK_ERROR_DEVICE_LOST : vk.vkSignalSemaphore(V.dev, &si);
        if (!sr && fence > V.tl_queued) __atomic_store_n(&V.tl_queued, fence, __ATOMIC_RELEASE);
        pthread_mutex_unlock(&V.qmu);
        if (sr) { vka_fence_cpu(fence, ops, n, post_run); g_cb = VK_NULL_HANDLE; return fence; }
    }
    vka_fence_post(fence, ops, n, post_run);
    if (atomic_load_explicit(&g_synclog, memory_order_relaxed) > 0)
        VLOG("sync sub: frame %llu fence %llu: %llu draws, %d parts, recorded %.2f-%.2f ms, queued", (unsigned long long)g_frame, (unsigned long long)fence,
             (unsigned long long)(g_stat_draws - g_sub_draws0), g_nx + 1, (g_sub_t0 % 1000000000000ull) / 1e6, ms_now());
    g_cb = VK_NULL_HANDLE;
    return fence;
}
uint64_t mtl_last_fence(void) { return vka_fence_last(); }
// DM_VK_SYNCLOG=n (live): the next n of the game's blocking waits (the fence against the last reserved and the last queued, when the wait began and ended), and each
// submission's recording and queueing times (vka_rp_sub_commit), on one clock (ms): where a frame's wait goes. The count starts again when the value changes
static int synclog(void) { static int last; int v = VKA_ENV("DM_VK_SYNCLOG", 0); if (v != last) { last = v; atomic_store(&g_synclog, v); } return atomic_load(&g_synclog) > 0; }
int mtl_wait(uint64_t fence, uint64_t timeout_ns) {
    if (timeout_ns && vka_done() < fence && synclog()) {
        double t0 = ms_now(); int r = vka_wait_done(fence, timeout_ns);
        if (atomic_fetch_sub(&g_synclog, 1) > 0) VLOG("sync wait: frame %llu fence %llu (last reserved %llu, queued %llu, done %s): %.2f-%.2f ms, %.2f ms", (unsigned long long)g_frame, (unsigned long long)fence,
                                                (unsigned long long)vka_fence_last(), (unsigned long long)vka_fence_queued(), r ? "yes" : "timeout", t0, ms_now(), ms_now() - t0);
        return r;
    }
    return vka_wait_done(fence, timeout_ns);
}
int mtl_wait_acq(uint64_t fence, uint64_t timeout_ns) { return vka_wait_done(fence, timeout_ns); }
void mtl_finish(void) { vka_wait_done(vka_fence_last(), UINT64_MAX); }
void mtl_present_flush(void);
void vka_pso_pause(int on);
void mtl_drain(void) { vka_pso_pause(1); vka_pso_save(1); mtl_present_flush(); mtl_finish(); if (vka_ok()) vka_queue_idle(); }

// ---------- Buffers ----------
static void buf_copy(uint64_t src, uint64_t dst, uint64_t n) {
    vka_tex_gw_copy(src, dst, n);
    while (n) {
        VkBuffer sb, db; VkDeviceSize so, dof;
        uint64_t a = vka_addr_run(src, n, &sb, &so), b = vka_addr_run(dst, n, &db, &dof), k = a < b ? a : b;
        if (!k) { static _Atomic int nr; if (atomic_fetch_add(&nr, 1) < 8) VRED("buffer copy %#llx → %#llx (%llu bytes): %s is not in placed guest RAM", (unsigned long long)src, (unsigned long long)dst, (unsigned long long)n, a ? "the destination" : "the source"); return; }
        VkBufferCopy c = {so, dof, k}; vk.vkCmdCopyBuffer(g_cb, sb, db, 1, &c);
        src += k; dst += k; n -= k;
    }
}
static void buf_fill(uint64_t dst, uint64_t n, uint32_t value) {
    vka_tex_gw_fill(dst, n, value);
    if ((dst | n) & 3) { VRED_ONCE("buffer fill at %#llx of %llu bytes is not 4-byte aligned", (unsigned long long)dst, (unsigned long long)n); return; }
    while (n) {
        VkBuffer b; VkDeviceSize o; uint64_t k = vka_addr_run(dst, n, &b, &o);
        if (!k) { VRED_ONCE("buffer fill at %#llx is not in placed guest RAM", (unsigned long long)dst); return; }
        vk.vkCmdFillBuffer(g_cb, b, o, k & ~3ull, value); dst += k; n -= k;
    }
}
static void buf_update(const void *data, uint64_t dst, uint64_t n) {   // inline data, in GPU order: through scratch memory
    vka_tex_gw_update(dst, n, data);
    vka_span s; if (!n || !vka_scratch(n, 16, &s)) return;
    memcpy(s.ptr, data, n);
    for (uint64_t done = 0; done < n;) {
        VkBuffer b; VkDeviceSize o; uint64_t k = vka_addr_run(dst + done, n - done, &b, &o);
        if (!k) { VRED_ONCE("inline buffer update at %#llx is not in placed guest RAM", (unsigned long long)(dst + done)); return; }
        VkBufferCopy c = {s.off + done, o, k}; vk.vkCmdCopyBuffer(g_cb, s.buf, b, 1, &c); done += k;
    }
}

// ---------- Clears ----------
static float srgb_enc(float c) { c = c < 0 ? 0 : c > 1 ? 1 : c; return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1 / 2.4f) - 0.055f; }
static VkClearColorValue color_value(const vka_fmt *f, const float rgba[4]) {
    VkClearColorValue v;
    for (int i = 0; i < 4; i++) {
        if (f->kind == 1) v.uint32[i] = (uint32_t)rgba[i];
        else if (f->kind == 2) v.int32[i] = (int32_t)rgba[i];
        else v.float32[i] = f->srgb && i < 3 ? srgb_enc(rgba[i]) : rgba[i];
    }
    return v;
}
static int clear_zero(const VkClearColorValue *v) { return !(v->uint32[0] | v->uint32[1] | v->uint32[2] | v->uint32[3]); }   // all-zero bits: zero in every format a view may take
static void level_size(const vka_img *im, uint32_t lv, uint32_t *w, uint32_t *h) { *w = im->w >> lv ? im->w >> lv : 1; *h = im->h >> lv ? im->h >> lv : 1; }
static void rt_range(const vka_img *im, const view_t *v, int has_v, uint32_t *lv, uint32_t *l0, uint32_t *nl) {
    *lv = has_v && v->set_levels ? v->base_level : 0;
    *l0 = has_v && v->set_layers ? v->min_layer : 0;
    if (!vka_img_lv(im, lv)) { *lv = im->levels; *nl = 0; return; }   // a level the image leaves out (a sparse texture's unmapped levels)
    uint32_t all = vka_img_slices(im, *lv);
    *nl = has_v && v->set_layers && v->nlayers ? v->nlayers : all - (*l0 < all ? *l0 : 0);
    if (*l0 + *nl > all) *nl = *l0 < all ? all - *l0 : 0;
}
// A clear of part of a level: one dynamic-rendering pass over the attachment, vkCmdClearAttachments in the rectangle
static void clear_rect(vka_img *im, VkFormat fmt, VkImageAspectFlags asp, uint32_t lv, uint32_t l0, uint32_t nl, VkRect2D rc, const VkClearValue *val) {
    int color = asp == VK_IMAGE_ASPECT_COLOR_BIT;
    if (!(im->usage & (color ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))) {
        VRED_ONCE("partial clear of a texture that cannot be an attachment (serial %llu format %#x)", (unsigned long long)im->serial, im->d.format); return; }
    VkImageView view = vka_view(im, fmt, nl > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, im->aspect, lv, 1, l0, nl, NULL);
    if (!view) return;
    uint32_t w, h; level_size(im, lv, &w, &h);
    VkRenderingAttachmentInfo a = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = nl, .colorAttachmentCount = color, .pColorAttachments = color ? &a : NULL,
                          .pDepthAttachment = im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? &a : NULL, .pStencilAttachment = im->aspect & VK_IMAGE_ASPECT_STENCIL_BIT ? &a : NULL};
    vk.vkCmdBeginRendering(g_cb, &ri);
    VkClearAttachment ca = {asp, 0, *val}; VkClearRect cr = {rc, 0, nl};
    vk.vkCmdClearAttachments(g_cb, 1, &ca, 1, &cr);
    vk.vkCmdEndRendering(g_cb);
}
static int scissored(const vka_img *im, uint32_t lv, VkRect2D *rc) {   // 1 = the scissor leaves part of the level out; rc = what is left (may be empty)
    uint32_t w, h; level_size(im, lv, &w, &h);
    *rc = (VkRect2D){{0, 0}, {w, h}};
    if (!S.has_sc) return 0;
    int64_t x0 = S.sc.x > 0 ? S.sc.x : 0, y0 = S.sc.y > 0 ? S.sc.y : 0, x1 = (int64_t)S.sc.x + S.sc.w, y1 = (int64_t)S.sc.y + S.sc.h;
    if (x1 > w) x1 = w; if (y1 > h) y1 = h;
    if (x1 <= x0 || y1 <= y0) { rc->extent = (VkExtent2D){0, 0}; return 1; }
    *rc = (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
    return !(x0 == 0 && y0 == 0 && x1 == w && y1 == h);
}
// Clears of the bound targets go into the render pass over them (vka_rp_clear): the title's 63 draws took 59 passes when each clear and rebinding ended the pass. Outside a
// pass otherwise (vka_clear_outside); DM_VK_PASS_CLEARS=0 keeps them all outside
static int clear_in_pass(const vka_pclear *c) {
    int on = VKA_ENV("DM_VK_PASS_CLEARS", 1);
    if (on && vka_rp_clear(&S.rt, S.has_rt, &S.d, c)) { g_n_pclear++; return 1; }
    vka_rp_end();
    return 0;
}
void vka_clear_outside(const vka_pclear *c) {
    vka_img *im = c->im; VkClearValue val = c->out;
    if (c->ci >= 0) {
        int is3d = im->d.target == NVN_TEXTURE_TARGET_3D, some = is3d && (c->l0 || c->nl < vka_img_slices(im, c->lv));   // some of a 3D texture's slices: through a 2D-array view of them
        if (c->whole && c->fmt == im->f.vk && !some) {
            VkImageSubresourceRange rg = {VK_IMAGE_ASPECT_COLOR_BIT, c->lv, 1, is3d ? 0 : c->l0, is3d ? 1 : c->nl};
            vk.vkCmdClearColorImage(g_cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &val.color, 1, &rg);
        } else clear_rect(im, c->fmt, VK_IMAGE_ASPECT_COLOR_BIT, c->lv, c->l0, c->nl, c->rc, &val);
    } else {
        if (c->whole) { VkImageSubresourceRange rg = {c->asp, c->lv, 1, c->l0, c->nl}; vk.vkCmdClearDepthStencilImage(g_cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &val.depthStencil, 1, &rg); }
        else clear_rect(im, im->f.vk, c->asp, c->lv, c->l0, c->nl, c->rc, &val);
    }
    vka_barrier(g_cb);
}
static void clear_color(const p_clear_color *c) {
    vka_rp_prep_clear(&S.rt, S.has_rt);
    if (S.rc_skip) return;
    if (!S.has_rt || c->index < 0 || c->index >= S.rt.n || c->index >= 8 || !S.rt.color[c->index]) { VRED_ONCE("color clear of target %d, which is not bound", c->index); return; }
    if (!c->mask) return;
    const tex_t *t = S.rt.color[c->index]; const view_t *v = &S.rt.cview[c->index]; int hv = S.rt.has_cview[c->index];
    vka_rl_site = VKA_RL_CLEAR; vka_img *im = vka_tex_get(t, g_cb); vka_rl_site = 0; vka_fmt f;   // (vka_rl_site: DM_VK_RTLOG_MB's path)
    if (!im || !vka_fmt_of(hv && v->set_format ? v->format : t->d.format, &f)) return;
    if (f.kind) { VRED_ONCE("color clear of a depth texture (format %#x)", t->d.format); return; }
    if ((c->mask & 0xf) != 0xf) { VRED_ONCE("color clear with channel mask %#x is not on Vulkan yet (needs a pipeline): skipped", c->mask); return; }
    VkClearColorValue cz = color_value(&f, c->rgba); int zero = clear_zero(&cz);
    if (zero && vka_tex_zclear(im)) { vka_rtw_clear(im, 1, 0); g_stat_ops++; return; }   // (a zero-shared image keeps its zeros: vka_tex.c "Zero-shared images", DM_VK_ZSHARE_ZCLEAR)
    vka_tex_own(im, g_cb);   // (zero-shared: memory of its own before the clear, which may also become the next pass's loadOp)
    uint32_t lv, l0, nl; rt_range(im, v, hv, &lv, &l0, &nl);
    if (lv >= im->levels || !nl) return;
    VkRect2D rc; int part = scissored(im, lv, &rc);
    if (!rc.extent.width) return;
    vka_fmt fl = f; fl.srgb = 0;   // in a pass the attachment is the sRGB view: the clear colour stays linear there
    vka_pclear pc = {c->index, im, f.vk, VK_IMAGE_ASPECT_COLOR_BIT, lv, l0, nl, rc, !part, {.color = color_value(&f, c->rgba)}, {.color = color_value(&fl, c->rgba)}};
    vka_img_written(im); vka_rtw_clear(im, zero, cz.uint32[0]); g_stat_ops++;
    if (!clear_in_pass(&pc)) vka_clear_outside(&pc);
}
// A stencil clear with a write mask (agl clears single stencil bits it uses as masks): Vulkan's clears write every bit, so it is a draw. A full-viewport triangle (the present's
// rotation vertex shader, no fragment shader) with the stencil test always passing, REPLACE with the reference = the clear value, and the clear's write mask as the dynamic
// write mask; depth untouched. One pipeline per depth-stencil format and sample count
static struct { VkFormat f; uint32_t samples; VkPipeline p; } g_smc[8]; static int g_nsmc; static VkPipelineLayout g_smc_lay; static VkShaderModule g_smc_vs;
static VkPipeline smc_pipeline(const vka_img *im) {
    for (int i = 0; i < g_nsmc; i++) if (g_smc[i].f == im->f.vk && g_smc[i].samples == im->samples) return g_smc[i].p;
    if (g_nsmc == 8) return VK_NULL_HANDLE;
    VkResult r = 0;
    if (!g_smc_lay) {
        VkPushConstantRange pc = {VK_SHADER_STAGE_VERTEX_BIT, 0, 6 * sizeof(float)};
        VkPipelineLayoutCreateInfo li = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1, .pPushConstantRanges = &pc};
        VkShaderModuleCreateInfo mv = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof vka_rot_vert, .pCode = vka_rot_vert};
        if ((r = vk.vkCreatePipelineLayout(V.dev, &li, NULL, &g_smc_lay)) || (r = vk.vkCreateShaderModule(V.dev, &mv, NULL, &g_smc_vs))) { VRED("masked stencil clear: layout / shader %d", r); return VK_NULL_HANDLE; }
    }
    VkPipelineShaderStageCreateInfo st = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = g_smc_vs, .pName = "main"};
    VkPipelineVertexInputStateCreateInfo vi = {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vp = {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1};
    VkPipelineMultisampleStateCreateInfo ms = {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = (VkSampleCountFlagBits)im->samples};
    VkStencilOpState so = {VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE, VK_COMPARE_OP_ALWAYS, 0xff, 0, 0};
    VkPipelineDepthStencilStateCreateInfo dss = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO, .stencilTestEnable = VK_TRUE, .front = so, .back = so};
    VkPipelineColorBlendStateCreateInfo cb = {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkDynamicState dyn[4] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE};
    VkPipelineDynamicStateCreateInfo ds = {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO, .dynamicStateCount = 4, .pDynamicStates = dyn};
    VkPipelineRenderingCreateInfo rci = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .depthAttachmentFormat = im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? im->f.vk : VK_FORMAT_UNDEFINED, .stencilAttachmentFormat = im->f.vk};
    VkGraphicsPipelineCreateInfo gi = {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rci, .stageCount = 1, .pStages = &st, .pVertexInputState = &vi, .pInputAssemblyState = &ia,
        .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &dss, .pColorBlendState = &cb, .pDynamicState = &ds, .layout = g_smc_lay};
    VkPipeline p = VK_NULL_HANDLE;
    if ((r = vk.vkCreateGraphicsPipelines(V.dev, VK_NULL_HANDLE, 1, &gi, NULL, &p))) { VRED_ONCE("masked stencil clear pipeline (format %d): %d", im->f.vk, r); return VK_NULL_HANDLE; }
    g_smc[g_nsmc++] = (typeof(g_smc[0])){im->f.vk, im->samples, p};
    return p;
}
static void stencil_masked(vka_img *im, uint32_t lv, uint32_t l0, uint32_t nl, VkRect2D rc, uint32_t mask, uint32_t value) {
    VkPipeline p = smc_pipeline(im);
    if (!p || !(im->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) { VRED_ONCE("stencil clear with write mask %#x: no pipeline, stencil left", mask); return; }
    uint32_t w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1;
    for (uint32_t l = l0; l < l0 + nl; l++) {   // a layer at a time: the vertex shader writes no layer
        VkImageView view = vka_view(im, im->f.vk, VK_IMAGE_VIEW_TYPE_2D, im->aspect, lv, 1, l, 1, NULL);
        if (!view) return;
        VkRenderingAttachmentInfo a = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
        VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = 1,
                              .pDepthAttachment = im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? &a : NULL, .pStencilAttachment = &a};
        vk.vkCmdBeginRendering(g_cb, &ri);
        float pc[6] = {0};
        VkViewport v = {0, 0, (float)w, (float)h, 0, 1};
        vk.vkCmdBindPipeline(g_cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
        vk.vkCmdPushConstants(g_cb, g_smc_lay, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof pc, pc);
        vk.vkCmdSetViewport(g_cb, 0, 1, &v); vk.vkCmdSetScissor(g_cb, 0, 1, &rc);
        vk.vkCmdSetStencilWriteMask(g_cb, VK_STENCIL_FACE_FRONT_AND_BACK, mask); vk.vkCmdSetStencilReference(g_cb, VK_STENCIL_FACE_FRONT_AND_BACK, value);
        vk.vkCmdDraw(g_cb, 3, 1, 0, 0);
        vk.vkCmdEndRendering(g_cb);
    }
    vka_dyn_invalidate();   // its pipeline's static state replaced the draws' dynamic state
}
static void clear_ds(const p_clear_ds *c) {
    vka_rp_prep_clear(&S.rt, S.has_rt);
    if (S.rc_skip || !S.has_rt || !S.rt.depth) return;
    vka_rl_site = VKA_RL_CLEAR; vka_img *im = vka_tex_get(S.rt.depth, g_cb); vka_rl_site = 0;
    if (!im || im->f.kind < 3) return;
    VkImageAspectFlags asp = 0; unsigned sm = (unsigned)c->stencil_mask & 0xff;
    if (c->depth_mask && (im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) asp |= VK_IMAGE_ASPECT_DEPTH_BIT;
    int masked = sm && sm != 0xff && (im->aspect & VK_IMAGE_ASPECT_STENCIL_BIT);   // drawn (stencil_masked)
    if (sm == 0xff && (im->aspect & VK_IMAGE_ASPECT_STENCIL_BIT)) asp |= VK_IMAGE_ASPECT_STENCIL_BIT;
    // All-zero bits written (depth +0.0 in any depth format, stencil 0, a masked stencil write of 0 into zeros): a zero-shared image keeps its zeros (DM_VK_ZSHARE_ZCLEAR)
    int zero = (!(asp & VK_IMAGE_ASPECT_DEPTH_BIT) || (c->depth == 0.0f && !signbit(c->depth))) && (!((asp & VK_IMAGE_ASPECT_STENCIL_BIT) || masked) || !((uint32_t)c->stencil & 0xff));
    uint32_t dbits; memcpy(&dbits, &c->depth, 4);
    if ((asp || masked) && zero && vka_tex_zclear(im)) { vka_rtw_clear(im, 1, 0); g_stat_ops++; return; }
    vka_tex_own(im, g_cb);
    if (!asp && !masked) return;
    uint32_t lv, l0, nl; rt_range(im, &S.rt.dview, S.rt.has_dview, &lv, &l0, &nl);
    if (lv >= im->levels || !nl) return;
    VkRect2D rc; int part = scissored(im, lv, &rc);
    if (!rc.extent.width) return;
    VkClearValue val = {.depthStencil = {c->depth, (uint32_t)c->stencil & 0xff}};
    vka_pclear pc = {-1, im, im->f.vk, asp, lv, l0, nl, rc, !part, val, val};
    vka_rtw_clear(im, zero, (asp & VK_IMAGE_ASPECT_DEPTH_BIT) ? dbits : (uint32_t)c->stencil & 0xff);   // (DM_VK_RTWATCH_MB)
    if (!masked && asp && clear_in_pass(&pc)) { vka_img_written(im); g_stat_ops++; return; }
    vka_rp_end();
    if (masked) { stencil_masked(im, lv, l0, nl, rc, sm, (uint32_t)c->stencil & 0xff); vka_img_written(im); g_stat_ops++; vka_barrier(g_cb); if (!asp) return; }
    vka_clear_outside(&pc);
    vka_img_written(im); g_stat_ops++;
}
static void clear_tex(const p_clear_tex *c) {
    if (S.rc_skip || !c->tex) return;
    vka_rl_site = VKA_RL_CLEAR; vka_img *im = vka_tex_get(c->tex, g_cb); vka_rl_site = 0; vka_fmt f;
    if (!im || !vka_fmt_of(c->has_view && c->view.set_format ? c->view.format : c->tex->d.format, &f)) return;
    if (f.kind >= 3) { VRED_ONCE("ClearTexture of a depth / stencil texture (format %#x)", c->tex->d.format); return; }
    if ((c->mask & 0xf) != 0xf) { if (c->mask & 0xf) VRED_ONCE("ClearTexture with channel mask %#x is not on Vulkan yet: skipped", c->mask); return; }
    VkClearColorValue cz; if (c->is_uint) for (int i = 0; i < 4; i++) cz.uint32[i] = c->c.u[i]; else cz = color_value(&f, c->c.f);
    int zero = clear_zero(&cz);
    if (zero && vka_tex_zclear(im)) { vka_rtw_clear(im, 1, 0); g_stat_ops++; return; }   // (a zero-shared image keeps its zeros, DM_VK_ZSHARE_ZCLEAR)
    vka_tex_own(im, g_cb);
    uint32_t lv = c->has_view && c->view.set_levels ? c->view.base_level : 0, l0 = (c->has_view && c->view.set_layers ? c->view.min_layer : 0) + (uint32_t)c->r.zoffset, nl = c->r.depth > 0 ? (uint32_t)c->r.depth : 1;
    if (!vka_img_lv(im, &lv)) return;
    uint32_t all = vka_img_slices(im, lv); int is3d = im->d.target == NVN_TEXTURE_TARGET_3D;
    if (l0 >= all) return; if (l0 + nl > all) nl = all - l0;
    uint32_t w, h; level_size(im, lv, &w, &h);
    VkClearValue val;
    if (c->is_uint) for (int i = 0; i < 4; i++) val.color.uint32[i] = c->c.u[i]; else val.color = color_value(&f, c->c.f);
    if (c->r.xoffset == 0 && c->r.yoffset == 0 && (uint32_t)c->r.width >= w && (uint32_t)c->r.height >= h && f.vk == im->f.vk && (!is3d || (!l0 && nl == all))) {
        VkImageSubresourceRange rg = {VK_IMAGE_ASPECT_COLOR_BIT, lv, 1, is3d ? 0 : l0, is3d ? 1 : nl}; vk.vkCmdClearColorImage(g_cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &val.color, 1, &rg);
    } else clear_rect(im, f.vk, VK_IMAGE_ASPECT_COLOR_BIT, lv, l0, nl, (VkRect2D){{c->r.xoffset, c->r.yoffset}, {(uint32_t)c->r.width, (uint32_t)c->r.height}}, &val);
    vka_img_written(im); vka_rtw_clear(im, zero, cz.uint32[0]); g_stat_ops++;
    vka_barrier(g_cb);
}

// ---------- Copies ----------
static VkImageAspectFlags copy_aspect(const vka_img *im, const char *what) {
    if (im->f.kind == 0 || im->f.kind == 1 || im->f.kind == 2) return VK_IMAGE_ASPECT_COLOR_BIT;
    if (im->f.vk == VK_FORMAT_D16_UNORM || im->f.vk == VK_FORMAT_D32_SFLOAT) return VK_IMAGE_ASPECT_DEPTH_BIT;
    if (im->f.vk == VK_FORMAT_S8_UINT) return VK_IMAGE_ASPECT_STENCIL_BIT;
    VRED_ONCE("%s of a packed depth-stencil texture (format %#x) is not on Vulkan yet: NVN's buffer layout interleaves what Vulkan copies by aspect", what, im->d.format);
    return 0;
}
// Buffer ↔ texture: c->src is the buffer's guest address. Rows are SetCopyRowStride bytes apart (tightly packed without it), layers / slices SetCopyImageStride
static void copy_bt(const p_copy_b2t *c, int to_tex) {
    if (!c->tex) return;
    vka_rl_site = VKA_RL_COPY; vka_img *im = vka_tex_get(c->tex, g_cb); vka_rl_site = 0;
    if (!im) return;
    if (to_tex) vka_tex_own(im, g_cb); else vka_tex_whole(im, g_cb);   // (a texture mip drop left without its finer levels: whole again, vka_tex.c "Mip drop")
    VkImageAspectFlags asp = copy_aspect(im, to_tex ? "buffer → texture copy" : "texture → buffer copy"); if (!asp) return;
    if (im->samples > 1) { VRED_ONCE("buffer copy of a multisampled texture"); return; }
    const vka_fmt *f = &im->f;
    unsigned lv = c->has_view && c->view.set_levels ? c->view.base_level : 0, l0 = c->has_view && c->view.set_layers ? c->view.min_layer : 0;
    if (!vka_img_lv(im, &lv) || c->r.width <= 0 || c->r.height <= 0) return;
    size_t row = S.copy_stride ? (size_t)S.copy_stride : (size_t)((c->r.width + f->bw - 1) / f->bw) * f->bpb, rows = (size_t)(c->r.height + f->bh - 1) / f->bh;
    size_t img = S.copy_img_stride ? (size_t)S.copy_img_stride : row * rows;
    int is3d = im->depth > 1 || c->tex->d.target == NVN_TEXTURE_TARGET_3D, nz = c->r.depth > 0 ? c->r.depth : 1;
    if (row % f->bpb) { VRED_ONCE("buffer copy row stride %zu is not a whole number of %u-byte blocks", row, f->bpb); return; }
    uint64_t span = img * (size_t)(nz - 1) + row * (rows - 1) + (size_t)((c->r.width + f->bw - 1) / f->bw) * f->bpb;
    VkBuffer b; VkDeviceSize o;
    if (!vka_addr(c->src, span, &b, &o, NULL)) { static _Atomic int nr; if (atomic_fetch_add(&nr, 1) < 8) VRED("%s at %#llx (%llu bytes) is not in placed guest RAM", to_tex ? "buffer → texture copy source" : "texture → buffer copy destination", (unsigned long long)c->src, (unsigned long long)span); return; }
    vka_span s = {0}; VkDeviceSize al = asp == VK_IMAGE_ASPECT_COLOR_BIT ? (f->bpb % 4 == 0 ? f->bpb : f->bpb % 2 == 0 ? 2 * f->bpb : 4 * f->bpb) : 4;   // a whole block and a multiple of 4
    int bounce = o % al != 0 || (img % row) != 0;   // Vulkan wants block-aligned offsets and layers a whole number of rows apart
    if (bounce) {
        size_t wb = (size_t)((c->r.width + f->bw - 1) / f->bw) * f->bpb;
        if (!vka_scratch(wb * rows * (size_t)nz, al * 16, &s)) return;
        if (to_tex) for (int z = 0; z < nz; z++) for (size_t y = 0; y < rows; y++) { VkBufferCopy bc = {o + z * img + y * row, s.off + (z * rows + y) * wb, wb}; vk.vkCmdCopyBuffer(g_cb, b, s.buf, 1, &bc); }
        if (to_tex) vka_barrier(g_cb);
        row = wb; img = wb * rows;
    }
    VkBufferImageCopy rg[64]; int n = 0;
    for (int z = 0; z < (is3d ? 1 : nz) && n < 64; z++)
        rg[n++] = (VkBufferImageCopy){(bounce ? s.off : o) + (is3d ? 0 : z * img), (uint32_t)(row / f->bpb * f->bw), (uint32_t)(img / row * f->bh),
            {asp, lv, is3d ? 0 : l0 + (uint32_t)c->r.zoffset + (uint32_t)z, 1}, {c->r.xoffset, c->r.yoffset, is3d ? c->r.zoffset : 0}, {(uint32_t)c->r.width, (uint32_t)c->r.height, is3d ? (uint32_t)nz : 1}};
    if (to_tex) { vk.vkCmdCopyBufferToImage(g_cb, bounce ? s.buf : b, im->img, VK_IMAGE_LAYOUT_GENERAL, (uint32_t)n, rg); vka_img_written(im); vka_rtw(im, VKA_RTW_COPY); }
    else {
        vka_tex_gw_opaque(c->src, span);
        vk.vkCmdCopyImageToBuffer(g_cb, im->img, VK_IMAGE_LAYOUT_GENERAL, bounce ? s.buf : b, (uint32_t)n, rg);
        if (bounce) {
            vka_barrier(g_cb);
            size_t wb = row, srow = S.copy_stride ? (size_t)S.copy_stride : wb, simg = S.copy_img_stride ? (size_t)S.copy_img_stride : srow * rows;
            for (int z = 0; z < nz; z++) for (size_t y = 0; y < rows; y++) { VkBufferCopy bc = {s.off + (z * rows + y) * wb, o + z * simg + y * srow, wb}; vk.vkCmdCopyBuffer(g_cb, s.buf, b, 1, &bc); }
        }
    }
    g_stat_ops++;
    vka_barrier(g_cb);
}
static void copy_t2t(const p_copy_t2t *c) {
    if (!c->src || !c->dst) return;
    vka_rl_site = VKA_RL_COPY; vka_img *s = vka_tex_get(c->src, g_cb), *d = vka_tex_get(c->dst, g_cb); vka_rl_site = 0;
    if (!s || !d) return;
    vka_tex_own(d, g_cb); vka_tex_whole(s, g_cb);   // (a source mip drop left without its finer levels: whole again)
    unsigned sl = c->has_sv && c->sv.set_levels ? c->sv.base_level : 0, dl = c->has_dv && c->dv.set_levels ? c->dv.base_level : 0;
    unsigned sz0 = c->has_sv && c->sv.set_layers ? c->sv.min_layer : 0, dz0 = c->has_dv && c->dv.set_layers ? c->dv.min_layer : 0;
    if (!vka_img_lv(s, &sl) || !vka_img_lv(d, &dl)) return;
    VkImageAspectFlags sa = s->aspect & (s->f.kind >= 3 ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT), da = d->aspect;
    if ((sa == VK_IMAGE_ASPECT_COLOR_BIT) != (da == VK_IMAGE_ASPECT_COLOR_BIT)) { VRED_ONCE("texture copy between a color and a depth texture (formats %#x → %#x)", c->src->d.format, c->dst->d.format); return; }
    sa &= da;
    int s3 = s->depth > 1, d3 = d->depth > 1, nz = c->sr.depth > 0 ? c->sr.depth : 1;
    VkImageSubresourceLayers ss = {sa, sl, s3 ? 0 : sz0 + (uint32_t)c->sr.zoffset, s3 ? 1 : (uint32_t)nz}, ds = {sa, dl, d3 ? 0 : dz0 + (uint32_t)c->dr.zoffset, d3 ? 1 : (uint32_t)nz};
    if (ss.baseArrayLayer + ss.layerCount > s->layers || ds.baseArrayLayer + ds.layerCount > d->layers) { VRED_ONCE("texture copy layers out of range"); return; }
    if (c->flags & DM_COPY_RESOLVE && s->samples > 1 && d->samples == 1) {
        VkImageResolve r = {ss, {c->sr.xoffset, c->sr.yoffset, 0}, ds, {c->dr.xoffset, c->dr.yoffset, 0}, {(uint32_t)c->sr.width, (uint32_t)c->sr.height, 1}};
        if (sa == VK_IMAGE_ASPECT_COLOR_BIT) vk.vkCmdResolveImage(g_cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
        else VRED_ONCE("depth resolve is not on Vulkan yet");
    } else if (c->sr.width == c->dr.width && c->sr.height == c->dr.height && !(c->flags & (NVN_COPY_FLAGS_MIRROR_X_BIT | NVN_COPY_FLAGS_MIRROR_Y_BIT | NVN_COPY_FLAGS_MIRROR_Z_BIT))
               && s->samples == d->samples && s->f.bpb == d->f.bpb && s->f.bw == d->f.bw && s->f.bh == d->f.bh) {
        VkImageCopy r = {ss, {c->sr.xoffset, c->sr.yoffset, s3 ? c->sr.zoffset : 0}, ds, {c->dr.xoffset, c->dr.yoffset, d3 ? c->dr.zoffset : 0}, {(uint32_t)c->sr.width, (uint32_t)c->sr.height, s3 ? (uint32_t)nz : 1}};
        vk.vkCmdCopyImage(g_cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    } else {   // a scaled, mirrored or format-changing copy: a blit (filtered with LINEAR_FILTER)
        if (s->f.kind != d->f.kind || (s->f.kind && s->f.kind < 3) || s->f.bw > 1 || d->f.bw > 1 || s->samples > 1 || d->samples > 1) {
            VRED_ONCE("texture copy %#x → %#x needs a draw (integer, compressed, multisampled or depth with scaling)", c->src->d.format, c->dst->d.format); return; }
        int x0 = c->sr.xoffset, x1 = x0 + c->sr.width, y0 = c->sr.yoffset, y1 = y0 + c->sr.height, t;
        if (c->flags & NVN_COPY_FLAGS_MIRROR_X_BIT) { t = x0; x0 = x1; x1 = t; }
        if (c->flags & NVN_COPY_FLAGS_MIRROR_Y_BIT) { t = y0; y0 = y1; y1 = t; }
        VkImageBlit r = {ss, {{x0, y0, 0}, {x1, y1, 1}}, ds, {{c->dr.xoffset, c->dr.yoffset, 0}, {c->dr.xoffset + c->dr.width, c->dr.yoffset + c->dr.height, 1}}};
        VkFormatProperties fp; vk.vkGetPhysicalDeviceFormatProperties(V.pd, s->f.vk, &fp);
        VkFilter fl = (c->flags & NVN_COPY_FLAGS_LINEAR_FILTER_BIT) && sa == VK_IMAGE_ASPECT_COLOR_BIT && (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        vk.vkCmdBlitImage(g_cb, s->img, VK_IMAGE_LAYOUT_GENERAL, d->img, VK_IMAGE_LAYOUT_GENERAL, 1, &r, fl);
    }
    vka_img_written(d); vka_rtw(d, VKA_RTW_COPY); g_stat_ops++;
    vka_barrier(g_cb);
}

static void not_yet(uint32_t op) {
    g_stat_skipped++;
    VRED_ONCE("DrawTexture (op %u) is not on Vulkan yet: skipped from now on", op);
}
int vka_mem_in_pool(uint64_t a, uint64_t n);
static int vka_mem_in_pool_(uint64_t a, uint64_t n) { return vka_addr(a, n, NULL, NULL, NULL) || vka_mem_in_pool(a, n); }
static int rc_known(uint64_t a, uint64_t *v) {   // the counter at a: a pending report's (rc_pend_get), else what memory holds now (0 = unknown)
    if (rc_pend_get(a, v)) return 1;   // a report recorded earlier and not written yet: the counter it will hold
    // A counter in a pool not placed yet is still in the guest's memory, where the completion thread wrote it (post): read there too. Only placed memory was read
    // before, so at a scene's start, while most of its pools were not placed yet, every condition over them read as unknown and its draws were drawn: Ocarina of
    // Time's first cutscene drew the draws its occlusion results ruled out for its first ~10 s (large holes of sky over the ground, changing each frame)
    { static int on = -1; if (on < 0) { const char *e_ = getenv("DM_TMP_RC_POOL"); on = !e_ || atoi(e_); }
      int vka_mem_in_pool(uint64_t a, uint64_t n);
      if (!vka_addr(a, 8, NULL, NULL, NULL) && !(on && vka_mem_in_pool(a, 8))) return 0; }
    memcpy(v, (const void *)(uintptr_t)a, 8); return 1;
}

static void seg_write(const void *a) { struct { VkCommandBuffer cb; VkPipelineStageFlagBits st; uint32_t q; } w; memcpy(&w, a, sizeof w); qf.write(w.cb, w.st, g_sqp, w.q); }
static void exec(const uint8_t *p, int depth) {
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        const void *a = p + 8;
        int bar_kept = 0;   // a game barrier taken inside the open pass (vka_rp_game_barrier): no pass end, no full barrier
        uint64_t ra_ = RP_T();   // (DM_VK_RPROF: part ends, then the pass end / state step, then the command itself)
        if (h.op == OP_DRAW || h.op == OP_DRAW_IDX) { chunk_check(); if (vka_rprof_on) { uint64_t t_ = vka_rprof_t(); g_rp.k[RK_PARTS] += t_ - ra_; ra_ = t_; } }   // before a draw that opens a new pass
        switch (h.op) {   // a render pass a draw opened stays open across draws and state; anything else closes it first
        case OP_END: case OP_BEGIN: case OP_JUMP: case OP_CALL: case OP_SIGNAL_EVENT: case OP_REPORT_COUNTER: case OP_FENCE_SYNC: case OP_RENDER_COND: case OP_RESET_COUNTER:
        case OP_SCISSOR: case OP_COPY_STRIDE: case OP_DRAW: case OP_DRAW_IDX: break;
        case OP_SET_RT: if (g_cb && !vka_pass_keep()) vka_rp_end(); break;
        case OP_BARRIER: if (g_cb) { int32_t bits; memcpy(&bits, a, 4); bar_kept = vka_rp_game_barrier((uint32_t)bits); if (!bar_kept) vka_rp_end(); } break;
        case OP_CLEAR_COLOR: case OP_CLEAR_DS: break;   // the pass is ended by the next draw over other targets, or the clear (vka_rp_clear)
        default: if (g_cb && !vka_state_op(&S.d, h.op, a)) vka_rp_end(); else if (!g_cb) vka_state_op(&S.d, h.op, a); break;
        }
        if (vka_rprof_on) { uint64_t t_ = vka_rprof_t(); g_rp.k[RK_STATE] += t_ - ra_; ra_ = t_; }
        switch (h.op) {
        case OP_END: return;
        case OP_BEGIN: break;
        case OP_JUMP: memcpy(&p, a, 8); continue;
        case OP_CALL: { uint64_t t; memcpy(&t, a, 8); if (depth < 16) exec((const uint8_t *)(uintptr_t)t, depth + 1); else VRED_ONCE("command lists nested deeper than 16"); break; }
        case OP_SIGNAL_EVENT: { uint64_t val; uint32_t m, v; memcpy(&val, a, 8); memcpy(&m, (const uint8_t *)a + 8, 4); memcpy(&v, (const uint8_t *)a + 12, 4); post((pop){1, m, v, 0, val}); break; }
        case OP_REPORT_COUNTER: { uint64_t t, ad; memcpy(&t, a, 8); memcpy(&ad, (const uint8_t *)a + 8, 8);
            { static uint64_t tc[32], tn; tc[t & 31]++; if (!(++tn & 4095) && getenv("DM_TMP_RC")) { char b_[300]; int o_ = 0; for (int i_ = 0; i_ < 32; i_++) if (tc[i_]) o_ += snprintf(b_ + o_, sizeof b_ - o_, " type%d:%llu", i_, (unsigned long long)tc[i_]); VLOG("TEMP reports:%s", b_); } }   // TEMP (S6)
            uint32_t q = 0;
            if (g_sqp && g_cb && g_sq_n < SEGQ && (t == NVN_COUNTER_TYPE_TIMESTAMP || t == NVN_COUNTER_TYPE_TIMESTAMP_TOP)) {
                q = (uint32_t)(g_sq_base + g_sq_n++) + 1;
                struct { VkCommandBuffer cb; VkPipelineStageFlagBits st; uint32_t q; } w = {g_cb, t == NVN_COUNTER_TYPE_TIMESTAMP_TOP ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, q - 1};
                if (!vka_enc_call(seg_write, &w, sizeof w)) seg_write(&w);
            }
            if (DM_GAME_RENDER_COND) rc_pend_add(ad, rc_counter(t));   // before post: post_run takes it down once written
            post((pop){2, 0, q, (uint32_t)t, ad});
            if (atomic_load_explicit(&g_synclog, memory_order_relaxed) > 0) VLOG("sync report: frame %llu type %llu at draw %llu of the submission, part %d, recording at %.2f ms", (unsigned long long)g_frame,
                                                                           (unsigned long long)t, (unsigned long long)(g_stat_draws - g_sub_draws0), g_nx + 1, ms_now());
            break; }
        case OP_FENCE_SYNC: { sync_t *s; memcpy(&s, a, 8); s->fence = g_cur_fence ? g_cur_fence : vka_fence_last() + 1; s->fenced = 1; break; }   // the fence this submission commits with
        case OP_RENDER_COND: { const p_render_cond *rc = a;
            if (rc->mode >= 2) S.rc_skip = rc->mode == 3;
            else { uint64_t x, y; int known = rc_known(rc->addr, &x) && rc_known(rc->addr + 16, &y); S.rc_skip = known && (rc->mode ? x == y : x != y); }
            { static int tr = -1; if (tr < 0) { const char *e_ = getenv("DM_TMP_RC"); tr = e_ ? atoi(e_) : 0; }   // TEMP (S6): 1 = never skip, 2 = log
              static uint64_t nset, nskip; nset++; if (S.rc_skip) nskip++;
              if (tr == 2 && (nset < 40 || !(nset & 1023))) VLOG("TEMP rc: mode %d addr %#llx -> skip %d (%llu of %llu skipped)", rc->mode, (unsigned long long)rc->addr, S.rc_skip, (unsigned long long)nskip, (unsigned long long)nset);
              if (tr == 3 && rc->mode < 2) { static uint64_t nl3; uint64_t x3 = 0, y3 = 0; int k3 = rc_known(rc->addr, &x3), k4 = rc_known(rc->addr + 16, &y3); uint64_t z3[4] = {0}; if (vka_mem_in_pool_(rc->addr, 32)) memcpy(z3, (const void *)(uintptr_t)rc->addr, 32);
                if (nl3++ < 3000) VLOG("TEMP rc3 f%llu m%d a%#llx k%d%d x%#llx y%#llx raw %#llx %#llx %#llx %#llx skip %d", (unsigned long long)g_frame, rc->mode, (unsigned long long)rc->addr, k3, k4, (unsigned long long)x3, (unsigned long long)y3, (unsigned long long)z3[0], (unsigned long long)z3[1], (unsigned long long)z3[2], (unsigned long long)z3[3], S.rc_skip); }
              if (tr == 1 || (tr == 4 && rc->mode < 2)) S.rc_skip = 0;
              if (tr == 5 && rc->mode < 2) S.rc_skip = 1; }
            break; }
        case OP_RESET_COUNTER: break;
        case OP_CLEAR_BUF: if (g_cb) { const p_clear_buf *cb = a; buf_fill(cb->addr, cb->size, cb->value); vka_barrier(g_cb); } break;
        case OP_COPY_B2B: if (g_cb) { uint64_t s_, d, n; memcpy(&s_, a, 8); memcpy(&d, (const uint8_t *)a + 8, 8); memcpy(&n, (const uint8_t *)a + 16, 8); buf_copy(s_, d, n); vka_barrier(g_cb); } break;
        case OP_UPDATE_UB: g_n_ub++; if (g_cb) { uint64_t d, n; memcpy(&d, a, 8); memcpy(&n, (const uint8_t *)a + 8, 8); buf_update((const uint8_t *)a + 16, d, n); vka_barrier(g_cb); } break;
        case OP_SET_RT: S.rt = *(const p_set_rt *)a; S.has_rt = 1; break;
        case OP_SCISSOR: S.sc = *(const p_scissor *)a; S.has_sc = 1; break;
        case OP_COPY_STRIDE: { const p_copy_stride *cs = a; if (cs->image) S.copy_img_stride = cs->stride; else S.copy_stride = cs->stride; break; }
        case OP_CLEAR_COLOR: if (g_cb) clear_color(a); break;
        case OP_CLEAR_DS: if (g_cb) clear_ds(a); break;
        case OP_CLEAR_TEX: if (g_cb) clear_tex(a); break;
        case OP_COPY_B2T: if (g_cb) copy_bt(a, 1); break;
        case OP_COPY_T2B: if (g_cb) copy_bt(a, 0); break;
        case OP_COPY_T2T: if (g_cb) copy_t2t(a); break;
        case OP_BARRIER: if (g_cb && !bar_kept) vka_barrier(g_cb); break;
        case OP_DRAW: if (!S.rc_skip && g_cb) { p_draw t; g_stat_draws++; vka_draw(&S.d, &S.rt, S.has_rt, &S.sc, S.has_sc, draw_rec(a, h.size, &t), NULL); } break;
        case OP_DRAW_IDX: if (!S.rc_skip && g_cb) { p_draw_idx t; g_stat_draws++; vka_draw(&S.d, &S.rt, S.has_rt, &S.sc, S.has_sc, NULL, draw_idx_rec(a, h.size, &t)); } break;
        case OP_DISPATCH: if (g_cb) vka_dispatch(&S.d, a); break;
        case OP_DRAW_TEX: not_yet(h.op); break;
        default: break;   // state: taken by vka_state_op above
        }
        if (vka_rprof_on && h.op != OP_CALL) g_rp.k[rp_kind(h.op)] += vka_rprof_t() - ra_;   // (a call's records are counted inside it)
        if (h.size < 8) { VRED("command record of size %u at %p (op %u): the list is cut here", h.size, (const void *)p, h.op); return; }
        p += h.size;
    }
}
void vka_rp_exec(const uint8_t *handle, uint64_t fence) { if (fence) g_cur_fence = fence; if (handle) exec(handle, 0); }
#endif
