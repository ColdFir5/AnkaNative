// GPU time by step (DM_VK_GPUPROF=n, live): one frame in every n (n >= 1) gets a GPU timestamp after each full barrier the replay records (vka_barrier) and at the start of
// each submission and of each part of one (vka_exec.c chunk). Every full barrier waits for all earlier work, so the time between two timestamps is the GPU time of what was
// recorded between them: one render pass, or the clears / copies / uploads of one op. Once the frame's last submission is done the log gets one "gpu prof" block:
// the frame's GPU time by kind (passes, transfers, barriers with nothing between them) and the steps that cost most, each with the recording site of the barrier that
// ended it (file:line), the draws in it and the pass's attachments. Gaps between submissions and parts (the GPU waiting for recording) are left out.
// It costs a stall per profiled frame (timestamps), nothing when off (one cached switch read per submission)
// Split of a pass (DM_VK_GPUPROF_SPLIT=1, live): the profiled frames alternate between a plain one (as above) and a split one. In a split frame every draw pass of the
// chosen size (DM_VK_GPUPROF_SPLIT_W / _H, 0 = any) gets a timestamp at its start, at its first draw, at every pipeline change and (DM_VK_GPUPROF_SPLIT_K=k) every k draws
// of one pipeline, and at its end, so each group of draws between two timestamps has one pipeline. On a tiler a timestamp inside a pass that runs in GMEM is written once
// per bin (Turnip: the last bin's stays), which splits nothing; so such a pass is made to run in sysmem first (DM_VK_GPUPROF_SPLIT_SYSMEM, default 1): Turnip renders a pass
// in sysmem when it holds a barrier with a stage outside the framebuffer-space ones (tu_cmd_buffer.cc tu_barrier: "Non-framebuffer-space barrier"), so one empty
// vertex-to-vertex memory barrier is recorded right after the pass begins. (Outside the Vulkan rules for a dynamic-rendering pass without local read; the profiled frame
// only, and only with the switch on.) The report ("gpu prof split:" lines, once the split frame is done) gives for the largest passes (DM_VK_GPUPROF_SPLIT_MIN draws, 32)
// their time between first and last timestamp, their draws and indices, the render mode asked, the same size pass's time in the last plain profiled frame (GMEM or sysmem
// as the driver chose: the ratio says what sysmem costs it), and the time by pipeline (vertex and fragment shader keys): total ms, share, draws, indices, groups. Normal
// frames and plain profiled frames are not changed. Bins are not visible to the replay: TU_DEBUG=perf (or TU_DEBUG_FILE) has Turnip log why it chose sysmem.
// The ratio compares like with like: every split pass of the reported pass's size (the small ones too) against every pass of that size in the plain frame, sums and counts
// on both sides (before 2026-11-08 the one reported pass was divided by all of the plain frame's passes of its size: 1.05 where the sums give 1.20).
// Pipeline statistics (DM_VK_GPUPROF_SPLIT_STATS=1, live, with the device's pipelineStatisticsQuery; Turnip has it): each group of a split pass also gets one
// VK_QUERY_TYPE_PIPELINE_STATISTICS query (vertex shader invocations, clipping invocations and primitives, fragment shader invocations), begun at the group's first draw
// and ended at the next group's or the pass's end, so the report gives per pipeline the fragment invocations, their count over the pass's pixels (w x h: shaded layers per
// pixel, overdraw and helper lanes as the hardware counts them, Turnip's RBBM PS invocations counter) and the time per invocation. No occlusion query: with one running,
// Turnip turns a discarding shader's EARLY_Z into EARLY_Z_LATE_Z (tu_cmd_buffer.cc), which would change what is measured. A stats query in a pass turns Turnip's
// concurrent binning off for it (a split pass runs in sysmem anyway). The GPU clock (KGSL sysfs, vka_dev.c vka_gpu_clock) is read when the profiled frame is armed and
// when its report is made, and given on the report's first line
#if defined(__ANDROID__)
#include "vka.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { NQ = 2048, DESC = 96 };
typedef struct { const char *file; int line; int kind; uint64_t draws, passes, ops; char desc[DESC]; } gp_mark;   // kind: 0 barrier, 1 submission start, 2 part start, 3 commit
static gp_mark g_m[NQ]; static int g_n;
static VkQueryPool g_qp; static int g_ok = -1; static double g_tick_ns;
static PFN_vkCmdWriteTimestamp g_write; static PFN_vkCmdResetQueryPool g_reset; static PFN_vkGetQueryPoolResults g_get;
static int g_state;                     // 0 idle, 1 recording the profiled frame, 2 waiting for its last fence
static uint64_t g_frame, g_last_frame, g_fence; static VkCommandBuffer g_cb;
static char g_pass[DESC]; static int g_pass_new;   // the last pass opened (vka_gp_pass), taken by the next mark
static uint64_t g_sub_frame; static int g_sub_frame_set;
int vka_gp_active;                      // marks are taken (read by vka_barrier)
// The split (above): groups of draws of one pipeline, each starting at its query in g_qps and ending at the next group's (or its pass's end)
enum { NQS = 4096, NSP = 64, NPIPE = 256 };
typedef struct { uint64_t vs, fs, idx; uint32_t q, draws, ind; } sp_grp;   // (group g's statistics query, when its pass has them, is query g of g_qst)
typedef struct { uint32_t w, h, q0, q1, g0, g1, draws, ind; uint64_t idx; int sysmem, cut, stats; } sp_pass;   // q1 0: not ended in the frame; stats: its groups have statistics queries
static sp_grp g_sgp[NQS]; static sp_pass g_spp[NSP];
static int g_nsq, g_nsgp, g_nspp, g_split_frame, g_split_cut, g_split_toggle, g_split_k;
static VkQueryPool g_qps; static int g_qps_ok = -1;
static VkQueryPool g_qst; static int g_qst_ok = -1, g_st_open, g_st_reset;   // DM_VK_GPUPROF_SPLIT_STATS: one pipeline statistics query a group; g_st_open: one is running
static PFN_vkCmdBeginQuery g_qbegin; static PFN_vkCmdEndQuery g_qend;
enum { NST = 4 };   // the statistics asked, in Vulkan's bit order (the order of the results)
#define ST_BITS (VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT | VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT | VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT | VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT)
static char g_clk0[200]; static int g_clk0_ok;   // the GPU clock when the profiled frame was armed (ok: read; not: the probe's answer, vka_dev.c vka_gpu_clock)
static VkPipeline g_sp_pipe; static uint32_t g_sp_since;
int vka_gp_split;                       // a split pass is open: vka_draw calls vka_gp_draw before each draw
typedef struct { uint32_t w, h, draws, n; double ms; } plain_pass;   // n: passes (steps with a pass) of that size
static plain_pass g_plain[32]; static int g_nplain; static uint64_t g_plain_frame;   // the last plain profiled frame's passes (by size)

void vka_gp_sub_frame(uint64_t f) { g_sub_frame = f; g_sub_frame_set = 1; }   // the replay worker: the game frame of the submission it replays next
static int init(void) {
    if (g_ok >= 0) return g_ok;
    g_ok = 0;
    g_write = (PFN_vkCmdWriteTimestamp)vk.vkGetDeviceProcAddr(V.dev, "vkCmdWriteTimestamp"); g_reset = (PFN_vkCmdResetQueryPool)vk.vkGetDeviceProcAddr(V.dev, "vkCmdResetQueryPool");
    g_get = (PFN_vkGetQueryPoolResults)vk.vkGetDeviceProcAddr(V.dev, "vkGetQueryPoolResults");
    PFN_vkCreateQueryPool create = (PFN_vkCreateQueryPool)vk.vkGetDeviceProcAddr(V.dev, "vkCreateQueryPool");
    VkQueryPoolCreateInfo ci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = NQ};
    if (!g_write || !g_reset || !g_get || !create || create(V.dev, &ci, NULL, &g_qp) || V.props.limits.timestampPeriod <= 0) { VRED("gpu prof: no timestamp query pool"); return 0; }
    g_tick_ns = V.props.limits.timestampPeriod; g_ok = 1;
    return 1;
}
static int split_init(void) {
    if (g_qps_ok >= 0) return g_qps_ok;
    PFN_vkCreateQueryPool create = (PFN_vkCreateQueryPool)vk.vkGetDeviceProcAddr(V.dev, "vkCreateQueryPool");
    VkQueryPoolCreateInfo ci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = NQS};
    g_qps_ok = create && !create(V.dev, &ci, NULL, &g_qps);
    if (!g_qps_ok) VRED("gpu prof split: no timestamp query pool for the split: profiled frames stay plain");
    return g_qps_ok;
}
static int stats_init(void) {   // DM_VK_GPUPROF_SPLIT_STATS: 1 = the statistics pool is there
    if (g_qst_ok >= 0) return g_qst_ok;
    g_qst_ok = 0;
    if (!V.pstats) { VLOG("gpu prof split: DM_VK_GPUPROF_SPLIT_STATS: the device has no pipelineStatisticsQuery: no statistics"); return 0; }
    PFN_vkCreateQueryPool create = (PFN_vkCreateQueryPool)vk.vkGetDeviceProcAddr(V.dev, "vkCreateQueryPool");
    g_qbegin = (PFN_vkCmdBeginQuery)vk.vkGetDeviceProcAddr(V.dev, "vkCmdBeginQuery"); g_qend = (PFN_vkCmdEndQuery)vk.vkGetDeviceProcAddr(V.dev, "vkCmdEndQuery");
    VkQueryPoolCreateInfo ci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS, .queryCount = NQS, .pipelineStatistics = ST_BITS};
    g_qst_ok = create && g_qbegin && g_qend && !create(V.dev, &ci, NULL, &g_qst);
    if (!g_qst_ok) VRED("gpu prof split: no pipeline statistics query pool: split frames have no statistics");
    return g_qst_ok;
}
static void st_end(VkCommandBuffer cb) { if (g_st_open) { g_qend(cb, g_qst, (uint32_t)g_st_open - 1); g_st_open = 0; } }   // (g_st_open: the running query + 1)
static void mark(VkCommandBuffer cb, int kind, const char *file, int line) {
    if (g_n >= NQ) return;
    gp_mark *m = &g_m[g_n]; m->file = file; m->line = line; m->kind = kind;
    uint64_t sk, di, pi; vka_draw_stats(&m->draws, &sk, &di, &pi); (void)sk; (void)di; (void)pi;
    extern uint64_t vka_n_rp; m->passes = vka_n_rp; uint64_t subs, rec; vka_exec_counts(&m->ops, &subs, &rec);
    if (g_pass_new) { memcpy(m->desc, g_pass, DESC); g_pass_new = 0; } else m->desc[0] = 0;
    vka_enc_sync();
    g_write(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_qp, (uint32_t)g_n);
    g_n++;
}
void vka_gp_mark(VkCommandBuffer cb, const char *file, int line) { if (cb == g_cb) mark(cb, 0, file, line); }
void vka_gp_pass(const char *desc) { if (vka_gp_active) { snprintf(g_pass, DESC, "%s", desc); g_pass_new = 1; } }
void vka_gp_part(VkCommandBuffer cb) { if (!vka_gp_active) return; g_cb = cb; mark(cb, 2, "part", 0); }
// A draw pass began (vka_draw.c rp_begin, with the profile on): in a split frame, one of the chosen size is split (and made to run in sysmem)
void vka_gp_rp_begin(VkCommandBuffer cb, uint32_t w, uint32_t h) {
    if (!g_split_frame || cb != g_cb || vka_gp_split) return;
    uint32_t fw = (uint32_t)VKA_ENV("DM_VK_GPUPROF_SPLIT_W", 0), fh = (uint32_t)VKA_ENV("DM_VK_GPUPROF_SPLIT_H", 0);
    if ((fw && w != fw) || (fh && h != fh)) return;
    if (g_nspp == NSP || g_nsq + 3 > NQS || g_nsgp == NQS) { g_split_cut++; return; }
    sp_pass *p = &g_spp[g_nspp++]; *p = (sp_pass){w, h, 0, 0, (uint32_t)g_nsgp, (uint32_t)g_nsgp, 0, 0, 0, 0, 0};
    if (VKA_ENV("DM_VK_GPUPROF_SPLIT_SYSMEM", 1)) {   // one barrier with a vertex stage inside the pass: Turnip renders the pass in sysmem (see the top)
        VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = 0, .dstAccessMask = 0};
        vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        p->sysmem = 1;
    }
    p->q0 = (uint32_t)g_nsq; g_write(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_qps, (uint32_t)g_nsq++);
    p->stats = g_st_reset && VKA_ENV("DM_VK_GPUPROF_SPLIT_STATS", 0) > 0;   // (the pool was reset at this frame's start: a switch turned on meanwhile waits for the next split frame)
    g_st_open = 0;
    g_sp_pipe = VK_NULL_HANDLE; g_sp_since = 0; g_split_k = VKA_ENV("DM_VK_GPUPROF_SPLIT_K", 0); vka_gp_split = 1;
}
// Before a draw of a split pass (vka_draw.c): a new group at its first draw, at a pipeline change, and every DM_VK_GPUPROF_SPLIT_K draws of one pipeline
void vka_gp_draw(VkCommandBuffer cb, VkPipeline pipe, uint64_t vs, uint64_t fs, uint64_t idx, int ind) {
    if (!vka_gp_split || cb != g_cb) return;
    sp_pass *p = &g_spp[g_nspp - 1];
    int nb = p->g1 == p->g0 || pipe != g_sp_pipe || (g_split_k > 0 && g_sp_since >= (uint32_t)g_split_k);
    if (nb && (g_nsq + 2 > NQS || g_nsgp == NQS)) { nb = 0; p->cut = 1; if (p->g1 == p->g0) return; }   // (full: the draw joins the group before; one query kept for the end)
    if (nb) { g_sgp[g_nsgp] = (sp_grp){vs, fs, 0, (uint32_t)g_nsq, 0, 0}; g_write(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_qps, (uint32_t)g_nsq++);
              if (p->stats) { st_end(cb); g_qbegin(cb, g_qst, (uint32_t)g_nsgp, 0); g_st_open = g_nsgp + 1; }   // the group's statistics (one query a group, the pool as large as the groups' table)
              g_nsgp++; p->g1 = (uint32_t)g_nsgp; g_sp_since = 0; g_sp_pipe = pipe; }
    sp_grp *g = &g_sgp[g_nsgp - 1]; g->draws++; g->idx += idx; g->ind += ind != 0; g_sp_since++;
    p->draws++; p->idx += idx; p->ind += ind != 0;
}
// A draw pass ends (vka_draw.c vka_rp_end, before vkCmdEndRendering)
void vka_gp_rp_end(VkCommandBuffer cb) {
    if (!vka_gp_split) return;
    vka_gp_split = 0;
    if (cb != g_cb) return;
    st_end(cb);   // (a query begun in the pass ends in it)
    g_spp[g_nspp - 1].q1 = (uint32_t)g_nsq; g_write(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_qps, (uint32_t)g_nsq++);   // (a query was kept for it)
}

typedef struct { int i; double ms; } step;
static int by_ms(const void *a, const void *b) { double d = ((const step *)b)->ms - ((const step *)a)->ms; return d > 0 ? 1 : d < 0 ? -1 : 0; }
static void report(void) {
    static uint64_t t[NQ];
    if (g_get(V.dev, g_qp, 0, (uint32_t)g_n, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT)) { VLOG("gpu prof: frame %llu: results not ready", (unsigned long long)g_frame); return; }
    static step s[NQ]; int ns = 0; double tot = 0, pass = 0, xfer = 0, bare = 0, first = 0; int npass = 0;
    for (int i = 1; i < g_n; i++) {
        if (g_m[i].kind == 1 || g_m[i].kind == 2) continue;   // the gap before a submission or part: the GPU waited for recording
        double ms = t[i] > t[i - 1] ? (t[i] - t[i - 1]) * g_tick_ns / 1e6 : 0;
        tot += ms; s[ns++] = (step){i, ms};
        if (g_m[i].passes > g_m[i - 1].passes) { pass += ms; npass++; } else if (g_m[i].ops > g_m[i - 1].ops || g_m[i].draws > g_m[i - 1].draws) xfer += ms; else bare += ms;
        if (g_m[i - 1].kind == 1) first += ms;
    }
    char clk[200], clks[420]; int vka_gpu_clock(char *out, size_t n); int ok = vka_gpu_clock(clk, sizeof clk);
    if (ok || g_clk0_ok) snprintf(clks, sizeof clks, "%s at the report, %s when armed", clk, g_clk0); else snprintf(clks, sizeof clks, "%s", clk);
    VLOG("gpu prof: frame %llu: %d timestamps, %.2f ms GPU: passes %.2f ms (%d), clears / copies / dispatches %.2f ms, steps with nothing recorded (barriers, uploads, writebacks) %.2f ms; "
         "%s", (unsigned long long)g_frame, g_n, tot, pass, npass, xfer, bare, clks);
    if (VKA_ENV("DM_VK_GPUPROF_LIST", 0))   // every step in order
        for (int k = 0; k < ns; k++) { const gp_mark *a = &g_m[s[k].i - 1], *b = &g_m[s[k].i];
            VLOG("gpu prof: step #%d %.3f ms: %s:%d, %llu draws, %llu ops%s%s", s[k].i, s[k].ms, b->file, b->line, (unsigned long long)(b->draws - a->draws),
                 (unsigned long long)(b->ops - a->ops), b->desc[0] ? "; pass " : "", b->desc); }
    qsort(s, (size_t)ns, sizeof *s, by_ms);
    for (int k = 0; k < ns && k < 16; k++) {
        const gp_mark *a = &g_m[s[k].i - 1], *b = &g_m[s[k].i];
        VLOG("gpu prof:  #%d %.3f ms: %s:%d, %llu draws, %llu passes, %llu ops%s%s", s[k].i, s[k].ms, b->file, b->line, (unsigned long long)(b->draws - a->draws),
             (unsigned long long)(b->passes - a->passes), (unsigned long long)(b->ops - a->ops), b->desc[0] ? "; pass " : "", b->desc);
    }
    // by barrier site: count and time
    struct { const char *f; int l; int n; double ms; } site[64]; int nsite = 0;
    for (int k = 0; k < ns; k++) { const gp_mark *b = &g_m[s[k].i]; int j = 0;
        while (j < nsite && !(site[j].l == b->line && site[j].f == b->file)) j++;
        if (j == nsite) { if (nsite == 64) continue; site[nsite++] = (typeof(site[0])){b->file, b->line, 0, 0}; }
        site[j].n++; site[j].ms += s[k].ms; }
    char line[1200]; size_t ln = 0; line[0] = 0;
    for (int j = 0; j < nsite && ln < sizeof line; j++) ln += (size_t)snprintf(line + ln, sizeof line - ln, " %s:%d %dx %.2fms;", site[j].f, site[j].l, site[j].n, site[j].ms);
    VLOG("gpu prof: by site:%s", line);
    if (!g_split_frame) {   // a plain frame: its passes by size, for the next split report
        g_nplain = 0; g_plain_frame = g_frame;
        for (int k = 0; k < ns; k++) { const gp_mark *a = &g_m[s[k].i - 1], *b = &g_m[s[k].i]; unsigned w, h;
            if (!b->desc[0] || b->passes == a->passes || sscanf(b->desc, "%ux%u", &w, &h) != 2) continue;
            int j = 0; while (j < g_nplain && !(g_plain[j].w == w && g_plain[j].h == h)) j++;
            if (j == g_nplain) { if (g_nplain == 32) continue; g_plain[g_nplain++] = (plain_pass){w, h, 0, 0, 0}; }
            g_plain[j].ms += s[k].ms; g_plain[j].draws += (uint32_t)(b->draws - a->draws); g_plain[j].n++; }
    }
}
typedef struct { uint64_t vs, fs, idx; double ms; uint32_t draws, groups, ind; uint64_t st[NST]; int st_n; } sp_pipe;   // st: statistics summed over st_n groups
static int by_pipe_ms(const void *a, const void *b) { double d = ((const sp_pipe *)b)->ms - ((const sp_pipe *)a)->ms; return d > 0 ? 1 : d < 0 ? -1 : 0; }
typedef struct { int i; double ms; } sp_rank;
static int by_rank(const void *a, const void *b) { double d = ((const sp_rank *)b)->ms - ((const sp_rank *)a)->ms; return d > 0 ? 1 : d < 0 ? -1 : 0; }
static void split_report(void) {
    static uint64_t t[NQS], st[NQS][NST]; static sp_pipe P[NPIPE];
    if (!g_nsq) { VLOG("gpu prof split: frame %llu: no pass of %ux%u (DM_VK_GPUPROF_SPLIT_W / _H; 0 any) was drawn", (unsigned long long)g_frame,
                       (unsigned)VKA_ENV("DM_VK_GPUPROF_SPLIT_W", 0), (unsigned)VKA_ENV("DM_VK_GPUPROF_SPLIT_H", 0)); return; }
    if (g_get(V.dev, g_qps, 0, (uint32_t)g_nsq, sizeof t, t, 8, VK_QUERY_RESULT_64_BIT)) { VLOG("gpu prof split: frame %llu: results not ready", (unsigned long long)g_frame); return; }
    #define TMS(a, b) ((b) > (a) ? ((b) - (a)) * g_tick_ns / 1e6 : 0.0)
    sp_rank rk[NSP]; int nr = 0, minw = VKA_ENV("DM_VK_GPUPROF_SPLIT_MIN", 32); double small = 0; int nsmall = 0;
    for (int i = 0; i < g_nspp; i++) { const sp_pass *p = &g_spp[i]; if (!p->q1) continue;
        double ms = TMS(t[p->q0], t[p->q1]);
        if ((int)p->draws >= minw) rk[nr++] = (sp_rank){i, ms}; else { small += ms; nsmall++; } }
    qsort(rk, (size_t)nr, sizeof *rk, by_rank);
    VLOG("gpu prof split: frame %llu: %d passes split (%d with %d draws or more), %d timestamps%s; passes under %d draws: %d, %.2f ms", (unsigned long long)g_frame, g_nspp, nr, minw, g_nsq,
         g_split_cut ? " (table full: later passes or groups not split)" : "", minw, nsmall, small);
    for (int r = 0; r < nr && r < 3; r++) {
        const sp_pass *p = &g_spp[rk[r].i]; int np = 0;
        // this pass's groups' statistics (the queries of its groups, one read; not ready or none: the pass's lines go without them)
        int stv = p->stats && p->g1 > p->g0 && !g_get(V.dev, g_qst, p->g0, p->g1 - p->g0, sizeof st[0] * (p->g1 - p->g0), st[p->g0], sizeof st[0], VK_QUERY_RESULT_64_BIT);
        uint64_t pst[NST] = {0}; double setup = p->g1 > p->g0 ? TMS(t[p->q0], t[g_sgp[p->g0].q]) : 0, grp = 0; uint32_t other = 0;
        for (uint32_t g = p->g0; g < p->g1; g++) { const sp_grp *x = &g_sgp[g]; uint32_t qe = g + 1 < p->g1 ? g_sgp[g + 1].q : p->q1; double ms = TMS(t[x->q], t[qe]); grp += ms;
            int j = 0; while (j < np && !(P[j].vs == x->vs && P[j].fs == x->fs)) j++;
            if (j == np) { if (np == NPIPE) { j = NPIPE - 1; other++; } else P[np++] = (sp_pipe){x->vs, x->fs, 0, 0, 0, 0, 0, {0}, 0}; }
            P[j].ms += ms; P[j].draws += x->draws; P[j].idx += x->idx; P[j].groups++; P[j].ind += x->ind;
            if (stv) { for (int q = 0; q < NST; q++) { P[j].st[q] += st[g][q]; pst[q] += st[g][q]; } P[j].st_n++; } }
        // split / plain over the same set: every split pass of this size (the small ones too) against every plain pass of this size
        const plain_pass *pp = NULL; for (int j = 0; j < g_nplain && !pp; j++) if (g_plain[j].w == p->w && g_plain[j].h == p->h) pp = &g_plain[j];
        double sms = 0; int sn = 0; uint32_t sd = 0;
        for (int i = 0; i < g_nspp; i++) { const sp_pass *x = &g_spp[i]; if (x->q1 && x->w == p->w && x->h == p->h) { sms += TMS(t[x->q0], t[x->q1]); sn++; sd += x->draws; } }
        char pl[320]; if (pp) snprintf(pl, sizeof pl, "all passes of this size: split frame %d passes %.2f ms %u draws, plain profiled frame %llu %d passes %.2f ms %u draws (driver's choice of GMEM or sysmem); "
                                       "split / plain %.2f%s", sn, sms, sd, (unsigned long long)g_plain_frame, pp->n, pp->ms, pp->draws, pp->ms > 0 ? sms / pp->ms : 0.0,
                                       sd != pp->draws ? " (draw counts differ: not the same work)" : "");
        else snprintf(pl, sizeof pl, "no pass of this size in the last plain profiled frame");
        VLOG("gpu prof split: pass #%d %ux%u: %.2f ms from its first to its last timestamp (%.2f ms before its first draw), %u draws (%u indirect), %llu indices, %u groups, %d pipelines; "
             "render mode %s; %s%s", r + 1, p->w, p->h, rk[r].ms, setup, p->draws, p->ind, (unsigned long long)p->idx, p->g1 - p->g0, np,
             p->sysmem ? "sysmem asked (an in-pass vertex barrier: Turnip's \"Non-framebuffer-space barrier\")" : "the driver's (DM_VK_GPUPROF_SPLIT_SYSMEM=0: under GMEM the groups are one bin's)", pl,
             p->cut ? "; table full: the last group holds the rest" : "");
        double px = (double)p->w * p->h;
        if (p->stats && !stv) VLOG("gpu prof split:   pipeline statistics of pass #%d: not read (results not ready)", r + 1);
        else if (stv) VLOG("gpu prof split:   pipeline statistics of pass #%d (%ux%u = %.0f pixels): fragment shader invocations %llu = %.2f per pixel; vertex shader invocations %llu, "
                           "clipping invocations %llu, primitives out of clipping %llu", r + 1, p->w, p->h, px, (unsigned long long)pst[3], pst[3] / px, (unsigned long long)pst[0],
                           (unsigned long long)pst[1], (unsigned long long)pst[2]);
        qsort(P, (size_t)np, sizeof *P, by_pipe_ms);
        double rest = 0; int nrest = 0;
        for (int j = 0; j < np; j++) {
            if (j >= 16) { rest += P[j].ms; nrest++; continue; }
            // (each "ns" is the group's whole time over that stage's invocations: an upper bound for the stage; with no fragments the time is the vertex stage's (and
            // the fixed function's), with no vertices the fragment stage's)
            char sl[320]; sl[0] = 0;
            if (P[j].st_n) snprintf(sl, sizeof sl, "; fs invocations %llu (%.2f per pixel of the pass, %.1f ns each), vs invocations %llu (%.1f ns each), primitives %llu%s", (unsigned long long)P[j].st[3],
                                    P[j].st[3] / px, P[j].st[3] ? P[j].ms * 1e6 / (double)P[j].st[3] : 0.0, (unsigned long long)P[j].st[0], P[j].st[0] ? P[j].ms * 1e6 / (double)P[j].st[0] : 0.0,
                                    (unsigned long long)P[j].st[2], !P[j].st[3] && P[j].st[0] ? "; no fragments: the vertex stage's time" : !P[j].st[0] && P[j].st[3] ? "; no vertices counted: the fragment stage's time" : "");
            VLOG("gpu prof split:   vs %016llx fs %016llx: %.2f ms (%.0f%%), %u draws%s, %llu indices, %u groups%s", (unsigned long long)P[j].vs, (unsigned long long)P[j].fs, P[j].ms,
                 grp > 0 ? 100.0 * P[j].ms / grp : 0.0, P[j].draws, P[j].ind ? " (some indirect: their indices not counted)" : "", (unsigned long long)P[j].idx, P[j].groups, sl);
        }
        if (nrest || other) VLOG("gpu prof split:   %d more pipelines: %.2f ms%s", nrest, rest, other ? " (past the table: in the last line)" : "");
    }
    #undef TMS
}
// mtl_sub_begin, after the command buffer begins: arms the profile at a new frame, reports a finished one
void vka_gp_sub_begin(VkCommandBuffer cb, uint64_t frame) {
    if (g_sub_frame_set) frame = g_sub_frame;
    if (g_state == 2 && vka_done() >= g_fence) { report(); if (g_split_frame) split_report(); g_state = 0; }
    if (g_state == 1 && frame != g_frame) { g_state = 2; vka_gp_active = 0; vka_gp_split = 0; g_cb = VK_NULL_HANDLE; }
    if (g_state == 2 && vka_done() >= g_fence) { report(); if (g_split_frame) split_report(); g_state = 0; }
    if (g_state == 0) {
        int every = VKA_ENV("DM_VK_GPUPROF", 0), split = VKA_ENV("DM_VK_GPUPROF_SPLIT", 0) > 0;
        if (split && every <= 0) every = 30;   // (the split alone: one frame in 30 profiled, every other one split)
        if (every <= 0 || frame == g_frame || (g_last_frame && frame < g_last_frame + (uint64_t)every) || !init()) return;
        g_state = 1; g_frame = g_last_frame = frame; g_n = 0; vka_gp_active = 1; g_pass_new = 0;
        { int vka_gpu_clock(char *out, size_t n); g_clk0_ok = vka_gpu_clock(g_clk0, sizeof g_clk0); }
        g_reset(cb, g_qp, 0, NQ);
        g_split_frame = split && !(g_split_toggle ^= 1) && split_init();   // plain and split frames alternate, a plain one first (the reference)
        if (!split) g_split_toggle = 0;
        if (g_split_frame) { g_reset(cb, g_qps, 0, NQS); g_nsq = g_nsgp = g_nspp = g_split_cut = 0; vka_gp_split = 0; g_st_open = 0;
                             g_st_reset = VKA_ENV("DM_VK_GPUPROF_SPLIT_STATS", 0) > 0 && stats_init(); if (g_st_reset) g_reset(cb, g_qst, 0, NQS); }
    }
    if (g_state == 1) { g_cb = cb; mark(cb, 1, "submission", 0); }
}
void vka_gp_commit(VkCommandBuffer cb, uint64_t fence) { if (g_state == 1 && cb == g_cb) { mark(cb, 3, "commit", 0); g_fence = fence; } }
#endif
