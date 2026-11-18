// The replay worker: the game's submissions and presents are replayed on a thread of their own, so the game's thread goes on to its next frame while the
// Vulkan commands of this one are recorded (on the Adreno 840 phone Donkey Kong Bananza's level spent ~48 ms a frame recording and ~29 ms in its own
// work on the same thread, one after the other).
//
// What the game sees stays what NVN promises, as the inline replay gave it:
// - Order: submissions, presents and queue finalizations are replayed in the order the game made them (one FIFO, one thread).
// - Fences: a submission's fence is reserved when the game submits (vka_fence_next, in order), so mtl_last_fence, QueueFenceSync and the window's acquire syncs name
//   it at once; a sync fenced inside a command buffer (OP_FENCE_SYNC) is given it before the submission is queued (fence_syncs), as the game may wait on it right
//   after. A wait for a fence the worker has not submitted yet waits for it (vka_wait_done waits for completion, not for submission); the hang detection only
//   counts fences handed to the GPU.
// - Presents: the ticket is reserved when the game presents (vka_present_reserve: its order and the bound of the present queue), the frame is filled when the
//   worker reaches it, after every submission before it was handed to the queue.
// - CPU reads at replay time (deferred texture binds, render conditions, indirect counts, texture uploads from storage) happen later than inline, which is
//   closer to the GPU reading them when it runs the commands; the game cannot rely on the old timing (its GPU runs a frame or more behind).
// - mtl_finish / mtl_drain wait for the last fence reserved, so for the worker too.
// The game runs at most DM_VK_WORKER_FRAMES (default 1) presents ahead of the worker. DM_VK_WORKER=0 replays inline on the submitting thread, as before.
#if defined(__ANDROID__)
#include "vka.h"
#include "../nvn/nvn_mtl.h"
#include "../../host/plat/plat.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
uint64_t vka_frame(void);
void vka_exec_sub_frame(uint64_t f);
static uint64_t wns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
// Where the time goes (logged every 300 presents): the worker busy replaying, idle waiting for a job; the game waiting for room in the queue
static uint64_t g_busy_ns, g_idle_ns, g_room_ns, g_sub_ns;
static _Atomic uint64_t g_room_all, g_sub_all, g_sub_cpu_all;   // cumulative (vka_worker_waits: the [longframe] line's attribution); sub_cpu: the worker's CPU time in submissions
void vka_worker_waits(uint64_t *room_ns, uint64_t *sub_ns) { if (room_ns) *room_ns = atomic_load(&g_room_all); if (sub_ns) *sub_ns = atomic_load(&g_sub_all); }
uint64_t vka_worker_sub_cpu(void) { return atomic_load(&g_sub_cpu_all); }   // a recording time far above it: the replay thread was blocked in the submission, not busy
static uint64_t thread_cpu_ns(void) { struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }

enum { J_SUB = 1, J_PRESENT, J_FORGET };
enum { MESH_ARG_LIMIT = 2048, MESH_OP_LIMIT = 100000 };
typedef struct { const uint8_t *record; uint64_t addr; uint32_t words[2]; int readable; } mesh_arg_sample;
typedef struct {
    mesh_arg_sample samples[MESH_ARG_LIMIT];
    unsigned count, seen, ops, incomplete, mismatch, unsampled, skipped, changed, same, unreadable, zero_to_positive, positive_to_zero;
    int pending, invalid;
    int gw_enabled; uint64_t gw_key;
} mesh_arg_probe;
static _Thread_local mesh_arg_probe *g_mesh_arg_probe;
enum { MESH_ARG_SHADER_LIMIT = 64 };
typedef struct { uint64_t key; unsigned zero_zero, positive_positive, zero_positive, positive_zero, unreadable; } mesh_arg_shader;
static mesh_arg_shader g_mesh_arg_shaders[MESH_ARG_SHADER_LIMIT];
static unsigned g_mesh_arg_shader_count, g_mesh_arg_shader_overflow;
static uint64_t g_mesh_arg_shader_start_ns;

static void mesh_arg_collect(mesh_arg_probe *probe, const uint8_t *record, int depth) {
    while (record && !probe->incomplete) {
        if (++probe->ops > MESH_OP_LIMIT) { probe->incomplete = 1; return; }
        op_hdr header; if (plat_mem_read((uint64_t)(uintptr_t)record, &header, sizeof header)) { probe->incomplete = 1; return; }
        if (header.op == OP_END) return;
        if (header.size < 8 || (header.size & 7) || header.size > (64u << 20) || header.op >= OP_COUNT_ || (uintptr_t)record > UINTPTR_MAX - header.size) { probe->incomplete = 1; return; }
        const uint8_t *args = record + 8;
        if (header.op == OP_JUMP || header.op == OP_CALL) {
            if (header.size < 16) { probe->incomplete = 1; return; }
            uint64_t target; if (plat_mem_read((uint64_t)(uintptr_t)args, &target, sizeof target)) { probe->incomplete = 1; return; }
            if (header.op == OP_JUMP) { record = (const uint8_t *)(uintptr_t)target; if (!record) probe->incomplete = 1; continue; }
            if (depth >= 16 || !target) { probe->incomplete = 1; return; }
            mesh_arg_collect(probe, (const uint8_t *)(uintptr_t)target, depth + 1);
        } else if (header.op == OP_DRAW && header.size >= 8 + offsetof(p_draw, mdi_cnt)) {
            p_draw draw = {0}; if (plat_mem_read((uint64_t)(uintptr_t)args, &draw, offsetof(p_draw, mdi_cnt))) { probe->incomplete = 1; return; }
            if (draw.prim == NVN_PRIM_MESH && draw.ind) {
                if (probe->count == MESH_ARG_LIMIT) { probe->incomplete = 1; return; }
                mesh_arg_sample *sample = &probe->samples[probe->count++];
                sample->record = record; sample->addr = draw.ind;
                sample->readable = !plat_mem_read(draw.ind, sample->words, sizeof sample->words);
            }
        }
        record += header.size;
    }
}

void vka_mesh_arg_probe_visit(const uint8_t *record, const p_draw *draw) {
    mesh_arg_probe *probe = g_mesh_arg_probe;
    if (!probe || !draw || draw->prim != NVN_PRIM_MESH || !draw->ind) return;
    if (probe->pending) { probe->skipped++; probe->pending = 0; }
    if (probe->invalid) { probe->unsampled++; return; }
    if (probe->seen >= probe->count) { if (probe->incomplete) probe->unsampled++; else { probe->mismatch++; probe->invalid = 1; } return; }
    mesh_arg_sample *sample = &probe->samples[probe->seen++];
    if (sample->record != record || sample->addr != draw->ind) { probe->mismatch++; probe->invalid = 1; return; }
    probe->pending = 1;
}

void vka_mesh_arg_probe_read(uint64_t addr, int readable, const uint32_t words[2], uint64_t mesh_key) {
    mesh_arg_probe *probe = g_mesh_arg_probe;
    if (!probe || !probe->pending) return;
    probe->pending = 0;
    mesh_arg_sample *sample = &probe->samples[probe->seen - 1];
    if (sample->addr != addr) { probe->mismatch++; probe->invalid = 1; return; }
    if (probe->gw_enabled && probe->gw_key == mesh_key) {
        static _Thread_local unsigned observed, emitted;
        if (!(observed++ % 32) && emitted < 96) {
            unsigned flags = vka_tex_gw_probe(addr, sizeof sample->words, NULL); emitted++;
            VLOG("TEMP mesh-arg write-candidates mono_ns=%llu key=%016llx submit=%s%u replay=%s%u uncommitted_op_record=%u uncommitted_opaque_intent_or_page=%u flight_op_record=%u flight_opaque_intent_or_page=%u done_op_record=%u done_opaque_intent_or_page=%u scan_capped=%u invalid_range=%u retained_ledger_only=1 execution_and_exact_writes_unverified=1", (unsigned long long)wns(), (unsigned long long)mesh_key, sample->readable ? "" : "unreadable:", sample->words[0], readable ? "" : "unreadable:", words[0], !!(flags & VKA_GW_CURRENT_KNOWN), !!(flags & VKA_GW_CURRENT_OPAQUE), !!(flags & VKA_GW_FLIGHT_KNOWN), !!(flags & VKA_GW_FLIGHT_OPAQUE), !!(flags & VKA_GW_DONE_KNOWN), !!(flags & VKA_GW_DONE_OPAQUE), !!(flags & VKA_GW_TRUNCATED), !!(flags & VKA_GW_INVALID_RANGE));
        }
    }
    if (!g_mesh_arg_shader_start_ns) g_mesh_arg_shader_start_ns = wns();
    unsigned shader_index = 0;
    while (shader_index < g_mesh_arg_shader_count && g_mesh_arg_shaders[shader_index].key != mesh_key) shader_index++;
    mesh_arg_shader *shader = NULL;
    if (shader_index < MESH_ARG_SHADER_LIMIT) {
        shader = &g_mesh_arg_shaders[shader_index];
        if (shader_index == g_mesh_arg_shader_count) { shader->key = mesh_key; g_mesh_arg_shader_count++; }
    } else g_mesh_arg_shader_overflow++;
    if (shader) {
        if (!sample->readable || !readable) shader->unreadable++;
        else if (!sample->words[0] && !words[0]) shader->zero_zero++;
        else if (!sample->words[0] && words[0]) shader->zero_positive++;
        else if (sample->words[0] && !words[0]) shader->positive_zero++;
        else shader->positive_positive++;
    }
    if (!sample->readable || !readable) { probe->unreadable++; return; }
    if (sample->words[0] == words[0] && sample->words[1] == words[1]) probe->same++;
    else { probe->changed++; if (!sample->words[0] && words[0]) probe->zero_to_positive++; if (sample->words[0] && !words[0]) probe->positive_to_zero++; }
}

typedef struct {
    int kind;
    uint64_t serial, fence, frame; int origin_ll;   // J_SUB: the queue, its window origin, the fence reserved for it, the game's frame when it submitted
    const uint8_t **h; int nh;                   // J_SUB: the command handles, in order
    mesh_arg_probe *mesh_probe;
    uint64_t ticket; tex_t t; double period; int has_crop, crop[4];   // J_PRESENT
} job_t;
enum { NJOB = 256 };
static job_t g_q[NJOB]; static uint64_t g_qin, g_qout, g_pres_in, g_pres_out;   // jobs queued / done; presents queued / done
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_cv = PTHREAD_COND_INITIALIZER;
static int g_on = -1, g_frames = 1, g_up;
// The submission being gathered on the game's thread (the host submits under its own lock: one at a time)
static job_t g_cur; static int g_cur_cap, g_in_sub; static uint64_t g_pend_serial; static int g_pend_origin;

int vka_worker_on(void) {
    if (g_on < 0) { const char *e = getenv("DM_VK_WORKER"); g_on = !e || atoi(e); e = getenv("DM_VK_WORKER_FRAMES"); g_frames = e && atoi(e) > 0 ? atoi(e) : 1; }
    return g_on;
}
// Core placement experiments (live, CPU masks as decimal numbers, 0 = leave as is: 192 = cpus 6-7): DM_VK_AFF_REPLAY for this thread, DM_VK_AFF_ALL for every thread of the process (read at each job)
#include <sched.h>
#include <dirent.h>
#include <errno.h>
static void aff_set(pid_t tid, unsigned mask) {
    cpu_set_t cs; CPU_ZERO(&cs); for (int c = 0; c < 32; c++) if (mask >> c & 1) CPU_SET(c, &cs);
    if (sched_setaffinity(tid, sizeof cs, &cs)) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 4) VRED("sched_setaffinity(%d, %#x): errno %d", (int)tid, mask, errno); }
}
static void aff_check(void) {
    static unsigned r_was, a_was;
    unsigned r = (unsigned)VKA_ENV("DM_VK_AFF_REPLAY", 0), a = (unsigned)VKA_ENV("DM_VK_AFF_ALL", 0);
    if (a != a_was) { a_was = a; if (a) { DIR *d = opendir("/proc/self/task"); struct dirent *e; int n = 0;
            while (d && (e = readdir(d))) if (e->d_name[0] != '.') { aff_set((pid_t)atoi(e->d_name), a); n++; }
            if (d) closedir(d); VLOG("DM_VK_AFF_ALL=%#x: %d threads", a, n); } }
    if (r != r_was) { r_was = r; if (r) { aff_set(0, r); VLOG("DM_VK_AFF_REPLAY=%#x", r); } }
}
static void *worker_main(void *arg) {
    (void)arg; pthread_setname_np(pthread_self(), "vk-replay");
    for (;;) {
        uint64_t t0 = wns();
        pthread_mutex_lock(&g_mu);
        while (g_qout == g_qin) pthread_cond_wait(&g_cv, &g_mu);
        job_t j = g_q[g_qout % NJOB];
        pthread_mutex_unlock(&g_mu);
        uint64_t t1 = wns(), c1 = j.kind == J_SUB ? thread_cpu_ns() : 0; g_idle_ns += t1 - t0;
        aff_check();
        if (j.kind == J_SUB) {
            g_mesh_arg_probe = j.mesh_probe;
            vka_rp_queue_activate(j.serial); vka_rp_set_queue_origin(j.origin_ll);
            vka_gp_sub_frame(j.frame); vka_exec_sub_frame(j.frame); vka_rp_sub_begin();
            for (int i = 0; i < j.nh; i++) vka_rp_exec(j.h[i], j.fence);
            if (j.mesh_probe) {
                mesh_arg_probe *probe = j.mesh_probe;
                if (probe->pending) probe->skipped++;
                static uint64_t last_ns; static unsigned emitted, jobs, same, changed, unreadable, skipped, mismatch, unsampled, incomplete, ztop, ptoz, unmatched;
                jobs++; same += probe->same; changed += probe->changed; unreadable += probe->unreadable; skipped += probe->skipped;
                mismatch += probe->mismatch; unsampled += probe->unsampled; incomplete += probe->incomplete; ztop += probe->zero_to_positive; ptoz += probe->positive_to_zero;
                unmatched += probe->count - probe->seen;
                uint64_t now = wns();
                if (!last_ns) last_ns = now;
                if (now - last_ns >= 2000000000ull && emitted < 96) {
                    VLOG("TEMP mesh-arg submit-replay mono_ns=%llu jobs=%u same=%u changed=%u zero_to_positive=%u positive_to_zero=%u unreadable=%u skipped=%u mismatch=%u unsampled=%u incomplete=%u unmatched=%u", (unsigned long long)now, jobs, same, changed, ztop, ptoz, unreadable, skipped, mismatch, unsampled, incomplete, unmatched);
                    char shader_text[8192]; size_t used = 0; shader_text[0] = 0;
                    for (unsigned i = 0; i < g_mesh_arg_shader_count; i++) {
                        mesh_arg_shader *shader = &g_mesh_arg_shaders[i];
                        int length = snprintf(shader_text + used, sizeof shader_text - used, "%s%016llx:%u/%u/%u/%u/%u", i ? "," : "", (unsigned long long)shader->key, shader->zero_zero, shader->zero_positive, shader->positive_zero, shader->positive_positive, shader->unreadable);
                        if (length < 0 || (size_t)length >= sizeof shader_text - used) { g_mesh_arg_shader_overflow++; break; }
                        used += (size_t)length;
                    }
                    VLOG("TEMP mesh-arg shaders start_mono_ns=%llu end_mono_ns=%llu counts[zero_zero,zero_positive,positive_zero,positive_positive,unreadable]={%s} overflow=%u", (unsigned long long)(g_mesh_arg_shader_start_ns ? g_mesh_arg_shader_start_ns : last_ns), (unsigned long long)now, shader_text, g_mesh_arg_shader_overflow);
                    memset(g_mesh_arg_shaders, 0, sizeof g_mesh_arg_shaders); g_mesh_arg_shader_count = g_mesh_arg_shader_overflow = 0; g_mesh_arg_shader_start_ns = 0;
                    last_ns = now; emitted++; jobs = same = changed = unreadable = skipped = mismatch = unsampled = incomplete = ztop = ptoz = unmatched = 0;
                }
                free(probe); g_mesh_arg_probe = NULL;
            }
            vka_rp_sub_commit(j.fence);
            free(j.h);
        } else if (j.kind == J_PRESENT) vka_present_fill(j.ticket, &j.t, j.period, j.has_crop ? j.crop : NULL);
        else if (j.kind == J_FORGET) vka_rp_queue_forget(j.serial);
        uint64_t t2 = wns(); g_busy_ns += t2 - t1; if (j.kind == J_SUB) { g_sub_ns += t2 - t1; atomic_fetch_add_explicit(&g_sub_all, t2 - t1, memory_order_relaxed); atomic_fetch_add_explicit(&g_sub_cpu_all, thread_cpu_ns() - c1, memory_order_relaxed); }
        if (j.kind == J_PRESENT && j.ticket % 300 == 0) {
            VLOG("replay worker up to present %llu: busy %.1f ms a frame (submissions %.1f), idle %.1f; the game waited %.1f ms a frame for the worker",
                 (unsigned long long)j.ticket, g_busy_ns / 300e6, g_sub_ns / 300e6, g_idle_ns / 300e6, g_room_ns / 300e6);
            g_busy_ns = g_idle_ns = g_room_ns = g_sub_ns = 0; }
        pthread_mutex_lock(&g_mu);
        g_qout++; if (j.kind == J_PRESENT) g_pres_out++;
        pthread_cond_broadcast(&g_cv);
        pthread_mutex_unlock(&g_mu);
    }
    return NULL;
}
static int up(void) {
    if (!vka_worker_on()) return 0;
    if (g_up) return g_up > 0;
    pthread_t th;
    if (pthread_create(&th, NULL, worker_main, NULL)) { VRED("cannot start the replay worker: submissions are replayed inline"); g_up = -1; g_on = 0; return 0; }
    pthread_detach(th); g_up = 1;
    VLOG("replay worker: submissions and presents are replayed on their own thread, at most %d frame%s behind the game (DM_VK_WORKER=0: inline)", g_frames, g_frames == 1 ? "" : "s");
    return 1;
}
static void push(const job_t *j) {
    uint64_t t0 = wns();
    pthread_mutex_lock(&g_mu);
    while (g_qin - g_qout >= NJOB || (j->kind == J_PRESENT && g_pres_in - g_pres_out >= (uint64_t)g_frames)) pthread_cond_wait(&g_cv, &g_mu);
    { uint64_t w = wns() - t0; g_room_ns += w; atomic_fetch_add_explicit(&g_room_all, w, memory_order_relaxed); }
    g_q[g_qin % NJOB] = *j; g_qin++; if (j->kind == J_PRESENT) g_pres_in++;
    pthread_cond_broadcast(&g_cv);
    pthread_mutex_unlock(&g_mu);
}

// Syncs fenced inside the submission's command lists (OP_FENCE_SYNC), given the submission's fence now: the lists are walked as exec walks them (jumps, calls)
static void fence_syncs(const uint8_t *p, uint64_t fence, int depth) {
    for (;;) {
        op_hdr h; memcpy(&h, p, sizeof h);
        const void *a = p + 8;
        if (h.op == OP_END) return;
        if (h.op == OP_JUMP) { memcpy(&p, a, 8); continue; }
        if (h.op == OP_CALL) { uint64_t t; memcpy(&t, a, 8); if (depth < 16) fence_syncs((const uint8_t *)(uintptr_t)t, fence, depth + 1); }
        if (h.op == OP_FENCE_SYNC) { sync_t *s; memcpy(&s, a, 8); s->fence = fence; s->fenced = 1; }
        if (h.size < 8) return;
        p += h.size;
    }
}

// ---------- The mtl_* calls of the host's submit (hle_nvn.c nvnQueueSubmitCommands) and present ----------
void mtl_queue_activate(uint64_t serial) { if (up()) g_pend_serial = serial; else vka_rp_queue_activate(serial); }
void mtl_set_queue_origin_lower_left(int on) { if (up()) g_pend_origin = on != 0; else vka_rp_set_queue_origin(on); }
void mtl_queue_forget(uint64_t serial) { if (up()) { job_t j = {J_FORGET}; j.serial = serial; push(&j); } else vka_rp_queue_forget(serial); }
void mtl_sub_begin(void) {
    if (!up()) { if (VKA_ENV("DM_VK_MESH_ARG_DIFF", 0)) VRED_ONCE("mesh argument diagnostic requires the replay worker"); vka_rp_sub_begin(); return; }
    g_in_sub = 1; g_cur.nh = 0;
}
void mtl_exec(const uint8_t *handle) {
    if (!g_in_sub) { vka_rp_exec(handle, 0); return; }
    if (!handle) return;
    if (g_cur.nh == g_cur_cap) { g_cur_cap = g_cur_cap ? 2 * g_cur_cap : 64; g_cur.h = realloc(g_cur.h, (size_t)g_cur_cap * sizeof *g_cur.h); }
    g_cur.h[g_cur.nh++] = handle;
}
uint64_t mtl_sub_commit(void) {
    if (!g_in_sub) return vka_rp_sub_commit(0);
    g_in_sub = 0;
    job_t j = {J_SUB}; j.serial = g_pend_serial; j.origin_ll = g_pend_origin; j.nh = g_cur.nh;
    j.h = malloc((size_t)(j.nh ? j.nh : 1) * sizeof *j.h); if (j.nh) memcpy(j.h, g_cur.h, (size_t)j.nh * sizeof *j.h);
    j.fence = vka_fence_next(); j.frame = vka_frame();
    if (VKA_ENV("DM_VK_MESH_ARG_DIFF", 0)) {
        j.mesh_probe = calloc(1, sizeof *j.mesh_probe);
        const char *key = getenv("DM_VK_MESH_GW_KEY");
        if (j.mesh_probe && key && *key) {
            size_t length = strnlen(key, 17); int valid = length <= 16;
            for (size_t i = 0; valid && i < length; i++) if (!strchr("0123456789abcdefABCDEF", key[i])) valid = 0;
            if (valid) { j.mesh_probe->gw_enabled = 1; j.mesh_probe->gw_key = strtoull(key, NULL, 16); }
            else VRED_ONCE("mesh write-candidate diagnostic key must be 1-16 hexadecimal digits");
        }
        if (j.mesh_probe) for (int i = 0; i < j.nh && !j.mesh_probe->incomplete; i++) mesh_arg_collect(j.mesh_probe, j.h[i], 0);
        else VRED_ONCE("mesh argument diagnostic allocation failed");
    }
    for (int i = 0; i < j.nh; i++) fence_syncs(j.h[i], j.fence, 0);
    push(&j);
    return j.fence;
}
uint64_t mtl_present(const tex_t *t, double period_s, const int crop[4]) {
    if (!up()) return vka_present_inline(t, period_s, crop);
    job_t j = {J_PRESENT}; j.ticket = vka_present_reserve(); j.t = *t; j.period = period_s; j.has_crop = crop != NULL; if (crop) memcpy(j.crop, crop, sizeof j.crop);
    push(&j);
    return j.ticket;
}
#endif
