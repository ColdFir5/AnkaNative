// Draws and dispatches of the Vulkan replay: nvn_mtl.m's draw path on Vulkan. vka_exec.c walks the records; state records land in the queue's vka_dstate (vka_state_op),
// and each draw resolves its program to shaders (vka_shader.c), opens the dynamic-rendering pass of the bound targets if none is open, takes a pipeline keyed on the state
// Vulkan cannot set dynamically here (vka_pso.c), writes set 0 from the binding records (vka_bind.c's walker: push descriptors when they fit, else a set from the submission
// slot's pool), sets the dynamic state and draws. The pass stays open across draws and closes at anything else (vka_rp_end), followed by a full barrier.
// What is not here yet, each logged once: SINT attachments and integer vertex formats (no variants), transform feedback, DrawTexture, line loops and quads through indirect draws,
// tessellation's patch size, texel buffers, viewport swizzles other than a y flip
#if defined(__ANDROID__)
#include "vka.h"
#include "../nvn/nvn_mtl.h"
#include "../../host/plat/plat.h"
#include "vka_selftest_spv.h"
#include "vka_pidx.h"
#include "vka_rprof.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>

static VkCommandBuffer g_cb; static unsigned g_slot; static int g_prof, g_tm, g_dyn_filter = 1, g_pass_keep = 1, g_vb_filter = 1;   // g_prof: DM_VK_PROF; g_tm: it or DM_VK_RPROF (vka_rprof.h)
uint64_t vka_n_rp;   // render passes the draws opened (vka_exec.c replay statistics)
static uint64_t g_draws, g_skipped, g_dispatches;
static uint32_t g_claim_words;   // the draw's claim words (an upper bound of its primitives): 0 = its geometry shader takes no claim
#define SKIP(...) do { g_skipped++; static _Atomic int n_; if (atomic_fetch_add(&n_, 1) < 4) VRED(__VA_ARGS__); return; } while (0)

// Whether draws and dispatches run. Without placed guest memory (an emulator's driver) the GPU sees no memory pool, so a game's draws would read nothing and only cost:
// pipeline compiles there take minutes each. Off there by default, on with placed memory; DM_VK_DRAW=0 / 1 overrides, the selftest turns them on for its own memory
static int g_force_draws;
void vka_draw_force(int on) { g_force_draws = on; }
static int draws_on(void) {
    int env = VKA_ENV("DM_VK_DRAW", -1); if (env > 0) env = 1;
    int on = g_force_draws || (env >= 0 ? env : V.placed);
    if (!on) VRED_ONCE("draws and dispatches are off (%s): their records are skipped; events, counters and fences still signal", env == 0 ? "DM_VK_DRAW=0" : "no placed guest memory, so the GPU would see no pool; DM_VK_DRAW=1 turns them on");
    return on;
}

// ---------- State ----------
void vka_dstate_reset(vka_dstate *d) {   // NVN's defaults (nvn_mtl.m state_reset)
    memset(d, 0, sizeof *d); d->inited = 1;
    for (int i = 0; i < 8; i++) d->blend[i] = (blend_st){i, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD};
    d->color = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS}; d->chmask.mask = 0xffffffffu;
    d->poly = (poly_st){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0};
    d->ds = (ds_st){0, 1, NVN_DEPTH_FUNC_LESS, 0, NVN_STENCIL_FUNC_ALWAYS, NVN_STENCIL_FUNC_ALWAYS, 1, 1, 1, 1, 1, 1};
    d->st_vmask[0] = d->st_vmask[1] = d->st_wmask[0] = d->st_wmask[1] = 0xff; d->dr[1] = 1; d->b.samples = 1;
}
int nvn_dpool_base(const void *pool, int tex);   // hle_nvn.c: where a descriptor pool's IDs start in the host table
int vka_state_op(vka_dstate *d, uint32_t op, const void *a) {
    if (!d->inited) vka_dstate_reset(d);
    switch (op) {
    case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: { int tex = op == OP_SET_TEX_POOL, b = nvn_dpool_base(((const p_set_pool *)a)->pool, tex); if (tex) d->b.tex_base = b; else d->b.samp_base = b; break; }
    case OP_BIND_PROGRAM: { const p_bind_program *b = a;
        if (b->stages & 0x20) d->cprog = b->prog;
        for (int st = 0; st < VKA_NSTAGE; st++) if (st != 5 && (b->stages >> st & 1)) d->bst[st] = b->prog;   // NVN binds per stage (nvn_mtl.m bind_stages); 6 / 7: NVN2's task and mesh
        break; }
    case OP_BIND_STATE: {
        const p_bind_state *b = a; blend_st bs;
        switch (b->kind) {
        case 0: memcpy(&bs, b->bytes, sizeof bs); d->blend[bs.target] = bs; break;
        case 1: memcpy(&d->color, b->bytes, sizeof d->color); break;
        case 2: memcpy(&d->chmask, b->bytes, sizeof d->chmask); break;
        case 3: memcpy(&d->ms, b->bytes, sizeof d->ms); break;
        case 4: memcpy(&d->poly, b->bytes, sizeof d->poly); break;
        case 5: memcpy(&d->ds, b->bytes, sizeof d->ds); break;
        }
        break; }
    case OP_BIND_VATTR: { const p_bind_vattr *b = a; if (b->first >= 0 && b->first + b->n <= 16) { memcpy(&d->attr[b->first], b->st, 4 * (size_t)b->n); if (b->first + b->n > d->nattr) d->nattr = b->first + b->n; } break; }
    case OP_BIND_VSTREAM: { const p_bind_vstream *b = a; if (b->first >= 0 && b->first + b->n <= 16) { memcpy(&d->stream[b->first], b->st, 8 * (size_t)b->n); if (b->first + b->n > d->nstream) d->nstream = b->first + b->n; } break; }
    case OP_BIND_VBUF: case OP_BIND_VBUF1: { p_bind_vbuf tmp; const p_bind_vbuf *b = vbuf_rec(op, a, &tmp); if (b->first >= 0 && b->first + b->n <= 16) memcpy(&d->vbuf[b->first], b->r, sizeof *b->r * (size_t)b->n); break; }
    case OP_BIND_UBO: case OP_BIND_UBO_SMALL: { p_bind_buf tmp; const p_bind_buf *b = bind_buf_rec(op, a, &tmp); if (b->stage >= 0 && b->stage < VKA_NSTAGE && b->index >= 0 && b->index < 16) d->b.ubo[b->stage][b->index] = (vka_range){b->addr, b->size}; break; }
    case OP_BIND_SSBO: { const p_bind_buf *b = a; if (b->stage >= 0 && b->stage < VKA_NSTAGE && b->index >= 0 && b->index < 16) d->b.ssbo[b->stage][b->index] = (vka_range){b->addr, b->size}; break; }
    case OP_BIND_TEX: case OP_BIND_TEX1: { p_bind_handles t1; const p_bind_handles *b = bind_rec(op, a, &t1);
        if (b->n < 0 || b->n > 32) { VRED_ONCE("texture bind record with count %d: overwritten after recording", b->n); break; }
        uint64_t h[32]; memcpy(h, b->h, 8 * (size_t)b->n);
        if (b->defer && b->src && b->n) {   // BindTexturesDeferred: the handles as memory holds them now (nvn_mtl.m bh_fresh); placed guest memory read directly, else word by word through the fault-safe read
            uint32_t w[32];
            if (vka_addr(b->src, 4u * (uint64_t)b->n, NULL, NULL, NULL)) { memcpy(w, (const void *)(uintptr_t)b->src, 4 * (size_t)b->n); for (int k = 0; k < b->n; k++) h[k] = w[k]; }
            else if (!plat_mem_read(b->src, w, 4 * (size_t)b->n)) for (int k = 0; k < b->n; k++) h[k] = w[k];   // one read for the run (they are in the game's heap; profile p2: 3.4 % in process_vm_readv)
            else for (int k = 0; k < b->n; k++) if (!plat_mem_read(b->src + 4u * k, &w[k], 4)) h[k] = w[k];   // a run across an unreadable page: word by word
        }
        if (b->stage >= 0 && b->stage < VKA_NSTAGE && b->first >= 0 && b->first + b->n <= VKA_NTEX) memcpy(&d->b.tex[b->stage][b->first], h, 8 * (size_t)b->n);
        break; }
    case OP_BIND_SAMP: { const p_bind_handles *b = a; if (b->stage >= 0 && b->stage < VKA_NSTAGE && b->first >= 0 && b->n >= 0 && b->first + b->n <= 128) memcpy(&d->b.samp[b->stage][b->first], b->h, 8 * (size_t)b->n); break; }
    case OP_BIND_IMG: { const p_bind_handles *b = a; if (b->stage >= 0 && b->stage < VKA_NSTAGE && b->first >= 0 && b->n >= 0 && b->first + b->n <= 32) memcpy(&d->b.img[b->stage][b->first], b->h, 8 * (size_t)b->n); break; }
    case OP_STENCIL: { const p_stencil *b = a;
        for (int f = 0; f < 2; f++) if (b->face & (1 << f)) { if (b->which == 0) d->st_vmask[f] = (uint32_t)b->value_mask; else if (b->which == 1) d->st_wmask[f] = (uint32_t)b->mask; else d->st_ref[f] = (uint32_t)b->ref; }
        break; }
    case OP_POLY_OFFSET: d->po = *(const p_poly_offset *)a; break;
    case OP_VIEWPORTS: { const p_viewports *b = a; for (int i = 0; i < b->n && b->first + i < 16; i++) memcpy(d->vp[b->first + i], b->v[i], 16); d->has_vp = 1; break; }
    case OP_DEPTH_RANGE: { const p_depth_range *b = a; d->dr[0] = b->n; d->dr[1] = b->f; break; }
    case OP_DEPTH_CLAMP: d->depth_clamp = ((const p_depth_clamp *)a)->on; break;
    case OP_VP_SWIZZLE: { const p_vp_swizzle *b = a; if (b->first >= 0 && b->first + b->n <= 16) memcpy(d->vp_sw[b->first], b->sw, 4 * (size_t)b->n); break; }
    case OP_BLEND_COLOR: memcpy(d->blend_color, ((const p_blend_color *)a)->rgba, 16); break;
    case OP_ALPHA_REF: d->alpha_ref = ((const p_alpha_ref *)a)->ref; break;
    case OP_PUSH_CONST: { const p_push_const *b = a; if ((uint32_t)b->offset + b->size <= DMSH_PUSH_SIZE) memcpy(d->b.push[b->compute ? 1 : 0] + b->offset, b->data, b->size); break; }
    case OP_TF_BIND: break;
    case OP_TF_STATE: { const p_tf_state *b = a; d->tf_on = b->what == 0 || b->what == 3; break; }
    case OP_PRIM_RESTART: d->restart = *(const p_prim_restart *)a; break;
    case OP_RAST_DISCARD: d->rast_discard = ((const p_rast_discard *)a)->on; break;
    default: return 0;
    }
    return 1;
}

// ---------- Static buffers: zeros, a write sink, the support buffer ----------
static vka_span g_zero, g_sink, g_support;
static int statics(void) {
    static int ok = -1; if (ok >= 0) return ok;
    ok = vka_static(65536, V.ub_align > V.sb_align ? V.ub_align : V.sb_align, &g_zero) && vka_static(65536, V.sb_align, &g_sink) && vka_static(VKA_SUPPORT_SIZE, V.ub_align, &g_support);
    if (ok) { memset(g_zero.ptr, 0, 65536); memset(g_sink.ptr, 0, 65536); vka_support_fill(g_support.ptr, VKA_SUPPORT_SIZE); }
    return ok;
}

// ---------- Descriptor sets that do not fit a push (per submission slot) ----------
enum { NSLOT = 8, MAXPOOLS = 64 };
static VkDescriptorPool g_dp[NSLOT][MAXPOOLS]; static int g_ndp[NSLOT], g_cur_dp[NSLOT];
void vka_dyn_invalidate(void);
static void drv_memo_reset(void);
static void reach_line(void);
void vka_draw_sub_begin(VkCommandBuffer cb, unsigned slot) {
    g_cb = cb; g_slot = slot % NSLOT; vka_dyn_invalidate();
    reach_line();
    { void vka_mem_rec_thread(void); vka_mem_rec_thread(); }   // the address lookups' cache outside TLS for the recording thread (vka_mem.c)
    drv_memo_reset();   // the last submission's scratch spans go with its fence
    g_prof = VKA_ENV("DM_VK_PROF", 0) != 0; g_tm = g_prof || vka_rprof_on; g_dyn_filter = VKA_ENV("DM_VK_DYN_FILTER", 1); g_pass_keep = VKA_ENV("DM_VK_PASS_KEEP", 1); g_vb_filter = VKA_ENV("DM_VK_VB_FILTER", 1);   // live: from this submission
    for (int i = 0; i < g_ndp[g_slot]; i++) vk.vkResetDescriptorPool(V.dev, g_dp[g_slot][i], 0);   // the slot's last submission is done (mtl_sub_begin waited for it)
    g_cur_dp[g_slot] = 0;
    vka_pso_save(0);
    static int ph; VkImageView placeholder(uint8_t dim, int storage);
    if (!V.null_desc && !ph && statics()) {   // no null descriptors: the placeholders are made here, outside any pass (their first use records a clear)
        ph = 1; static const uint8_t dims[] = {1, 2, 3, 1 | DMSH_VK_DIM_ARRAY, 3 | DMSH_VK_DIM_ARRAY};
        for (size_t i = 0; i < sizeof dims; i++) for (int c = 0; c < 3; c++) placeholder(dims[i] | (c == 1 ? DMSH_VK_DIM_SINT : c == 2 ? DMSH_VK_DIM_UINT : 0), 0);
    }
}
static VkDescriptorSet set_alloc(VkDescriptorSetLayout l) {
    for (;;) {
        int i = g_cur_dp[g_slot];
        if (i == g_ndp[g_slot]) {
            if (i == MAXPOOLS) { VRED_ONCE("descriptor pools exhausted in one submission"); return VK_NULL_HANDLE; }
            VkDescriptorPoolSize ps[] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8192}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8192},
                                         {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1024}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024},
                                         {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256}, {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256}};
            VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1024, .poolSizeCount = sizeof ps / sizeof *ps, .pPoolSizes = ps};
            if (vk.vkCreateDescriptorPool(V.dev, &pi, NULL, &g_dp[g_slot][i])) { VRED_ONCE("descriptor pool not made"); return VK_NULL_HANDLE; }
            g_ndp[g_slot]++;
        }
        VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = g_dp[g_slot][i], .descriptorSetCount = 1, .pSetLayouts = &l};
        VkDescriptorSet s; VkResult r = vk.vkAllocateDescriptorSets(V.dev, &ai, &s);
        if (!r) return s;
        if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL) { VRED_ONCE("vkAllocateDescriptorSets %d", r); return VK_NULL_HANDLE; }
        g_cur_dp[g_slot]++;
    }
}

// ---------- The render pass ----------
static struct {
    int on; VkRect2D area; uint32_t layers, samples;
    uint32_t ncolor; VkFormat cf[8]; uint8_t ckind[8]; VkFormat df, sf;
    vka_img *cim[8]; uint32_t clv[8], cl0[8]; int nci;   // the colour attachments (DM_VK_NANSCAN)
    p_set_rt rt; int has_rt;                             // the targets it was opened for: a later SET_RT of the same targets keeps it open
    uint32_t cnl[8], dnl; VkExtent2D cext[8], dext;      // each attachment's layers and level size (clears in the pass must cover what they would outside it)
    int swrite;                                          // a draw of the pass wrote memory (storage buffers or images, raw global stores, a geometry shader's claim)
    int bar_pending;                                     // a game barrier came inside the pass and was taken there (vka_rp_game_barrier): the draws after it are checked (bar_hazard)
    vka_img *wim[9]; int8_t wslot[9]; int nw;            // its watched big attachments (DM_VK_RTWATCH_MB, vka_tex.c "Big targets"): colour index, or -1 depth / stencil
} R;
// Clears waiting for the pass of their targets (vka_rp_clear), in order: the next draw over the same targets opens its pass with them (loadOp CLEAR, or vkCmdClearAttachments),
// anything else does them outside a pass first (pc_flush), as before
enum { NPC = 16 };
static struct { int n; p_set_rt rt; vka_pclear c[NPC]; } PC;
static void pc_flush(void) { int n = PC.n; PC.n = 0; for (int i = 0; i < n; i++) vka_clear_outside(&PC.c[i]); }
// The same targets: the same textures and the same views of them (field by field: the records' padding is not compared)
static int view_same(const view_t *a, const view_t *b) {
    return a->set_levels == b->set_levels && a->set_layers == b->set_layers && a->set_format == b->set_format && (!a->set_levels || (a->base_level == b->base_level && a->nlevels == b->nlevels)) &&
           (!a->set_layers || (a->min_layer == b->min_layer && a->nlayers == b->nlayers)) && (!a->set_format || a->format == b->format);
}
static int rt_same(const p_set_rt *a, int ha, const p_set_rt *b, int hb) {
    if (!ha || !hb) return !ha && !hb;
    if (a->n != b->n || a->depth != b->depth || !!a->has_dview != !!b->has_dview || (a->has_dview && !view_same(&a->dview, &b->dview))) return 0;
    for (int i = 0; i < a->n && i < 8; i++) {
        if (a->color[i] != b->color[i] || !!a->has_cview[i] != !!b->has_cview[i]) return 0;
        if (a->color[i] && a->has_cview[i] && !view_same(&a->cview[i], &b->cview[i])) return 0;
    }
    return 1;
}
// Game barriers (OP_BARRIER) that came with a pass open: taken inside it (DM_VK_BAR_KEEP), ended it at once, ended it later at a draw that needed it (bar_hazard)
static uint64_t g_bar_in, g_bar_kept, g_bar_late;
void vka_rp_end(void) {
    if (PC.n) pc_flush();
    if (!R.on) return;
    if (vka_gp_split) vka_gp_rp_end(g_cb);   // DM_VK_GPUPROF_SPLIT: the split pass's last timestamp, inside it
    vk.vkCmdEndRendering(g_cb);
    vka_barrier(g_cb);   // the pass's writes before whatever reads them next (copies, the present, the next pass's samples)
    R.on = 0; vka_tex_feedback(NULL, NULL, 0);
    if (vka_nan_on()) vka_nan_pass(g_cb, R.cim, R.clv, R.cl0, R.nci);
}
static void rt_range(const vka_img *im, const view_t *v, int has_v, uint32_t *lv, uint32_t *l0, uint32_t *nl) {
    *lv = has_v && v->set_levels ? v->base_level : 0;
    *l0 = has_v && v->set_layers ? v->min_layer : 0;
    if (!vka_img_lv(im, lv)) { *lv = im->levels; *nl = 0; return; }   // a level the image leaves out (a sparse texture's unmapped levels)
    uint32_t all = vka_img_slices(im, *lv);
    *nl = has_v && v->set_layers && v->nlayers ? v->nlayers : all - (*l0 < all ? *l0 : 0);
    if (*l0 + *nl > all) *nl = *l0 < all ? all - *l0 : 0;
}
// Opens the pass over the bound targets: 0 = it cannot (no target can be an attachment), logged
static int rp_begin(const p_set_rt *rt, int has_rt, const vka_dstate *d) {
    if (R.on) { if (!g_pass_keep || rt_same(rt, has_rt, &R.rt, R.has_rt)) return 1; vka_rp_end(); }   // a new SET_RT of other targets ends the pass at its first draw (exec does not)
    if (PC.n && !rt_same(rt, has_rt, &PC.rt, 1)) pc_flush();                             // clears of other targets: done before
    VkRenderingAttachmentInfo ca[8], da = {0}, sa; uint32_t w = UINT32_MAX, h = UINT32_MAX, layers = UINT32_MAX, samples = 0;
    vka_img *fb_im[9]; uint32_t fb_lv[9]; int nfb = 0;   // the attachments' images and levels (vka_tex_feedback)
    memset(&R, 0, sizeof R);
    if (has_rt) R.rt = *rt; R.has_rt = has_rt;
    int n = has_rt ? (rt->n < 8 ? rt->n : 8) : 0;
    for (int i = 0; i < n; i++) {
        ca[i] = (VkRenderingAttachmentInfo){VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
        R.cf[i] = VK_FORMAT_UNDEFINED;
        const tex_t *t = rt->color[i]; if (!t) continue;
        vka_rl_site = VKA_RL_ATT; vka_img *im = vka_tex_get(t, g_cb); vka_rl_site = 0; vka_fmt f;   // (vka_rl_site: DM_VK_RTLOG_MB's path)
        if (!im || !vka_fmt_of(rt->has_cview[i] && rt->cview[i].set_format ? rt->cview[i].format : t->d.format, &f)) continue;
        if (!(im->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) || f.kind >= 3) { VRED_ONCE("colour target %d (format %#x) cannot be a Vulkan colour attachment: not drawn into", i, t->d.format); continue; }
        if (im->depth > 1 && !(im->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) continue;   // its slices are drawn into through 2D-array views (img_create)
        vka_tex_whole(im, g_cb);   // (a coarse-first image made without its finer levels: whole before rt_range maps the level, vka_tex.c "Coarse-first")
        uint32_t lv, l0, nl; rt_range(im, &rt->cview[i], rt->has_cview[i], &lv, &l0, &nl);
        if (lv >= im->levels || !nl) continue;
        if (samples && im->samples != samples) { VRED_ONCE("colour targets of different sample counts"); continue; }
        vka_tex_own(im, g_cb);   // (a zero-shared image gets memory of its own before it is drawn into: no pass is open here)
        VkImageView v = vka_view(im, f.view, nl > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, lv, 1, l0, nl, NULL);
        if (!v) continue;
        ca[i].imageView = v; R.cf[i] = f.view; R.ckind[i] = f.kind; samples = im->samples; fb_im[nfb] = im; fb_lv[nfb++] = lv;
        R.cim[R.nci] = im; R.clv[R.nci] = lv; R.cl0[R.nci++] = l0;
        uint32_t lw = im->w >> lv ? im->w >> lv : 1, lh = im->h >> lv ? im->h >> lv : 1;
        R.cnl[i] = nl; R.cext[i] = (VkExtent2D){lw, lh};
        if (lw < w) w = lw; if (lh < h) h = lh; if (nl < layers) layers = nl;
        if (im->rtw && R.nw < 9) { R.wim[R.nw] = im; R.wslot[R.nw++] = (int8_t)i; vka_rtw(im, VKA_RTW_PASS); }
        vka_img_written(im);
    }
    R.ncolor = (uint32_t)n;
    VkRenderingAttachmentInfo *dp = NULL, *sp = NULL;
    if (has_rt && rt->depth) {
        vka_rl_site = VKA_RL_ATT; vka_img *im = vka_tex_get(rt->depth, g_cb); vka_rl_site = 0;
        if (im && im->f.kind >= 3 && (im->usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) && (!samples || im->samples == samples)) {
            uint32_t lv, l0, nl; rt_range(im, &rt->dview, rt->has_dview, &lv, &l0, &nl);
            if (lv < im->levels && nl) vka_tex_own(im, g_cb);
            VkImageView v = lv < im->levels && nl ? vka_view(im, im->f.vk, nl > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, im->aspect, lv, 1, l0, nl, NULL) : VK_NULL_HANDLE;
            if (v) {
                da = (VkRenderingAttachmentInfo){VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = v, .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
                if (im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) { dp = &da; R.df = im->f.vk; }
                if (im->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) { sp = &da; R.sf = im->f.vk; }
                samples = im->samples; fb_im[nfb] = im; fb_lv[nfb++] = lv;
                uint32_t lw = im->w >> lv ? im->w >> lv : 1, lh = im->h >> lv ? im->h >> lv : 1;
                R.dnl = nl; R.dext = (VkExtent2D){lw, lh};
                if (lw < w) w = lw; if (lh < h) h = lh; if (nl < layers) layers = nl;
                if (im->rtw && R.nw < 9) { R.wim[R.nw] = im; R.wslot[R.nw++] = -1; vka_rtw(im, VKA_RTW_PASS); }
                vka_img_written(im);
            }
        } else if (im) VRED_ONCE("depth target (format %#x) cannot be a Vulkan depth attachment: drawn without depth", rt->depth->d.format);
    }
    if (w == UINT32_MAX) {   // no attachment at all (rasterizer discard, or targets Vulkan cannot take): the viewport's size, for side effects
        if (!d->has_vp) { VRED_ONCE("a draw with no render target and no viewport: skipped"); return 0; }
        w = (uint32_t)fabsf(d->vp[0][2]); h = (uint32_t)fabsf(d->vp[0][3]); if (!w) w = 1; if (!h) h = 1; layers = 1;
    }
    R.area = (VkRect2D){{0, 0}, {w, h}}; R.layers = layers; R.samples = samples ? samples : 1;
    // The pending clears of these targets (PC): loadOp CLEAR when one covers its attachment and the attachment is the render area, else vkCmdClearAttachments once the pass
    // begins when the rectangle is inside the render area and the attachment has the pass's layers, else outside before it begins
    int nlate = 0; vka_pclear *late[NPC];
    sa = da;
    for (int k = 0; k < PC.n; k++) {
        vka_pclear *c = &PC.c[k]; int ci = c->ci;
        int att = ci >= 0 ? ci < n && ca[ci].imageView && R.cnl[ci] == layers && R.cnl[ci] == c->nl
                          : R.dnl == layers && R.dnl == c->nl && (!(c->asp & VK_IMAGE_ASPECT_DEPTH_BIT) || dp) && (!(c->asp & VK_IMAGE_ASPECT_STENCIL_BIT) || sp);
        VkExtent2D e = ci >= 0 ? (ci < 8 ? R.cext[ci] : (VkExtent2D){0, 0}) : R.dext;
        int whole = att && !c->rc.offset.x && !c->rc.offset.y && c->rc.extent.width == w && c->rc.extent.height == h && e.width == w && e.height == h;
        int inarea = att && (uint32_t)c->rc.offset.x + c->rc.extent.width <= w && (uint32_t)c->rc.offset.y + c->rc.extent.height <= h;
        if (whole && !nlate) {   // (after a late one it would come first: late too)
            if (ci >= 0) { ca[ci].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca[ci].clearValue = c->in; }
            else { if (c->asp & VK_IMAGE_ASPECT_DEPTH_BIT) { da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; da.clearValue.depthStencil.depth = c->in.depthStencil.depth; }
                   if (c->asp & VK_IMAGE_ASPECT_STENCIL_BIT) { sa.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; sa.clearValue.depthStencil.stencil = c->in.depthStencil.stencil; } }
        } else if (inarea) late[nlate++] = c;
        else vka_clear_outside(c);
    }
    if (sp) sp = &sa;
    #undef INAREA
    #undef WHOLE
    vka_tex_feedback(fb_im, fb_lv, nfb);
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = R.area, .layerCount = layers, .colorAttachmentCount = (uint32_t)n, .pColorAttachments = ca,
                          .pDepthAttachment = dp, .pStencilAttachment = sp};
    if (vka_gp_active) {   // DM_VK_GPUPROF: what the pass is, for the step it ends
        char b[96]; size_t bn = (size_t)snprintf(b, sizeof b, "%ux%ux%u", w, h, layers);
        for (int i = 0; i < n && bn < sizeof b; i++) if (ca[i].imageView) bn += (size_t)snprintf(b + bn, sizeof b - bn, " c%d:%d%s", i, R.cf[i], ca[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ? "C" : "");
        if (dp || sp) snprintf(b + bn, bn < sizeof b ? sizeof b - bn : 0, " d:%d%s", R.df ? R.df : R.sf, (dp ? dp : sp)->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR ? "C" : "");
        vka_gp_pass(b);
    }
    if (VKA_ENV("DM_VK_DRAW_LOG", 0) >= 1) {   // (the draws that follow draw into these: a UI's text layers)
        char b[256]; int o = 0; for (int i = 0; i < R.nci && o < 200; i++) o += snprintf(b + o, sizeof b - o, " serial %llu %ux%u fmt %d;", (unsigned long long)R.cim[i]->serial, R.cim[i]->w, R.cim[i]->h, R.cim[i]->f.vk);
        VLOG("draw pass: %d colour targets%s %ux%u", R.nci, o ? b : "", R.area.extent.width, R.area.extent.height); }
    vk.vkCmdBeginRendering(g_cb, &ri);
    if (vka_gp_active) vka_gp_rp_begin(g_cb, w, h);   // DM_VK_GPUPROF_SPLIT: a split frame's pass of the chosen size
    R.on = 1; vka_n_rp++; if (g_dyn_filter == 2) vka_dyn_invalidate();   // DM_VK_DYN_FILTER=2: the dynamic state set again at each pass (bisect)
    for (int k = 0; k < nlate; k++) { VkClearAttachment c = {late[k]->asp, late[k]->ci >= 0 ? (uint32_t)late[k]->ci : 0, late[k]->in}; VkClearRect cr = {late[k]->rc, 0, layers}; vk.vkCmdClearAttachments(g_cb, 1, &c, 1, &cr); }
    PC.n = 0;
    return 1;
}
int vka_pass_keep(void) { return g_pass_keep; }
int vka_rp_open(void) { return R.on; }
// A game barrier (nvnCommandBufferBarrier) with a draw pass open. NVN orders the work before it against the work after it (ORDER_PRIMITIVES / ORDER_FRAGMENTS, the cache
// invalidations); the replay used to end the pass there and record a full barrier, so every barrier between draws into the same targets cost a pass end, a store and a
// load of every attachment on a tiler (the next draw reopens it with LOAD). Inside a pass Vulkan already keeps the attachment writes of draws in rasterization order, so
// what a barrier between two draws of one pass can order is: memory the earlier draws wrote with shader stores (the pass then ends at once, as nvn_mtl.m's OP_BARRIER
// does for a pass with shader writes), and an attachment of the pass that a later draw samples (a feedback read: the pass ends before that draw, bar_hazard). Everything
// else the draws read was written before the pass began, behind the barrier that ended what wrote it. DM_VK_BAR_KEEP=1 takes such barriers inside the pass (live;
// default 0 until phone-checked); the counts are in the replay statistics line either way
int vka_rp_game_barrier(uint32_t bits) {
    (void)bits;
    if (!R.on) return 0;
    g_bar_in++;
    if (R.swrite || !VKA_ENV("DM_VK_BAR_KEEP", 0)) return 0;
    R.bar_pending = 1; g_bar_kept++;
    return 1;
}
void vka_rp_bar_stats(char *b, size_t n, double frames) {
    static uint64_t i0, k0, l0;
    if (frames <= 0 || g_bar_in == i0) { if (n) b[0] = 0; i0 = g_bar_in; k0 = g_bar_kept; l0 = g_bar_late; return; }
    snprintf(b, n, "game barriers inside a draw pass per frame: %.1f, %.1f taken inside it (DM_VK_BAR_KEEP), %.1f of those ended it later at a draw sampling its targets or writing memory",
             (g_bar_in - i0) / frames, (g_bar_kept - k0) / frames, (g_bar_late - l0) / frames);
    i0 = g_bar_in; k0 = g_bar_kept; l0 = g_bar_late;
}
int vka_rp_is_open(void) { return R.on; }
int vka_rp_open_for(const p_set_rt *rt, int has_rt) { return R.on && rt_same(rt, has_rt, &R.rt, R.has_rt); }
// The draw about to come may start a new part of the submission: no pass is open over its targets and no clear waits for them (an open pass over other targets ends at it anyway)
int vka_rp_split_ok(const p_set_rt *rt, int has_rt) { return R.on ? !rt_same(rt, has_rt, &R.rt, R.has_rt) : PC.n ? has_rt && !rt_same(rt, has_rt, &PC.rt, 1) : 1; }
void vka_draw_cb_switch(VkCommandBuffer cb) { g_cb = cb; vka_dyn_invalidate(); }   // the submission goes on in another command buffer (vka_exec.c chunk): its bound state is gone
// Before a clear of these targets records anything (vka_tex_get may record a first upload): a pass over other targets ends, pending clears of other targets are done
void vka_rp_prep_clear(const p_set_rt *rt, int has_rt) {
    if (R.on) { if (!rt_same(rt, has_rt, &R.rt, R.has_rt)) vka_rp_end(); }
    else if (PC.n && !rt_same(rt, has_rt, &PC.rt, 1)) pc_flush();
}
// A clear of one of the bound targets (colour index ci, or depth / stencil when ci < 0) inside the render pass: an open pass over the same targets takes it as
// vkCmdClearAttachments (rasterization-ordered with its draws), a closed one opens over them with it (loadOp CLEAR). 0 = not here: the caller ends the pass and clears outside.
// rc is in the attachment's level (the scissored part), nl the layers the clear names: they must be the attachment's own, within the render area
int vka_rp_clear(const p_set_rt *rt, int has_rt, const vka_dstate *d, const vka_pclear *c) {
    (void)d;
    if (!g_cb || !has_rt || !draws_on()) return 0;
    if (R.on && !rt_same(rt, has_rt, &R.rt, R.has_rt)) return 0;
    if (!R.on) {
        if (PC.n && (!rt_same(rt, has_rt, &PC.rt, 1) || PC.n == NPC)) pc_flush();
        if (!PC.n) PC.rt = *rt;
        PC.c[PC.n++] = *c;
        return 1;
    }
    int ci = c->ci; uint32_t nl = c->nl; VkRect2D rc = c->rc;
    if (ci >= 0 ? ci >= 8 || !R.cf[ci] || R.cnl[ci] != nl || nl != R.layers : ((c->asp & VK_IMAGE_ASPECT_DEPTH_BIT) && !R.df) || ((c->asp & VK_IMAGE_ASPECT_STENCIL_BIT) && !R.sf) || R.dnl != nl || nl != R.layers) return 0;
    if ((uint32_t)rc.offset.x + rc.extent.width > R.area.extent.width || (uint32_t)rc.offset.y + rc.extent.height > R.area.extent.height) return 0;
    VkClearAttachment ca = {c->asp, ci >= 0 ? (uint32_t)ci : 0, c->in}; VkClearRect cr = {rc, 0, nl};
    vk.vkCmdClearAttachments(g_cb, 1, &ca, 1, &cr);
    return 1;
}

// ---------- Formats and fixed-function mappings ----------
// NVN vertex formats 0x01-0x30: four rows (1-4 components) of {UNORM8, SNORM8, UINT8, SINT8, FLOAT16, UNORM16, SNORM16, UINT16, SINT16, FLOAT32, UINT32, SINT32}
#define VF(n, s) VK_FORMAT_##n##_##s
static const VkFormat VROW[4][12] = {
    {VF(R8, UNORM), VF(R8, SNORM), VF(R8, UINT), VF(R8, SINT), VF(R16, SFLOAT), VF(R16, UNORM), VF(R16, SNORM), VF(R16, UINT), VF(R16, SINT), VF(R32, SFLOAT), VF(R32, UINT), VF(R32, SINT)},
    {VF(R8G8, UNORM), VF(R8G8, SNORM), VF(R8G8, UINT), VF(R8G8, SINT), VF(R16G16, SFLOAT), VF(R16G16, UNORM), VF(R16G16, SNORM), VF(R16G16, UINT), VF(R16G16, SINT), VF(R32G32, SFLOAT), VF(R32G32, UINT), VF(R32G32, SINT)},
    {VF(R8G8B8, UNORM), VF(R8G8B8, SNORM), VF(R8G8B8, UINT), VF(R8G8B8, SINT), VF(R16G16B16, SFLOAT), VF(R16G16B16, UNORM), VF(R16G16B16, SNORM), VF(R16G16B16, UINT), VF(R16G16B16, SINT), VF(R32G32B32, SFLOAT), VF(R32G32B32, UINT), VF(R32G32B32, SINT)},
    {VF(R8G8B8A8, UNORM), VF(R8G8B8A8, SNORM), VF(R8G8B8A8, UINT), VF(R8G8B8A8, SINT), VF(R16G16B16A16, SFLOAT), VF(R16G16B16A16, UNORM), VF(R16G16B16A16, SNORM), VF(R16G16B16A16, UINT), VF(R16G16B16A16, SINT), VF(R32G32B32A32, SFLOAT), VF(R32G32B32A32, UINT), VF(R32G32B32A32, SINT)},
};
static const VkFormat VSCALED[4][2][2] = {   // [components][8 / 16 bits][unsigned / signed]
    {{VF(R8, USCALED), VF(R8, SSCALED)}, {VF(R16, USCALED), VF(R16, SSCALED)}}, {{VF(R8G8, USCALED), VF(R8G8, SSCALED)}, {VF(R16G16, USCALED), VF(R16G16, SSCALED)}},
    {{VF(R8G8B8, USCALED), VF(R8G8B8, SSCALED)}, {VF(R16G16B16, USCALED), VF(R16G16B16, SSCALED)}}, {{VF(R8G8B8A8, USCALED), VF(R8G8B8A8, SSCALED)}, {VF(R16G16B16A16, USCALED), VF(R16G16B16A16, SSCALED)}}};
#undef VF
// The Vulkan format for NVN vertex format f whose shader input has class want (DMSH_VK_TYPE_*): the format's own when its class matches, else the same layout in that class
// (Vulkan wants the classes equal). An integer format into a float input keeps its own format and *retype says the input's class must change (DMSH_VK_TYPE_SINT / UINT):
// Maxwell's fetch puts the integer's bits in the register the shader bit-casts, so the vertex module is rewritten to declare that input as an integer vector (vka_spvpatch.c).
// VK_FORMAT_UNDEFINED = not mapped
static VkFormat vin_format(int f, int want, int *changed, int *retype) {
    *changed = 0; *retype = 0;
    int n = 0, col = -1;   // row (components) and column of VROW, or a scaled / packed format
    if (f >= 0x01 && f <= 0x30) { n = (f - 1) / 12 + 1; col = (f - 1) % 12; }
    else if (f >= 0x51 && f <= 0x68) {   // integers read as floats (nvn_mtl.m vconv_i2f): USCALED / SSCALED, none for 32 bits
        int i = (f - 0x51) >> 1, sg = (f - 0x51) & 1, bits = 8 << (i % 3); n = i / 3 + 1;
        if (bits < 32 && want == DMSH_VK_TYPE_FLOAT) return VSCALED[n - 1][bits == 16][sg];
        col = bits == 8 ? 2 + sg : bits == 16 ? 7 + sg : 10 + sg; *changed = want == DMSH_VK_TYPE_FLOAT;
        if (want == DMSH_VK_TYPE_FLOAT) return VROW[n - 1][col];
    } else {
        VkFormat p = f == 0x3D ? VK_FORMAT_A2B10G10R10_UNORM_PACK32 : f == 0x69 ? VK_FORMAT_A2B10G10R10_SNORM_PACK32 : f == 0x3F ? VK_FORMAT_B10G11R11_UFLOAT_PACK32 : f == 0x9A ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_UNDEFINED;
        if (p && want != DMSH_VK_TYPE_FLOAT) { *changed = 1; return want == DMSH_VK_TYPE_UINT ? VK_FORMAT_R32_UINT : VK_FORMAT_R32_SINT; }
        return p;
    }
    static const int8_t cls[12] = {0, 0, 2, 1, 0, 0, 0, 2, 1, 0, 2, 1};   // DMSH_VK_TYPE_* of each column
    if (cls[col] == want) return VROW[n - 1][col];
    if (want == DMSH_VK_TYPE_FLOAT && retype) { *retype = cls[col]; return VROW[n - 1][col]; }
    *changed = 1;
    int bits = col < 4 ? 8 : col < 9 ? 16 : 32, sg = col == 1 || col == 3 || col == 6 || col == 8 || col == 11;
    if (want == DMSH_VK_TYPE_FLOAT) return bits < 32 ? VSCALED[n - 1][bits == 16][sg] : VROW[n - 1][9];
    int base = bits == 8 ? 2 : bits == 16 ? 7 : 10;   // the integer of the same width, in the class asked
    return VROW[n - 1][base + (want == DMSH_VK_TYPE_SINT)];
}
static VkBlendFactor bfac(int f) {
    switch (f) {
    case NVN_BLEND_FUNC_ZERO: return VK_BLEND_FACTOR_ZERO; case NVN_BLEND_FUNC_ONE: return VK_BLEND_FACTOR_ONE;
    case NVN_BLEND_FUNC_SRC_COLOR: return VK_BLEND_FACTOR_SRC_COLOR; case NVN_BLEND_FUNC_ONE_MINUS_SRC_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case NVN_BLEND_FUNC_SRC_ALPHA: return VK_BLEND_FACTOR_SRC_ALPHA; case NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case NVN_BLEND_FUNC_DST_ALPHA: return VK_BLEND_FACTOR_DST_ALPHA; case NVN_BLEND_FUNC_ONE_MINUS_DST_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case NVN_BLEND_FUNC_DST_COLOR: return VK_BLEND_FACTOR_DST_COLOR; case NVN_BLEND_FUNC_ONE_MINUS_DST_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case NVN_BLEND_FUNC_SRC_ALPHA_SATURATE: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case NVN_BLEND_FUNC_SRC1_COLOR: return VK_BLEND_FACTOR_SRC1_COLOR; case NVN_BLEND_FUNC_ONE_MINUS_SRC1_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case NVN_BLEND_FUNC_SRC1_ALPHA: return VK_BLEND_FACTOR_SRC1_ALPHA; case NVN_BLEND_FUNC_ONE_MINUS_SRC1_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case NVN_BLEND_FUNC_CONSTANT_COLOR: return VK_BLEND_FACTOR_CONSTANT_COLOR; case NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_COLOR: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case NVN_BLEND_FUNC_CONSTANT_ALPHA: return VK_BLEND_FACTOR_CONSTANT_ALPHA; case NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_ALPHA: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    }
    return VK_BLEND_FACTOR_ONE;
}
static VkBlendOp bop(int e) { return e >= 1 && e <= 5 ? (VkBlendOp)(e - 1) : VK_BLEND_OP_ADD; }
static int is_src1(int f) { return f >= NVN_BLEND_FUNC_SRC1_COLOR && f <= NVN_BLEND_FUNC_ONE_MINUS_SRC1_ALPHA; }
static int zero_src1(int f) { return f == NVN_BLEND_FUNC_SRC1_COLOR || f == NVN_BLEND_FUNC_SRC1_ALPHA ? NVN_BLEND_FUNC_ZERO : f == NVN_BLEND_FUNC_ONE_MINUS_SRC1_COLOR || f == NVN_BLEND_FUNC_ONE_MINUS_SRC1_ALPHA ? NVN_BLEND_FUNC_ONE : f; }
// An advanced blend mode as fixed-function factors where one exists (nvn_mtl.m adv_blend: the Porter-Duff modes with uncorrelated overlap)
static blend_st adv_blend(blend_st b) {
    if (!b.adv) return b;
    enum { Z = NVN_BLEND_FUNC_ZERO, O = NVN_BLEND_FUNC_ONE, SA = NVN_BLEND_FUNC_SRC_ALPHA, ISA = NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA, DA = NVN_BLEND_FUNC_DST_ALPHA, IDA = NVN_BLEND_FUNC_ONE_MINUS_DST_ALPHA };
    static const uint8_t pd[][4] = {
        [NVN_BLEND_ADVANCED_MODE_ZERO] = {Z, Z, Z, Z},        [NVN_BLEND_ADVANCED_MODE_SRC] = {O, Z, O, Z},           [NVN_BLEND_ADVANCED_MODE_DST] = {Z, O, Z, O},
        [NVN_BLEND_ADVANCED_MODE_SRC_OVER] = {O, ISA, O, ISA}, [NVN_BLEND_ADVANCED_MODE_DST_OVER] = {IDA, O, IDA, O},
        [NVN_BLEND_ADVANCED_MODE_SRC_IN] = {DA, Z, DA, Z},     [NVN_BLEND_ADVANCED_MODE_DST_IN] = {Z, SA, Z, SA},
        [NVN_BLEND_ADVANCED_MODE_SRC_OUT] = {IDA, Z, IDA, Z},  [NVN_BLEND_ADVANCED_MODE_DST_OUT] = {Z, ISA, Z, ISA},
        [NVN_BLEND_ADVANCED_MODE_SRC_ATOP] = {DA, ISA, Z, O},  [NVN_BLEND_ADVANCED_MODE_DST_ATOP] = {IDA, SA, O, Z},
        [NVN_BLEND_ADVANCED_MODE_XOR] = {IDA, ISA, IDA, ISA},  [NVN_BLEND_ADVANCED_MODE_PLUS] = {O, O, O, O},
        [NVN_BLEND_ADVANCED_MODE_PLUS_CLAMPED] = {O, O, O, O}, [NVN_BLEND_ADVANCED_MODE_PLUS_CLAMPED_ALPHA] = {O, O, O, O},
    };
    const uint8_t *f = b.adv < sizeof pd / sizeof *pd && pd[b.adv][0] ? pd[b.adv] : NULL;
    if (!f || b.adv_ovl || (b.adv_npm && f[0] != O && f[0] != Z)) { VRED_ONCE("advanced blend mode %#x has no fixed-function equivalent: the state's blend functions are used", (unsigned)b.adv); return b; }
    b.src_rgb = b.adv_npm && f[0] == O ? SA : f[0]; b.dst_rgb = f[1]; b.src_a = f[2]; b.dst_a = f[3];
    b.eq_rgb = b.eq_a = NVN_BLEND_EQUATION_ADD;
    return b;
}
// NVN primitive → Vulkan topology; *expand: 1 quads, 2 line loop (indices made on the CPU), 0 none; -1 = not drawable
static VkPrimitiveTopology topology(int prim, int *expand) {
    *expand = 0;
    switch (prim) {
    case NVN_DRAW_PRIMITIVE_POINTS: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case NVN_DRAW_PRIMITIVE_LINES: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case NVN_DRAW_PRIMITIVE_LINE_LOOP: *expand = 2; return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case NVN_DRAW_PRIMITIVE_LINE_STRIP: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case NVN_DRAW_PRIMITIVE_TRIANGLES: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP: case NVN_DRAW_PRIMITIVE_QUAD_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;   // a quad strip rasterizes as the triangle strip of its vertices
    case NVN_DRAW_PRIMITIVE_TRIANGLE_FAN: case NVN_DRAW_PRIMITIVE_POLYGON: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    case NVN_DRAW_PRIMITIVE_QUADS: *expand = 1; return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    case NVN_DRAW_PRIMITIVE_LINES_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_LINE_STRIP_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_TRIANGLES_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_PATCHES: return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    }
    *expand = -1; return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
}
static VkPrimitiveTopology topo_class(VkPrimitiveTopology t) {   // the pipeline's topology: one of each class, the draw sets its own dynamically
    switch (t) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: return t;
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY: case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST: return t;
    default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
}

// ---------- Guest memory as descriptors ----------
// How far the guest memory descriptors reach against what the bindings ask (the [reach] line, at most every 5 s while a count moved; vka_mem.c Span buffers):
// sb: storage buffers bound; sb_hard: ones whose bound size reaches past their placed run (the shader may read what the descriptor does not cover: a certain cut);
// sb_cut: runs shorter than the reach asked (the device's range, or the bound size under DM_VK_SSBO_EXACT); sb_more: of those, ones with placed memory right after the
// run (a span buffer or a window would reach on), sb_min the shortest such run; ub / ub_hard and vb / vb_hard: uniform buffers and vertex streams bound in place, and
// ones whose bound size reaches past the run; gm: draws and dispatches binding raw global memory's table, gm_short: while it left segments out (vka_mem_chunks);
// sink: checks that found the sink's bytes changed (a store to no pool); noview: texture bindings with no view
static struct { _Atomic uint64_t sb, sb_hard, sb_cut, sb_more, sb_min, ub, ub_hard, vb, vb_hard, gm, gm_short, sink, noview; } g_rch = {.sb_min = UINT64_MAX};
// The last ubo_desc / ssbo_desc call's cut, for its caller's DM_VK_REACH_LOG line (what = 0 none, 1 hard, 2 placed memory after the run)
static struct { int what; uint64_t want, run; } g_cut;
static void reach_cut(int hard, uint64_t addr, uint64_t want, uint64_t run, _Atomic uint64_t *n_hard) {
    VkBuffer b; VkDeviceSize o; int more = n_hard == &g_rch.sb_hard && vka_addr_run(addr + run, 1, &b, &o) != 0;   // (storage buffers: what a longer run would reach)
    if (hard) atomic_fetch_add_explicit(n_hard, 1, memory_order_relaxed);
    if (more) {
        atomic_fetch_add_explicit(&g_rch.sb_more, 1, memory_order_relaxed);
        uint64_t m = atomic_load_explicit(&g_rch.sb_min, memory_order_relaxed);
        while (run < m && !atomic_compare_exchange_weak_explicit(&g_rch.sb_min, &m, run, memory_order_relaxed, memory_order_relaxed)) {}
    }
    g_cut.what = hard ? 1 : more ? 2 : 0; g_cut.want = want; g_cut.run = run;
}
static int cpu_read(uint64_t addr, void *out, size_t n) {   // guest bytes as the CPU sees them now
    if (vka_addr(addr, n, NULL, NULL, NULL)) { memcpy(out, (const void *)(uintptr_t)addr, n); return 1; }
    return !plat_mem_read(addr, out, n);
}
static int rd_cb(void *ctx, uint64_t addr, void *out, size_t n) { (void)ctx; return cpu_read(addr, out, n); }
// A uniform buffer: bound in place when Vulkan's offset alignment allows, else a copy of what the CPU sees now (a GPU write earlier in this submission is missed: logged)
static int ubo_desc(uint64_t addr, uint64_t size, VkDescriptorBufferInfo *bi) {
    uint64_t given = size; g_cut.what = 0;
    if (!size || size > V.ub_range) size = size ? V.ub_range : 65536 < V.ub_range ? 65536 : V.ub_range;
    VkBuffer b; VkDeviceSize o;
    uint64_t run = vka_addr_run(addr, size, &b, &o);
    if (run && !(o % V.ub_align) && !(addr % V.ub_align)) {
        atomic_fetch_add_explicit(&g_rch.ub, 1, memory_order_relaxed);
        if (given && run < size) reach_cut(1, addr, size, run, &g_rch.ub_hard);
        *bi = (VkDescriptorBufferInfo){b, o, run}; return 1;
    }
    vka_span s; if (!vka_scratch(size, V.ub_align, &s)) return 0;
    if (!cpu_read(addr, s.ptr, size)) { memset(s.ptr, 0, size); VRED_ONCE("uniform buffer at %#llx is not readable: it reads zeros", (unsigned long long)addr); }
    else if (run) VRED_ONCE("uniform buffer at %#llx is not %llu-byte aligned: bound as a copy of the CPU's bytes", (unsigned long long)addr, (unsigned long long)V.ub_align);
    else VRED_ONCE("uniform buffer at %#llx is not in placed guest RAM: bound as a copy of the CPU's bytes", (unsigned long long)addr);
    *bi = (VkDescriptorBufferInfo){s.buf, s.off, size}; return 1;
}
// A storage buffer: NVN's are global memory with no bounds (the bound size only informs the shader's length queries), and shaders read outside it, past its end and before
// its address. A read before the address needs a descriptor that starts earlier (robust access gives zeros below it), which only a shader that reads its base from c0
// can use (entry flag DMSH_VK_F_SSBO_BASE: r.c0base, vka_c0_bases). So the descriptor covers the placed run from the address on, up to the device's range limit, and for
// such a shader also ssbo_back bytes before it in the same buffer (vka_ssbo_back: as far as the buffer goes, at most half the range). *back: the bytes before addr it
// starts. No shader is known to read before its address so far: the window was made for Ocarina of Time's title text (b19e1bdd247555ec) on a wrong diagnosis; that
// shader reads its glyph records within the bound size, and the text's cause was the translator's (docs/reference/shaders.md "A deferred write under a false guard").
// DM_VK_SSBO_EXACT=1 (live): the bound size from the address and no window before it (the behaviour before both; the c0 base words then hold the rounded address too,
// so a DMSH_VK_F_SSBO_BASE shader reads as an older one)
static int g_ssbo_exact;   // DM_VK_SSBO_EXACT, read once a draw or dispatch (walk_all) so the descriptors and the c0 base words of one draw agree
static uint64_t ssbo_back(void *ctx, uint64_t a) {   // vka_c0_bases' answer for an address rounded down to ssbo_align: where in its buffer the descriptor starts
    (void)ctx; VkBuffer b; VkDeviceSize o;
    if (g_ssbo_exact || !vka_addr_run(a, 1, &b, &o)) return 0;
    return vka_ssbo_back(o, V.sb_align, V.sb_range);
}
// *own: the run in addr's own segment (a part of a span buffer reaches further: vka_mem.c Span buffers), what a store through it is followed in (vka_tex_gw_opaque)
static int ssbo_desc(uint64_t addr, uint64_t size, int c0base, VkDescriptorBufferInfo *bi, uint64_t *back, uint64_t *own) {
    VkBuffer b; VkDeviceSize o; g_cut.what = 0;
    uint64_t w = c0base ? ssbo_back(NULL, addr) : 0, want = size && g_ssbo_exact ? size : V.sb_range - w;
    uint64_t run = vka_addr_run(addr, want, &b, &o);
    if (!run) { VRED_ONCE("storage buffer at %#llx is not in placed guest RAM: it is not bound", (unsigned long long)addr); return 0; }
    if (run + w > V.sb_range) run = V.sb_range - w;
    atomic_fetch_add_explicit(&g_rch.sb, 1, memory_order_relaxed);
    if (run < want) { atomic_fetch_add_explicit(&g_rch.sb_cut, 1, memory_order_relaxed); reach_cut(size && run < size, addr, size && run < size ? size : want, run, &g_rch.sb_hard); }
    if (own) { *own = vka_addr_run_own(addr, run); if (!*own) *own = run; }
    *bi = (VkDescriptorBufferInfo){b, o - w, w + run}; if (back) *back = w; return 1;
}
// Driver segments (c0, c2, the push bank) and raw global memory's table, kept from the last draw of this submission that filled them: a draw whose segment holds the same
// bytes binds the same scratch span (DM_VK_DRV_MEMO, live, default 1). Each draw used to fill a fresh 2.5 KB span per segment it reads (memset and fill into scratch, under
// the scratch lock: ~10 KB of writes a draw, ~80 MB a frame in DKB's level), so no two draws' pushes were ever the same and the push dedupe (push_same) never matched them.
// Graphics stages only: an indirect dispatch's counts are copied into its c0 by the GPU (vka_dispatch). A span lives until the submission's fence (vka_scratch_retire at
// its commit), so the memo is forgotten at each submission's start (vka_draw_sub_begin)
typedef struct { VkBuffer buf; VkDeviceSize off; } c0_at;
static struct { int valid; vka_span s; uint8_t b[DMSH_DRV_SIZE]; } g_dm[VKA_NSTAGE][3];
static struct { int valid; uint64_t gen; vka_span s; } g_gm;
static struct { int valid; uint32_t ref; vka_span s; } g_sm_ref;   // the support buffer with the alpha test's reference (stage_writes)
static int g_drv_memo = 1; static uint64_t g_dm_hit, g_dm_miss;
static void drv_memo_reset(void) { for (int st = 0; st < VKA_NSTAGE; st++) for (int b = 0; b < 3; b++) g_dm[st][b].valid = 0; g_gm.valid = 0; g_sm_ref.valid = 0; g_drv_memo = VKA_ENV("DM_VK_DRV_MEMO", 1); }
// c0base: the stage's shader reads its storage buffers from the base words in c0 (DMSH_VK_F_SSBO_BASE), which are written after the fill (vka_c0_bases, ssbo_back)
static int drv_seg(const vka_dstate *d, int st, int bank, int pool, VkDescriptorBufferInfo *bi, c0_at *c0, uint32_t c0base) {
    size_t sz = vka_drv_size(bank); vka_span s;
    // One memo slot per segment kind: c0, c2 and the push bank. Any other driver bank (an all-zero c17-style read of DMSH_DRV_SIZE bytes) takes the fill below: sharing
    // the push bank's slot, an all-zero push block matched it and the 0x100-byte span was bound with a 0xA00-byte range (other draws' scratch, or past the buffer)
    int bi_ = bank == 0 ? 0 : bank == 2 ? 1 : bank == DMSH_PUSH_BANK ? 2 : -1;
    if (g_drv_memo && bi_ >= 0 && st != 5 && sz <= DMSH_DRV_SIZE) {
        static uint8_t tmp[DMSH_DRV_SIZE];
        vka_drv_fill(&d->b, st, bank, pool, tmp);
        if (bank == 0 && c0base) vka_c0_bases(&d->b, st, c0base, ssbo_back, NULL, tmp);
        typeof(g_dm[0][0]) *m = &g_dm[st][bi_];
        if (m->valid && !memcmp(m->b, tmp, sz)) { g_dm_hit++; *bi = (VkDescriptorBufferInfo){m->s.buf, m->s.off, sz}; return 1; }
        if (!vka_scratch(sz, V.ub_align, &s)) return 0;
        memcpy(s.ptr, tmp, sz); memcpy(m->b, tmp, sz); m->s = s; m->valid = 1; g_dm_miss++;
        *bi = (VkDescriptorBufferInfo){s.buf, s.off, sz};
        return 1;
    }
    if (!vka_scratch(sz, V.ub_align, &s)) return 0;
    vka_drv_fill(&d->b, st, bank, pool, s.ptr);
    if (bank == 0 && c0base) vka_c0_bases(&d->b, st, c0base, ssbo_back, NULL, s.ptr);
    *bi = (VkDescriptorBufferInfo){s.buf, s.off, sz};
    if (bank == 0 && c0) { c0->buf = s.buf; c0->off = s.off; }
    if (st == 5 && bank == 0 && sz >= 848 && VKA_ENV("DM_VK_DISPATCH_LOG", 0) >= 2) {   // DM_VK_DISPATCH_LOG=2: the compute storage slots as the shader reads them in c0 (0x310 + 16 k)
        const uint32_t *w = (const uint32_t *)((const uint8_t *)s.ptr + 0x310);
        VLOG("dispatch c0 0x310-0x34f: %08x %08x %08x %08x | %08x %08x %08x %08x | %08x %08x %08x %08x | %08x %08x %08x %08x", w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15]);
    }
    return 1;
}
// Raw global memory's table (dmsh_abi.h DMSH_GMEM_MAP_*): entry 0 empty, then the placed chunks by address with delta = device address - guest address, unused entries at ~0
// (kept while no segment comes or goes: vka_mem_gen, and DM_VK_DRV_MEMO)
uint64_t vka_mem_gen(void);
// Raw global memory's sink (dmsh_abi_vk.h DMSH_VK_GMEM_SINK_WORD): a device buffer for the process's life, made at the first table; 0 when the device has no buffer
// device address or the buffer was not made (logged once; the helpers' access of an address in no pool then faults instead of being skipped). Host-visible where the device
// has such memory, filled with a pattern: only a store whose address is in no pool writes it (the helpers' skipped probes write nothing, their skipped loads only read),
// so a changed word means a shader's store was lost (gmem_sink_check, logged once)
// DM_VK_GMEM_SINK_FILL=<byte> (read once, 0xA5): the pattern. 0xFF makes a load from the sink a NaN (a float read of 0xFFFFFFFF), which DM_VK_NANSCAN finds where it lands
static uint8_t *g_sink_map; static uint8_t g_sink_fill = 0xA5;
static uint64_t gmem_sink(void) {
    static uint64_t addr; static int tried;
    if (tried) return addr;
    tried = 1;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = DMSH_VK_GMEM_SINK_BYTES, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
    VkBuffer b = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkMemoryRequirements mr; uint32_t t = UINT32_MAX;
    if (V.bda && !vk.vkCreateBuffer(V.dev, &bi, NULL, &b)) {
        vk.vkGetBufferMemoryRequirements(V.dev, b, &mr);
        t = vka_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
        if (t == UINT32_MAX) t = vka_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        if (t == UINT32_MAX) t = vka_mem_type(mr.memoryTypeBits, 0, 0);
        VkMemoryAllocateFlagsInfo af = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &af, .allocationSize = mr.size, .memoryTypeIndex = t};
        if (t != UINT32_MAX && !vk.vkAllocateMemory(V.dev, &ai, NULL, &mem) && !vk.vkBindBufferMemory(V.dev, b, mem, 0)) {
            VkBufferDeviceAddressInfo ba = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b};
            addr = vk.vkGetBufferDeviceAddress(V.dev, &ba);
            void *p = NULL;
            if ((V.memp.memoryTypes[t].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                && !vk.vkMapMemory(V.dev, mem, 0, VK_WHOLE_SIZE, 0, &p)) {
                g_sink_fill = (uint8_t)VKA_ENV("DM_VK_GMEM_SINK_FILL", 0xA5);
                if (g_sink_fill != 0xA5) VLOG("DM_VK_GMEM_SINK_FILL=%#x: raw global memory's sink holds that byte (a load from an address in no pool or past the table reads it)", g_sink_fill);
                g_sink_map = p; memset(g_sink_map, g_sink_fill, DMSH_VK_GMEM_SINK_BYTES);
            }
        }
    }
    if (!addr) VRED("no raw global memory sink (buffer device address %d): an access through a pool table to an address in no pool faults", V.bda);
    return addr;
}
// (the recording thread; the GPU's writes show up some submissions later). Each find is counted ([reach] sink) and the pattern written again, so later stores show too;
// the line once. (A shader's store meanwhile may land before the refill and be missed: a count, not every store)
static void gmem_sink_check(void) {
    static int told;
    if (!g_sink_map) return;
    int n = 0, first = -1;
    for (int i = 0; i < DMSH_VK_GMEM_SINK_BYTES; i++) if (g_sink_map[i] != g_sink_fill) { n++; if (first < 0) first = i; }
    if (!n) return;
    atomic_fetch_add_explicit(&g_rch.sink, 1, memory_order_relaxed);
    memset(g_sink_map, g_sink_fill, DMSH_VK_GMEM_SINK_BYTES);
    if (told) return;
    told = 1;
    VRED("raw global memory: a shader stored to an address in no memory pool (%d bytes of the sink changed, the first at +%d): that store is lost", n, first);
}
static int g_gm_short;   // the table last built left segments out
static void gm_table(uint8_t *p, const uint64_t *lo, const uint64_t *hi, const uint64_t *bda, int k) {   // the table from vka_mem_chunks' k entries
    memset(p, 0xff, (size_t)DMSH_GMEM_MAP_ENTRIES * DMSH_GMEM_MAP_ENTRY); memset(p, 0, DMSH_GMEM_MAP_ENTRY);
    uint64_t sink = gmem_sink(); memcpy(p + DMSH_VK_GMEM_SINK_WORD * 4, &sink, sizeof sink);   // entry 0's padding: the sink (gmem_flatten)
    for (int i = 0; i < k && i + 1 < DMSH_GMEM_MAP_ENTRIES; i++) { uint64_t e[4] = {lo[i], hi[i], bda[i] - lo[i], 0}; memcpy(p + (size_t)(i + 1) * DMSH_GMEM_MAP_ENTRY, e, sizeof e); }
    { vka_table_stats t; vka_mem_table_stats(&t); g_gm_short = t.dropped > 0; }
}
static int gmem_map(VkDescriptorBufferInfo *bi) {
    size_t n = (size_t)DMSH_GMEM_MAP_ENTRIES * DMSH_GMEM_MAP_ENTRY;
    gmem_sink_check();
    uint64_t gen = vka_mem_gen();
    atomic_fetch_add_explicit(&g_rch.gm, 1, memory_order_relaxed);
    if (g_drv_memo && g_gm.valid && g_gm.gen == gen) { if (g_gm_short) atomic_fetch_add_explicit(&g_rch.gm_short, 1, memory_order_relaxed); *bi = (VkDescriptorBufferInfo){g_gm.s.buf, g_gm.s.off, n}; return 1; }
    vka_span s; if (!vka_scratch(n, V.sb_align, &s)) return 0;
    static uint64_t lo[DMSH_GMEM_MAP_ENTRIES], hi[DMSH_GMEM_MAP_ENTRIES], bda[DMSH_GMEM_MAP_ENTRIES];
    int k = vka_mem_chunks(lo, hi, bda, DMSH_GMEM_MAP_ENTRIES - 1);
    gm_table(s.ptr, lo, hi, bda, k);
    if (g_gm_short) atomic_fetch_add_explicit(&g_rch.gm_short, 1, memory_order_relaxed);
    g_gm.valid = 1; g_gm.gen = gen; g_gm.s = s;   // (a segment that came between the read of gen and the table: the next draw sees a new gen and builds it again)
    *bi = (VkDescriptorBufferInfo){s.buf, s.off, n}; return 1;
}
// Raw global memory's page index (dmsh_abi_vk.h DMSH_VK_GMEM_PIDX_*, vka_pidx.h; vk_android_spv.py gmem_pidx): a shader whose gmem_map record has DMSH_VK_BF_PIDX reads
// the table with the index after it: an 8-byte record per 4 KiB page of the window the entries span (Ocarina of Time: 5.6-8.9 GiB of guest addresses, 11-18 MiB of
// records), too large to copy into scratch at each generation (about 45 placements a second while it loads). So the table and index live in buffers of their own
// (host-visible, coherent, like scratch), each kept with the generation and entries it holds: a new generation takes a buffer the GPU is done with (its last submission's
// fence reached, vka_gmem_retire) and brings it up to date by the entries that differ (vka_pidx_update: only their pages written; a full build when the window no longer
// fits), so each draw reads the generation it was recorded with, as with the scratch table. The table part is rewritten each time (128 KiB). With every buffer taken
// (DM_VK_GMEM_PIDX_BUFS, default 4, at most 8), the draw waits for the oldest one handed to the queue; when all of them belong to the submission being recorded (that many
// generations in one submission), the newest is brought up to date in place: draws recorded earlier in that submission then read the newer generation (no GPU race: the
// submission has not gone to the queue), counted in the [reach] line. Kept on the generation whatever DM_VK_DRV_MEMO says. Window: the entries' pages rounded out to
// 256 MiB (VKA_PIDX_GRAN), at most DM_VK_GMEM_PIDX_MB of records (default 256) and the device's maxStorageBufferRange; entries off the page, past 2^44 or past that
// go to the sink in these shaders ([RED] once, counted). The buffers stay for the process's life; each is logged when made and their sum is in the [reach] line
enum { PX_MAX = 8 };
typedef struct { VkBuffer buf; VkDeviceMemory mem; uint8_t *ptr; uint64_t cap, bytes, fence, gen; int open; vka_pidx s; uint64_t lo[DMSH_GMEM_MAP_ENTRIES], hi[DMSH_GMEM_MAP_ENTRIES], dl[DMSH_GMEM_MAP_ENTRIES]; } px_buf;
static px_buf *g_px[PX_MAX]; static int g_npx, g_px_cur = -1;   // g_px_cur: the buffer of the last generation built
static pthread_mutex_t g_pxmu = PTHREAD_MUTEX_INITIALIZER;
static struct { _Atomic uint64_t full, inc, written, waits, wait_ns, inplace, bytes, base, pages, left; _Atomic int bufs; } g_pxs;
void vka_gmem_retire(uint64_t fence) {   // (vka_exec.c, beside vka_scratch_retire) the buffers the submission being recorded used belong to fence
    pthread_mutex_lock(&g_pxmu);
    for (int i = 0; i < g_npx; i++) if (g_px[i]->open) { g_px[i]->open = 0; g_px[i]->fence = fence; }
    pthread_mutex_unlock(&g_pxmu);
}
static uint64_t px_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec; }
static void px_free(px_buf *x) {
    if (x->ptr) vk.vkUnmapMemory(V.dev, x->mem);
    if (x->buf) vk.vkDestroyBuffer(V.dev, x->buf, NULL);
    if (x->mem) vk.vkFreeMemory(V.dev, x->mem, NULL);
    if (x->bytes) atomic_fetch_sub(&g_pxs.bytes, x->bytes);
    x->buf = VK_NULL_HANDLE; x->mem = VK_NULL_HANDLE; x->ptr = NULL; x->cap = x->bytes = 0; x->s.valid = 0;
}
static int px_alloc(int at, uint64_t recs, int k) {   // a buffer for the table and recs records (an idle one: its old memory is freed)
    px_buf *x = g_px[at]; px_free(x);
    uint64_t n = (uint64_t)DMSH_GMEM_MAP_ENTRIES * DMSH_GMEM_MAP_ENTRY + vka_pidx_bytes(recs - 1);
    n = (n + 0xFFFF) & ~(uint64_t)0xFFFF;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    if (vk.vkCreateBuffer(V.dev, &bi, NULL, &x->buf)) { x->buf = VK_NULL_HANDLE; return 0; }
    VkMemoryRequirements mr; vk.vkGetBufferMemoryRequirements(V.dev, x->buf, &mr);
    uint32_t t = vka_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
    if (t == UINT32_MAX) t = vka_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = t};
    void *p = NULL;
    if (t == UINT32_MAX || vk.vkAllocateMemory(V.dev, &ai, NULL, &x->mem) || vk.vkBindBufferMemory(V.dev, x->buf, x->mem, 0) || vk.vkMapMemory(V.dev, x->mem, 0, VK_WHOLE_SIZE, 0, &p)) {
        x->mem = x->mem ? x->mem : VK_NULL_HANDLE; px_free(x); return 0; }
    x->ptr = p; x->cap = recs; x->bytes = mr.size; x->s = (vka_pidx){.max = DMSH_GMEM_MAP_ENTRIES, .lo = x->lo, .hi = x->hi, .dl = x->dl};
    atomic_fetch_add(&g_pxs.bytes, mr.size); vka_mem_grew(mr.size);
    VLOG("raw global memory page index: buffer %d of %.1f MiB (the table and %llu records: %.1f GiB of guest addresses; %d table entries now), %d buffers %.1f MiB in all",
         at, mr.size / 1048576.0, (unsigned long long)recs, (double)(recs - 1) * 4096 / 1073741824.0, k, atomic_load(&g_pxs.bufs),
         atomic_load(&g_pxs.bytes) / 1048576.0);
    return 1;
}
static int gmem_pidx(VkDescriptorBufferInfo *bi) {
    gmem_sink_check();
    uint64_t gen = vka_mem_gen();
    atomic_fetch_add_explicit(&g_rch.gm, 1, memory_order_relaxed);
    pthread_mutex_lock(&g_pxmu);
    px_buf *x = g_px_cur >= 0 ? g_px[g_px_cur] : NULL;
    if (x && x->s.valid && x->gen == gen) goto bind;
    static uint64_t lo[DMSH_GMEM_MAP_ENTRIES], hi[DMSH_GMEM_MAP_ENTRIES], bda[DMSH_GMEM_MAP_ENTRIES], ilo[DMSH_GMEM_MAP_ENTRIES], ihi[DMSH_GMEM_MAP_ENTRIES], idl[DMSH_GMEM_MAP_ENTRIES];
    int k = vka_mem_chunks(lo, hi, bda, DMSH_GMEM_MAP_ENTRIES - 1);
    // the window's limit: DM_VK_GMEM_PIDX_MB of records and the binding range the device allows (less one rounding step: vka_pidx_take counts from the rounded start)
    uint64_t lim = (uint64_t)VKA_ENV("DM_VK_GMEM_PIDX_MB", 256) << 17, rng = V.sb_range > (uint64_t)DMSH_GMEM_MAP_ENTRIES * DMSH_GMEM_MAP_ENTRY + 32 ? (V.sb_range - (uint64_t)DMSH_GMEM_MAP_ENTRIES * DMSH_GMEM_MAP_ENTRY - 16) / DMSH_VK_GMEM_PIDX_REC - 1 : 0;
    if (rng < lim) lim = rng;
    lim = lim > 2 * VKA_PIDX_GRAN ? lim - VKA_PIDX_GRAN : VKA_PIDX_GRAN;
    vka_pidx_stats st = {0};
    memcpy(ilo, lo, (size_t)k * 8); memcpy(ihi, hi, (size_t)k * 8);
    int n = vka_pidx_take(ilo, ihi, bda, idl, k, VKA_PIDX_GRAN, lim, &st);
    atomic_store(&g_pxs.left, (uint64_t)(k - n));
    if (n < k) VRED_ONCE("raw global memory page index: %d of %d table entries left out (%d not on 4 KiB pages, %d past 2^44, %d past a window of %llu pages): page-indexed shaders "
                         "send their addresses to the sink", k - n, k, st.unaligned, st.high, st.over, (unsigned long long)lim);
    uint64_t wb, wp; vka_pidx_window(ilo, ihi, n, VKA_PIDX_GRAN, &wb, &wp);
    // the buffer: one the GPU is done with (the closest generation first: fewest entries differ), a new one, the oldest one handed to the queue, or the newest in place
    int maxb = VKA_ENV("DM_VK_GMEM_PIDX_BUFS", 4); if (maxb < 1) maxb = 1; if (maxb > PX_MAX) maxb = PX_MAX;
    uint64_t done = vka_done(); int pick = -1;
    for (int i = 0; i < g_npx; i++) if (!g_px[i]->open && g_px[i]->fence <= done && (pick < 0 || g_px[i]->gen > g_px[pick]->gen)) pick = i;
    if (pick < 0 && g_npx < maxb) { px_buf *nb = calloc(1, sizeof *nb); if (nb) { pick = g_npx; g_px[g_npx++] = nb; atomic_store(&g_pxs.bufs, g_npx); } }
    if (pick < 0) {
        for (int i = 0; i < g_npx; i++) if (!g_px[i]->open && (pick < 0 || g_px[i]->fence < g_px[pick]->fence)) pick = i;
        if (pick >= 0) {
            uint64_t t0 = px_ns(); int ok = vka_wait_done(g_px[pick]->fence, vka_hang_ns());
            atomic_fetch_add(&g_pxs.waits, 1); atomic_fetch_add(&g_pxs.wait_ns, px_ns() - t0);
            if (!ok) { VRED_ONCE("raw global memory page index: fence %llu not reached: the draw's index is not bound", (unsigned long long)g_px[pick]->fence); pthread_mutex_unlock(&g_pxmu); return 0; }
        } else { pick = g_px_cur; atomic_fetch_add(&g_pxs.inplace, 1); }   // every buffer belongs to this submission
    }
    if (pick < 0) { pthread_mutex_unlock(&g_pxmu); return 0; }
    x = g_px[pick];
    int r = 0;
    if (x->ptr) r = vka_pidx_update(x->ptr + DMSH_VK_GMEM_PIDX_WORD * 4, x->cap, &x->s, ilo, ihi, idl, n, VKA_PIDX_GRAN, &st);   // (0: nothing written)
    if (!r) {   // no buffer yet, or the window outgrew it: made again with room for 1 GiB more (not one this submission uses: its memory may not go)
        if (x->open || !px_alloc(pick, wp + 1 + 4 * VKA_PIDX_GRAN, k)) { VRED_ONCE("raw global memory page index: no buffer for a window of %llu pages: the draw's index is not bound", (unsigned long long)wp);
            pthread_mutex_unlock(&g_pxmu); return 0; }
        r = vka_pidx_update(x->ptr + DMSH_VK_GMEM_PIDX_WORD * 4, x->cap, &x->s, ilo, ihi, idl, n, VKA_PIDX_GRAN, &st);
        if (!r) { pthread_mutex_unlock(&g_pxmu); return 0; }
    }
    gm_table(x->ptr, lo, hi, bda, k);
    atomic_fetch_add(&g_pxs.full, st.full); atomic_fetch_add(&g_pxs.inc, st.inc); atomic_fetch_add(&g_pxs.written, st.written);
    atomic_store(&g_pxs.base, x->s.base); atomic_store(&g_pxs.pages, x->s.pages);
    x->gen = gen; g_px_cur = pick;   // (a segment that came between the read of gen and the chunks: the next draw sees a new gen, as gmem_map)
bind:
    x->open = 1;
    if (g_gm_short) atomic_fetch_add_explicit(&g_rch.gm_short, 1, memory_order_relaxed);
    // the buffer's whole room, not the window's: an in-place full build may move the window (a lower base, more pages) under draws this submission bound earlier
    // (vka_pidx_take's limit keeps any window within maxStorageBufferRange)
    uint64_t room = (uint64_t)DMSH_GMEM_MAP_ENTRIES * DMSH_GMEM_MAP_ENTRY + vka_pidx_bytes(x->cap - 1);
    if (V.sb_range && room > V.sb_range) room = V.sb_range;
    *bi = (VkDescriptorBufferInfo){x->buf, 0, room};
    pthread_mutex_unlock(&g_pxmu);
    return 1;
}
// The [reach] line (vka_draw_sub_begin, the recording thread): the counts above, the raw global memory table's last build and the span buffers (vka_mem.c), at most
// every 5 s and only when a count moved since the last line
static void reach_line(void) {
    static uint64_t last_ns, last_sum;
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); uint64_t now = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
    if (last_ns && now - last_ns < 5000000000ull) return;
    uint64_t c[13] = {atomic_load(&g_rch.sb), atomic_load(&g_rch.sb_hard), atomic_load(&g_rch.sb_cut), atomic_load(&g_rch.sb_more), atomic_load(&g_rch.sb_min), atomic_load(&g_rch.ub),
                      atomic_load(&g_rch.ub_hard), atomic_load(&g_rch.vb), atomic_load(&g_rch.vb_hard), atomic_load(&g_rch.gm), atomic_load(&g_rch.gm_short), atomic_load(&g_rch.sink), atomic_load(&g_rch.noview)};
    vka_table_stats t; vka_mem_table_stats(&t);
    uint64_t px[10] = {atomic_load(&g_pxs.full), atomic_load(&g_pxs.inc), atomic_load(&g_pxs.written), atomic_load(&g_pxs.waits), atomic_load(&g_pxs.wait_ns), atomic_load(&g_pxs.inplace),
                       atomic_load(&g_pxs.bytes), atomic_load(&g_pxs.base), atomic_load(&g_pxs.pages), atomic_load(&g_pxs.left)};
    uint64_t sum = (uint64_t)t.live + (uint64_t)t.spans * 7 + (uint64_t)t.span_parts * 11; for (int i = 0; i < 13; i++) if (i != 4) sum += c[i] * (i + 2);
    for (int i = 0; i < 10; i++) if (i != 4) sum += px[i] * (i + 17);
    if (sum == last_sum) return;
    last_sum = sum; last_ns = now;
    VLOG("[reach] storage buffers %llu bound: %llu with the bound size past the placed run, %llu runs shorter than the reach asked, %llu of them with placed memory right after (shortest %#llx); "
         "uniform buffers %llu in place, %llu cut; vertex streams %llu, %llu cut; raw global memory table: %d segments, %d entries (%d merged), %d left out (%.1f MiB from %#llx), "
         "bound %llu times, %llu of them while segments were left out; sink stores found %llu times; texture bindings with no view %llu; span buffers %d (%d parts bound, check %s)",
         (unsigned long long)c[0], (unsigned long long)c[1], (unsigned long long)c[2], (unsigned long long)c[3], (unsigned long long)(c[4] == UINT64_MAX ? 0 : c[4]),
         (unsigned long long)c[5], (unsigned long long)c[6], (unsigned long long)c[7], (unsigned long long)c[8], t.live, t.entries, t.merged, t.dropped, t.dropped_bytes / 1048576.0,
         (unsigned long long)t.dropped_lo, (unsigned long long)c[9], (unsigned long long)c[10], (unsigned long long)c[11], (unsigned long long)c[12], t.spans, t.span_parts,
         t.span_check == 1 ? "passed" : t.span_check < 0 ? "FAILED (spans off)" : t.span_check == 3 ? "not made" : "not yet");
    if (atomic_load(&g_pxs.bufs))   // the page index (gmem_pidx): its buffers, window, builds and waits
        VLOG("[reach] raw global memory page index: %d buffers, %.1f MiB; window %#llx + %llu pages (%.2f GiB); %llu full builds, %llu updates, %llu records written; "
             "%llu waits for a buffer (%.1f ms), %llu updates in place; %llu table entries left out",
             atomic_load(&g_pxs.bufs), px[6] / 1048576.0, (unsigned long long)(px[7] << 12), (unsigned long long)px[8], px[8] * 4096.0 / 1073741824.0, (unsigned long long)px[0],
             (unsigned long long)px[1], (unsigned long long)px[2], (unsigned long long)px[3], px[4] / 1e6, (unsigned long long)px[5], (unsigned long long)px[9]);
}
// A compute program's shared memory past the device's limit (DMSH_VK_KIND_SHARED_SPILL; Ryujinx StructuredIr/SharedSpillInfo.cs): one buffer for the process
// (DM_VK_SHARED_SPILL_MB, default 16 MiB, at least one slice), reused by every such dispatch: each dispatch ends with a barrier (vka_dispatch), so the next one finds it free.
// A dispatch binds slices for as many workgroups as it has, or as the buffer holds; workgroup w takes slice w mod the slices bound. A dispatch with more workgroups than
// that runs in parts (vkCmdDispatchBase, a barrier between them), each part's linear indices a contiguous run no longer than the slices, so no two workgroups that may run
// at once share a slice. An indirect dispatch's group count is known only on the GPU: it gets the whole buffer (open: correct while no two workgroups the buffer's slice
// count apart run at once, which in-order workgroup launch makes very likely but Vulkan does not promise)
static vka_span g_spill; static VkDeviceSize g_spill_cap, g_spill_range;
static uint32_t spill_bytes(const vka_shader *cs) {
    for (unsigned i = 0; i < cs->e->bind_count; i++) if (cs->b[i].kind == DMSH_VK_KIND_SHARED_SPILL) return cs->b[i].bytes;
    return 0;
}
static int spill_buf(uint32_t slice) {
    static int failed;
    if (!g_spill.buf && !failed) {   // made once, at the first such dispatch (one try: a failure is logged once and their dispatches are skipped)
        VkDeviceSize cap = (VkDeviceSize)VKA_ENV("DM_VK_SHARED_SPILL_MB", 16) << 20, lim = V.props.limits.maxStorageBufferRange;
        if (lim && cap > lim) cap = lim;
        if (cap < (VkDeviceSize)DMSH_VK_SHARED_MAX * 2) cap = (VkDeviceSize)DMSH_VK_SHARED_MAX * 2;   // at least one slice of any program the build moves (they declare under twice the limit)
        if ((lim && cap > lim) || !vka_static(cap, V.sb_align, &g_spill)) { failed = 1; g_spill.buf = VK_NULL_HANDLE; VRED("no shared memory spill buffer of %llu bytes", (unsigned long long)cap); return 0; }
        g_spill_cap = cap;
        VLOG("shared memory spill buffer: %llu KiB (%llu slices of the first program's %u bytes)", (unsigned long long)(cap >> 10), (unsigned long long)(cap / slice), slice);
    }
    return g_spill.buf && g_spill_cap >= slice;
}
static void spill_dispatch(uint32_t x, uint32_t y, uint32_t z, uint64_t ns) {
    int first = 1;
    #define PART(bx, by, bz, cx, cy, cz) do { if (!first) vka_barrier(g_cb); first = 0; vk.vkCmdDispatchBase(g_cb, bx, by, bz, cx, cy, cz); } while (0)
    #define MIN(a, b) ((a) < (b) ? (a) : (b))
    if ((uint64_t)x * y <= ns) { uint32_t c = (uint32_t)(ns / ((uint64_t)x * y)); for (uint32_t z0 = 0; z0 < z; z0 += c) PART(0, 0, z0, x, y, MIN(c, z - z0)); }
    else if (x <= ns) { uint32_t c = (uint32_t)(ns / x); for (uint32_t zz = 0; zz < z; zz++) for (uint32_t y0 = 0; y0 < y; y0 += c) PART(0, y0, zz, x, MIN(c, y - y0), 1); }
    else for (uint32_t zz = 0; zz < z; zz++) for (uint32_t yy = 0; yy < y; yy++) for (uint32_t x0 = 0; x0 < x; x0 += (uint32_t)ns) PART(x0, yy, zz, MIN((uint32_t)ns, x - x0), 1, 1);
    #undef PART
    #undef MIN
}
static int gmem_log(VkDescriptorBufferInfo *bi) {
    if (vka_tex_gml_take(bi)) return 1;   // read when the submission is done: the textures over the pages its stores wrote are uploaded again
    size_t n = ((size_t)1 + (1u << DMSH_GMEM_LOG_LOG2)) * 4;
    vka_span s; if (!vka_scratch(n, V.sb_align, &s)) return 0;
    memset(s.ptr, 0, n);
    *bi = (VkDescriptorBufferInfo){s.buf, s.off, n}; return 1;
}

// Placeholders for unbound images when the device has no null descriptors: 1x1 images of the shapes a shader may declare, zero (float, sint and uint views)
VkImageView placeholder(uint8_t dim, int storage) {
    static vka_img *im2, *im3, *im1; static VkImageView cache[3][7][3];
    int cls = dim & DMSH_VK_DIM_SINT ? 1 : dim & DMSH_VK_DIM_UINT ? 2 : 0, d = dim & DMSH_VK_DIM_MASK, a = (dim & DMSH_VK_DIM_ARRAY) != 0;
    if ((dim & DMSH_VK_DIM_MS) || d == 0) return VK_NULL_HANDLE;   // 1D shapes: the replay makes no 1D images (vka_tex.c img_create) and the translator declares none
    VkImageViewType vt = d == 0 ? (a ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D) : d == 2 ? VK_IMAGE_VIEW_TYPE_3D : d == 3 ? (a ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE) : a ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    if (cache[storage][vt][cls]) return cache[storage][vt][cls];
    vka_img **pim = d == 2 ? &im3 : d == 0 ? &im1 : &im2;
    if (!*pim) {   // a texture of our own, made through the same path as the game's (zeroed at first use)
        static tex_t t[3]; tex_t *tt = &t[d == 2 ? 1 : d == 0 ? 2 : 0]; memset(tt, 0, sizeof *tt);
        tt->d.target = d == 2 ? NVN_TEXTURE_TARGET_3D : d == 0 ? NVN_TEXTURE_TARGET_1D_ARRAY : NVN_TEXTURE_TARGET_CUBEMAP_ARRAY; tt->d.format = NVN_FORMAT_RGBA8;
        tt->d.width = 1; tt->d.height = 1; tt->d.depth = d == 2 ? 1 : 6; tt->d.levels = 1; tt->d.flags = NVN_TEXTURE_FLAGS_IMAGE;
        nvn_tex_layout(&tt->d, &tt->L); tt->serial = nvn_new_serial();
        *pim = vka_tex_get(tt, g_cb);
        if (!*pim) return VK_NULL_HANDLE;
    }
    static const VkFormat f[3] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SINT, VK_FORMAT_R8G8B8A8_UINT};
    uint32_t na = vt == VK_IMAGE_VIEW_TYPE_CUBE || vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY ? 6 : vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY || vt == VK_IMAGE_VIEW_TYPE_1D_ARRAY ? (*pim)->layers : 1;
    return cache[storage][vt][cls] = vka_view(*pim, f[cls], vt, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, na, NULL);
}

// Where a draw's CPU time goes (DM_VK_PROF=1, logged with the replay statistics): the virtual counter at each phase boundary of vka_draw
static uint64_t g_pt_last, g_pt[9], g_ps[6]; static int g_pt_ph;
// The same, never reset, for DM_VK_RPROF's per-frame differences (vka_rprof.h, vka_exec.c rprof_line)
static uint64_t g_pt_all[VKA_RP_PT], g_ps_all[VKA_RP_PS];
#define PS_ADD(i, dt) do { uint64_t d_ = (dt); g_ps[i] += d_; g_ps_all[i] += d_; } while (0)
void vka_draw_rprof(uint64_t pt[VKA_RP_PT], uint64_t ps[VKA_RP_PS]) { memcpy(pt, g_pt_all, sizeof g_pt_all); memcpy(ps, g_ps_all, sizeof g_ps_all); }
static inline uint64_t cntvct(void) { uint64_t v; __asm__ volatile("mrs %0, cntvct_el0" : "=r"(v)); return v; }
// ---------- Set 0 from the walk ----------
typedef struct { VkWriteDescriptorSet w[512]; VkDescriptorBufferInfo bi[512]; VkDescriptorImageInfo ii[512]; VkBufferView bv[512]; uint32_t n; } writes_t;
static writes_t g_w;
static vka_res g_res[512];   // the NaN scan's own walk (DM_VK_NANSCAN_FS)
// DM_VK_DRAW_DUMP=<vertex shader key, hex>:<folder>[:<draws>] (diagnostic, read at the first draw): the guest bytes the first <draws> (default 4) draws of that vertex shader
// read, as the CPU sees them at recording, one file a buffer in <folder> (made when missing): each buffer binding of every stage from 4 KB before its address to 4 KB past its
// bound size (dd<n>_s<stage>_b<binding>_<address>.bin; unreadable 256-byte pieces as zeros), each driver segment as the replay fills it (dd<n>_s<stage>_drv<bank>.bin), each
// vertex stream and the index buffer; dd.txt lists each file with its address, the draw's arguments and its storage descriptors' offsets. For evaluating one draw's shader
// off the phone (a UI draw that writes no pixels: the inputs its positions come from)
static struct { int init, on, n, max; uint64_t key; char dir[192]; } g_dd;
static void dd_note(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void dd_note(const char *fmt, ...) {
    char p[224]; snprintf(p, sizeof p, "%s/dd.txt", g_dd.dir); FILE *f = fopen(p, "a"); if (!f) return;
    va_list ap; va_start(ap, fmt); vfprintf(f, fmt, ap); va_end(ap); fputc('\n', f); fclose(f);
}
static void dd_file(const char *name, const void *b, size_t n) {
    char p[256]; snprintf(p, sizeof p, "%s/dd%d_%s.bin", g_dd.dir, g_dd.n, name); FILE *f = fopen(p, "wb"); if (!f) { VRED_ONCE("DM_VK_DRAW_DUMP: cannot write %s", p); return; }
    fwrite(b, 1, n, f); fclose(f);
}
static void dd_guest(const char *what, uint64_t addr, uint64_t size) {   // [addr - 4 KB, addr + size + 4 KB)
    if (!addr) return;
    uint64_t a0 = addr > 4096 ? addr - 4096 : 0, n = (addr - a0) + (size ? size : 256) + 4096; if (n > (4u << 20)) n = 4u << 20;
    uint8_t *b = calloc(1, n); if (!b) return; int bad = 0;
    for (uint64_t o = 0; o < n; o += 256) if (!cpu_read(a0 + o, b + o, n - o < 256 ? n - o : 256)) bad++;
    char nm[96]; snprintf(nm, sizeof nm, "%s_%llx", what, (unsigned long long)addr); dd_file(nm, b, n); free(b);
    dd_note("dd%d_%s.bin: %#llx + %#llx, file from %#llx (%llu bytes, %d unreadable 256-byte pieces)", g_dd.n, nm, (unsigned long long)addr, (unsigned long long)size, (unsigned long long)a0, (unsigned long long)n, bad);
}
static void dd_begin(const vka_dstate *d, vka_shader *const sh[VKA_NSTAGE], const p_draw *dw, const p_draw_idx *di) {
    if (!g_dd.init) {
        g_dd.init = 1; const char *e = getenv("DM_VK_DRAW_DUMP"); char *c;
        if (e && (g_dd.key = strtoull(e, &c, 16)) && *c == ':') { snprintf(g_dd.dir, sizeof g_dd.dir, "%s", c + 1); char *m = strchr(g_dd.dir, ':'); if (m) { *m = 0; g_dd.max = atoi(m + 1); } if (g_dd.max <= 0) g_dd.max = 4;
                                                                  mkdir(g_dd.dir, 0777); VLOG("DM_VK_DRAW_DUMP: draws of %016llx into %s", (unsigned long long)g_dd.key, g_dd.dir); }
        else g_dd.key = 0;
    }
    g_dd.on = g_dd.key && g_dd.n < g_dd.max && sh[0] && sh[0]->key == g_dd.key;
    if (!g_dd.on) return;
    if (dw) dd_note("draw dd%d: vs %016llx non-indexed prim %d first %d count %d instances %d base instance %d indirect %#llx", g_dd.n, (unsigned long long)g_dd.key, dw->prim, dw->first, dw->count, dw->instances, dw->base_instance, (unsigned long long)dw->ind);
    else dd_note("draw dd%d: vs %016llx indexed prim %d itype %d count %d base vertex %d instances %d base instance %d ibuf %#llx indirect %#llx", g_dd.n, (unsigned long long)g_dd.key, di->prim, di->itype, di->count, di->base_vertex, di->instances, di->base_instance, (unsigned long long)di->ibuf, (unsigned long long)di->ind);
    (void)d;
}
static void dd_end(const vka_dstate *d, const p_draw_idx *di) {
    if (!g_dd.on) return;
    for (int s = 0; s < 16; s++) if (d->vbuf[s].address) { char w[16]; snprintf(w, sizeof w, "vb%d", s); dd_guest(w, d->vbuf[s].address, d->vbuf[s].size); }
    if (di && di->ibuf) dd_guest("ib", di->ibuf, (uint64_t)di->count << (di->itype == 0 ? 0 : di->itype == 1 ? 1 : 2));
    g_dd.on = 0; g_dd.n++;
}

// A texture ID bound with no view (stage_writes): one line per shader key and binding, the 2D-in-3D / cube fallback and the "has no view" [RED] line alike, for the first
// DM_VK_NOVIEW_LOG (64) distinct pairs (at most 128 kept). A per-line count let one binding that misses every frame (Ocarina of Time's f8965bffbdf51d78 binding 3) take every line
static int noview_new(uint64_t key, unsigned binding) {
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static struct { uint64_t key; unsigned b; } seen[128]; static int n;
    int lim = VKA_ENV("DM_VK_NOVIEW_LOG", 64); if (lim > 128) lim = 128;
    pthread_mutex_lock(&mu);
    int i = 0; while (i < n && !(seen[i].key == key && seen[i].b == binding)) i++;
    int fresh = i == n && n < lim; if (fresh) { seen[n].key = key; seen[n].b = binding; n++; }
    pthread_mutex_unlock(&mu);
    return fresh;
}
static const char *kind_name(int k) {
    return k == DMSH_VK_KIND_CBUF ? "CBUF" : k == DMSH_VK_KIND_SBUF ? "SBUF" : k == DMSH_VK_KIND_SBUF_UBO ? "SBUF_UBO" : k == DMSH_VK_KIND_UBO_PTR ? "UBO_PTR" : "other";
}

// DM_VK_DRAWDUMP_FS=<fragment shader key, hex> (diagnostic, live through ctl env; one-shot: each time the switch is set, the next draw with that fragment shader): the buffers
// that draw's fragment stage binds, read on the CPU at recording (what the GPU reads too: a bound guest buffer is placed memory, the same pages). The log gets one line per
// buffer binding (record kind and slot: c<3 + reg> uniform, s<reg> storage, c0 the driver segment): the guest's binding (NVN address and size) against the host's (the walk's
// address and size, the descriptor's offset and range, the c0 storage table words the shader indexes from: address, size and the DMSH_VK_C0_SSBO_BASE word), and the first
// 4 KiB of each buffer go to <HOME>/drawdump/ld<n>_<key>_<c3|s0|...>.bin (HOME: the app's files folder), the c0 segment as the host filled it too.
// Light lists: the fragment shaders of Ocarina of Time's /Shader/Model.Product.110 (the forest's terrain material, fs 560bba8ac990993d and its variants) loop over a clustered
// light list. From their SPIR-V (library v-a25cc83eed63; the same accesses in 560bba8ac990993d, 5684f5e658eda176, 2be1ceac9d5eddb6, 377081b689f3af21, b0bfb2d3d90d296c,
// 8b9969b6e9e86fda): the cluster of a pixel is (x, y, z), clamped to [0, c3[149].y - 1] x [0, c3[148].x - 1] x [0, c3[149].z - 1] (x, y from its screen position, z from its
// depth); its word in s0 is c3[148].z + ((y * c3[149].y + x) + z * c3[148].y) * c3[148].w; that word is the cluster's light count n, the next its list start l; the loop runs
// min(n, c6[113].x) + 1 times (iteration 0 the directional light), iteration i >= 1 reading light index s0[c3[149].x + l + i - 1] and a 96-byte record at s1 + 96 * index
// (its type word at +16, type 5 skipped). s0 and s1 are read at (c0 table address low word + byte offset - base word) >> 2 into their descriptors. So the dump gives the
// grid, the cap c6[113].x, every cluster's count (min / median / p90 / max, how many exceed the cap, the mean loop trip), how far the counts and lists reach in s0 and the
// records in s1 against the guest's bound sizes and the descriptors' ranges, and the light types read; the counts in ld<n>_<key>_counts.bin (one uint32 a cluster, index
// order). For another shader that binds c3, c6, s0 and s1 the same analysis runs, marked as not verified for its key.
// The same draw's vertex stage is dumped first (lines "drawdump vs", files ld<n>_<fs key>_vs_<name>.bin); when it reads raw global memory (a gmem_map binding: loads
// through the pool table, vk_android_spv.py gmem_flatten / gmem_cache) every non-zero 64-bit word in the first 64 bytes of its storage buffers is looked up as a pointer:
// placed (in the table: the loads read it) or not (the loads read the sink and give 0). Ocarina of Time's forest vertex shaders (b1f06157308e3b22 ...) read their matrix
// palette through the pointer at s0 + 32 (lights/NEXT.md)
uint64_t vka_frame(void);
static const uint64_t g_ld_known[] = {0x560bba8ac990993dull, 0x5684f5e658eda176ull, 0x2be1ceac9d5eddb6ull, 0x377081b689f3af21ull, 0xb0bfb2d3d90d296cull, 0x8b9969b6e9e86fdaull};
static struct { uint32_t gen, done_seq, seq; uint64_t key; int n; } g_ld = {.done_seq = UINT32_MAX};
static int ld_on(uint64_t fs) {
    uint32_t g = atomic_load_explicit(&vka_env_gen, memory_order_relaxed);
    if (g != g_ld.gen) { g_ld.gen = g; char b[64]; g_ld.key = vka_env_str("DM_VK_DRAWDUMP_FS", b, sizeof b) ? strtoull(b, NULL, 16) : 0; g_ld.seq = vka_env_seq("DM_VK_DRAWDUMP_FS"); }
    return g_ld.key && g_ld.key == fs && g_ld.seq != g_ld.done_seq;
}
static void ld_file(int st, const char *name, const void *b, size_t n, char *note, size_t nn) {   // <HOME>/drawdump/ld<n>_<fs key>_[vs_]<name>.bin; note: where, or why not
    const char *h = getenv("HOME"); char dir[256], p[320];
    snprintf(dir, sizeof dir, "%s/drawdump", h && *h ? h : "/data/local/tmp"); mkdir(dir, 0777);
    snprintf(p, sizeof p, "%s/ld%d_%016llx_%s%s.bin", dir, g_ld.n, (unsigned long long)g_ld.key, st == 0 ? "vs_" : "", name);
    FILE *f = fopen(p, "wb"); size_t w = f ? fwrite(b, 1, n, f) : 0; if (f && fclose(f)) w = 0;
    if (w == n) snprintf(note, nn, "%s", p); else snprintf(note, nn, "not written (%s: %s)", p, strerror(errno));
}
static int u32cmp(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return x < y ? -1 : x > y; }
// w0: the descriptor write of r[0] (stage_writes writes one a result, in order); pool: the stage reads the pools (vka_drv_fill)
// st: 0 the vertex stage (dumped first, the one-shot left to the fragment stage's dump that follows in the same draw), 1 the fragment stage
static void ld_dump(const vka_dstate *d, int st, const vka_shader *sh, const vka_res *r, int n, uint32_t w0, int pool) {
    if (st == 1) g_ld.done_seq = g_ld.seq;   // one-shot, also when something below fails
    uint64_t fr = vka_frame(); const uint32_t al = sh->lib->h->ssbo_align; int known = 0, gm = 0;
    for (size_t k = 0; k < sizeof g_ld_known / sizeof *g_ld_known; k++) known |= g_ld_known[k] == sh->key;
    for (unsigned i = 0; i < sh->e->bind_count; i++) gm |= sh->b[i].kind == DMSH_VK_KIND_GMEM_MAP;
    VLOG("drawdump %s %016llx #%d: frame %llu, draw %llu, %d bindings in the %s stage%s%s", st ? "fs" : "vs", (unsigned long long)sh->key, g_ld.n, (unsigned long long)fr, (unsigned long long)g_draws, n,
         st ? "fragment" : "vertex", st && d->rast_discard ? " (rasterizer discard: the stage does not run)" : "", gm ? "; reads raw global memory (the pool table)" : "");
    // the c0 segment as the host fills it (with the base words of a DMSH_VK_F_SSBO_BASE shader)
    static uint8_t c0[DMSH_DRV_SIZE > DMSH_PUSH_SIZE ? DMSH_DRV_SIZE : DMSH_PUSH_SIZE]; memset(c0, 0, sizeof c0);
    vka_drv_fill(&d->b, st, 0, pool, c0); if (sh->e->flags & DMSH_VK_F_SSBO_BASE) vka_c0_bases(&d->b, st, al, ssbo_back, NULL, c0);
    { char note[352]; ld_file(st, "c0", c0, vka_drv_size(0), note, sizeof note); VLOG("drawdump:   c0 (driver segment, %zu bytes as the host fills it%s) -> %s", vka_drv_size(0),
                                                                                (sh->e->flags & DMSH_VK_F_SSBO_BASE) ? ", base words in" : "", note); }
    if (gm) { vka_table_stats t; vka_mem_table_stats(&t);
        VLOG("drawdump:   pool table (last build): %d entries of %d placed segments (%d merged), %d left out%s", t.entries, t.live, t.merged, t.dropped, t.dropped ? " (their loads read the sink)" : ""); }
    int ic3 = -1, ic6 = -1, is0 = -1, is1 = -1;   // the results of c3, c6, s0, s1
    for (int i = 0; i < n; i++) {
        if (r[i].what != VKA_R_BUF) continue;
        const dmsh_vk_bind *b = NULL; for (int j = 0; j < sh->e->bind_count && !b; j++) if (sh->b[j].binding == r[i].binding && sh->b[j].kind == r[i].kind) b = &sh->b[j];
        int reg = b ? b->reg : -1, sb = r[i].kind == DMSH_VK_KIND_SBUF, cb = r[i].kind == DMSH_VK_KIND_CBUF;
        const vka_range *g = reg >= 0 && reg < 16 && (sb || cb) ? (sb ? &d->b.ssbo[st][reg] : &d->b.ubo[st][reg]) : NULL;
        const VkDescriptorBufferInfo *bi = &g_w.bi[w0 + (uint32_t)i];
        char nm[16]; if (cb) snprintf(nm, sizeof nm, "c%d", 3 + reg); else if (sb) snprintf(nm, sizeof nm, "s%d", reg); else snprintf(nm, sizeof nm, "b%u", r[i].binding);
        char tbl[160] = ""; if (sb && reg >= 0 && reg < DMSH_C0_SSBO_SLOTS) { uint32_t w[4]; memcpy(w, c0 + DMSH_C0_SSBO_PS + DMSH_C0_ROW * reg, 16);
            snprintf(tbl, sizeof tbl, "; c0 table slot %d (vec4 %d): address %#x%08x, size %#x, base word %#x", reg, (DMSH_C0_SSBO_PS + DMSH_C0_ROW * reg) / 16, w[1], w[0], w[2], w[3]); }
        uint8_t raw[4096]; memset(raw, 0, sizeof raw); uint64_t ga = g ? g->addr : r[i].addr; int rd = ga && cpu_read(ga, raw, sizeof raw);
        char note[352]; if (rd) ld_file(st, nm, raw, sizeof raw, note, sizeof note); else snprintf(note, sizeof note, "the first 4 KiB not readable");
        VkBuffer pb; VkDeviceSize po; uint64_t placed = ga ? vka_addr_run(ga, UINT64_MAX >> 1, &pb, &po) : 0;
        VLOG("drawdump:   %s (binding %u, %s): guest %#llx + %#llx; host walk %#llx + %#llx, descriptor %s offset %#llx range %#llx; placed run from the guest address %#llx bytes%s; raw -> %s",
             nm, r[i].binding, kind_name(r[i].kind), g ? (unsigned long long)g->addr : 0ull, g ? (unsigned long long)g->size : 0ull, (unsigned long long)r[i].addr, (unsigned long long)r[i].size,
             bi->buffer ? "buffer" : "NULL", (unsigned long long)bi->offset, (unsigned long long)bi->range, (unsigned long long)placed, tbl, note);
        if (cb && reg == 0) ic3 = i; if (cb && reg == 3) ic6 = i; if (sb && reg == 0) is0 = i; if (sb && reg == 1) is1 = i;
        if (gm && sb && rd) {   // pointers the raw loads may follow: each non-zero 64-bit word of the first 64 bytes, placed or not (vka_addr_run: the table lists placed memory)
            char pl[640]; size_t o = 0; pl[0] = 0;
            for (int q = 0; q < 8 && o < sizeof pl; q++) {
                uint64_t v; memcpy(&v, raw + 8 * q, 8); if (!v) continue;
                VkBuffer qb; VkDeviceSize qo; uint64_t run = vka_addr_run(v, UINT64_MAX >> 1, &qb, &qo);
                o += (size_t)snprintf(pl + o, sizeof pl - o, "%s+%d %#llx %s", o ? ", " : "", 8 * q, (unsigned long long)v, run ? "placed" : "NOT PLACED");
                if (run && o < sizeof pl) o += (size_t)snprintf(pl + o, sizeof pl - o, " (run %#llx)", (unsigned long long)run);
            }
            VLOG("drawdump:   %s as pointers (raw global memory: a load through one that is not placed reads the sink): %s", nm, o ? pl : "all zero");
        }
    }
    if (st == 0) return;
    if (ic3 < 0 || ic6 < 0 || is0 < 0 || is1 < 0) { VLOG("drawdump:   no light-list analysis: the stage binds %s%s%s%s", ic3 < 0 ? "no c3 " : "", ic6 < 0 ? "no c6 " : "", is0 < 0 ? "no s0 " : "", is1 < 0 ? "no s1" : ""); g_ld.n++; return; }
    const vka_range *g3 = &d->b.ubo[st][0], *g6 = &d->b.ubo[st][3], *g0 = &d->b.ssbo[st][0], *g1 = &d->b.ssbo[st][1];
    uint32_t c3w[8] = {0}, cap = 0; int ok3 = cpu_read(g3->addr + 148 * 16, c3w, sizeof c3w), ok6 = cpu_read(g6->addr + 113 * 16, &cap, 4);
    const VkDescriptorBufferInfo *b3 = &g_w.bi[w0 + (uint32_t)ic3], *b6 = &g_w.bi[w0 + (uint32_t)ic6], *b0 = &g_w.bi[w0 + (uint32_t)is0], *b1 = &g_w.bi[w0 + (uint32_t)is1];
    uint32_t Y = c3w[0], slice = c3w[1], off = c3w[2], stride = c3w[3], lbase = c3w[4], X = c3w[5], Z = c3w[6];
    VLOG("drawdump:   light lists (%s): c3[148] = %u %u %u %u, c3[149] = %u %u %u %u%s; grid x %u, y %u, z %u, slice stride %u clusters, word %u + cluster * %u, lists from word %u; "
         "cap c6[113].x = %d (%#x)%s; c3 bound %#llx bytes (needs %#x; descriptor range %#llx), c6 bound %#llx (needs %#x; descriptor range %#llx)", known ? "layout from this shader's SPIR-V" : "layout of Model.Product.110, NOT verified for this key",
         c3w[0], c3w[1], c3w[2], c3w[3], c3w[4], c3w[5], c3w[6], c3w[7], ok3 ? "" : " (NOT READABLE)", X, Y, Z, slice, off, stride, lbase, (int32_t)cap, cap, ok6 ? "" : " (NOT READABLE)",
         (unsigned long long)g3->size, 150 * 16, (unsigned long long)b3->range, (unsigned long long)g6->size, 114 * 16, (unsigned long long)b6->range);
    uint64_t ncl = (uint64_t)X * Y * Z;
    // (stride 0: every cluster reads the same count and list start, s0 word off: Ocarina of Time's 1x1x1 grid, c3[148] = 1 1 0 0)
    if (!ok3 || !ok6 || !X || !Y || !Z || X > 4096 || Y > 4096 || Z > 4096 || ncl > (1u << 22)) { VLOG("drawdump:   light lists: grid not usable (%u x %u x %u = %llu clusters%s): no counts read", X, Y, Z, (unsigned long long)ncl, ok3 && ok6 ? "" : ", c3 or c6 not readable"); g_ld.n++; return; }
    if (!stride) VLOG("drawdump:   light lists: cluster stride 0: every cluster reads s0 word %u (count) and %u (list start)", off, off + 1);
    // the counts and list starts: one read of s0 from its first word to the last cluster's second
    uint64_t maxidx = (uint64_t)(Y - 1) * X + (X - 1) + (uint64_t)(Z - 1) * slice, wlast = off + maxidx * stride + 2;
    if (wlast > (64u << 20) / 4) { VLOG("drawdump:   light lists: s0 reach %llu words is past 64 MiB: not read", (unsigned long long)wlast); g_ld.n++; return; }
    uint32_t *s0 = malloc((size_t)wlast * 4), *cnt = malloc((size_t)ncl * 4), *srt = malloc((size_t)ncl * 4);
    if (!s0 || !cnt || !srt || !cpu_read(g0->addr, s0, (size_t)wlast * 4)) { VLOG("drawdump:   light lists: s0 %#llx + %#llx not readable (or no memory)", (unsigned long long)g0->addr, (unsigned long long)wlast * 4); free(s0); free(cnt); free(srt); g_ld.n++; return; }
    uint64_t over = 0, zero = 0, trips = 0, unwalked = 0, lmax_w = 0, rec_max = 0, nlist = 0, bad_idx = 0; uint32_t type_n[8] = {0}, type_other = 0, idx_max = 0; uint64_t k = 0;
    uint64_t s1n = g1->size ? g1->size / 96 : 0;
    for (uint32_t z = 0; z < Z; z++) for (uint32_t y = 0; y < Y; y++) for (uint32_t x = 0; x < X; x++, k++) {
        uint64_t idx = (uint64_t)y * X + x + (uint64_t)z * slice, w = off + idx * stride; uint32_t c = s0[w], l = s0[w + 1];
        cnt[k] = c; over += (int32_t)c > (int32_t)cap; zero += !c;
        int32_t trip = ((int32_t)c < (int32_t)cap ? (int32_t)c : (int32_t)cap) + 1; if (trip < 0) trip = 0; trips += (uint64_t)trip;
        if (trip > 1 && (uint64_t)lbase + l + (uint64_t)(trip - 1) > lmax_w) lmax_w = (uint64_t)lbase + l + (uint64_t)(trip - 1);   // (the reach from the count, walked or not)
        // the light indices this cluster's loop reads, and their records' type words: walked up to 256 a cluster and 1M in all (a garbage cap and count, the case looked
        // for, would be 2^31 reads on the recording thread); the rest counted as not walked
        for (int32_t i = 1; i < trip; i++) {
            if (i > 256 || nlist >= (1u << 20)) { unwalked += (uint64_t)(trip - i); break; }
            uint64_t lw = (uint64_t)lbase + l + (uint64_t)(i - 1); nlist++;
            uint32_t li; if (lw < wlast) li = s0[lw]; else if (!cpu_read(g0->addr + lw * 4, &li, 4)) { bad_idx++; continue; }
            if (li > idx_max) idx_max = li; if ((uint64_t)li * 96 + 96 > rec_max) rec_max = (uint64_t)li * 96 + 96;
            uint32_t ty; if (cpu_read(g1->addr + (uint64_t)li * 96 + 16, &ty, 4)) { if (ty < 8) type_n[ty]++; else type_other++; } else bad_idx++;
        }
    }
    memcpy(srt, cnt, (size_t)ncl * 4); qsort(srt, (size_t)ncl, 4, u32cmp);
    { char note[352]; ld_file(st, "counts", cnt, (size_t)ncl * 4, note, sizeof note);
      VLOG("drawdump:   light counts over %llu clusters: min %u, median %u, p90 %u, p99 %u, max %u; %llu zero, %llu above the cap %d; mean loop trip %.2f (min(count, cap) + 1); counts -> %s",
           (unsigned long long)ncl, srt[0], srt[ncl / 2], srt[ncl * 9 / 10], srt[ncl * 99 / 100], srt[ncl - 1], (unsigned long long)zero, (unsigned long long)over, (int32_t)cap, (double)trips / ncl, note); }
    // how far the shader reads, against what the guest bound and what the descriptors cover. The GPU's element is (address low word + byte offset - base word) >> 2 from the
    // descriptor's start, so the descriptor covers byte offsets up to range - (address - start); the start is the base word's address (vka_c0_bases)
    uint32_t w0s[4], w1s[4]; memcpy(w0s, c0 + DMSH_C0_SSBO_PS, 16); memcpy(w1s, c0 + DMSH_C0_SSBO_PS + DMSH_C0_ROW, 16);
    uint64_t r0 = (uint64_t)(lmax_w > wlast ? lmax_w : wlast) * 4, d0 = (uint32_t)(w0s[0] - w0s[3]), d1 = (uint32_t)(w1s[0] - w1s[3]);
    VLOG("drawdump:   reach: s0 counts and lists to byte %#llx (guest bound %#llx; descriptor range %#llx from %#llx before the address: covers %#llx) %s; s1 records to byte %#llx "
         "(highest light index %u of the entries read; guest bound %#llx = %llu records; descriptor covers %#llx) %s; %llu list entries read, %llu not walked (past 256 a cluster or 1M), %llu reads not readable",
         (unsigned long long)r0, (unsigned long long)g0->size, (unsigned long long)b0->range, (unsigned long long)d0, (unsigned long long)(b0->range > d0 ? b0->range - d0 : 0),
         r0 <= (b0->range > d0 ? b0->range - d0 : 0) ? "inside" : "PAST THE DESCRIPTOR", (unsigned long long)rec_max, idx_max, (unsigned long long)g1->size, (unsigned long long)s1n,
         (unsigned long long)(b1->range > d1 ? b1->range - d1 : 0), rec_max <= (b1->range > d1 ? b1->range - d1 : 0) ? "inside" : "PAST THE DESCRIPTOR", (unsigned long long)nlist, (unsigned long long)unwalked, (unsigned long long)bad_idx);
    { char tl[160]; size_t o = 0; for (int t = 0; t < 8; t++) if (type_n[t]) o += (size_t)snprintf(tl + o, sizeof tl - o, " %d: %u", t, type_n[t]); if (!o) snprintf(tl, sizeof tl, " none");
      VLOG("drawdump:   light record types read (type word at +16; 5 is skipped by the loop):%s, other %u; base words: s0 %#x (address low %#x, %#llx before it), s1 %#x (address low %#x, %#llx before it)",
           tl, type_other, w0s[3], w0s[0], (unsigned long long)d0, w1s[3], w1s[0], (unsigned long long)d1); }
    free(s0); free(cnt); free(srt); g_ld.n++;
}
// One stage's results as descriptor writes. 0 = the draw cannot run (logged)
static int stage_writes(const vka_dstate *d, int st, const vka_shader *sh, const vka_res *r, int n, int pool, c0_at *c0) {
    for (int i = 0; i < n; i++) {
        if (g_w.n == 512) { VRED_ONCE("more than 512 descriptors in one draw"); return 0; }
        uint32_t k = g_w.n;
        VkWriteDescriptorSet *w = &g_w.w[k]; VkDescriptorBufferInfo *bi = &g_w.bi[k]; VkDescriptorImageInfo *ii = &g_w.ii[k];
        *w = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = r[i].binding, .dstArrayElement = r[i].elem, .descriptorCount = 1, .descriptorType = (VkDescriptorType)r[i].desc};
        int ok = 1;
        switch (r[i].desc) {
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: {
            int storage = r[i].desc == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            switch (r[i].what) {
            case VKA_R_BUF: { uint64_t back = 0, own = 0;
                ok = storage ? ssbo_desc(r[i].addr, r[i].size, r[i].c0base, bi, &back, &own) : ubo_desc(r[i].addr, r[i].size, bi);
                // the shader may store into it: its bound size, or with none the run in its own segment (a span buffer's run over 100+ MiB would re-upload every texture there)
                if (ok && storage && (r[i].flags & DMSH_VK_BF_RW)) vka_tex_gw_opaque(r[i].addr, r[i].size && r[i].size < bi->range - back ? r[i].size : own);
                // DM_VK_REACH_LOG=<n>: the first n bindings cut short (the bound size past the run, or placed memory right after the run)
                if (ok && g_cut.what) { static _Atomic int nl; if (atomic_fetch_add(&nl, 1) < VKA_ENV("DM_VK_REACH_LOG", 0))
                    VLOG("[reach] %s %016llx stage %d binding %u (%s): %#llx+%#llx bound to %#llx of %#llx asked (%s)", st == 5 ? "dispatch" : "draw", (unsigned long long)sh->key, st, r[i].binding,
                         kind_name(r[i].kind), (unsigned long long)r[i].addr, (unsigned long long)r[i].size, (unsigned long long)g_cut.run, (unsigned long long)g_cut.want,
                         g_cut.what == 1 ? "the bound size reaches past it" : "placed memory follows"); }
                break; }
            case VKA_R_C1: if (sh->c1_len) *bi = (VkDescriptorBufferInfo){sh->c1.buf, sh->c1.off, (sh->c1_len + 15) & ~(VkDeviceSize)15}; else ok = 0; break;
            case VKA_R_SUPPORT:
                if (st == 1 && (sh->e->flags & DMSH_VK_F_ALPHA_TEST) && d->color.alpha_func && d->color.alpha_func != NVN_ALPHA_FUNC_ALWAYS) {   // word 0: the alpha test's reference
                    vka_span s;
                    if (g_drv_memo && g_sm_ref.valid && !memcmp(&g_sm_ref.ref, &d->alpha_ref, 4)) s = g_sm_ref.s;   // the same reference as the last alpha-tested draw: its copy (DM_VK_DRV_MEMO)
                    else { if (!vka_scratch(VKA_SUPPORT_SIZE, V.ub_align, &s)) return 0;
                           memcpy(s.ptr, g_support.ptr, VKA_SUPPORT_SIZE); memcpy(s.ptr, &d->alpha_ref, 4);
                           g_sm_ref.valid = 1; memcpy(&g_sm_ref.ref, &d->alpha_ref, 4); g_sm_ref.s = s; }
                    *bi = (VkDescriptorBufferInfo){s.buf, s.off, VKA_SUPPORT_SIZE};
                } else *bi = (VkDescriptorBufferInfo){g_support.buf, g_support.off, VKA_SUPPORT_SIZE};
                break;
            case VKA_R_DRV: { uint64_t td_ = g_tm ? cntvct() : 0;
                if (!drv_seg(d, st, r[i].bank, pool, bi, c0, (sh->e->flags & DMSH_VK_F_SSBO_BASE) ? sh->lib->h->ssbo_align : 0)) return 0;
                if (g_tm) PS_ADD(3, cntvct() - td_);
                break; }
            case VKA_R_GMEM_MAP: { int px_ = (r[i].flags & DMSH_VK_BF_PIDX) != 0; uint64_t tg_ = VKA_RP_T0(); ok = px_ ? gmem_pidx(bi) : gmem_map(bi); VKA_RP_ADD(px_ ? VKA_RPX_PIDX : VKA_RPX_GMAP, tg_); break; }   // (an older library's records: no flag, the table alone)
            case VKA_R_GMEM_LOG: ok = gmem_log(bi); break;
            case VKA_R_SHARED_SPILL: ok = st == 5 && g_spill_range && g_spill.buf; if (ok) *bi = (VkDescriptorBufferInfo){g_spill.buf, g_spill.off, g_spill_range}; break;
            default: ok = 0; break;
            }
            if (!ok) { if (V.null_desc) *bi = (VkDescriptorBufferInfo){VK_NULL_HANDLE, 0, VK_WHOLE_SIZE}; else *bi = (VkDescriptorBufferInfo){storage ? g_sink.buf : g_zero.buf, storage ? g_sink.off : g_zero.off, 65536}; }
            // DM_VK_DISPATCH_LOG=2: each buffer a dispatch binds (what the shader reads at binding offset + its own index; ok 0: the null, zero or sink buffer instead)
            if (st == 5 && VKA_ENV("DM_VK_DISPATCH_LOG", 0) >= 2)
                VLOG("dispatch %016llx binding %u: buffer kind %d at %#llx + %#llx: %s, offset %#llx, range %#llx", (unsigned long long)sh->key, r[i].binding, (int)r[i].what,
                     (unsigned long long)r[i].addr, (unsigned long long)r[i].size, ok ? "bound" : "NOT BOUND", (unsigned long long)bi->offset, (unsigned long long)bi->range);
            // DM_VK_DRAW_LOG=1: each buffer a draw's stage binds, with its first words as the CPU sees them at recording (zeros here and a shader that reads its constants: content missing)
            if (st != 5 && VKA_ENV("DM_VK_DRAW_LOG", 0) >= 1) { uint32_t wd[8] = {0}; int rd = r[i].what == VKA_R_BUF && r[i].addr && cpu_read(r[i].addr, wd, sizeof wd);
                VLOG("draw stage %d %016llx binding %u: buffer kind %d at %#llx + %#llx: %s; words %s %08x %08x %08x %08x %08x %08x %08x %08x", st, (unsigned long long)sh->key, r[i].binding, (int)r[i].what,
                     (unsigned long long)r[i].addr, (unsigned long long)r[i].size, ok ? "bound" : "NOT BOUND", rd ? "" : "(not read)", wd[0], wd[1], wd[2], wd[3], wd[4], wd[5], wd[6], wd[7]); }
            if (g_dd.on) { char w_[32];   // DM_VK_DRAW_DUMP: the bytes behind this binding
                if (r[i].what == VKA_R_BUF) { snprintf(w_, sizeof w_, "s%d_b%u", st, r[i].binding); dd_guest(w_, r[i].addr, r[i].size);
                    dd_note("  stage %d binding %u: %s at %#llx + %#llx, descriptor offset %#llx range %#llx", st, r[i].binding, storage ? "storage" : "uniform", (unsigned long long)r[i].addr, (unsigned long long)r[i].size, (unsigned long long)bi->offset, (unsigned long long)bi->range); }
                else if (r[i].what == VKA_R_DRV) { static uint8_t t_[DMSH_DRV_SIZE > DMSH_PUSH_SIZE ? DMSH_DRV_SIZE : DMSH_PUSH_SIZE];
                    vka_drv_fill(&d->b, st, r[i].bank, pool, t_); if (r[i].bank == 0 && (sh->e->flags & DMSH_VK_F_SSBO_BASE)) vka_c0_bases(&d->b, st, sh->lib->h->ssbo_align, ssbo_back, NULL, t_);
                    snprintf(w_, sizeof w_, "s%d_drv%d", st, r[i].bank); dd_file(w_, t_, vka_drv_size(r[i].bank)); dd_note("  stage %d binding %u: driver bank %d -> dd%d_%s.bin", st, r[i].binding, r[i].bank, g_dd.n, w_); } }
            w->pBufferInfo = bi;
            break; }
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE: case VK_DESCRIPTOR_TYPE_SAMPLER: case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
            int storage = r[i].desc == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; uint64_t ti_ = g_tm ? cntvct() : 0;
            VkImageView v = VK_NULL_HANDLE; VkSampler s = VK_NULL_HANDLE; VkImageLayout lay = VK_IMAGE_LAYOUT_GENERAL;
            if (r[i].desc != VK_DESCRIPTOR_TYPE_SAMPLER) {
                int tid = r[i].what == VKA_R_TEX || r[i].what == VKA_R_IMG ? r[i].tid : 0;
                if (tid) v = vka_tex_id_view_rec(tid, g_cb, r[i].dim, storage, &lay);
                // DM_VK_DISPATCH_LOG=2: each texture and image a dispatch binds, with its serial (nxctl.sh texshot <serial> then shows what a step read or wrote)
                if (st == 5 && tid && VKA_ENV("DM_VK_DISPATCH_LOG", 0) >= 2) { char why[320]; vka_tex_id_desc(tid, why, sizeof why);
                    VLOG("dispatch %016llx binding %u: %s texture ID %d (%s)", (unsigned long long)sh->key, r[i].binding, storage ? "image" : "sampled", tid, why); }
                // DM_VK_DRAW_LOG=1 (live): the same for each draw's stages (the text of a UI: which texture a draw samples, for nxctl.sh texshot)
                if (st != 5 && VKA_ENV("DM_VK_DRAW_LOG", 0) >= 1) { char why[320]; why[0] = 0; if (tid) vka_tex_id_desc(tid, why, sizeof why);
                    VLOG("draw stage %d %016llx binding %u: %s texture ID %d (%s)%s", st, (unsigned long long)sh->key, r[i].binding, storage ? "image" : "sampled", tid, tid ? why : "none", v ? "" : " NO VIEW"); }
                // A 2D texture in a 3D or cube slot: what the game leaves bound where a shader's branch does not sample (DKB's materials keep a 1x1 2D texture in their volume slot 12).
                // It reads as unbound, zeros, as nvn_mtl.m's zero 3D / zero cube fallback does; noted once. Other mismatches are reported
                int dd = r[i].dim & DMSH_VK_DIM_MASK, tt = tid && !v ? vka_tex_id_target(tid) : -1;
                if (tid && !v) atomic_fetch_add_explicit(&g_rch.noview, 1, memory_order_relaxed);
                if (tid && !v && (dd == 2 || dd == 3) && (tt == NVN_TEXTURE_TARGET_2D || tt == NVN_TEXTURE_TARGET_2D_ARRAY)) {
                    if (noview_new(sh->key, r[i].binding)) { char why[320]; vka_tex_id_desc(tid, why, sizeof why);
                        VLOG("%s binding fallback: shader %016llx binding %u, texture ID %d is a 2D texture (%s): it reads as zeros (the shader may leave it unused)", dd == 2 ? "3D" : "cube", (unsigned long long)sh->key, r[i].binding, tid, why); }
                } else if (tid && !v) { if (noview_new(sh->key, r[i].binding)) { char why[320]; vka_tex_id_desc(tid, why, sizeof why);
                    VRED("shader %016llx binding %u: texture ID %d has no view of shape %#x%s: it reads as unbound (%s)", (unsigned long long)sh->key, r[i].binding, tid, r[i].dim, storage ? " (storage)" : "", why); } }
                if (!v && !V.null_desc) v = placeholder(r[i].dim, storage);
                if (!v && !V.null_desc) { VRED_ONCE("an unbound image of shape %#x and no null descriptors: the draw is skipped", r[i].dim); return 0; }
            }
            if (r[i].desc == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || r[i].desc == VK_DESCRIPTOR_TYPE_SAMPLER) {
                s = r[i].sid ? vka_samp_id_rec(r[i].sid) : VK_NULL_HANDLE;
                if (!s) s = vka_samp_default();
            }
            *ii = (VkDescriptorImageInfo){s, v, lay};
            w->pImageInfo = ii; if (g_tm) { PS_ADD(4, cntvct() - ti_); g_ps[5]++; }
            break; }
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            if (r[i].what != VKA_R_NULL) VRED_ONCE("texel buffers are not on Vulkan yet (shader %016llx binding %u): bound as null", (unsigned long long)sh->key, r[i].binding);
            if (!V.null_desc) { VRED_ONCE("a texel buffer binding and no null descriptors: the draw is skipped"); return 0; }
            g_w.bv[k] = VK_NULL_HANDLE; w->pTexelBufferView = &g_w.bv[k];
            break;
        default: VRED_ONCE("descriptor type %u in a binding record", r[i].desc); return 0;
        }
        g_w.n++;
    }
    return 1;
}
// Each stage's bindings, walked once a draw or dispatch (walk_all): the images first used are made from them (make_images), then set 0 is written from them (bind_sets)
static struct { int n[VKA_NSTAGE], pool[VKA_NSTAGE]; char why[VKA_NSTAGE][200]; vka_res r[VKA_NSTAGE][512]; } g_wk;
static void walk_all(const vka_dstate *d, vka_shader *const sh[VKA_NSTAGE]) {
    uint64_t t0_ = g_tm ? cntvct() : 0;
    g_ssbo_exact = VKA_ENV("DM_VK_SSBO_EXACT", 0);
    for (int st = 0; st < VKA_NSTAGE; st++) {
        vka_shader *s = sh[st]; g_wk.n[st] = 0; g_wk.pool[st] = 0; if (!s) continue;
        g_wk.n[st] = vka_walk(&d->b, st, s->e, s->b, s->lib->h->ssbo_align, V.sb_align, rd_cb, NULL, g_wk.r[st], 512, &g_wk.pool[st], g_wk.why[st], sizeof g_wk.why[st]);
    }
    if (g_tm) PS_ADD(0, cntvct() - t0_);
}
// Writes set 0 from the walk (walk_all, just before) and binds it (and set 1 for shaders that read the pools). c0: where the compute stage's c0 went (an indirect dispatch copies its counts there)
// The last push descriptor set per bind point, as the words that decide it (layout, each write's binding, type and count, each descriptor's info): a draw whose writes
// are the same word for word skips its push (DM_VK_PUSH_DEDUP=0: every draw pushes). Pushed descriptors stay bound across draws and pipeline binds while the layout stays
// the same (Vulkan's pipeline layout compatibility); a new command buffer or a set bound in their place forgets them (vka_dyn_invalidate, bind_sets)
static struct { uint64_t *w; size_t n, cap; } g_pd[2];
static uint64_t g_pd_skip, g_pd_push, g_pd_skip_all, g_pd_push_all, g_pd_delta_w, g_pd_delta_all;
// The replay statistics' line: driver segments bound again (DM_VK_DRV_MEMO) and pushes skipped (DM_VK_PUSH_DEDUP) since the last line
void vka_drv_memo_stats(char *b, size_t n) {
    static uint64_t h0, m0, s0, p0;
    uint64_t h = g_dm_hit - h0, m = g_dm_miss - m0, sk = g_pd_skip_all - s0, pu = g_pd_push_all - p0;
    h0 = g_dm_hit; m0 = g_dm_miss; s0 = g_pd_skip_all; p0 = g_pd_push_all;
    if (!h && !m && !sk && !pu) { if (n) b[0] = 0; return; }
    snprintf(b, n, "descriptors: driver segments %.0f %% the same bytes as the draw before (bound again: DM_VK_DRV_MEMO); push descriptor sets %.0f %% skipped as the same (DM_VK_PUSH_DEDUP); "
             "partial pushes (DM_VK_PUSH_DELTA) wrote %.0f %% of their descriptors", h + m ? 100.0 * h / (double)(h + m) : 0.0, sk + pu ? 100.0 * sk / (double)(sk + pu) : 0.0,
             g_pd_delta_all ? 100.0 * g_pd_delta_w / (double)g_pd_delta_all : 0.0);
    g_pd_delta_w = g_pd_delta_all = 0;
}
static VkWriteDescriptorSet g_dw[512]; static uint32_t g_dwn;   // the writes that changed (push_same returns 2)
static int push_same(VkPipelineBindPoint bp, VkPipelineLayout layout) {
    static uint64_t *buf; static size_t cap;
    size_t need = 2 + (size_t)g_w.n * 5, k = 0;
    if (need > cap) { cap = need * 2; buf = realloc(buf, cap * sizeof *buf); if (!buf) { cap = 0; return 0; } }
    buf[k++] = (uint64_t)layout; buf[k++] = g_w.n;
    for (uint32_t i = 0; i < g_w.n; i++) {
        const VkWriteDescriptorSet *w = &g_w.w[i];
        buf[k++] = (uint64_t)w->dstBinding | (uint64_t)w->dstArrayElement << 16 | (uint64_t)w->descriptorType << 32 | (uint64_t)w->descriptorCount << 48;
        if (w->descriptorCount != 1) return (g_pd[bp == VK_PIPELINE_BIND_POINT_COMPUTE].n = 0), 0;   // (the walk writes one descriptor per write)
        if (w->pBufferInfo) { buf[k++] = (uint64_t)w->pBufferInfo->buffer; buf[k++] = w->pBufferInfo->offset; buf[k++] = w->pBufferInfo->range; }
        else if (w->pImageInfo) { buf[k++] = (uint64_t)w->pImageInfo->sampler; buf[k++] = (uint64_t)w->pImageInfo->imageView; buf[k++] = w->pImageInfo->imageLayout; }
        else if (w->pTexelBufferView) { buf[k++] = (uint64_t)*w->pTexelBufferView; buf[k++] = 0; buf[k++] = 0; }
        else { buf[k++] = 1; buf[k++] = 1; buf[k++] = 1; }
    }
    int c = bp == VK_PIPELINE_BIND_POINT_COMPUTE;
    if (VKA_ENV("DM_VK_PUSH_DEDUP", 1) && g_pd[c].n == k && !memcmp(g_pd[c].w, buf, k * sizeof *buf)) { g_pd_skip++; g_pd_skip_all++; return 1; }
    // DM_VK_PUSH_DELTA=1 (live, default 0 until phone-checked): the same layout and the same writes in the same order as the last push, some of them with other descriptors:
    // only those are pushed. Vulkan lets push descriptors be updated incrementally; the others stay as the last push left them (until the set is disturbed: a new command
    // buffer or part, a set bound in its place, which forget g_pd)
    g_dwn = 0;
    if (VKA_ENV("DM_VK_PUSH_DELTA", 0) && g_pd[c].n == k && g_pd[c].w[0] == buf[0] && g_pd[c].w[1] == buf[1]) {
        int ok = 1;
        for (uint32_t i = 0; i < g_w.n && ok; i++) {
            const uint64_t *a = g_pd[c].w + 2 + 4 * (size_t)i, *b = buf + 2 + 4 * (size_t)i;
            if (a[0] != b[0]) ok = 0; else if (memcmp(a + 1, b + 1, 3 * sizeof *a)) g_dw[g_dwn++] = g_w.w[i];
        }
        if (!ok) g_dwn = 0;
    }
    if (k > g_pd[c].cap) { g_pd[c].cap = k * 2; g_pd[c].w = realloc(g_pd[c].w, g_pd[c].cap * sizeof *buf); if (!g_pd[c].w) { g_pd[c].cap = g_pd[c].n = 0; return 0; } }
    memcpy(g_pd[c].w, buf, k * sizeof *buf); g_pd[c].n = k; g_pd_push++; g_pd_push_all++;
    if (g_dwn) { g_pd_delta_w += g_dwn; g_pd_delta_all += g_w.n; return 2; }
    return 0;
}
// Draws and dispatches whose shaders read the texture pool through the bindless table (set 1: any registered texture, indexed on the GPU), for the ctl stats line
static _Atomic uint64_t g_bs_all, g_bs_pool;
uint64_t now_ns(void);   // (the host's monotonic clock: the pool scan's memo)
uint64_t vka_frame(void);
void vka_draw_pool_stats(uint64_t *all, uint64_t *pool) { *all = g_bs_all; *pool = g_bs_pool; }
// Lazy images (vka_tex.c write_tables): the texture IDs a draw or dispatch whose shaders read the texture pool may index get their images and slots before it, from the walk
// (walk_all, just before). Those its storage buffers name, of every stage (a handle one stage reads may reach the stage that reads the pool as a varying), or every registered
// one when a stage that reads the pool has no storage buffer or more than POOL_SCAN bytes of them (the handles then come from elsewhere, or reading them costs too much).
// Returns 0 (nothing to make), 1 (texture IDs marked) or 2 (every registered one): the caller ends the render pass, then writes them (vka_tex_pool_write), since making an
// image records uploads and barriers. A buffer already scanned in this frame is not read again (the same material table is bound by draw after draw): a handle the game
// writes into it meanwhile names a slot that still reads as null (zeros) until the next frame's scan. DM_VK_POOL_RESCAN_MS=<ms> scans a buffer at most that often instead
// (250: as before, up to a quarter of a second of null reads, two or three frames at 10 fps). IDs a rescan of a memoized buffer queued (named since its last scan, or
// deferred again by image eviction) are counted: draws in between may have read them as null (the pool-reading draws line, the first 8 logged)
// Texture-use feedback (vka_tex.c, DM_VK_TEXFB): a stage that reads the texture pool (binding DMSH_VK_KIND_TEX_POOL, not only the sampler pool) reports the IDs it indexes
// when its entry has DMSH_VK_F_TEXFB; a draw with such a stage without it is blind (its scan keeps today's images). fb: the scan's IDs may wait for the GPU's report
static int sh_texpool(const vka_shader *s) { for (unsigned i = 0; i < s->e->bind_count; i++) if (s->b[i].kind == DMSH_VK_KIND_TEX_POOL) return 1; return 0; }
static int pool_scan(const vka_dstate *d, vka_shader *const sh[VKA_NSTAGE]) {
    int any_pool = 0, texpool = 0, blind = 0;
    for (int st = 0; st < VKA_NSTAGE; st++) if (sh[st] && g_wk.pool[st] && g_wk.n[st] >= 0 && !(st == 1 && d->rast_discard)) {
        any_pool = 1;
        if (sh_texpool(sh[st])) { texpool = 1; if (!(sh[st]->e->flags & DMSH_VK_F_TEXFB)) blind = 1; }
    }
    if (texpool) vka_tex_fb_draw(!blind);
    int fb = !blind && vka_shader_texfb(NULL);
    if (!any_pool || !vka_tex_pool_lazy()) return 0;   // (after the first draw that reads every registered texture, or with lazy images off, there is nothing to defer)
    enum { POOL_SCAN = 64 << 20 };
    static struct { uint64_t addr, size, t, f; } memo[16]; static unsigned memo_next;
    static uint64_t n_late, n_rescan;   // texture IDs queued by rescans of memoized buffers, and those rescans
    uint64_t tnow = now_ns(), fr = vka_frame(); int rescan_ms = VKA_ENV("DM_VK_POOL_RESCAN_MS", 0);
#define VLOG_ONCE_POOL(fmt, ...) do { static _Atomic int nl_; if (atomic_fetch_add(&nl_, 1) < 4) VLOG("texture pool: shader %016llx (stage %d) reads it, every registered texture gets its image: " fmt, (unsigned long long)sh[st]->key, st, __VA_ARGS__); } while (0)
    int all = 0, marked = 0; uint64_t scanned = 0;
    for (int st = 0; st < VKA_NSTAGE && !all; st++) if (sh[st] && g_wk.pool[st] && g_wk.n[st] >= 0) {   // a stage that reads the pool with no storage buffer anywhere in the draw
        int nb = 0;
        for (int s2 = 0; s2 < VKA_NSTAGE; s2++) if (sh[s2] && g_wk.n[s2] > 0) for (int i = 0; i < g_wk.n[s2]; i++) nb += g_wk.r[s2][i].what == VKA_R_BUF && g_wk.r[s2][i].desc == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && g_wk.r[s2][i].addr;
        if (!nb) {
            all = 1;
            static _Atomic int nl; if (atomic_fetch_add(&nl, 1) < 4) {
                char b[1200]; size_t o = 0; b[0] = 0;
                for (int i = 0; i < g_wk.n[st] && o < sizeof b - 80; i++) { const vka_res *r = &g_wk.r[st][i]; o += (size_t)snprintf(b + o, sizeof b - o, " [%d desc %d b%u %#llx+%#llx]", r->what, r->desc, r->binding, (unsigned long long)r->addr, (unsigned long long)r->size); }
                VLOG("texture pool: shader %016llx (stage %d) reads it and its draw has no storage buffer to scan; its records:%s", (unsigned long long)sh[st]->key, st, b);
            }
        }
        break;   // (the storage buffers are the draw's: scanned once below)
    }
    for (int st = 0; st < VKA_NSTAGE && !all; st++) if (sh[st] && g_wk.n[st] > 0 && !(st == 1 && d->rast_discard)) {
        for (int i = 0; i < g_wk.n[st] && !all; i++) { const vka_res *r = &g_wk.r[st][i];
            if (r->what != VKA_R_BUF || r->desc != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER || !r->addr) continue;
            uint64_t sz = r->size ? r->size : 65536;   // (size 0: to the end of its placed chunk; the first 64 KiB of it)
            int m = 0; while (m < 16 && !(memo[m].addr == r->addr && memo[m].size == sz)) m++;
            if (m < 16 && (rescan_ms > 0 ? tnow - memo[m].t < (uint64_t)rescan_ms * 1000000ull : memo[m].f == fr)) continue;
            int again = m < 16; uint64_t t_was = again ? memo[m].t : 0, f_was = again ? memo[m].f : 0;
            if (m == 16) { m = (int)(memo_next++ & 15); memo[m].addr = r->addr; memo[m].size = sz; }
            memo[m].t = tnow; memo[m].f = fr;
            if (scanned + sz > POOL_SCAN) { all = 1; VLOG_ONCE_POOL("its draw's storage buffer binding %u at %#llx is %#llx bytes, over the scan's %u", r->binding, (unsigned long long)r->addr, (unsigned long long)sz, (unsigned)POOL_SCAN); break; }
            scanned += sz;
            static uint32_t buf[16384]; int mk = 0;
            for (uint64_t o = 0; o < sz; o += sizeof buf) { size_t c = sz - o < sizeof buf ? (size_t)(sz - o) : sizeof buf;
                if (!cpu_read(r->addr + o, buf, c)) { all = 1; VLOG_ONCE_POOL("its draw's storage buffer binding %u at %#llx+%#llx could not be read at +%#llx", r->binding, (unsigned long long)r->addr, (unsigned long long)sz, (unsigned long long)o); break; }
                mk += vka_tex_pool_mark(buf, c / 4, r->size, fb); }   // (r->size 0: unknown, counted as a table)
            marked += mk;
            if (again) { n_rescan++; n_late += (uint64_t)mk;
                static int nl; if (mk && nl < 8) { nl++; VLOG("texture pool: a rescan of storage buffer %#llx+%#llx queued %d texture IDs, %.0f ms (%llu frames) after its last scan: pool-reading draws in between read them as null (DM_VK_POOL_RESCAN_MS %d)",
                    (unsigned long long)r->addr, (unsigned long long)sz, mk, (tnow - t_was) / 1e6, (unsigned long long)(fr - f_was), rescan_ms); } }
        }
    }
#undef VLOG_ONCE_POOL
    static uint64_t n_pd, b_pd; n_pd++; b_pd += scanned;   // the scan's cost, logged now and then (recording thread)
    if (!(n_pd & (n_pd - 1)) && n_pd >= 64) VLOG("texture pool: %llu pool-reading draws so far, %.1f KiB of storage buffers scanned for texture IDs on average; %llu rescans of a buffer scanned before queued %llu texture IDs (read as null until then)",
                                                 (unsigned long long)n_pd, b_pd / 1024.0 / n_pd, (unsigned long long)n_rescan, (unsigned long long)n_late);
    return all ? 2 : marked ? 1 : 0;
}
static int bind_sets(const vka_dstate *d, vka_shader *const sh[VKA_NSTAGE], const vka_layout *lay, VkPipelineBindPoint bp, c0_at *c0) {
    g_w.n = 0;
    int any_pool = 0;
    uint64_t t1_ = g_tm ? cntvct() : 0;
    for (int st = 0; st < VKA_NSTAGE; st++) {
        vka_shader *s = sh[st]; if (!s) continue;
        if (st == 1 && d->rast_discard) continue;   // no fragments: its stage's records are not read (the layout keeps them unwritten: partially bound is not needed for push descriptors)
        int n = g_wk.n[st];
        if (n < 0) { static _Atomic int k; if (atomic_fetch_add(&k, 1) < 16) VRED("shader %016llx: %s; skipped", (unsigned long long)s->key, g_wk.why[st]); return 0; }
        any_pool |= g_wk.pool[st];
        uint32_t w0 = g_w.n;
        if (!stage_writes(d, st, s, g_wk.r[st], n, g_wk.pool[st], st == 5 ? c0 : NULL)) return 0;
        if (st == 1 && ld_on(s->key)) ld_dump(d, 1, s, g_wk.r[st], n, w0, g_wk.pool[st]);   // DM_VK_DRAWDUMP_FS (one-shot)
        else if (st == 0 && sh[1] && !d->rast_discard && ld_on(sh[1]->key)) ld_dump(d, 0, s, g_wk.r[st], n, w0, g_wk.pool[st]);   // (its vertex stage first)
    }
    if (g_tm) PS_ADD(1, cntvct() - t1_);
    if (sh[2] && sh[2]->gs_fx > 0 && bp == VK_PIPELINE_BIND_POINT_GRAPHICS) {   // the claim words, zeroed (vka_spv_gs_claim); bound also when the draw takes no claim (the layout has them)
        if (g_w.n == 512) { VRED_ONCE("more than 512 descriptors in one draw"); return 0; }
        VkDeviceSize n = 4 * (VkDeviceSize)(g_claim_words ? g_claim_words : 1); vka_span cs;
        if (!vka_scratch(n, V.sb_align, &cs)) return 0;
        memset(cs.ptr, 0, (size_t)n);
        uint32_t k = g_w.n++;
        g_w.bi[k] = (VkDescriptorBufferInfo){cs.buf, cs.off, n};
        g_w.w[k] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = VKA_GS_CLAIM_BINDING, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &g_w.bi[k]};
    }
    uint64_t t2_ = g_tm ? cntvct() : 0;
    if (lay->push) { int ps = g_w.n ? push_same(bp, lay->layout) : 1;
                     if (ps == 0) vk.vkCmdPushDescriptorSetKHR(g_cb, bp, lay->layout, 0, g_w.n, g_w.w);
                     else if (ps == 2) vk.vkCmdPushDescriptorSetKHR(g_cb, bp, lay->layout, 0, g_dwn, g_dw); }
    else {
        g_pd[bp == VK_PIPELINE_BIND_POINT_COMPUTE].n = 0;   // set 0 bound as a set: the last push is no longer what the draws see
        VkDescriptorSet set = set_alloc(lay->set0); if (!set) return 0;
        for (uint32_t i = 0; i < g_w.n; i++) g_w.w[i].dstSet = set;
        if (g_w.n) vk.vkUpdateDescriptorSets(V.dev, g_w.n, g_w.w, 0, NULL);
        vk.vkCmdBindDescriptorSets(g_cb, bp, lay->layout, 0, 1, &set, 0, NULL);
    }
    if (g_tm) PS_ADD(2, cntvct() - t2_);
    atomic_fetch_add_explicit(&g_bs_all, 1, memory_order_relaxed); if (any_pool) atomic_fetch_add_explicit(&g_bs_pool, 1, memory_order_relaxed);
    if (any_pool) {
        static uint64_t seen[32]; static int nseen;   // the first 32 shaders that read the texture pool, once each (recording thread)
        for (int st = 0; st < VKA_NSTAGE && nseen < 32; st++) if (sh[st] && g_wk.pool[st]) {
            int k = 0; while (k < nseen && seen[k] != sh[st]->key) k++;
            if (k == nseen) { seen[nseen++] = sh[st]->key; VLOG("texture pool: shader %016llx (stage %d) reads it (a bindless texture or sampler the translator found no fixed slot for)", (unsigned long long)sh[st]->key, st); }
        }
    }
    if (any_pool) { VkDescriptorSet ps = vka_bindless_set(); if (ps) vk.vkCmdBindDescriptorSets(g_cb, bp, lay->layout, 1, 1, &ps, 0, NULL); else VRED_ONCE("a shader reads the texture pool and there is no pool set"); }
    return 1;
}
// Images first used by a draw (from the walk, walk_all), or remade with storage use (vka_tex.c img_storage_upgrade): dry = only whether there is one
static int make_images(const vka_dstate *d, vka_shader *const sh[VKA_NSTAGE], int dry) {
    (void)d; int any = 0;
    for (int st = 0; st < VKA_NSTAGE; st++) {
        if (!sh[st]) continue;
        const vka_res *r = g_wk.r[st];
        for (int i = 0; i < g_wk.n[st]; i++) if ((r[i].what == VKA_R_TEX || r[i].what == VKA_R_IMG) && r[i].tid
                                                 && (vka_tex_id_pending_rec(r[i].tid, r[i].dim) || (r[i].what == VKA_R_IMG && vka_tex_id_storage_pending(r[i].tid)))) {   // (an image remade with storage use copies its content: outside the pass)
            any = 1; if (dry) return 1;
            vka_tex_id_view(r[i].tid, g_cb, r[i].dim, r[i].what == VKA_R_IMG, NULL);
        }
    }
    return any;
}

// The draw writes memory: a storage buffer or image it writes, raw global stores (their write log), a geometry shader's claimed writes (from the walk, walk_all)
static int draw_writes(vka_shader *const sh[VKA_NSTAGE], const vka_dstate *d) {
    if (sh[2] && sh[2]->gs_fx > 0) return 1;
    for (int st = 0; st < VKA_NSTAGE; st++) {
        if (!sh[st] || (st == 1 && d->rast_discard)) continue;
        const vka_res *r = g_wk.r[st];
        for (int i = 0; i < g_wk.n[st]; i++) if ((r[i].flags & DMSH_VK_BF_RW) || r[i].what == VKA_R_IMG || r[i].what == VKA_R_GMEM_LOG) return 1;
    }
    return 0;
}
// After a game barrier taken inside the pass (vka_rp_game_barrier): this draw needs the pass ended first when it writes memory (earlier draws of the pass may read what it
// overwrites), reads the texture pool directly (any texture, the pass's targets among them), or samples a texture whose image the pass draws into
static int bar_hazard(vka_shader *const sh[VKA_NSTAGE], const vka_dstate *d) {
    if (draw_writes(sh, d)) return 1;
    for (int st = 0; st < VKA_NSTAGE; st++) {
        if (!sh[st] || (st == 1 && d->rast_discard)) continue;
        if (g_wk.pool[st] || g_wk.n[st] < 0) return 1;
        const vka_res *r = g_wk.r[st];
        for (int i = 0; i < g_wk.n[st]; i++) if ((r[i].what == VKA_R_TEX || r[i].what == VKA_R_TBUF) && r[i].tid && vka_tex_id_in_pass(r[i].tid)) return 1;
    }
    return 0;
}

// ---------- Draws ----------
// A program's shaders by its serial (a program that gets new shaders gets a new serial, hle_nvn.c), remembered for the draws after: the shader table's lock and lookup
// were taken once per stage of every draw
static struct { uint64_t serial; vka_progsh ps; } g_pm[64];
static int resolve(const prog_t *p, int st, vka_shader **out, unsigned *bad) {
    *out = NULL; if (!p) return 1;
    vka_progsh ps;
    uint64_t sr = p->serial; typeof(g_pm[0]) *m = &g_pm[(sr ^ sr >> 6) & 63];
    if (sr && m->serial == sr) ps = m->ps;
    else { if (!vka_prog_shaders(p, &ps)) return 0; if (sr) { m->serial = sr; m->ps = ps; } }
    if (ps.bad & (1u << st | VKA_BAD_MISSING)) { *bad |= 1u << st; return 1; }
    *out = ps.sh[st]; return 1;
}
// Vertex input: an attribute per input location the vertex shader reads (Location = NVN attribute index), each from its stream; an input no attribute feeds reads a zero buffer.
// With VK_EXT_vertex_input_dynamic_state the descriptions are set per draw, else they go into the pipeline key
static struct { VkVertexInputBindingDescription2EXT b[33]; VkVertexInputAttributeDescription2EXT a[32]; uint32_t nb, na, used, zb; int zero; } VI;
static void vin_build(const vka_dstate *d, const vka_shader *vs, vka_gkey *k) {
    VI.nb = VI.na = VI.used = 0; VI.zero = 0;
    VI.zb = V.max_vin_bindings > 32 ? 31 : V.max_vin_bindings > 16 ? V.max_vin_bindings - 1 : 15;   // the zero buffer's binding: past the 16 streams when the device has room
    for (uint32_t loc = 0; loc < 32; loc++) {
        if (!(vs->e->vin_mask >> loc & 1)) continue;
        int want = (int)(vs->e->vin_types >> (2 * loc) & 3), changed = 0, retype = 0;
        const va_st *a = loc < 16 && (int)loc < d->nattr ? &d->attr[loc] : NULL;
        VkFormat f = a && a->fmt ? vin_format(a->fmt, want, &changed, &retype) : VK_FORMAT_UNDEFINED;
        if (retype == DMSH_VK_TYPE_UINT) k->vin_u |= 1u << loc; else if (retype == DMSH_VK_TYPE_SINT) k->vin_s |= 1u << loc;
        if (a && a->fmt && !f) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 4) VRED("vertex attribute format %#x is not mapped: the input reads zeros", a->fmt); }
        if (changed) { static uint64_t seen[4]; unsigned b = a->fmt & 255; if (!(seen[b >> 6] >> (b & 63) & 1)) { seen[b >> 6] |= 1ull << (b & 63);
            VRED("vertex attribute format %#x feeds a %s input: bound as VkFormat %d of that class (values may differ from Maxwell's fetch)", a->fmt, want == 0 ? "float" : want == 1 ? "sint" : "uint", f); } }
        if (!f) {
            f = want == DMSH_VK_TYPE_UINT ? VK_FORMAT_R32G32B32A32_UINT : want == DMSH_VK_TYPE_SINT ? VK_FORMAT_R32G32B32A32_SINT : VK_FORMAT_R32G32B32A32_SFLOAT;
            VI.a[VI.na++] = (VkVertexInputAttributeDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, NULL, loc, VI.zb, f, 0};
            VI.zero = 1; continue;
        }
        uint32_t st = a->stream & 15;
        VI.a[VI.na++] = (VkVertexInputAttributeDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, NULL, loc, st, f, a->offset};
        VI.used |= 1u << st;
    }
    for (uint32_t st = 0; st < 16; st++) if (VI.used >> st & 1) {
        uint32_t div = d->stream[st].divisor;
        if (div > 1 && !V.divisor) { VRED_ONCE("instance divisor %u and no vertexAttributeInstanceRateDivisor: drawn with divisor 1", div); div = 1; }
        VI.b[VI.nb++] = (VkVertexInputBindingDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, NULL, st, d->stream[st].stride,
                                                               div ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX, div ? div : 1};
    }
    if (VI.zero) VI.b[VI.nb++] = (VkVertexInputBindingDescription2EXT){VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, NULL, VI.zb, 0, VK_VERTEX_INPUT_RATE_VERTEX, 1};
    if (!V.vin_dyn) {
        k->vin_static = 1;
        for (uint32_t i = 0; i < VI.na; i++) { uint32_t l = VI.a[i].location; k->va[l].fmt = VI.a[i].format; k->va[l].offset = VI.a[i].offset; k->va[l].binding = (uint8_t)VI.a[i].binding; k->va[l].used = 1; }
        for (uint32_t i = 0; i < VI.nb; i++) { uint32_t b = VI.b[i].binding; k->vb[b].stride = VI.b[i].stride; k->vb[b].divisor = VI.b[i].inputRate ? VI.b[i].divisor : 0; k->vb[b].used = 1; }
    }
}
// The vertex buffers the command buffer holds (slot: buffer, offset, size): a draw binds only the slots that changed, consecutive ones in one call. Vulkan keeps vertex
// buffer bindings across pipeline binds and passes for the life of the command buffer; forgotten with the dynamic state (vka_dyn_invalidate: a new command buffer or
// part). Profile p2: vin_bind was 13 % of the replay thread, one vkCmdBindVertexBuffers2 a stream a draw. DM_VK_VB_FILTER=0 (live): every slot every draw
static struct { int valid, vin_stale; VkBuffer b[33]; VkDeviceSize o[33], sz[33]; uint32_t vnb, vna; VkVertexInputBindingDescription2EXT vb[33]; VkVertexInputAttributeDescription2EXT va[32]; } VB;
static void vb_set(uint32_t slot, VkBuffer b, VkDeviceSize o, VkDeviceSize sz, uint32_t *n0, uint32_t *nn, VkBuffer *rb, VkDeviceSize *ro, VkDeviceSize *rs) {
    if (slot >= 33) return;
    if (VB.valid && g_vb_filter && VB.b[slot] == b && VB.o[slot] == o && VB.sz[slot] == sz) return;
    VB.b[slot] = b; VB.o[slot] = o; VB.sz[slot] = sz;
    if (*nn && *n0 + *nn != slot) { vk.vkCmdBindVertexBuffers2(g_cb, *n0, *nn, rb, ro, rs, NULL); *nn = 0; }   // a run ends: bound in one call
    if (!*nn) *n0 = slot;
    rb[*nn] = b; ro[*nn] = o; rs[*nn] = sz; (*nn)++;
}
static void vin_bind(const vka_dstate *d) {
    if (!VB.valid) memset(&VB, 0, sizeof VB);
    // The vertex input state too only when it changed (it was set every draw: up to 65 descriptions copied into the encoder's ring and the driver), and again after
    // every pipeline bind: Vulkan keeps dynamic vertex input across vkCmdBindPipeline, but drivers have lost it there before (hardening; the buffers stay filtered)
    if (V.vin_dyn) {
        int same = VB.valid && !VB.vin_stale && g_vb_filter && VB.vnb == VI.nb && VB.vna == VI.na && !memcmp(VB.vb, VI.b, VI.nb * sizeof *VI.b) && !memcmp(VB.va, VI.a, VI.na * sizeof *VI.a);
        if (!same) { vk.vkCmdSetVertexInputEXT(g_cb, VI.nb, VI.b, VI.na, VI.a); VB.vnb = VI.nb; VB.vna = VI.na; memcpy(VB.vb, VI.b, VI.nb * sizeof *VI.b); memcpy(VB.va, VI.a, VI.na * sizeof *VI.a); }
        VB.vin_stale = 0;
    }
    VkBuffer rb[33]; VkDeviceSize ro[33], rs[33]; uint32_t n0 = 0, nn = 0;
    for (uint32_t st = 0; st < 16; st++) if (VI.used >> st & 1) {   // each stream at its address, as far as it is bound and placed
        VkBuffer b = g_zero.buf; VkDeviceSize o = g_zero.off, sz = 65536;
        uint64_t addr = d->vbuf[st].address, size = d->vbuf[st].size;
        if (addr) { VkBuffer cb; VkDeviceSize co; uint64_t run = vka_addr_run(addr, size ? size : 1ull << 30, &cb, &co);
                    if (run) { b = cb; o = co; sz = run; atomic_fetch_add_explicit(&g_rch.vb, 1, memory_order_relaxed);
                               if (size && run < size) { atomic_fetch_add_explicit(&g_rch.vb_hard, 1, memory_order_relaxed);
                                   static _Atomic int nl; if (atomic_fetch_add(&nl, 1) < VKA_ENV("DM_VK_REACH_LOG", 0))
                                       VLOG("[reach] vertex stream %u: %#llx+%#llx bound to %#llx", st, (unsigned long long)addr, (unsigned long long)size, (unsigned long long)run); } }
                    else VRED_ONCE("vertex stream at %#llx is not in placed guest RAM: it reads zeros", (unsigned long long)addr); }
        vb_set(st, b, o, sz, &n0, &nn, rb, ro, rs);
    }
    if (VI.zero) vb_set(VI.zb, g_zero.buf, g_zero.off, 65536, &n0, &nn, rb, ro, rs);
    if (VKA_ENV("DM_VK_DRAW_LOG", 0) >= 1)   // (a vertex attribute that reads zeros collapses the primitives)
        for (uint32_t i = 0; i < VI.na; i++) { uint32_t bnd = VI.a[i].binding, j = 0; uint64_t addr = bnd < 16 ? d->vbuf[bnd].address : 0; float fv[4] = {0}; int rd = addr && cpu_read(addr + VI.a[i].offset, fv, sizeof fv);
            while (j < VI.nb && VI.b[j].binding != bnd) j++;   // its binding's description (VI.b is in stream order, not indexed by binding: the stride logged was another stream's)
            VLOG("draw vertex input location %u: format %d binding %u offset %u stride %u %s divisor %u%s, stream at %#llx + %#llx: %s %g %g %g %g", VI.a[i].location, (int)VI.a[i].format, bnd, VI.a[i].offset,
                 j < VI.nb ? VI.b[j].stride : 0, j < VI.nb && VI.b[j].inputRate == VK_VERTEX_INPUT_RATE_INSTANCE ? "per instance" : "per vertex", j < VI.nb ? VI.b[j].divisor : 0,
                 VI.zero && bnd == VI.zb ? " (the zero buffer)" : "", (unsigned long long)addr, (unsigned long long)(bnd < 16 ? d->vbuf[bnd].size : 0), rd ? "first" : "not read", fv[0], fv[1], fv[2], fv[3]); }
    if (nn) vk.vkCmdBindVertexBuffers2(g_cb, n0, nn, rb, ro, rs, NULL);
    VB.valid = 1;
}

// The guest indices of an indexed draw as 32-bit values (the CPU's view): for primitives Vulkan has no topology for and 8-bit indices on a device without them
static uint32_t *read_indices(uint64_t addr, int itype, uint32_t count) {
    size_t isz = itype == 0 ? 1 : itype == 1 ? 2 : 4; uint8_t *raw = malloc(isz * count + 4); uint32_t *o = malloc(4 * (size_t)count + 4);
    if (!raw || !o || !cpu_read(addr, raw, isz * count)) { free(raw); free(o); return NULL; }
    for (uint32_t i = 0; i < count; i++) o[i] = isz == 1 ? raw[i] : isz == 2 ? ((uint16_t *)raw)[i] : ((uint32_t *)raw)[i];
    free(raw); return o;
}

// The dynamic state the command buffer holds: each draw sets only what changed (the draw path set some 25 states a draw). Valid from the first draw of a submission until
// something else binds a graphics pipeline with static state (the masked stencil clear: vka_dyn_invalidate) or the submission ends; every graphics pipeline of the draws has
// the same dynamic states (vka_pso.c), so binding one keeps them, except a mesh pipeline, which has no topology or primitive restart (vka_draw resets DY.topo / DY.pr after one)
static struct { int valid; VkPipeline pipe; VkViewport vp; VkRect2D sc; uint32_t cull, ff, topo, dt, dw, dc, dbt, stt, dbe, rd, pr, sop[8], smask[6]; float db[3], bc[4]; } DY;
#define DY_SET(f, val, call) do { if (!DY.valid || !g_dyn_filter || memcmp(&DY.f, &(val), sizeof DY.f)) { memcpy(&DY.f, &(val), sizeof DY.f); call; } } while (0)
#define DY_SET1(f, val, call) do { uint32_t v_ = (val); if (!DY.valid || !g_dyn_filter || DY.f != v_) { DY.f = v_; call; } } while (0)
// The index buffer the command buffer holds: bound again only when it changes (DM_VK_VB_FILTER), forgotten with the dynamic state
static struct { int valid; VkBuffer b; VkDeviceSize o; VkIndexType t; } IB;
static void ib_bind(VkBuffer b, VkDeviceSize o, VkIndexType t) {
    if (IB.valid && g_vb_filter && IB.b == b && IB.o == o && IB.t == t) return;
    vk.vkCmdBindIndexBuffer(g_cb, b, o, t); IB.valid = 1; IB.b = b; IB.o = o; IB.t = t;
}
void vka_dyn_invalidate(void) { DY.valid = 0; VB.valid = 0; IB.valid = 0; g_pd[0].n = g_pd[1].n = 0; }
#define PT(i) do { if (g_tm) { uint64_t t_ = cntvct(); if ((i) > 0 && g_pt_ph == (i) - 1) { g_pt[(i) > 0 ? (i) - 1 : 0] += t_ - g_pt_last; g_pt_all[(i) > 0 ? (i) - 1 : 0] += t_ - g_pt_last; } g_pt_last = t_; g_pt_ph = (i); } } while (0)
void vka_draw_prof(char *b, size_t n, double draws) {
    if (!g_prof || draws <= 0) { if (n) b[0] = 0; if (!g_prof) { memset(g_ps, 0, sizeof g_ps); memset(g_pt, 0, sizeof g_pt); } return; }   // (DM_VK_RPROF alone also fills them)
    uint64_t f; __asm__ volatile("mrs %0, cntfrq_el0" : "=r"(f)); double us = 1e6 / (double)f / draws;
    static const char *nm[8] = {"resolve", "layout", "images+pass+key", "vertex input", "pipeline", "sets", "vertex buffers+dynamic", "draw"};
    size_t o = 0; for (int i = 0; i < 8 && o < n; i++) o += snprintf(b + o, n - o, "%s%s %.2f", i ? ", " : "", nm[i], g_pt[i] * us);
    if (o < n) o += snprintf(b + o, n - o, "; pushes %.0f%% skipped as the last one", g_pd_skip + g_pd_push ? 100.0 * g_pd_skip / (double)(g_pd_skip + g_pd_push) : 0.0);
    g_pd_skip = g_pd_push = 0;
    if (o < n) o += snprintf(b + o, n - o, "; sets: walk %.2f, writes %.2f (driver segments %.2f, %.1f images %.2f), push %.2f", g_ps[0] * us, g_ps[1] * us, g_ps[3] * us, g_ps[5] / draws, g_ps[4] * us, g_ps[2] * us);
    memset(g_ps, 0, sizeof g_ps);
    memset(g_pt, 0, sizeof g_pt);
}
void vka_draw(vka_dstate *d, const p_set_rt *rt, int has_rt, const p_scissor *sc, int has_sc, const p_draw *dw, const p_draw_idx *di) {
    if (!g_cb || !draws_on() || !statics()) { g_skipped++; return; }
    if (!d->inited) vka_dstate_reset(d);
    uint64_t mdi = dw ? dw->mdi_cnt : di->mdi_cnt; int mdi_i = dw ? dw->mdi_i : di->mdi_i;
    if (mdi) { uint32_t cnt = 0; if (!cpu_read(mdi, &cnt, 4) || (uint32_t)mdi_i >= cnt) return;   // past the multi-draw's count (read now: a count the GPU writes in this submission is not seen)
               VRED_ONCE("indirect-count multi-draws read their count on the CPU at replay"); }
    d->b.base_instance = (uint32_t)(dw ? dw->base_instance : di->base_instance);
    vka_shader *sh[VKA_NSTAGE] = {0}; unsigned bad = 0;
    PT(0);
    // An NVN2 mesh draw (nvnCommandBufferDrawMeshTasks, hle_nvn.c): the task (optional) and mesh shaders of one program, the fragment stage's shader, in a mesh pipeline
    // (vka_pso.c pstate_fill) drawn with vkCmdDrawMeshTasksEXT: no vertex input, primitive type or index buffer
    const int mesh = dw && dw->prim == NVN_PRIM_MESH;
    if (mesh) {
        if (!V.mesh) SKIP("mesh draw (nvnCommandBufferDrawMeshTasks) on a device without VK_EXT_mesh_shader (or DM_VK_MESH=0): skipped");
        // The program whose task and mesh shaders run: the fragment stage's when it has a mesh shader (a program's stages are bound together), else the one bound at the
        // mesh stage bit. Task and mesh come from the same program (a program's task shader feeds its own mesh shader)
        const prog_t *mp = d->bst[7];
        { vka_shader *m = NULL; unsigned b2 = 0; if (d->bst[1] && d->bst[1] != mp && resolve(d->bst[1], 7, &m, &b2) && (m || (b2 & 1u << 7))) mp = d->bst[1]; }
        if (!resolve(mp, 6, &sh[6], &bad) || !resolve(mp, 7, &sh[7], &bad) || !resolve(d->bst[1], 1, &sh[1], &bad)) SKIP("mesh draw skipped: no shader library");
        if (bad) { g_skipped++; return; }   // a refused or missing shader (logged when it was made)
        if (!sh[7]) SKIP("mesh draw with no mesh shader bound: skipped");
        if (d->bst[6] && d->bst[6] != mp) VRED_ONCE("mesh draw: a program bound at the task stage bit apart from the mesh program; drawn with the mesh program's own task shader, if any");
        // A pixel shader with per-primitive inputs reads what its mesh shader writes per primitive (Vulkan matches only equally decorated interfaces). The other way round
        // (a mesh shader with per-primitive outputs, a pixel shader without the decoration: vk_android_build.py leaves a pixel shader also drawn with vertex shaders per
        // vertex) the flags cannot tell whether it reads those locations: a red line
        if (sh[1] && (sh[1]->e->flags & DMSH_VK_F_MESH_PRIM) && !(sh[7]->e->flags & DMSH_VK_F_MESH_PRIM))
            SKIP("mesh draw: pixel shader %016llx reads per-primitive inputs that mesh shader %016llx does not write: skipped", (unsigned long long)sh[1]->key, (unsigned long long)sh[7]->key);
        if (sh[1] && !(sh[1]->e->flags & DMSH_VK_F_MESH_PRIM) && (sh[7]->e->flags & DMSH_VK_F_MESH_PRIM))
            VRED_ONCE("mesh draw: mesh shader %016llx writes per-primitive outputs and pixel shader %016llx has no per-primitive inputs: any it reads there are undefined", (unsigned long long)sh[7]->key, (unsigned long long)sh[1]->key);
        if (dw->first) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 4) VRED("mesh draw with first workgroup %d (count %d): drawn from 0 (the translated task / mesh shaders read their workgroup index with no base; open)", dw->first, dw->count); }
        if (dw->count <= 0) return;
        if (d->tf_on) VRED_ONCE("transform feedback with a mesh draw: drawn without capture");
    } else {
        for (int st = 0; st < 5; st++) if (!resolve(d->bst[st], st, &sh[st], &bad)) SKIP("draw skipped: no shader library");
        if (bad) { g_skipped++; return; }   // a refused or missing shader (logged when it was made)
        if (!sh[0]) SKIP("draw with no vertex shader bound: skipped");
        if (sh[1] && (sh[1]->e->flags & DMSH_VK_F_MESH_PRIM)) SKIP("pixel shader %016llx reads per-primitive mesh outputs and is drawn with a vertex shader: skipped", (unsigned long long)sh[1]->key);
        if (d->rast_discard && d->tf_on) SKIP("transform feedback is not on Vulkan yet (F_TF): the draw is skipped");
        if (d->tf_on) VRED_ONCE("transform feedback is not on Vulkan yet: the draw is drawn without capture");
    }
    if (d->poly.cull == NVN_FACE_FRONT_AND_BACK && !d->rast_discard) return;
    int prim = dw ? dw->prim : di->prim, expand = 0;
    VkPrimitiveTopology topo = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;   // (a mesh pipeline's: every translated mesh shader outputs triangles)
    if (!mesh) {
        topo = topology(prim, &expand);
        if (expand < 0) SKIP("primitive type %d not on Vulkan: the draw is skipped", prim);
        if (topo == VK_PRIMITIVE_TOPOLOGY_PATCH_LIST) SKIP("tessellation patches are not on Vulkan yet (no patch size is recorded): the draw is skipped");
    }
    uint64_t ind = dw ? dw->ind : di->ind;
    // A geometry shader that writes memory makes those writes in the run that claims its primitive (vka_spv_gs_claim): a zeroed word per primitive at VKA_GS_CLAIM_BINDING.
    // Primitive IDs restart with each instance and an indirect draw's count is the GPU's, so those draws take the shader as it is (each run writing)
    g_claim_words = 0;
    if (sh[2] && sh[2]->gs_fx > 0) {
        int inst_n = dw ? dw->instances : di->instances;
        if (!ind && inst_n <= 1) { int32_t c = dw ? dw->count : di->count; g_claim_words = c > 0 ? (uint32_t)c : 1; }
        else VRED_ONCE("geometry shader %016llx writes memory in an %s draw: drawn without the claim, its writes repeated by the device's runs of each primitive", (unsigned long long)sh[2]->key, ind ? "indirect" : "instanced");
    }
    if (expand && ind) SKIP("an indirect draw of quads or a line loop is not on Vulkan yet: skipped");

    PT(1);
    // The last draw's shaders' layout (layouts live as long as the device): the layout table's lock and hash a draw
    static vka_shader *lsh[VKA_NSTAGE]; static vka_layout llay; static int lok;
    vka_layout lay;
    if (lok && !memcmp(lsh, sh, sizeof lsh)) lay = llay;
    else { lok = 0; if (!vka_layout_of(sh, 0, &lay)) SKIP("draw skipped: no pipeline layout"); memcpy(lsh, sh, sizeof lsh); llay = lay; lok = 1; }
    // Images first used by this draw are made (cleared or uploaded) outside the render pass, then the pass opens over the bound targets
    PT(2);
    walk_all(d, sh);
    { uint64_t ti_ = VKA_RP_T0(); if (make_images(d, sh, 1)) { vka_rp_end(); make_images(d, sh, 0); } VKA_RP_ADD(VKA_RPX_IMAGES, ti_); }
    { uint64_t tp_ = VKA_RP_T0(); int ps = pool_scan(d, sh); if (ps) { vka_rp_end(); vka_tex_pool_write(g_cb, ps == 2); } VKA_RP_ADD(VKA_RPX_POOLSCAN, tp_); }   // (lazy images the draw may index through the pool: outside the pass, as make_images)
    if (R.on && R.bar_pending && bar_hazard(sh, d)) { g_bar_late++; vka_rp_end(); }   // the game barrier taken inside the pass is needed here: the pass ends (its full barrier) first
    if (vka_nan_on() >= 2 && sh[1] && !d->rast_discard && vka_nan_probe(sh[1]->key)) {   // DM_VK_NANSCAN_FS: this draw's sampled textures, scanned before it
        char why[8]; int n = vka_walk(&d->b, 1, sh[1]->e, sh[1]->b, sh[1]->lib->h->ssbo_align, 0, rd_cb, NULL, g_res, 512, NULL, why, sizeof why), m = 0;
        static vka_img *ims[64]; static int bs[64], ts[64];
        for (int i = 0; i < n && m < 64; i++) if (g_res[i].what == VKA_R_TEX && g_res[i].tid) { ims[m] = vka_tex_id_img(g_res[i].tid, g_cb); bs[m] = g_res[i].binding; ts[m++] = g_res[i].tid;
            static _Atomic int nd; if (atomic_fetch_add(&nd, 1) < 32) { char why[320]; vka_tex_id_desc(g_res[i].tid, why, sizeof why); vka_img *im = ims[m - 1];
                if (getenv("DM_VK_NANSCAN_DUMP") && im && im->d.format >= 0x79 && im->d.format <= 0x94) vka_tex_id_dump(g_res[i].tid);
                vka_tex_id_check(g_res[i].tid);
                VLOG("NaN scan: input binding %u dim %#x texture ID %d sampler %d: %s; image inited %d gpu_owned %d track %d", g_res[i].binding, g_res[i].dim, g_res[i].tid, g_res[i].sid, why, im ? im->inited : -1, im ? im->gpu_owned : -1, im ? im->track : -1); } }
        int on = R.on; vka_rp_end(); (void)on;
        vka_nan_inputs(g_cb, ims, bs, ts, m);
    }
    if (!rp_begin(rt, has_rt, d)) { g_skipped++; return; }
    if (!R.swrite && draw_writes(sh, d)) R.swrite = 1;   // (a game barrier in this pass from now on ends it: vka_rp_game_barrier)
    d->b.samples = R.samples;
    vka_gkey k; memset(&k, 0, V.vin_dyn ? offsetof(vka_gkey, va) : sizeof k);   // (va, vb: only without vertex input dynamic state, vin_build; the pipeline lookup reads no further then)
    k.ncolor = (uint8_t)R.ncolor; k.samples = (uint8_t)R.samples; k.dfmt = R.df; k.sfmt = R.sf;
    k.topo = mesh ? 0 : (uint8_t)topo_class(topo);
    k.poly_mode = (uint8_t)(d->poly.mode == NVN_POLYGON_MODE_LINE && V.fill_lines ? VK_POLYGON_MODE_LINE : d->poly.mode == NVN_POLYGON_MODE_POINT && V.fill_lines ? VK_POLYGON_MODE_POINT : VK_POLYGON_MODE_FILL);
    k.depth_clamp = d->depth_clamp && V.depth_clamp; k.a2c = d->ms.a2c != 0;
    if (d->color.logic_op != NVN_LOGIC_OP_COPY && V.logic_op) { k.logic_en = 1; k.logic_op = (uint8_t)d->color.logic_op; }
    k.gl_depth = !vka_depth_zero();
    const vka_shader *fs = d->rast_discard ? NULL : sh[1];
    // The pixel shader's module (android/SHADERS.md "Variants"): MOD_UINT when every target it writes is a UINT attachment, MOD_UINT_MIXED when the UINT ones are exactly its
    // uint_mask, MOD_DUAL when target 0 blends with a SRC1 factor; else the base. The alpha test is the base's specialization constant in every variant
    uint32_t umask = 0, amask = 0;
    for (uint32_t i = 0; i < R.ncolor; i++) if (R.cf[i]) { amask |= 1u << i; if (R.ckind[i] == 1) umask |= 1u << i; }
    int variant = DMSH_VK_MOD_BASE;
    if (fs) {
        uint32_t fm = fs->e->fout_mask;
        if ((umask & fm) && (umask & fm) == (amask & fm) && vka_shlib_module(fs->lib, fs->e, DMSH_VK_MOD_UINT)) variant = DMSH_VK_MOD_UINT;
        else if ((umask & fm) && fs->e->uint_mask == (umask & fm) && vka_shlib_module(fs->lib, fs->e, DMSH_VK_MOD_UINT_MIXED)) variant = DMSH_VK_MOD_UINT_MIXED;
        else if ((d->color.blend_en & 1) && V.dual_src && vka_shlib_module(fs->lib, fs->e, DMSH_VK_MOD_DUAL)) {
            blend_st b0 = adv_blend(d->blend[0]);
            if (is_src1(b0.src_rgb) || is_src1(b0.dst_rgb) || is_src1(b0.src_a) || is_src1(b0.dst_a)) variant = DMSH_VK_MOD_DUAL;
        }
        if ((fs->e->flags & DMSH_VK_F_ALPHA_TEST) && d->color.alpha_func != NVN_ALPHA_FUNC_ALWAYS) k.alpha_func = (uint8_t)d->color.alpha_func;
    }
    // No library variant for a float output into an integer attachment (the library has MOD_UINT only for outputs made of integer bits or constants): the base module
    // with those outputs retyped as the integer vectors of their bits (VKA_FS_RETYPE, vka_spv_fout_retype), which is what the hardware writes. DM_VK_FOUT_RETYPE=0: not written, as before
    // Only a module that was rewritten is used (one the rewrite refuses falls back to the base module, whose float outputs those targets then do not take: written = 0 below).
    // MOD_UINT_MIXED / MOD_DUAL pipelines keep their variant; a float output there into another integer target stays unwritten (open)
    if (fs && variant == DMSH_VK_MOD_BASE && VKA_ENV("DM_VK_FOUT_RETYPE", 1)) {
        uint32_t fu = 0, fsm = 0;
        for (uint32_t i = 0; i < R.ncolor; i++) if (R.cf[i] && (R.ckind[i] == 1 || R.ckind[i] == 2) && (fs->e->fout_mask >> i & 1) && (fs->e->fout_types >> (2 * i) & 3) == DMSH_VK_TYPE_FLOAT) { if (R.ckind[i] == 1) fu |= 1u << i; else fsm |= 1u << i; }
        if ((fu | fsm) && vka_shader_fout_retyped((vka_shader *)fs, fu, fsm)) variant = VKA_FS_RETYPE;
    }
    k.fs_variant = (uint8_t)variant; k.gs_fx = g_claim_words != 0;
    int src1_used = 0;
    for (uint32_t i = 0; i < R.ncolor; i++) {
        k.cfmt[i] = R.cf[i];
        if (!R.cf[i]) continue;
        uint32_t m = d->chmask.mask >> (4 * i) & 0xf;
        int written = fs && (fs->e->fout_mask >> i & 1);
        int out_cls = !fs ? 0 : variant == DMSH_VK_MOD_UINT || (variant == DMSH_VK_MOD_UINT_MIXED && (fs->e->uint_mask >> i & 1)) ? DMSH_VK_TYPE_UINT : (int)(fs->e->fout_types >> (2 * i) & 3);
        int att_cls = R.ckind[i] == 1 ? DMSH_VK_TYPE_UINT : R.ckind[i] == 2 ? DMSH_VK_TYPE_SINT : DMSH_VK_TYPE_FLOAT;
        if (variant == VKA_FS_RETYPE && out_cls == DMSH_VK_TYPE_FLOAT && att_cls != DMSH_VK_TYPE_FLOAT) out_cls = att_cls;   // (retyped in the module: vka_pso.c fout_masks)
        if (written && out_cls != att_cls) {   // Vulkan leaves a class mismatch undefined: the target is not written (SINT attachments have no variant yet)
            static _Atomic int n; if (atomic_fetch_add(&n, 1) < 4) VRED("fragment shader %016llx writes a %s output to a %s target %u: that target is not written", (unsigned long long)fs->key,
                                                                       out_cls == 0 ? "float" : out_cls == 1 ? "sint" : "uint", att_cls == 0 ? "float" : att_cls == 1 ? "sint" : "uint", i);
            written = 0;
        }
        k.att[i].mask = written ? (uint8_t)m : 0;   // outputs the shader does not write leave the target as it was (Maxwell writes only the shader's outputs)
        if (written && (d->color.blend_en >> i & 1) && att_cls == DMSH_VK_TYPE_FLOAT) {
            blend_st b = adv_blend(d->blend[i]);
            if (variant != DMSH_VK_MOD_DUAL && (is_src1(b.src_rgb) || is_src1(b.dst_rgb) || is_src1(b.src_a) || is_src1(b.dst_a))) {   // no second source: it reads as zero
                src1_used = 1; b.src_rgb = zero_src1(b.src_rgb); b.dst_rgb = zero_src1(b.dst_rgb); b.src_a = zero_src1(b.src_a); b.dst_a = zero_src1(b.dst_a); }
            k.att[i] = (typeof(k.att[i])){1, (uint8_t)bfac(b.src_rgb), (uint8_t)bfac(b.dst_rgb), (uint8_t)bop(b.eq_rgb), (uint8_t)bfac(b.src_a), (uint8_t)bfac(b.dst_a), (uint8_t)bop(b.eq_a), (uint8_t)m};
        }
    }
    if (src1_used) VRED_ONCE("a SRC1 blend factor with no dual-source module (MOD_DUAL; the profile's dual_want.tsv) or no dualSrcBlend: the second source reads as zero");

    PT(3);
    if (!mesh) vin_build(d, sh[0], &k);
    PT(4);
    VkPipeline p = vka_gfx_pipeline(sh, &lay, &k);
    if (!p) { g_skipped++; return; }
    if (!DY.valid || !g_dyn_filter || DY.pipe != p) { vk.vkCmdBindPipeline(g_cb, VK_PIPELINE_BIND_POINT_GRAPHICS, p); DY.pipe = p; VB.vin_stale = 1; }
    if (vka_nan_on()) vka_nan_draw(mesh ? sh[7]->key : sh[0]->key, fs ? fs->key : 0);
    PT(5);
    dd_begin(d, sh, dw, di);
    if (!bind_sets(d, sh, &lay, VK_PIPELINE_BIND_POINT_GRAPHICS, NULL)) { g_dd.on = 0; g_skipped++; return; }
    PT(6);
    if (!mesh) vin_bind(d);
    dd_end(d, dw ? NULL : di);

    // Viewport and scissor. The translator writes clip space for Metal (y up, origin upper left): Metal's viewport, flipped for a lower-left window origin as on Metal, then given
    // a negative height, maps it as Metal does (android/SHADERS.md "Depth and Y")
    float vx = d->has_vp ? d->vp[0][0] : 0, vy = d->has_vp ? d->vp[0][1] : 0, vw = d->has_vp ? d->vp[0][2] : (float)R.area.extent.width, vh = d->has_vp ? d->vp[0][3] : (float)R.area.extent.height;
    if (vka_origin_ll()) { vy += vh; vh = -vh; }
    { uint32_t sw; memcpy(&sw, d->vp_sw[0], 4);
      if (sw == 0x100) { vy += vh; vh = -vh; }
      else if (sw) VRED_ONCE("viewport swizzle %d,%d,%d,%d not implemented (drawn unswizzled)", d->vp_sw[0][0], d->vp_sw[0][1] ^ 2, d->vp_sw[0][2] ^ 4, d->vp_sw[0][3] ^ 6); }
    float n0 = d->dr[0] < 0 ? 0 : d->dr[0] > 1 ? 1 : d->dr[0], n1 = d->dr[1] < 0 ? 0 : d->dr[1] > 1 ? 1 : d->dr[1];
    if (!vw) vw = 1; if (!vh) vh = 1;
    VkViewport vp = {vx, vy + vh, vw, -vh, n0, n1};
    DY_SET(vp, vp, vk.vkCmdSetViewport(g_cb, 0, 1, &vp));
    VkRect2D scr = R.area;
    if (has_sc && !d->rast_discard) {
        int64_t x0 = sc->x < 0 ? 0 : sc->x, y0 = sc->y < 0 ? 0 : sc->y, x1 = (int64_t)sc->x + sc->w, y1 = (int64_t)sc->y + sc->h;
        if (x1 > R.area.extent.width) x1 = R.area.extent.width; if (y1 > R.area.extent.height) y1 = R.area.extent.height;
        if (x1 <= x0 || y1 <= y0) {   // an empty scissor: nothing drawn (DM_VK_DRAW_LOG: said, as the draw's state line below is not reached)
            if (VKA_ENV("DM_VK_DRAW_LOG", 0) >= 1) VLOG("draw fs %016llx: scissor %d %d %d %d is empty in the %ux%u render area: not drawn", fs ? (unsigned long long)fs->key : 0ull, sc->x, sc->y, sc->w, sc->h, R.area.extent.width, R.area.extent.height);
            return;
        }
        scr = (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
    }
    { int xs = VKA_ENV("DM_VK_XP_SCISSOR", 0);   // experiment (wrong picture): every draw's scissor cut to xs x xs pixels, to split a pass's GPU time into per-pixel and per-draw work
      if (xs > 0) { if (scr.extent.width > (uint32_t)xs) scr.extent.width = (uint32_t)xs; if (scr.extent.height > (uint32_t)xs) scr.extent.height = (uint32_t)xs; } }
    DY_SET(sc, scr, vk.vkCmdSetScissor(g_cb, 0, 1, &scr));
    if (VKA_ENV("DM_VK_DRAW_LOG", 0) >= 1) {   // (why a draw writes no pixels: its target masks, blend, viewport, scissor, culling, depth test)
        char mk[64]; int mo = 0; mk[0] = 0; for (int i = 1; i < (int)R.ncolor && i < 8 && mo < 56; i++) mo += snprintf(mk + mo, sizeof mk - mo, " %d:%#x", i, k.att[i].mask);
        VLOG("draw fs %016llx count %d inst %d: target 0 mask %#x%s blend %d (%d %d %d / %d %d %d), viewport %.0f %.0f %.0f %.0f depth %.2f-%.2f, scissor %d %d %u %u, cull %d front %d, depth test %d write %d func %d, stencil %d, discard %d, "
             "prim %d first %d base instance %u indirect %d, module variant %d, alpha test %d, alpha to coverage %d at %u samples",   // (Vulkan applies alpha to coverage at 1 sample too)
             fs ? (unsigned long long)fs->key : 0ull, dw ? dw->count : di->count, dw ? dw->instances : di->instances, k.att[0].mask, mk, k.att[0].en, k.att[0].src, k.att[0].dst, k.att[0].op, k.att[0].srca, k.att[0].dsta, k.att[0].opa,
             vp.x, vp.y, vp.width, vp.height, vp.minDepth, vp.maxDepth, scr.offset.x, scr.offset.y, scr.extent.width, scr.extent.height, d->poly.cull, d->poly.front, d->ds.dt, d->ds.dw, d->ds.df, d->ds.st, d->rast_discard,
             prim, dw ? dw->first : 0, d->b.base_instance, ind != 0, (int)k.fs_variant, (int)k.alpha_func, (int)k.a2c, R.samples); }
    // Rasterizer and depth / stencil
    DY_SET1(cull, (uint32_t)(d->poly.cull == NVN_FACE_FRONT ? VK_CULL_MODE_FRONT_BIT : d->poly.cull == NVN_FACE_BACK ? VK_CULL_MODE_BACK_BIT : d->poly.cull == NVN_FACE_FRONT_AND_BACK ? VK_CULL_MODE_FRONT_AND_BACK : VK_CULL_MODE_NONE), vk.vkCmdSetCullMode(g_cb, DY.cull));
    DY_SET1(ff, (uint32_t)((d->poly.front == NVN_FRONT_FACE_CCW) != vka_queue_origin_ll() ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE), vk.vkCmdSetFrontFace(g_cb, DY.ff));   // as nvn_mtl.m's winding
    // A mesh pipeline has no input assembly: the topology and primitive restart are not set for it, and set again at the next draw that has them (its pipeline bind may
    // have left them undefined)
    if (mesh) DY.topo = DY.pr = ~0u;
    else DY_SET1(topo, (uint32_t)(topo), vk.vkCmdSetPrimitiveTopology(g_cb, DY.topo));
    int hd = R.df != VK_FORMAT_UNDEFINED, hs = R.sf != VK_FORMAT_UNDEFINED;
    DY_SET1(dt, (uint32_t)(hd && d->ds.dt), vk.vkCmdSetDepthTestEnable(g_cb, DY.dt));
    DY_SET1(dw, (uint32_t)(hd && d->ds.dt && d->ds.dw), vk.vkCmdSetDepthWriteEnable(g_cb, DY.dw));
    DY_SET1(dc, (uint32_t)(d->ds.df >= 1 && d->ds.df <= 8 ? (VkCompareOp)(d->ds.df - 1) : VK_COMPARE_OP_ALWAYS), vk.vkCmdSetDepthCompareOp(g_cb, DY.dc));
    DY_SET1(dbt, (uint32_t)(VK_FALSE), vk.vkCmdSetDepthBoundsTestEnable(g_cb, DY.dbt));
    DY_SET1(stt, (uint32_t)(hs && d->ds.st), vk.vkCmdSetStencilTestEnable(g_cb, DY.stt));
    for (int i = 0; i < R.nw; i++) if (R.wslot[i] < 0 ? DY.dw || DY.stt : k.att[R.wslot[i]].mask != 0) vka_rtw(R.wim[i], VKA_RTW_DRAW);   // (DM_VK_RTWATCH_MB: a draw with writes on)
    #define SOP(x) ((x) >= 1 && (x) <= 8 ? (VkStencilOp)((x) - 1) : VK_STENCIL_OP_KEEP)
    #define SCMP(x) ((x) >= 1 && (x) <= 8 ? (VkCompareOp)((x) - 1) : VK_COMPARE_OP_ALWAYS)
    { uint32_t so[8] = {SOP(d->ds.fail_f), SOP(d->ds.zpass_f), SOP(d->ds.zfail_f), SCMP(d->ds.sf_f), SOP(d->ds.fail_b), SOP(d->ds.zpass_b), SOP(d->ds.zfail_b), SCMP(d->ds.sf_b)};
      DY_SET(sop, so, (vk.vkCmdSetStencilOp(g_cb, VK_STENCIL_FACE_FRONT_BIT, so[0], so[1], so[2], so[3]), vk.vkCmdSetStencilOp(g_cb, VK_STENCIL_FACE_BACK_BIT, so[4], so[5], so[6], so[7]))); }
    #undef SOP
    #undef SCMP
    { uint32_t sm[6] = {d->st_vmask[0], d->st_vmask[1], d->st_wmask[0], d->st_wmask[1], d->st_ref[0], d->st_ref[1]};
      DY_SET(smask, sm, (vk.vkCmdSetStencilCompareMask(g_cb, VK_STENCIL_FACE_FRONT_BIT, sm[0]), vk.vkCmdSetStencilCompareMask(g_cb, VK_STENCIL_FACE_BACK_BIT, sm[1]),
                         vk.vkCmdSetStencilWriteMask(g_cb, VK_STENCIL_FACE_FRONT_BIT, sm[2]), vk.vkCmdSetStencilWriteMask(g_cb, VK_STENCIL_FACE_BACK_BIT, sm[3]),
                         vk.vkCmdSetStencilReference(g_cb, VK_STENCIL_FACE_FRONT_BIT, sm[4]), vk.vkCmdSetStencilReference(g_cb, VK_STENCIL_FACE_BACK_BIT, sm[5]))); }
    int bias_bit = d->poly.mode == NVN_POLYGON_MODE_POINT ? NVN_POLYGON_OFFSET_ENABLE_POINT : d->poly.mode == NVN_POLYGON_MODE_LINE ? NVN_POLYGON_OFFSET_ENABLE_LINE : NVN_POLYGON_OFFSET_ENABLE_FILL;
    int bias = (d->poly.offset_en & bias_bit) != 0;
    DY_SET1(dbe, (uint32_t)(bias), vk.vkCmdSetDepthBiasEnable(g_cb, DY.dbe));
    { float db[3] = {bias ? d->po.units : 0, bias && V.depth_bias_clamp ? d->po.clamp : 0, bias ? d->po.factor : 0}; DY_SET(db, db, vk.vkCmdSetDepthBias(g_cb, db[0], db[1], db[2])); }
    DY_SET1(rd, (uint32_t)(d->rast_discard != 0), vk.vkCmdSetRasterizerDiscardEnable(g_cb, DY.rd));
    DY_SET(bc, d->blend_color, vk.vkCmdSetBlendConstants(g_cb, d->blend_color));
    if (!DY.valid) vk.vkCmdSetLineWidth(g_cb, 1.0f);
    int restart = di && d->restart.on;
    if (restart) { uint32_t all = di->itype == 0 ? 0xff : di->itype == 1 ? 0xffff : 0xffffffffu; if (d->restart.index != all) VRED_ONCE("primitive restart index %#x is not the all-ones index Vulkan restarts at", d->restart.index); }
    if (!mesh) DY_SET1(pr, (uint32_t)(restart), vk.vkCmdSetPrimitiveRestartEnable(g_cb, DY.pr));
    DY.valid = 1;

    PT(7);
    // The draw
    uint32_t inst = (uint32_t)((dw ? dw->instances : di->instances) > 0 ? (dw ? dw->instances : di->instances) : 1), binst = d->b.base_instance;
    if (vka_gp_split) vka_gp_draw(g_cb, p, mesh ? (sh[7] ? sh[7]->key : 0) : sh[0] ? sh[0]->key : 0, fs ? fs->key : 0, ind ? 0 : (uint64_t)(uint32_t)(dw ? dw->count : di->count) * inst, ind != 0);   // DM_VK_GPUPROF_SPLIT
    if (mesh) {   // count workgroups of the task shader, or of the mesh shader without one (the device's limit per dimension: maxTaskWorkGroupCount / maxMeshWorkGroupCount, 65535 on the Adreno 840)
        static uint32_t largest; if ((uint32_t)dw->count > largest) { largest = (uint32_t)dw->count; static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VLOG("mesh draw: largest so far %u workgroups (task %016llx mesh %016llx)", largest, sh[6] ? (unsigned long long)sh[6]->key : 0ull, (unsigned long long)sh[7]->key); }
        uint32_t maxg = sh[6] ? V.task_groups : V.mesh_groups;
        if ((uint32_t)dw->count > maxg) SKIP("mesh draw of %d workgroups: more than the device's %u in one dimension; skipped", dw->count, maxg);
        vk.vkCmdDrawMeshTasksEXT(g_cb, (uint32_t)dw->count, 1, 1);
    } else if (expand) {   // quads or a line loop: the indices made here, 32-bit, in scratch memory
        uint32_t count = (uint32_t)(dw ? dw->count : di->count), *src = NULL;
        if (di && !(src = read_indices(di->ibuf, di->itype, count))) SKIP("indices at %#llx are not readable: the draw is skipped", (unsigned long long)di->ibuf);
        uint32_t nout = expand == 1 ? count / 4 * 6 : count + 1; vka_span s;
        if (!nout || !vka_scratch(4 * (VkDeviceSize)nout, 4, &s)) { free(src); return; }
        uint32_t *o = (uint32_t *)s.ptr, base = dw ? (uint32_t)dw->first : 0;
        #define IX(i) (src ? src[i] : base + (i))
        if (expand == 1) for (uint32_t q = 0; q < count / 4; q++) { static const uint8_t t[6] = {0, 1, 2, 0, 2, 3}; for (int j = 0; j < 6; j++) o[6 * q + j] = IX(4 * q + t[j]); }
        else { for (uint32_t i = 0; i < count; i++) o[i] = IX(i); o[count] = IX(0); }
        #undef IX
        free(src);
        ib_bind(s.buf, s.off, VK_INDEX_TYPE_UINT32);
        vk.vkCmdDrawIndexed(g_cb, nout, inst, 0, di ? di->base_vertex : 0, binst);
    } else if (dw) {
        if (ind) { VkBuffer b; VkDeviceSize o; if (!vka_addr(ind, 16, &b, &o, NULL) || (o & 3)) SKIP("indirect draw arguments at %#llx are not in placed guest RAM: skipped", (unsigned long long)ind);
                   VRED_ONCE("indirect draws: the driver segment's base instance (c0 + 0x644) reads the record's, not the GPU's"); vk.vkCmdDrawIndirect(g_cb, b, o, 1, 16); }
        else vk.vkCmdDraw(g_cb, (uint32_t)dw->count, inst, (uint32_t)dw->first, binst);
    } else {
        VkBuffer ib; VkDeviceSize io; VkIndexType it = di->itype == 0 ? VK_INDEX_TYPE_UINT8 : di->itype == 1 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
        size_t isz = di->itype == 0 ? 1 : di->itype == 1 ? 2 : 4;
        if (it == VK_INDEX_TYPE_UINT8 && !V.idx8) {   // 8-bit indices on a device without them: widened on the CPU
            if (ind) SKIP("an indirect draw of 8-bit indices on a device without them: skipped");
            uint32_t *src = read_indices(di->ibuf, 0, (uint32_t)di->count); vka_span s;
            if (!src || !vka_scratch(2 * (VkDeviceSize)di->count + 4, 4, &s)) { free(src); SKIP("8-bit indices at %#llx not readable: skipped", (unsigned long long)di->ibuf); }
            for (int i = 0; i < di->count; i++) ((uint16_t *)s.ptr)[i] = (uint16_t)src[i];
            free(src); ib = s.buf; io = s.off; it = VK_INDEX_TYPE_UINT16;
            if (restart) VRED_ONCE("primitive restart with widened 8-bit indices");
        } else if (!vka_addr(di->ibuf, ind ? isz : isz * (uint64_t)(di->count > 0 ? di->count : 1), &ib, &io, NULL) || io % isz) SKIP("index buffer at %#llx is not in placed guest RAM: skipped", (unsigned long long)di->ibuf);
        ib_bind(ib, io, it);
        if (ind) { VkBuffer b; VkDeviceSize o; if (!vka_addr(ind, 20, &b, &o, NULL) || (o & 3)) SKIP("indirect draw arguments at %#llx are not in placed guest RAM: skipped", (unsigned long long)ind);
                   vk.vkCmdDrawIndexedIndirect(g_cb, b, o, 1, 20); }
        else vk.vkCmdDrawIndexed(g_cb, (uint32_t)di->count, inst, 0, di->base_vertex, binst);
    }
    g_draws++;
    PT(8);
    if (vka_nan_on() >= 2 && vka_nan_split(R.cim, R.nci)) vka_rp_end();   // DM_VK_NANSCAN=2: a scan inside the game's pass
}

// ---------- The movie library's NDVE video network (NintendoSDK movie, Movie/Ndve/Ndve-Nvn2-vid_seq5.dat) ----------
// A movie frame is one colour conversion (NV12 -> RGBA8 1920x1080), then 12 dispatches of an HMMA network and a last pass that writes the 2560x1440 movie output
// (docs/reference/shaders.md "NDVE video enhancement packs"). On Turnip that network cannot run right and poisons other programs: d15370a382630d4e needs about 96 KiB of
// private memory a fiber, which tu_setup_pvtmem rounds to 128 KiB, past what SP_xS_PVT_MEM_PARAM's 8-bit MEMSIZEPERITEM (512-byte units, at most 130560) holds: its
// field wraps to 0, so every spill of that program, and of every program that spills per fiber and is made after it (they share the device's one private-memory buffer
// and its layout), reads 0 (docs/reference/shaders.md "Programs made after one with more than 64 KiB of private memory lose their spills on Turnip"). The
// network also takes 550-800 ms of GPU a frame on the Adreno 830. DM_VK_NDVE_DIRECT (default 1) scales the converted frame into the output with a linear blit
// instead: the network's dispatches are not recorded and their pipelines never made (vka_pso.c prewarm too). The picture differs from the network's by about one
// level in 255 on average (measured on the Mac, nxnative #1375). Matched by the pack's shader keys after a conversion (the movie library may submit the conversion and the network in different frames: the replay's frame count is not
// a boundary); the last pass ends the match. Anything else runs as recorded
static const uint64_t k_ndve_convert = 0x49dff010b9832251ull, k_ndve_final = 0x4f41ccf24f163e0dull;
static const uint64_t k_ndve_net[] = {0xba079f68ae288dbaull, 0xa8b4fb48fee0ccf8ull, 0xe2eebd7ba18fe35full, 0xe6be82d835fbe4d3ull, 0x9442573ee4ace011ull,
                                      0x0a56002cdc7e1fceull, 0xd15370a382630d4eull, 0x7f7ff0f36b54d90full, 0x5a7d69b067a7f011ull};
static struct { int src; unsigned long long skipped, scaled; } g_ndve;   // src: the texture ID the last conversion wrote, until the last pass (replay thread only)
int vka_ndve_direct(void) { return VKA_ENV("DM_VK_NDVE_DIRECT", 1) != 0; }
static int ndve_net(uint64_t key) {   // a network or last-pass key the direct path replaces
    if (!vka_ndve_direct()) return 0;
    if (key == k_ndve_final) return 1;
    for (size_t i = 0; i < sizeof k_ndve_net / sizeof *k_ndve_net; i++) if (key == k_ndve_net[i]) return 1;
    return 0;
}
static const uint64_t k_ndsr = 0xd99fbeeb53d38847ull;   // agl's NDSR upscaler network (DM_VK_NDSR_STANDIN, below)
static int ndsr_standin_on(void) { return VKA_ENV("DM_VK_NDSR_STANDIN", 0) != 0; }
// A compute key whose dispatches the replay replaces (the NDVE network, the NDSR network under DM_VK_NDSR_STANDIN): its pipeline is not made ahead (vka_pso.c pw_main), so its
// private memory never sizes Turnip's per-fiber buffer. One that still runs (the switch turned off live, a dispatch the replacement refuses) is made at its dispatch
int vka_ndve_replaced(uint64_t key) { return ndve_net(key) || (key == k_ndsr && ndsr_standin_on()); }
// The one storage image a compute shader writes (from the walk), 0 when it has none or several
static int cs_image_tid(uint8_t *dim) {
    int tid = 0, n = 0;
    for (int i = 0; i < g_wk.n[5]; i++) if (g_wk.r[5][i].what == VKA_R_IMG && g_wk.r[5][i].tid) { tid = g_wk.r[5][i].tid; if (dim) *dim = g_wk.r[5][i].dim; n++; }
    return n == 1 ? tid : 0;
}
// The last pass's output: the converted frame blitted into it (linear filter, level 0, whole images). 0 = not possible here (the pass then runs as recorded)
static int ndve_scale(vka_dstate *d, vka_shader *const sh[VKA_NSTAGE]) {
    walk_all(d, sh);
    uint8_t dim = 0; int dst = cs_image_tid(&dim);
    if (!dst || !vka_tex_id_view(dst, g_cb, dim, 1, NULL)) return 0;   // the output made as the dispatch would bind it: memory of its own, storage use
    vka_img *s = vka_tex_id_img(g_ndve.src, g_cb), *t = vka_tex_id_img(dst, g_cb);
    if (!s || !t || s == t) return 0;
    int ok = s->aspect == VK_IMAGE_ASPECT_COLOR_BIT && t->aspect == VK_IMAGE_ASPECT_COLOR_BIT && s->samples == 1 && t->samples == 1 && !s->base && !t->base && s->depth == 1 && t->depth == 1
          && s->f.kind == 0 && t->f.kind == 0 && (s->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) && (t->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    if (!ok) { VRED_ONCE("NDVE movie: the converted frame (texture ID %d) or the output (texture ID %d) cannot be blitted: the network's last pass runs as recorded", g_ndve.src, dst); return 0; }
    VkImageBlit b = {{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {(int32_t)s->w, (int32_t)s->h, 1}}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {{0, 0, 0}, {(int32_t)t->w, (int32_t)t->h, 1}}};
    vk.vkCmdBlitImage(g_cb, s->img, VK_IMAGE_LAYOUT_GENERAL, t->img, VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_LINEAR);
    vka_barrier(g_cb);
    vka_img_written(t);
    if (!g_ndve.scaled++) VLOG("NDVE movie: the converted frame (serial %llu, %ux%u) scaled into the movie output (serial %llu, %ux%u) by a linear blit; the network's %llu dispatches before it not recorded (DM_VK_NDVE_DIRECT=0 runs them)",
                               (unsigned long long)s->serial, s->w, s->h, (unsigned long long)t->serial, t->w, t->h, g_ndve.skipped);
    return 1;
}

// ---------- agl's NDSR upscaler network replaced by a bilinear stand-in (DM_VK_NDSR_STANDIN, default 0) ----------
// Ocarina of Time's d99fbeeb53d38847 (agl_ndsr_vr_light.pck) upscales the frame, 683x384 at the dynamic-resolution floor, into a B10G11R11 storage image (2048x1152,
// 1536x864): an HMMA network of about 46 ms of GPU a frame on the Adreno 830 under Turnip, whose 4,608 bytes of per-fiber private memory also size the device's spill
// buffer (docs/reference/shaders.md "agl's NDSR upscaler replaced by a bilinear stand-in on the Vulkan replay"). Its input texture holds the square root of the tonemapped
// frame (an E5B9G9R9 view of a B10G11R11 image); one term of the network is that input sampled bilinearly at each output texel's place, from the constants in its uniform
// buffer (c[1].xy the output origin, c[4].zw the workgroup tile, c[6].xy the extent its writes keep to, c[6].zw the UV step per output texel). With the switch on, that
// term squared is written instead
// (selftest/ndsr.comp) and the network is not recorded, its pipeline never made (vka_ndve_replaced). The picture is softer than the network's (no detail added); a
// workaround for the network's cost, which is the driver's (spills, per-fiber layout). A dispatch whose bindings are not the network's runs as recorded (a [RED] line)
static struct { int st; VkPipeline p; VkPipelineLayout lay; VkDescriptorSetLayout dsl; VkSampler samp; uint64_t seen[32]; unsigned nseen; unsigned long long n; } g_ndsr;
static int ndsr_ready(void) {
    if (g_ndsr.st) return g_ndsr.st > 0;
    g_ndsr.st = -1;
    if (!V.push_desc) { VRED("NDSR stand-in: the device has no push descriptors: d99fbeeb53d38847 runs as recorded"); return 0; }
    VkPhysicalDeviceFeatures pf; vk.vkGetPhysicalDeviceFeatures(V.pd, &pf);   // (ndsr.comp writes its output without a declared format; the device enables every supported feature)
    if (!pf.shaderStorageImageWriteWithoutFormat) { VRED("NDSR stand-in: the device has no shaderStorageImageWriteWithoutFormat: d99fbeeb53d38847 runs as recorded"); return 0; }
    VkDescriptorSetLayoutBinding b[3] = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
                                         {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = 3, .pBindings = b};
    VkPushConstantRange pc = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo li = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &g_ndsr.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pc};
    VkSamplerCreateInfo si = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof SPV_NDSR_COMP, .pCode = SPV_NDSR_COMP};
    VkShaderModule m = VK_NULL_HANDLE;
    VkResult r = vk.vkCreateDescriptorSetLayout(V.dev, &dl, NULL, &g_ndsr.dsl);
    if (!r) r = vk.vkCreatePipelineLayout(V.dev, &li, NULL, &g_ndsr.lay);
    if (!r) r = vk.vkCreateSampler(V.dev, &si, NULL, &g_ndsr.samp);
    if (!r) r = vk.vkCreateShaderModule(V.dev, &mi, NULL, &m);
    if (!r) { VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"}, .layout = g_ndsr.lay};
              r = vk.vkCreateComputePipelines(V.dev, VK_NULL_HANDLE, 1, &ci, NULL, &g_ndsr.p); }
    if (m) vk.vkDestroyShaderModule(V.dev, m, NULL);
    if (r) { VRED("NDSR stand-in: its layout, sampler or pipeline: VkResult %d: d99fbeeb53d38847 runs as recorded", r); return 0; }
    g_ndsr.st = 1; return 1;
}
// The format a dispatch's view of texture ID tid reads (vka_tex.c view_of, as vka_nanscan.c works it out): the registration's NVN format as Vulkan (storage: the image format,
// texture: the view format), the image's own where its format list lacks that one
static VkFormat ndsr_view_fmt(int tid, const vka_img *im, int storage) {
    int ireg = 0, q = vka_tex_id_fmt(tid, &ireg); vka_fmt f;
    if (!q || !vka_fmt_of(q, &f)) return im->f.vk;
    VkFormat want = storage || ireg ? f.vk : f.view;
    return vka_view_listed(im, want) ? want : im->f.vk;
}
// The game's AA before the network (DM_VK_NDSR_STANDIN_AA, live, default 0; Ocarina of Time 1): f8965bffbdf51d78 finds edges, d17eefaff4f0bc23 ORs SMAA blend weights into an
// RGBA8 image, a335c1db6270c355 (agl_ndsr_vr_light) blends the scene with them, takes the square root and packs it as RGB9E5 words into the image d99 reads (the same texture
// serial each frame: session-2026-11-07 ex/j4.logcat, aa/WIRE.md). With the switch the stand-in reads the image a335c wrote in this frame: its own input when that is the
// same image (logged), else a335c's output in the input's view format. No a335c dispatch in this frame (the chain absent, skipped, another game): the input, as before
static const uint64_t k_aa_blend = 0xa335c1db6270c355ull, k_aa_weights = 0xd17eefaff4f0bc23ull;
static struct { uint64_t fr_blend, fr_weights; int out; int last; unsigned nlog; } g_ndsr_aa;   // out: a335c's storage image texture ID in frame fr_blend; last: the last case logged
static void ndsr_aa_note(uint64_t key) {   // after a recorded dispatch (the walk is its own)
    if (key == k_aa_weights) g_ndsr_aa.fr_weights = vka_frame() + 1;
    else if (key == k_aa_blend) { g_ndsr_aa.fr_blend = vka_frame() + 1; g_ndsr_aa.out = cs_image_tid(NULL); }
}
static const char *ndsr_fmt_name(VkFormat f, char *b, size_t n) {
    const char *s = f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ? "B10G11R11" : f == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 ? "E5B9G9R9" : f == VK_FORMAT_R16G16B16A16_SFLOAT ? "RGBA16F"
                  : f == VK_FORMAT_R8G8B8A8_UNORM ? "RGBA8" : f == VK_FORMAT_R32_UINT ? "R32_UINT" : NULL;
    if (s) snprintf(b, n, "%s", s); else snprintf(b, n, "VkFormat %d", (int)f);
    return b;
}
// 1 = the stand-in was recorded in place of the dispatch; 0 = the network runs as recorded (logged)
static int ndsr_standin(vka_dstate *d, vka_shader *const sh[VKA_NSTAGE], const p_dispatch *dp) {
    if (dp->ind) { VRED_ONCE("NDSR stand-in: an indirect dispatch of d99fbeeb53d38847 (its extent unknown here): it runs as recorded"); return 0; }
    if (!ndsr_ready()) return 0;
    walk_all(d, sh);
    if (g_wk.n[5] < 0) return 0;   // (the dispatch path logs the refusal)
    int in = 0, out = 0, nin = 0, nout = 0, ncb = 0; uint8_t din = 0, dout = 0; const vka_res *cb = NULL;
    for (int i = 0; i < g_wk.n[5]; i++) { const vka_res *r = &g_wk.r[5][i];
        if (r->what == VKA_R_TEX && r->tid && r->desc != VK_DESCRIPTOR_TYPE_SAMPLER) { in = r->tid; din = r->dim; nin++; }
        else if (r->what == VKA_R_IMG && r->tid) { out = r->tid; dout = r->dim; nout++; }
        else if (r->what == VKA_R_BUF && r->desc == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) { cb = r; ncb++; } }
    if (nin != 1 || nout != 1 || ncb > 1 || din != 1 || dout != 1) {   // (dim 1: a 2D float view, as the network declares both)
        VRED_ONCE("NDSR stand-in: the dispatch of d99fbeeb53d38847 binds %d textures, %d storage images and %d constant buffers (shapes %#x, %#x), not one 2D texture, one 2D image and its constants: it runs as recorded",
                  nin, nout, ncb, din, dout);
        return 0;
    }
    VkImageLayout ls, lt;
    VkImageView vs = vka_tex_id_view(in, g_cb, din, 0, &ls), vt = vka_tex_id_view(out, g_cb, dout, 1, &lt);   // as the dispatch would bind them (the output with storage use)
    vka_img *s = vka_tex_id_img(in, g_cb), *t = vka_tex_id_img(out, g_cb);
    if (!vs || !vt || !s || !t || s == t || s->f.kind != 0 || t->f.kind != 0) {
        VRED_ONCE("NDSR stand-in: d99fbeeb53d38847's input (texture ID %d) or output (texture ID %d) has no float view here: it runs as recorded", in, out);
        return 0;
    }
    if (VKA_ENV("DM_VK_NDSR_STANDIN_AA", 0)) {   // ("The game's AA before the network" above): 1 its input is a335c's output, 2 a335c's output read instead, 3 none this frame
        uint64_t fr = vka_frame() + 1; int how = 3; vka_img *a = NULL;
        if (g_ndsr_aa.fr_blend == fr && g_ndsr_aa.out && (a = vka_tex_id_img(g_ndsr_aa.out, g_cb))) {
            VkFormat vf = ndsr_view_fmt(in, s, 0); VkImageView av;
            if (a == s) how = 1;
            else if (a->w == s->w && a->h == s->h && a->f.kind == 0 && a->samples == 1 && !a->base && vka_view_listed(a, vf)
                     && (av = vka_view(a, vf, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1, NULL))) { how = 2; vs = av; ls = VK_IMAGE_LAYOUT_GENERAL; }
        }
        if (how != g_ndsr_aa.last && g_ndsr_aa.nlog < 16) { g_ndsr_aa.nlog++;
            VLOG("NDSR stand-in AA: %s (input image serial %llu%s%llu; d17eefaff4f0bc23 %s in this frame; DM_VK_NDSR_STANDIN_AA)",
                 how == 1 ? "its input is a335c1db6270c355's output of this frame: the game's AA result is what it upscales"
                 : how == 2 ? "its input is not a335c1db6270c355's output of this frame: that output is read instead"
                 : a ? "a335c1db6270c355's output of this frame has another size, format list or shape: the input is read" : "no a335c1db6270c355 dispatch in this frame: the input is read",
                 (unsigned long long)s->serial, a ? ", a335c output image serial " : "", a ? (unsigned long long)a->serial : 0ull, g_ndsr_aa.fr_weights == fr ? "ran" : "did not run"); }
        g_ndsr_aa.last = how;
    }
    // Its constants as a uniform buffer (mode 1); without one the whole input over the whole output (mode 0), the zero buffer bound
    VkDescriptorBufferInfo bi = {g_zero.buf, g_zero.off, 112}; uint32_t mode = 0;
    if (cb && cb->addr && ubo_desc(cb->addr, cb->size, &bi) && bi.range >= 112) mode = 1;
    else bi = (VkDescriptorBufferInfo){g_zero.buf, g_zero.off, 112};
    VkDescriptorImageInfo ii[2] = {{g_ndsr.samp, vs, ls}, {VK_NULL_HANDLE, vt, lt}};
    VkWriteDescriptorSet w[3] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii[0]},
                                 {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &ii[1]},
                                 {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 2, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &bi}};
    uint32_t pc[4] = {(uint32_t)(dp->x > 0 ? dp->x : 0), (uint32_t)(dp->y > 0 ? dp->y : 0), mode, 0};
    vk.vkCmdBindPipeline(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ndsr.p);
    vk.vkCmdPushDescriptorSetKHR(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_ndsr.lay, 0, 3, w); vka_dyn_invalidate();   // the draws' last push is no longer bound (push_same)
    vk.vkCmdPushConstants(g_cb, g_ndsr.lay, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vk.vkCmdDispatch(g_cb, (t->w + 7) / 8, (t->h + 7) / 8, 1);   // every texel of level 0; the shader keeps to the view's size and, in mode 1, to the network's own extent
    if (vka_gp_active) { char b[96]; snprintf(b, sizeof b, "dispatch %016llx NDSR stand-in %ux%u", (unsigned long long)k_ndsr, t->w, t->h); vka_gp_pass(b); }   // DM_VK_GPUPROF
    vka_barrier(g_cb);
    vka_img_written(t);
    g_ndsr.n++;
    // Once per output image: the sizes and formats, and the network's constants as the CPU sees them at recording (the GPU reads them in order)
    unsigned k = 0; while (k < g_ndsr.nseen && g_ndsr.seen[k] != t->serial) k++;
    if (k == g_ndsr.nseen && g_ndsr.nseen < sizeof g_ndsr.seen / sizeof *g_ndsr.seen) {
        g_ndsr.seen[g_ndsr.nseen++] = t->serial;
        uint32_t c[28] = {0}; int rd = mode && cpu_read(cb->addr, c, sizeof c);
        float uvs[2]; memcpy(uvs, &c[26], sizeof uvs);
        char fi[24], fo[24], cs[200];
        if (rd) snprintf(cs, sizeof cs, "origin %d,%d, extent %d,%d, tile %d,%d, UV step %g,%g (%.1f x %.1f output texels to the input)", (int)c[4], (int)c[5], (int)c[24], (int)c[25],
                         (int)c[18], (int)c[19], (double)uvs[0], (double)uvs[1], uvs[0] > 0 ? 1.0 / uvs[0] : 0.0, uvs[1] > 0 ? 1.0 / uvs[1] : 0.0);
        else snprintf(cs, sizeof cs, mode ? "not readable by the CPU" : "no constant buffer bound: the whole input over the whole output");
        VLOG("NDSR stand-in: %ux%u %s -> %ux%u %s (input serial %llu image VkFormat %d, output serial %llu, groups %d,%d; constants: %s); d99fbeeb53d38847 not run (DM_VK_NDSR_STANDIN=0 runs it)",
             s->w, s->h, ndsr_fmt_name(ndsr_view_fmt(in, s, 0), fi, sizeof fi), t->w, t->h, ndsr_fmt_name(ndsr_view_fmt(out, t, 1), fo, sizeof fo), (unsigned long long)s->serial, (int)s->f.vk,
             (unsigned long long)t->serial, dp->x, dp->y, cs);
    }
    return 1;
}

int dm_sdk_ounce(void);
static int g_nvn2 = -1;
void vka_set_nvn2(int on) { g_nvn2 = on != 0; }
int vka_nvn2(void) { if (g_nvn2 < 0) g_nvn2 = dm_sdk_ounce(); return g_nvn2; }   // first called at a dispatch or a capture, after the modules are loaded
void vka_dispatch(vka_dstate *d, const p_dispatch *dp) {
    if (!g_cb || !draws_on() || !statics()) { g_skipped++; return; }
    if (!d->inited) vka_dstate_reset(d);
    vka_rp_end();
    vka_shader *sh[VKA_NSTAGE] = {0}; unsigned bad = 0;
    if (!resolve(d->cprog, 5, &sh[5], &bad)) SKIP("dispatch skipped: no shader library");
    if (bad) { g_skipped++; return; }
    if (!sh[5]) SKIP("dispatch with no compute shader bound: skipped");
    // The NDVE network after its conversion in this frame (DM_VK_NDVE_DIRECT): its dispatches not recorded, its last pass a blit (ndve_scale)
    int ndve = 0;
    if (!dp->ind && vka_ndve_direct()) {
        uint64_t key = sh[5]->key;
        int after = g_ndve.src != 0;
        if (key == k_ndve_convert) ndve = 1;
        else if (after && key == k_ndve_final) { if (ndve_scale(d, sh)) { g_ndve.src = 0; g_dispatches++; return; } }
        else if (after && ndve_net(key)) { g_ndve.skipped++; g_dispatches++; return; }
        else if (ndve_net(key)) VRED_ONCE("NDVE movie: network shader %016llx dispatched with no colour conversion before it: it runs as recorded (its pipeline is made)", (unsigned long long)key);
    }
    // agl's NDSR network replaced by its bilinear term (DM_VK_NDSR_STANDIN)
    if (sh[5]->key == k_ndsr && ndsr_standin_on() && ndsr_standin(d, sh, dp)) { g_dispatches++; return; }
    vka_layout lay; if (!vka_layout_of(sh, 1, &lay)) SKIP("dispatch skipped: no pipeline layout");
    VkPipeline p = vka_cs_pipeline(sh[5], &lay);
    if (vka_nan_on()) vka_tex_watch_dispatch(sh[5]->key);
    if (!p) { g_skipped++; return; }
    d->b.dispatch[0] = (uint32_t)dp->x; d->b.dispatch[1] = (uint32_t)dp->y; d->b.dispatch[2] = (uint32_t)dp->z; d->b.nvn2 = (uint32_t)vka_nvn2();
    // DM_VK_DISPATCH_LOG=2: the compute storage slots the game set (bound ones are read through descriptors; a program with raw global memory, the pool table, may also
    // start its raw accesses from any of them: the translator binds a slot only when it can follow the address, and through the table an address in no placed segment
    // is lost: gmem_sink_check)
    if (VKA_ENV("DM_VK_DISPATCH_LOG", 0) >= 2) {
        int raw = 0; for (unsigned i = 0; i < sh[5]->e->bind_count; i++) raw |= sh[5]->b[i].kind == DMSH_VK_KIND_GMEM_MAP;
        if (!raw) VLOG("dispatch %016llx: no raw global memory", (unsigned long long)sh[5]->key);
        for (int k = 0; k < 16; k++) { const vka_range *r = &d->b.ssbo[5][k]; if (!r->addr && !r->size) continue;
            char a[200], z[200]; void vka_mem_describe(uint64_t, char *, size_t); vka_mem_describe(r->addr, a, sizeof a); vka_mem_describe(r->addr + (r->size ? r->size - 1 : 0), z, sizeof z);
            VLOG("dispatch %016llx storage slot %d: %#llx + %#llx, %s; first byte: %s; last byte: %s", (unsigned long long)sh[5]->key, k, (unsigned long long)r->addr, (unsigned long long)r->size,
                 vka_addr(r->addr, r->size ? r->size : 1, NULL, NULL, NULL) ? "placed" : "NOT in one placed segment", a, z); }
    }
    VkBuffer ib = VK_NULL_HANDLE; VkDeviceSize io = 0;
    if (dp->ind && (!vka_addr(dp->ind, 12, &ib, &io, NULL) || (io & 3))) SKIP("indirect dispatch arguments at %#llx are not in placed guest RAM: skipped", (unsigned long long)dp->ind);
    uint32_t sb = spill_bytes(sh[5]); uint64_t ns = 0;   // shared memory in the spill buffer: the slices this dispatch binds (spill_buf)
    if (sb) {
        if (!spill_buf(sb)) SKIP("dispatch of %016llx skipped: no shared memory spill buffer of %u bytes", (unsigned long long)sh[5]->key, sb);
        uint64_t cap = g_spill_cap / sb, total = dp->ind ? cap : (uint64_t)(dp->x > 0 ? dp->x : 0) * (uint64_t)(dp->y > 0 ? dp->y : 0) * (uint64_t)(dp->z > 0 ? dp->z : 0);
        ns = total < cap ? total : cap; if (!ns) ns = 1;
        g_spill_range = ns * sb;
        if (dp->ind) VRED_ONCE("indirect dispatch of a program with its shared memory in the spill buffer (%016llx): its %llu slices are shared by workgroups that many apart", (unsigned long long)sh[5]->key, (unsigned long long)ns);
    }
    c0_at c0 = {0};
    walk_all(d, sh);
    { uint64_t tp_ = VKA_RP_T0(); int ps = pool_scan(d, sh); if (ps) vka_tex_pool_write(g_cb, ps == 2); VKA_RP_ADD(VKA_RPX_POOLSCAN, tp_); }   // (no pass is open at a dispatch)
    if (!bind_sets(d, sh, &lay, VK_PIPELINE_BIND_POINT_COMPUTE, &c0)) { g_skipped++; return; }
    if (dp->ind && c0.buf) {   // the group counts the GPU wrote, into c0's dispatch words before the dispatch reads them (nvn_mtl.m g_c0_dim_buf); NVN2 reads them at c0 + 4 too
        VkBufferCopy c[2] = {{io, c0.off + DMSH_C0_DISPATCH, 12}, {io, c0.off + DMSH_C0_DISPATCH_NVN2, 12}};
        vk.vkCmdCopyBuffer(g_cb, ib, c0.buf, d->b.nvn2 ? 2 : 1, c);
        vka_barrier(g_cb);
    }
    // DM_VK_DISPATCH_LOG=1: one line per dispatch (its index in the submission's run of dispatches since the last log reset, key, groups, workgroup, shared memory, flags);
    // DM_VK_DISPATCH_SKIP=<from>[,<to>]: dispatches from..to (default: to the end) are recorded without their vkCmdDispatch, to find the one that writes a wrong picture
    uint64_t vka_frame(void); static unsigned long long nd; static uint64_t fr = ~0ull; static int lg; if (vka_frame() != fr) { fr = vka_frame(); lg = (int)VKA_ENV("DM_VK_DISPATCH_LOG", 0); nd = 0; }
    unsigned long long di = nd++; const char *sk = getenv("DM_VK_DISPATCH_SKIP"); int skip = 0;
    if (sk && *sk) { char *e; unsigned long long a = strtoull(sk, &e, 10), b = *e == ',' ? strtoull(e + 1, NULL, 10) : ~0ull; skip = di >= a && di <= b; }
    if (lg) VLOG("dispatch %llu: %016llx groups %d,%d,%d%s local %u,%u,%u shared %u flags %#x%s", di, (unsigned long long)sh[5]->key, dp->x, dp->y, dp->z, dp->ind ? " (indirect)" : "",
                 sh[5]->e->local[0], sh[5]->e->local[1], sh[5]->e->local[2], sh[5]->e->shared, sh[5]->e->flags, skip ? " SKIPPED" : "");
    if (skip) { vka_barrier(g_cb); g_dispatches++; return; }
    vk.vkCmdBindPipeline(g_cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    if (dp->ind) vk.vkCmdDispatchIndirect(g_cb, ib, io);
    else if (dp->x > 0 && dp->y > 0 && dp->z > 0) {
        if (sb && (uint64_t)dp->x * dp->y * dp->z > ns) spill_dispatch((uint32_t)dp->x, (uint32_t)dp->y, (uint32_t)dp->z, ns);
        else vk.vkCmdDispatch(g_cb, (uint32_t)dp->x, (uint32_t)dp->y, (uint32_t)dp->z);
    }
    if (vka_gp_active) { char b[96]; snprintf(b, sizeof b, "dispatch %llu %016llx groups %d,%d,%d%s", di, (unsigned long long)sh[5]->key, dp->x, dp->y, dp->z, dp->ind ? " indirect" : ""); vka_gp_pass(b); }   // DM_VK_GPUPROF: the step it ends
    vka_barrier(g_cb);   // (also frees the shared memory spill buffer for the next dispatch)
    if (vka_nan_on() && vka_nan_cs(sh[5]->key)) vka_nan_dispatch(g_cb, sh[5]->key, g_wk.r[5], g_wk.n[5]);   // DM_VK_NANSCAN_CS: the images it binds (the walk is this dispatch's)
    { int sl; VkBuffer sbuf; VkDeviceSize soff; const vka_range *r;   // the ctl bufshot: what this dispatch left in one storage slot
      if (vka_ctl_bufshot_want(di, &sl)) { r = &d->b.ssbo[5][sl];
          if (r->size && vka_addr(r->addr, r->size, &sbuf, &soff, NULL)) vka_ctl_bufshot_record(g_cb, sh[5]->key, r->addr, sbuf, soff, r->size);
          else VLOG("ctl: bufshot: dispatch %llu (%016llx) storage slot %d is %#llx + %#llx, not in one placed segment", di, (unsigned long long)sh[5]->key, sl, (unsigned long long)r->addr, (unsigned long long)r->size); } }
    if (ndve == 1) g_ndve.src = cs_image_tid(NULL);   // (the walk is this dispatch's)
    if (sh[5]->key == k_aa_blend || sh[5]->key == k_aa_weights) ndsr_aa_note(sh[5]->key);   // (DM_VK_NDSR_STANDIN_AA; the walk is this dispatch's)
    g_dispatches++;
}
void vka_draw_stats(uint64_t *draws, uint64_t *skipped, uint64_t *dispatches, uint64_t *pipelines) {
    if (draws) *draws = g_draws; if (skipped) *skipped = g_skipped; if (dispatches) *dispatches = g_dispatches; if (pipelines) *pipelines = 0;
}
#endif
