// DM_VK_NANSCAN=1: a diagnostic that finds the pass that first writes NaN or infinite texels. After every render pass, each float colour attachment's level / layer is
// scanned by a compute shader (selftest/nanscan.comp) that counts NaN and infinite texels and keeps the first one's position; when the submission is done the completion
// thread logs each pass with any, with its frame, its place in the submission, the attachment (serial, format, size) and the vertex / fragment shaders of its draws.
// DM_VK_NANSCAN=2 also splits long passes to find the draw: the Vulkan pass is ended (and reopened by the next draw, its attachments loaded) after every
// DM_VK_NANSCAN_STEP draws (default 64), in passes into the image DM_VK_NANSCAN_SERIAL (default any), of the game's pass whose index is in DM_VK_NANSCAN_DRAWS=<from>-<to> (default all), at most MAXSPLIT times a submission,
// and only the scans after which an attachment has more NaN / infinite texels than at its previous scan in the submission are logged, with the pass's draw range.
// Bisect with a coarse step over the whole pass, then step 1 over the range it names. Costs a dispatch per attachment per pass (per split with 2): for finding a
// cause, not for play
// DM_VK_NANSCAN_CS=<key>[,<key>…] (hex compute shader keys, at most 16) or all (with DM_VK_NANSCAN=1 or 2): after each dispatch of such a shader, level 0 (every layer, at most 6)
// of each image it binds, the storage images it writes and the textures it samples, is scanned through the format the dispatch's own view reads it in (the texture ID's
// registration format, its view's or its texture's, as Vulkan: vka_tex.c view_of; an E5B9G9R9 texture over a B10G11R11 image reads as RGB9E5; the image's own format
// where its format list lacks the view's, as vka_view makes the dispatch's view then) by selftest/nanscan_cs.comp (float view formats: NaN, infinite, all-zero and big
// texels, the largest component) or nanscan_9e5.comp (an R32_UINT view, its words decoded as RGB9E5: a frame packed through an R32_UINT storage view; no NaN or infinity
// there, so all-zero, big and the largest component tell). Big: a component of 256 or more. Logged per dispatch key and binding: its first two scans, each scan whose
// verdict changes (bad: a NaN or infinite texel, a big one, or all texels zero) and every 120th, at most 600 lines; each binding is first described once per image and
// view format (texture ID, the view format scanned, image; a texture ID registered anew each frame over the same image is not described again). Other view formats
// (sint, other uint, depth, 3D, multisampled) are named and not scanned. Costs a
// dispatch per image per matched dispatch. DM_VK_NANSCAN=3 is DM_VK_NANSCAN_CS alone: no pass scans (they share a submission's 1024 results with the dispatch scans)
#if defined(__ANDROID__)
#include "vka.h"
#include "vka_selftest_spv.h"
#include <stdlib.h>
#include <string.h>

enum { NBATCH = 16, MAXR = 1024, NKEY = 8, MAXSPLIT = 256, NCS = 16 };
typedef struct { uint32_t nan, inf, first, zero, maxv, big, fbig, pad; } res_t;   // nanscan.comp / nanscan3.comp write the first three; nanscan_cs / nanscan_9e5 all
// cs, out, mode: a DM_VK_NANSCAN_CS scan (mode 1 float, 2 R32_UINT read as RGB9E5; 0 = a pass or DM_VK_NANSCAN_FS input scan), the dispatch's key, whether the binding is a storage image;
// fmt / vkf: the format scanned (a dispatch scan: the binding's view format, NVN and Vulkan), ivk: a dispatch scan's image's own Vulkan format
typedef struct { uint64_t frame, serial; uint32_t pass, fmt, w, h, lv, layer, d0, ndraws, nkeys; uint64_t vs[NKEY], fs[NKEY]; int bind, tid; uint64_t cs; uint32_t vkf; uint8_t out, mode; uint32_t ivk; } rec_t;
typedef struct { rec_t r[MAXR]; int n; res_t *res; VkDeviceSize off; } batch_t;
static batch_t g_b[NBATCH]; static unsigned g_bi; static vka_span g_res;
static VkPipeline g_pipe, g_pipe3, g_fetch, g_pcs, g_p9e5; static VkPipelineLayout g_lay; static VkDescriptorSetLayout g_dsl;
static uint32_t g_pass; static uint64_t g_pvs[NKEY], g_pfs[NKEY]; static uint32_t g_pn, g_pdraws, g_d0;   // g_pdraws: draws of the game's pass so far, g_d0: the segment's first
static uint64_t g_probe; static int g_split, g_nsplit; static uint32_t g_step = 64, g_from, g_to = UINT32_MAX; static uint64_t g_only;
static _Atomic int g_lines; static int g_csonly;   // g_csonly: DM_VK_NANSCAN=3

int vka_nan_on(void) {
    static int v = -1; if (v >= 0) return v;
    const char *e = getenv("DM_VK_NANSCAN"); v = e && atoi(e) > 0 && V.push_desc ? atoi(e) : 0;
    if (v == 3) { g_csonly = 1; v = 1; VLOG("DM_VK_NANSCAN=3: no pass scans, only DM_VK_NANSCAN_CS's dispatch scans"); return v; }   // 1: the pass splits and DM_VK_NANSCAN_FS (>= 2) stay off
    if (v >= 2) {
        const char *s = getenv("DM_VK_NANSCAN_STEP"), *r = getenv("DM_VK_NANSCAN_DRAWS"); if (s && atoi(s) > 0) g_step = (uint32_t)atoi(s);
        const char *o = getenv("DM_VK_NANSCAN_SERIAL"); if (o) g_only = strtoull(o, NULL, 10);
        if (r) { char *x; g_from = (uint32_t)strtoul(r, &x, 10); if (*x == '-') g_to = (uint32_t)strtoul(x + 1, NULL, 10); }
        VLOG("DM_VK_NANSCAN=2: passes split every %u draws in draws %u-%u of a pass%s, at most %d splits a submission", g_step, g_from, g_to, g_only ? " into DM_VK_NANSCAN_SERIAL" : "", MAXSPLIT);
    } else if (v) VLOG("DM_VK_NANSCAN: every pass's float attachments are scanned for NaN / infinite texels");
    return v;
}
// DM_VK_NANSCAN=2, after a draw: 1 = end the Vulkan pass now for a scan (the game's pass goes on)
int vka_nan_split(vka_img *const *ims, int n) {
    uint32_t i = g_pdraws;   // the draws of the game's pass so far
    if (i <= g_from || i > g_to || (i - g_from) % g_step) return 0;
    if (g_nsplit >= MAXSPLIT) { static _Atomic int n; if (!atomic_exchange(&n, 1)) VLOG("NaN scan: %d splits in one submission, no more in it (at draw %u of pass %u)", MAXSPLIT, i, g_pass); return 0; }
    if (g_only) { int k = 0; while (k < n && !(ims[k] && ims[k]->serial == g_only)) k++; if (k == n) return 0; }   // DM_VK_NANSCAN_SERIAL: only passes into that image
    g_nsplit++; g_split = 1; return 1;
}
static int ready(void) {
    static int st; if (st) return st > 0;
    st = -1;
    if (!vka_static((VkDeviceSize)NBATCH * MAXR * sizeof(res_t), V.sb_align, &g_res)) return 0;
    VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = 2, .pBindings = b};
    VkPushConstantRange pc = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo li = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &g_dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pc};
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof SPV_NANSCAN_COMP, .pCode = SPV_NANSCAN_COMP};
    VkResult r;
    if ((r = vk.vkCreateDescriptorSetLayout(V.dev, &dl, NULL, &g_dsl)) || (r = vk.vkCreatePipelineLayout(V.dev, &li, NULL, &g_lay))) { VRED("NaN scan: %d", r); return 0; }
    for (int i = 0; i < 5; i++) {   // 2D-array views (attachments, 2D / cube inputs), 3D inputs, the ASTC check's fetch, DM_VK_NANSCAN_CS's float and RGB9E5-in-R32_UINT scans
        VkShaderModule m = VK_NULL_HANDLE;
        if (i == 1) { mi.codeSize = sizeof SPV_NANSCAN3_COMP; mi.pCode = SPV_NANSCAN3_COMP; }
        if (i == 2) { mi.codeSize = sizeof SPV_FETCH_COMP; mi.pCode = SPV_FETCH_COMP; }
        if (i == 3) { mi.codeSize = sizeof SPV_NANSCAN_CS_COMP; mi.pCode = SPV_NANSCAN_CS_COMP; }
        if (i == 4) { mi.codeSize = sizeof SPV_NANSCAN_9E5_COMP; mi.pCode = SPV_NANSCAN_9E5_COMP; }
        if ((r = vk.vkCreateShaderModule(V.dev, &mi, NULL, &m))) { VRED("NaN scan: %d", r); return 0; }
        VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"}, .layout = g_lay};
        VkPipeline *out[5] = {&g_pipe, &g_pipe3, &g_fetch, &g_pcs, &g_p9e5};
        r = vk.vkCreateComputePipelines(V.dev, VK_NULL_HANDLE, 1, &ci, NULL, out[i]);
        vk.vkDestroyShaderModule(V.dev, m, NULL);
        if (r) { VRED("NaN scan pipeline: %d", r); return 0; }
    }
    for (int i = 0; i < NBATCH; i++) { g_b[i].res = (res_t *)(g_res.ptr + (size_t)i * MAXR * sizeof(res_t)); g_b[i].off = g_res.off + (VkDeviceSize)i * MAXR * sizeof(res_t); }
    st = 1; return 1;
}
void vka_nan_draw(uint64_t vs, uint64_t fs) {
    g_pdraws++;   // counted before the draw records: a draw skipped after this still has its index
    for (uint32_t i = 0; i < g_pn; i++) if (g_pvs[i] == vs && g_pfs[i] == fs) return;
    if (g_pn < NKEY) { g_pvs[g_pn] = vs; g_pfs[g_pn] = fs; g_pn++; }
}
uint64_t vka_frame(void);
// The ASTC check (once per run with DM_VK_NANSCAN): an 8x8 ASTC 4x4 texture of four void-extent (constant colour) blocks, sampled by the device and logged. A void-extent block's
// colour is UNORM16 when its bit 9 (dynamic range) is 0 and FP16 when it is 1 (ASTC specification, "Void-Extent Blocks")
static const uint16_t g_vcol[4][4] = {{0x1212, 0x1212, 0x1212, 0xffff}, {0x8000, 0x4000, 0xc000, 0x8000}, {0x3c00, 0x3800, 0x3400, 0x3c00}, {0xffff, 0x0000, 0xffff, 0xffff}};
static const int g_vhdr[4] = {0, 0, 1, 0};
static vka_span g_fetch_out; static int g_astc_state;   // 1 = recorded, report at the next commit
static void astc_report(void *arg) {
    (void)arg; const float *t = (const float *)g_fetch_out.ptr;
    for (int b = 0; b < 4; b++) {
        const float *v = t + 4 * ((b / 2) * 4 * 8 + (b % 2) * 4);   // the block's first texel in the 8x8 texture
        VLOG("NaN scan: ASTC check: void-extent block %d (%s, colour %04x %04x %04x %04x) samples as (%g, %g, %g, %g)", b, g_vhdr[b] ? "HDR: FP16" : "LDR: UNORM16",
             g_vcol[b][0], g_vcol[b][1], g_vcol[b][2], g_vcol[b][3], v[0], v[1], v[2], v[3]);
    }
}
static void astc_check(VkCommandBuffer cb) {
    static uint8_t blocks[4 * 16]; static tex_t t;
    for (int b = 0; b < 4; b++) {
        uint64_t lo = 0x1fc | (uint64_t)g_vhdr[b] << 9 | 3ull << 10 | 0xfffffffffffffull << 12, hi = 0;   // extent all ones: the whole texture
        for (int c = 0; c < 4; c++) hi |= (uint64_t)g_vcol[b][c] << (16 * c);
        memcpy(blocks + 16 * b, &lo, 8); memcpy(blocks + 16 * b + 8, &hi, 8);
    }
    memset(&t, 0, sizeof t);
    t.d.target = NVN_TEXTURE_TARGET_2D; t.d.format = 0x79; t.d.width = 8; t.d.height = 8; t.d.depth = 1; t.d.levels = 1; t.d.flags = NVN_TEXTURE_FLAGS_LINEAR; t.d.stride = 32;
    nvn_tex_layout(&t.d, &t.L); t.mem = blocks; t.serial = nvn_new_serial();
    vka_img *im = vka_tex_get(&t, cb);
    if (!im || !vka_static(64 * 16, V.sb_align, &g_fetch_out)) { VLOG("NaN scan: ASTC check: no test texture"); return; }
    memset(g_fetch_out.ptr, 0xff, 64 * 16);
    VkImageView v = vka_view(im, im->f.view, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1, NULL);
    if (!v) return;
    VkDescriptorImageInfo ii = {vka_samp_default(), v, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo bi = {g_fetch_out.buf, g_fetch_out.off, 64 * 16};
    VkWriteDescriptorSet wr[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii},
                                  {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi}};
    uint32_t pc[4] = {8, 8, 0, 0};
    vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_fetch);
    vk.vkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_lay, 0, 2, wr); vka_dyn_invalidate();   // the draws' last push is no longer bound (vka_draw.c push_same)
    vk.vkCmdPushConstants(cb, g_lay, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vk.vkCmdDispatch(cb, 1, 1, 1);
    vka_barrier(cb);
    g_astc_state = 1;
}
// The scan shader pipe over view v into result k of batch b (pc: size, layer or depth; groups of 8x8 over the size, z of them)
static void scan_dispatch(VkCommandBuffer cb, VkPipeline pipe, VkImageView v, const batch_t *b, int k, const uint32_t pc[4], uint32_t z) {
    VkDescriptorImageInfo ii = {vka_samp_default(), v, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo bi = {g_res.buf, b->off + (VkDeviceSize)k * sizeof(res_t), sizeof(res_t)};
    VkWriteDescriptorSet wr[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii},
                                  {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi}};
    vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vk.vkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_lay, 0, 2, wr); vka_dyn_invalidate();   // the draws' last push is no longer bound (vka_draw.c push_same)
    vk.vkCmdPushConstants(cb, g_lay, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
    vk.vkCmdDispatch(cb, (pc[0] + 7) / 8, (pc[1] + 7) / 8, z);
}
// One dispatch: level lv, layer l0 of im into the batch's next result
static int scan_one(VkCommandBuffer cb, batch_t *b, vka_img *im, uint32_t lv, uint32_t l0, uint32_t pass, int bind, int tid) {
    if (b->n == MAXR) return 0;
    int v3 = im->d.target == NVN_TEXTURE_TARGET_3D;   // a whole level of a 3D texture (inputs only)
    VkImageView v = vka_view(im, im->f.view, v3 ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_ASPECT_COLOR_BIT, lv, 1, v3 ? 0 : l0, 1, NULL);
    if (!v) return 0;
    uint32_t w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1, dz = v3 ? (im->depth >> lv ? im->depth >> lv : 1) : 1;
    int k = b->n++;
    b->res[k] = (res_t){0, 0, 0xffffffffu, 0, 0, 0, 0xffffffffu, 0};
    b->r[k] = (rec_t){vka_frame(), im->serial, pass, (uint32_t)im->d.format, w, h, lv, l0, g_d0, g_pdraws, g_pn, {0}, {0}, bind, tid, 0, (uint32_t)im->f.vk, 0, 0, 0};
    memcpy(b->r[k].vs, g_pvs, sizeof g_pvs); memcpy(b->r[k].fs, g_pfs, sizeof g_pfs);
    uint32_t pc[4] = {w, h, v3 ? dz : 0, 0};
    scan_dispatch(cb, v3 ? g_pipe3 : g_pipe, v, b, k, pc, dz);
    return 1;
}
// DM_VK_NANSCAN_FS=<fragment shader key> (with DM_VK_NANSCAN=2): the first draw of each frame with that fragment shader has its sampled float textures scanned before it
// (every level and layer, outside the pass): which input holds NaN / infinite texels. 1 = this draw is one (the caller ends the pass and calls vka_nan_inputs)
int vka_nan_probe(uint64_t fs) {
    static uint64_t last = ~0ull; static int init; uint64_t key = g_probe;
    if (!init) { init = 1; const char *e = getenv("DM_VK_NANSCAN_FS"); if (e) g_probe = key = strtoull(e, NULL, 16); }
    if (!key || fs != key || last == vka_frame()) return 0;
    last = vka_frame(); return 1;
}
void vka_nan_inputs(VkCommandBuffer cb, vka_img *const *ims, const int *binds, const int *tids, int n) {
    batch_t *b = &g_b[g_bi % NBATCH];
    if (!ready()) return;
    int any = 0;
    for (int i = 0; i < n; i++) {
        vka_img *im = ims[i];
        if (!im || im->f.kind != 0 || im->samples > 1 || !(im->usage & VK_IMAGE_USAGE_SAMPLED_BIT)) continue;
        uint32_t nl = im->d.target == NVN_TEXTURE_TARGET_3D ? 1 : im->layers;
        for (uint32_t lv = 0; lv < im->levels && lv < 14; lv++) for (uint32_t l = 0; l < nl && l < 6; l++) any |= scan_one(cb, b, im, lv, l, g_pass, binds[i], tids[i]);
    }
    if (any) vka_barrier(cb);
    static _Atomic int k; if (atomic_fetch_add(&k, 1) < 2) VLOG("NaN scan: frame %llu: %d inputs of fragment shader %016llx scanned before its first draw (only those with NaN / infinite texels are logged)", (unsigned long long)vka_frame(), n, (unsigned long long)g_probe);
}
// After a pass ends (its writes made visible by the caller's barrier): one dispatch per float colour attachment
void vka_nan_pass(VkCommandBuffer cb, vka_img *const *ims, const uint32_t *lvs, const uint32_t *l0s, int n) {
    batch_t *b = &g_b[g_bi % NBATCH];
    int split = g_split; g_split = 0;
    uint32_t pass = split ? g_pass : g_pass++;
    if (!ready()) return;
    if (!g_astc_state) { g_astc_state = -1; astc_check(cb); }
    int any = 0;
    for (int i = 0; i < n && !g_csonly; i++) {
        vka_img *im = ims[i];
        if (!im || im->f.kind != 0 || im->samples > 1 || im->d.target == NVN_TEXTURE_TARGET_3D || !(im->usage & VK_IMAGE_USAGE_SAMPLED_BIT)) continue;
        if (b->n == MAXR) { static _Atomic int f; if (!atomic_exchange(&f, 1)) VLOG("NaN scan: %d scans in one submission, the rest of its pass scans dropped (first at pass %u)", MAXR, pass); break; }
        any |= scan_one(cb, b, im, lvs[i], l0s[i], pass, -1, 0);
    }
    if (any) vka_barrier(cb);
    g_pn = 0;
    if (split) g_d0 = g_pdraws; else { g_pdraws = 0; g_d0 = 0; }
}
// DM_VK_NANSCAN_CS: 1 = the dispatch of compute shader key is one whose images are scanned after it (vka_nan_dispatch)
static uint64_t g_cs[NCS]; static int g_ncs = -1, g_cs_all;
int vka_nan_cs(uint64_t key) {
    if (g_ncs < 0) {
        g_ncs = 0; const char *e = getenv("DM_VK_NANSCAN_CS");
        if (e && !strcmp(e, "all")) g_cs_all = 1;
        else for (const char *p = e; p && *p && g_ncs < NCS;) { char *x; uint64_t k = strtoull(p, &x, 16); if (x == p) break; if (k) g_cs[g_ncs++] = k; p = *x == ',' ? x + 1 : x; }
        if (g_cs_all || g_ncs) VLOG("DM_VK_NANSCAN_CS: the images of %s are scanned after each dispatch through the dispatch's view formats (float: NaN / infinite / all-zero / big texels; an R32_UINT view read as RGB9E5)", g_cs_all ? "every compute shader" : "the listed compute shaders");
        else if (g_csonly) VLOG("DM_VK_NANSCAN=3 without DM_VK_NANSCAN_CS: nothing is scanned");
    }
    if (g_cs_all) return 1;
    for (int i = 0; i < g_ncs; i++) if (g_cs[i] == key) return 1;
    return 0;
}
// One DM_VK_NANSCAN_CS dispatch: level 0, layer l of im through a view in vf (NVN nf; mode 1 float, 2 R32_UINT as RGB9E5)
static int scan_cs(VkCommandBuffer cb, batch_t *b, vka_img *im, VkFormat vf, uint32_t nf, uint32_t l, uint64_t key, int bind, int tid, int out, int mode) {
    if (b->n == MAXR) { static _Atomic int n; if (!atomic_exchange(&n, 1)) VLOG("NaN scan: %d scans in one submission, no more in it (DM_VK_NANSCAN_CS)", MAXR); return 0; }
    VkImageView v = vka_view(im, vf, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, l, 1, NULL);
    if (!v) return 0;
    int k = b->n++;
    b->res[k] = (res_t){0, 0, 0xffffffffu, 0, 0, 0, 0xffffffffu, 0};
    b->r[k] = (rec_t){vka_frame(), im->serial, g_pass, nf, im->w, im->h, 0, l, 0, 0, 0, {0}, {0}, bind, tid, key, (uint32_t)vf, (uint8_t)out, (uint8_t)mode, (uint32_t)im->f.vk};
    uint32_t pc[4] = {im->w, im->h, 0, 0};
    scan_dispatch(cb, mode == 2 ? g_p9e5 : g_pcs, v, b, k, pc, 1);
    return 1;
}
// After a dispatch of a DM_VK_NANSCAN_CS shader (its writes made visible by the caller's barrier): its compute stage's walk r[0..n) (the images it binds), scanned
void vka_nan_dispatch(VkCommandBuffer cb, uint64_t key, const vka_res *r, int n) {
    batch_t *b = &g_b[g_bi % NBATCH];
    if (!ready()) return;
    // seen: the bindings described, by dispatch key, binding, image and view format (not texture ID: a game that registers its frame's texture anew each frame has a new ID
    // every frame over the same image, which filled the table in a few frames and left later bindings undescribed)
    static struct { uint64_t key, serial; int bind; uint32_t nf; } seen[64]; static int nseen;
    struct { vka_img *im; VkFormat vf; } done[64]; int nd = 0, any = 0;
    for (int i = 0; i < n; i++) {
        if ((r[i].what != VKA_R_IMG && r[i].what != VKA_R_TEX) || r[i].tid <= 0) continue;
        int out = r[i].what == VKA_R_IMG;
        vka_img *im = vka_tex_id_img(r[i].tid, cb);
        // The format the dispatch's view reads the image in (vka_tex.c view_of): the registration's NVN format (its view's, else its texture's) as Vulkan, the storage
        // format for an image binding, the view format (an sRGB twin) for a texture; where the image's format list lacks it, vka_view makes the view in the image's own
        // format, and so does the scan (what the dispatch reads). An E5B9G9R9 texture over a B10G11R11 image is scanned as RGB9E5, not as the image's B10G11R11
        // (an image registration's view is in the storage format too, as view_of's e->img; rnf: the format asked for, nf: the one read, the image's where the list lacks it)
        uint32_t nf = 0, rnf = 0; vka_fmt f = {0}; int listed = 1, ireg = 0;
        if (im) { int q = vka_tex_id_fmt(r[i].tid, &ireg); if (q && vka_fmt_of(q, &f)) nf = (uint32_t)q; else { f = im->f; nf = im->d.format; } }
        rnf = nf;
        VkFormat want = out || ireg ? f.vk : f.view, vf = want;
        if (im && !vka_view_listed(im, want)) { listed = 0; vf = im->f.vk; f = im->f; nf = im->d.format; }
        int mode = !im ? 0 : f.kind == 0 ? 1 : f.kind == 1 && vf == VK_FORMAT_R32_UINT ? 2 : 0;
        const char *skip = !im ? "no image" : !mode ? "not a float or R32_UINT view format" : im->samples > 1 ? "multisampled" : im->d.target == NVN_TEXTURE_TARGET_3D ? "3D"
                         : !(im->usage & VK_IMAGE_USAGE_SAMPLED_BIT) ? "no sampled use" : NULL;
        uint64_t ser = im ? im->serial : 0;
        int s = 0; while (s < nseen && !(seen[s].key == key && seen[s].bind == (int)r[i].binding && seen[s].serial == ser && seen[s].nf == rnf)) s++;
        if (s == nseen && nseen == 64) { static int full; if (!full) { full = 1; VLOG("NaN scan: 64 dispatch bindings described, later new bindings are not described (dispatch %016llx binding %u first)", (unsigned long long)key, r[i].binding); } }
        if (s == nseen && nseen < 64) {   // each binding described once per image and view format (its texture ID's registration and view, and the image the scan reads)
            seen[nseen].key = key; seen[nseen].bind = (int)r[i].binding; seen[nseen].serial = ser; seen[nseen++].nf = rnf;
            char why[320]; vka_tex_id_desc(r[i].tid, why, sizeof why);
            int view = vka_tex_id_view(r[i].tid, cb, r[i].dim, out, NULL) != VK_NULL_HANDLE;   // (as bind_sets asked for it: none = the shader read an unbound descriptor, not this image)
            char vfs[160]; snprintf(vfs, sizeof vfs, listed ? "NVN %#x VkFormat %d" : "NVN %#x VkFormat %d, NOT in the image's format list: the dispatch and the scan read it as the image's VkFormat %d", rnf, (int)want, (int)vf);
            VLOG("NaN scan: dispatch %016llx %s binding %u dim %#x texture ID %d (later IDs of this image and view format are not described again): %s; the dispatch's view of it: %s; "
                 "scanned through view format %s; image serial %llu Vulkan format %d (NVN %#x) %ux%u, %u layers, %u levels%s%s%s",
                 (unsigned long long)key, out ? "storage image" : "texture", r[i].binding, r[i].dim, r[i].tid, why, view ? "made" : "NONE (the binding reads as unbound)", vfs, (unsigned long long)ser, im ? (int)im->f.vk : 0,
                 im ? (unsigned)im->d.format : 0u, im ? im->w : 0, im ? im->h : 0, im ? im->layers : 0, im ? im->levels : 0, mode == 2 ? ", R32_UINT read as RGB9E5" : "", skip ? "; not scanned: " : "", skip ? skip : "");
        }
        if (skip) continue;
        int j = 0; while (j < nd && !(done[j].im == im && done[j].vf == vf)) j++;
        if (j < nd) continue;   // (one image bound twice in one view format: scanned once, as its first binding)
        if (nd < 64) { done[nd].im = im; done[nd++].vf = vf; }
        for (uint32_t l = 0; l < im->layers && l < 6; l++) any |= scan_cs(cb, b, im, vf, nf, l, key, r[i].binding, r[i].tid, out, mode);
    }
    if (any) vka_barrier(cb);
}
// A DM_VK_NANSCAN_CS result, logged on the verdict's changes (report, completion side)
static void report_cs(const rec_t *c, const res_t *r) {
    static struct { uint64_t cs; int bind; uint32_t layer, n; int bad; } st[128]; static int nst; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static _Atomic int lines;
    uint32_t all = c->w * c->h;
    int bad = r->nan || r->inf || r->big || r->zero == all;
    pthread_mutex_lock(&mu);
    int s = 0; while (s < nst && !(st[s].cs == c->cs && st[s].bind == c->bind && st[s].layer == c->layer)) s++;
    if (s == nst) { if (nst == 128) { static int full; int first = !full; full = 1; pthread_mutex_unlock(&mu);   // (a dispatch key, binding and layer each; list the keys instead of all)
            if (first) VLOG("NaN scan: 128 dispatch bindings tracked, scans of later new ones are not logged (dispatch %016llx binding %d first)", (unsigned long long)c->cs, c->bind);
            return; } st[nst++] = (__typeof__(st[0])){c->cs, c->bind, c->layer, 0, -1}; }
    uint32_t n = st[s].n++; int was = st[s].bad; st[s].bad = bad;
    pthread_mutex_unlock(&mu);
    if (!(n < 2 || bad != was || n % 120 == 0) || atomic_fetch_add(&lines, 1) >= 600) return;
    float mx; memcpy(&mx, &r->maxv, 4);
    char f1[32] = "none", f2[32] = "none";
    if (r->first != 0xffffffffu) snprintf(f1, sizeof f1, "(%u, %u)", r->first & 0xffff, r->first >> 16);
    if (r->fbig != 0xffffffffu) snprintf(f2, sizeof f2, "(%u, %u)", r->fbig & 0xffff, r->fbig >> 16);
    VLOG("NaN scan: frame %llu dispatch %016llx (after pass %u of its submission) %s binding %d texture ID %d serial %llu (view VkFormat %u, NVN %#x; image VkFormat %u; %ux%u layer %u%s), scan %u: %s; "
         "%u NaN + %u infinite texels (first %s), %u of %u all-zero, %u big (a component >= 256, first %s), largest component %g",
         (unsigned long long)c->frame, (unsigned long long)c->cs, c->pass, c->out ? "storage image" : "texture", c->bind, c->tid, (unsigned long long)c->serial, c->vkf, c->fmt, c->ivk, c->w, c->h, c->layer,
         c->mode == 2 ? ", R32_UINT read as RGB9E5" : "", n, bad ? "BAD" : "fine", r->nan, r->inf, f1, r->zero, all, r->big, f2, (double)mx);
}
static void report(void *arg) {
    batch_t *b = arg;
    for (int k = 0; k < b->n; k++) {
        const res_t *r = &b->res[k]; const rec_t *c = &b->r[k];
        if (c->mode) { report_cs(c, r); continue; }
        if (!r->nan && !r->inf) continue;
        if (c->bind >= 0) {
            if (atomic_load(&g_lines) >= 200 || atomic_fetch_add(&g_lines, 1) >= 200) continue;   // (continue: the batch's DM_VK_NANSCAN_CS results are still reported)
            VLOG("NaN scan: frame %llu pass %u (fragment %016llx): input binding %d texture ID %d serial %llu (format %#x %ux%u level %u layer %u): %u NaN + %u infinite texels, first at (%u, %u)",
                 (unsigned long long)c->frame, c->pass, (unsigned long long)g_probe, c->bind, c->tid, (unsigned long long)c->serial, c->fmt, c->w, c->h, c->lv, c->layer, r->nan, r->inf, r->first & 0xffff, r->first >> 16);
            continue;
        }
        if (vka_nan_on() >= 2) {   // only where the count of this attachment grew since its previous scan in this submission
            int j = k - 1; while (j >= 0 && (b->r[j].serial != c->serial || b->r[j].lv != c->lv || b->r[j].layer != c->layer)) j--;
            if (j >= 0 && b->res[j].nan + b->res[j].inf >= r->nan + r->inf) continue;
        }
        if (atomic_load(&g_lines) >= 200 || atomic_fetch_add(&g_lines, 1) >= 200) continue;   // (continue: the batch's DM_VK_NANSCAN_CS results are still reported)
        char keys[400] = ""; size_t o = 0;
        for (uint32_t i = 0; i < c->nkeys && o < sizeof keys - 40; i++) o += (size_t)snprintf(keys + o, sizeof keys - o, " %016llx/%016llx", (unsigned long long)c->vs[i], (unsigned long long)c->fs[i]);
        VLOG("NaN scan: frame %llu pass %u: %u NaN + %u infinite texels in serial %llu (format %#x %ux%u level %u layer %u), first at (%u, %u); draws %u-%u of the pass, vertex/fragment%s%s",
             (unsigned long long)c->frame, c->pass, r->nan, r->inf, (unsigned long long)c->serial, c->fmt, c->w, c->h, c->lv, c->layer, r->first & 0xffff, r->first >> 16, c->d0, c->ndraws, keys, c->nkeys == NKEY ? " …" : "");
    }
}
void vka_nan_commit(uint64_t fence) {   // the submission's batch is read when fence is done; the next submission starts a new batch
    batch_t *b = &g_b[g_bi % NBATCH];
    if (b->n) vka_after(fence, report, b);
    if (g_astc_state == 1) { g_astc_state = 2; vka_after(fence, astc_report, NULL); }
    g_bi++; g_b[g_bi % NBATCH].n = 0; g_pass = 0; g_nsplit = 0;
}
#endif
