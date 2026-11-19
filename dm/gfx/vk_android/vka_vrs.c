// NVN2 variable-rate shading (shading-rate images) via VK_KHR_fragment_shading_rate (DM_VK_VRS=1, restart only; off by default).
// NVN uses 16x16-pixel tiles storing R8_UINT palette indices (VkShadingRatePaletteEntryNV); Turnip / Adreno 840 requires 8x8-pixel
// R8_UINT attachment texels holding Vulkan encoded rates ((log2 w << 2) | log2 h). Before each eligible render pass opens, a tiny
// compute shader (vka_vrs_spv.h) expands and LUT-maps the bound NVN image into a reused per-size R8_UINT image, bracketed by
// FRAGMENT_SHADING_RATE_ATTACHMENT_READ <-> SHADER_WRITE barriers on the single replay queue.
#if defined(__ANDROID__)
#include "vka.h"
#include "vka_vrs_spv.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint64_t now_ns(void);
uint64_t vka_frame(void);

uint64_t vka_vrs_pal_hash(int count, const uint32_t *entries) {
    uint64_t h = 1469598103934665603ull;
    h ^= (uint64_t)(uint32_t)count; h *= 1099511628211ull;
    for (int i = 0; i < count; i++) { h ^= entries[i]; h *= 1099511628211ull; }
    return h;
}

// ---------- Replay stats (written on replay thread, read by DM_VK_VRS_LOG summary) ----------
static _Atomic uint64_t g_vrs_passes, g_vrs_dispatches;
static _Atomic uint64_t g_vrs_fb[VKA_VRS_FB_N];
static _Atomic uint32_t g_vrs_last_w, g_vrs_last_h;

void vka_vrs_note_fallback(int reason) {
    if (reason > VKA_VRS_FB_NONE && reason < VKA_VRS_FB_N)
        atomic_fetch_add_explicit(&g_vrs_fb[reason], 1, memory_order_relaxed);
}

// ---------- LUT mapping (VkShadingRatePaletteEntryNV -> Vulkan R8_UINT encoding, clamped to driver rates) ----------
static void nv_entry_size(uint32_t e, uint32_t *w, uint32_t *h) {
    // 0 (no invocations) and 1..5 (16/8/4/2/1 invocations per pixel) -> 1x1: never drop pixels.
    // 6=2x1, 7=1x2, 8=2x2, 9=4x2, 10=2x4, 11=4x4; unknown -> 1x1.
    switch (e) {
    case 6:  *w = 2; *h = 1; break;
    case 7:  *w = 1; *h = 2; break;
    case 8:  *w = 2; *h = 2; break;
    case 9:  *w = 4; *h = 2; break;
    case 10: *w = 2; *h = 4; break;
    case 11: *w = 4; *h = 4; break;
    default: *w = 1; *h = 1; break;
    }
}

static uint32_t clamp_encode_rate(uint32_t w, uint32_t h, uint32_t samples) {
    uint32_t bw = 1, bh = 1, best_area = 1, best_match = 0;
    VkSampleCountFlags sbit = (VkSampleCountFlags)(samples ? samples : 1);
    for (uint32_t i = 0; i < V.vrs_nrates; i++) {
        if (!(V.vrs_rates[i].samples & sbit)) continue;
        uint32_t rw = V.vrs_rates[i].w, rh = V.vrs_rates[i].h;
        if ((rw != 1 && rw != 2 && rw != 4) || (rh != 1 && rh != 2 && rh != 4)) continue;
        if (rw > w || rh > h || rw > V.vrs_max_w || rh > V.vrs_max_h) continue;
        uint32_t area = rw * rh;
        uint32_t match = (rw == w) + (rh == h);
        if (area > best_area || (area == best_area && (match > best_match || (match == best_match && rw > bw)))) {
            bw = rw; bh = rh; best_area = area; best_match = match;
        }
    }
    uint32_t lw = bw == 4 ? 2u : bw == 2 ? 1u : 0u;
    uint32_t lh = bh == 4 ? 2u : bh == 2 ? 1u : 0u;
    return (lw << 2) | lh;
}

static int build_lut(const vka_vrs_state *st, uint32_t samples, int show, uint32_t lut[16], int *raw_coarse) {
    int any_raw = 0, any_enc = 0;
    int cnt = st->pal_count[0] > 16 ? 16 : (int)st->pal_count[0];
    for (int i = 0; i < 16; i++) {
        if (i >= cnt) { lut[i] = 0; continue; }
        uint32_t w = 1, h = 1;
        nv_entry_size(st->pal[0][i], &w, &h);
        if (w > 1 || h > 1) {
            any_raw = 1;
            if (show) { w = 4; h = 4; }
        }
        uint32_t enc = clamp_encode_rate(w, h, samples);
        lut[i] = enc;
        if (enc) any_enc = 1;
    }
    if (raw_coarse) *raw_coarse = any_raw;
    return any_enc;
}

// ---------- Conversion compute pipeline + per-size converted R8_UINT image pool (replay thread only) ----------
static struct {
    int state;   // 0 uninit, 1 ready, -1 failed
    VkDescriptorSetLayout dsl;
    VkPipelineLayout lay;
    VkPipeline pipe;
} g_conv;

typedef struct {
    uint32_t w, h;
    VkImage img;
    VkDeviceMemory mem;
    VkImageView view;
    int inited;
} vrs_dst;

enum { VRS_POOL_MAX = 16 };
static vrs_dst g_pool[VRS_POOL_MAX];
static int g_npool;

static int conv_init(void) {
    if (g_conv.state) return g_conv.state > 0;
    g_conv.state = -1;
    VkDescriptorSetLayoutBinding b[2] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dl = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = 2, .pBindings = b};
    VkPushConstantRange pc = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 84};
    VkPipelineLayoutCreateInfo li = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &g_conv.dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pc};
    VkShaderModuleCreateInfo mi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof vka_vrs_conv_spv, .pCode = vka_vrs_conv_spv};
    VkShaderModule mod = VK_NULL_HANDLE;
    VkResult r = vk.vkCreateDescriptorSetLayout(V.dev, &dl, NULL, &g_conv.dsl);
    if (!r) r = vk.vkCreatePipelineLayout(V.dev, &li, NULL, &g_conv.lay);
    if (!r) r = vk.vkCreateShaderModule(V.dev, &mi, NULL, &mod);
    if (!r) {
        VkComputePipelineCreateInfo ci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod, .pName = "main"},
            .layout = g_conv.lay};
        r = vk.vkCreateComputePipelines(V.dev, VK_NULL_HANDLE, 1, &ci, NULL, &g_conv.pipe);
    }
    if (mod) vk.vkDestroyShaderModule(V.dev, mod, NULL);
    if (r) {
        VRED("vrs: conversion compute pipeline failed (%d): falling back to full-rate shading", (int)r);
        return 0;
    }
    g_conv.state = 1;
    return 1;
}

static vrs_dst *pool_get(uint32_t w, uint32_t h) {
    for (int i = 0; i < g_npool; i++)
        if (g_pool[i].w == w && g_pool[i].h == h) return &g_pool[i];
    if (g_npool >= VRS_POOL_MAX) {
        static int once;
        if (!once++) VLOG("vrs: more than %d distinct shading-rate image sizes; falling back to full-rate shading for %ux%u", VRS_POOL_MAX, w, h);
        return NULL;
    }
    VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8_UINT,
        .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage img = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE;
    VkResult r = vk.vkCreateImage(V.dev, &ii, NULL, &img);
    if (!r) {
        VkMemoryRequirements mr; vk.vkGetImageMemoryRequirements(V.dev, img, &mr);
        uint32_t mt = vka_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        if (mt == UINT32_MAX) mt = vka_mem_type(mr.memoryTypeBits, 0, 0);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = mt};
        if (mt == UINT32_MAX || (r = vk.vkAllocateMemory(V.dev, &ai, NULL, &mem)) || (r = vk.vkBindImageMemory(V.dev, img, mem, 0))) {
            if (!r) r = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        }
    }
    if (!r) {
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R8_UINT, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        r = vk.vkCreateImageView(V.dev, &vi, NULL, &view);
    }
    if (r || !view) {
        if (view) vk.vkDestroyImageView(V.dev, view, NULL);
        if (img) vk.vkDestroyImage(V.dev, img, NULL);
        if (mem) vk.vkFreeMemory(V.dev, mem, NULL);
        static int once;
        if (!once++) VRED("vrs: converted %ux%u R8_UINT image creation failed (%d): falling back to full-rate shading", w, h, (int)r);
        return NULL;
    }
    vrs_dst *d = &g_pool[g_npool++];
    *d = (vrs_dst){.w = w, .h = h, .img = img, .mem = mem, .view = view, .inited = 0};
    return d;
}

VkImageView vka_vrs_prepare(VkCommandBuffer cb, const vka_vrs_state *st, uint32_t rp_w, uint32_t rp_h, uint32_t layers, uint32_t samples, uint32_t *out_w, uint32_t *out_h, int *fb_reason) {
    if (out_w) *out_w = 0;
    if (out_h) *out_h = 0;
    if (fb_reason) *fb_reason = VKA_VRS_FB_NONE;
    if (!V.vrs || !cb || !st || !st->bound) return VK_NULL_HANDLE;

    if (layers > 1) {
        if (fb_reason) *fb_reason = VKA_VRS_FB_LAYERS;
        static int once; if (!once++) VLOG("vrs: pass fallback to full rate: layers=%u > 1", layers);
        return VK_NULL_HANDLE;
    }
    if (samples > V.vrs_max_samples) {
        if (fb_reason) *fb_reason = VKA_VRS_FB_SAMPLES;
        static int once; if (!once++) VLOG("vrs: pass fallback to full rate: samples=%u > max %u", samples, V.vrs_max_samples);
        return VK_NULL_HANDLE;
    }
    int show = VKA_ENV("DM_VK_VRS_SHOW", 0) != 0;
    uint32_t lut[16]; int raw_coarse = 0;
    if (!build_lut(st, samples, show, lut, &raw_coarse)) {
        if (raw_coarse && fb_reason) *fb_reason = VKA_VRS_FB_SAMPLES;
        return VK_NULL_HANDLE;   // all entries map to 1x1: no attachment or conversion needed
    }

    vka_img *im = vka_tex_get(&st->tex, cb);
    if (!im || !im->inited || im->f.vk != VK_FORMAT_R8_UINT || im->samples != 1 || !(im->usage & VK_IMAGE_USAGE_SAMPLED_BIT)) {
        if (fb_reason) *fb_reason = VKA_VRS_FB_NOIMG;
        static int once;
        if (!once++) VLOG("vrs: pass fallback to full rate: host image unavailable (ser=%llu im=%p inited=%d vk_fmt=%d samples=%u)",
                          (unsigned long long)st->tex.serial, (void *)im, im ? im->inited : 0, im ? (int)im->f.vk : -1, im ? im->samples : 0u);
        return VK_NULL_HANDLE;
    }
    if (im->mdrop) vka_tex_whole(im, cb);
    uint32_t lv = st->has_view && st->view.set_levels ? st->view.base_level : 0;
    uint32_t l0 = st->has_view && st->view.set_layers ? st->view.min_layer : 0;
    if (!vka_img_lv(im, &lv) || l0 >= im->layers) {
        if (fb_reason) *fb_reason = VKA_VRS_FB_NOIMG;
        return VK_NULL_HANDLE;
    }
    uint32_t src_w = im->w >> lv ? im->w >> lv : 1;
    uint32_t src_h = im->h >> lv ? im->h >> lv : 1;
    uint32_t dst_w = src_w * 2, dst_h = src_h * 2;
    uint32_t need_w = (rp_w + V.vrs_texel_w - 1) / V.vrs_texel_w;
    uint32_t need_h = (rp_h + V.vrs_texel_h - 1) / V.vrs_texel_h;
    if (dst_w < need_w || dst_h < need_h || dst_w > V.props.limits.maxImageDimension2D || dst_h > V.props.limits.maxImageDimension2D) {
        if (fb_reason) *fb_reason = VKA_VRS_FB_AREA;
        static int once;
        if (!once++) VLOG("vrs: pass fallback to full rate: converted %ux%u does not cover %ux%u render area (needs %ux%u)",
                          dst_w, dst_h, rp_w, rp_h, need_w, need_h);
        return VK_NULL_HANDLE;
    }

    VkImageView src_view = vka_view(im, VK_FORMAT_R8_UINT, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, lv, 1, l0, 1, NULL);
    if (!src_view || !conv_init()) {
        if (fb_reason) *fb_reason = VKA_VRS_FB_NOIMG;
        return VK_NULL_HANDLE;
    }
    vrs_dst *dst = pool_get(dst_w, dst_h);
    if (!dst) {
        if (fb_reason) *fb_reason = VKA_VRS_FB_NOIMG;
        return VK_NULL_HANDLE;
    }

    // Order conversion write after any prior pass's FSR read of this reused image (or transition UNDEFINED -> GENERAL on first use).
    VkImageMemoryBarrier pre = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = dst->inited ? VK_ACCESS_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR : 0,
        .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .oldLayout = dst->inited ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = dst->img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    vk.vkCmdPipelineBarrier(cb,
        dst->inited ? VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &pre);
    dst->inited = 1;

    struct { uint32_t lut[16], src_w, src_h, dst_w, dst_h, flip_y; } pc;
    _Static_assert(sizeof pc == 84, "vrs push constants size mismatch");
    memcpy(pc.lut, lut, sizeof lut);
    pc.src_w = src_w; pc.src_h = src_h; pc.dst_w = dst_w; pc.dst_h = dst_h; pc.flip_y = (uint32_t)(V.vrs_flip != 0);

    VkDescriptorImageInfo ii[2] = {{vka_samp_default(), src_view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, dst->view, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet w[2] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii[0]},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &ii[1]},
    };
    vk.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_conv.pipe);
    vk.vkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_conv.lay, 0, 2, w);
    vk.vkCmdPushConstants(cb, g_conv.lay, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vk.vkCmdDispatch(cb, (dst_w + 7) / 8, (dst_h + 7) / 8, 1);

    VkImageMemoryBarrier post = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = dst->img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR,
                            0, 0, NULL, 0, NULL, 1, &post);

    atomic_fetch_add_explicit(&g_vrs_passes, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_vrs_dispatches, 1, memory_order_relaxed);
    atomic_store_explicit(&g_vrs_last_w, dst_w, memory_order_relaxed);
    atomic_store_explicit(&g_vrs_last_h, dst_h, memory_order_relaxed);
    if (out_w) *out_w = dst_w;
    if (out_h) *out_h = dst_h;

    static int first_pass;
    if (!first_pass++) {
        char lb[128]; int o = 0; lb[0] = 0;
        for (int i = 0; i < 16 && o < (int)sizeof lb - 8; i++) o += snprintf(lb + o, sizeof lb - (size_t)o, "%s%x", i ? " " : "", lut[i]);
        VLOG("vrs: first pass applied: src %ux%u (ser=%llu gpu=%d mode=%u) -> dst %ux%u over %ux%u (samples=%u flip=%d show=%d lut=[%s])",
             src_w, src_h, (unsigned long long)im->serial, im->gpu_owned, (unsigned)im->mode, dst_w, dst_h, rp_w, rp_h, samples, V.vrs_flip, show, lb);
    }
    return dst->view;
}

// ---------- NVN call logging (DM_VK_VRS_LOG=1; called from hle_nvn.c on the recording thread) ----------
// Logs the first 200 calls, then only when a new palette hash or new image size appears, plus a once-a-second summary.
enum { VRS_CB_MAX = 64, VRS_SIZE_MAX = 32, VRS_PAL_MAX = 64, VRS_LOG_CAP = 200 };
typedef struct { uint32_t w, h, nvn_fmt; VkFormat vk_fmt; } vrs_size_ent;
static pthread_mutex_t g_vrs_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_vrs_cbs[VRS_CB_MAX];
static uint32_t g_vrs_ncb;
static vrs_size_ent g_vrs_sizes[VRS_SIZE_MAX];
static uint32_t g_vrs_nsizes;
static uint64_t g_vrs_pals[VRS_PAL_MAX];
static uint32_t g_vrs_npals;
static uint8_t g_vrs_once[3];
static uint64_t g_vrs_logged;
static uint64_t g_vrs_sum_t0, g_vrs_sum_f0, g_vrs_w_bind, g_vrs_w_unbind, g_vrs_w_pal, g_vrs_w_mode;
static uint64_t g_vrs_p0, g_vrs_d0, g_vrs_fb0[VKA_VRS_FB_N];

static void vrs_note_cb(uint64_t cmd) {
    for (uint32_t i = 0; i < g_vrs_ncb && i < VRS_CB_MAX; i++) if (g_vrs_cbs[i] == cmd) return;
    if (g_vrs_ncb < VRS_CB_MAX) g_vrs_cbs[g_vrs_ncb] = cmd;
    g_vrs_ncb++;
}

static int vrs_note_size(uint32_t w, uint32_t h, uint32_t nvn_fmt, VkFormat vk_fmt) {
    for (uint32_t i = 0; i < g_vrs_nsizes && i < VRS_SIZE_MAX; i++)
        if (g_vrs_sizes[i].w == w && g_vrs_sizes[i].h == h && g_vrs_sizes[i].nvn_fmt == nvn_fmt) return 0;
    if (g_vrs_nsizes < VRS_SIZE_MAX) g_vrs_sizes[g_vrs_nsizes] = (vrs_size_ent){w, h, nvn_fmt, vk_fmt};
    g_vrs_nsizes++;
    return 1;
}

static int vrs_note_pal(uint64_t h) {
    for (uint32_t i = 0; i < g_vrs_npals && i < VRS_PAL_MAX; i++) if (g_vrs_pals[i] == h) return 0;
    if (g_vrs_npals < VRS_PAL_MAX) g_vrs_pals[g_vrs_npals] = h;
    g_vrs_npals++;
    return 1;
}

static void vrs_summary_tick(void) {
    uint64_t now = now_ns(), fr = vka_frame();
    if (!g_vrs_sum_t0) { g_vrs_sum_t0 = now; g_vrs_sum_f0 = fr; return; }
    if (now - g_vrs_sum_t0 < 1000000000ull) return;
    uint64_t df = fr > g_vrs_sum_f0 ? fr - g_vrs_sum_f0 : 0;
    uint64_t tot = g_vrs_w_bind + g_vrs_w_unbind + g_vrs_w_pal + g_vrs_w_mode;
    char sb[256]; int o = 0; sb[0] = 0;
    uint32_t show = g_vrs_nsizes < 4 ? g_vrs_nsizes : 4;
    for (uint32_t i = 0; i < show && o < (int)sizeof sb - 32; i++)
        o += snprintf(sb + o, sizeof sb - (size_t)o, "%s%ux%u(fmt=%#x/vk=%d)", i ? ", " : "",
                      g_vrs_sizes[i].w, g_vrs_sizes[i].h, g_vrs_sizes[i].nvn_fmt, (int)g_vrs_sizes[i].vk_fmt);
    if (g_vrs_nsizes > show && o < (int)sizeof sb - 8) snprintf(sb + o, sizeof sb - (size_t)o, ", ...");
    if (V.vrs) {
        uint64_t p = atomic_load_explicit(&g_vrs_passes, memory_order_relaxed);
        uint64_t dp = p - g_vrs_p0; g_vrs_p0 = p;
        uint64_t d = atomic_load_explicit(&g_vrs_dispatches, memory_order_relaxed);
        uint64_t dd = d - g_vrs_d0; g_vrs_d0 = d;
        uint64_t fbd[VKA_VRS_FB_N] = {0}, fbt = 0;
        for (int i = 1; i < VKA_VRS_FB_N; i++) {
            uint64_t v = atomic_load_explicit(&g_vrs_fb[i], memory_order_relaxed);
            fbd[i] = v - g_vrs_fb0[i]; g_vrs_fb0[i] = v; fbt += fbd[i];
        }
        uint32_t lw = atomic_load_explicit(&g_vrs_last_w, memory_order_relaxed);
        uint32_t lh = atomic_load_explicit(&g_vrs_last_h, memory_order_relaxed);
        VLOG("vrs summary: %.1f calls/frame (%llu over %llu frames: bind=%llu unbind=%llu pal=%llu mode=%llu) distinct_cb=%u sizes=%u[%s] distinct_pal=%u | applied: passes=%llu conv_dispatches=%llu conv_size=%ux%u fb_draws=%llu(interlock=%llu viewport=%llu samples=%llu layers=%llu area=%llu noimg=%llu)",
             df ? (double)tot / (double)df : (double)tot, (unsigned long long)tot, (unsigned long long)df,
             (unsigned long long)g_vrs_w_bind, (unsigned long long)g_vrs_w_unbind, (unsigned long long)g_vrs_w_pal, (unsigned long long)g_vrs_w_mode,
             g_vrs_ncb, g_vrs_nsizes, sb, g_vrs_npals,
             (unsigned long long)dp, (unsigned long long)dd, lw, lh, (unsigned long long)fbt,
             (unsigned long long)fbd[VKA_VRS_FB_INTERLOCK], (unsigned long long)fbd[VKA_VRS_FB_VIEWPORT],
             (unsigned long long)fbd[VKA_VRS_FB_SAMPLES], (unsigned long long)fbd[VKA_VRS_FB_LAYERS],
             (unsigned long long)fbd[VKA_VRS_FB_AREA], (unsigned long long)fbd[VKA_VRS_FB_NOIMG]);
    } else {
        VLOG("vrs summary: %.1f calls/frame (%llu over %llu frames: bind=%llu unbind=%llu pal=%llu mode=%llu) distinct_cb=%u sizes=%u[%s] distinct_pal=%u",
             df ? (double)tot / (double)df : (double)tot, (unsigned long long)tot, (unsigned long long)df,
             (unsigned long long)g_vrs_w_bind, (unsigned long long)g_vrs_w_unbind, (unsigned long long)g_vrs_w_pal, (unsigned long long)g_vrs_w_mode,
             g_vrs_ncb, g_vrs_nsizes, sb, g_vrs_npals);
    }
    g_vrs_sum_t0 = now; g_vrs_sum_f0 = fr;
    g_vrs_w_bind = g_vrs_w_unbind = g_vrs_w_pal = g_vrs_w_mode = 0;
}

void dm_gfx_vrs_bind(uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3, uint64_t x4, uint64_t x5,
                     const tex_t *tex, int tex_valid, const view_t *view, int has_view, const uint64_t raw[4], int raw_ok) {
    int vlog = VKA_ENV("DM_VK_VRS_LOG", 0);
    vka_fmt vf = {0}; VkFormat vk_fmt = VK_FORMAT_UNDEFINED;
    if (vlog && tex_valid && vka_ok() && vka_fmt_of((int)tex->d.format, &vf)) vk_fmt = vf.vk;

    pthread_mutex_lock(&g_vrs_mu);
    vrs_note_cb(x0);
    int new_size = 0;
    if (!x1) g_vrs_w_unbind++;
    else {
        g_vrs_w_bind++;
        if (tex_valid) new_size = vrs_note_size(tex->d.width, tex->d.height, (uint32_t)tex->d.format, vk_fmt);
    }
    if (!vlog) {
        if (!V.vrs && !g_vrs_once[0]++)
            VLOG("nvnCommandBufferBindShadingRateImage: shading-rate images are recorded but not applied (full-rate shading; DM_VK_VRS=1 enables, DM_VK_VRS_LOG=1 for details)");
    } else {
        if (g_vrs_logged < VRS_LOG_CAP || new_size) {
            g_vrs_logged++;
            if (!x1) {
                VLOG("vrs BindShadingRateImage: unbind (x0=%#llx x1=%#llx x2=%#llx)",
                     (unsigned long long)x0, (unsigned long long)x1, (unsigned long long)x2);
            } else if (tex_valid) {
                char vb[96]; vb[0] = 0;
                if (has_view) snprintf(vb, sizeof vb, " view(lv=%u+%u ly=%u+%u fmt=%#x tgt=%u)", view->base_level, view->nlevels, view->min_layer, view->nlayers, view->format, view->target);
                VLOG("vrs BindShadingRateImage%s: tex ser=%llu fmt=%#x(vk=%d) %ux%ux%u lv=%u tgt=%u fl=%#x%s",
                     new_size && g_vrs_logged > VRS_LOG_CAP ? " [new size]" : "",
                     (unsigned long long)tex->serial, (unsigned)tex->d.format, (int)vk_fmt, tex->d.width, tex->d.height, tex->d.depth, tex->d.levels, (unsigned)tex->d.target, tex->d.flags, vb);
            } else {
                VLOG("vrs BindShadingRateImage: tex=INVALID raw=%d[%#llx %#llx %#llx %#llx] (x0=%#llx x1=%#llx x2=%#llx x3=%#llx x4=%#llx x5=%#llx)",
                     raw_ok, (unsigned long long)raw[0], (unsigned long long)raw[1], (unsigned long long)raw[2], (unsigned long long)raw[3],
                     (unsigned long long)x0, (unsigned long long)x1, (unsigned long long)x2, (unsigned long long)x3, (unsigned long long)x4, (unsigned long long)x5);
            }
        }
        vrs_summary_tick();
    }
    pthread_mutex_unlock(&g_vrs_mu);
}

void dm_gfx_vrs_palette(uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3, uint64_t x4, uint64_t x5,
                        int vp, int count, uint64_t ptr, const char *ptr_reg, const uint32_t *entries, int nread) {
    (void)x1; (void)x2; (void)x3; (void)x4; (void)x5;
    int vlog = VKA_ENV("DM_VK_VRS_LOG", 0);
    int nstore = nread < 0 ? 0 : (nread > VKA_VRS_MAX_PAL ? VKA_VRS_MAX_PAL : nread);
    uint64_t h = vka_vrs_pal_hash(nstore, entries);

    pthread_mutex_lock(&g_vrs_mu);
    vrs_note_cb(x0);
    int new_pal = vrs_note_pal(h);
    g_vrs_w_pal++;
    if (!vlog) {
        if (!V.vrs && !g_vrs_once[1]++)
            VLOG("nvnCommandBufferShadingRateImagePalette: shading-rate palettes are recorded but not applied (full-rate shading; DM_VK_VRS=1 enables, DM_VK_VRS_LOG=1 for details)");
    } else {
        if (g_vrs_logged < VRS_LOG_CAP || new_pal) {
            g_vrs_logged++;
            char eb[384]; int o = 0; eb[0] = 0;
            for (int i = 0; i < nstore && o < (int)sizeof eb - 16; i++)
                o += snprintf(eb + o, sizeof eb - (size_t)o, "%s%#x", i ? " " : "", entries[i]);
            VLOG("vrs ShadingRateImagePalette%s: vp=%d count=%d %s=%#llx nread=%d hash=%016llx [%s]",
                 new_pal && g_vrs_logged > VRS_LOG_CAP ? " [new palette]" : "",
                 vp, count, ptr_reg ? ptr_reg : "ptr", (unsigned long long)ptr, nstore, (unsigned long long)h, eb);
        }
        vrs_summary_tick();
    }
    pthread_mutex_unlock(&g_vrs_mu);
}

void dm_gfx_vrs_palette_mode(uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3, uint64_t x4, uint64_t x5) {
    (void)x2; (void)x3; (void)x4; (void)x5;
    int vlog = VKA_ENV("DM_VK_VRS_LOG", 0);
    pthread_mutex_lock(&g_vrs_mu);
    vrs_note_cb(x0);
    g_vrs_w_mode++;
    if (!vlog) {
        if (!V.vrs && !g_vrs_once[2]++)
            VLOG("nvnCommandBufferSetShadingRatePaletteMode: shading-rate palette mode is recorded but not applied (full-rate shading; DM_VK_VRS=1 enables, DM_VK_VRS_LOG=1 for details)");
    } else {
        if (g_vrs_logged < VRS_LOG_CAP) {
            g_vrs_logged++;
            VLOG("vrs SetShadingRatePaletteMode: mode=%d (x0=%#llx)", (int)x1, (unsigned long long)x0);
        }
        vrs_summary_tick();
    }
    pthread_mutex_unlock(&g_vrs_mu);
}
#endif
