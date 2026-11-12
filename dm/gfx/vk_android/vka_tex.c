// NVN textures as Vulkan images, and the texture / sampler pools as update-after-bind descriptor arrays.
// An image lives in its own device memory, not over the pool: Vulkan's optimal tiling is not the Tegra block-linear layout. A texture the CPU filled (packaged data, WriteTexels,
// a CPU write the dirty tracker saw: mtl_tex_cpu_written) is deswizzled from its pool storage (nvn_tex.c layout, tex/nvn_deswizzle.c) into scratch memory and copied in;
// every other texture starts cleared to zero and is GPU content from then on (render targets, copy destinations), as nvn_mtl.m's private textures are.
// Images are found by the texture's serial (the game copies texture objects by value). A texture over the same storage and shape as a live one shares its image, so a clear through
// one object and a present through another meet. Every image stays in VK_IMAGE_LAYOUT_GENERAL after its first use: no layout tracking until the draw path needs it
#if defined(__ANDROID__)
#include "vka.h"
#include "../nvn/nvn_mtl.h"
#include "../tex/nvn_deswizzle.h"
#include "../nvn/dmsh_abi_vk.h"
#include "../nvn/nvn_dirty.h"
#include "vka_rprof.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "vka_zshare.h"
#include "../../host/mem_foot.h"

// ---------- Formats ----------
typedef struct { VkFormat vk; uint8_t kind, srgb; } fmt_ent;
static fmt_ent fmt_ent_of(int f) {
#define C(v) (fmt_ent){VK_FORMAT_##v, 0, 0}
#define U(v) (fmt_ent){VK_FORMAT_##v, 1, 0}
#define S(v) (fmt_ent){VK_FORMAT_##v, 2, 0}
#define SRGB(v) (fmt_ent){VK_FORMAT_##v, 0, 1}
    if (f >= 0x79 && f <= 0x86) return (fmt_ent){(VkFormat)(VK_FORMAT_ASTC_4x4_UNORM_BLOCK + 2 * (f - 0x79)), 0, 0};   // the 14 ASTC block sizes are in the same order in both enums
    if (f >= 0x87 && f <= 0x94) return (fmt_ent){(VkFormat)(VK_FORMAT_ASTC_4x4_UNORM_BLOCK + 2 * (f - 0x87)), 0, 1};
    switch (f) {
    case 0x01: case 0x51: case 0x52: return f == 0x01 ? C(R8_UNORM) : f == 0x51 ? U(R8_UINT) : S(R8_SINT);
    case 0x02: return C(R8_SNORM); case 0x03: return U(R8_UINT); case 0x04: return S(R8_SINT);
    case 0x05: return C(R16_SFLOAT); case 0x06: return C(R16_UNORM); case 0x07: return C(R16_SNORM); case 0x08: case 0x53: return U(R16_UINT); case 0x09: case 0x54: return S(R16_SINT);
    case 0x0A: return C(R32_SFLOAT); case 0x0B: case 0x55: return U(R32_UINT); case 0x0C: case 0x56: return S(R32_SINT);
    case 0x0D: return C(R8G8_UNORM); case 0x0E: return C(R8G8_SNORM); case 0x0F: case 0x57: return U(R8G8_UINT); case 0x10: case 0x58: return S(R8G8_SINT);
    case 0x11: return C(R16G16_SFLOAT); case 0x12: return C(R16G16_UNORM); case 0x13: return C(R16G16_SNORM); case 0x14: case 0x59: return U(R16G16_UINT); case 0x15: case 0x5A: return S(R16G16_SINT);
    case 0x16: return C(R32G32_SFLOAT); case 0x17: case 0x5B: return U(R32G32_UINT); case 0x18: case 0x5C: return S(R32G32_SINT);
    case 0x22: return C(R32G32B32_SFLOAT); case 0x23: case 0x61: return U(R32G32B32_UINT); case 0x24: case 0x62: return S(R32G32B32_SINT);
    case 0x25: case 0x6D: return C(R8G8B8A8_UNORM); case 0x26: case 0x6E: return C(R8G8B8A8_SNORM); case 0x27: case 0x6F: case 0x63: return U(R8G8B8A8_UINT); case 0x28: case 0x70: case 0x64: return S(R8G8B8A8_SINT);
    case 0x29: case 0x71: return C(R16G16B16A16_SFLOAT); case 0x2A: case 0x72: return C(R16G16B16A16_UNORM); case 0x2B: case 0x73: return C(R16G16B16A16_SNORM);
    case 0x2C: case 0x74: case 0x65: return U(R16G16B16A16_UINT); case 0x2D: case 0x75: case 0x66: return S(R16G16B16A16_SINT);
    case 0x2E: case 0x76: return C(R32G32B32A32_SFLOAT); case 0x2F: case 0x77: case 0x67: return U(R32G32B32A32_UINT); case 0x30: case 0x78: case 0x68: return S(R32G32B32A32_SINT);
    case 0x31: return (fmt_ent){VK_FORMAT_S8_UINT, 4, 0};
    case 0x32: return (fmt_ent){VK_FORMAT_D16_UNORM, 3, 0};
    case 0x33: return (fmt_ent){VK_FORMAT_X8_D24_UNORM_PACK32, 3, 0};
    case 0x34: return (fmt_ent){VK_FORMAT_D32_SFLOAT, 3, 0};
    case 0x35: return (fmt_ent){VK_FORMAT_D24_UNORM_S8_UINT, 5, 0};
    case 0x36: return (fmt_ent){VK_FORMAT_D32_SFLOAT_S8_UINT, 5, 0};
    case 0x37: case 0x38: return SRGB(R8G8B8A8_UNORM);
    case 0x39: return C(A4B4G4R4_UNORM_PACK16);   // Maxwell's packed formats start with R in the low bits
    case 0x3A: case 0x3B: return C(A1B5G5R5_UNORM_PACK16);
    case 0x3C: return C(B5G6R5_UNORM_PACK16); case 0x95: return C(R5G6B5_UNORM_PACK16);
    case 0x3D: return C(A2B10G10R10_UNORM_PACK32); case 0x3E: case 0x6B: return U(A2B10G10R10_UINT_PACK32); case 0x69: return C(A2B10G10R10_SNORM_PACK32); case 0x6A: case 0x6C: return S(A2B10G10R10_SINT_PACK32);
    case 0x3F: return C(B10G11R11_UFLOAT_PACK32); case 0x40: return C(E5B9G9R9_UFLOAT_PACK32);
    case 0x41: return C(BC1_RGB_UNORM_BLOCK); case 0x42: return C(BC1_RGBA_UNORM_BLOCK); case 0x43: return C(BC2_UNORM_BLOCK); case 0x44: return C(BC3_UNORM_BLOCK);
    case 0x45: return SRGB(BC1_RGB_UNORM_BLOCK); case 0x46: return SRGB(BC1_RGBA_UNORM_BLOCK); case 0x47: return SRGB(BC2_UNORM_BLOCK); case 0x48: return SRGB(BC3_UNORM_BLOCK);
    case 0x49: return C(BC4_UNORM_BLOCK); case 0x4A: return C(BC4_SNORM_BLOCK); case 0x4B: return C(BC5_UNORM_BLOCK); case 0x4C: return C(BC5_SNORM_BLOCK);
    case 0x4D: return C(BC7_UNORM_BLOCK); case 0x4E: return SRGB(BC7_UNORM_BLOCK); case 0x4F: return C(BC6H_SFLOAT_BLOCK); case 0x50: return C(BC6H_UFLOAT_BLOCK);
    case 0x99: case 0x9A: return C(B8G8R8A8_UNORM); case 0x9B: case 0x9C: return SRGB(B8G8R8A8_UNORM);
    }
#undef C
#undef U
#undef S
#undef SRGB
    return (fmt_ent){VK_FORMAT_UNDEFINED, 0, 0};
}
static VkFormat srgb_twin(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8G8B8A8_UNORM: return VK_FORMAT_R8G8B8A8_SRGB; case VK_FORMAT_B8G8R8A8_UNORM: return VK_FORMAT_B8G8R8A8_SRGB;
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK: return VK_FORMAT_BC1_RGB_SRGB_BLOCK; case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
    case VK_FORMAT_BC2_UNORM_BLOCK: return VK_FORMAT_BC2_SRGB_BLOCK; case VK_FORMAT_BC3_UNORM_BLOCK: return VK_FORMAT_BC3_SRGB_BLOCK; case VK_FORMAT_BC7_UNORM_BLOCK: return VK_FORMAT_BC7_SRGB_BLOCK;
    default: return f >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK ? (VkFormat)(f + 1) : f;
    }
}
static VkFormatFeatureFlags feats(VkFormat f) {
    static struct { VkFormat f; VkFormatFeatureFlags ff; } cache[128]; static int n; static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu);
    for (int i = 0; i < n; i++) if (cache[i].f == f) { VkFormatFeatureFlags ff = cache[i].ff; pthread_mutex_unlock(&mu); return ff; }
    VkFormatProperties p; vk.vkGetPhysicalDeviceFormatProperties(V.pd, f, &p);
    if (n < 128) cache[n++].f = f, cache[n - 1].ff = p.optimalTilingFeatures;
    pthread_mutex_unlock(&mu);
    return p.optimalTilingFeatures;
}
static int fmt_of_slow(int nf, vka_fmt *o);
// By NVN format, answered once (the draws ask for every target and texture view: ~3% of the replay thread in format lookups and the feature cache's lock)
int vka_fmt_of(int nf, vka_fmt *o) {
    static _Atomic uint8_t st[512]; static vka_fmt memo[512];   // st: 0 unknown, 1 has a format (memo), 2 none
    if (nf >= 0 && nf < 512) {
        int k = atomic_load_explicit(&st[nf], memory_order_acquire);
        if (k == 1) { *o = memo[nf]; return 1; }
        if (k == 2) return 0;
        vka_fmt f; int ok = fmt_of_slow(nf, &f);
        if (ok) { memo[nf] = f; *o = f; }
        atomic_store_explicit(&st[nf], ok ? 1 : 2, memory_order_release);
        return ok;
    }
    return fmt_of_slow(nf, o);
}
static int fmt_of_slow(int nf, vka_fmt *o) {
    fmt_ent e = fmt_ent_of(nf); int bw, bh, bpb = nvn_fmt_bpb(nf, &bw, &bh);
    if (e.vk == VK_FORMAT_D24_UNORM_S8_UINT && !(feats(e.vk) & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) e.vk = VK_FORMAT_D32_SFLOAT_S8_UINT;
    if (e.vk == VK_FORMAT_X8_D24_UNORM_PACK32 && !(feats(e.vk) & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) e.vk = VK_FORMAT_D32_SFLOAT;
    if (e.vk == VK_FORMAT_UNDEFINED || !bpb || !(feats(e.vk) & (VK_FORMAT_FEATURE_TRANSFER_DST_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))) {
        static uint64_t logged[4]; unsigned b = (unsigned)nf & 255;
        if (!(__atomic_fetch_or(&logged[b >> 6], 1ull << (b & 63), __ATOMIC_RELAXED) >> (b & 63) & 1)) VRED("NVN format %#x has no Vulkan format on this device (VkFormat %d)", nf, e.vk);
        return 0;
    }
    *o = (vka_fmt){e.vk, e.srgb ? srgb_twin(e.vk) : e.vk, e.kind, e.srgb, (uint8_t)bpb, (uint8_t)bw, (uint8_t)bh};
    return 1;
}
static VkImageAspectFlags aspect_of(int kind) { return kind == 3 ? VK_IMAGE_ASPECT_DEPTH_BIT : kind == 4 ? VK_IMAGE_ASPECT_STENCIL_BIT : kind == 5 ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT; }
static int compressed(const vka_fmt *f) { return f->bw > 1 || f->bh > 1; }

uint64_t vka_n_bar;   // full barriers recorded (vka_exec.c replay statistics)
void vka_barrier_at(VkCommandBuffer cb, const char *file, int line) {
    vka_n_bar++;
    VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT, .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT};
    vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    if (vka_gp_active) vka_gp_mark(cb, file, line);
}
void vka_img_barrier(VkCommandBuffer cb, VkImage img, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT, .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        .oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
        .subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
    vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &b);
}

// ---------- Images by serial ----------
typedef struct view_ent { struct view_ent *next; VkImageView v; VkFormat f; VkImageViewType t; VkImageAspectFlags asp; uint32_t l0, nl, a0, na; VkComponentMapping swz; } view_ent;
typedef struct { uint64_t serial; vka_img *im; } smap_ent;   // serial 0 = empty, im NULL with a serial = a texture that has no image (format refused)
static smap_ent *g_map; static size_t g_mapcap, g_mapn;
static vka_img **g_live; static int g_nlive, g_livecap;
static uint64_t *g_cpuw; static size_t g_cpuwcap, g_cpuwn;   // serials the CPU wrote before their image existed (a set, open addressing)
static pthread_mutex_t g_tmu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_img_bytes;
// Images' last use (used_s, made_s): seconds of submitting, not of the wall clock. vka_tex_prepare adds at most 1 per submission whose coarse monotonic second
// differs from the last one's, so a time without submissions (the app in the background, a long stall) does not age every image at once: on the wall clock the first
// submission after a minute away found every image idle and let all that could go go, to be uploaded again in the next frames
static _Atomic uint32_t g_now_s = 1;
uint64_t vka_frame(void);
static inline void img_used(vka_img *im) { im->used_s = g_now_s; im->used_f = (uint32_t)vka_frame(); }   // a CPU lookup: used_s, and used_f ("Texture-use feedback": the release window's frames)
// Image eviction's counts (tex_evict, under g_tmu): images let go, their bytes, and serials whose image was made again after it (g_evs: the serials let go, direct-mapped,
// so a collision forgets one and the count is a floor)
static uint64_t g_ev_n, g_ev_bytes, g_ev_remade, g_evs[4096];
// Of those, images a newer texture over their storage made dead (tex_evict, EV_DEAD): how many and their bytes, the serials (g_evd, direct-mapped like g_evs), how many
// were made again after it, and the serials pinned for that (g_evpin: the overlap rule leaves their images alone from then on)
static uint64_t g_evd_n, g_evd_bytes, g_evd_remade, g_evd[4096], g_evpin[4096];
static uint64_t g_evu_n, g_evu_bytes;   // of those, let go as named and unseen (DM_VK_EVICT=2)
static uint64_t g_nreg;   // texture registrations so far (treg; the eviction line's registrations a second)
static int g_zwant;   // (under g_tmu) img_create_ may make a zero-shared or guest-aliased image: vka_tex_get's new image whose content is ST_ZERO ("Zero-shared images", "Guest-aliased targets" below)
// ---------- Guest-aliased targets (DM_VK_RTALIAS) ----------
// On the Switch a texture is its storage, and an engine's transient heap gives one storage to many textures over time. Ocarina of Time draws a square DEPTH32F target at one
// pool address (0x412f25d600 in session-2026-11-07 rt/FB6.md) whose size follows the camera: a new texture object of a new size almost every frame while Link walks (1112² to
// 1482²), each written and sampled for a frame or two and never used again, all at offset 0 of the same 8.6-14 MiB of guest storage. Here each new size was a new image with
// memory of its own (zero-shared until its first write, then adopted): 51 own (345 MiB) and 19 zero-shared (129 MiB) in about 110 frames, and the low-memory killer 20 s
// into the walk; eviction let the dead sizes go only after DM_VK_EVICT_S, and the table scan made the ones whose IDs a storage buffer still named again.
// DM_VK_RTALIAS=1 (live: images made from then on; 0, the default, as before; Ocarina of Time 1 in its profile): a new image whose content is the GPU's (ST_ZERO) and that a
// zero-shared one could be (dense, single-sampled, not 3D, not block-compressed, not linear, not a display texture, at least DM_VK_RTALIAS_MB of storage, default 4) over a
// live pool is bound at offset 0 of one allocation per key (vka_imgmem.c vka_imem_bind_alias, vka_rtalias.h): the pool, its Initialize serial and the storage address, and the
// create info without its extent (flags, format, type, levels, layers, samples, usage, format list) and memory type. Textures that differ only in size at one storage address
// are the same bytes on the Switch: writing one leaves the other's bytes read in the wrong shape there, so only the newest one written holds content, as here. An image
// that does not fit the key's current allocation makes a new one of max(its size, 1.5 times the current one's) (a target that grows a little every frame: a few allocations,
// not one per size); older allocations go with their last image (eviction, Finalize; vka_after, after the GPU is done). Vulkan aliasing: a new image on an allocation others
// hold is not cleared at its first use (only its transition from UNDEFINED, as an alias's first use must be: its content is undefined, as the Switch's bytes would be read in its
// shape), and its first image on a new allocation is cleared as before; it is never zero-shared, so it gets no memory of its own at its first write (img_own). Nothing else
// changes: references, holds, the open pass's attachments, eviction, and frees after the submissions that may use it. Counted on the `memory:` line and the ctl stats
// (`guest-aliased: N images in K allocations (X MiB, Y MiB saved), M made, G of them grown`)
static uint64_t g_ra_n, g_ra_b, g_ra_noclr;   // (under g_tmu) guest-aliased images made, the bytes they would have had alone, first uses left without a clear
static uint64_t ra_min(void) { if (!VKA_ENV("DM_VK_RTALIAS", 0)) return 0; double mb = VKA_ENVF("DM_VK_RTALIAS_MB", 4.0); return mb > 0 ? (uint64_t)(mb * 1048576.0) : 0; }
static uint64_t g_img_made;   // (under g_tmu) image memory allocated in all (img_create: an image's own, a new zero-shared allocation): the pressure's creation-burst test (mem_low)

// The images' memory for the stats line: all of it, and the part whose content the GPU wrote (render targets, copy and storage destinations); the rest was filled from the
// texture's storage in guest memory, which stays in its memory pool, so those bytes are held twice
uint64_t vka_tex_mem_nolock(int *n) { if (n) *n = g_nlive; return g_img_bytes + vka_imem_zbytes_nolock() + vka_imem_abytes_nolock(); }   // (zero-shared and guest-aliased allocations counted once each)
uint64_t vka_tex_mem(int *n, uint64_t *gpu_written) {
    pthread_mutex_lock(&g_tmu); uint64_t all = g_img_bytes + vka_imem_zbytes() + vka_imem_abytes(), gw = 0;
    for (int i = 0; i < g_nlive; i++) if (g_live[i]->gpu_owned) gw += g_live[i]->bytes;
    if (n) *n = g_nlive; if (gpu_written) *gpu_written = gw;
    pthread_mutex_unlock(&g_tmu); return all;
}
// The images' bytes by where their content came from, and the part whose memory requirements would let it live on its texture's storage in the pool (the memory type the
// pools use, a size within the storage, the storage's offset aligned as the image needs): what binding images onto pool memory could give back. Only measured: an image over
// its storage would overwrite the guest's block-linear bytes with the driver's layout, so a texture the CPU rewrites in part (streamed mip levels) would re-upload garbage;
// only content no CPU write follows is a candidate (docs/reference/rendering.md "Images filled from pool storage hold texture bytes twice on the Vulkan replay")
void vka_tex_alias_stats(char *b, size_t n) {
    uint64_t by[3][2] = {{0}}, gw = 0, st = 0, dead = 0, idle[2] = {0}; int cnt[3] = {0}, ndead = 0, nidle[2] = {0};
    pthread_mutex_lock(&g_tmu);
    uint32_t now = g_now_s;
    for (int i = 0; i < g_nlive; i++) { const vka_img *im = g_live[i];
        if (im->pdead) { dead += im->bytes; ndead++; }
        for (int k = 0; k < 2; k++) if (now - im->used_s >= (k ? 60u : 10u)) { idle[k] += im->bytes; nidle[k]++; }   // (the flag as last checked: img_pool_dead is not called here)
        if (im->gpu_owned) { gw += im->bytes; continue; }
        int m = im->mode < 3 ? im->mode : 0; by[m][0] += im->bytes; cnt[m]++; if (im->pool_fit) by[m][1] += im->bytes; st += im->L.size; }
    uint64_t evb = g_ev_bytes, evn = g_ev_n, evr = g_ev_remade;   // (image eviction, tex_evict)
    pthread_mutex_unlock(&g_tmu);
    snprintf(b, n, "images not GPU-written: cleared %.0f MB (%d, %.0f MB would fit their storage), filled once %.0f MB (%d, %.0f MB fit), tracked %.0f MB (%d, %.0f MB fit), their storage %.0f MB; GPU-written %.0f MB; "
             "of a finalized pool %.0f MB (%d); not used for 10 s %.0f MB (%d), for 60 s %.0f MB (%d); evicted %.0f MB (%llu, %llu made again)",
             by[0][0] / 1e6, cnt[0], by[0][1] / 1e6, by[1][0] / 1e6, cnt[1], by[1][1] / 1e6, by[2][0] / 1e6, cnt[2], by[2][1] / 1e6, st / 1e6, gw / 1e6, dead / 1e6, ndead, idle[0] / 1e6, nidle[0], idle[1] / 1e6, nidle[1],
             evb / 1e6, (unsigned long long)evn, (unsigned long long)evr);
}
static size_t hslot(uint64_t s, size_t cap) { return (size_t)((s * 0x9E3779B97F4A7C15ull) >> 20) & (cap - 1); }
static smap_ent *map_get(uint64_t s) { if (!g_mapcap) return NULL; for (size_t i = hslot(s, g_mapcap);; i = (i + 1) & (g_mapcap - 1)) { if (g_map[i].serial == s) return &g_map[i]; if (!g_map[i].serial) return NULL; } }
static _Atomic uint32_t g_tgen, *g_tver, *g_sver;   // the recording thread's view memo: see unref_locked
static void map_put(uint64_t s, vka_img *im);
static void map_grow(void) {
    size_t oc = g_mapcap; smap_ent *om = g_map;
    g_mapcap = oc ? oc * 2 : 1024; g_map = calloc(g_mapcap, sizeof *g_map); g_mapn = 0;
    for (size_t i = 0; i < oc; i++) if (om[i].serial && om[i].serial != ~0ull) map_put(om[i].serial, om[i].im);
    free(om);
}
static void map_put(uint64_t s, vka_img *im) {
    if ((g_mapn + 1) * 2 > g_mapcap) map_grow();
    size_t i = hslot(s, g_mapcap);
    while (g_map[i].serial && g_map[i].serial != ~0ull && g_map[i].serial != s) i = (i + 1) & (g_mapcap - 1);
    if (g_map[i].serial != s) g_mapn++;
    g_map[i] = (smap_ent){s, im};
}
static void map_del(uint64_t s) { smap_ent *e = map_get(s); if (e) { e->serial = ~0ull; e->im = NULL; atomic_fetch_add_explicit(&g_tgen, 1, memory_order_release); } }   // a tombstone (never a valid serial)
// (the ctl texshot, on the present thread, records a copy of it without a reference: marked used, so image eviction leaves it for DM_VK_EVICT_S seconds)
vka_img *vka_img_of_serial(uint64_t serial) { pthread_mutex_lock(&g_tmu); smap_ent *e = map_get(serial); vka_img *im = e ? e->im : NULL; if (im) img_used(im); pthread_mutex_unlock(&g_tmu); return im; }
static int cpuw_take(uint64_t s, int add) {   // add: insert; else: remove and say whether it was there
    if (add && (g_cpuwn + 1) * 2 > g_cpuwcap) {
        size_t oc = g_cpuwcap; uint64_t *o = g_cpuw; g_cpuwcap = oc ? oc * 2 : 256; g_cpuw = calloc(g_cpuwcap, 8); g_cpuwn = 0;
        for (size_t i = 0; i < oc; i++) if (o[i] && o[i] != ~0ull) cpuw_take(o[i], 1);
        free(o);
    }
    if (!g_cpuwcap) return 0;
    for (size_t i = hslot(s, g_cpuwcap);; i = (i + 1) & (g_cpuwcap - 1)) {
        if (g_cpuw[i] == s) { if (!add) g_cpuw[i] = ~0ull; return 1; }
        if (!g_cpuw[i]) { if (add) { g_cpuw[i] = s; g_cpuwn++; } return 0; }
    }
}

static void img_destroy(void *arg) {
    vka_img *im = arg;
    if (im->zold) { img_destroy(im->zold); im->zold = NULL; }   // (the zero-shared image it left: no holder is left either)
    for (view_ent *v = im->views, *n; v; v = n) { n = v->next; vk.vkDestroyImageView(V.dev, v->v, NULL); free(v); }
    if (im->rs) { vks_free(im->rs); vks_va_give(im->va); }   // (its binds dropped or run first)
    if (im->img) vk.vkDestroyImage(V.dev, im->img, NULL);
    vka_imem_free(&im->m);   // after the image: its run or allocation (vka_after: the GPU is done with it)
    free(im->snap); free(im);
}
#define VIEWS(im) (*(view_ent **)&(im)->views)

// The recording thread's memo of texture ID -> view (vka_tex_id_view_rec) and sampler ID -> sampler (vka_samp_id_rec): an entry holds while the ID was not registered
// again (g_tver / g_sver, per ID) and no texture mapping was dropped or image let go since (g_tgen). It saves the table lock and the lookups 3-4 times a texture binding
static vka_img *g_big[64]; static int g_nbig;   // large images (DM_VK_BIGIMG_MB), for the storage overlap check (big_note)
static void rtw_forget(vka_img *im); static void rtw_log(const vka_img *im, const char *when);   // ("Big targets")
static void rl_note(const vka_img *im, const char *what);   // ("Render target log")
static void unref_locked(vka_img *im) {
    atomic_fetch_add_explicit(&g_tgen, 1, memory_order_release);
    if (--im->refs) return;
    for (int i = 0; i < g_nlive; i++) if (g_live[i] == im) { g_live[i] = g_live[--g_nlive]; break; }
    for (int i = 0; i < g_nbig; i++) if (g_big[i] == im) { g_big[i] = g_big[--g_nbig]; break; }
    rtw_forget(im); rl_note(im, "gone");
    g_img_bytes -= im->bytes;
    vka_after(vka_fence_last(), img_destroy, im);
}

// Textures of one memory and shape share an image (one storage, many texture objects: a clear through one and a present through another meet), and so do ones of another
// uncompressed colour format of the same texel size (float, normalized or integer: an R32_UINT texture over a B10G11R11 target's memory is one) whose formats (storage and
// view) the image's format list takes: they see the image through views in their own format.
// A texture whose format the list lacks gets its own image of its format, and the pair is learned (Learned view formats, below), so the images made from then on, and in
// the next runs from their first use, take it and are shared. Shared without the format, its views fell back to the image's format (vka_view): DKB's R11G11B10 HDR target
// over an RGBA8 texture's memory was drawn and sampled as RGBA8 UNORM, and a colour attachment view in another format than the pipeline's rendering format breaks a valid
// usage rule. The split loses a bit-for-bit reinterpretation between the two formats through the memory in that run, which the format list never allowed anyway; it stays
// for the run without DM_TEMP's file, once the table is full, or under DM_VK_VIEW_LEARN=0. Not compressed textures of another format: a view takes only a format of the image's own compatibility class (BC1 and BC4 blocks have one size, as BC3, BC5, BC7 and ASTC 4x4 blocks have, but no view
// reinterprets one as the other), and view learning takes uncompressed pairs only, so a BC4 texture over a BC1 image's memory was sampled through the BC1 view: garbage.
// Such a texture gets its own image, filled from its own storage; what it loses is GPU content passing between the two through the memory, which compressed textures
// (never render targets) do not have
static int img_lists(const vka_img *a, VkFormat v) {   // the image's format list takes views in v (img_create)
    if (v == a->f.vk || v == a->alt || a->any_fmt) return 1;
    for (int k = 0; k < a->nx; k++) if (a->xfmt[k] == v) return 1;
    return 0;
}
static int vl_learn(VkFormat a, VkFormat b, uint32_t w, uint32_t h);
static pthread_mutex_t g_vlmu;
static int same_shape(const vka_img *a, const tex_t *t, const vka_fmt *f, int learn) {   // learn = 0: a diagnostic's question (vka_cap_shared), nothing learned or logged
    if (a->mem != t->mem || a->d.target != t->d.target || a->d.width != t->d.width || a->d.height != t->d.height || a->d.depth != t->d.depth || a->d.levels != t->d.levels || a->d.samples != t->d.samples) return 0;
    if (a->f.vk == f->vk) return 1;
    if (!(a->f.kind < 3 && f->kind < 3 && a->f.bpb == f->bpb && !compressed(&a->f) && !compressed(f))) return 0;   // (kind < 3: colour, float or integer)
    int lv = img_lists(a, f->vk), lw = img_lists(a, f->view);
    if (lv && lw) return 1;
    if (!learn) return 0;
    pthread_mutex_lock(&g_vlmu);   // (under g_tmu: the order vka_view takes them in)
    int learned = (!lv && vl_learn(a->f.vk, f->vk, a->w, a->h)) | (!lw && f->view != f->vk && vl_learn(a->f.vk, f->view, a->w, a->h));
    pthread_mutex_unlock(&g_vlmu);
    static struct { VkFormat a, b; } seen[16]; static int ns; int k = 0;   // (under g_tmu)
    while (k < ns && !(seen[k].a == a->f.vk && seen[k].b == f->vk)) k++;
    if (k == ns && ns < 16) { seen[ns].a = a->f.vk; seen[ns++].b = f->vk;
        VLOG("textures of VkFormat %d and %d over the same memory (%ux%u): separate images, the first's format list cannot take the second%s", a->f.vk, f->vk, a->w, a->h,
             learned ? " (learned: images of the first format and this size made from now on, and in the next runs, take it and are shared)" : ""); }
    return 0;
}

// ---------- Learned view formats ----------
// Textures of one memory and shape share one image (same_shape) when the image's format list takes their formats: content passes between them through it, whatever the
// operation (draws, copies, clears, presents, write-back). Images keep Adreno's compression with a short format list (own format, sRGB twin) instead of a bare
// MUTABLE_FORMAT, and an image cannot take a new format once made, so the formats are learned: a texture of another format over the memory (same_shape), or a view asked
// in a format outside the list (vka_view: a texture view's own format), records the pair (image format, other format, size), and every image made from then on with that
// format and size takes the other format in its list (MUTABLE_FORMAT). Turnip keeps UBWC on such an image when every listed format can have it and the GPU's UBWC is
// format-independent (tu_image.cc: ubwc_possible on each listed format; a7xx gen 3 and the Adreno 830 have ubwc_all_formats_compatible), so B10G11R11 + R32_UINT keeps
// it while E5B9G9R9 in the list drops it; older Adreno keep it only for lists of one UBWC compatibility class. The pairs are kept in $DM_TEMP/vk_view_formats.txt, so the next run makes
// the image right from its first use. Meanwhile the texture gets its own image (same_shape), and a view keeps the fallback to the image's own format ([RED]) for the life
// of an image made before the pair was learned (its views are in use; replacing it would mean rebinding every view and descriptor); so it stays in a run without DM_TEMP,
// once the table is full (64 pairs) or with 4 formats learned for that image format and size (VL_PER_IMAGE).
// A pair is taken (learned, or read from the file) only when both formats are uncompressed colour formats of one texel size, float, normalized or integer (kinds 0-2;
// Vulkan's compatibility classes of uncompressed colour formats are by texel size, so R32_UINT is in B10G11R11's and RGBA8's 32-bit class): a format list may hold only
// formats compatible with the image's, and a bad pair in the file (an older build's, a hand edit) would break that rule on every image of its format in every run. Integer
// pairs were refused before, so an R32_UINT view of a B10G11R11 or RGBA8 target (a pass packing RGB9E5 or colour into words) fell back to the target's own format for
// good: what the pass wrote was not its words (Ocarina of Time's gameplay picture black, the upscaler's input near zero). The file is
// written whole (a temporary file renamed over it), so a run that ends while writing leaves the old one. DM_VK_VIEW_LEARN=0: nothing learned or read (views in the image's
// own format, as before)
typedef struct { VkFormat a, b; uint32_t w, h; } vlearn;
enum { NVL = 64 };
static vlearn g_vl[NVL]; static int g_nvl, g_vl_loaded; static char g_vl_path[1024]; static pthread_mutex_t g_vlmu = PTHREAD_MUTEX_INITIALIZER;
enum { VL_PER_IMAGE = 4 };   // learned formats one image takes at most (vka_img.xfmt)   // under g_vlmu
static int vl_bytes(VkFormat v) {   // bytes per texel of an uncompressed colour (kind 0-2: float, normalized or integer) format of the textures (an image's own format or its sRGB view format), else 0
    for (int nf = 1; nf < 0x100; nf++) {
        fmt_ent e = fmt_ent_of(nf); int bw = 0, bh = 0, bpb;
        if (e.vk == VK_FORMAT_UNDEFINED || !(e.vk == v || (e.srgb && srgb_twin(e.vk) == v))) continue;
        return e.kind < 3 && (bpb = nvn_fmt_bpb(nf, &bw, &bh)) && bw == 1 && bh == 1 ? bpb : 0;
    }
    return 0;
}
static int vl_pair_ok(VkFormat a, VkFormat b) { int n = vl_bytes(a); return n && a != b && n == vl_bytes(b); }
static void vl_save(void) {   // caller holds g_vlmu
    if (!g_vl_path[0]) return;
    char tmp[1100]; snprintf(tmp, sizeof tmp, "%s.%d.tmp", g_vl_path, (int)getpid());
    FILE *f = fopen(tmp, "we"); if (!f) { VRED("learned view formats: cannot write %s (errno %d): the pairs learned now are not kept", tmp, errno); return; }
    for (int i = 0; i < g_nvl; i++) fprintf(f, "%u %u %u %u\n", (unsigned)g_vl[i].a, (unsigned)g_vl[i].b, g_vl[i].w, g_vl[i].h);
    int bad = fflush(f) != 0 || ferror(f); bad |= fclose(f) != 0;
    if (bad || rename(tmp, g_vl_path)) { VRED("learned view formats: cannot write %s (errno %d): the pairs learned now are not kept", g_vl_path, errno); unlink(tmp); }
}
static void vl_load(void) {   // caller holds g_vlmu
    if (g_vl_loaded) return; g_vl_loaded = 1;
    const char *t = getenv("DM_TEMP"); if (!t || !*t) return;
    snprintf(g_vl_path, sizeof g_vl_path, "%s/vk_view_formats.txt", t);
    FILE *f = fopen(g_vl_path, "re"); if (!f) return;
    char ln[128]; int dropped = 0;
    while (g_nvl < NVL && fgets(ln, sizeof ln, f)) {
        unsigned a, b, w, h; char more;
        if (sscanf(ln, "%u %u %u %u %c", &a, &b, &w, &h, &more) != 4 || !w || !h || !vl_pair_ok((VkFormat)a, (VkFormat)b)) { dropped++; continue; }
        int dup = 0, same = 0;
        for (int i = 0; i < g_nvl; i++) { dup |= g_vl[i].a == (VkFormat)a && g_vl[i].b == (VkFormat)b && g_vl[i].w == w && g_vl[i].h == h; same += g_vl[i].a == (VkFormat)a && g_vl[i].w == w && g_vl[i].h == h; }
        if (dup || same >= VL_PER_IMAGE) { dropped++; continue; }
        g_vl[g_nvl++] = (vlearn){(VkFormat)a, (VkFormat)b, w, h};
    }
    fclose(f);
    if (g_nvl || dropped) VLOG("learned view formats %s: %d pairs (images of those formats and sizes take the second format in their format list)%s", g_vl_path, g_nvl,
                               dropped ? "; dropped lines that are not a pair of uncompressed colour formats of one texel size, duplicates and those past 4 for one format and size (the file is written again without them)" : "");
    if (dropped) vl_save();
}
static int vl_formats(VkFormat a, uint32_t w, uint32_t h, VkFormat *out, int max) {   // caller holds g_vlmu
    if (!VKA_ENV("DM_VK_VIEW_LEARN", 1)) return 0;
    vl_load(); int n = 0;
    for (int i = 0; i < g_nvl && n < max; i++) if (g_vl[i].a == a && g_vl[i].w == w && g_vl[i].h == h) { int dup = 0; for (int k = 0; k < n; k++) dup |= out[k] == g_vl[i].b; if (!dup) out[n++] = g_vl[i].b; }   // (vl_learn and vl_load keep at most VL_PER_IMAGE a format and size)
    return n;
}
static int vl_learn(VkFormat a, VkFormat b, uint32_t w, uint32_t h) {   // caller holds g_vlmu; 1 = new
    if (!VKA_ENV("DM_VK_VIEW_LEARN", 1) || !vl_pair_ok(a, b)) return 0;
    vl_load();
    int same = 0;
    for (int i = 0; i < g_nvl; i++) { if (g_vl[i].a == a && g_vl[i].b == b && g_vl[i].w == w && g_vl[i].h == h) return 0; same += g_vl[i].a == a && g_vl[i].w == w && g_vl[i].h == h; }
    if (g_nvl == NVL || same >= VL_PER_IMAGE) {
        static int nr; if (nr++ < 4) VRED("learned view formats: VkFormat %d as a view of VkFormat %d (%ux%u) not learned: %s", b, a, w, h, same >= VL_PER_IMAGE ? "that image format and size has 4 already" : "the table is full (64 pairs)");
        return 0;
    }
    g_vl[g_nvl++] = (vlearn){a, b, w, h};
    vl_save();
    return 1;
}

static VkImageViewType view_type(int target, uint32_t layers) {
    switch (target) {
    case NVN_TEXTURE_TARGET_1D: return VK_IMAGE_VIEW_TYPE_2D; case NVN_TEXTURE_TARGET_1D_ARRAY: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;   // one-row 2D images (img_create)
    case NVN_TEXTURE_TARGET_3D: return VK_IMAGE_VIEW_TYPE_3D;
    case NVN_TEXTURE_TARGET_2D_ARRAY: case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case NVN_TEXTURE_TARGET_CUBEMAP: return layers >= 6 ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: return layers >= 12 ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    default: return VK_IMAGE_VIEW_TYPE_2D;
    }
}

uint64_t nvn_pool_live(const void *pool, unsigned *flags, int *sparse); void nvn_pool_use(int on);   // hle_nvn.c: the replay reads a texture's pool only through it (the game may have finalized it)
uint64_t nvn_pool_fin(void);
static int pool_sparse(const void *pool) { int sp = 0; return pool && nvn_pool_live(pool, NULL, &sp) && sp; }
static unsigned pool_flags(const void *pool) { unsigned f = 0; nvn_pool_live(pool, &f, NULL); return f; }
uint64_t nvn_sparse_gen(void); uint64_t nvn_sparse_copy(const void *pool, void *dst, uint64_t a, uint64_t n);   // hle_nvn.c: sparse virtual pools
int nvn_sparse_since(const void *pool, uint64_t a, uint64_t n, uint64_t gen, uint64_t stamp); void nvn_sparse_protect(const void *pool, uint64_t a, uint64_t n);
int nvn_sparse_any(const void *pool, uint64_t a, uint64_t n); int nvn_sparse_mapped(const void *pool, uint64_t a, uint64_t n);
static void upload(vka_img *im, const tex_t *t, VkCommandBuffer cb);
static uint64_t g_rows_up;   // bytes of partial row re-uploads (upload_rows_tracked)

// ---------- Sparse textures: images of their mapped levels ----------
// A sampled texture of a sparse virtual pool holds what the physical pages mapped behind it hold (hle_nvn.c), and games map only part of it: Mario Kart World's 8192x8192 x 68-layer
// ASTC 8x8 arrays (1.45 GB each, 12 levels) have 25 to 130 MB mapped at the title, its levels from 2048 or 1024 texels down; 45 such textures were 8.8 GB uploaded in one frame
// ("Sparse texture arrays made at their full size" in docs/reference/nvn-objects.md). Vulkan's answer, sparse residency, does not fit them on the Adreno 840: its 3 GiB of sparse
// address space is under the 6.8 GB these textures span. So the image holds only the levels where pages are mapped: image level 0 is the texture's finest level with a mapped
// page in any layer (base), the finer ones are left out. A view or a target of a left-out level starts at the image's level 0 (the finest data there is). Sampling with an
// implicit level of detail picks the same texels as with the whole chain (normalized coordinates on an image 2^base times smaller: the level of detail is base lower, which is
// the level the image numbers from 0), except that the levels finer than base read base; an explicit level of detail, a texel fetch with a level or a size query sees the image's
// numbering (open, nvn-objects.md). Each level and layer is uploaded from the page table only when its pages or mappings change, unmapped pages as zero texels; when a finer level
// is mapped the image is made again with it (rebase), and when the finest mapped level becomes two or more levels coarser it is made again without the finer ones, which frees them.
// DM_VK_SPARSE_LEVELS=0 makes every sparse texture at its full size (the earlier behaviour)
static int sparse_sampled(const tex_t *t, const vka_fmt *f) {   // a sparse pool's texture whose content is its storage (storage_mode ST_SPARSE)
    const nvn_texdesc *d = &t->d;
    return t->mem && t->pool && pool_sparse(t->pool) && d->samples <= 1 && f->kind < 3 && d->target != NVN_TEXTURE_TARGET_BUFFER
        && !(d->flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_IMAGE));
}
// A sparse pool's SPARSE texture whose content is the GPU's (a render target or shader image: COMPRESSIBLE or IMAGE, which sparse_sampled leaves out): on the Switch only
// its mapped 64 KB pages have memory. Mario Kart World 1.8.0 makes ten of them after its first frames (16384x16384 R8, 8192x8192 RG16 / R16 terrain render targets, kartnative's
// note: 8-26 MB mapped each), 3.4 GB as dense images. A residency image from level 0 (the game renders into it), its tiles bound behind the mapped pages and nothing uploaded (sparse_bind)
static int sparse_gpu(const tex_t *t, const vka_fmt *f) {
    const nvn_texdesc *d = &t->d;
    return t->mem && t->pool && (d->flags & NVN_TEXTURE_FLAGS_SPARSE) && (d->flags & (NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_IMAGE)) && pool_sparse(t->pool) && d->samples <= 1
        && f->kind < 3 && (d->target == NVN_TEXTURE_TARGET_2D || d->target == NVN_TEXTURE_TARGET_2D_ARRAY) && !(d->flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET));
}
static int sparse_levels_on(void) { static int on = -1; if (on < 0) { const char *e = getenv("DM_VK_SPARSE_LEVELS"); on = !e || atoi(e) != 0; } return on; }
static void sub_range(const nvn_texdesc *d, const nvn_texlayout *L, uint32_t lv, uint32_t ly, uint64_t *off, uint64_t *len) {   // level lv of layer ly in the storage
    uint32_t n = d->levels ? d->levels : 1;
    uint64_t o = nvn_tex_level_offset(d, lv), e = lv + 1 < n ? nvn_tex_level_offset(d, lv + 1) : L->layer_size;
    *off = (uint64_t)ly * L->layer_size + o; *len = e > o ? e - o : 0;
}
// The finest level with a mapped page in any layer (the image's base); the coarsest level when nothing is mapped. 3D textures keep every level (their depth halves with the level too).
// *vmin: the finest level from which every level of every layer is wholly mapped (the coarsest when none is), where views start (nvn_mtl.m sparse_view: the translator drops
// TEX's level-of-detail clamp and residency predicate, so a game that clamps its samples to the mapped levels would read the unmapped tiles of a partly mapped level as zeros)
static uint32_t sparse_base(const void *pool, const uint8_t *mem, const nvn_texdesc *d, const nvn_texlayout *L, uint32_t *vmin) {
    uint32_t n = d->levels ? d->levels : 1, layers = L->layers ? L->layers : 1, b = n - 1, r = n;
    if (vmin) *vmin = 0;
    if (d->target == NVN_TEXTURE_TARGET_3D || !sparse_levels_on()) return 0;
    for (uint32_t lv = 0; lv + 1 < n && b == n - 1; lv++)
        for (uint32_t ly = 0; ly < layers; ly++) { uint64_t o, l; sub_range(d, L, lv, ly, &o, &l); if (l && nvn_sparse_any(pool, (uint64_t)(uintptr_t)mem + o, l)) { b = lv; break; } }
    for (int full = 1; r > b && full; ) {
        for (uint32_t ly = 0; ly < layers && full; ly++) { uint64_t o, l; sub_range(d, L, r - 1, ly, &o, &l); full = !l || nvn_sparse_mapped(pool, (uint64_t)(uintptr_t)mem + o, l); }
        if (full) r--;
    }
    if (vmin) *vmin = r < n ? r : n - 1;
    return b;
}
// A sparse residency image (vka_sparse.c) instead of a dense one: a sampled sparse pool texture, 2D or 2D array, single-sampled, with NVN tiled levels from base on, whose dense
// image from base would take DM_VK_SPARSE_RES_MB (default 32) or more; or a GPU-written one (sparse_gpu) from level 0. Smaller ones stay dense (the tiles' bookkeeping and binds cost more than they save)
static int res_candidate(const tex_t *t, const vka_fmt *f, uint32_t base) {
    const nvn_texdesc *d = &t->d; uint32_t tw, th, td;
    if (!vks_on() || !(sparse_sampled(t, f) || (base == 0 && sparse_gpu(t, f))) || (d->target != NVN_TEXTURE_TARGET_2D && d->target != NVN_TEXTURE_TARGET_2D_ARRAY) || d->samples > 1) return 0;
    if (nvn_tex_sparse_tiles(d, &tw, &th, &td) <= (int)base || td != 1) return 0;
    uint64_t lo = nvn_tex_level_offset(d, base), ly = t->L.layer_size > lo ? t->L.layer_size - lo : 0;
    return (double)ly * (t->L.layers ? t->L.layers : 1) >= VKA_ENVF("DM_VK_SPARSE_RES_MB", 32.0) * 1048576.0;
}
static vka_img *img_create_(const tex_t *t, const vka_fmt *f, uint32_t base, int res, int *why, uint64_t *need);
// A sparse texture with no residency image (its format has no sparse residency, or the residency image was refused) is a dense image from base: Mario Kart World's 8192x8192 x68
// terrain arrays are 1.45 GB of ASTC 8x8 and 6.08 GB of BC6H each that way, of which the game maps 3-12%. The Adreno 840 gives sparse residency to no block-compressed format
// (probe 2026-11-03: vkGetPhysicalDeviceSparseImageFormatProperties returns no properties for any BC, ETC2 or ASTC format, with any usage, flags or format list; RGBA8 and
// RGBA16F have 128x128 / 128x64 tiles), and an uncompressed residency image would take 16x (ASTC 8x8 to RGBA8) or 8x (BC6H to RGBA16F) the bytes of each mapped tile and more
// sparse address space than the device has (4 GiB). So a dense image of a sparse texture starts at the finest level from base whose levels take at most DM_VK_SPARSE_DENSE_MB
// (default 128; NVN storage bytes, which the Vulkan image's size is close to; 0: no cap): its finer levels are left out and sampled from that level, as for the address space (bfloor). A platform limit, logged
static uint64_t dense_bytes(const nvn_texdesc *d, const nvn_texlayout *L, uint32_t b) {
    uint64_t lo = nvn_tex_level_offset(d, b); return (L->layer_size > lo ? L->layer_size - lo : 0) * (L->layers ? L->layers : 1);
}
static uint32_t dense_floor(const nvn_texdesc *d, const nvn_texlayout *L, uint32_t b) {
    double cap = VKA_ENVF("DM_VK_SPARSE_DENSE_MB", 128.0) * 1048576.0; uint32_t n = d->levels ? d->levels : 1;
    if (cap <= 0 || d->target == NVN_TEXTURE_TARGET_3D || !sparse_levels_on()) return b;
    while (b + 1 < n && (double)dense_bytes(d, L, b) > cap) b++;
    return b;
}
// The image of texture t from level base: a sparse residency image when it is one (res_candidate). One that does not fit the sparse address space starts at a coarser level
// (bfloor: its finer levels are left out, sampled from the coarser one; a platform limit, logged), down to where it is no candidate any more, then dense from there
static vka_img *img_create(const tex_t *t, const vka_fmt *f, uint32_t base) {
    uint32_t b = base; uint64_t need = 0; int wlast = 0;   // (the address space the last finer level asked for and did not get: once that much is free, a finer start is tried again, sparse_refresh)
    for (; res_candidate(t, f, b); b++) {
        int why = 0; vka_img *im = img_create_(t, f, b, 1, &why, &need); wlast = why;
        if (im) {
            im->bfloor = b > base ? b : 0; im->bneed = need;   // (0: nothing left out; a later rebase may go finer)
            if (b > base) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 16) { char st[256]; vks_stats(st, sizeof st);
                VRED("sparse texture serial %llu %ux%u layers %u format %#x: levels %u-%u left out, the sparse address space is full (%s)", (unsigned long long)t->serial, t->d.width, t->d.height,
                     im->layers, t->d.format, base, b - 1, st); } }
            return im;
        }
        if (why != 1 || !sparse_sampled(t, f)) { b = base; break; }   // refused for another reason (3: the format has no sparse residency at this shape), or a GPU-written one (drawn into at level 0: no coarser start): dense from base
    }
    uint32_t bd = sparse_sampled(t, f) ? dense_floor(&t->d, &t->L, b) : b;   // (at most DM_VK_SPARSE_DENSE_MB from there)
    if (bd > b) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 16)
        VRED("sparse texture serial %llu %ux%u layers %u format %#x: no residency image; dense from level %u (levels %u-%u left out: %.0f MiB from level %u, %.0f MiB from level %u, DM_VK_SPARSE_DENSE_MB)",
             (unsigned long long)t->serial, t->d.width, t->d.height, t->L.layers, t->d.format, bd, b, bd - 1, dense_bytes(&t->d, &t->L, b) / 1048576.0, b, dense_bytes(&t->d, &t->L, bd) / 1048576.0, bd); }
    int capped = bd > b; b = bd;
    vka_img *im = img_create_(t, f, b, 0, NULL, NULL);
    // dcap only when residency is not to be tried again (a transient refusal, 2, retries it at the next finer start: sparse_refresh does not clamp it)
    if (im) { im->bfloor = b > base ? b : 0; im->bneed = need; im->dcap = (uint8_t)(capped && wlast != 2); }
    if (im && b > base && !capped) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 16)
        VRED("sparse texture serial %llu %ux%u format %#x: dense from level %u (levels %u-%u left out: the sparse address space is full)", (unsigned long long)t->serial, t->d.width, t->d.height, t->d.format, b, base, b - 1); }
    return im;
}
// res: a sparse residency image; *why (with res, on NULL): 1 = the sparse address space is full, 2 = refused otherwise (a dense one may be made), 3 = the format has no sparse
// residency at this shape (a dense one may be made)
static vka_img *img_create_(const tex_t *t, const vka_fmt *f, uint32_t base, int res, int *why, uint64_t *need) {
    const nvn_texdesc *d = &t->d;
    if (d->target == NVN_TEXTURE_TARGET_BUFFER) { VRED_ONCE("texture buffers are not images yet (first: serial %llu)", (unsigned long long)t->serial); return NULL; }
    vka_img *im = calloc(1, sizeof *im);
    im->serial = t->serial; im->mem = t->mem; im->d = *d; im->L = t->L; im->f = *f; im->refs = 1;
    if (t->pool && pool_sparse(t->pool)) im->vpool = t->pool;
    im->pool = t->pool; im->pfin = nvn_pool_fin(); im->pgen = nvn_pool_live(t->pool, NULL, NULL);   // (the count first: a pool event in between is seen at the next check)
    // 1D and 1D-array textures are one-row 2D / 2D-array images, as nvn_mtl.m makes them: the translator declares every such read as a 2D one (no Dim1D in any DKB module),
    // and Vulkan has no 2D view of a 1D image, so a 1D image read as unbound
    int is3d = d->target == NVN_TEXTURE_TARGET_3D, is1d = d->target == NVN_TEXTURE_TARGET_1D || d->target == NVN_TEXTURE_TARGET_1D_ARRAY;
    im->w = d->width ? d->width : 1; im->h = is1d ? 1 : d->height ? d->height : 1; im->depth = is3d ? (d->depth ? d->depth : 1) : 1;
    im->layers = is3d ? 1 : t->L.layers ? t->L.layers : 1; im->levels = d->levels ? d->levels : 1; im->samples = d->samples > 1 ? (uint32_t)d->samples : 1;
    if (base >= im->levels || is3d) base = 0;
    im->base = base; im->levels -= base; im->w = im->w >> base ? im->w >> base : 1; im->h = im->h >> base ? im->h >> base : 1;
    im->aspect = aspect_of(f->kind);
    VkFormatFeatureFlags ff = feats(f->vk);
    VkImageUsageFlags u = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (ff & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) u |= VK_IMAGE_USAGE_SAMPLED_BIT;
    if ((ff & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) && !compressed(f)) u |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (ff & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) u |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    // An 8-bit sRGB texture is stored as its UNORM twin (vka_fmt), which has storage use: its storage views are UNORM (tex_view: f.vk), the raw bytes a shader image writes,
    // and its sRGB views leave storage out (vka_view). Ocarina of Time's movie network (NDVE) writes its 2560x1440 RGBA8 sRGB output through an RGBA8 image view
    if ((ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) && (d->flags & NVN_TEXTURE_FLAGS_IMAGE)) u |= VK_IMAGE_USAGE_STORAGE_BIT;
    // Views in another format: only an 8-bit / BC / ASTC UNORM image as its sRGB twin (NVN sRGB formats are stored as UNORM, and UNORM textures have sRGB views; the
    // only reinterpretation DKB's level makes), and the formats learned for its format and size (Learned view formats: float or integer ones of its texel size). A short
    // format list instead of a bare MUTABLE_FORMAT keeps Adreno's UBWC compression on the image; every other image takes views of its own format only
    // DM_VK_UBWC=0: every colour image (float or integer) MUTABLE_FORMAT without a list, as before (any view format; no UBWC on Adreno)
    int ubwc = VKA_ENV("DM_VK_UBWC", 1);   // live: images made from now on
    VkFormat alt = f->kind == 0 && srgb_twin(f->vk) != f->vk ? srgb_twin(f->vk) : VK_FORMAT_UNDEFINED;
    VkFormat fls[6] = {f->vk, alt}; int nfl = 2;
    if (ubwc && f->kind < 3) { pthread_mutex_lock(&g_vlmu); im->nx = vl_formats(f->vk, im->w, im->h, im->xfmt, VL_PER_IMAGE); pthread_mutex_unlock(&g_vlmu); for (int k = 0; k < im->nx; k++) fls[nfl++] = im->xfmt[k]; }   // learned view formats
    if (!alt && im->nx) fls[1] = fls[--nfl];   // (no sRGB twin: no UNDEFINED entry in the list)
    VkImageFormatListCreateInfo fli = {VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO, .viewFormatCount = (uint32_t)nfl, .pViewFormats = fls};
    VkImageCreateFlags fl = alt || im->nx || (!ubwc && f->kind < 3) ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
    im->alt = ubwc ? alt : VK_FORMAT_UNDEFINED; im->any_fmt = !ubwc && f->kind < 3;
    // A shader image over a colour format that has no storage use of its own (RGB9E5: Ocarina of Time's AA pass a335c1db6270c355 writes its 683x384 target through an
    // R32_UINT view): storage use through a listed view format that has it (EXTENDED_USAGE: the image's own format need not support every usage, its views do; view_usage
    // leaves storage out of the views in the image's own format). Without it the binding read as unbound and the whole picture came out dark in false colours
    if ((d->flags & NVN_TEXTURE_FLAGS_IMAGE) && !(u & VK_IMAGE_USAGE_STORAGE_BIT) && f->kind < 3 && !compressed(f) && (fl & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT))
        for (int k = 0; k < nfl; k++) if (fls[k] && fls[k] != f->vk && (feats(fls[k]) & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) { u |= VK_IMAGE_USAGE_STORAGE_BIT; fl |= VK_IMAGE_CREATE_EXTENDED_USAGE_BIT; break; }
    if (!res && !is3d && !is1d && im->w == im->h && im->layers >= 6 && im->layers % 6 == 0 && im->samples == 1) fl |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    uint32_t rtw = 0, rth = 0, rtd = 0; int ntl = res ? nvn_tex_sparse_tiles(d, &rtw, &rth, &rtd) : 0;
    if (res) {   // the format takes sparse residency at this shape, with the usage it would have (storage use dropped: sampled sparse textures have none)
        VkImageFormatProperties sp = {0}; uint32_t nsp = 0;
        VkImageCreateFlags sfl = fl | VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
        int gpu = !sparse_sampled(t, f);   // (sparse_gpu: drawn into, so its attachment and storage use stay, or no residency image)
        if (!gpu) u &= ~(VkImageUsageFlags)VK_IMAGE_USAGE_STORAGE_BIT;
        VkResult r1 = vk.vkGetPhysicalDeviceImageFormatProperties(V.pd, f->vk, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, u, sfl, &sp), r2 = r1;
        if (r1 && !gpu) { u &= ~(VkImageUsageFlags)VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT; r2 = vk.vkGetPhysicalDeviceImageFormatProperties(V.pd, f->vk, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, u, sfl, &sp); }
        if (!r2 && sp.maxArrayLayers >= im->layers && sp.maxMipLevels >= im->levels)
            vk.vkGetPhysicalDeviceSparseImageFormatProperties(V.pd, f->vk, VK_IMAGE_TYPE_2D, VK_SAMPLE_COUNT_1_BIT, u, VK_IMAGE_TILING_OPTIMAL, &nsp, NULL);
        if (!nsp || ntl <= (int)base) {   // logged once per VkFormat (up to 8 formats), with what refused it
            static _Atomic int seen[8]; int k = 0; for (; k < 8; k++) { int e = 0; if (atomic_compare_exchange_strong(&seen[k], &e, (int)f->vk) || e == (int)f->vk) break; }
            static _Atomic int logged[8]; if (k < 8 && !atomic_exchange(&logged[k], 1))
                VRED("VkFormat %d (serial %llu): no sparse residency at %ux%u x%u levels %u: format query %d/%d (layers %u levels %u), %u sparse format properties, usage %#x, flags %#x%s, NVN tiled levels %d from %u: dense images instead",
                     f->vk, (unsigned long long)t->serial, im->w, im->h, im->layers, im->levels, r1, r2, sp.maxArrayLayers, sp.maxMipLevels, nsp, u, sfl, alt || im->nx ? " (format list)" : "", ntl, base);
            *why = 3; free(im); return NULL; }
        fl = sfl;
    }
    // A 3D texture is drawn into slice by slice (NVN render target views of its slices, or all of them layered): 2D / 2D-array views of slices need this (Vulkan 1.1 maintenance1)
    if (is3d && (u & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) fl |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    // Zero-shared images (below): a large GPU-content image the caller makes (g_zwant: vka_tex_get, its content ST_ZERO) is made to alias the zeroed memory of its create info
    int zs = g_zwant && VKA_ENV("DM_VK_ZSHARE", 1) && vkz_candidate(t->L.size, (uint64_t)(VKA_ENVF("DM_VK_ZSHARE_MB", 4.0) * 1048576.0), res, im->samples, is3d, compressed(f),
                                                                    (d->flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_DISPLAY)) != 0);
    // Guest-aliased targets (below): such an image over live pool storage is bound to the allocation of the images over the same storage start instead (zero-shared memory not taken)
    int ra = g_zwant && ra_min() && !im->vpool && t->mem && t->pool && im->pgen && vkz_candidate(t->L.size, ra_min(), res, im->samples, is3d, compressed(f),
                                                                    (d->flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_DISPLAY)) != 0);
    if (ra) zs = 0;
    if (zs) fl |= VK_IMAGE_CREATE_ALIAS_BIT;
    VkImageType it = is3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    VkImageFormatProperties ip;
    for (int k = 0; k < 3 && vk.vkGetPhysicalDeviceImageFormatProperties(V.pd, f->vk, it, VK_IMAGE_TILING_OPTIMAL, u, fl, &ip); k++)
        u &= k == 0 ? ~(VkImageUsageFlags)VK_IMAGE_USAGE_STORAGE_BIT : ~(VkImageUsageFlags)(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    im->usage = u;
    VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = (alt || im->nx) && ubwc ? &fli : NULL, .flags = fl, .imageType = it, .format = f->vk, .extent = {im->w, im->h, im->depth}, .mipLevels = im->levels,
        .arrayLayers = im->layers, .samples = (VkSampleCountFlagBits)im->samples, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = u, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkResult r = vk.vkCreateImage(V.dev, &ci, NULL, &im->img);
    if (r && res) { VRED_ONCE("sparse residency image of serial %llu: vkCreateImage %d: dense images instead", (unsigned long long)t->serial, r); *why = 2; free(im); return NULL; }
    if (r) { VRED("texture serial %llu %ux%ux%u format %#x (VkFormat %d) levels %u layers %u samples %u: vkCreateImage %d", (unsigned long long)t->serial, im->w, im->h, im->depth, d->format, f->vk, im->levels, im->layers, im->samples, r); free(im); return NULL; }
    VkMemoryRequirements mr; vk.vkGetImageMemoryRequirements(V.dev, im->img, &mr);
    uint32_t mt = vka_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0); if (mt == UINT32_MAX) mt = vka_mem_type(mr.memoryTypeBits, 0, 0);
    if (res) {   // no memory of its own: tiles bound as the page table maps them (vks_tile), the rest now (vks_new)
        int ok = mt != UINT32_MAX && mr.alignment && 0x10000 % mr.alignment == 0;
        if (ok && !vks_va_take(mr.size)) { vk.vkDestroyImage(V.dev, im->img, NULL); free(im); *why = 1; *need = mr.size; return NULL; }
        if (ok && !(im->rs = vks_new(im->img, mt, im->w, im->h, im->levels, im->layers, rtw, rth, (uint32_t)ntl - base))) { vks_va_give(mr.size); ok = 0; }
        else if (!ok) VRED_ONCE("sparse residency image of serial %llu: memory alignment %llu, type %u: dense images instead", (unsigned long long)t->serial, (unsigned long long)mr.alignment, mt);
        if (!ok) { vk.vkDestroyImage(V.dev, im->img, NULL); free(im); *why = 2; return NULL; }
        im->va = mr.size; im->rs_tw = rtw; im->rs_th = rth; im->bytes = 0;
        static _Atomic int n; if (atomic_fetch_add(&n, 1) < 32) { char st[256]; vks_stats(st, sizeof st);
            VLOG("sparse residency image: serial %llu %ux%u layers %u levels %u+%u format %#x, tiles %ux%u, %u levels by tile, %.0f MiB of address space (%s)", (unsigned long long)t->serial, im->w, im->h,
                 im->layers, base, im->levels, d->format, rtw, rth, vks_tiled(im->rs), mr.size / 1048576.0, st); }
        return im;
    }
    int fresh = 0; uint64_t rafresh = 0;
    if (ra && mt != UINT32_MAX) {   // the key: the guest storage start (pool, its Initialize serial, address) and the create info without its extent, the memory type
        struct { const void *pool; uint64_t pgen, mem; VkImageCreateFlags fl; VkFormat f; VkImageType it; uint32_t lv, ly, sm; VkImageUsageFlags u; uint32_t mt, nfl; VkFormat fls[6]; } k;
        memset(&k, 0, sizeof k);
        k.pool = t->pool; k.pgen = im->pgen; k.mem = (uint64_t)(uintptr_t)t->mem; k.fl = fl; k.f = f->vk; k.it = it; k.lv = im->levels; k.ly = im->layers; k.sm = im->samples; k.u = u; k.mt = mt;
        if (ci.pNext) { k.nfl = (uint32_t)nfl; memcpy(k.fls, fls, (size_t)nfl * sizeof *fls); }
        if ((r = vka_imem_bind_alias(im->img, mt, &mr, &k, (uint32_t)sizeof k, &im->m, &rafresh))) ra = 0;   // (refused: the image's own memory, as before)
        else if (!im->m.a) ra = 0;   // (the table had no room: the allocation is its own)
    } else ra = 0;
    if (zs && mt != UINT32_MAX) {   // the key: the whole create info (ALIAS_BIT in its flags), the format list, the memory type and requirements (images that may alias consistently)
        struct { VkImageCreateFlags fl; VkFormat f; VkImageType it; uint32_t w, h, dp, lv, ly, sm; VkImageUsageFlags u; uint32_t mt, nfl; VkFormat fls[6]; VkDeviceSize size, al; } k;
        memset(&k, 0, sizeof k);
        k.fl = fl; k.f = f->vk; k.it = it; k.w = im->w; k.h = im->h; k.dp = im->depth; k.lv = im->levels; k.ly = im->layers; k.sm = im->samples; k.u = u; k.mt = mt; k.size = mr.size; k.al = mr.alignment;
        if (ci.pNext) { k.nfl = (uint32_t)nfl; memcpy(k.fls, fls, (size_t)nfl * sizeof *fls); }
        if ((r = vka_imem_bind_zero(im->img, mt, &mr, &k, (uint32_t)sizeof k, &im->m, &fresh))) zs = 0;   // (refused: the image's own memory, as before)
        else if (!im->m.z) zs = 0;   // (the table had no room: the allocation is its own)
    } else zs = 0;
    if (!im->m.dm && (r = vka_imem_bind(im->img, mt, &mr, &im->m))) {   // a run of a shared block, or a dedicated allocation (vka_imgmem.c)
        VRED("texture serial %llu: %llu bytes of image memory not allocated (%d; %.0f MiB in %d images, %llu image allocations, at most %u allocations)", (unsigned long long)t->serial,
             (unsigned long long)mr.size, r, g_img_bytes / 1048576.0, g_nlive, (unsigned long long)vka_imem_allocations(), V.props.limits.maxMemoryAllocationCount);
        vk.vkDestroyImage(V.dev, im->img, NULL); free(im); return NULL;
    }
    if (ra) {   // its bytes are the allocation's, counted once (vka_tex_mem)
        im->bytes = 0; im->ra_fresh = rafresh != 0; if (rafresh) { vka_mem_grew(rafresh); g_img_made += rafresh; }
        g_ra_n++; g_ra_b += mr.size;
        static _Atomic int n; if (atomic_fetch_add(&n, 1) < 32)
            VLOG("guest-aliased image: texture serial %llu %ux%u layers %u levels %u format %#x flags %#x at guest %p (pool %p): %.1f MiB at offset 0 of %s allocation of %.1f MiB",
                 (unsigned long long)t->serial, im->w, im->h, im->layers, im->levels, d->format, d->flags, (const void *)t->mem, t->pool, mr.size / 1048576.0,
                 rafresh ? "a new" : "the storage's", im->m.asz / 1048576.0);
    } else if (zs) {   // its bytes are the shared allocation's, counted once (vka_tex_mem)
        im->bytes = 0; if (fresh) { vka_mem_grew(mr.size); g_img_made += mr.size; }
        static _Atomic int n; if (atomic_fetch_add(&n, 1) < 32)
            VLOG("zero-shared image: texture serial %llu %ux%u layers %u levels %u format %#x flags %#x: %.1f MiB of zeroed memory %s (memory of its own at its first write)", (unsigned long long)t->serial,
                 im->w, im->h, im->layers, im->levels, d->format, d->flags, mr.size / 1048576.0, fresh ? "made for its create info" : "shared with images of the same create info");
    } else { im->bytes = mr.size; g_img_bytes += mr.size; g_img_made += mr.size; vka_mem_grew(mr.size); }
    if (mr.size >= (VkDeviceSize)(VKA_ENVF("DM_VK_BIGIMG_MB", 64.0) * 1048576.0)) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 32)   // large images, for memory walls
        VLOG("large image: texture serial %llu %ux%ux%u layers %u levels %u (base %u) samples %u format %#x target %d flags %#x: %.1f MiB of image memory (storage %.1f MiB at %p, pool flags %#x)",
             (unsigned long long)t->serial, im->w, im->h, im->depth, im->layers, im->levels, im->base, im->samples, t->d.format, t->d.target, t->d.flags, mr.size / 1048576.0,
             t->L.size / 1048576.0, (void *)t->mem, pool_flags(t->pool)); }
    { VkBuffer sb; VkDeviceSize so;   // could the image be bound on its storage's own pool memory (HANDOFF "pool-aliased images")? Measured only (vka_tex_alias_stats)
      im->pool_fit = t->mem && V.guest_type < 32 && (mr.memoryTypeBits >> V.guest_type & 1) && mr.size <= t->L.size && vka_addr((uintptr_t)t->mem, t->L.size ? t->L.size : 1, &sb, &so, NULL)
                     && mr.alignment && so % mr.alignment == 0; }
    if ((u & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) && im->w >= 1024 && im->levels == 1 && im->layers == 1) { static int n; if (n++ < 6)   // UBWC shows as room for its metadata
        VLOG("target %ux%u VkFormat %d%s: %llu bytes (%.2f a texel)", im->w, im->h, f->vk, im->alt ? " (format list)" : fl ? " (mutable)" : "", (unsigned long long)mr.size, (double)mr.size / ((double)im->w * im->h)); }
    return im;
}

// Where a new image's content comes from (nvn_mtl.m tex_mtl_, tex_from_storage, tex_cpu_content, tex_init_fill, tex_linear). On the Switch a texture is its storage: the GPU reads
// what the pool holds, whoever wrote it. The replay keeps images in memory of their own, so content the CPU put into the storage is copied in:
//   ST_TRACK: uploaded now and again whenever the CPU writes its pages (write-protected after each upload, nvn_dirty.c; vka_tex_prepare re-uploads at the next submission):
//     a packaged layout (BNTX assets: the game loads the file into the pool and sets the layout; only compressed formats used to be uploaded, so every uncompressed asset
//     texture, font sheets among them, read zeros: Donkey Kong Bananza's menus had their panels but no text), CPU-written ones (WriteTexels, packaged data), linear textures in a
//     pool the CPU can write (Metal makes those on the pool buffer itself: SMBW's glyph cache, rasterized into an R8 linear texture by the CPU), and block-compressed textures
//     whose storage holds data at their first use (the GPU cannot render into them)
//   ST_FILL: uploaded once: what a CPU_NO_ACCESS or CPU_UNCACHED pool, or a range the CPU flushed before the first use, holds (the CPU cannot or did not say it would write it again)
//   ST_ZERO: GPU content (render targets, copy destinations): cleared
// The GPU writing an image (gpu_owned) ends its tracking, as on Metal: its content is the GPU's from then on. DM_VK_TEX_TRACK=0 uploads ST_TRACK textures once (no write protection)
enum { ST_ZERO, ST_FILL, ST_TRACK, ST_SPARSE };
// The images over each pool's storage, by content (hle_svc.c's footprint lines, DM_MEM_FOOT; dm/host/mem_foot.h says what each count means): p sorted by address. An image
// counts in the last pool that starts at or below its storage and holds it; images of sparse virtual pools (read through their page table) and without storage are left out
void vka_tex_pool_use(dm_pool_foot *p, int n) {
    if (n <= 0) return;
    pthread_mutex_lock(&g_tmu);
    for (int i = 0; i < g_nlive; i++) { const vka_img *im = g_live[i];
        if (!im->mem || im->vpool) continue;
        uint64_t a = (uint64_t)(uintptr_t)im->mem; int lo = 0, hi = n;
        while (lo < hi) { int c = (lo + hi) / 2; if (p[c].lo <= a) lo = c + 1; else hi = c; }
        int k = lo - 1; for (int b = 0; k >= 0 && a - p[k].lo >= p[k].size && b < 8; b++) k--;   // (pools that share a start or lie in another: a few back)
        if (k < 0 || a - p[k].lo >= p[k].size) continue;
        dm_pool_foot *f = &p[k]; uint64_t st = im->L.size, by = im->m.size ? im->m.size : im->bytes;
        if (im->mdrop || im->prov) { f->n_drop++; f->s_drop += st; }
        else if (im->gpu_owned || im->mode == ST_ZERO) { f->n_gpu++; f->s_gpu += st; }
        else if (im->mode == ST_TRACK && im->track) { f->n_track++; f->s_track += st; }
        else { f->n_fill++; f->s_fill += st; }
        f->img += by;
    }
    pthread_mutex_unlock(&g_tmu);
}
int nvn_flushed(const void *p, uint64_t n);   // hle_nvn.c
static int mem_any(const uint8_t *p, uint64_t n) { uint64_t i = 0, v; for (; i + 8 <= n; i += 8) { memcpy(&v, p + i, 8); if (v) return 1; } for (; i < n; i++) if (p[i]) return 1; return 0; }
static uint64_t g_st_up[4], g_reup;
// Time and bytes of uploads from guest memory (deswizzle / row copy into scratch and the copy commands), on the replay thread: the [longframe] line's "texture uploads ... MB / ms"
// (hle_nvn.c mtl_upload_ns_take, mtl_upload_bytes; stubs that said 0 until 2026-11-03)
static _Atomic uint64_t g_up_ns, g_up_bytes;
uint64_t mtl_upload_ns_take(void) { return atomic_exchange(&g_up_ns, 0); }
uint64_t mtl_upload_bytes(void) { return atomic_load(&g_up_bytes); }
// mem_any over guest storage: pages the guest has not mapped are skipped (inaccessible; they read as zeros: unmapped_content)
static int mem_any_mapped(const uint8_t *c, const tex_t *t) {
    if (c != t->mem || !dm_guest_unmapped((uintptr_t)c, t->L.size)) return mem_any(c, t->L.size);
    for (uintptr_t p = (uintptr_t)c, e = p + t->L.size; p < e; ) {
        uintptr_t q = (p & ~(uintptr_t)0xFFF) + 0x1000; if (q > e) q = e;
        uint8_t pg[0x1000];   // copied, not read in place: the guest may unmap the page after the check (dm_guest_read_mapped)
        if (!dm_guest_unmapped(p, 1) && (dm_guest_read_mapped(pg, p, q - p), mem_any(pg, q - p))) return 1;
        p = q;
    }
    return 0;
}
// Switch 2 (NVN2) memory is coherent: the driver's FlushMappedRange / FlushTexels are bare returns, so the GPU reads what the CPU stored, flushed or not. A texture whose first
// use is a sample (a texture ID written into the sampled table: write_tex / id_view, g_first_sampled) that the GPU has not written holds what its storage holds, whatever its
// format. So on NVN2 an uncompressed texture in a pool the CPU can write is taken from its storage (tracked: re-uploaded when the CPU writes its pages, flushed or uncached pools
// too) when its first use is a sample, unless an image the GPU drew into already lies over the same storage (a render target sampled through another texture object keeps the GPU's
// content; one drawn only later is not seen here, and its draw ends the tracking: gpu_owned). Storage still all zero is tracked too: a later texture object of the same shape over a
// reused slot shares this image (same_shape) and never has a first use of its own, so the CPU's later fill must reach it through the tracking. The sampled table's descriptors are
// written for every dirty registered ID at each submission (vka_tex_prepare → write_tex), so a render target registered for sampling counts as sampled first when its draw comes later:
// its stale storage is uploaded (what the Switch shows too) until its first draw makes it GPU content.
// Mario Kart World stores its terrain tiles (260x260 R16 height, RG8 material, RGBA8 and R11G11B10F colour, 68x68 RGBA16; MINIMAL_LAYOUT, one CPU_CACHED | GPU_CACHED pool,
// flags 0x24) into reused slots before nvnTextureInitialize and never flushes them: as zero images its ground beside the roads came out flat and untextured. A texture first used
// as an attachment, a copy destination or a shader image stays GPU content. The rule and the finding follow kartnative (pahorema/kartnative, MIT: "Switch 2: uncompressed storage
// the CPU wrote before a texture's first sample", nvn_mtl.m tex_cpu_content). DM_VK_STORAGE_FIRST_USE=0 turns it off (to measure it)
static _Thread_local int g_first_sampled;
static int img_pool_dead(vka_img *im);
static int gpu_image_over(const uint8_t *mem) {   // caller holds g_tmu; images of finalized pools do not count (a new pool may lie at the same address)
    for (int i = 0; i < g_nlive; i++) if (g_live[i]->mem == mem && g_live[i]->gpu_owned && !img_pool_dead(g_live[i])) return 1;
    return 0;
}
static int nvn2_sampled_content(const tex_t *t, const vka_fmt *f, unsigned pf, int sampled) {
    static int on = -1; if (on < 0) on = VKA_ENV("DM_VK_STORAGE_FIRST_USE", 1) != 0 && vka_nvn2();   // (vka_nvn2: captures too)
    const nvn_texdesc *d = &t->d;
    if (!on || !sampled || !t->pool || t->has_layout || compressed(f) || f->kind >= 3) return 0;
    if (pf & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL)) return 0;
    if (d->flags & (NVN_TEXTURE_FLAGS_IMAGE | NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_SPARSE)) return 0;
    return !gpu_image_over(t->mem);
}
static uint64_t g_nvn2_first[2];   // textures taken from their storage at a first sample (nvn2_sampled_content): [0] with data, [1] storage all zero (tracked from zeros)
static int storage_mode(const tex_t *t, const uint8_t *c, const vka_fmt *f, int cpuw, int sampled) {   // c: the storage's content as the GPU will see it (t->mem or gw_content's copy)
    const nvn_texdesc *d = &t->d;
    if (!t->mem || d->samples > 1 || f->kind >= 3 || d->target == NVN_TEXTURE_TARGET_BUFFER) return ST_ZERO;
    // A sparse virtual pool's texture (nvn_mtl.m tex_from_storage): a sampled one holds what the CPU wrote into the physical pages mapped behind it (streamed textures), read
    // through the page table and uploaded again when its mappings or those pages change (ST_SPARSE); render targets and images are GPU content. Its address is reserved, inaccessible
    // memory where nothing is mapped (no host alias on Android), so mem_any and the other checks below must not read it
    int sp = 0; unsigned pf = 0; if (t->pool && !nvn_pool_live(t->pool, &pf, &sp)) return ST_ZERO;   // a finalized pool: its memory may be unmapped (treg_dead)
    if (sp) return sparse_sampled(t, f) ? ST_SPARSE : ST_ZERO;
    int linear = (d->flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) != 0, comp = compressed(f);
    int want = ST_ZERO;
    if (cpuw || t->has_layout) want = ST_TRACK;
    else if (linear && t->pool) want = ST_TRACK;
    else if (comp && !(d->flags & (NVN_TEXTURE_FLAGS_IMAGE | NVN_TEXTURE_FLAGS_COMPRESSIBLE | NVN_TEXTURE_FLAGS_SPARSE)) && mem_any_mapped(c, t)) want = ST_TRACK;
    else if (nvn2_sampled_content(t, f, pf, sampled)) {
        int any = mem_any_mapped(c, t); uint64_t k = g_nvn2_first[!any]++; want = ST_TRACK;
        if (any && k < 4) VLOG("texture serial %llu %ux%ux%u format %#x pool flags %#x: written by the CPU before its first use, a sample (NVN2): uploaded from its storage, re-uploaded when it changes (%llu so far, %llu more with empty storage)",
                               (unsigned long long)t->serial, d->width, d->height, d->depth, d->format, pf, (unsigned long long)k + 1, (unsigned long long)g_nvn2_first[1]);
    }
    else if (!(d->flags & NVN_TEXTURE_FLAGS_COMPRESSIBLE) && t->pool && ((pf & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED)) || nvn_flushed(t->mem, t->L.size))
             && mem_any_mapped(c, t)) want = ST_FILL;
    // No later CPU writes to follow: a pool the CPU cannot access, a virtual pool (its pages are not the storage's), storage outside every tracked pool, or tracking off
    int trk = VKA_ENV("DM_VK_TEX_TRACK", 1) != 0;   // live: textures made from now on
    if (want == ST_TRACK && (!trk || (pf & (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_VIRTUAL)) || !nvn_dirty_span(t->mem, t->L.size))) want = ST_FILL;
    // Diagnostic DM_VK_ZEROLOG=<n>: the first n textures made as zero images whose storage holds data (virtual pools left out: their range may be reserved, inaccessible memory) (what they are, and whether their first use was a sample)
    { int zl = VKA_ENV("DM_VK_ZEROLOG", 0); static _Atomic int nz;
      if (zl > 0 && want == ST_ZERO && !(pf & NVN_MEMORY_POOL_FLAGS_VIRTUAL) && atomic_load(&nz) < zl && mem_any_mapped(c, t) && atomic_fetch_add(&nz, 1) < zl)
          VLOG("zero image over data: serial %llu %ux%ux%u levels %u format %#x target %d flags %#x pool flags %#x layout %d sampled %d cpuw %d gpu over %d",
               (unsigned long long)t->serial, d->width, d->height, d->depth, d->levels, d->format, d->target, d->flags, pf, t->has_layout, sampled, cpuw, gpu_image_over(t->mem)); }
    return want;
}
static uint64_t mem_hash(const uint8_t *p, uint64_t n) { uint64_t h = 1469598103934665603ull, v; uint64_t i = 0; for (; i + 8 <= n; i += 8) { memcpy(&v, p + i, 8); h = (h ^ v) * 1099511628211ull; } for (; i < n; i++) h = (h ^ p[i]) * 1099511628211ull; return h; }
// GPU writes into guest memory. On the Switch a texture is its storage, so a copy, fill or inline update into that memory (or a shader's store into a storage buffer over it)
// changes what the texture reads from then on, in GPU order. Here a texture is an image of its own, which the CPU uploads from the storage when the replay records its first use
// (or again after a CPU write the dirty tracker saw). A GPU write is not a CPU write: it does not fault, and the CPU sees its bytes only when the submission holding it completes.
// Donkey Kong Bananza streams textures that way (a buffer copy into the pool, then the first draw with the texture, in the same submission): the image got the bytes the storage held
// before the copy (an earlier texture's or zeros; ASTC blocks of another block size or all zero decode to NaN on the Adreno 840, so its lamps, rocks and signs rendered as black
// cores with white bloom), and kept them, since no fault ever came. So each GPU write is kept here, in recording order (seq), with its submission's fence:
//   - an upload reads the storage as the GPU will see it at that point: the copies, fills and inline updates not yet done are applied over a copy of the storage (gw_content);
//   - once a write's fence is done, the images over its range whose upload did not include it (gw_seen) are uploaded again (vka_tex_prepare): writes whose bytes the host cannot
//     know at recording (shader stores, texture → buffer copies), and images first used before the write
// Images whose content is the GPU's (gpu_owned: render targets, copy destinations) keep it, as for CPU writes
enum { GW_COPY, GW_FILL, GW_UPDATE, GW_OPAQUE };
typedef struct { uint64_t lo, hi, src, seq, fence; uint32_t kind, value; uint8_t *data; } gw_t;
static gw_t *g_gw; static int g_ngw, g_gwcap; static uint64_t g_gwseq = 1, g_gw_redo, g_gw_over, g_gw_late;
static void gw_add(int kind, uint64_t dst, uint64_t n, uint64_t src, uint32_t value, const void *data) {
    if (!n) return;
    pthread_mutex_lock(&g_tmu);
    if (g_ngw == g_gwcap) { g_gwcap = g_gwcap ? 2 * g_gwcap : 256; g_gw = realloc(g_gw, (size_t)g_gwcap * sizeof *g_gw); }
    if (kind == GW_OPAQUE)   // the same range again in this submission (a storage buffer bound by draw after draw): one entry, at the newest place
        for (int i = g_ngw - 1; i >= 0 && i >= g_ngw - 16 && !g_gw[i].fence; i--) if (g_gw[i].kind == GW_OPAQUE && g_gw[i].lo == dst && g_gw[i].hi == dst + n) {
            gw_t w = g_gw[i]; memmove(&g_gw[i], &g_gw[i + 1], (size_t)(g_ngw - 1 - i) * sizeof *g_gw); w.seq = g_gwseq++; g_gw[g_ngw - 1] = w; pthread_mutex_unlock(&g_tmu); return; }
    uint8_t *cp = NULL; if (kind == GW_UPDATE && (cp = malloc(n))) memcpy(cp, data, n); else if (kind == GW_UPDATE) kind = GW_OPAQUE;
    g_gw[g_ngw++] = (gw_t){dst, dst + n, src, g_gwseq++, 0, (uint32_t)kind, value, cp};
    pthread_mutex_unlock(&g_tmu);
}
void vka_tex_gw_copy(uint64_t src, uint64_t dst, uint64_t n) { gw_add(GW_COPY, dst, n, src, 0, NULL); }
void vka_tex_gw_fill(uint64_t dst, uint64_t n, uint32_t value) { gw_add(GW_FILL, dst, n, 0, value, NULL); }
void vka_tex_gw_update(uint64_t dst, uint64_t n, const void *data) { gw_add(GW_UPDATE, dst, n, 0, 0, data); }
void vka_tex_gw_opaque(uint64_t dst, uint64_t n) { gw_add(GW_OPAQUE, dst, n, 0, 0, NULL); }
// Stores through raw global memory (an address the shader loaded: dmsh_abi.h DMSH_GMEM_LOG_*) land where only the GPU knows: each draw or dispatch with such stores gets a write
// log of its own, read once its submission is done; its pages become writes of unknown bytes, done (the images over them are uploaded again)
enum { GML_N = 256, GML_BYTES = (1 + (1 << DMSH_GMEM_LOG_LOG2)) * 4 };
static struct { vka_span s; uint64_t fence; int used; } g_gml[GML_N]; static int g_gml_init; static uint64_t g_gml_pages, g_gml_full, g_gml_none;
int vka_tex_gml_take(VkDescriptorBufferInfo *bi) {   // 0 = no log free (the caller binds an unread one)
    pthread_mutex_lock(&g_tmu);
    if (!g_gml_init) { g_gml_init = 1; for (int i = 0; i < GML_N; i++) if (!vka_static(GML_BYTES, V.sb_align, &g_gml[i].s)) break; }
    for (int i = 0; i < GML_N; i++) if (!g_gml[i].used && g_gml[i].s.ptr) {
        g_gml[i].used = 1; g_gml[i].fence = 0; memset(g_gml[i].s.ptr, 0, GML_BYTES);
        *bi = (VkDescriptorBufferInfo){g_gml[i].s.buf, g_gml[i].s.off, GML_BYTES}; pthread_mutex_unlock(&g_tmu); return 1; }
    g_gml_none++; pthread_mutex_unlock(&g_tmu);
    VRED_ONCE("more than %d raw global memory write logs in flight: stores of later draws are not followed into textures", GML_N);
    return 0;
}
static int img_by_mem(const void *a, const void *b) { uintptr_t x = (uintptr_t)(*(vka_img *const *)a)->mem, y = (uintptr_t)(*(vka_img *const *)b)->mem; return x < y ? -1 : x > y; }
static void gml_poll(void) {   // caller holds g_tmu
    uint64_t done = vka_done();
    for (int i = 0; i < GML_N; i++) {
        if (!g_gml[i].used || !g_gml[i].fence || g_gml[i].fence > done) continue;
        const uint32_t *w = (const uint32_t *)g_gml[i].s.ptr; g_gml_full += w[0] != 0;
        if (w[0]) VRED_ONCE("a raw global memory write log ran out of slots: some stores are not followed into textures");
        for (int k = 1; k <= 1 << DMSH_GMEM_LOG_LOG2; k++) if (w[k]) {
            if (g_ngw == g_gwcap) { g_gwcap = g_gwcap ? 2 * g_gwcap : 256; g_gw = realloc(g_gw, (size_t)g_gwcap * sizeof *g_gw); }
            uint64_t a = (uint64_t)w[k] << 12; g_gw[g_ngw++] = (gw_t){a, a + 4096, 0, g_gwseq++, g_gml[i].fence, GW_OPAQUE, 0, NULL}; g_gml_pages++;
        }
        g_gml[i].used = 0;
    }
}
// DM_VK_NANSCAN storage watch: the first 64 bytes of every tracked texture's storage, checked at each submission; a change the dirty tracker did not see (no CPU fault) is logged
// with the compute shaders the submissions since its upload dispatched (a GPU writer)
enum { WD_N = 4096 }; static struct { uint64_t fence, key; } g_wd[WD_N]; static uint64_t g_nwd;
void vka_tex_watch_dispatch(uint64_t key) {
    uint64_t f = vka_fence_last() + 1;
    for (uint64_t i = g_nwd > 64 ? g_nwd - 64 : 0; i < g_nwd; i++) if (g_wd[i % WD_N].fence == f && g_wd[i % WD_N].key == key) return;
    g_wd[g_nwd % WD_N].fence = f; g_wd[g_nwd % WD_N].key = key; g_nwd++;
}
uint64_t vka_frame(void);
// The GPU's view of the same bytes: copied out of the pool's buffer in this submission, compared with what the CPU reads once it is done
typedef struct { vka_span s; const uint8_t *mem; uint64_t serial; } gv_t;
static void gview_report(void *arg) {
    gv_t *g = arg; char a[40], b[40]; for (int j = 0; j < 16; j++) { snprintf(a + 2 * j, 3, "%02x", g->s.ptr[j]); snprintf(b + 2 * j, 3, "%02x", g->mem[j]); }
    char sd[400]; void vka_mem_describe(uint64_t, char *, size_t); vka_mem_describe((uint64_t)(uintptr_t)g->mem, sd, sizeof sd);
    char mp_[200] = ""; { FILE *f = fopen("/proc/self/maps", "re"); char ln[512]; uintptr_t ad = (uintptr_t)g->mem;
        while (f && fgets(ln, sizeof ln, f)) { unsigned long lo, hi; if (sscanf(ln, "%lx-%lx", &lo, &hi) == 2 && ad >= lo && ad < hi) { ln[strcspn(ln, "\n")] = 0; snprintf(mp_, sizeof mp_, "%.120s", ln); break; } } if (f) fclose(f); }
    VLOG("NaN scan: watch: serial %llu at %p: the GPU reads %s, the CPU reads %s (%s); %s; maps %s", (unsigned long long)g->serial, (const void *)g->mem, a, b, memcmp(g->s.ptr, g->mem, 256) ? "DIFFERENT memory" : "the same", sd, mp_);
}
static void gview(VkCommandBuffer cb, const vka_img *im) {
    static gv_t g[16]; static int n; if (n >= 16 || !cb) return;
    VkBuffer b; VkDeviceSize o; if (!vka_addr((uint64_t)(uintptr_t)im->mem, 256, &b, &o, NULL)) return;
    gv_t *x = &g[n]; if (!vka_static(256, 16, &x->s)) return; n++;
    x->mem = im->mem; x->serial = im->serial;
    vka_barrier(cb); VkBufferCopy c = {o, x->s.off, 256}; vk.vkCmdCopyBuffer(cb, b, x->s.buf, 1, &c); vka_barrier(cb);
    vka_after(vka_fence_last() + 1, gview_report, x);
}
// DM_VK_NANSCAN: placed memory against mprotect, once. Six 4 KB pages of a placed pool: each written by the CPU, some protected read-only and back as the dirty tracker
// does, then copied by the GPU into a buffer; when the submission is done the GPU's bytes are compared with what the CPU reads
static uint8_t *g_pt; static vka_span g_pt_out;
static void ptest_report(void *arg) {
    (void)arg; static const char *what[6] = {"written, untouched", "written, then read-only, then read-write", "written, then read-only (still read-only)", "read-only then read-write, then written",
                                             "written, read-only, read-write, written again", "written, read-only and back 64 times"};
    for (int p = 0; p < 6; p++) { const uint8_t *c = g_pt + 4096 * p, *g = g_pt_out.ptr + 4096 * p; int d = 0; for (int i = 0; i < 4096; i++) d += c[i] != g[i];
        VLOG("NaN scan: placed memory test: page %d (%s): %d of 4096 bytes differ between the CPU (%02x%02x%02x%02x) and the GPU (%02x%02x%02x%02x)", p, what[p], d, c[0], c[1], c[2], c[3], g[0], g[1], g[2], g[3]); }
}
static void ptest(VkCommandBuffer cb) {
    static int done; if (done || !cb) return; done = 1;
    uint8_t *m = mmap(NULL, 1 << 16, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); if (m == MAP_FAILED) return;
    memset(m, 0x11, 1 << 16);
    mtl_pool_add(m, 1 << 16);
    VkBuffer b; VkDeviceSize o; if (!vka_addr((uint64_t)(uintptr_t)m, 6 * 4096, &b, &o, NULL) || !vka_static(6 * 4096, 16, &g_pt_out)) { VLOG("NaN scan: placed memory test: not placed"); return; }
    g_pt = m; uint8_t *P[6]; for (int i = 0; i < 6; i++) P[i] = m + 4096 * i;
    memset(P[0], 0xa0, 4096);
    memset(P[1], 0xa1, 4096); mprotect(P[1], 4096, PROT_READ); mprotect(P[1], 4096, PROT_READ | PROT_WRITE);
    memset(P[2], 0xa2, 4096); mprotect(P[2], 4096, PROT_READ);
    mprotect(P[3], 4096, PROT_READ); mprotect(P[3], 4096, PROT_READ | PROT_WRITE); memset(P[3], 0xa3, 4096);
    memset(P[4], 0x44, 4096); mprotect(P[4], 4096, PROT_READ); mprotect(P[4], 4096, PROT_READ | PROT_WRITE); memset(P[4], 0xa4, 4096);
    memset(P[5], 0xa5, 4096); for (int k = 0; k < 64; k++) { mprotect(P[5], 4096, PROT_READ); mprotect(P[5], 4096, PROT_READ | PROT_WRITE); }
    vka_barrier(cb); VkBufferCopy c = {o, g_pt_out.off, 6 * 4096}; vk.vkCmdCopyBuffer(cb, b, g_pt_out.buf, 1, &c); vka_barrier(cb);
    vka_after(vka_fence_last() + 1, ptest_report, NULL);
}
// An image whose texture's pool was finalized since the image was made (or is another pool now): the game may have unmapped its storage, so it is not read again (re-uploads,
// scans); the image keeps its content. Caller holds g_tmu and nvn_pool_use from this check until it is done reading the storage. Cheap: the table is looked up only when a
// pool was finalized since the image's last check
static int img_pool_dead(vka_img *im) {
    if (im->pdead) return 1;
    if (!im->pool || !im->pgen) return 0;
    uint64_t f = nvn_pool_fin(); if (f == im->pfin) return 0;
    im->pfin = f;
    if (nvn_pool_live(im->pool, NULL, NULL) == im->pgen) return 0;
    im->pdead = 1; im->track = 0;   // (vtrack stays: view_of's level cut follows it; every loop that refreshes checks pdead first)
    static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VLOG("texture serial %llu (storage %p): its pool %p was finalized: its image keeps its content, its storage is not read again", (unsigned long long)im->serial, (void *)im->mem, im->pool);
    return 1;
}
static void watch_scan(VkCommandBuffer cb) {   // caller holds g_tmu and nvn_pool_use
    ptest(cb);
    static int nlog;
    if (nlog >= 24) return;
    for (int i = 0; i < g_nlive && nlog < 24; i++) {
        vka_img *im = g_live[i];
        if (!im->track || im->gpu_owned || !im->mem || !im->up_fence || img_pool_dead(im)) continue;
        uint64_t h = mem_hash(im->mem, im->L.size < 64 ? im->L.size : 64);
        if (h == im->head) continue;
        im->head = h;
        if (nvn_dirty_since_span(im->span, im->mem, im->L.size, im->stamp)) continue;   // a CPU write the tracker saw
        char keys[300] = ""; size_t o = 0; int nk = 0;
        for (uint64_t j = g_nwd > WD_N ? g_nwd - WD_N : 0; j < g_nwd && nk < 12; j++) if (g_wd[j % WD_N].fence >= im->up_fence) { o += (size_t)snprintf(keys + o, sizeof keys - o, " %016llx@%llu", (unsigned long long)g_wd[j % WD_N].key, (unsigned long long)g_wd[j % WD_N].fence); nk++; }
        nlog++;
        char why[600], mp_[160] = ""; nvn_dirty_debug(im->span, im->mem, im->L.size, im->stamp, why, sizeof why);
        { FILE *f = fopen("/proc/self/maps", "re"); char ln[512]; uintptr_t a = (uintptr_t)im->mem; int nv = 0;
          while (f && fgets(ln, sizeof ln, f)) { nv++; unsigned long lo, hi; if (!mp_[0] && sscanf(ln, "%lx-%lx", &lo, &hi) == 2 && a >= lo && a < hi) { ln[strcspn(ln, "\n")] = 0; snprintf(mp_, sizeof mp_, "%.100s", ln); } }
          if (f) fclose(f); size_t k = strlen(mp_); snprintf(mp_ + k, sizeof mp_ - k, " (%d mappings)", nv); }
        char df[200] = ""; if (im->snap) { size_t n = im->L.size < 256 ? im->L.size : 256, k = 0; while (k < n && im->snap[k] == im->mem[k]) k++; k &= ~(size_t)15;
            size_t o = (size_t)snprintf(df, sizeof df, "first change at +%#zx: ", k); for (size_t j = k; j < k + 16 && j < n; j++) o += (size_t)snprintf(df + o, sizeof df - o, "%02x", im->snap[j]);
            o += (size_t)snprintf(df + o, sizeof df - o, " -> "); for (size_t j = k; j < k + 16 && j < n; j++) o += (size_t)snprintf(df + o, sizeof df - o, "%02x", im->mem[j]);
            memcpy(im->snap, im->mem, n); }
        VLOG("NaN scan: watch: %s; frame %llu", df, (unsigned long long)vka_frame());
        gview(cb, im);
        VLOG("NaN scan: watch: storage of texture serial %llu (%p+%#llx, format %#x %ux%u) changed with no CPU fault; uploaded in submission %llu, now %llu done / %llu recorded; compute since:%s; %s; maps: %s",
             (unsigned long long)im->serial, (const void *)im->mem, (unsigned long long)im->L.size, im->d.format, im->w, im->h, (unsigned long long)im->up_fence, (unsigned long long)vka_done(), (unsigned long long)vka_fence_last(), keys, why, mp_);
    }
}
void vka_tex_gw_commit(uint64_t fence) {   // the writes recorded since the last commit belong to this submission
    pthread_mutex_lock(&g_tmu);
    for (int i = g_ngw - 1; i >= 0 && !g_gw[i].fence; i--) g_gw[i].fence = fence;
    for (int i = 0; i < GML_N; i++) if (g_gml[i].used && !g_gml[i].fence) g_gml[i].fence = fence;
    pthread_mutex_unlock(&g_tmu);
}
// The storage [mem, mem + n) as the GPU sees it after the writes recorded so far: NULL = memory as it is (no pending write over it), else a malloc'd copy with the pending writes
// applied in order. *seen = the newest write the content includes with every older one (the writes not done and not applied, opaque ones, cap it). Caller holds g_tmu
static uint8_t *gw_content(const uint8_t *mem, uint64_t n, uint64_t *seen) {
    uint64_t a = (uint64_t)(uintptr_t)mem, b = a + n, done = vka_done(), cap = g_gwseq - 1; uint8_t *c = NULL;
    for (int i = 0; i < g_ngw; i++) {
        gw_t *w = &g_gw[i];
        if (w->fence && w->fence <= done) continue;   // landed: memory has it
        if (w->hi <= a || w->lo >= b) continue;
        if (w->kind == GW_OPAQUE) { if (w->seq - 1 < cap) cap = w->seq - 1; continue; }
        if (!c) { if (!(c = malloc(n))) { cap = 0; break; } dm_guest_read_mapped(c, (uintptr_t)mem, n); }
        uint64_t lo = w->lo > a ? w->lo : a, hi = w->hi < b ? w->hi : b;
        if (w->kind == GW_FILL) { for (uint64_t k = lo; k < hi; k++) { uint32_t v = w->value; c[k - a] = ((const uint8_t *)&v)[(k - w->lo) & 3]; } }
        else if (w->kind == GW_UPDATE) memcpy(c + (lo - a), w->data + (lo - w->lo), hi - lo);
        else {   // a copy: its source as the GPU will read it, which an earlier pending write over the source would change (then the write lands later: re-uploaded)
            const uint8_t *s = (const uint8_t *)(uintptr_t)(w->src + (lo - w->lo));
            for (int j = 0; j < i; j++) { const gw_t *x = &g_gw[j]; if ((!x->fence || x->fence > done) && x->lo < w->src + (hi - w->lo) && x->hi > w->src + (lo - w->lo)) { if (w->seq - 1 < cap) cap = w->seq - 1; break; } }
            memmove(c + (lo - a), s, hi - lo);
        }
    }
    *seen = cap;
    return c;
}
// Storage the guest has not mapped (all or part of it): its content as zeros, in a copy (NULL: all mapped). A memory pool is placed memory only once the GPU gets an address
// in it (vka_mem.c mtl_pool_need); before, placing mapped the whole pool read-write, and Mario Kart World 1.8.0 uploads a texture of a CPU_NO_ACCESS pool whose pages its
// SDK has not mapped (SIGSEGV in the deswizzle at 0x4074003000, phone m22)
static uint8_t *unmapped_content(const tex_t *t) {
    uint64_t u = dm_guest_unmapped((uintptr_t)t->mem, t->L.size); if (!u) return NULL;
    uint8_t *c = malloc(t->L.size); if (!c) return NULL;
    dm_guest_read_mapped(c, (uintptr_t)t->mem, t->L.size);
    static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VRED("texture serial %llu: %llu of its %llu storage pages at %p are not mapped by the guest: read as zeros", (unsigned long long)t->serial,
                                                              (unsigned long long)u, (unsigned long long)((t->L.size + 4095) >> 12), (const void *)t->mem);
    return c;
}
static void upload_tracked(vka_img *im, const tex_t *t, VkCommandBuffer cb) {   // stamp, protect, then read (nvn_mtl.m pkg_upload): a write during the copy is seen at the next check
    im->stamp = nvn_dirty_now(); nvn_dirty_protect(t->mem, t->L.size);
    if (vka_nan_on()) im->hash = mem_hash(t->mem, t->L.size);
    uint8_t *c = im->vpool ? NULL : gw_content(t->mem, t->L.size, &im->gw_seen);   // (a sparse pool's storage is read through its page table: upload_)
    if (!c && !im->vpool) c = unmapped_content(t);
    if (c) { tex_t tc = *t; tc.mem = c; upload(im, &tc, cb); free(c); g_gw_over++; } else upload(im, t, cb);
    if (vka_nan_on()) { im->head = mem_hash(t->mem, t->L.size < 64 ? t->L.size : 64); im->up_fence = vka_fence_last() + 1;
        if (!im->snap) im->snap = malloc(256); if (im->snap) memcpy(im->snap, t->mem, t->L.size < 256 ? t->L.size : 256); }
}
static void upload_fill(vka_img *im, const tex_t *t, VkCommandBuffer cb) {   // the same without tracking (ST_FILL)
    uint8_t *c = im->vpool ? NULL : gw_content(t->mem, t->L.size, &im->gw_seen);
    if (!c && !im->vpool) c = unmapped_content(t);
    if (c) { tex_t tc = *t; tc.mem = c; upload(im, &tc, cb); free(c); g_gw_over++; } else upload(im, t, cb);
}
// A linear 2D texture of one level and layer whose pages the CPU wrote between byte lo and hi of its storage: only those rows again (Donkey Kong Bananza's glyph cache,
// 1024x8192 R8: the whole 8 MB was deswizzled and copied each time a glyph was rasterized into it). 0 = not such a texture (the caller uploads all of it)
static int upload_rows_tracked(vka_img *im, const tex_t *t, VkCommandBuffer cb, size_t lo, size_t hi) {
    const nvn_texdesc *d = &t->d; const vka_fmt *f = &im->f;
    if (!(d->flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) || im->levels > 1 || im->layers > 1 || im->depth > 1 || f->kind >= 3) return 0;
    uint32_t bx = (im->w + f->bw - 1) / f->bw, by = (im->h + f->bh - 1) / f->bh; size_t pitch = (size_t)bx * f->bpb, st = d->stride ? d->stride : pitch;
    if (st < pitch) return 0;
    uint32_t r0 = (uint32_t)(lo / st), r1 = (uint32_t)((hi + st - 1) / st); if (r1 > by) r1 = by;
    if (r0 >= r1) r0 = 0, r1 = by;
    im->stamp = nvn_dirty_now(); nvn_dirty_protect(t->mem, t->L.size);   // stamp, protect, then read (upload_tracked)
    vka_span s; if (!vka_scratch(pitch * (r1 - r0), 16, &s)) return 1;
    for (uint32_t y = r0; y < r1; y++) memcpy(s.ptr + (size_t)(y - r0) * pitch, t->mem + (size_t)y * st, pitch);
    uint32_t y0 = r0 * f->bh, y1 = r1 * f->bh > im->h ? im->h : r1 * f->bh;
    VkBufferImageCopy c = {s.off, bx * f->bw, (r1 - r0) * f->bh, {im->aspect, 0, 0, 1}, {0, (int32_t)y0, 0}, {im->w, y1 - y0, 1}};
    vk.vkCmdCopyBufferToImage(cb, s.buf, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    g_rows_up += pitch * (r1 - r0);
    return 1;
}
// A zero texel block: zeros, except ASTC, whose all-zero block is a reserved mode (the error colour): a void-extent block of colour (0, 0, 0, 0) there
static int is_astc(VkFormat v) { return v >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && v <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; }
static const uint8_t k_astc_zero[16] = {0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
static void astc_zero_blocks(uint8_t *p, size_t n) {   // the zeros of unmapped pages, as zero texels
    for (size_t i = 0; i + 16 <= n; i += 16) { uint64_t a, b; memcpy(&a, p + i, 8); memcpy(&b, p + i + 8, 8); if (!(a | b)) memcpy(p + i, k_astc_zero, 16); }
}
// Zero texels without staging: a buffer of zero blocks (first half) and of ASTC's zero blocks (second half), the source of every unmapped part of a sparse texture
enum { ZHALF = 2 << 20 };
static VkBuffer g_zbuf; static int g_zstate;   // 0 not tried, 1 made, -1 failed
static int zero_src(void) {
    if (g_zstate) return g_zstate > 0;
    g_zstate = -1;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 2 * ZHALF, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT};
    VkBuffer b; VkDeviceMemory m = VK_NULL_HANDLE; void *p = NULL;
    if (vk.vkCreateBuffer(V.dev, &bi, NULL, &b)) return 0;
    VkMemoryRequirements mr; vk.vkGetBufferMemoryRequirements(V.dev, b, &mr);
    uint32_t t = vka_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = t};
    if (t == UINT32_MAX || vk.vkAllocateMemory(V.dev, &ai, NULL, &m) || vk.vkBindBufferMemory(V.dev, b, m, 0) || vk.vkMapMemory(V.dev, m, 0, VK_WHOLE_SIZE, 0, &p)) {
        VRED("zero texel buffer not made: unmapped parts of sparse textures go through scratch memory"); if (m) vk.vkFreeMemory(V.dev, m, NULL); vk.vkDestroyBuffer(V.dev, b, NULL); return 0; }
    memset(p, 0, ZHALF);
    for (size_t k = 0; k < ZHALF; k += 16) memcpy((uint8_t *)p + ZHALF + k, k_astc_zero, 16);
    vk.vkUnmapMemory(V.dev, m);
    g_zbuf = b; g_zstate = 1; return 1;
}
// Block rows [y0, y1) of image level i, layer ly (w x h texels, bx blocks a row, pitch bytes a row) as zero texels from the zero buffer: 0 = not done (no buffer, or a row wider
// than it), the caller zeroes them through scratch memory
static int zero_rows(vka_img *im, VkCommandBuffer cb, uint32_t i, uint32_t ly, uint32_t w, uint32_t h, uint32_t bx, size_t pitch, uint32_t y0, uint32_t y1) {
    const vka_fmt *f = &im->f; uint32_t per = (uint32_t)(ZHALF / pitch);
    if (!per || !zero_src()) return 0;
    VkBufferImageCopy rg[64]; uint32_t n = 0;
    for (uint32_t y = y0; y < y1; y += per) {
        uint32_t ye = y + per < y1 ? y + per : y1, py0 = y * f->bh, py1 = ye * f->bh > h ? h : ye * f->bh;
        rg[n++] = (VkBufferImageCopy){is_astc(f->vk) ? ZHALF : 0, bx * f->bw, 0, {im->aspect, i, ly, 1}, {0, (int32_t)py0, 0}, {w, py1 - py0, 1}};
        if (n == 64) { vk.vkCmdCopyBufferToImage(cb, g_zbuf, im->img, VK_IMAGE_LAYOUT_GENERAL, n, rg); n = 0; }
    }
    if (n) vk.vkCmdCopyBufferToImage(cb, g_zbuf, im->img, VK_IMAGE_LAYOUT_GENERAL, n, rg);
    return 1;
}
// Tiles across image level i of a sparse residency image (its tile width in texels from the format's granularity)
static uint32_t rs_ntx(const vka_img *im, uint32_t i) { uint32_t w = im->w >> i ? im->w >> i : 1; return (w + im->rs_tw - 1) / im->rs_tw; }
// Image level i is bound tile by tile, and its bands are its rows of 64 KB tiles (one band = one tile high, tiles across x 64 KB): 1. A tiled level whose bands are not that
// shape (never seen) has all its tiles bound and is uploaded as a dense one
static int res_level(vka_img *im, uint32_t i, uint32_t band_rows, uint64_t band_bytes) {
    if (!im->rs || i >= vks_tiled(im->rs)) return 0;
    uint32_t ntx = rs_ntx(im, i), h = im->h >> i ? im->h >> i : 1, nty = (h + im->rs_th - 1) / im->rs_th;
    if (band_rows * im->f.bh == im->rs_th && band_bytes == (uint64_t)ntx * 0x10000) return 1;
    VRED_ONCE("sparse texture serial %llu level %u: bands of %u block rows / %llu bytes are not rows of %ux%u tiles: the level is bound whole", (unsigned long long)im->serial, i + im->base,
              band_rows, (unsigned long long)band_bytes, im->rs_tw, im->rs_th);
    for (uint32_t ly = 0; ly < im->layers; ly++) for (uint32_t ty = 0; ty < nty; ty++) for (uint32_t tx = 0; tx < ntx; tx++) vks_tile(im->rs, i, ly, tx, ty, 1);
    return 0;
}
// The bound tiles of bands [k, k1) of image level i, layer ly from staging s (rows from block row y0 at pitch): one copy region per run of bound tiles in a band
static void copy_bound(vka_img *im, VkCommandBuffer cb, const vka_span *s, uint32_t i, uint32_t ly, uint32_t w, uint32_t h, size_t pitch, uint32_t band_rows, uint32_t k, uint32_t k1, uint32_t y0) {
    const vka_fmt *f = &im->f; uint32_t ntx = rs_ntx(im, i), bx = (w + f->bw - 1) / f->bw;
    VkBufferImageCopy rg[32]; uint32_t n = 0;
    for (uint32_t kk = k; kk < k1; kk++) {
        uint32_t r0 = kk * band_rows, py0 = r0 * f->bh; if (py0 >= h) break;
        uint32_t py1 = (r0 + band_rows) * f->bh > h ? h : (r0 + band_rows) * f->bh;
        for (uint32_t tx = 0, t1; tx < ntx; tx = t1 + 1) {
            for (t1 = tx; t1 < ntx && vks_tile_bound(im->rs, i, ly, t1, kk); t1++) {}
            if (t1 == tx) continue;
            uint32_t px0 = tx * im->rs_tw, px1 = t1 * im->rs_tw > w ? w : t1 * im->rs_tw;
            rg[n++] = (VkBufferImageCopy){s->off + (VkDeviceSize)(r0 - y0) * pitch + (VkDeviceSize)(px0 / f->bw) * f->bpb, bx * f->bw, (py1 - py0 + f->bh - 1) / f->bh * f->bh,
                                          {im->aspect, i, ly, 1}, {(int32_t)px0, (int32_t)py0, 0}, {px1 - px0, py1 - py0, 1}};
            if (n == 32) { vk.vkCmdCopyBufferToImage(cb, s->buf, im->img, VK_IMAGE_LAYOUT_GENERAL, n, rg); n = 0; }
        }
    }
    if (n) vk.vkCmdCopyBufferToImage(cb, s->buf, im->img, VK_IMAGE_LAYOUT_GENERAL, n, rg);
}
// Levels and layers of a sparse texture's image whose pages or mappings changed since (gen, stamp), or all of them, gathered from the page table and deswizzled by bands:
// a band is one row of GOB blocks (8 << block height block rows), which is one range of the storage. Bands with nothing mapped are zero texels from the zero buffer, bands that
// did not change (all = 0) are left alone, and runs of mapped bands go through scratch memory of at most 16 MiB a run, gathered from the page table (unmapped pages of a
// band as zero texels): the staging is about what is mapped, never a whole level (Mario Kart World 1.8.0's 6144x6144 RGBA8 array levels, 144 MiB each, have about half of
// their pages mapped at boot, and its 8192x8192 x68 arrays' levels a few MiB in all). 3D textures go by whole levels, as before. Each run is a point where the upload
// batch may go to the queue (vka_up_point)
static void upload_sparse_parts(vka_img *im, VkCommandBuffer cb, uint64_t gen, uint64_t stamp, int all, int bind_only) {
    const nvn_texdesc *d = &im->d; const vka_fmt *f = &im->f; uint64_t mem = (uint64_t)(uintptr_t)im->mem;
    uint8_t *part = NULL, *bst = NULL; uint64_t cap = 0, up = 0, mapped = 0, zeroed = 0; unsigned nsub = 0, nzero = 0; uint32_t bst_cap = 0;
    int is3d = d->target == NVN_TEXTURE_TARGET_3D;
    for (uint32_t i = 0; i < im->levels && cb; i++) {
        uint32_t lv = i + im->base, w = im->w >> i ? im->w >> i : 1, h = im->h >> i ? im->h >> i : 1;
        uint32_t bx = (w + f->bw - 1) / f->bw, by = (h + f->bh - 1) / f->bh;
        size_t pitch = (size_t)bx * f->bpb;
        // Bands: block rows and storage bytes of each (a 3D level: one band of all of it)
        uint32_t band_rows = by; uint64_t band_bytes = 0;
        if (is3d || nvn_deswizzle_block_rows(d, lv) != (int)by || nvn_deswizzle_bands(d, lv, &band_rows, &band_bytes)) band_rows = by, band_bytes = 0;
        for (uint32_t ly = 0; ly < im->layers && cb; ly++) {
            uint64_t off, len; sub_range(d, &im->L, lv, ly, &off, &len);
            if (!len || (!all && !nvn_sparse_since(im->vpool, mem + off, len, gen, stamp))) continue;
            uint32_t nb = band_bytes ? (by + band_rows - 1) / band_rows : 1;
            // Band k: 0 = left alone, 1 = zero texels, 2 = from the page table
            #define BAND_OFF(k) ((uint64_t)(k) * band_bytes)
            #define BAND_LEN(k) (band_bytes ? (BAND_OFF(k) < len ? (len - BAND_OFF(k) < band_bytes ? len - BAND_OFF(k) : band_bytes) : 0) : len)
            // (a band is redone only when it changed: unmapped since the last upload counts, so a run never mixes changed and unchanged bands)
            #define BAND_CHANGED(k) (all || nvn_sparse_since(im->vpool, mem + off + BAND_OFF(k), BAND_LEN(k), gen, stamp))
            #define BAND_STATE(k) (!BAND_LEN(k) ? all : !BAND_CHANGED(k) ? 0 : nvn_sparse_any(im->vpool, mem + off + BAND_OFF(k), BAND_LEN(k)) ? 2 : 1)
            int any = 0;
            // A sparse residency image's tiled level (vka_sparse.c): a band is one row of 64 KB tiles. Each changed tile is bound when a page of it is mapped and unbound when
            // none is; a band with a bound tile is gathered (2) and only its bound tiles copied, the rest is left alone (0: unbound tiles read 0, no zero texels)
            int rt = res_level(im, i, band_rows, band_bytes);
            if (bind_only && !rt) continue;   // (a GPU-written image's levels past the tiled ones are bound whole, vks_new; their content is the GPU's)
            if (bst_cap < nb) { free(bst); bst_cap = nb; if (!(bst = malloc(bst_cap))) { bst_cap = 0; VRED("sparse texture serial %llu: band states not allocated", (unsigned long long)im->serial); goto out; } }
            for (uint32_t k = 0; k < nb; k++) {
                if (!rt) { bst[k] = (uint8_t)BAND_STATE(k); continue; }
                int b2 = 0;
                if (BAND_CHANGED(k)) for (uint32_t tx = 0; tx < rs_ntx(im, i); tx++) {
                    uint64_t po = off + BAND_OFF(k) + (uint64_t)tx * 0x10000;
                    b2 |= vks_tile(im->rs, i, ly, tx, k, po < off + len && nvn_sparse_any(im->vpool, mem + po, 0x10000));
                }
                bst[k] = b2 ? 2 : 0;
            }
            if (bind_only) continue;
            for (uint32_t k = 0, k1; k < nb && cb; k = k1) {
                int st = bst[k]; k1 = k + 1;
                size_t maxrun = 16u << 20; if (maxrun < (size_t)band_rows * pitch) maxrun = (size_t)band_rows * pitch;
                while (k1 < nb && (size_t)(k1 + 1 - k) * band_rows * pitch <= maxrun && bst[k1] == st) k1++;
                if (!band_bytes) k1 = nb;
                uint32_t y0 = k * band_rows, y1 = k1 * band_rows < by ? k1 * band_rows : by;
                if (!st) continue;
                any = 1;
                size_t sz = (size_t)(y1 - y0) * pitch * (is3d ? (im->depth >> i ? im->depth >> i : 1) : 1);
                if (st == 1 && !is3d && zero_rows(im, cb, i, ly, w, h, bx, pitch, y0, y1)) { zeroed += sz; continue; }
                vka_span s; if (!vka_scratch(sz, 16, &s)) { VRED("sparse texture serial %llu: no scratch memory for level %u layer %u (%zu bytes)", (unsigned long long)im->serial, lv, ly, sz); goto out; }
                if (st == 1) { if (is_astc(f->vk)) for (size_t q = 0; q + 16 <= sz; q += 16) memcpy(s.ptr + q, k_astc_zero, 16); else memset(s.ptr, 0, sz); zeroed += sz; }
                else {
                    uint64_t so = off + BAND_OFF(k), sl = band_bytes ? BAND_OFF(k1) - BAND_OFF(k) : len;
                    if (so + sl > off + len) sl = off + len - so;
                    if (sl > cap) { free(part); cap = sl; if (!(part = malloc(cap))) { VRED("sparse texture serial %llu: copy of %llu bytes not allocated", (unsigned long long)im->serial, (unsigned long long)sl); goto out; } }
                    mapped += sl - nvn_sparse_copy(im->vpool, part, mem + so, sl);
                    if (is_astc(f->vk)) astc_zero_blocks(part, sl);
                    // the deswizzler reads level lv of layer ly at its offset in the whole storage, and writes row y at y * pitch: part holds just this run's bytes
                    if (nvn_deswizzle_rows(d, lv, ly, (const uint8_t *)((uintptr_t)part - (uintptr_t)so), s.ptr - (size_t)y0 * pitch, pitch, y0, y1)) {
                        VRED_ONCE("texture serial %llu format %#x target %d: no CPU deswizzle", (unsigned long long)im->serial, d->format, d->target); goto out; }
                    up += sz;
                }
                uint32_t py0 = y0 * f->bh, py1 = y1 * f->bh > h ? h : y1 * f->bh;
                uint32_t dz = is3d ? (im->depth >> i ? im->depth >> i : 1) : 1;
                VkBufferImageCopy c = {s.off, bx * f->bw, (y1 - y0) * f->bh, {im->aspect, i, ly, 1}, {0, (int32_t)py0, 0}, {w, py1 - py0, dz}};
                if (rt) copy_bound(im, cb, &s, i, ly, w, h, pitch, band_rows, k, k1, y0);   // only the bound tiles (an unbound one takes no write)
                else vk.vkCmdCopyBufferToImage(cb, s.buf, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
                cb = vka_up_point(cb);
            }
            #undef BAND_STATE
            #undef BAND_CHANGED
            #undef BAND_LEN
            #undef BAND_OFF
            nsub += any; if (any && !nvn_sparse_any(im->vpool, mem + off, len)) nzero++;
        }
    }
out:
    free(part); free(bst);
    static _Atomic int nl; if (nsub && atomic_fetch_add(&nl, 1) < 64)
        VLOG("sparse texture serial %llu %ux%u layers %u levels %u+%u of %u format %#x: %u levels x layers uploaded (%u with nothing mapped), %.1f MiB through staging, %.1f MiB of zero texels, %.1f MiB of pages mapped behind them%s",
             (unsigned long long)im->serial, im->d.width, im->d.height, im->layers, im->base, im->levels, im->base + im->levels, d->format, nsub, nzero, up / 1048576.0, zeroed / 1048576.0, mapped / 1048576.0, all ? "" : " (changed since the last upload)");
}
static void upload_sparse(vka_img *im, VkCommandBuffer cb, int all) {   // generation and stamp, protect the physical pages behind it, then read (nvn_mtl.m pkg_upload)
    uint64_t gen = im->vgen, stamp = im->stamp;
    im->vgen = nvn_sparse_gen(); im->stamp = nvn_dirty_now(); nvn_sparse_protect(im->vpool, (uint64_t)(uintptr_t)im->mem, im->L.size);
    upload_sparse_parts(im, cb, gen, stamp, all, 0);
}
// A GPU-written sparse texture's residency image (sparse_gpu): its tiles bound where pages are mapped and unbound where none is, nothing read (all: every tile; else those whose
// mappings changed since). Open: a tile bound again starts with what its memory slot held (the Switch keeps the physical page's content), not cleared
static void sparse_bind(vka_img *im, VkCommandBuffer cb, int all) {
    uint64_t gen = im->vgen, stamp = im->stamp;
    im->vgen = nvn_sparse_gen(); im->stamp = nvn_dirty_now();
    upload_sparse_parts(im, cb, gen, stamp, all, 1);
    static _Atomic int n; if (atomic_fetch_add(&n, 1) < 24)
        VLOG("sparse GPU-written texture serial %llu %ux%u layers %u levels %u format %#x: tiles bound %s, %.1f MiB bound of %.1f MiB of address space", (unsigned long long)im->serial, im->w, im->h, im->layers,
             im->levels, im->d.format, all ? "at its first use" : "again (its mappings changed)", vks_img_bytes(im->rs) / 1048576.0, im->va / 1048576.0);
}
static void dirty(int **list, int *n, int *cap, uint8_t **flag, int nslots, int id);
static int *g_tdirty, g_ntd, g_tdcap; static uint8_t *g_tdflag;
typedef struct { tex_t t; view_t v; uint8_t has_v, img; uint64_t pgen; } treg_ent;   // pgen: the texture's pool's Initialize serial at registration (nvn_pool_live)
static treg_ent *g_treg; static int g_ntreg;   // indexed by host texture ID
// The image made again from texture level base (caller holds g_tmu): the old image, its memory and views go when the GPU is done with them, and the texture IDs that name it
// are written again at this submission (vka_tex_prepare); then everything is uploaded
static uint64_t g_rebase;
static unsigned redirty_ids(vka_img *im) {   // the texture IDs registered with im's texture, written again at this submission (caller holds g_tmu)
    unsigned nid = 0;
    for (int id = 1; id < g_ntreg; id++) { if (!g_treg[id].t.serial) continue; smap_ent *e = map_get(g_treg[id].t.serial); if (e && e->im == im) { dirty(&g_tdirty, &g_ntd, &g_tdcap, &g_tdflag, 1 << 20, id); nid++; } }
    return nid;
}
static int img_rebase(vka_img *im, uint32_t base, VkCommandBuffer cb) {
    tex_t t = {0}; t.serial = im->serial; t.d = im->d; t.L = im->L; t.mem = (uint8_t *)im->mem; t.pool = (pool_t *)im->vpool;   // (the pool: a sparse residency image again)
    vka_img *n = img_create(&t, &im->f, base);
    if (!n) return 0;
    // Going finer but the address space (which still counts this image's own) did not take it: the image stays as it is, rather than one at a coarser level or dense
    // (and a sparse image is not swapped for a dense one from a finer level: a large memory jump when only the residency image was refused)
    if ((base < im->base && (n->base >= im->base || (im->rs && !n->rs))) || (n->base == im->base && n->levels == im->levels && !!n->rs == !!im->rs)) {
        int floored = base < im->base;
        uint64_t need = n->bneed > im->va ? n->bneed : im->va + 1;   // (a start finer than im->base asks for more than im itself: n's ask may be of a coarser level)
        if (n->dcap) im->dcap = 1;   // (its dense images are capped: sparse_refresh does not make this one again)
        g_img_bytes -= n->bytes; img_destroy(n);
        if (floored) { im->bfloor = im->base; im->bneed = need; }   // (tried again once what the finer level asked for is free)
        return 0;
    }
    vka_img *old = calloc(1, sizeof *old);
    if (!old) { g_img_bytes -= n->bytes; img_destroy(n); return 0; }
    old->img = im->img; old->m = im->m; old->views = im->views; old->rs = im->rs; old->va = im->va;
    uint32_t ob = im->base, ol = im->levels; uint64_t obytes = im->bytes;
    g_img_bytes -= im->bytes;
    im->img = n->img; im->m = n->m; im->views = NULL; im->usage = n->usage; im->w = n->w; im->h = n->h; im->levels = n->levels; im->base = n->base; im->bytes = n->bytes;
    im->rs = n->rs; im->va = n->va; im->rs_tw = n->rs_tw; im->rs_th = n->rs_th; im->bfloor = n->bfloor; im->bneed = n->bneed; im->dcap = n->dcap;   // (old's address space, still counted, goes back
                                                                                                                           // at its img_destroy: a finer start is tried again once enough is free)
    free(n);
    vka_after(vka_fence_last(), img_destroy, old);
    vka_img_barrier(cb, im->img, im->aspect, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    unsigned nid = redirty_ids(im);
    if (g_rebase++ < 32) VLOG("sparse texture serial %llu %ux%u format %#x: levels %u+%u → %u+%u (%.1f → %.1f MiB of image memory, %u texture IDs written again)", (unsigned long long)im->serial, im->d.width, im->d.height,
                              im->d.format, ob, ol, im->base, im->levels, obytes / 1048576.0, im->bytes / 1048576.0, nid);
    return 1;
}
// A sparse texture whose pages or mappings changed: its finest mapped level moved finer, or two or more levels coarser, → made again; else the changed parts uploaded
static void sparse_refresh(vka_img *im, VkCommandBuffer cb) {
    uint32_t vmin, b = sparse_base(im->vpool, im->mem, &im->d, &im->L, &vmin);
    if (b < im->bfloor && vks_va_free() < im->bneed) b = im->bfloor;   // (levels left out for the sparse address space; tried again once what the finer level asked for is free: a
                                                                          // rebase counts the old image's address space until it goes, vka_after, and a finer start must fit beside the current image)
    if (im->dcap && !im->rs) { uint32_t bd = dense_floor(&im->d, &im->L, b); if (bd > b) b = bd; }   // (a dense image under DM_VK_SPARSE_DENSE_MB: no rebase to a start the cap moves back)
    if (vmin != im->vmin) {   // the views start elsewhere: the texture IDs that name it are written again
        if (g_rebase < 32) VLOG("sparse texture serial %llu %ux%u format %#x: sampled from level %u (was %u)", (unsigned long long)im->serial, im->d.width, im->d.height, im->d.format, vmin, im->vmin);
        im->vmin = vmin; redirty_ids(im);
    }
    // A new image (rebase): its barrier and its whole upload in the upload batch when cb is the replay's (nothing recorded before reads the new image)
    if (im->d.target != NVN_TEXTURE_TARGET_3D && (b < im->base || b >= im->base + 2)) {
        VkCommandBuffer ucb = cb && cb == vka_rp_cb() && vka_up_on() ? vka_up_begin() : VK_NULL_HANDLE;
        int made = img_rebase(im, b, ucb ? ucb : cb);
        if (made) upload_sparse(im, ucb ? ucb : cb, 1);
        if (ucb) vka_up_end();
        if (made) return;
    }
    upload_sparse(im, cb, 0);
}
// ---------- Large images (diagnostic, DM_VK_BIGIMG_MB) ----------
// For each image of at least DM_VK_BIGIMG_MB (default 64; memrun-style runs set 16): its first GPU write (vka_img_written_at: frame and site), the draws that bound it for sampling
// (id_view), and the other textures found over its storage (a texture of the same storage and shape shares the image, same_shape; one of another shape or format gets an image
// of its own, so content drawn through one is not in the other). The ctl stats add the largest ones the GPU has not written (vka_tex_big_stats). Caller holds g_tmu
static uint64_t big_min(void) { double mb = VKA_ENVF("DM_VK_BIGIMG_MB", 64.0); return mb > 0 ? (uint64_t)(mb * 1048576.0) : UINT64_MAX; }
static int ranges_meet(const uint8_t *a, uint64_t an, const uint8_t *b, uint64_t bn) { return a && b && a < b + bn && b < a + an; }
static void ovl_add(vka_img *big, uint64_t serial) { if (big->novl < 4) big->ovl[big->novl] = serial; big->novl++; }
static void big_note(vka_img *im, const tex_t *t) {   // a new image of texture t
    uint64_t mn = big_min(); static _Atomic int nl;
    if (!t->mem || mn == UINT64_MAX) return;
    for (int i = 0; i < g_nbig; i++) { vka_img *b = g_big[i];
        if (b == im || !ranges_meet(b->mem, b->L.size, t->mem, t->L.size)) continue;
        ovl_add(b, t->serial);
        if (atomic_fetch_add(&nl, 1) < 32) VLOG("large image serial %llu (%ux%u format %#x, storage %p+%#llx): texture serial %llu %ux%ux%u layers %u format %#x flags %#x target %d over its storage (%p+%#llx), an image of its own%s",
            (unsigned long long)b->serial, b->w, b->h, b->d.format, (const void *)b->mem, (unsigned long long)b->L.size, (unsigned long long)t->serial, t->d.width, t->d.height, t->d.depth, t->L.layers,
            t->d.format, t->d.flags, t->d.target, (const void *)t->mem, (unsigned long long)t->L.size, im->m.z ? " (zero-shared)" : ""); }
    if (im->m.size < mn) return;
    for (int i = 0; i < g_nlive; i++) { vka_img *o = g_live[i];
        if (o == im || !ranges_meet(o->mem, o->L.size, t->mem, t->L.size)) continue;
        ovl_add(im, o->serial);
        if (atomic_fetch_add(&nl, 1) < 32) VLOG("large image serial %llu (%ux%u format %#x, storage %p+%#llx): made over the storage of texture serial %llu %ux%u format %#x flags %#x (%p+%#llx, %s)",
            (unsigned long long)im->serial, im->w, im->h, im->d.format, (const void *)t->mem, (unsigned long long)t->L.size, (unsigned long long)o->serial, o->w, o->h, o->d.format, o->d.flags,
            (const void *)o->mem, (unsigned long long)o->L.size, o->gpu_owned ? "GPU-written" : "not GPU-written"); }
    if (g_nbig < 64) g_big[g_nbig++] = im;
}
static void big_share(vka_img *im, const tex_t *t) {   // texture t shares im (same storage and shape)
    if (im->m.size < big_min() || t->serial == im->serial) return;
    ovl_add(im, t->serial);
    static _Atomic int n; if (atomic_fetch_add(&n, 1) < 16) VLOG("large image serial %llu (%ux%u format %#x): texture serial %llu (format %#x) of the same storage and shape shares it", (unsigned long long)im->serial,
                                                                  im->w, im->h, im->d.format, (unsigned long long)t->serial, t->d.format);
}
void vka_tex_big_stats(char *b, size_t n) {
    uint64_t mn = big_min(); if (n) b[0] = 0;
    if (mn == UINT64_MAX || !n) return;
    pthread_mutex_lock(&g_tmu);
    vka_img *top[8]; int nt = 0, all = 0;
    for (int i = 0; i < g_nlive; i++) { vka_img *im = g_live[i];
        if (im->gpu_owned || im->mode != ST_ZERO || im->from_mem || im->m.size < mn) continue;
        all++;
        int k = nt < 8 ? nt++ : 8; if (k == 8) { if (im->m.size <= top[7]->m.size) continue; k = 7; }
        while (k > 0 && top[k - 1]->m.size < im->m.size) { top[k] = top[k - 1]; k--; }
        top[k] = im; }
    size_t l = (size_t)snprintf(b, n, "large images the GPU has not written: %d", all);
    for (int i = 0; i < nt && l < n; i++) { vka_img *im = top[i];
        l += (size_t)snprintf(b + l, n - l, "; serial %llu %ux%u format %#x %.0f MiB%s, sampled by %u draws, %d refs, %u other textures over its storage", (unsigned long long)im->serial, im->w, im->h, im->d.format,
                              im->m.size / 1048576.0, im->m.z ? " zero-shared" : "", im->nsamp, im->refs, im->novl);
        for (uint32_t k = 0; k < im->novl && k < 4 && l < n; k++) { smap_ent *e = map_get(im->ovl[k]);
            l += (size_t)snprintf(b + l, n - l, "%s%llu%s", k ? " " : " (", (unsigned long long)im->ovl[k], !e || !e->im ? "?" : e->im == im ? "=" : e->im->gpu_owned ? "W" : "-"); }
        if (im->novl && l < n) l += (size_t)snprintf(b + l, n - l, ")"); }
    pthread_mutex_unlock(&g_tmu);
}

// The large images whose storage lies in placed pool memory (ctl stats): a pool placed as Vulkan memory is committed when it is allocated (vka_mem.c), and an image the
// GPU writes (or one filled once) never reads its storage again, so such storage is held twice: once as the pool's placed memory, once as the image. Whether a pool is placed
// depends on what the game gave the GPU an address in (vka_mem.c Pools: a buffer, the pool's address, a texture's address, a linear texture); the big render target pools
// of Ocarina of Time (the 12288² DEPTH16 shadow atlas, 288 MiB of storage) are only GPU-written images, and this line says whether their storage is placed as well
void vka_tex_placed_stats(char *b, size_t n) {
    struct { uint64_t serial, lo, size, bytes; uint32_t w, h; int fmt, gw; } e[64]; int ne = 0;
    if (!n) return;
    b[0] = 0;
    pthread_mutex_lock(&g_tmu);
    for (int i = 0; i < g_nbig && ne < 64; i++) { const vka_img *im = g_big[i]; if (!im->mem || im->vpool) continue;
        e[ne].serial = im->serial; e[ne].lo = (uint64_t)(uintptr_t)im->mem; e[ne].size = im->L.size; e[ne].bytes = im->bytes; e[ne].w = im->w; e[ne].h = im->h; e[ne].fmt = im->d.format; e[ne].gw = im->gpu_owned; ne++; }
    pthread_mutex_unlock(&g_tmu);
    uint64_t all = 0, allp = 0; int np = 0; size_t l = 0; char tb[700]; tb[0] = 0;
    for (int i = 0; i < ne; i++) {
        uint64_t a = e[i].lo, end = a + e[i].size, got = 0;   // (placed segments start on the placement alignment: a miss steps 64 KiB, so a part may be missed by less than that)
        while (a < end) { uint64_t r = vka_addr_run_own(a, end - a); if (r) { got += r; a += r; } else a = (a | 0xFFFFull) + 1; }
        if (!got) continue;
        np++; all += e[i].bytes; allp += got;
        if (l < sizeof tb) l += (size_t)snprintf(tb + l, sizeof tb - l, "; serial %llu %ux%u format %#x %s: image %.1f MiB, storage %.1f MiB of which placed %.1f MiB", (unsigned long long)e[i].serial,
                                                 e[i].w, e[i].h, e[i].fmt, e[i].gw ? "GPU-written" : "not GPU-written", e[i].bytes / 1048576.0, e[i].size / 1048576.0, got / 1048576.0);
    }
    snprintf(b, n, "large images whose storage is placed pool memory too: %d of %d (images %.0f MiB, placed storage %.0f MiB)%s", np, ne, all / 1048576.0, allp / 1048576.0, tb);
}

// ---------- Zero-shared images ----------
// A large image whose content is the GPU's (ST_ZERO: render targets, copy destinations, shader images) holds zeros until the GPU first writes it, and many engines make such
// targets long before they draw into them, or never: Mario Kart World 1.8.0 keeps four 16384² R8 terrain targets (341 MiB each, flags SPARSE | COMPRESSIBLE, no sparse
// residency for 8-bit formats on the Adreno 840) and about 300 MB of other cleared images that its scene never writes. On the Switch such a texture costs only its pool
// storage. So a new image of at least DM_VK_ZSHARE_MB (default 4; NVN storage bytes) that is dense, single-sampled, not 3D, not block-compressed, not linear and not a display
// texture, made with ST_ZERO content, is created with VK_IMAGE_CREATE_ALIAS_BIT and bound at offset 0 of one zeroed allocation per identical create info (vka_imgmem.c
// vka_imem_bind_zero; Vulkan's "Memory Aliasing": images of identical create info with ALIAS_BIT bound at the same offset read the memory alike). Each image's first use clears
// it to zero behind a full barrier as before (zeros onto zeros, which also sets that image's layout and the driver's compression metadata). Before its first write (vka_tex_own:
// an attachment at render-pass begin, a clear, a copy or blit destination, a buffer-to-image copy, a shader image, a CPU re-upload) it gets memory of its own (img_own): a new
// image of the same texture, cleared to zero in the upload batch (nothing recorded before reads it), swapped in place as img_storage_upgrade does (every serial and view that
// shares it follows; the recording memo and the texture IDs that name it are made again), the old one destroyed after the submission, or once its holders let it go.
// A write that comes without it is logged [RED] (vka_img_written_at). DM_VK_ZSHARE=0 (live: images made from then on): every image has memory of its own, as before.
// Adopt (DM_VK_ZSHARE_ADOPT, live, 0; Ocarina of Time 1): an image alone in its zero-shared allocation takes that allocation as its own (vka_imem_zadopt) instead of a new
// image: same VkImage, views and layout, its zeros its content (nothing wrote it), no clear, no texture IDs written again. A new image held both allocations until the
// submission ended: Ocarina of Time's 12288² DEPTH16 target (alone in its 299 MiB allocation) took 598 MiB at its first write, during the load that killed the game (fb3).
// A clear to zero (DM_VK_ZSHARE_ZCLEAR, live, 0; Ocarina of Time 1): a clear whose value is all-zero bits writes zeros onto the zeros a zero-shared image holds, so it is left
// out and the image keeps the shared memory (vka_tex_zclear; the next real write gives it memory of its own, cleared to zero, which is what the clear would have left)
static uint64_t g_zown, g_zown_b, g_zadopt, g_zadopt_b, g_zclear;   // (under g_tmu) zero-shared images given memory of their own, and its bytes; of them adopted; clears to zero left out
int vka_tex_zclear(vka_img *im) {
    if (!im || !VKA_ENV("DM_VK_ZSHARE_ZCLEAR", 0)) return 0;
    pthread_mutex_lock(&g_tmu);
    int r = im->m.z != 0;
    // (gpu_owned as vka_img_written would set it: the storage is the GPU's content from now on, so a CPU write, a texture over the storage or eviction does not take stale storage for it)
    if (r) { im->gpu_owned = 1; im->rl_wf = (uint32_t)vka_frame(); vka_rtw(im, VKA_RTW_ZKEEP); if (g_zclear++ < 8) VLOG("zero-shared image serial %llu %ux%u format %#x: a clear to zero left out, it keeps the shared zeroed memory (%.1f MiB; %llu so far)",
                                                                  (unsigned long long)im->serial, im->w, im->h, im->d.format, im->m.size / 1048576.0, (unsigned long long)g_zclear); }
    pthread_mutex_unlock(&g_tmu);
    return r;
}
static int img_own(vka_img *im, VkCommandBuffer cb) {   // caller holds g_tmu; 1 = im has memory of its own
    if (!im->m.z) return 1;
    if (VKA_ENV("DM_VK_ZSHARE_ADOPT", 0) && vka_imem_zadopt(&im->m)) {   // (alone in its allocation: it becomes the image's own)
        im->bytes = im->m.size; g_img_bytes += im->bytes; g_zown_b += im->bytes; g_zadopt++; g_zadopt_b += im->bytes;
        if (g_zown++ < 16) VLOG("zero-shared image serial %llu %ux%u layers %u format %#x: memory of its own at its first write, the zeroed allocation it was alone in adopted (%.1f MiB, frame %llu; %llu adopted)",
                                (unsigned long long)im->serial, im->w, im->h, im->layers, im->d.format, im->bytes / 1048576.0, (unsigned long long)vka_frame(), (unsigned long long)g_zadopt);
        if (im->rtw) rtw_log(im, "memory of its own (adopted)");
        rl_note(im, "own (the zero-shared allocation adopted)");
        return 1;
    }
    tex_t t = {0}; t.serial = im->serial; t.mem = (uint8_t *)im->mem; t.d = im->d; t.L = im->L;
    if (im->usage & VK_IMAGE_USAGE_STORAGE_BIT) t.d.flags |= NVN_TEXTURE_FLAGS_IMAGE;   // (made with storage use, or upgraded to it)
    VkCommandBuffer ucb = !vka_scr_up && cb && cb == vka_rp_cb() && vka_up_on() ? vka_up_begin() : VK_NULL_HANDLE, rc = ucb ? ucb : cb;
    vka_img *n = rc ? img_create(&t, &im->f, im->base) : NULL;   // (g_zwant is 0: memory of its own)
    if (!n) { if (ucb) vka_up_end();
        static _Atomic int k; if (atomic_fetch_add(&k, 1) < 8) VRED("zero-shared image serial %llu %ux%u format %#x: no memory of its own (%.0f MiB%s): its writes go to the memory it shares", (unsigned long long)im->serial,
                                                                      im->w, im->h, im->d.format, im->m.size / 1048576.0, rc ? "" : ", no command buffer");
        return 0; }
    vka_img_barrier(rc, n->img, n->aspect, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    VkImageSubresourceRange rg = {n->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    if (n->f.kind >= 3) { VkClearDepthStencilValue z = {0, 0}; vk.vkCmdClearDepthStencilImage(rc, n->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rg); }
    else { VkClearColorValue z = {{0}}; vk.vkCmdClearColorImage(rc, n->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rg); }
    vka_barrier(rc);
    if (ucb) vka_up_end();
    n->inited = 1;
    VkImage oi = im->img; vka_imem om = im->m; void *ov = im->views;
    im->img = n->img; im->m = n->m; im->views = NULL; im->usage = n->usage; im->bytes = n->bytes; im->alt = n->alt; im->any_fmt = n->any_fmt; im->nx = n->nx; memcpy(im->xfmt, n->xfmt, sizeof im->xfmt);
    n->img = oi; n->m = om; n->views = ov; n->bytes = 0;
    if (im->holds) { n->zold = im->zold; im->zold = n; }   // (a holder may read im->img unlocked: the old image stays until it lets go, vka_tex_drop)
    else vka_after(vka_fence_last() + 1, img_destroy, n);   // (after the submission being recorded, which may hold commands that read it)
    atomic_fetch_add_explicit(&g_tgen, 1, memory_order_release);
    unsigned nid = redirty_ids(im);
    vka_imem_zowned();
    g_zown_b += im->bytes;
    if (g_zown++ < 16) VLOG("zero-shared image serial %llu %ux%u layers %u format %#x: memory of its own at its first write (%.1f MiB, frame %llu, %u texture IDs written again)", (unsigned long long)im->serial,
                            im->w, im->h, im->layers, im->d.format, im->bytes / 1048576.0, (unsigned long long)vka_frame(), nid);
    if (im->rtw) rtw_log(im, "memory of its own (a new image; the shared allocation goes after the submission)");
    rl_note(im, "own (a new image)");
    return 1;
}
static void md_whole(vka_img *im, VkCommandBuffer cb);
void vka_tex_whole(vka_img *im, VkCommandBuffer cb) {   // a copy reads a dropped image ("Mip drop"): whole again first, its finer levels from its storage
    if (!im || !im->mdrop) return;
    if (!vka_up_on() && cb && cb == vka_rp_cb() && vka_rp_is_open()) vka_rp_end();
    nvn_pool_use(1); pthread_mutex_lock(&g_tmu); if (im->mdrop) md_whole(im, cb); pthread_mutex_unlock(&g_tmu); nvn_pool_use(0);
}
void vka_tex_own(vka_img *im, VkCommandBuffer cb) {
    if (!im || (!im->m.z && !im->mdrop)) return;   // (m.z and mdrop change on the recording thread only)
    if (!vka_up_on() && cb && cb == vka_rp_cb() && vka_rp_is_open()) vka_rp_end();   // (no upload batch: the clear or upload goes into cb, outside a pass)
    int md = im->mdrop != 0;
    if (md) nvn_pool_use(1);   // (md_whole reads the storage; taken before g_tmu, as write_tex does)
    pthread_mutex_lock(&g_tmu); if (im->mdrop) md_whole(im, cb); img_own(im, cb); pthread_mutex_unlock(&g_tmu);
    if (md) nvn_pool_use(0);
}

// The content a new image of t starts with (storage_mode), decided before the image is made (vka_tex_get: a zero-shared image is made only for ST_ZERO)
static int first_mode(const tex_t *t, const vka_fmt *f, int cpu) {
    int samples = t->d.samples > 1 ? t->d.samples : 1, vpool = t->pool && pool_sparse(t->pool);
    uint64_t seen; uint8_t *c = samples == 1 && t->mem && !vpool ? gw_content(t->mem, t->L.size, &seen) : NULL;   // the content the mode's "holds data" checks look at
    int mode = samples == 1 ? storage_mode(t, c ? c : t->mem, f, cpu, g_first_sampled) : ST_ZERO;
    free(c);
    return mode;
}
static void img_first_use(vka_img *im, const tex_t *t, VkCommandBuffer cb, int mode) {
    vka_img_barrier(cb, im->img, im->aspect, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    im->inited = 1;
    if (mode != ST_ZERO) im->from_mem = 1;
    im->mode = (uint8_t)mode;
    if (im->rs && mode == ST_ZERO && sparse_gpu(t, &im->f)) { im->vbind = 1; sparse_bind(im, cb, 1); }   // (a GPU-written sparse texture: its tiles bound before the clear below)
    // (an upload into the upload batch may send it and go on in the next one: vka_up_cur is where it ended; NULL: none could be begun after a GPU hang)
    if (mode == ST_TRACK) { im->track = 1; im->span = nvn_dirty_span(t->mem, t->L.size); upload_tracked(im, t, cb); g_st_up[mode]++; if ((cb = vka_up_cur(cb))) vka_barrier(cb); return; }
    if (mode == ST_SPARSE) { im->vtrack = 1; upload_sparse(im, cb, 1); g_st_up[mode]++; if ((cb = vka_up_cur(cb))) vka_barrier(cb); return; }
    if (mode == ST_FILL) { upload_fill(im, t, cb); g_st_up[mode]++; if ((cb = vka_up_cur(cb))) vka_barrier(cb); return; }   // the copy before the draw or copy that reads it
    g_st_up[ST_ZERO]++;
    // (guest-aliased on an allocation other images hold: no clear, its content is the storage's, which the newest image written there owns: "Guest-aliased targets")
    if (im->m.a && !im->ra_fresh) { g_ra_noclr++; if (cb) vka_barrier(cb); return; }
    VkImageSubresourceRange rg = {im->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    if (im->f.kind >= 3) { VkClearDepthStencilValue z = {0, 0}; vk.vkCmdClearDepthStencilImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rg); }
    else if (!compressed(&im->f)) { VkClearColorValue z = {{0}}; vk.vkCmdClearColorImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rg); }
    else if (t->mem) { im->from_mem = 1; upload_fill(im, t, cb); cb = vka_up_cur(cb); }   // a compressed texture with no data yet: its storage (zeros) is a valid block pattern; Vulkan cannot clear it
    if (cb) vka_barrier(cb);
}

static void tfb_remade(uint64_t s);
static void ev_remade(uint64_t s) {
    tfb_remade(s);
    uint64_t *e = &g_evs[s & 4095]; if (*e == s) { *e = 0; g_ev_remade++; }
    uint64_t *d = &g_evd[s & 4095]; if (*d == s) { *d = 0; g_evd_remade++; g_evpin[s & 4095] = s; }   // (a texture let go as dead was used again: the rule guessed wrong for it)
}
// ---------- Colour textures over a depth image ----------
// A single-channel colour texture over the storage of a depth texture of the same shape whose depth is stored the same way (R32_SFLOAT over D32_SFLOAT, R16_UNORM over
// D16_UNORM: one texel size, one encoding, one block-linear layout) reads the depth on the Switch. Ocarina of Time samples its 4096² DEPTH32F target through an R32F
// texture (q1-q4: serials 4132 / 4069 / 4075 / 4086 over 15722 / 15202 / 15409 / 14986; never written, sampled by 2-12 draws): same_shape cannot share a colour format with
// a depth image (no view reinterprets one as the other), so the R32F texture got a zero-shared image of its own, 64 MiB, and its draws sampled zeros.
// Vulkan samples a depth image through a view of its depth aspect in its own format, and the depth comes back in R (G, B 0 and A 1, as an R32F / R16 view reads), so
// such a texture shares the depth image at its first sampled lookup (vka_tex_get with g_look_sample: write_tex of a sampled ID, id_view without storage) and view_of
// makes its views in the depth format, depth aspect (all images are in GENERAL, which sampling a depth image allows). Only when the depth format is the NVN depth
// format itself (not D24 widened to D32_SFLOAT by fmt_of_slow), the depth image can be sampled with the filtering the colour format has, it is dense and its pool is
// live. A lookup that is no sample (an attachment, a clear, a copy or blit, a shader image or an image registration: a write as colour, which a depth image cannot take)
// ends the sharing for that serial: its map entry goes, the texture IDs registered with it are written again at this submission, and the lookup makes an image of its
// own as before (dalias_end). A texture looked up before the depth image exists keeps the image of its own. DM_VK_DEPTH_ALIAS=0 (live: lookups from then on): no sharing
static _Thread_local int g_look_sample;
static uint64_t g_dalias_n, g_dalias_end;   // sharings made, sharings ended by a write lookup (under g_tmu)
static VkFormat depth_twin(VkFormat c) { return c == VK_FORMAT_R32_SFLOAT ? VK_FORMAT_D32_SFLOAT : c == VK_FORMAT_R16_UNORM ? VK_FORMAT_D16_UNORM : VK_FORMAT_UNDEFINED; }
static int depth_alias_ok(vka_img *a, const tex_t *t, const vka_fmt *f) {   // caller holds g_tmu
    VkFormat dv = depth_twin(f->vk);
    if (!dv || a->f.vk != dv || a->f.kind != 3 || fmt_ent_of(a->d.format).vk != dv) return 0;
    if (a->mem != t->mem || a->d.target != t->d.target || a->d.width != t->d.width || a->d.height != t->d.height || a->d.depth != t->d.depth || a->d.levels != t->d.levels || a->d.samples != t->d.samples) return 0;
    if (!a->inited || a->rs || a->vpool || a->base || !(a->usage & VK_IMAGE_USAGE_SAMPLED_BIT) || img_pool_dead(a)) return 0;
    VkFormatFeatureFlags want = feats(f->vk) & (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
    return !(want & ~feats(dv));
}
static vka_img *depth_alias(const tex_t *t, const vka_fmt *f) {   // caller holds g_tmu; the depth image t shares (a reference taken), or NULL
    if (!g_look_sample || !t->mem || f->kind >= 3 || !depth_twin(f->vk) || t->has_layout || !VKA_ENV("DM_VK_DEPTH_ALIAS", 1)) return NULL;
    if (t->d.flags & (NVN_TEXTURE_FLAGS_IMAGE | NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) return NULL;   // (made to be written as colour)
    for (int i = 0; i < g_nlive; i++) if (depth_alias_ok(g_live[i], t, f)) {
        vka_img *a = g_live[i]; a->refs++; g_dalias_n++;
        if (g_dalias_n <= 8) VLOG("texture serial %llu %ux%u format %#x over the storage of depth texture serial %llu (format %#x, %s): shares its image, sampled through a depth view (%llu so far)",
                                  (unsigned long long)t->serial, t->d.width, t->d.height, t->d.format, (unsigned long long)a->serial, a->d.format, a->gpu_owned ? "GPU-written" : "not GPU-written yet",
                                  (unsigned long long)g_dalias_n);
        return a;
    }
    return NULL;
}
static int is_dalias(const tex_t *t, const vka_img *im) {   // t's serial maps to im through depth_alias
    vka_fmt cf;
    return im && im->f.kind >= 3 && t->d.format != im->d.format && vka_fmt_of(t->d.format, &cf) && cf.kind < 3;
}
static void dalias_end(smap_ent *e, vka_img *im, const tex_t *t) {   // caller holds g_tmu; a lookup that is no sample: the serial leaves the depth image
    e->serial = ~0ull; e->im = NULL;
    unsigned nid = 0;
    for (int id = 1; id < g_ntreg; id++) if (g_treg[id].t.serial == t->serial) { dirty(&g_tdirty, &g_ntd, &g_tdcap, &g_tdflag, 1 << 20, id); nid++; }
    g_dalias_end++;
    if (g_dalias_end <= 8) VLOG("texture serial %llu format %#x: looked up for a write as colour while it shared depth image serial %llu: an image of its own from now on (%u texture IDs written again)",
                                (unsigned long long)t->serial, t->d.format, (unsigned long long)im->serial, nid);
    // The reference dropped as unref_locked does (the recording memo goes), but a last one destroys the image after the submission being recorded: this submission's
    // table slots still show its depth view until write_tables rewrites them at the next one (the depth texture finalized before, so this serial held it alone)
    atomic_fetch_add_explicit(&g_tgen, 1, memory_order_release);
    if (--im->refs) return;
    for (int i = 0; i < g_nlive; i++) if (g_live[i] == im) { g_live[i] = g_live[--g_nlive]; break; }
    for (int i = 0; i < g_nbig; i++) if (g_big[i] == im) { g_big[i] = g_big[--g_nbig]; break; }
    rtw_forget(im); rl_note(im, "gone");
    g_img_bytes -= im->bytes;
    vka_after(vka_fence_last() + 1, img_destroy, im);
}
// ---------- Mip drop under memory pressure (DM_VK_MIPDROP) ----------
// A phone's low-memory killer counts image memory as the game's and cannot swap it. Ocarina of Time was killed at every Deku Tree → Kokiri Forest load: about 830 images
// (390-436 MiB) of guest textures made within 8 s, with MemAvailable already under 0.9 GB (session-2026-11-07 forest/ANALYSIS.md). Under pressure a texture whose content
// is its tracked storage (ST_TRACK), block-compressed, 2D / 2D array / cube, single-sampled, not linear and with no IMAGE, COMPRESSIBLE, SPARSE or DISPLAY flag, at least
// DM_VK_MIPDROP_MIN (256) texels on both sides and with more levels than it drops, is made from texture level DM_VK_MIPDROP_LEVELS (1) on: its finer levels are left out
// as for a sparse texture ("Sparse textures": image level 0 is texture level base; views, render targets and copies go through vka_img_lv), and its upload reads texture
// level lv + base. One level saves about 3/4 of the image. Sampling with an implicit level of detail reads the coarser level where the finer one was (blurrier, never black);
// an explicit level of detail, a texel fetch with a level or a size query sees the image's numbering. Compressed textures are never render targets; a copy or clear into
// one makes the image whole again first (vka_tex_own → md_whole: made from level 0 and uploaded from its storage, which is its content until the GPU writes it).
// At each second of submitting, idle images of the same kind (no lookup for DM_VK_MIPDROP_IDLE_S, 10, seconds: bindless samples are no lookups) are made again from that
// level too, at most DM_VK_MIPDROP_SHRINK_MB (64) of their memory a second (md_shrink). When the pressure ends, md_restore (DM_VK_MIPDROP_RESTORE_MB) makes them whole again.
// Pressure: MemAvailable (/proc/meminfo, read at most every 0.5 s) under DM_VK_MIPDROP_AVAIL_MB (1024), and for DM_VK_MIPDROP_HOLD_S (60) seconds after the last such
// reading (the killer's kills of other apps lift MemAvailable for a moment in the middle of a load). DM_VK_MIPDROP (live): 0 (default) off; 1 under pressure; 2 always.
// Pressure from the rate (DM_VK_PRESS_RATE, live, 0; Ocarina of Time 1): MemAvailable lags a load. Loading the Kokiri Forest save from the title (session-2026-11-07 fb3) took it
// from 2707 to 658 MiB in about 20 s (images +1.8 GB, 743 MiB of them in 1.1 s while it still read 2201), and the 1024 MiB line came 3 s before the end of the burst that killed
// the game, with 280 MiB of binding-path images already made. So pressure is also on while MemAvailable is under DM_VK_PRESS_RATE_AVAIL_MB (2048) and either fell more than
// DM_VK_PRESS_RATE_MB (400) within the last DM_VK_PRESS_RATE_S (5) seconds of readings, or more than DM_VK_PRESS_BURST_MB (384) of image memory was made in them (g_img_made: a
// burst the kernel has not shown yet), and for DM_VK_PRESS_RATE_HOLD_S (20) seconds after. Mip drop, coarse-first and the feedback gate all ask mem_low; critical pressure
// (mem_crit, half of DM_VK_MIPDROP_AVAIL_MB) stays the absolute reading. The `mip drop:` line says which test turned it on
// The `memory:` line gives the images made dropped, shrunk and made whole again, and new images by the path that made them (table: a scan named the ID since its
// registration / the ID was wanted before this registration (DM_VK_TWANT_KEEP) / every ID written; binding; other)
static uint64_t g_md_n, g_md_saved, g_mds_n, g_mds_saved, g_mdw_n;   // (under g_tmu) made dropped and the bytes saved, shrunk and saved, made whole again
static uint64_t g_mrs_n, g_mrs_b0, g_mrs_b1;   // (under g_tmu) made whole again by md_restore, their bytes before and after
static long g_md_avail = -1;   // the last MemAvailable reading (MiB; -1: none)
static uint64_t g_md_fall_ns;   // when mem_low's rate test last fired (a fast fall or an image burst; 0: never): the feedback release window's fast case (tfb_fast)
static long md_avail_mib(void) {
    FILE *f = fopen("/proc/meminfo", "r"); if (!f) return -1;
    char ln[128]; long kb = -1; while (fgets(ln, sizeof ln, f)) if (sscanf(ln, "MemAvailable: %ld kB", &kb) == 1) break;
    fclose(f); return kb < 0 ? -1 : kb >> 10;
}
// (under g_tmu) "Coarse-first" below: provisional images made and their bytes; promoted (made from level 0, or mip drop's level under pressure) by trigger, and their bytes after
enum { CF_LOOK, CF_WRITE, CF_BG, CF_N };
static uint64_t g_sc_prov_n, g_sc_prov_b, g_cf_pn[CF_N], g_cf_pb[CF_N];
enum { MP_HIST = 32 };   // readings kept for the rate test (one each 0.5 s at most: 16 s)
static int mem_low(void) {   // caller holds g_tmu: MemAvailable under DM_VK_MIPDROP_AVAIL_MB now or within DM_VK_MIPDROP_HOLD_S, or the rate test (mip drop, coarse-first, the feedback gate)
    static uint64_t next, until, ht[MP_HIST], hm[MP_HIST]; static long ha[MP_HIST], wv; static int on, nlog, hn, hi, why; uint64_t t = now_ns();
    if (t >= next) {
        next = t + 500000000ull;
        long a = md_avail_mib(), lim = VKA_ENV("DM_VK_MIPDROP_AVAIL_MB", 1024); g_md_avail = a;
        if (a >= 0) { ht[hi] = t; ha[hi] = a; hm[hi] = g_img_made; hi = (hi + 1) % MP_HIST; if (hn < MP_HIST) hn++; }
        if (a >= 0 && a < lim) { until = t + (uint64_t)VKA_ENV("DM_VK_MIPDROP_HOLD_S", 60) * 1000000000ull; if (!(t < until && on)) why = 0; }
        else if (a >= 0 && VKA_ENV("DM_VK_PRESS_RATE", 0) && a < VKA_ENV("DM_VK_PRESS_RATE_AVAIL_MB", 2048)) {   // ("Pressure from the rate" above)
            double ws = VKA_ENVF("DM_VK_PRESS_RATE_S", 5.0); uint64_t w = ws > 0 ? (uint64_t)(ws * 1e9) : 0, m0 = g_img_made; long top = a;
            for (int k = 0; k < hn; k++) { int j = (hi - 1 - k + MP_HIST) % MP_HIST; if (t - ht[j] > w) break; if (ha[j] > top) top = ha[j]; m0 = hm[j]; }
            long rmb = VKA_ENV("DM_VK_PRESS_RATE_MB", 400), bmb = VKA_ENV("DM_VK_PRESS_BURST_MB", 384); uint64_t made = (g_img_made - m0) >> 20;
            int r = rmb > 0 && top - a > rmb, b = bmb > 0 && made > (uint64_t)bmb;
            if (r || b) g_md_fall_ns = t;
            if (r || b) { uint64_t u = t + (uint64_t)(VKA_ENVF("DM_VK_PRESS_RATE_HOLD_S", 20.0) * 1e9); if (!(t < until && on)) { why = r ? 1 : 2; wv = r ? top - a : (long)made; } if (u > until) until = u; }
        }
    }
    // Hysteresis (DM_VK_PRESS_HYST_MB, live, 0; Ocarina of Time 256): pressure that is on stays on while the last reading is under DM_VK_MIPDROP_AVAIL_MB + this, and
    // ends only after DM_VK_PRESS_OFF_S (5) seconds in which no test held it. Each end makes what the pressure kept back at once (the feedback gate's deferred IDs, full-size
    // textures): OoT's walk sat at MemAvailable 0.98-1.12 GB, on the 1024 MiB line, and every end made 2.2-3.9k textures (380-600 MB of uploads, a 1.4-3.2 s frame) whose
    // burst turned the pressure on again within 1 s (session-2026-11-07 day2/D2.md section 3). 0: ends at the hold's end, as before
    long hyst = VKA_ENV("DM_VK_PRESS_HYST_MB", 0); static uint64_t off_since;
    int now = t < until;
    if (hyst > 0 && on) {
        if (!now && g_md_avail >= 0 && g_md_avail < VKA_ENV("DM_VK_MIPDROP_AVAIL_MB", 1024) + hyst) now = 1;
        if (now) off_since = 0;
        else { double os = VKA_ENVF("DM_VK_PRESS_OFF_S", 5.0); if (!off_since) off_since = t; if (os > 0 && t - off_since < (uint64_t)(os * 1e9)) now = 1; }
    } else off_since = 0;
    if (now != on && nlog < 16) { nlog++;
        char wb[160]; if (why == 1) snprintf(wb, sizeof wb, "; on as it fell %ld MiB within %.0f s, under DM_VK_PRESS_RATE_AVAIL_MB", wv, VKA_ENVF("DM_VK_PRESS_RATE_S", 5.0));
        else if (why == 2) snprintf(wb, sizeof wb, "; on as %ld MiB of image memory was made within %.0f s, under DM_VK_PRESS_RATE_AVAIL_MB", wv, VKA_ENVF("DM_VK_PRESS_RATE_S", 5.0)); else wb[0] = 0;
        VLOG("mip drop: %s (MemAvailable %ld MiB, DM_VK_MIPDROP_AVAIL_MB %d%s; %llu images made without their finer levels, %.0f MiB saved; %llu shrunk, %.0f MiB saved; %llu made whole again; "
             "coarse-first: %llu provisional, %.0f MiB)",
             !now ? "the pressure is over: new textures whole again" : VKA_ENV("DM_VK_SCAN_COARSE", 0) == 1
                 ? "memory pressure: textures from storage are made without their finest level(s) from now on (DM_VK_MIPDROP), idle ones shrunk; scan-named ones coarse first (DM_VK_SCAN_COARSE)"
                 : "memory pressure: textures from storage are made without their finest level(s) from now on (DM_VK_MIPDROP), idle ones shrunk",
             g_md_avail, VKA_ENV("DM_VK_MIPDROP_AVAIL_MB", 1024), now ? wb : "", (unsigned long long)g_md_n, g_md_saved / 1048576.0, (unsigned long long)g_mds_n, g_mds_saved / 1048576.0, (unsigned long long)g_mdw_n,
             (unsigned long long)g_sc_prov_n, g_sc_prov_b / 1048576.0); }
    on = now;
    return now;
}
// Critical pressure (caller holds g_tmu): the MemAvailable reading (refreshed through mem_low) under half of DM_VK_MIPDROP_AVAIL_MB ("Texture-use feedback": pinned serials released)
static int mem_crit(void) { (void)mem_low(); long lim = VKA_ENV("DM_VK_MIPDROP_AVAIL_MB", 1024) / 2; return g_md_avail >= 0 && g_md_avail < lim; }
// The feedback release window's fast case (tfb_idle_s; caller holds g_tmu): MemAvailable under DM_VK_MIPDROP_AVAIL_MB, or the rate test fired within DM_VK_PRESS_RATE_S.
// mem_crit's half (512 MiB) is under the low-memory killer's line on the phone (OoT kills at MemAvailable 0.5-0.7 GB: fb3 658 MiB, play1 ~0.7 GB, 525 MiB), so a 30 s window
// held until mem_crit kept a load's dead images until the kill (fb1 got through the forest load on the 2 s window)
static int tfb_fast(void) {
    if (mem_crit()) return 1;
    if (g_md_avail >= 0 && g_md_avail < VKA_ENV("DM_VK_MIPDROP_AVAIL_MB", 1024)) return 1;
    double ws = VKA_ENVF("DM_VK_PRESS_RATE_S", 5.0); uint64_t t = now_ns();
    return g_md_fall_ns && ws > 0 && t - g_md_fall_ns < (uint64_t)(ws * 1e9);
}
static int md_pressure(void) {   // caller holds g_tmu
    int mode = VKA_ENV("DM_VK_MIPDROP", 0);
    return mode <= 0 ? 0 : mode >= 2 ? 1 : mem_low();
}
// Why a texture made while mip drop's pressure is on keeps its finer levels (the `memory:` line's "mip drop while on", by the path that made it; checked in this order): its
// content (ST_ZERO: GPU content, render targets and copy destinations; ST_FILL under DM_VK_MIPDROP_FILL=0; sparse: ST_SPARSE, a sparse base or pool), DM_VK_MIPDROP_LEVELS 0,
// single-level, shape (depth / stencil, multisample, not 2D / 2D array / cube), flags, not block-compressed, under DM_VK_MIPDROP_MIN; coarse-first made it provisional
// instead; img_create made it from another level. fb3 (session-2026-11-07) made 530 images (297 MiB) on the binding path under pressure and none without finer levels.
// Taken since 2026-11-08 (each with its own switch, live, default 1; 0 leaves them whole as before), as coarse-first takes them:
//   - ST_FILL content (DM_VK_MIPDROP_FILL): a CPU_NO_ACCESS or uncached pool, or a range the CPU flushed (streamed textures a GPU copy filled are such). Its content is its storage
//     too, read through upload_fill (the GPU's pending writes applied) and uploaded again after a GPU write into the storage (vka_tex_prepare), so img_relevel to level 0
//     (md_whole) reads what the first upload would have, or newer storage, as the Switch's GPU does; md_shrink leaves it alone (tracked images only);
//   - COMPRESSIBLE (DM_VK_MIPDROP_COMPRESSIBLE): the flag only lets the Switch's GPU keep its writes compressed; a block-compressed texture is ST_TRACK / ST_FILL only from a
//     packaged layout, a CPU write or a fill, so its bytes are plain blocks, and a GPU write makes it whole first (vka_tex_own → md_whole);
//   - IMAGE on a block-compressed format (DM_VK_MIPDROP_IMAGE): no Vulkan block-compressed format is a storage image here (img_create drops the use), and every shader-image
//     binding or registration of the texture makes a dropped image whole first (id_view / write_tex: vka_tex_whole), as copies and clears do.
// An uncompressed texture stays whole: an uncompressed mipmapped texture is often data a shader reads with texelFetch or an explicit level, which a dropped image would answer
// from the wrong level (a wrong value, not a blur), and its learned view formats are keyed by its whole size
enum { MDR_DROP, MDR_PROV, MDR_ZERO, MDR_FILL, MDR_SPARSE, MDR_OFF, MDR_ONE, MDR_SHAPE, MDR_FLAGS, MDR_UNCOMP, MDR_SMALL, MDR_BASE, MDR_N };
static const char *const mdr_name[MDR_N] = {"made without finer levels", "coarse-first", "ST_ZERO", "ST_FILL", "sparse", "DM_VK_MIPDROP_LEVELS 0", "single-level", "shape", "flags",
                                            "not compressed", "under DM_VK_MIPDROP_MIN", "made from another level"};
static uint32_t md_xflags(const vka_fmt *f) {   // the texture flags that keep a texture whole
    uint32_t m = NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_SPARSE | NVN_TEXTURE_FLAGS_DISPLAY;
    if (!VKA_ENV("DM_VK_MIPDROP_COMPRESSIBLE", 1)) m |= NVN_TEXTURE_FLAGS_COMPRESSIBLE;
    if (!compressed(f) || !VKA_ENV("DM_VK_MIPDROP_IMAGE", 1)) m |= NVN_TEXTURE_FLAGS_IMAGE;
    return m;
}
static uint32_t md_levels_why(const nvn_texdesc *d, const vka_fmt *f, int *why) {   // the levels mip drop leaves out of such a texture (0: none, *why the reason when why is given)
    int k = VKA_ENV("DM_VK_MIPDROP_LEVELS", 1), min = VKA_ENV("DM_VK_MIPDROP_MIN", 256), w = -1;
    uint32_t n = d->levels ? d->levels : 1;
    if (k <= 0) w = MDR_OFF;
    else if (n < 2) w = MDR_ONE;
    else if (f->kind >= 3 || d->samples > 1 || (d->target != NVN_TEXTURE_TARGET_2D && d->target != NVN_TEXTURE_TARGET_2D_ARRAY && d->target != NVN_TEXTURE_TARGET_CUBEMAP
                                                 && d->target != NVN_TEXTURE_TARGET_CUBEMAP_ARRAY)) w = MDR_SHAPE;
    else if (d->flags & md_xflags(f)) w = MDR_FLAGS;
    else if (!compressed(f)) w = MDR_UNCOMP;
    else if ((int)d->width < min || (int)d->height < min) w = MDR_SMALL;
    if (why) *why = w < 0 ? MDR_DROP : w;
    return w >= 0 ? 0 : (uint32_t)k < n ? (uint32_t)k : n - 1;
}
static uint32_t md_levels(const nvn_texdesc *d, const vka_fmt *f) { return md_levels_why(d, f, NULL); }
static int md_content(int mode) { return mode == ST_TRACK || (mode == ST_FILL && VKA_ENV("DM_VK_MIPDROP_FILL", 1)); }   // content mip drop takes (ST_FILL: md_levels_why's comment)
// A tracked (or ST_FILL) image made again from texture level base, in place (every serial and view sharing it follows): its content uploaded from its storage, through the upload batch
// when cb is the replay's; the old image, its views and memory go after the submission being recorded; the recording memo is dropped and the IDs naming it written again.
// Caller holds g_tmu and nvn_pool_use; no render pass open in cb unless the upload batch is on. 1 = done
static int img_relevel(vka_img *im, uint32_t base, VkCommandBuffer cb) {
    int fill = !im->track && im->mode == ST_FILL && im->from_mem;   // (an ST_FILL image: its content is its storage too, with the GPU's writes into it re-uploaded: "Coarse-first")
    if (!cb || (!im->track && !fill) || im->gpu_owned || im->vpool || im->rs || im->holds || im->m.z || img_pool_dead(im)) return 0;
    tex_t t = {0}; t.serial = im->serial; t.d = im->d; t.L = im->L; t.mem = (uint8_t *)im->mem;
    VkCommandBuffer ucb = !vka_scr_up && cb == vka_rp_cb() && vka_up_on() ? vka_up_begin() : VK_NULL_HANDLE, rc = ucb ? ucb : cb;
    vka_img *n = img_create(&t, &im->f, base);
    if (!n || n->base != base) { if (n) { g_img_bytes -= n->bytes; img_destroy(n); } if (ucb) vka_up_end(); return 0; }
    vka_img *old = calloc(1, sizeof *old);
    if (!old) { g_img_bytes -= n->bytes; img_destroy(n); if (ucb) vka_up_end(); return 0; }
    old->img = im->img; old->m = im->m; old->views = im->views;
    g_img_bytes -= im->bytes;
    im->img = n->img; im->m = n->m; im->views = NULL; im->usage = n->usage; im->w = n->w; im->h = n->h; im->levels = n->levels; im->base = n->base; im->bytes = n->bytes;
    im->alt = n->alt; im->any_fmt = n->any_fmt; im->nx = n->nx; memcpy(im->xfmt, n->xfmt, sizeof im->xfmt);
    free(n);
    vka_after(vka_fence_last() + 1, img_destroy, old);   // (after the submission being recorded, which may hold commands that read it)
    atomic_fetch_add_explicit(&g_tgen, 1, memory_order_release);
    redirty_ids(im);
    vka_img_barrier(rc, im->img, im->aspect, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    if (fill) upload_fill(im, &t, rc); else upload_tracked(im, &t, rc);
    if ((rc = vka_up_cur(rc))) vka_barrier(rc);
    if (ucb) { vka_up_point(vka_up_cur(ucb)); vka_up_end(); }
    return 1;
}
// A dropped image a GPU write is about to reach (vka_tex_own): whole again first (caller holds g_tmu and nvn_pool_use)
static void md_whole(vka_img *im, VkCommandBuffer cb) {
    uint32_t was = im->base; uint64_t b0 = im->bytes;
    if (!img_relevel(im, 0, cb)) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8)
        VRED("mip drop: texture serial %llu %ux%u format %#x (levels %u+%u) not made whole before a GPU write or copy (held, pool finalized or no memory): its levels under %u are not there",
             (unsigned long long)im->serial, im->d.width, im->d.height, im->d.format, im->base, im->levels, im->base); return; }
    int prov = im->prov; im->mdrop = 0; im->prov = 0; im->cf_want = 0;
    if (prov) { g_cf_pn[CF_WRITE]++; g_cf_pb[CF_WRITE] += im->bytes;   // ("Coarse-first": a GPU write, copy or attachment promotes a provisional image at once, outside the rate caps)
        if (g_cf_pn[CF_WRITE] <= 8) VLOG("coarse-first: provisional texture serial %llu %ux%u format %#x made whole before a GPU write or copy (from level %u, %.2f → %.2f MiB)",
                                         (unsigned long long)im->serial, im->d.width, im->d.height, im->d.format, was, b0 / 1048576.0, im->bytes / 1048576.0);
        return; }
    g_mdw_n++;
    if (g_mdw_n <= 8) VLOG("mip drop: texture serial %llu %ux%u format %#x made whole again before a GPU write or copy (from level %u, %.1f → %.1f MiB)", (unsigned long long)im->serial,
                           im->d.width, im->d.height, im->d.format, was, b0 / 1048576.0, im->bytes / 1048576.0);
}
// New images by the path that asked for them (the `memory:` line): MK_FRESH a table slot whose ID a scanned storage buffer named since its registration, MK_STICKY one
// wanted before this registration (g_twant 2: DM_VK_TWANT_KEEP), MK_ALL a slot written with no lazy deferral (lazy images ended or off, an image registration),
// MK_BIND a draw's or dispatch's binding (id_view), MK_OTHER the rest (attachments, copies, clears, the present)
enum { MK_OTHER, MK_BIND, MK_FRESH, MK_STICKY, MK_ALL, MK_N };
static _Thread_local int g_mk_path, g_mk_id;   // g_mk_id: the table slot write_tables is writing (0: none)
static uint64_t g_mk_n[MK_N], g_mk_b[MK_N];   // (under g_tmu)
// Of those (DM_VK_TEXFB >= 2, the `memory:` line): made for a serial image eviction let go before (g_evs; a floor, direct-mapped), of them after a feedback release (g_tfbr)
// and within 30 s of it; zero-shared at creation (their bytes count 0 in g_mk_b until their first write gives them memory of their own: m.size here)
static uint64_t g_mk_an[MK_N], g_mk_ab[MK_N], g_mk_rn[MK_N], g_mk_rb[MK_N], g_mk_r30[MK_N], g_mk_zn[MK_N], g_mk_zb[MK_N];   // (under g_tmu)
// Counters only (the `memory:` line; host-only.md §5): scan-named images (MK_FRESH) by class: SC_MIPC mip drop's class (compressed, mipmapped, at least 256x256), SC_MIPO
// other mipmapped, SC_ONE single-level, SC_SMALL any under 64 KiB (checked first)
enum { SC_MIPC, SC_MIPO, SC_ONE, SC_SMALL, SC_N };
static uint64_t g_sc_n[SC_N], g_sc_b[SC_N];   // (under g_tmu)
// The scanned ranges that named an ID (vka_tex_pool_mark's range): a draw's window (NM_WIN or less) or only larger ones (scene-wide tables; size 0, unknown, counts as one).
// g_tnamed per ID, reset when another texture takes it: 1 named by a window, 2 by a table, 4 its scan-named image(s) counted under "only tables" (g_tnkib KiB), moved to
// "windows" when a window names it later. g_nm_n/kib[0] windows, [1] only tables: scan-named images made and their KiB; g_nm_ids: IDs named
enum { NM_WIN = 256 << 10 };
static uint8_t *g_tnamed; static uint32_t *g_tnkib; static uint64_t g_nm_n[2], g_nm_kib[2], g_nm_ids[2];   // (under g_tmu)
static void nm_named(int id, uint8_t b) {   // caller holds g_tmu; b 1 or 2, not yet in g_tnamed[id]
    uint8_t o = g_tnamed[id]; g_tnamed[id] = o | b;
    if (b == 2) { if (!(o & 1)) g_nm_ids[1]++; return; }
    g_nm_ids[0]++; if (o & 2) g_nm_ids[1]--;
    if (o & 4) { uint64_t k = g_tnkib[id]; g_tnamed[id] &= (uint8_t)~4; g_tnkib[id] = 0; g_nm_n[1]--; g_nm_kib[1] -= k; g_nm_n[0]++; g_nm_kib[0] += k; }
}
static void sc_count(int id, const nvn_texdesc *d, const vka_fmt *f, uint64_t bytes, uint64_t whole) {   // caller holds g_tmu: a scan-named image made (whole: its class's size, a provisional one's whole size)
    int c = whole < (64 << 10) ? SC_SMALL : (d->levels ? d->levels : 1) < 2 ? SC_ONE : compressed(f) && d->width >= 256 && d->height >= 256 ? SC_MIPC : SC_MIPO;
    g_sc_n[c]++; g_sc_b[c] += bytes;
    if (!g_tnamed || id <= 0 || id >= (1 << 20)) return;
    uint32_t k = (uint32_t)(bytes >> 10); int w = g_tnamed[id] & 1;
    g_nm_n[!w]++; g_nm_kib[!w] += k;
    if (!w) { g_tnamed[id] |= 4; g_tnkib[id] += k; }
}
// ---------- Coarse-first scan-named images under memory pressure (DM_VK_SCAN_COARSE) ----------
// At Ocarina of Time's Kokiri Forest load the per-frame pool scan names about 1,500 texture IDs in scene-wide material tables (99 % only by tables of MiBs; a draw's own
// window names 1 %), and their images (+505 MiB in 4 s, R1) outran the low-memory killer. No host evidence says which of them a pool-reading draw samples, and a slot left
// empty reads black (o4). So under memory pressure (mip drop's MemAvailable test, mem_low) a scan-named texture (MK_FRESH / MK_STICKY: no lookup asked for it) that is
// tracked (ST_TRACK, or ST_FILL: cf_levels' comment), colour, single-sampled, 2D / 2D array / cube, not linear, with no IMAGE, COMPRESSIBLE, SPARSE or DISPLAY flag, at least 64 KiB whole and mipmapped
// is made "provisional": from its first level whose larger side is at most DM_VK_SCAN_COARSE_DIM (128) texels (img_create base, as mip drop; im->mdrop == im->base,
// im->prov). Its slot shows that real level and the coarser ones, never a null view. An uncompressed one whose format and size have learned view formats is made whole
// (the coarse image would miss them). Single-level and smaller textures are made whole as before (counted). A window-named one (g_tnamed 1) is made whole while the frame
// budget lasts. Promotion (img_relevel to level 0, or to mip drop's level under DM_VK_MIPDROP's pressure): a GPU write, copy or attachment at once (md_whole through
// vka_tex_own / vka_tex_whole, rp_begin, a shader-image binding or registration); a lookup (a draw's binding, id_view and the recording memo; other non-table lookups) at
// the next submission's start (cf_promote); in the background, when MemAvailable is at least DM_VK_SCAN_PROMOTE_AVAIL_MB (1536) and no load runs (fewer than CF_BURST
// new images in each of the last CF_QUIET seconds of submitting), live-list order (about oldest first). Lookup and background promotions: at most DM_VK_SCAN_PROMOTE_MB
// (32) of new image memory a second, and while the pressure is on, full-size scan-named images (made whole or promoted) at most DM_VK_SCAN_FULL_MB (48) a frame (a
// single larger one when it is the frame's first). DM_VK_SCAN_COARSE (live): 0 (default) off, 1 under pressure, 2 always (a visual check; no background promotion).
// What a provisional texture costs: sampling with an implicit LOD reads the coarse level (blur, pop-in at promotion); textureSize / texelFetch / explicit LOD see the
// image's numbering (base k); the `memory:` line's coarse-first counts and `coarse-first:` lines
static int in_fb(const vka_img *im);
enum { CF_BURST = 32, CF_QUIET = 2 };
static uint64_t g_cf_full_b, g_cf_wn, g_cf_wb, g_cf_xn, g_cf_xb;   // (under g_tmu) provisional ones' whole bytes; window-named made whole; not eligible made whole while on
static uint64_t g_cf_frame, g_cf_frame_b, g_cf_sec_b; static uint32_t g_cf_sec = ~0u, g_cf_new_s, g_cf_burst_s; static int g_cf_new_n;   // (under g_tmu)
static _Atomic int g_cf_any;   // a lookup flagged a provisional image (cf_want) since the last cf_promote
static int cf_mode(void) { return VKA_ENV("DM_VK_SCAN_COARSE", 0); }
static int cf_on(void) { int m = cf_mode(); return m <= 0 ? 0 : m >= 2 ? 1 : mem_low(); }   // caller holds g_tmu
static int cf_frame_fits(uint64_t b) {   // caller holds g_tmu: b more bytes of full-size scan-named images fit this frame's budget
    uint64_t fr = vka_frame(); if (fr != g_cf_frame) { g_cf_frame = fr; g_cf_frame_b = 0; }
    return !g_cf_frame_b || g_cf_frame_b + b <= (uint64_t)(VKA_ENVF("DM_VK_SCAN_FULL_MB", 48.0) * 1048576.0);
}
static void cf_frame_add(uint64_t b) { uint64_t fr = vka_frame(); if (fr != g_cf_frame) { g_cf_frame = fr; g_cf_frame_b = 0; } g_cf_frame_b += b; }
// Reported textures first (DM_VK_TEXFB_REPORT_MB=<MiB>, live; default 0 = off, as before; Ocarina of Time 128): an ID the GPU indexed (DM_VK_TEXFB's report: wanted by a
// readback, or let through by the gate's evidence) is made at full size (mip drop's level under its pressure) instead of coarse-first's provisional image, and a provisional
// image a report asked for is promoted, both outside coarse-first's caps (DM_VK_SCAN_FULL_MB a frame, DM_VK_SCAN_PROMOTE_MB a second) while their own budget of this many MiB
// a replay frame lasts (one larger image when it is the frame's first). seedds/nxnative (memory.md, the texture-use feedback path) makes reported textures whatever the budget, up
// to 128 MB a frame. The play run (session-2026-11-07 play/PLAY.md) had GPU-indexed textures blurry or grey for frames behind the 48 MiB / 32 MiB caps
static uint64_t g_rep_frame, g_rep_frame_b, g_rep_n, g_rep_b, g_rep_pn, g_rep_pb, g_rep_over;   // (under g_tmu) this frame's bytes; made at full size, promoted; budget hits
static int rep_on(void) { return VKA_ENVF("DM_VK_TEXFB_REPORT_MB", 0.0) > 0; }
static int rep_fits(uint64_t b) {   // caller holds g_tmu
    uint64_t fr = vka_frame(); if (fr != g_rep_frame) { g_rep_frame = fr; g_rep_frame_b = 0; }
    double mb = VKA_ENVF("DM_VK_TEXFB_REPORT_MB", 0.0);
    if (mb <= 0) return 0;
    int ok = !g_rep_frame_b || g_rep_frame_b + b <= (uint64_t)(mb * 1048576.0);
    if (!ok) g_rep_over++;
    return ok;
}
static void rep_add(uint64_t b) { uint64_t fr = vka_frame(); if (fr != g_rep_frame) { g_rep_frame = fr; g_rep_frame_b = 0; } g_rep_frame_b += b; }
static void cf_note_new(void) { if (g_cf_new_s != g_now_s) { g_cf_new_s = g_now_s; g_cf_new_n = 0; } if (++g_cf_new_n >= CF_BURST) g_cf_burst_s = g_now_s; }   // (a load's image burst)
// Why a scan-named texture made while coarse-first is on was made whole ("not eligible", the `memory:` line's reasons; checked in this order): its content mode (ST_ZERO,
// ST_SPARSE, or ST_FILL under DM_VK_SCAN_COARSE_FILL=0), a sparse base, a sparse pool, single-level, shape (depth / stencil, multisample, a target other than 2D / 2D array /
// cube), flags, under 64 KiB whole, learned view formats (uncompressed), already at most DM_VK_SCAN_COARSE_DIM texels a side, mip drop's level as coarse (cfk <= drop), the
// image made from another level (base != cfk: a sparse or dense floor).
// ST_FILL content (a CPU_NO_ACCESS or virtual pool, an uncached pool or a flushed range: uploaded once, not tracked) is taken too (DM_VK_SCAN_COARSE_FILL, default 1): its
// content is also its storage, read through the same upload (upload_fill: the GPU's pending writes applied, unmapped pages as zeros) and re-uploaded after a GPU write
// into it (vka_tex_prepare), so a promotion (img_relevel) reads what the first upload would have, or newer storage, as the Switch's GPU does
enum { CFW_FILL, CFW_ZERO, CFW_SPARSE, CFW_SPBASE, CFW_SPPOOL, CFW_ONE, CFW_SHAPE, CFW_FLAGS, CFW_SMALL, CFW_LEARNED, CFW_COARSE, CFW_DROP, CFW_BASE, CFW_N };
static const char *const cfw_name[CFW_N] = {"ST_FILL", "ST_ZERO", "ST_SPARSE", "sparse base", "sparse pool", "single-level", "shape", "flags", "under 64 KiB", "learned formats",
                                            "already coarse", "cfk <= drop", "base != cfk"};
static uint64_t g_cfw_n[CFW_N], g_cfw_b[CFW_N], g_cf_filln, g_cf_fillb;   // (under g_tmu) not eligible by reason; provisional ones whose content is ST_FILL
// The "flags" reason by flag (an image counted under each excluding flag it has), and provisional images with the COMPRESSIBLE flag. Why each flag excludes:
//   - IMAGE: shaders may write it (an image binding or registration); not shown safe for a coarse image here (mip drop leaves it out too), so it stays out;
//   - SPARSE: a page table may map its storage; LINEAR / LINEAR_RENDER_TARGET: pitch-linear storage (row uploads, linear write-back), not block-linear levels;
//     DISPLAY: a presentable target. These stay out;
//   - COMPRESSIBLE: only lets the Switch's GPU keep its writes compressed in memory. It changes neither sampling nor the texture's layout, and whether the storage is the
//     content is storage_mode's decision (a COMPRESSIBLE texture is ST_TRACK only after a CPU write the dirty tracker saw, or with a layout: its bytes are then plain texels;
//     never ST_FILL by itself): the provisional image uploads a subset of the levels the whole one would from the same storage, and a GPU write or attachment promotes it
//     first (md_whole, as for any provisional image). So it is taken (DM_VK_SCAN_COARSE_COMPRESSIBLE, default 1; 0: excluded as before 2026-11-08)
enum { CFF_IMAGE, CFF_COMP, CFF_SPARSE, CFF_LINEAR, CFF_LRT, CFF_DISPLAY, CFF_N };
static const uint32_t cff_bit[CFF_N] = {NVN_TEXTURE_FLAGS_IMAGE, NVN_TEXTURE_FLAGS_COMPRESSIBLE, NVN_TEXTURE_FLAGS_SPARSE, NVN_TEXTURE_FLAGS_LINEAR, NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET, NVN_TEXTURE_FLAGS_DISPLAY};
static const char *const cff_name[CFF_N] = {"IMAGE", "COMPRESSIBLE", "SPARSE", "LINEAR", "LINEAR_RENDER_TARGET", "DISPLAY"};
static uint64_t g_cff_n[CFF_N], g_cff_b[CFF_N], g_cf_compn, g_cf_compb;   // (under g_tmu)
static uint32_t cf_xflags(void) {   // the texture flags that keep a scan-named texture whole
    uint32_t m = NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET | NVN_TEXTURE_FLAGS_IMAGE | NVN_TEXTURE_FLAGS_SPARSE | NVN_TEXTURE_FLAGS_DISPLAY;
    return VKA_ENV("DM_VK_SCAN_COARSE_COMPRESSIBLE", 1) ? m : m | NVN_TEXTURE_FLAGS_COMPRESSIBLE;
}
static uint32_t cf_levels(const tex_t *t, const vka_fmt *f, int *why) {   // the texture level a provisional image of t starts at (0: t is not one, *why the reason)
    const nvn_texdesc *d = &t->d; uint32_t n = d->levels ? d->levels : 1;
    if (n < 2) { *why = CFW_ONE; return 0; }
    if (f->kind >= 3 || d->samples > 1) { *why = CFW_SHAPE; return 0; }
    if (d->target != NVN_TEXTURE_TARGET_2D && d->target != NVN_TEXTURE_TARGET_2D_ARRAY && d->target != NVN_TEXTURE_TARGET_CUBEMAP && d->target != NVN_TEXTURE_TARGET_CUBEMAP_ARRAY) { *why = CFW_SHAPE; return 0; }
    if (d->flags & cf_xflags()) { *why = CFW_FLAGS; return 0; }
    if (dense_bytes(d, &t->L, 0) < (64u << 10)) { *why = CFW_SMALL; return 0; }
    uint32_t dim = (uint32_t)VKA_ENV("DM_VK_SCAN_COARSE_DIM", 128), w = d->width ? d->width : 1, h = d->height ? d->height : 1, k = 0;
    if (!dim) dim = 1;
    while (k + 1 < n && ((w >> k) > dim || (h >> k) > dim)) k++;
    if (!k) { *why = CFW_COARSE; return 0; }
    if (!compressed(f)) { VkFormat x[VL_PER_IMAGE]; pthread_mutex_lock(&g_vlmu); int nx = vl_formats(f->vk, w, h, x, VL_PER_IMAGE); pthread_mutex_unlock(&g_vlmu); if (nx) { *why = CFW_LEARNED; return 0; } }
    return k;
}
static void cf_look(vka_img *im) {   // caller holds g_tmu: a lookup that is no table write asks for a provisional image's promotion
    if (im && im->prov && !im->cf_want && (g_mk_path == MK_BIND || g_mk_path == MK_OTHER)) { im->cf_want = 1; atomic_store(&g_cf_any, 1); }
}
// At a submission's start (vka_tex_prepare; caller holds g_tmu and nvn_pool_use; no render pass open in cb): provisional images a lookup asked for, then in the background
static void cf_promote(VkCommandBuffer cb) {
    int want = atomic_exchange(&g_cf_any, 0), mode = cf_mode();
    static uint32_t bg_last = ~0u; static int nlog;
    uint64_t gone = g_cf_pn[CF_LOOK] + g_cf_pn[CF_WRITE] + g_cf_pn[CF_BG];
    int bg = 0;
    if (mode < 2 && g_sc_prov_n > gone && bg_last != g_now_s) {   // (provisional images let go uncounted keep this true: one live-list walk a second at most)
        bg_last = g_now_s; (void)mem_low();   // (g_md_avail fresh)
        bg = g_md_avail >= VKA_ENV("DM_VK_SCAN_PROMOTE_AVAIL_MB", 1536) && g_now_s - g_cf_burst_s >= CF_QUIET;
    }
    if (!want && !bg) return;
    if (g_cf_sec != g_now_s) { g_cf_sec = g_now_s; g_cf_sec_b = 0; }
    uint64_t cap = (uint64_t)(VKA_ENVF("DM_VK_SCAN_PROMOTE_MB", 32.0) * 1048576.0);
    if (!cap) return;   // (0: only GPU writes, copies and attachments promote)
    int pr = cf_on(), left = 0, n[CF_N] = {0}; uint64_t b0s = 0, b1s = 0;
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 0 ? !want : !bg) continue;
        for (int i = 0; i < g_nlive; i++) {
            vka_img *im = g_live[i];
            if (!im->prov || (pass == 0 && !im->cf_want)) continue;
            if (img_pool_dead(im)) { im->cf_want = 0; continue; }   // (its storage is not read again: stays as it is)
            if (im->holds || in_fb(im)) { left |= im->cf_want; continue; }
            uint32_t tgt = md_pressure() ? md_levels(&im->d, &im->f) : 0; if (tgt >= im->base) tgt = 0;
            uint64_t est = dense_bytes(&im->d, &im->L, tgt);
            int byrep = pass == 0 && im->cf_rep && rep_on();   // ("Reported textures first": its own frame budget instead of the caps)
            if (byrep && !rep_fits(est)) { left |= want; continue; }
            if (!byrep && ((g_cf_sec_b && g_cf_sec_b + est > cap) || (pr && !cf_frame_fits(est)))) { left |= want; if (rep_on()) continue; goto out; }
            uint64_t b0 = im->bytes; uint32_t ob = im->base;
            if (!img_relevel(im, tgt, cb)) { static _Atomic int nf; if (atomic_fetch_add(&nf, 1) < 8)
                VRED("coarse-first: provisional texture serial %llu %ux%u format %#x (levels %u+%u) not promoted (pool finalized, no memory or not tracked): it stays coarse",
                     (unsigned long long)im->serial, im->d.width, im->d.height, im->d.format, im->base, im->levels);
                im->cf_want = 0; continue; }
            int k = pass == 0 ? CF_LOOK : CF_BG;
            im->mdrop = (uint8_t)tgt; im->prov = 0; im->cf_want = 0; im->cf_rep = 0;
            if (byrep) { rep_add(im->bytes); g_rep_pn++; g_rep_pb += im->bytes; }
            else { g_cf_sec_b += im->bytes; if (pr) cf_frame_add(im->bytes); }
            g_cf_pn[k]++; g_cf_pb[k] += im->bytes; n[k]++; b0s += b0; b1s += im->bytes;
            if (g_cf_pn[k] <= 4) VLOG("coarse-first: provisional texture serial %llu %ux%u format %#x promoted %s (from level %u to %u, %.2f → %.2f MiB)", (unsigned long long)im->serial,
                                      im->d.width, im->d.height, im->d.format, k == CF_LOOK ? "on a lookup" : "in the background", ob, tgt,
                                      b0 / 1048576.0, im->bytes / 1048576.0);
        }
    }
out:
    if (left) atomic_store(&g_cf_any, 1);
    if ((n[CF_LOOK] || n[CF_BG]) && (nlog < 8 || !(nlog & 63))) VLOG("coarse-first: %d promoted on lookups, %d in the background (%.1f → %.1f MiB); in all %llu provisional (%.0f MiB), "
        "promoted on lookups %llu (%.0f MiB), on writes %llu (%.0f MiB), in the background %llu (%.0f MiB); MemAvailable %ld MiB", n[CF_LOOK], n[CF_BG], b0s / 1048576.0, b1s / 1048576.0,
        (unsigned long long)g_sc_prov_n, g_sc_prov_b / 1048576.0, (unsigned long long)g_cf_pn[CF_LOOK], g_cf_pb[CF_LOOK] / 1048576.0, (unsigned long long)g_cf_pn[CF_WRITE],
        g_cf_pb[CF_WRITE] / 1048576.0, (unsigned long long)g_cf_pn[CF_BG], g_cf_pb[CF_BG] / 1048576.0, g_md_avail);
    if (n[CF_LOOK] || n[CF_BG]) nlog++;
}
// ---------- Texture-use feedback (DM_VK_TEXFB): the texture IDs the GPU really indexes ----------
// The per-frame pool scan (vka_tex_pool_mark) names every texture ID a pool-reading draw's storage buffers hold: at Ocarina of Time's Kokiri Forest load +2,235 images
// (+665 MiB) in 5.3 s, most of them only in scene-wide material tables (CF1), and pool-reading dispatches name IDs no shader of theirs samples. A slot left empty that the
// GPU samples from a handle the scan never saw reads black (o4), so holding images back needs the GPU's own report. The library's DMSH_VK_F_TEXFB modules (vk_android_spv.py
// tex_feedback; dmsh_abi_vk.h DMSH_VK_POOL_FB_BINDING) set bit i of a 2^20-bit bitmap (set 1 binding 3, 128 KiB of device memory) before each pool access of texture ID i,
// when their bool constant DMSH_VK_SPEC_TEXFB is true (vka_pso.c: every such module's pipelines, fixed for the process). Each submission with a reporting draw ends with the
// bitmap's first ceil(registered IDs / 8) bytes copied into a host-visible ring slot (TFB_RING slots, sized for the registry, 4 KiB for 32,768 IDs) and the bitmap cleared
// (vka_tex_fb_commit, vka_exec.c); the next vka_tex_prepare after that submission's fence takes the slot (tfb_consume): per ID the second it was last indexed (g_tfb_last),
// per image the second and the readback count (tfb_s, tfb_sub), the serial in a direct-mapped set (g_tfb_ser). A draw with a stage that reads the texture pool and lacks
// the flag (an older library, a module the pass refused) is blind: it reports nothing, and while one was recorded within TFB_BLIND_S nothing is gated or released.
// DM_VK_TEXFB (restart: the constant is in the pipelines): 0 (default) off, nothing made or copied; 1 step 0, the reports logged only (the `memory:` line's texfb segment
// and a "texture-use feedback:" line every TFB_LOG_S: scan-named images whose ID the GPU indexed within 2 s of their creation (whole seconds: 2-3 s), IDs indexed a second);
// 2 the gate below under memory pressure (mip drop's MemAvailable test, mem_low); 3 the gate always (visual checks). With 2 or 3:
//   - gate: a reporting draw's scan (vka_tex_pool_mark fb; a draw that reads no texture of the pool counts as reporting) makes a deferred ID's image only when the GPU
//     indexed the ID within TFB_RECENT_S (10 s; whatever texture held it then), or the scan's word has the NVN2 handle form (a registered sampler ID in bits 20-31) and
//     the GPU indexed the ID's serial before (at any ID). Other IDs are held back: their slots show a 1x1 grey placeholder (alpha 1) of the registration's view type when it is a single-sampled float-class
//     colour one (DM_VK_TEXFB_PLACEHOLDER=0: the null descriptor, as a deferred slot), else null;
//   - any deferred ID the GPU indexed (held back, or named by no scan: a handle in a uniform buffer) is wanted from that readback on: write_tables makes its image at the
//     next submission's start, through the upload batch (one submission of placeholder, never a lasting hole);
//   - an index of a coarse-first provisional image asks for its promotion ("Coarse-first", within its caps);
//   - release: a named image the GPU never wrote and no image registration names, with no index of any of its IDs in DM_VK_TEXFB_IDLE_READBACKS (8) readbacks, in
//     DM_VK_TEXFB_IDLE_FRAMES (8) replay frames and in TFB_IDLE_S (2) s, no CPU lookup in those frames and seconds (used_f, used_s) and no blind draw in that window, is
//     let go as dead by image eviction (DM_VK_EVICT >= 1 with its demotion on: lazy images, seen stamps): its IDs back to deferred (placeholder), at most TFB_REL_MAX a
//     second. The window was 3 readbacks and 2 s (fb1, session-2026-11-07 fb/FB.md): at 0.1-1 fps one frame lasts longer than 2 s and one readback is about one frame, so
//     textures a draw binds or samples every frame went and 77 % of the 2,300 released were made again within 30 s (REL.md). A serial made again within 30 s of its
//     release is pinned for the session (g_tpin, a set that grows by one serial per wrong guess) and never released again unless the pressure is critical (MemAvailable
//     under half of DM_VK_MIPDROP_AVAIL_MB, mem_crit). A released serial does not take image eviction's dead pin (g_evd / g_evpin, which also keeps the overlap rule from
//     letting it go): a wrong release guess said nothing about its storage. Counted: made again within 30 s, and by the path that made it again (the `memory:` line)
// DM_VK_TEXFB_IDLE_S=<s> (live; default TFB_IDLE_S, 2): the release window's seconds under ordinary pressure; TFB_IDLE_S as before while MemAvailable is under
// DM_VK_MIPDROP_AVAIL_MB or the rate test fired within DM_VK_PRESS_RATE_S (tfb_fast: mem_crit's 512 MiB is under the phone's kill line).
// Ocarina of Time's play run (session-2026-11-07 play/PLAY.md section 2): 5,957 images released (1,358 MiB), 3,959 of them made again within 30 s (66 %): turning the
// camera let go what was behind it and turning back brought the pop-in. Its profile has 30: an image goes only when the GPU indexed none of its IDs for 30 s
// Memory: the bitmap 128 KiB (device), the ring 8 x 4 KiB, 5 bytes per registered ID, 96 KiB of serial sets (OoT, 32,768 IDs: about 0.41 MiB)
enum { TFB_RING = 8, TFB_BYTES = 1 << 17, TFB_RECENT_S = 10, TFB_BLIND_S = 2, TFB_IDLE_S = 2, TFB_SUBT = 64, TFB_LOG_S = 10 };   // (TFB_SUBT: the readbacks' seconds kept, pow2)
#define TFB_REL_MAX (64ull << 20)
enum { TFB_HELD = 1, TFB_PASS = 2, TFB_LATE = 4, TFB_PH = 8 };   // g_tfbs per ID: held back (counted), made by the gate's evidence, wanted by a readback, shows the placeholder
static uint32_t tfb_idle_s(int crit) { int v = crit ? TFB_IDLE_S : VKA_ENV("DM_VK_TEXFB_IDLE_S", TFB_IDLE_S); return v < 1 ? 1u : (uint32_t)v; }
static int tfb_mode(void) { static int m = -1; if (m < 0) { int v = VKA_ENV("DM_VK_TEXFB", 0); m = v < 0 ? 0 : v > 3 ? 3 : v; } return m; }
static _Atomic int g_tfb_ok;   // the bitmap is made and written into the set (bindless_init)
static _Atomic int g_tfb_any;   // (recording thread) the submission being recorded has a reporting draw
static _Atomic uint32_t g_tfb_blind_s;   // the coarse clock's second + 1 of the last blind draw (0: none)
static _Atomic uint64_t g_tfb_blind_n, g_tfb_rep_n;   // blind and reporting pool-reading draws
static uint32_t *g_tfb_last; static uint8_t *g_tfbs;   // (under g_tmu) per registered ID, sized with the registry (treg)
static uint64_t g_tfb_ser[4096];   // (under g_tmu) serials the GPU indexed, direct mapped by serial & 4095
static struct { uint64_t serial; uint32_t s; } g_tfbr[4096];   // (under g_tmu) serials the release rule let go, and when
static uint32_t g_tfb_nsub, g_tfb_subt[TFB_SUBT], g_tfb_sec, g_tfb_sec_n, g_tfb_ids_ps;   // (under g_tmu) readbacks taken (the second of the last TFB_SUBT); distinct IDs this second, the last whole second
static uint64_t g_tfb_skip, g_tfb_s0n[2], g_tfb_s0b[2], g_tfb_mkn, g_tfb_mkb, g_tfb_hn, g_tfb_hb, g_tfb_ln, g_tfb_lb, g_tfb_wn, g_tfb_rn, g_tfb_rb, g_tfb_rem, g_tfb_cfw;   // (under g_tmu)
static uint64_t g_tfb_pcn, g_tfb_pcb;   // (under g_tmu) pinned serials' images released under critical pressure
// Serials the release guessed wrong for (made again within 30 s of it): open addressing, 0 empty, grown at half load (16 bytes a pinned serial; fb1 had 1,766 such remakes)
static uint64_t *g_tpin; static uint32_t g_tpin_cap, g_tpin_n;   // (under g_tmu)
static uint32_t tpin_slot(uint64_t s, uint32_t cap) { return (uint32_t)((s * 0x9E3779B97F4A7C15ull) >> 32) & (cap - 1); }
static int tpin_has(uint64_t s) {
    if (!g_tpin_n || !s) return 0;
    for (uint32_t i = tpin_slot(s, g_tpin_cap);; i = (i + 1) & (g_tpin_cap - 1)) { if (g_tpin[i] == s) return 1; if (!g_tpin[i]) return 0; }
}
static void tpin_add(uint64_t s) {
    if (!s || tpin_has(s)) return;
    if ((g_tpin_n + 1) * 2 > g_tpin_cap) {   // (no memory: not pinned, the release may guess wrong for it once more)
        uint32_t nc = g_tpin_cap ? g_tpin_cap * 2 : 1024; uint64_t *n = calloc(nc, sizeof *n); if (!n) return;
        for (uint32_t i = 0; i < g_tpin_cap; i++) if (g_tpin[i]) { uint32_t j = tpin_slot(g_tpin[i], nc); while (n[j]) j = (j + 1) & (nc - 1); n[j] = g_tpin[i]; }
        free(g_tpin); g_tpin = n; g_tpin_cap = nc; }
    uint32_t j = tpin_slot(s, g_tpin_cap); while (g_tpin[j]) j = (j + 1) & (g_tpin_cap - 1);
    g_tpin[j] = s; g_tpin_n++;
}
// Mip drop while its pressure is on (md_levels_why): new images by the path that made them (binding, a table: scan-named / wanted before / all, other) and the reason they
// were not made without their finer levels, and under "flags" each excluding flag (all paths). Zero-shared images count at their allocation's size (under g_tmu)
enum { MDP_BIND, MDP_TABLE, MDP_OTHER, MDP_N };
static uint64_t g_mdr_n[MDP_N][MDR_N], g_mdr_b[MDP_N][MDR_N], g_mdf_n[CFF_N], g_mdf_b[CFF_N];
static void md_why_add(int r, uint64_t bytes, uint32_t xflags) {
    int p = g_mk_path == MK_BIND ? MDP_BIND : g_mk_path == MK_OTHER ? MDP_OTHER : MDP_TABLE;
    g_mdr_n[p][r]++; g_mdr_b[p][r] += bytes;
    if (r == MDR_FLAGS) for (int k = 0; k < CFF_N; k++) if (xflags & cff_bit[k]) { g_mdf_n[k]++; g_mdf_b[k] += bytes; }
}
static size_t md_why_paths(char *b, size_t n) {   // the `memory:` line's part ("" before any pressure)
    static const char *const pn[MDP_N] = {"binding", "table", "other"};
    size_t o = 0; int any = 0;
    for (int p = 0; p < MDP_N && o < n; p++) {
        uint64_t an = 0, ab = 0; for (int r = 0; r < MDR_N; r++) { an += g_mdr_n[p][r]; ab += g_mdr_b[p][r]; }
        if (!an) continue;
        o += (size_t)snprintf(b + o, n - o, "%s %s %llu (%.0f MiB):", any++ ? ";" : "; mip drop while on, new images by path and reason:", pn[p], (unsigned long long)an, ab / 1048576.0);
        for (int r = 0, k = 0; r < MDR_N && o < n; r++) if (g_mdr_n[p][r])
            o += (size_t)snprintf(b + o, n - o, "%s %s %llu (%.0f MiB)", k++ ? "," : "", mdr_name[r], (unsigned long long)g_mdr_n[p][r], g_mdr_b[p][r] / 1048576.0);
    }
    if (any && o < n) for (int k = 0, j = 0; k < CFF_N && o < n; k++) if (g_mdf_n[k])
        o += (size_t)snprintf(b + o, n - o, "%s %s %llu (%.0f MiB)", j++ ? "," : "; mip drop's flags by flag:", cff_name[k], (unsigned long long)g_mdf_n[k], g_mdf_b[k] / 1048576.0);
    return o < n ? o : n;
}
// ---------- Big targets (diagnostic, DM_VK_RTWATCH_MB) ----------
// A new image whose content is the GPU's (ST_ZERO: render targets, copy destinations) and whose memory (a zero-shared one's allocation) is at least DM_VK_RTWATCH_MB (live: images
// made from then on; 0, the default, none; Ocarina of Time 32) is watched (at most RTW_MAX at once): what reached it from its creation on (vka.h VKA_RTW_*: sampled, cleared to
// zero or another value, clears to zero left out for a zero-shared one, passes and draws with writes as an attachment, copies, shader-image writes). A `big target:` line when it
// gets memory of its own (img_own) and every 10 s while it lives. Ocarina of Time's 12288² DEPTH16 target (299 MiB, session-2026-11-07 fb3 serial 15193) was sampled by 26
// draws as zeros and got memory of its own at a depth clear; whether a draw ever writes it decides what can be done about it (SAVELOAD.md)
enum { RTW_MAX = 32 };
static vka_img *g_rtw[RTW_MAX]; static int g_nrtw;   // (under g_tmu)
static void rtw_new(vka_img *im, int mode) {   // caller holds g_tmu: a new image
    double mb = VKA_ENVF("DM_VK_RTWATCH_MB", 0.0);
    if (!(mb > 0) || mode != ST_ZERO || im->rtw || g_nrtw >= RTW_MAX) return;
    if ((double)(im->m.z || im->m.a ? im->m.size : im->bytes) < mb * 1048576.0) return;
    im->rtw = 1; im->rtw_ns = now_ns(); memset(im->rtw_n, 0, sizeof im->rtw_n); g_rtw[g_nrtw++] = im;
}
static void rtw_forget(vka_img *im) { if (im->rtw) for (int i = 0; i < g_nrtw; i++) if (g_rtw[i] == im) { g_rtw[i] = g_rtw[--g_nrtw]; break; } }   // caller holds g_tmu: let go
static void rtw_log(const vka_img *im, const char *when) {
    const uint32_t *n = im->rtw_n; float cv; memcpy(&cv, &im->rtw_clr, 4);
    VLOG("big target: serial %llu %ux%u layers %u levels %u format %#x (VkFormat %d) flags %#x, %.1f MiB%s, %s, %.0f s after its creation (frame %llu): bound for sampling %u, cleared to zero %u "
         "(left out, zero-shared kept: %u), cleared to another value %u (last %#x, as depth %g), passes as an attachment %u, draws there with writes on %u, copies or blits into it %u, "
         "shader-image bindings %u%s", (unsigned long long)im->serial, im->w, im->h, im->layers, im->levels, im->d.format, im->f.vk, im->d.flags, (im->m.z || im->m.a ? im->m.size : im->bytes) / 1048576.0,
         im->m.z ? " zero-shared" : im->m.a ? " guest-aliased" : "", when, (double)(now_ns() - im->rtw_ns) / 1e9, (unsigned long long)vka_frame(), n[VKA_RTW_SAMPLE], n[VKA_RTW_CLEAR0], n[VKA_RTW_ZKEEP],
         n[VKA_RTW_CLEARV], im->rtw_clr, (double)cv, n[VKA_RTW_PASS], n[VKA_RTW_DRAW], n[VKA_RTW_COPY], n[VKA_RTW_STORE], im->gpu_owned ? "; GPU-written" : "");
}
static void rtw_tick(void) {   // caller holds g_tmu (vka_tex_prepare): every 10 s, a line for each watched image
    static uint64_t last; uint64_t t = now_ns();
    if (!g_nrtw || t - last < 10000000000ull) return;
    last = t;
    for (int i = 0; i < g_nrtw; i++) rtw_log(g_rtw[i], "alive");
}
// ---------- Render target log (diagnostic, DM_VK_RTLOG_MB) ----------
// Does the game keep transient targets on the same guest storage (its pools aliasing them over time) while each gets memory of its own here? Ocarina of Time grew +485 MiB of
// images walking in Link's house (session-2026-11-07 fb4: 172 zero-shared targets adopted, 1578 MiB, +760 MiB of it while walking) and the per-image lines were capped at 32.
// DM_VK_RTLOG_MB=<MiB> (live; 0, the default, none) logs, for each image of at least that memory (an image's memory requirements; a zero-shared one's allocation):
//   `rtlog new` when vka_tex_get makes it, `rtlog own (...)` when a zero-shared one gets memory of its own (img_own), `rtlog gone` / `gone (evicted)` when its last reference
//   goes (at most RL_LINES such lines, the rest counted in the `rtlog live` line): serial, creation order, guest storage (address, size, pool, its Initialize serial, offset),
//   host size and kind, format, shape, NVN flags, Vulkan usage, content mode, the path that made it (a binding, a table, an attachment, a clear, a copy, other), its creation
//   frame and its last write / sampling binding / lookup / GPU index frames;
//   every 10 s `rtlog live`: the live ones, largest first (RL_ROWS rows), and `rtlog overlap`: the live images of at least 1 MiB grouped by overlapping guest storage (each group
//   two or more images), the groups' host bytes against the guest bytes they span, the RL_GROUPS largest groups' members, and the host bytes of images whose storage a newer
//   image's (made later) overlaps, of them the ones not used since that newer image was made. Only logs: nothing it counts changes what is made or kept
_Thread_local uint8_t vka_rl_site;
enum { RL_LINES = 4000, RL_ROWS = 256, RL_GROUPS = 10, RL_MEMB = 48, RL_PER_LINE = 12 };
static uint32_t g_rl_seq; static uint64_t g_rl_lines, g_rl_cut;   // (under g_tmu) images made so far; new / own / gone lines logged, and left out past RL_LINES
static const char *const rl_pn[8] = {"other", "binding", "table scan-named", "table wanted before", "table all", "attachment", "clear", "copy"};
static const char *const rl_mn[4] = {"GPU (cleared)", "filled once", "tracked", "sparse"};
static double rl_mb(void) { return VKA_ENVF("DM_VK_RTLOG_MB", 0.0); }
static uint64_t rl_size(const vka_img *im) { return im->m.size ? im->m.size : im->bytes; }
static const char *rl_kind(const vka_img *im) { return im->rs ? "sparse" : im->m.z ? "zero-shared" : im->m.a ? "guest-aliased" : im->m.blk ? "in a block" : im->m.dm ? "own" : "none"; }
static uint32_t rl_last(const vka_img *im) {   // the last frame anything used it: a write, a sampling binding, a lookup, a GPU index (texture-use feedback)
    uint32_t f = im->rl_wf; if (im->rl_sf > f) f = im->rl_sf; if (im->used_f > f) f = im->used_f; if (im->tfb_s && im->tfb_f > f) f = im->tfb_f; return f;
}
static void rl_log(const vka_img *im, const char *what) {   // caller holds g_tmu
    VLOG("rtlog %s: serial %llu #%u guest %#llx+%#llx (pool %p gen %llu off %#llx), host %.2f MiB %s, format %#x (VkFormat %d) %ux%ux%u levels %u (base %u) layers %u samples %u, "
         "NVN flags %#x, usage %#x, content %s, path %s, made frame %u; last write frame %u, sampling binding %u, lookup %u, GPU index %u%s; frame %llu",
         what, (unsigned long long)im->serial, im->rl_seq, (unsigned long long)(uintptr_t)im->mem, (unsigned long long)im->L.size, im->pool, (unsigned long long)im->pgen,
         (unsigned long long)im->rl_off, rl_size(im) / 1048576.0, rl_kind(im), im->d.format, im->f.vk, im->w, im->h, im->depth, im->levels, im->base, im->layers, im->samples,
         im->d.flags, im->usage, rl_mn[im->mode & 3], rl_pn[im->rl_path & 7], im->rl_mf, im->rl_wf, im->rl_sf, im->used_f, im->tfb_s ? im->tfb_f : 0, im->gpu_owned ? ", GPU-written" : "",
         (unsigned long long)vka_frame());
}
static void rl_note(const vka_img *im, const char *what) {   // caller holds g_tmu
    double mb = rl_mb();
    if (!(mb > 0) || (double)rl_size(im) < mb * 1048576.0) return;
    if (g_rl_lines >= RL_LINES) { g_rl_cut++; return; }
    g_rl_lines++; rl_log(im, what);
}
static void rl_new(vka_img *im, const tex_t *t) {   // caller holds g_tmu: a new image (vka_tex_get, after its first use)
    im->rl_seq = ++g_rl_seq; im->rl_mf = (uint32_t)vka_frame(); im->rl_off = (uint64_t)t->off;
    im->rl_path = (uint8_t)(vka_rl_site ? 4 + vka_rl_site : g_mk_path);
    rl_note(im, "new");
}
typedef struct { uint64_t lo, hi; vka_img *im; } rl_ent;
typedef struct { int i0, n; uint64_t host, zs, lo, hi; } rl_grp;
static int rl_by_size(const void *a, const void *b) { uint64_t x = rl_size(((const rl_ent *)a)->im), y = rl_size(((const rl_ent *)b)->im); return x < y ? 1 : x > y ? -1 : 0; }
static int rl_by_lo(const void *a, const void *b) { const rl_ent *x = a, *y = b; return x->lo < y->lo ? -1 : x->lo > y->lo ? 1 : x->hi < y->hi ? -1 : x->hi > y->hi; }
static uint64_t rl_host(const vka_img *im) { return im->m.z ? 0 : im->bytes; }   // host memory the image holds alone (a zero-shared one's allocation is shared: counted apart)
static void rl_live(uint64_t min) {   // caller holds g_tmu: the live images of at least min bytes, largest first
    rl_ent *e = malloc((size_t)(g_nlive ? g_nlive : 1) * sizeof *e); if (!e) { VRED_ONCE("rtlog: no memory for the live table"); return; }
    int n = 0, nk[3] = {0}; uint64_t bk[3] = {0};
    for (int i = 0; i < g_nlive; i++) { vka_img *im = g_live[i]; if (rl_size(im) < min) continue;
        int k = im->m.z || im->m.a ? 1 : im->m.dm ? 0 : 2; nk[k]++; bk[k] += rl_size(im); e[n++].im = im; }
    qsort(e, (size_t)n, sizeof *e, rl_by_size);
    VLOG("rtlog live: frame %llu, %d images of %.0f MiB or more: own %d (%.0f MiB), zero-shared or guest-aliased %d (%.0f MiB of allocations, shared), other %d (%.0f MiB); "
         "%llu new / own / gone lines logged, %llu left out (past %d)", (unsigned long long)vka_frame(), n, min / 1048576.0, nk[0], bk[0] / 1048576.0, nk[1], bk[1] / 1048576.0,
         nk[2], bk[2] / 1048576.0, (unsigned long long)g_rl_lines, (unsigned long long)g_rl_cut, RL_LINES);
    uint64_t rest = 0;
    for (int i = 0; i < n; i++) { const vka_img *im = e[i].im;
        if (i >= RL_ROWS) { rest += rl_size(im); continue; }
        VLOG("rtlog live: serial %llu #%u %ux%u layers %u levels %u VkFormat %d, %.2f MiB %s, guest %#llx+%#llx, %s, path %s, made frame %u, last write %u, sampling binding %u, lookup %u, GPU index %u",
             (unsigned long long)im->serial, im->rl_seq, im->w, im->h, im->layers, im->levels, im->f.vk, rl_size(im) / 1048576.0, rl_kind(im), (unsigned long long)(uintptr_t)im->mem,
             (unsigned long long)im->L.size, rl_mn[im->mode & 3], rl_pn[im->rl_path & 7], im->rl_mf, im->rl_wf, im->rl_sf, im->used_f, im->tfb_s ? im->tfb_f : 0); }
    if (n > RL_ROWS) VLOG("rtlog live: %d more (%.0f MiB) not listed", n - RL_ROWS, rest / 1048576.0);
    free(e);
}
static void rl_overlap(void) {   // caller holds g_tmu: the live images of at least 1 MiB grouped by overlapping guest storage
    rl_ent *e = malloc((size_t)(g_nlive ? g_nlive : 1) * sizeof *e); rl_grp *g = malloc((size_t)(g_nlive ? g_nlive : 1) * sizeof *g);
    uint8_t *old = calloc((size_t)(g_nlive ? g_nlive : 1), 1); uint32_t *nf = calloc((size_t)(g_nlive ? g_nlive : 1), sizeof *nf);   // old: a newer image's storage overlaps it; nf: the newest such one's creation frame
    if (!e || !g || !old || !nf) { free(e); free(g); free(old); free(nf); VRED_ONCE("rtlog: no memory for the overlap groups"); return; }
    int n = 0;
    for (int i = 0; i < g_nlive; i++) { vka_img *im = g_live[i];
        if (rl_size(im) < (1u << 20) || !im->mem || !im->L.size) continue;
        e[n].lo = (uint64_t)(uintptr_t)im->mem; e[n].hi = e[n].lo + im->L.size; e[n++].im = im; }
    qsort(e, (size_t)n, sizeof *e, rl_by_lo);
    // Pairs that overlap: j after i (lo_j >= lo_i) with lo_j < hi_i. The older of the two (creation order) is overlapped by a newer one. Each pair once; at most RL_PAIRS looked at
    enum { RL_PAIRS = 4 << 20 };
    uint64_t pairs = 0; int cut = 0;
    for (int i = 0; i < n && !cut; i++)
        for (int j = i + 1; j < n && e[j].lo < e[i].hi; j++) {
            if (++pairs > RL_PAIRS) { cut = 1; break; }
            int o = e[i].im->rl_seq < e[j].im->rl_seq ? i : j; uint32_t f = e[o == i ? j : i].im->rl_mf;
            old[o] = 1; if (f > nf[o]) nf[o] = f;
        }
    int ng = 0; uint64_t th = 0, tz = 0, ts = 0; int tm = 0;
    for (int i = 0; i < n;) {
        int j = i + 1; uint64_t hi = e[i].hi;
        while (j < n && e[j].lo < hi) { if (e[j].hi > hi) hi = e[j].hi; j++; }
        if (j - i >= 2) { rl_grp *x = &g[ng++]; x->i0 = i; x->n = j - i; x->lo = e[i].lo; x->hi = hi; x->host = x->zs = 0;
            for (int k = i; k < j; k++) { const vka_img *a = e[k].im; x->host += rl_host(a); if (a->m.z) x->zs += a->m.size;
                if (a->m.a) { int seen = 0; for (int q = i; q < k && !seen; q++) seen = e[q].im->m.a && e[q].im->m.ab == a->m.ab; if (!seen) x->zs += a->m.asz; } }   // (a guest-aliased allocation once)
            th += x->host; tz += x->zs; ts += hi - e[i].lo; tm += x->n; }
        i = j;
    }
    uint64_t ob = 0, oz = 0, ub = 0; int on = 0, un = 0;
    for (int i = 0; i < n; i++) if (old[i]) { on++; ob += rl_host(e[i].im); if (e[i].im->m.z) oz += e[i].im->m.size;
        if (rl_last(e[i].im) < nf[i]) { un++; ub += rl_host(e[i].im); } }
    VLOG("rtlog overlap: frame %llu, %d live images of 1 MiB or more with guest storage; %d groups of overlapping storage, %d images: host %.1f MiB of their own (+%.1f MiB zero-shared or guest-aliased, "
         "shared) over %.1f MiB of guest storage; overlapped by a newer image's storage: %d images, %.1f MiB of their own (+%.1f MiB zero-shared), of them not used since that "
         "newer image was made %d, %.1f MiB%s", (unsigned long long)vka_frame(), n, ng, tm, th / 1048576.0, tz / 1048576.0, ts / 1048576.0, on, ob / 1048576.0, oz / 1048576.0,
         un, ub / 1048576.0, cut ? " (pair scan cut short: a floor)" : "");
    for (int r = 0; r < RL_GROUPS && r < ng; r++) {   // the largest groups (host bytes with zero-shared ones', then images), each moved to the front in turn
        int b = r; for (int k = r + 1; k < ng; k++) if (g[k].host + g[k].zs > g[b].host + g[b].zs || (g[k].host + g[k].zs == g[b].host + g[b].zs && g[k].n > g[b].n)) b = k;
        rl_grp x = g[b]; g[b] = g[r]; g[r] = x;
        if (!x.host && !x.zs) break;
        VLOG("rtlog group %d: %d images, host %.1f MiB of their own (+%.1f MiB zero-shared or guest-aliased) over guest %#llx+%#llx (%.1f MiB), pool %p", r + 1, x.n, x.host / 1048576.0, x.zs / 1048576.0,
             (unsigned long long)x.lo, (unsigned long long)(x.hi - x.lo), (x.hi - x.lo) / 1048576.0, e[x.i0].im->pool);
        char ln[2048]; size_t o = 0;
        for (int k = 0; k < x.n && k < RL_MEMB; k++) {   // RL_PER_LINE members a line: serial, creation order, format, shape, host size and kind, storage (offset in the group, size), frames
            const rl_ent *q = &e[x.i0 + k]; const vka_img *im = q->im;
            if (!o) o = (size_t)snprintf(ln, sizeof ln, "rtlog group %d members:", r + 1);
            if (o < sizeof ln) o += (size_t)snprintf(ln + o, sizeof ln - o, "%s %llu #%u VkFormat %d %ux%u", k % RL_PER_LINE ? ";" : "", (unsigned long long)im->serial, im->rl_seq, im->f.vk, im->w, im->h);
            if (o < sizeof ln && im->layers > 1) o += (size_t)snprintf(ln + o, sizeof ln - o, "x%u", im->layers);
            if (o < sizeof ln) o += (size_t)snprintf(ln + o, sizeof ln - o, " %.1f MiB %s at +%#llx+%#llx made %u last %u%s", rl_size(im) / 1048576.0, rl_kind(im),
                                                     (unsigned long long)(q->lo - x.lo), (unsigned long long)(q->hi - q->lo), im->rl_mf, rl_last(im), old[x.i0 + k] ? " (a newer one overlaps it)" : "");
            if (k % RL_PER_LINE == RL_PER_LINE - 1 || k + 1 == x.n || k + 1 == RL_MEMB) { VLOG("%s", ln); o = 0; }
        }
        if (x.n > RL_MEMB) VLOG("rtlog group %d: %d more members not listed", r + 1, x.n - RL_MEMB);
    }
    free(e); free(g); free(old); free(nf);
}
static void rl_tick(void) {   // caller holds g_tmu (vka_tex_prepare): every 10 s while DM_VK_RTLOG_MB is on
    double mb = rl_mb(); static uint64_t last; uint64_t t = now_ns();
    if (!(mb > 0) || t - last < 10000000000ull) return;
    last = t;
    rl_live((uint64_t)(mb * 1048576.0));
    rl_overlap();
}
static _Thread_local int g_mk_tfb;   // write_tables: the slot's TFB_PASS / TFB_LATE, for vka_tex_get's counts
static void tfb_paths(char *b, size_t n);
void vka_tex_mem_paths(char *b, size_t n) {   // (read without the lock: a log line)
    snprintf(b, n, "; new images by path: table scan-named %llu (%.0f MiB), table wanted before %llu (%.0f MiB), table all %llu (%.0f MiB), binding %llu (%.0f MiB), other %llu (%.0f MiB)"
             "; mip drop: %llu made without finer levels (-%.0f MiB), %llu shrunk (-%.0f MiB), %llu made whole again, MemAvailable %ld MiB"
             "; scan-named by class: mipmapped compressed >=256 %llu (%.0f MiB), mipmapped other %llu (%.0f MiB), single-level %llu (%.0f MiB), under 64 KiB %llu (%.0f MiB)"
             "; scan-named by range: window <=256 KiB %llu (%.0f MiB), only tables %llu (%.0f MiB); IDs named by a window %llu, only by tables %llu"
             "; coarse-first: provisional %llu (%.0f MiB, %.0f MiB whole), promoted on lookups %llu (%.0f MiB), on writes %llu (%.0f MiB), in the background %llu (%.0f MiB); made whole while on: window-named %llu (%.0f MiB), not eligible %llu (%.0f MiB)",
             (unsigned long long)g_mk_n[MK_FRESH], g_mk_b[MK_FRESH] / 1048576.0, (unsigned long long)g_mk_n[MK_STICKY], g_mk_b[MK_STICKY] / 1048576.0,
             (unsigned long long)g_mk_n[MK_ALL], g_mk_b[MK_ALL] / 1048576.0, (unsigned long long)g_mk_n[MK_BIND], g_mk_b[MK_BIND] / 1048576.0,
             (unsigned long long)g_mk_n[MK_OTHER], g_mk_b[MK_OTHER] / 1048576.0, (unsigned long long)g_md_n, g_md_saved / 1048576.0, (unsigned long long)g_mds_n,
             g_mds_saved / 1048576.0, (unsigned long long)g_mdw_n, g_md_avail,
             (unsigned long long)g_sc_n[SC_MIPC], g_sc_b[SC_MIPC] / 1048576.0, (unsigned long long)g_sc_n[SC_MIPO], g_sc_b[SC_MIPO] / 1048576.0,
             (unsigned long long)g_sc_n[SC_ONE], g_sc_b[SC_ONE] / 1048576.0, (unsigned long long)g_sc_n[SC_SMALL], g_sc_b[SC_SMALL] / 1048576.0,
             (unsigned long long)g_nm_n[0], g_nm_kib[0] / 1024.0, (unsigned long long)g_nm_n[1], g_nm_kib[1] / 1024.0, (unsigned long long)g_nm_ids[0], (unsigned long long)g_nm_ids[1],
             (unsigned long long)g_sc_prov_n, g_sc_prov_b / 1048576.0, g_cf_full_b / 1048576.0, (unsigned long long)g_cf_pn[CF_LOOK], g_cf_pb[CF_LOOK] / 1048576.0,
             (unsigned long long)g_cf_pn[CF_WRITE], g_cf_pb[CF_WRITE] / 1048576.0, (unsigned long long)g_cf_pn[CF_BG], g_cf_pb[CF_BG] / 1048576.0,
             (unsigned long long)g_cf_wn, g_cf_wb / 1048576.0, (unsigned long long)g_cf_xn, g_cf_xb / 1048576.0);
    size_t o = strlen(b);
    if (o < n) o += (size_t)snprintf(b + o, n - o, " (by reason:");
    for (int i = 0, k = 0; i < CFW_N && o < n; i++) if (g_cfw_n[i]) o += (size_t)snprintf(b + o, n - o, "%s %s %llu (%.0f MiB)", k++ ? "," : "", cfw_name[i], (unsigned long long)g_cfw_n[i], g_cfw_b[i] / 1048576.0);
    if (o < n) o += (size_t)snprintf(b + o, n - o, ")");
    if (g_cfw_n[CFW_FLAGS] && o < n) {   // (an image under each excluding flag it has)
        o += (size_t)snprintf(b + o, n - o, " (flags by flag:");
        for (int i = 0, k = 0; i < CFF_N && o < n; i++) if (g_cff_n[i]) o += (size_t)snprintf(b + o, n - o, "%s %s %llu (%.0f MiB)", k++ ? "," : "", cff_name[i], (unsigned long long)g_cff_n[i], g_cff_b[i] / 1048576.0);
        if (o < n) o += (size_t)snprintf(b + o, n - o, ")"); }
    if (o < n) o += (size_t)snprintf(b + o, n - o, "; provisional from ST_FILL storage %llu (%.0f MiB), with the COMPRESSIBLE flag %llu (%.0f MiB)", (unsigned long long)g_cf_filln,
                                     g_cf_fillb / 1048576.0, (unsigned long long)g_cf_compn, g_cf_compb / 1048576.0);
    if (o < n && rep_on()) o += (size_t)snprintf(b + o, n - o, "; reported textures first (DM_VK_TEXFB_REPORT_MB %.0f): made at full size %llu (%.0f MiB), promoted %llu (%.0f MiB), put off by the frame budget %llu",
                                                 VKA_ENVF("DM_VK_TEXFB_REPORT_MB", 0.0), (unsigned long long)g_rep_n, g_rep_b / 1048576.0, (unsigned long long)g_rep_pn, g_rep_pb / 1048576.0, (unsigned long long)g_rep_over);
    if (o < n && VKA_ENVF("DM_VK_MIPDROP_RESTORE_MB", 0.0) > 0) o += (size_t)snprintf(b + o, n - o, "; mip restore (DM_VK_MIPDROP_RESTORE_MB %.0f): %llu made whole again after the pressure (%.0f → %.0f MiB)",
                                                 VKA_ENVF("DM_VK_MIPDROP_RESTORE_MB", 0.0), (unsigned long long)g_mrs_n, g_mrs_b0 / 1048576.0, g_mrs_b1 / 1048576.0);
    if (o < n && tfb_mode() >= 2) {   // ("Texture-use feedback": made again by path; zero-shared ones count 0 bytes above until their first write)
        static const char *const pn[MK_N] = {"other", "binding", "table scan-named", "table wanted before", "table all"};
        o += (size_t)snprintf(b + o, n - o, "; made again after an eviction / after a feedback release (within 30 s) by path:");
        for (int i = 0; i < MK_N && o < n; i++) o += (size_t)snprintf(b + o, n - o, "%s %s %llu (%.0f MiB) / %llu (%.0f MiB, %llu)", i ? "," : "", pn[i], (unsigned long long)g_mk_an[i],
                                                           g_mk_ab[i] / 1048576.0, (unsigned long long)g_mk_rn[i], g_mk_rb[i] / 1048576.0, (unsigned long long)g_mk_r30[i]);
        if (o < n) o += (size_t)snprintf(b + o, n - o, "; first made %llu, zero-shared at creation by path:", (unsigned long long)(g_mk_n[0] + g_mk_n[1] + g_mk_n[2] + g_mk_n[3] + g_mk_n[4]
                                                       - g_mk_an[0] - g_mk_an[1] - g_mk_an[2] - g_mk_an[3] - g_mk_an[4]));
        for (int i = 0; i < MK_N && o < n; i++) if (g_mk_zn[i]) o += (size_t)snprintf(b + o, n - o, " %s %llu (%.0f MiB once written)", pn[i], (unsigned long long)g_mk_zn[i], g_mk_zb[i] / 1048576.0);
        if (o < n) o += (size_t)snprintf(b + o, n - o, "; zero-shared images given memory of their own %llu (%.0f MiB; the allocation adopted %llu, %.0f MiB); clears to zero left out %llu",
                                         (unsigned long long)g_zown, g_zown_b / 1048576.0, (unsigned long long)g_zadopt, g_zadopt_b / 1048576.0, (unsigned long long)g_zclear);
    }
    if (o < n && g_ra_n) { char a[200]; vka_imem_astats(a, sizeof a);
        o += (size_t)snprintf(b + o, n - o, "%s (made %llu, %.0f MiB alone; %llu first uses without a clear)", a, (unsigned long long)g_ra_n, g_ra_b / 1048576.0, (unsigned long long)g_ra_noclr); }
    if (o < n) o += md_why_paths(b + o, n - o);
    if (o < n) tfb_paths(b + o, n - o);
}
vka_img *vka_tex_get(const tex_t *t, VkCommandBuffer cb) {
    if (!t || !t->serial || !vka_ok()) return NULL;
    pthread_mutex_lock(&g_tmu);
    smap_ent *e = map_get(t->serial);
    if (e && !g_look_sample && is_dalias(t, e->im)) { dalias_end(e, e->im, t); e = NULL; }
    if (e) { vka_img *im = e->im; if (im) { img_used(im); cf_look(im); } pthread_mutex_unlock(&g_tmu); return im; }
    vka_fmt f; vka_img *im = NULL; int throttle = 0;
    if (vka_fmt_of(t->d.format, &f)) {
        // (not an image whose pool was finalized: a new pool over the same memory has new content. Shared with a texture of another live pool, the image follows the newest one's)
        for (int i = 0; i < g_nlive && !im; i++) if (t->mem && same_shape(g_live[i], t, &f, 1) && !img_pool_dead(g_live[i]) && !(g_live[i]->base && sparse_gpu(t, &f))) {   // (drawn into at level 0)
            im = g_live[i]; im->refs++;
            if (t->pool && im->pool != t->pool) { uint64_t fin = nvn_pool_fin(), g = nvn_pool_live(t->pool, NULL, NULL); if (g) { im->pool = t->pool; im->pfin = fin; im->pgen = g; } } }
        if (!im) im = depth_alias(t, &f);   // (a colour texture over a depth image's storage, sampled: "Colour textures over a depth image")
        if (im) big_share(im, t);   // (another texture object of the same storage and shape: DM_VK_BIGIMG_MB diagnostic)
        uint32_t vmin = 0, base = sparse_sampled(t, &f) ? sparse_base(t->pool, t->mem, &t->d, &t->L, &vmin) : 0;
        int mode = 0;
        uint32_t drop = 0, cfk = 0;
        if (!im) { mode = first_mode(t, &f, cpuw_take(t->serial, 0));
                   int spp = t->pool && pool_sparse(t->pool), st = md_content(mode) && !base && !spp, win = 0, rep = 0, why = -1, mdr = -1;
                   if (md_pressure()) {   // ("Mip drop"; mdr: why not, the `memory:` line's "mip drop while on")
                       if (!st) mdr = mode == ST_ZERO ? MDR_ZERO : mode == ST_FILL ? MDR_FILL : MDR_SPARSE;
                       else drop = md_levels_why(&t->d, &f, &mdr); }
                   int cfon = (g_mk_path == MK_FRESH || g_mk_path == MK_STICKY) && cf_on();   // ("Coarse-first": a scan-named texture under pressure)
                   int cfill = mode == ST_FILL && VKA_ENV("DM_VK_SCAN_COARSE_FILL", 1), stc = (mode == ST_TRACK || cfill) && !base && !spp;   // (ST_FILL: "Coarse-first")
                   if (cfon && !stc) why = mode == ST_FILL ? CFW_FILL : mode == ST_ZERO ? CFW_ZERO : mode == ST_SPARSE ? CFW_SPARSE : base ? CFW_SPBASE : CFW_SPPOOL;
                   if (cfon && stc && (cfk = cf_levels(t, &f, &why)) && cfk <= drop) { cfk = 0; why = CFW_DROP; }
                   if (cfk && (g_mk_tfb & (TFB_PASS | TFB_LATE)) && rep_on() && rep_fits(dense_bytes(&t->d, &t->L, drop))) { cfk = 0; rep = 1; }   // (the GPU indexed it: "Reported textures first")
                   if (cfk && g_tnamed && g_mk_id > 0 && g_mk_id < (1 << 20) && (g_tnamed[g_mk_id] & 1) && cf_frame_fits(dense_bytes(&t->d, &t->L, drop))) { cfk = 0; win = 1; }   // (a draw's window named it)
                   g_zwant = mode == ST_ZERO; im = img_create(t, &f, cfk ? cfk : drop ? drop : base); g_zwant = 0;
                   if (im && cfk && im->base == cfk) { uint64_t whole = dense_bytes(&t->d, &t->L, 0);
                       im->mdrop = (uint8_t)cfk; im->prov = 1; g_sc_prov_n++; g_sc_prov_b += im->bytes; g_cf_full_b += whole;
                       if (mode == ST_FILL) { g_cf_filln++; g_cf_fillb += im->bytes; }
                       if (t->d.flags & NVN_TEXTURE_FLAGS_COMPRESSIBLE) { g_cf_compn++; g_cf_compb += im->bytes; }
                       if (g_sc_prov_n <= 8) VLOG("coarse-first: scan-named texture serial %llu %ux%u levels %u format %#x made provisional from level %u (%ux%u, %.3f MiB instead of %.2f; MemAvailable %ld MiB)",
                                                  (unsigned long long)t->serial, t->d.width, t->d.height, t->d.levels, t->d.format, cfk, im->w, im->h, im->bytes / 1048576.0, whole / 1048576.0, g_md_avail); }
                   else if (im && rep) { rep_add(im->bytes); g_rep_n++; g_rep_b += im->bytes; }   // (its own budget, not coarse-first's)
                   else if (im && cfon) {   // made at full size (or mip drop's) while coarse-first is on: the frame budget counts it
                       cf_frame_add(im->bytes);
                       if (win) { g_cf_wn++; g_cf_wb += im->bytes; }
                       else { g_cf_xn++; g_cf_xb += im->bytes; if (why < 0) why = CFW_BASE; g_cfw_n[why]++; g_cfw_b[why] += im->bytes;
                              if (why == CFW_FLAGS) { uint32_t xf = t->d.flags & cf_xflags(); for (int k = 0; k < CFF_N; k++) if (xf & cff_bit[k]) { g_cff_n[k]++; g_cff_b[k] += im->bytes; } } } }
                   if (im) cf_note_new();
                   if (im && mdr >= 0) md_why_add(im->prov ? MDR_PROV : mdr == MDR_DROP && im->base != drop ? MDR_BASE : mdr, im->m.z || im->m.a ? im->m.size : im->bytes, t->d.flags & md_xflags(&f));
                   if (im) rtw_new(im, mode);   // ("Big targets")
                   if (im && !im->prov && drop && im->base == drop) { uint64_t whole = dense_bytes(&t->d, &t->L, 0), part = dense_bytes(&t->d, &t->L, drop);
                       im->mdrop = (uint8_t)drop; g_md_n++; g_md_saved += whole - part;
                       if (g_md_n <= 8) VLOG("mip drop: texture serial %llu %ux%u levels %u format %#x made from level %u (%.2f MiB of storage instead of %.2f; MemAvailable %ld MiB)",
                                             (unsigned long long)t->serial, t->d.width, t->d.height, t->d.levels, t->d.format, drop, part / 1048576.0, whole / 1048576.0, g_md_avail); }
                   if (im) { g_mk_n[g_mk_path]++; g_mk_b[g_mk_path] += im->bytes; if (g_mk_path == MK_FRESH) sc_count(g_mk_id, &t->d, &f, im->bytes, im->prov ? dense_bytes(&t->d, &t->L, 0) : im->bytes);
                             im->tfb_sub = g_tfb_nsub; im->tfb_f = (uint32_t)vka_frame();   // ("Texture-use feedback": its unindexed readbacks and frames count from here)
                             { uint64_t s = t->serial; int k = g_mk_path;   // (made again: before ev_remade / tfb_remade forget the serial)
                               if (g_evs[s & 4095] == s) { g_mk_an[k]++; g_mk_ab[k] += im->bytes; }
                               if (g_tfbr[s & 4095].serial == s) { g_mk_rn[k]++; g_mk_rb[k] += im->bytes; g_mk_r30[k] += g_now_s - g_tfbr[s & 4095].s <= 30; }
                               if (im->m.z) { g_mk_zn[k]++; g_mk_zb[k] += im->m.size; } }
                             if ((g_mk_path == MK_FRESH || g_mk_path == MK_STICKY) && tfb_mode() && atomic_load(&g_tfb_ok) && vka_shader_texfb(NULL)) im->tfb_st = 1;
                             if (g_mk_tfb & TFB_PASS) { g_tfb_mkn++; g_tfb_mkb += im->bytes; }
                             if (g_mk_tfb & TFB_LATE) { g_tfb_ln++; g_tfb_lb += im->bytes; } } }
        if (im && !im->inited) {
            im->vmin = vmin;
            if (g_nlive == g_livecap) { g_livecap = g_livecap ? g_livecap * 2 : 256; g_live = realloc(g_live, (size_t)g_livecap * sizeof *g_live); }
            g_live[g_nlive++] = im;
            im->made_s = g_now_s;
            ev_remade(t->serial);
            big_note(im, t);
            // Its first content through the upload batch when cb is the replay's (vka_dev.c "Upload batches"): staging that retires on the batch's own completion
            VkCommandBuffer ucb = cb && cb == vka_rp_cb() && vka_up_on() ? vka_up_begin() : VK_NULL_HANDLE;
            img_first_use(im, t, ucb ? ucb : cb, mode);
            if (ucb) { vka_up_point(vka_up_cur(ucb)); vka_up_end(); }   // (an upload of many pieces, upload_mt, checks the batch budget only here)
            throttle = ucb != VK_NULL_HANDLE;
            rl_new(im, t);   // ("Render target log")
        }
    }
    if (im) { img_used(im); cf_look(im); }
    map_put(t->serial, im);
    pthread_mutex_unlock(&g_tmu);
    if (throttle) vka_up_throttle();
    return im;
}
vka_img *vka_tex_acquire(uint64_t serial) {
    pthread_mutex_lock(&g_tmu); smap_ent *e = map_get(serial); vka_img *im = e ? e->im : NULL; if (im) { im->refs++; im->holds++; } pthread_mutex_unlock(&g_tmu); return im;
}
// References to the live images (at most max), for the replayer's dumps; each dropped with vka_tex_drop. Returns how many were taken
int vka_tex_live_refs(vka_img **out, int max) {
    pthread_mutex_lock(&g_tmu); int n = 0;
    for (int i = 0; i < g_nlive && n < max; i++) { vka_img *im = g_live[i]; im->refs++; im->holds++; out[n++] = im; }
    pthread_mutex_unlock(&g_tmu); return n;
}
// (an image made by its holder, frame generation's outputs, starts with a reference and no hold: its last drop finds holds at 0)
void vka_tex_drop(vka_img *im) {
    if (!im) return;
    pthread_mutex_lock(&g_tmu);
    if (im->holds > 0) im->holds--;
    if (!im->holds && im->zold) { vka_after(vka_fence_last() + 1, img_destroy, im->zold); im->zold = NULL; }   // (the zero-shared image it left while held: img_own)
    unref_locked(im);
    pthread_mutex_unlock(&g_tmu);
}
vka_img *vka_tex_ref(vka_img *im) { if (im) { pthread_mutex_lock(&g_tmu); im->refs++; im->holds++; pthread_mutex_unlock(&g_tmu); } return im; }

// A view asked in a format outside the image's format list (img_create): the pair is learned for the images made from now on (Learned view formats) and logged; then says
// what the asker does instead (vka_view: a view in the image's own format; id_view: the binding reads as unbound). Caller holds g_tmu
static void view_unlisted(const vka_img *im, VkFormat fmt, const char *then) {
    int learned = 0;   // (only uncompressed colour formats of one texel size, float or integer: vl_learn checks the pair)
    if (im->f.kind < 3 && !compressed(&im->f)) { pthread_mutex_lock(&g_vlmu); learned = vl_learn(im->f.vk, fmt, im->w, im->h); pthread_mutex_unlock(&g_vlmu); }
    if (learned)
        VLOG("image VkFormat %d (NVN %#x, %ux%u) asked for a view as VkFormat %d, which its format list does not have: %s; learned (images of this format and size made from now on, and in the next runs, take it)", im->f.vk, im->d.format, im->w, im->h, fmt, then);
    else { static struct { VkFormat a, b; } seen[32]; static int ns; int k = 0;
        while (k < ns && !(seen[k].a == im->f.vk && seen[k].b == fmt)) k++;
        if (k == ns && ns < 32) { seen[ns].a = im->f.vk; seen[ns++].b = fmt;
            VRED("image VkFormat %d (NVN %#x, %ux%u) asked for a view as VkFormat %d, which its format list does not have: %s%s", im->f.vk, im->d.format, im->w, im->h, fmt, then,
                 VKA_ENV("DM_VK_VIEW_LEARN", 1) ? " (an image made before the pair was learned, or the table is full)" : " (DM_VK_VIEW_LEARN=0)"); } }
}
// The usage a view in format f takes of an image made with usage u: each use whose format feature f lacks left out (a view in another format than the image's has the
// view format's features: VUID-VkImageViewCreateInfo-usage-02274 to 02277). An sRGB view of an image with storage use leaves storage out (sRGB formats have no storage
// support), an E5B9G9R9 one also colour attachment use; an R32_UINT view of a B10G11R11 or RGBA8 image keeps them both, so a pass writes the image as words through it.
// Every view in another format left storage out before, so such a view bound as a shader image was one without storage use
static VkImageUsageFlags view_usage(VkImageUsageFlags u, VkFormat f) {
    VkFormatFeatureFlags ff = feats(f);
    if (!(ff & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) u &= ~(VkImageUsageFlags)VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!(ff & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) u &= ~(VkImageUsageFlags)VK_IMAGE_USAGE_STORAGE_BIT;
    if (!(ff & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) u &= ~(VkImageUsageFlags)VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (!(ff & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) u &= ~(VkImageUsageFlags)VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    return u;   // (the transfer uses stay: never 0)
}
VkImageView vka_view(vka_img *im, VkFormat fmt, VkImageViewType type, VkImageAspectFlags asp, uint32_t level, uint32_t nlevels, uint32_t layer, uint32_t nlayers, const VkComponentMapping *swz) {
    VkComponentMapping id = {0}; if (!swz) swz = &id;
    pthread_mutex_lock(&g_tmu);
    if (!img_lists(im, fmt)) { view_unlisted(im, fmt, "viewed in its own format"); fmt = im->f.vk; }   // (outside the format list: its own format instead)
    // Moved to the front when found: an image shared by many textures (one storage, many views) has a long list, and the draws ask for the same few views again and again
    for (view_ent **pv = &VIEWS(im), *v; (v = *pv); pv = &v->next)
        if (v->f == fmt && v->t == type && v->asp == asp && v->l0 == level && v->nl == nlevels && v->a0 == layer && v->na == nlayers && !memcmp(&v->swz, swz, sizeof *swz)) {
            if (pv != &VIEWS(im)) { *pv = v->next; v->next = VIEWS(im); VIEWS(im) = v; }
            pthread_mutex_unlock(&g_tmu); return v->v; }
    // A view in another format: the uses its format takes (view_usage)
    VkImageViewUsageCreateInfo vu = {VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO, .usage = view_usage(im->usage, fmt)};   // (own format too: an EXTENDED_USAGE image's own format may lack its storage use
    VkImageViewCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = vu.usage != im->usage ? &vu : NULL, .image = im->img, .viewType = type, .format = fmt, .components = *swz, .subresourceRange = {asp, level, nlevels, layer, nlayers}};
    VkImageView view = VK_NULL_HANDLE; VkResult r = vk.vkCreateImageView(V.dev, &ci, NULL, &view);
    if (r) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VRED("image view of serial %llu format %d type %d levels %u+%u layers %u+%u: %d", (unsigned long long)im->serial, fmt, type, level, nlevels, layer, nlayers, r); }
    else { view_ent *v = malloc(sizeof *v); *v = (view_ent){VIEWS(im), view, fmt, type, asp, level, nlevels, layer, nlayers, *swz}; VIEWS(im) = v; }
    pthread_mutex_unlock(&g_tmu);
    return view;
}

// Guest storage → image: every level and layer deswizzled (block-linear) or copied by rows (LINEAR, pitch d.stride) into scratch memory, then one copy per piece
static void upload_(vka_img *im, const tex_t *t, VkCommandBuffer cb, uint64_t *nbytes);
// Large block-linear uploads deswizzled on several threads (DM_VK_UPLOAD_MT, live: helper threads, default 3, 0 = on the calling thread only): a texture's pieces (each level
// and layer) are cut into runs of block rows (nvn_deswizzle_rows, at least ~64 KB each) that the helpers and the calling thread take in turn; the copies are recorded after
// all of them, in the same order as before. Only uploads of at least DM_VK_UPLOAD_MT_MIN bytes (default 256 KB): the first uploads of a level's textures came in loading
// frames, 1-93 textures a frame, all deswizzled on the replay thread one after another
typedef struct { const nvn_texdesc *d; const uint8_t *src; uint8_t *dst; size_t pitch; uint32_t lv, ly, y0, y1; _Atomic int *bad; } up_part;   // bad: its piece's flag
// A part is claimed by compare-and-swap on next while next < n: a helper that wakes late, or comes back from the last batch, can only claim a part of the batch being set up
// once it is complete (p, then next = 0, then n; n = 0 between batches)
static struct { pthread_mutex_t mu; pthread_cond_t cv, done_cv; _Atomic(up_part *) p; _Atomic uint32_t n, next, done; _Atomic int fail; uint64_t gen; int nth; } UP = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, PTHREAD_COND_INITIALIZER};
static pthread_mutex_t g_up_one = PTHREAD_MUTEX_INITIALIZER;   // one parallel upload at a time (another thread's upload runs on its own thread)
static void up_run(void) {   // take parts until none is left
    for (;;) {
        uint32_t i = atomic_load(&UP.next), n = atomic_load(&UP.n);
        if (i >= n) return;
        if (!atomic_compare_exchange_weak(&UP.next, &i, i + 1)) continue;
        up_part *q = &atomic_load(&UP.p)[i];
        if (nvn_deswizzle_rows(q->d, q->lv, q->ly, q->src, q->dst, q->pitch, q->y0, q->y1)) { atomic_store(&UP.fail, 1); atomic_store(q->bad, 1); }
        if (atomic_fetch_add(&UP.done, 1) + 1 == n) { pthread_mutex_lock(&UP.mu); pthread_cond_broadcast(&UP.done_cv); pthread_mutex_unlock(&UP.mu); }
    }
}
static void *up_main(void *a) {
    (void)a; pthread_setname_np(pthread_self(), "vk-upload");
    uint64_t seen = 0;
    for (;;) {
        pthread_mutex_lock(&UP.mu); while (UP.gen == seen) pthread_cond_wait(&UP.cv, &UP.mu); seen = UP.gen; pthread_mutex_unlock(&UP.mu);
        up_run();
    }
    return NULL;
}
// 1 = done in parallel (fail: a piece could not be deswizzled); 0 = not here (the caller does it piece by piece)
static int up_parallel(up_part *p, uint32_t n, int *fail) {
    int want = VKA_ENV("DM_VK_UPLOAD_MT", 3); if (want <= 0 || n < 2) return 0;
    if (want > 6) want = 6;
    if (pthread_mutex_trylock(&g_up_one)) return 0;
    while (UP.nth < want) { pthread_t th; if (pthread_create(&th, NULL, up_main, NULL)) break; pthread_detach(th); UP.nth++; }
    if (!UP.nth) { pthread_mutex_unlock(&g_up_one); return 0; }
    pthread_mutex_lock(&UP.mu);
    atomic_store(&UP.fail, 0); atomic_store(&UP.done, 0); atomic_store(&UP.p, p); atomic_store(&UP.next, 0); atomic_store(&UP.n, n); UP.gen++;
    pthread_cond_broadcast(&UP.cv);
    pthread_mutex_unlock(&UP.mu);
    up_run();   // this thread takes parts too
    pthread_mutex_lock(&UP.mu); while (atomic_load(&UP.done) < n) pthread_cond_wait(&UP.done_cv, &UP.mu); atomic_store(&UP.n, 0); pthread_mutex_unlock(&UP.mu);
    *fail = atomic_load(&UP.fail);
    pthread_mutex_unlock(&g_up_one);
    return 1;
}
// The block-linear pieces of a texture of at least DM_VK_UPLOAD_MT_MIN bytes, deswizzled in parallel into their scratch spans and copied; 0 = not taken (the serial path does it)
static int upload_mt(vka_img *im, const tex_t *t, VkCommandBuffer cb, uint64_t *nbytes) {
    const nvn_texdesc *d = &t->d; const vka_fmt *f = &im->f;
    int is3d = d->target == NVN_TEXTURE_TARGET_3D;
    if (vka_nan_on() || (d->flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) || f->kind >= 3) return 0;
    uint64_t total = 0; uint32_t npieces = im->levels * im->layers;
    for (uint32_t lv = 0; lv < im->levels; lv++) {
        uint32_t w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1, z = is3d ? (im->depth >> lv ? im->depth >> lv : 1) : 1;
        total += (uint64_t)((w + f->bw - 1) / f->bw) * f->bpb * ((h + f->bh - 1) / f->bh) * z * im->layers;
    }
    int minb = VKA_ENV("DM_VK_UPLOAD_MT_MIN", 256 << 10);
    if (total < (uint64_t)(minb > 0 ? minb : 1) || npieces > 4096 || VKA_ENV("DM_VK_UPLOAD_MT", 3) <= 0) return 0;
    typedef struct { vka_span s; uint32_t lv, ly, w, h, z, bx, by; size_t pitch; _Atomic int bad; } piece_t;
    piece_t *pc = malloc(npieces * sizeof *pc); up_part *parts = NULL; uint32_t np = 0, cap = 0;
    if (!pc) return 0;
    uint32_t k = 0;
    for (uint32_t lv = 0; lv < im->levels; lv++) {
        uint32_t w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1, z = is3d ? (im->depth >> lv ? im->depth >> lv : 1) : 1;
        uint32_t bx = (w + f->bw - 1) / f->bw, by = (h + f->bh - 1) / f->bh; size_t pitch = (size_t)bx * f->bpb, piece = pitch * by * z;
        int rows = nvn_deswizzle_block_rows(d, lv + im->base);   // (texture level: the image's level 0 is the texture's level base, "Mip drop")
        if (rows < 0 || (uint32_t)rows != by) { free(pc); free(parts); return 0; }   // (a shape the serial path reports)
        for (uint32_t ly = 0; ly < im->layers; ly++, k++) {
            piece_t *q = &pc[k]; *q = (piece_t){{0}, lv, ly, w, h, z, bx, by, pitch, 0};
            if (!vka_scratch(piece, 16, &q->s)) goto cut;   // (no scratch: as the serial path, the pieces before it are uploaded)
            // Runs of block rows of at least ~64 KB (all slices of a 3D level in each)
            uint32_t per = (uint32_t)((65536 + pitch * z - 1) / (pitch * z)); if (!per) per = 1;
            for (uint32_t y = 0; y < by; y += per) {
                if (np == cap) { cap = cap ? cap * 2 : 64; up_part *n2 = realloc(parts, cap * sizeof *parts); if (!n2) { free(pc); free(parts); return 0; } parts = n2; }
                parts[np++] = (up_part){d, t->mem, q->s.ptr, pitch, lv + im->base, ly, y, y + per < by ? y + per : by, &q->bad};
            }
        }
    }
cut:;
    int fail = 0;
    if (!up_parallel(parts, np, &fail)) { for (uint32_t i = 0; i < np; i++) if (nvn_deswizzle_rows(parts[i].d, parts[i].lv, parts[i].ly, parts[i].src, parts[i].dst, parts[i].pitch, parts[i].y0, parts[i].y1)) { fail = 1; *parts[i].bad = 1; } }
    free(parts);
    // A piece that could not be deswizzled ends the upload there, as on the serial path: the pieces before it are copied, it and those after it are not
    if (fail) VRED_ONCE("texture serial %llu format %#x target %d: no CPU deswizzle", (unsigned long long)t->serial, d->format, d->target);
    uint64_t bytes = 0;
    for (uint32_t i = 0; i < k; i++) {
        piece_t *q = &pc[i];
        if (atomic_load(&q->bad)) break;
        VkBufferImageCopy c = {q->s.off, q->bx * f->bw, q->by * f->bh, {im->aspect, q->lv, q->ly, 1}, {0, 0, 0}, {q->w, q->h, q->z}};
        vk.vkCmdCopyBufferToImage(cb, q->s.buf, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
        bytes += q->pitch * q->by * q->z;
    }
    free(pc);
    *nbytes = bytes;
    return 1;
}
static int img_own(vka_img *im, VkCommandBuffer cb);
static void upload(vka_img *im, const tex_t *t, VkCommandBuffer cb) {   // (callers hold g_tmu)
    if (im->m.z) img_own(im, cb);   // a CPU write into a zero-shared image's storage (vka_tex_prepare): memory of its own before the upload
    uint64_t t0 = now_ns(), nb = 0; upload_(im, t, cb, &nb);
    atomic_fetch_add_explicit(&g_up_ns, now_ns() - t0, memory_order_relaxed); atomic_fetch_add_explicit(&g_up_bytes, nb, memory_order_relaxed);
}
static void upload_(vka_img *im, const tex_t *t, VkCommandBuffer cb, uint64_t *nbytes) {
    if (!im->vpool && upload_mt(im, t, cb, nbytes)) return;   // (a sparse pool's storage is read through its page table below, never at its address)
    const nvn_texdesc *d = &t->d; const vka_fmt *f = &im->f;
    int linear = d->flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET), is3d = d->target == NVN_TEXTURE_TARGET_3D;
    if (f->kind >= 3) { VRED_ONCE("depth / stencil textures are not uploaded from guest memory yet (format %#x)", d->format); return; }
    if (linear && (im->levels > 1 || im->layers > 1 || is3d)) { VRED_ONCE("linear texture with levels or layers (format %#x %ux%u): only level 0 uploaded", d->format, d->width, d->height); }
    uint64_t bytes = 0;
    // A sparse pool's texture: gathered from its page table by bands, unmapped ones as zero texels (until 2026-11-03 a copy of its whole range was made first)
    if (im->vpool) { if (!im->vbind) upload_sparse_parts(im, cb, 0, 0, 1, 0); return; }   // (a GPU-written one's content is the GPU's: sparse_bind)
    uint8_t *part = NULL; const uint8_t *src = t->mem;
    for (uint32_t lv = 0; lv < (linear ? 1 : im->levels); lv++) {
        uint32_t w = im->w >> lv ? im->w >> lv : 1, h = im->h >> lv ? im->h >> lv : 1, z = is3d ? (im->depth >> lv ? im->depth >> lv : 1) : 1;
        uint32_t bx = (w + f->bw - 1) / f->bw, by = (h + f->bh - 1) / f->bh;
        size_t pitch = (size_t)bx * f->bpb, piece = pitch * by * z;
        for (uint32_t ly = 0; ly < (linear ? 1 : im->layers); ly++) {
            vka_span s; if (!vka_scratch(piece, 16, &s)) { free(part); return; }
            if (linear) { size_t st = d->stride ? d->stride : pitch; for (uint32_t y = 0; y < by; y++) memcpy(s.ptr + y * pitch, src + y * st, pitch); }
            // (texture level lv + base: "Mip drop"; a sparse pool's image returned above)
            else if (nvn_deswizzle(d, lv + im->base, ly, src, s.ptr, pitch)) { VRED_ONCE("texture serial %llu format %#x target %d: no CPU deswizzle", (unsigned long long)t->serial, d->format, d->target); free(part); return; }
            if (d->format >= 0x79 && d->format <= 0x94 && vka_nan_on()) {   // DM_VK_NANSCAN: all-zero blocks of ASTC textures (an all-zero ASTC block is an error block)
                uint64_t zb = 0, nb = piece / 16; for (uint64_t k = 0; k < nb; k++) if (!mem_any(s.ptr + 16 * k, 16)) zb++;
                const char *dm = getenv("DM_VK_NANSCAN_DUMP"); static _Atomic int nd;   // <format>x<w>: level 0 of such textures written to DM_TEMP (diagnostic, at most 64)
                if (dm && lv == 0 && ly == 0 && (int)strtol(dm, NULL, 16) == d->format && (uint32_t)atoi(strchr(dm, 'x') ? strchr(dm, 'x') + 1 : "0") == im->w && atomic_fetch_add(&nd, 1) < 64) {
                    char fn[512]; snprintf(fn, sizeof fn, "%s/nan-%llu-%#x-%ux%u.blk", getenv("DM_TEMP") ? getenv("DM_TEMP") : "/data/local/tmp", (unsigned long long)t->serial, d->format, w, h);
                    FILE *fo = fopen(fn, "wb"); if (fo) { fwrite(s.ptr, 1, piece, fo); fclose(fo); VLOG("NaN scan: level 0 of serial %llu written to %s", (unsigned long long)t->serial, fn); }
                }
                static _Atomic int nz; if (zb && atomic_fetch_add(&nz, 1) < 64) VLOG("NaN scan: upload of serial %llu format %#x level %u layer %u: %llu of %llu blocks all zero (storage %p)", (unsigned long long)t->serial, d->format, lv, ly, (unsigned long long)zb, (unsigned long long)nb, (const void *)t->mem);
            }
            VkBufferImageCopy c = {s.off, bx * f->bw, by * f->bh, {im->aspect, lv, ly, 1}, {0, 0, 0}, {w, h, z}};
            vk.vkCmdCopyBufferToImage(cb, s.buf, im->img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
            bytes += piece;
            if (!(cb = vka_up_point(cb))) { free(part); *nbytes = bytes; return; }
        }
    }
    free(part);
    *nbytes = bytes;
    static _Atomic int n; if (atomic_fetch_add(&n, 1) < 4) VLOG("texture serial %llu %ux%u format %#x: %llu bytes uploaded from guest memory", (unsigned long long)t->serial, im->w, im->h, d->format, (unsigned long long)bytes);
}

void mtl_tex_init(const tex_t *t) { (void)t; }
void mtl_tex_release(tex_t *t) {
    pthread_mutex_lock(&g_tmu);
    smap_ent *e = map_get(t->serial); vka_img *im = e ? e->im : NULL;
    if (e) map_del(t->serial);
    cpuw_take(t->serial, 0);
    if (im) unref_locked(im);
    pthread_mutex_unlock(&g_tmu);
}
// Linear textures the GPU wrote in this submission. Metal makes a linear texture on the pool buffer, so what the GPU renders there is in guest memory for the CPU to read
// (agl reads small linear render targets back on the CPU, such as a 1x1 R16F one in Donkey Kong Bananza's level); here the image is copied back into the storage
static vka_img **g_wb; static int g_nwb, g_wbcap;
void vka_img_written_at(vka_img *im, const char *file, int line) {
    if (!im) return;
    pthread_mutex_lock(&g_tmu);
    // A zero-shared image must have memory of its own before any write (vka_tex_own at the write's site): this one did not, so every image sharing its memory sees the write
    if (im->m.z) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VRED("zero-shared image serial %llu %ux%u format %#x written at %s:%d without memory of its own (a write path without vka_tex_own): the images that share its memory see the write",
                                                                            (unsigned long long)im->serial, im->w, im->h, im->d.format, file, line); }
    if (!im->gpu_owned && im->m.size >= big_min()) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 64)   // large images: the first GPU write (DM_VK_BIGIMG_MB)
        VLOG("large image serial %llu %ux%u layers %u format %#x (%.1f MiB): first GPU write at frame %llu (%s:%d), sampled by %u draws before, %u other textures over its storage", (unsigned long long)im->serial,
             im->w, im->h, im->layers, im->d.format, im->m.size / 1048576.0, (unsigned long long)vka_frame(), file, line, im->nsamp, im->novl); }
    if (im->mdrop) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VRED("texture serial %llu %ux%u format %#x written at %s:%d while its levels under %u are left out (%s; a write path without vka_tex_own / vka_tex_whole): those levels miss the write",
                                                                             (unsigned long long)im->serial, im->d.width, im->d.height, im->d.format, file, line, im->base, im->prov ? "coarse-first" : "mip drop"); }
    im->gpu_owned = 1; im->rl_wf = (uint32_t)vka_frame();   // (DM_VK_RTLOG_MB: its last write)
    if ((im->d.flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) && im->mem && !im->wb_pending && im->samples == 1 && im->f.kind < 3 && im->f.bw == 1) {
        if (g_nwb == g_wbcap) { g_wbcap = g_wbcap ? 2 * g_wbcap : 32; g_wb = realloc(g_wb, (size_t)g_wbcap * sizeof *g_wb); }
        im->wb_pending = 1; im->refs++; g_wb[g_nwb++] = im;
    }
    pthread_mutex_unlock(&g_tmu);
}
void vka_tex_writeback(VkCommandBuffer cb) {
    pthread_mutex_lock(&g_tmu);
    int any = 0;
    for (int i = 0; i < g_nwb; i++) {
        vka_img *im = g_wb[i]; im->wb_pending = 0;
        uint32_t bpb = im->f.bpb, h = im->h; size_t row = im->d.stride ? im->d.stride : (size_t)im->w * bpb;
        VkBuffer b; VkDeviceSize o;
        if (row % bpb || !vka_addr((uint64_t)(uintptr_t)im->mem, (uint64_t)row * (h - 1) + (uint64_t)im->w * bpb, &b, &o, NULL) || o % (bpb % 4 ? 4 * bpb : bpb)) {
            VRED_ONCE("linear texture serial %llu (format %#x, row %zu) cannot be copied back into its storage: the CPU reads what it held", (unsigned long long)im->serial, im->d.format, row); unref_locked(im); continue; }
        if (!any) { vka_barrier(cb); any = 1; }
        VkBufferImageCopy c = {o, (uint32_t)(row / bpb), h, {im->aspect, 0, 0, 1}, {0, 0, 0}, {im->w, h, 1}};
        vk.vkCmdCopyImageToBuffer(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, b, 1, &c);
        static _Atomic int n; if (atomic_fetch_add(&n, 1) < 4) VLOG("linear texture serial %llu %ux%u format %#x: the GPU's content copied back into its storage", (unsigned long long)im->serial, im->w, h, im->d.format);
        unref_locked(im);
    }
    g_nwb = 0;
    pthread_mutex_unlock(&g_tmu);
}
int mtl_tex_gpu_owned(uint64_t serial) {   // hle_nvn.c nvn_ft_poll: the GPU has written the image (render target, copy or clear destination)
    pthread_mutex_lock(&g_tmu); smap_ent *e = map_get(serial); int r = e && e->im && e->im->gpu_owned; pthread_mutex_unlock(&g_tmu); return r;
}
unsigned mtl_cpu_tex_count(void) {   // live images whose content follows their storage
    pthread_mutex_lock(&g_tmu); unsigned n = 0; for (int i = 0; i < g_nlive; i++) n += g_live[i]->track && !g_live[i]->gpu_owned; pthread_mutex_unlock(&g_tmu); return n;
}
void mtl_pkg_stats(uint64_t *uploads, uint64_t *reuploads) { if (uploads) *uploads = g_st_up[ST_TRACK] + g_st_up[ST_FILL] + g_st_up[ST_SPARSE]; if (reuploads) *reuploads = g_reup; }
// The CPU wrote the texture's storage: uploaded at its first use, or again at the next submission unless the GPU has written it since (then the GPU's content wins)
static uint64_t *g_redo; static int g_nredo, g_redocap;
void mtl_tex_cpu_written(const tex_t *t) {
    pthread_mutex_lock(&g_tmu);
    smap_ent *e = map_get(t->serial);
    if (!e) cpuw_take(t->serial, 1);
    else if (e->im && !e->im->gpu_owned && !e->im->cpu_dirty) {
        e->im->cpu_dirty = 1;
        if (g_nredo == g_redocap) { g_redocap = g_redocap ? g_redocap * 2 : 64; g_redo = realloc(g_redo, (size_t)g_redocap * sizeof *g_redo); }
        g_redo[g_nredo++] = t->serial;
    }
    pthread_mutex_unlock(&g_tmu);
}

// ---------- Descriptor tables ----------
// One update-after-bind set: binding 0 the texture pool (sampled images, up to 2^20), 1 the sampler pool (up to 4096), 2 images (storage). A registration writes its slot at the next
// submission. Nothing draws yet, so no command buffer holds the set while it is written
static samp_par *g_sreg; static int g_nsreg;
static int *g_sdirty; static int g_nsd, g_sdcap;
static uint8_t *g_sdflag;
static VkDescriptorSetLayout g_dsl; static VkDescriptorPool g_dpool; static VkDescriptorSet g_set;
static uint32_t g_ntex_slots, g_nsamp_slots, g_nimg_slots;
typedef struct { samp_par p; VkSampler s; } samp_ent; static samp_ent *g_samp; static int g_nsamp;
static VkSampler *g_sres; static int g_nsres;   // the sampler of each registered ID, found once per registration (sampler_of searches every distinct sampler)
static VkImageView g_null_view;
// Lazy images (DM_VK_TEX_LAZY, on): sampled texture IDs whose slots wait for the first draw that reads the texture pool (write_tables); g_pool_read: one has (under g_tmu;
// read unlocked as a hint by vka_tex_pool_read)
static int *g_tdefer; static int g_ntdf, g_tdfcap; static uint8_t *g_tdfflag; static _Atomic int g_pool_read;
static uint8_t *g_twant;   // (under g_tmu) texture IDs a pool-reading draw's storage buffers named (vka_tex_pool_ids): written at once, as before lazy images
// (under g_tmu) per registered texture ID, g_now_s when a pool-reading draw's scanned storage buffer last held a word naming it (vka_tex_pool_mark; 0: not since its
// registration): g_tseen in the NVN handle form (texture ID in bits 0-19, a registered sampler ID other than 0 in bits 20-31), g_tseen20 by the low 20 bits alone. Float
// and integer data name IDs by chance by the low bits (about one word in 2^20 names a given ID, and draws scan about 10^6 words a second), far less often in the handle
// form. Sized with the registry (treg). The eviction line's seen / unseen split and the overlap rule's keep (tex_evict) read them; nothing is scanned once lazy images end
static uint32_t *g_tseen, *g_tseen20;
static uint8_t g_sok[4096 * 4];   // (under g_tmu) sampler IDs registered (mtl_reg_samp)
static int tex_lazy(void) { static int on = -1; if (on < 0) on = VKA_ENV("DM_VK_TEX_LAZY", 1) != 0; return on; }
static void write_tables(VkCommandBuffer cb);
static VkImageView tfb_placeholder(const treg_ent *e);

static void tfb_init(void);
static void bindless_init(void) {
    static int done; if (done) return; done = 1;
    if (!V.uab) { VRED("no update-after-bind descriptor indexing: the texture and sampler pools have no tables"); return; }
    VkPhysicalDeviceVulkan12Properties p12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &p12}; vk.vkGetPhysicalDeviceProperties2(V.pd, &p2);
#define MIN(a, b) ((a) < (b) ? (a) : (b))
    // Every pipeline layout holds this set beside set 0, whose combined image samplers and storage images count against the same limits: RESERVE of each is left to set 0
    // (a stage's set-0 range is DMSH_VK_STAGE_BINDINGS bindings; 5 graphics stages)
    enum { RESERVE = 5 * DMSH_VK_STAGE_BINDINGS };
#define HEAD(lim) ((lim) > 2 * RESERVE ? (lim) - RESERVE : (lim) / 2)
    g_ntex_slots = MIN(1u << 20, HEAD(MIN(p12.maxDescriptorSetUpdateAfterBindSampledImages, p12.maxPerStageDescriptorUpdateAfterBindSampledImages)));
    g_nsamp_slots = MIN(4096u, HEAD(MIN(p12.maxDescriptorSetUpdateAfterBindSamplers, p12.maxPerStageDescriptorUpdateAfterBindSamplers)));
    g_nimg_slots = MIN(1u << 16, HEAD(MIN(p12.maxDescriptorSetUpdateAfterBindStorageImages, p12.maxPerStageDescriptorUpdateAfterBindStorageImages)));
    // DM_VK_TEX_SLOTS / DM_VK_IMG_SLOTS (default 0: the sizes above): fewer slots for a game whose texture IDs stay low. Every slot is device memory in the driver's
    // descriptor pool: on Turnip 64 bytes, so the 1M-texture table is 64 MiB and the 64K-image table 4 MiB for the process's life; on the Adreno 840
    // system driver about 256 bytes (VK_EXT_device_memory_report, DM_VK_MEMREPORT: descriptor pools 282 MiB, 34.5 MiB with DM_VK_TEX_SLOTS=65536; 2026-11-10). An ID past the table is reported
    // once (write_tables: "beyond the table") and reads as nothing
    { uint32_t t = (uint32_t)VKA_ENV("DM_VK_TEX_SLOTS", 0), i = (uint32_t)VKA_ENV("DM_VK_IMG_SLOTS", 0);
      if (t) g_ntex_slots = MIN(g_ntex_slots, t);
      if (i) g_nimg_slots = MIN(g_nimg_slots, i); }
#undef HEAD
#undef MIN
    VkDescriptorBindingFlags bf = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT, bfs[4];
    VkDescriptorSetLayoutBinding b[4]; VkDescriptorPoolSize ps[4]; uint32_t nb = 0;
    b[nb] = (VkDescriptorSetLayoutBinding){0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, g_ntex_slots, VK_SHADER_STAGE_ALL, NULL}; ps[nb] = (VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, g_ntex_slots}; bfs[nb++] = bf;
    b[nb] = (VkDescriptorSetLayoutBinding){1, VK_DESCRIPTOR_TYPE_SAMPLER, g_nsamp_slots, VK_SHADER_STAGE_ALL, NULL}; ps[nb] = (VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_SAMPLER, g_nsamp_slots}; bfs[nb++] = bf;
    if (g_nimg_slots) { b[nb] = (VkDescriptorSetLayoutBinding){2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, g_nimg_slots, VK_SHADER_STAGE_ALL, NULL}; ps[nb] = (VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, g_nimg_slots}; bfs[nb++] = bf; }
    // The texture-use feedback bitmap ("Texture-use feedback"): in the layout whatever DM_VK_TEXFB says, since a DMSH_VK_F_TEXFB module declares it; partially bound (no
    // update after bind), so it is written only when feedback is on, and a module whose constant is false never reads the slot
    int fbb = p12.maxPerStageDescriptorUpdateAfterBindStorageBuffers > DMSH_VK_STAGE_BINDINGS && p12.maxDescriptorSetUpdateAfterBindStorageBuffers > DMSH_VK_STAGE_BINDINGS;
    if (fbb) { b[nb] = (VkDescriptorSetLayoutBinding){DMSH_VK_POOL_FB_BINDING, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL, NULL}; ps[nb] = (VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
               bfs[nb++] = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT; }
    else VRED("texture-use feedback: the device's update-after-bind storage buffer limits (%u per stage, %u per set) leave no room for the bitmap at set 1 binding %d: "
              "DMSH_VK_F_TEXFB modules' pipelines lack it", p12.maxPerStageDescriptorUpdateAfterBindStorageBuffers, p12.maxDescriptorSetUpdateAfterBindStorageBuffers, DMSH_VK_POOL_FB_BINDING);
    VkDescriptorSetLayoutBindingFlagsCreateInfo bfi = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO, .bindingCount = nb, .pBindingFlags = bfs};
    VkDescriptorSetLayoutCreateInfo li = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .pNext = &bfi, .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT, .bindingCount = nb, .pBindings = b};
    VkDescriptorPoolCreateInfo pi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, .maxSets = 1, .poolSizeCount = nb, .pPoolSizes = ps};
    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorSetCount = 1, .pSetLayouts = &g_dsl};
    VkResult r;
    if ((r = vk.vkCreateDescriptorSetLayout(V.dev, &li, NULL, &g_dsl)) || (r = vk.vkCreateDescriptorPool(V.dev, &pi, NULL, &g_dpool)) || ((ai.descriptorPool = g_dpool), (r = vk.vkAllocateDescriptorSets(V.dev, &ai, &g_set)))) {
        VRED("bindless set (%u textures, %u samplers, %u images): %d", g_ntex_slots, g_nsamp_slots, g_nimg_slots, r); g_set = VK_NULL_HANDLE; return; }
    VLOG("bindless tables: %u textures, %u samplers, %u images (update after bind; about %.1f MiB of descriptors)", g_ntex_slots, g_nsamp_slots, g_nimg_slots,
         ((double)g_ntex_slots + g_nsamp_slots + g_nimg_slots) * 64 / 1048576.0);
    if (fbb && tfb_mode()) tfb_init();
}
VkDescriptorSet vka_bindless_set(void) { return g_set; }
VkDescriptorSetLayout vka_bindless_layout(void) { return g_dsl; }

static void dirty(int **list, int *n, int *cap, uint8_t **flag, int nslots, int id) {
    if (!*flag) *flag = calloc((size_t)nslots, 1);
    if ((*flag)[id]) return;
    (*flag)[id] = 1;
    if (*n == *cap) { *cap = *cap ? *cap * 2 : 256; *list = realloc(*list, (size_t)*cap * sizeof **list); }
    (*list)[(*n)++] = id;
}
static void treg(int id, const tex_t *t, const view_t *v, int img) {
    if (id < 0 || id >= (1 << 20)) return;
    pthread_mutex_lock(&g_tmu);
    if (id >= g_ntreg) { int n = g_ntreg ? g_ntreg : 1024; while (n <= id) n *= 2; g_treg = realloc(g_treg, (size_t)n * sizeof *g_treg); memset(g_treg + g_ntreg, 0, (size_t)(n - g_ntreg) * sizeof *g_treg);
                         int from = g_tseen ? g_ntreg : 0;   // (stamps kept from a smaller table; none after a failed allocation)
                         uint32_t *s1 = realloc(g_tseen, (size_t)n * sizeof *g_tseen); if (s1) g_tseen = s1;
                         uint32_t *s2 = realloc(g_tseen20, (size_t)n * sizeof *g_tseen20); if (s2) g_tseen20 = s2;
                         if (s1 && s2) { memset(g_tseen + from, 0, (size_t)(n - from) * sizeof *g_tseen); memset(g_tseen20 + from, 0, (size_t)(n - from) * sizeof *g_tseen20); }
                         else { free(g_tseen); free(g_tseen20); g_tseen = g_tseen20 = NULL; }   // (no seen stamps: the overlap rule keeps every named image)
                         if (tfb_mode()) {   // ("Texture-use feedback": per-ID stamps; none after a failed allocation, and nothing is gated or released then)
                             int fr = g_tfb_last && g_tfbs ? g_ntreg : 0;
                             uint32_t *l = realloc(g_tfb_last, (size_t)n * sizeof *g_tfb_last); if (l) g_tfb_last = l;
                             uint8_t *st = realloc(g_tfbs, (size_t)n); if (st) g_tfbs = st;
                             if (l && st) { memset(g_tfb_last + fr, 0, (size_t)(n - fr) * sizeof *g_tfb_last); memset(g_tfbs + fr, 0, (size_t)(n - fr)); }
                             else { free(g_tfb_last); free(g_tfbs); g_tfb_last = NULL; g_tfbs = NULL; } }
                         g_ntreg = n; }
    if (g_tseen) g_tseen[id] = g_tseen20[id] = 0;   // (a new texture at the ID: not seen yet)
    // An ID a pool-reading draw's scan named stays wanted (written at once, its image made) for every later texture registered at it (g_twant 2: wanted before this
    // registration), whether a draw samples it or not: Ocarina of Time keeps 32768 IDs registered and registers about 900 a second. DM_VK_TWANT_KEEP=0 (live; default 1)
    // un-wants it when another texture takes the ID: its slot reads null (deferred) until a scan names the ID again. Opt-in: a texture whose handle no scanned storage
    // buffer holds (a uniform buffer, push constants, arithmetic) and that only this inherited want made would read as null
    if (g_twant && g_twant[id] && g_treg[id].t.serial != t->serial) g_twant[id] = VKA_ENV("DM_VK_TWANT_KEEP", 1) ? 2 : 0;
    if (g_tnamed && g_treg[id].t.serial != t->serial) { g_tnamed[id] = 0; g_tnkib[id] = 0; }   // (the `memory:` line's ranges: named again from scratch)
    if (g_tfbs && g_treg[id].t.serial != t->serial) g_tfbs[id] = 0;   // ("Texture-use feedback": another texture, not held back yet; g_tfb_last stays, the gate's "the ID, any serial")
    g_nreg++;
    treg_ent *e = &g_treg[id]; e->t = *t; e->has_v = v != NULL; if (v) e->v = *v; e->img = (uint8_t)img; e->pgen = nvn_pool_live(t->pool, NULL, NULL);
    dirty(&g_tdirty, &g_ntd, &g_tdcap, &g_tdflag, 1 << 20, id);
    if (!g_tver) g_tver = calloc(1 << 20, sizeof *g_tver);
    if (g_tver) atomic_fetch_add_explicit(&g_tver[id], 1, memory_order_release);
    pthread_mutex_unlock(&g_tmu);
}
void mtl_reg_tex(int id, const tex_t *t, const view_t *v) { treg(id, t, v, 0); }
void mtl_reg_image(int id, const tex_t *t, const view_t *v) { treg(id, t, v, 1); }
void mtl_reg_samp(int id, const samp_par *p) {
    if (id < 0 || id >= 4096 * 4) return;
    pthread_mutex_lock(&g_tmu);
    if (id >= g_nsreg) { int n = g_nsreg ? g_nsreg : 256; while (n <= id) n *= 2; g_sreg = realloc(g_sreg, (size_t)n * sizeof *g_sreg); memset(g_sreg + g_nsreg, 0, (size_t)(n - g_nsreg) * sizeof *g_sreg); g_nsreg = n; }
    if (id < g_nsres) g_sres[id] = VK_NULL_HANDLE;
    g_sreg[id] = *p; g_sok[id] = 1; dirty(&g_sdirty, &g_nsd, &g_sdcap, &g_sdflag, 4096 * 4, id);
    if (!g_sver) g_sver = calloc(4096 * 4, sizeof *g_sver);
    if (g_sver) atomic_fetch_add_explicit(&g_sver[id], 1, memory_order_release);
    pthread_mutex_unlock(&g_tmu);
}
// The capacity of the game's texture pools together (hle_nvn.c, at each nvnTexturePoolInitialize): the highest texture or image ID there can be is below it. Logged against
// the bindless tables, whose slots are descriptor memory in the driver whether used or not: what DM_VK_TEX_SLOTS / DM_VK_IMG_SLOTS can be set to for this game
void mtl_set_tex_pool_n(int n) {
    static int logged;
    if (n > logged) { logged = n; VLOG("texture pools: %d IDs in all; the bindless tables have %u texture and %u image slots%s (DM_VK_TEX_SLOTS / DM_VK_IMG_SLOTS)", n, g_ntex_slots, g_nimg_slots,
                                       !g_ntex_slots ? " (not made yet)" : (uint32_t)n > g_ntex_slots || (uint32_t)n > g_nimg_slots ? ": fewer than the pools' IDs" : ""); }
}

static VkSamplerAddressMode wrap(int w) {
    switch (w) {
    case NVN_WRAP_MODE_REPEAT: return VK_SAMPLER_ADDRESS_MODE_REPEAT; case NVN_WRAP_MODE_MIRRORED_REPEAT: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case NVN_WRAP_MODE_CLAMP_TO_BORDER: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case NVN_WRAP_MODE_MIRROR_CLAMP: case NVN_WRAP_MODE_MIRROR_CLAMP_TO_EDGE: case NVN_WRAP_MODE_MIRROR_CLAMP_TO_BORDER: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    default: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    }
}
static VkSampler sampler_of(const samp_par *p) {
    for (int i = 0; i < g_nsamp; i++) if (!memcmp(&g_samp[i].p, p, sizeof *p)) return g_samp[i].s;
    if (g_nsamp >= (int)g_nsamp_slots) { VRED_ONCE("more than %u distinct samplers", g_nsamp_slots); return VK_NULL_HANDLE; }
    int mip = p->minf >= NVN_MIN_FILTER_NEAREST_MIPMAP_NEAREST;
    float aniso = p->aniso > 1 ? (p->aniso < V.props.limits.maxSamplerAnisotropy ? p->aniso : V.props.limits.maxSamplerAnisotropy) : 1;
    // Each distinct guest LOD bias is logged once (16 values at most): a negative one sharpens textures for an upscaler and shimmers at a low render size without it
    // (Ocarina of Time at 683x384 with the NDSR stand-in; session-2026-11-07 shimmer/PLAN.md). DM_VK_LOD_BIAS_MIN=<bias> (samplers made from then on; default off)
    // raises a lower bias to it (0: no negative bias)
    float bias = p->lod_bias, bl = V.props.limits.maxSamplerLodBias, bmin = VKA_ENVF("DM_VK_LOD_BIAS_MIN", -1000.0);
    { static float seen[16]; static int ns; int k = 0; while (k < ns && seen[k] != bias) k++;
      if (k == ns && ns < 16) { seen[ns++] = bias;
          VLOG("sampler LOD bias %g first seen (min filter %d, anisotropy %g, LOD clamp %g..%g; %d samplers made before)%s", bias, p->minf, p->aniso, p->lod_min, p->lod_max, g_nsamp,
               bias < 0 && bmin > -1000 && bias < bmin ? ": raised to DM_VK_LOD_BIAS_MIN" : ""); } }
    if (bmin > -1000 && bias < bmin) bias = bmin;
    if (bias > bl) bias = bl; if (bias < -bl) bias = -bl;
    int white = p->border[0] > 0.5f && p->border[3] > 0.5f, opaque = p->border[3] > 0.5f;
    VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = p->magf == NVN_MAG_FILTER_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .minFilter = p->minf == NVN_MIN_FILTER_LINEAR || p->minf == NVN_MIN_FILTER_LINEAR_MIPMAP_NEAREST || p->minf == NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
        .mipmapMode = p->minf == NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR || p->minf == NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = wrap(p->ws), .addressModeV = wrap(p->wt), .addressModeW = wrap(p->wr), .mipLodBias = bias,
        .anisotropyEnable = aniso > 1, .maxAnisotropy = aniso, .compareEnable = p->cmp_mode == NVN_COMPARE_MODE_COMPARE_R_TO_TEXTURE,
        .compareOp = p->cmp_func >= 1 && p->cmp_func <= 8 ? (VkCompareOp)(p->cmp_func - 1) : VK_COMPARE_OP_NEVER,
        .minLod = p->lod_min, .maxLod = mip ? p->lod_max : (p->lod_min < 0.25f ? 0.25f : p->lod_min),
        .borderColor = white ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : opaque ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK};
    if (p->reduction) VRED_ONCE("sampler reduction mode %d not applied", p->reduction);
    VkSampler s = VK_NULL_HANDLE;
    if (vk.vkCreateSampler(V.dev, &ci, NULL, &s)) return VK_NULL_HANDLE;
    g_samp = realloc(g_samp, (size_t)(g_nsamp + 1) * sizeof *g_samp); g_samp[g_nsamp++] = (samp_ent){*p, s};
    return s;
}
static VkComponentSwizzle swz1(int n) { return n >= 0 && n <= 5 ? (VkComponentSwizzle)(n + 1) : VK_COMPONENT_SWIZZLE_IDENTITY; }   // NVN ZERO ONE R G B A = Vulkan's minus 1

// The view a registered texture ID stands for: its view_t (levels, layers, format, target, swizzle, depth-stencil mode) over the texture's image. vt: the view type, or -1 for
// the registration's own target; storage: a storage image view (no sRGB decode, no swizzle)
// The attachments of the open render pass (vka_draw.c rp_begin): a sampled view of the same image whose levels start below an attached level and include it is cut to the
// levels below (nvn_mtl.m fb_prepare, "Mip-chain pass sampling the level it renders" in docs/reference/rendering.md). DKB fills its environment cube's mip chain level by level,
// each drawn from the one above through a view of the whole chain; sampling the attached level is a feedback loop Vulkan leaves undefined (on Adreno: NaN texels, which bloom
// spread into white blotches and black speckles over the whole level)
static vka_img *g_fb_im[9]; static uint32_t g_fb_lv[9]; static int g_nfb;
void vka_tex_feedback(vka_img *const *ims, const uint32_t *levels, int n) { g_nfb = n > 9 ? 9 : n; for (int i = 0; i < g_nfb; i++) { g_fb_im[i] = ims[i]; g_fb_lv[i] = levels[i]; } }
static VkImageView view_of(const treg_ent *e, vka_img *im, int vt_want, int storage) {
    vka_fmt f; const view_t *v = e->has_v ? &e->v : NULL;
    if (!vka_fmt_of(v && v->set_format ? v->format : e->t.d.format, &f)) f = im->f;
    uint32_t tl = im->base + im->levels, l0 = v && v->set_levels ? v->base_level : 0, nl = v && v->set_levels && v->nlevels ? v->nlevels : tl > l0 ? tl - l0 : 1;
    if (im->vtrack && !storage && l0 < im->vmin) { uint32_t cut = im->vmin - l0; nl = nl > cut ? nl - cut : 1; l0 = im->vmin; }   // a sparse texture: from its wholly mapped levels
    if (l0 < im->base) { uint32_t cut = im->base - l0; nl = nl > cut ? nl - cut : 1; l0 = 0; } else l0 -= im->base;   // texture levels → the image's (a sparse texture's left-out levels)
    uint32_t a0 = v && v->set_layers ? v->min_layer : 0, na = v && v->set_layers && v->nlayers ? v->nlayers : im->layers - a0;
    if (im->d.target == NVN_TEXTURE_TARGET_3D) { a0 = 0; na = 1; }   // a 3D texture's view range names slices (a render target view's), which a 3D view does not take
    if (l0 >= im->levels || a0 >= im->layers) return VK_NULL_HANDLE;
    if (l0 + nl > im->levels) nl = im->levels - l0;
    if (a0 + na > im->layers) na = im->layers - a0;
    if (vt_want >= 0 && !storage) for (int i = 0; i < g_nfb; i++) if (g_fb_im[i] == im && g_fb_lv[i] > l0 && g_fb_lv[i] < l0 + nl) {
        static _Atomic int n; if (!atomic_exchange(&n, 1)) VLOG("a pass samples the texture it renders (serial %llu level %u) through levels %u+%u: sampled through the levels below it", (unsigned long long)im->serial, g_fb_lv[i], l0, nl);
        nl = g_fb_lv[i] - l0; }
    int tgt = v && v->set_target ? v->target : e->t.d.target;
    VkComponentMapping m = {0};
    if (storage) nl = 1;
    else if (v && v->set_swizzle) m = (VkComponentMapping){swz1(v->swz[0]), swz1(v->swz[1]), swz1(v->swz[2]), swz1(v->swz[3])};
    else if (e->t.set_swz) m = (VkComponentMapping){swz1(e->t.swz & 7), swz1(e->t.swz >> 3 & 7), swz1(e->t.swz >> 6 & 7), swz1(e->t.swz >> 9 & 7)};
    else { int fmt = v && v->set_format ? v->format : e->t.d.format; if (fmt == 0x37 || fmt == 0x6D || fmt == 0x6E || fmt == 0x6F || fmt == 0x70 || (fmt >= 0x71 && fmt <= 0x78) || fmt == 0x99 || fmt == 0x9B) m.a = VK_COMPONENT_SWIZZLE_ONE; }
    VkImageViewType vt = vt_want >= 0 ? (VkImageViewType)vt_want : view_type(tgt, na);
    // What the image's type lets the view be (img_create: 3D for a 3D target, 1D for the 1D targets, cube-compatible for square 2D images of whole cubes)
    int img3d = im->d.target == NVN_TEXTURE_TARGET_3D;
    if ((vt == VK_IMAGE_VIEW_TYPE_3D) != img3d || vt == VK_IMAGE_VIEW_TYPE_1D || vt == VK_IMAGE_VIEW_TYPE_1D_ARRAY) return VK_NULL_HANDLE;   // no 1D images (img_create)
    if (vt == VK_IMAGE_VIEW_TYPE_CUBE || vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) {
        if (im->w != im->h || im->layers < 6 || im->layers % 6 || im->samples > 1 || na < 6) return VK_NULL_HANDLE;
        if (vt == VK_IMAGE_VIEW_TYPE_CUBE) na = 6; else na -= na % 6;
    } else if (vt == VK_IMAGE_VIEW_TYPE_1D || vt == VK_IMAGE_VIEW_TYPE_2D) na = 1;
    else if (vt == VK_IMAGE_VIEW_TYPE_3D) { a0 = 0; na = 1; }
    VkImageAspectFlags asp = im->aspect == VK_IMAGE_ASPECT_COLOR_BIT ? VK_IMAGE_ASPECT_COLOR_BIT : e->t.stencil || (v && v->set_dsmode && v->dsmode) ? VK_IMAGE_ASPECT_STENCIL_BIT : im->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_STENCIL_BIT;
    VkFormat vf = e->img || storage ? f.vk : f.view;
    if (im->f.kind >= 3 && f.kind < 3 && depth_twin(f.vk) == im->f.vk) vf = im->f.vk;   // a colour texture or view over a depth image: its depth view ("Colour textures over a depth image")
    return vka_view(im, vf, vt, asp, l0, nl, a0, na, storage ? NULL : &m);
}

// DM_VK_IMG_UPGRADE=0: no image intent from a binding: made and kept without storage use, the binding reads as unbound (as before); live
// storage_able for an image whose own format has no storage use, through a view format that has (img_create: EXTENDED_USAGE)
static int storage_able_as(const vka_img *im, VkFormat f) { return !im->no_storage && !im->rs && VKA_ENV("DM_VK_IMG_UPGRADE", 1) && im->aspect == VK_IMAGE_ASPECT_COLOR_BIT && im->f.kind < 3 && (feats(f) & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT); }
static int storage_able(const vka_img *im) { return !im->no_storage && !im->rs && VKA_ENV("DM_VK_IMG_UPGRADE", 1) && im->aspect == VK_IMAGE_ASPECT_COLOR_BIT && (feats(im->f.vk) & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT); }
// A texture made without NVN's IMAGE flag and then bound to a shader image slot (nvn_mtl.m image_prepare: the binding is the image intent; some engines register a texture with
// RegisterTexture and use its handle in an image slot). Its image was made without storage use (img_create), so the shader's writes went nowhere and what read the texture
// afterwards saw what was there before: DKB's opening area drew purple (a compute pass writing a 1536x864 R11G11B10 target). The image is remade with storage use, its content
// copied, and swapped in place, so every serial and view that shares it follows; the old image goes once the submissions that may use it complete. Not while a thread outside
// recording holds it (the present reads img unlocked): that binding stays unbound and the next one tries again. Caller holds g_tmu; no render pass open in cb
// want: a view format the image's list must take (img_alias_view: a format just learned for it), remade for it even with storage use; VK_FORMAT_UNDEFINED = storage use only
static int img_remake(vka_img *im, VkCommandBuffer cb, VkFormat want) {
    if ((im->usage & VK_IMAGE_USAGE_STORAGE_BIT) && (!want || img_lists(im, want))) return 1;
    if (!cb || im->holds || !(storage_able(im) || (want && storage_able_as(im, want)))) return 0;
    tex_t t = {0}; t.serial = im->serial; t.mem = im->mem; t.d = im->d; t.d.flags |= NVN_TEXTURE_FLAGS_IMAGE; t.L = im->L;
    vka_img *n = img_create(&t, &im->f, im->base);
    if (!n) return 0;
    if (!(n->usage & VK_IMAGE_USAGE_STORAGE_BIT)) {   // the format's image limits refuse storage at this shape
        VRED("texture serial %llu %ux%u VkFormat %d: bound as a shader image, but the driver has no storage use for it at this shape: its writes go nowhere", (unsigned long long)im->serial, im->w, im->h, im->f.vk);
        g_img_bytes -= n->bytes; img_destroy(n); im->no_storage = 1; return 0; }
    if (want && !img_lists(n, want)) { g_img_bytes -= n->bytes; img_destroy(n); return 0; }   // (not learned: table full, VL_PER_IMAGE, DM_VK_VIEW_LEARN=0 or DM_VK_UBWC=0)
    vka_img_barrier(cb, n->img, im->aspect, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    VkImageCopy c[16]; uint32_t nc = 0;
    for (uint32_t lv = 0; lv < im->levels && nc < 16; lv++) {
        VkImageSubresourceLayers sr = {im->aspect, lv, 0, im->layers};
        c[nc++] = (VkImageCopy){sr, {0, 0, 0}, sr, {0, 0, 0}, {im->w >> lv ? im->w >> lv : 1, im->h >> lv ? im->h >> lv : 1, im->depth >> lv ? im->depth >> lv : 1}};
    }
    vk.vkCmdCopyImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, n->img, VK_IMAGE_LAYOUT_GENERAL, nc, c);
    vka_barrier(cb);
    VkImage oi = im->img; vka_imem od = im->m; void *ov = im->views; uint64_t ob = im->bytes;
    // (its format list with it: the new image takes the pairs learned since the old one was made, as img_own's swap does)
    im->img = n->img; im->m = n->m; im->views = NULL; im->usage = n->usage; im->bytes = n->bytes; im->alt = n->alt; im->any_fmt = n->any_fmt; im->nx = n->nx; memcpy(im->xfmt, n->xfmt, sizeof im->xfmt);
    n->img = oi; n->m = od; n->views = ov; n->bytes = ob; g_img_bytes -= ob;
    vka_after(vka_fence_last(), img_destroy, n);
    // Views made of the old image are gone with it: the recording memo (g_tgen) and the registered pool slots that name the image are made again
    atomic_fetch_add_explicit(&g_tgen, 1, memory_order_release);
    int named = 0;
    for (int id = 1; id < g_ntreg; id++) { smap_ent *e = g_treg[id].t.serial ? map_get(g_treg[id].t.serial) : NULL; if (e && e->im == im) { dirty(&g_tdirty, &g_ntd, &g_tdcap, &g_tdflag, 1 << 20, id); named++; } }
    static _Atomic int k; if (atomic_fetch_add(&k, 1) < 16) {
        if (want) VLOG("texture serial %llu %ux%u format %#x (VkFormat %d) flags %#x: bound as a shader image in VkFormat %d, remade with it in its format list (content copied, %d texture IDs rewritten)",
                       (unsigned long long)im->serial, im->w, im->h, im->d.format, im->f.vk, im->d.flags, want, named);
        else VLOG("texture serial %llu %ux%u format %#x (VkFormat %d) flags %#x: bound as a shader image without the image flag, remade with storage use (content copied, %d texture IDs rewritten)",
                  (unsigned long long)im->serial, im->w, im->h, im->d.format, im->f.vk, im->d.flags, named); }
    return 1;
}
static int img_storage_upgrade(vka_img *im, VkCommandBuffer cb) { return img_remake(im, cb, VK_FORMAT_UNDEFINED); }

// A texture whose pool was finalized since it was registered (or is another pool now, initialized at the same address) is not made or read: the game may have unmapped its
// storage and the pool object (Mario Kart World on the phone, m22 / m23). Its ID gets the null view: sampling a texture of a finalized pool is undefined in NVN, and an ID
// registered again is dirty again. pgen 0: its pool was not live at the registration (registered before the pool's Initialize, or after its Finalize): not dead by this
// test, but storage_mode makes it zero texels while the pool is not live. The caller holds nvn_pool_use from this check until the image is made and uploaded
static int treg_dead(int id, const treg_ent *e) {
    int dead = e->t.pool && e->pgen && nvn_pool_live(e->t.pool, NULL, NULL) != e->pgen;
    if (dead) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VLOG("texture ID %d (serial %llu, storage %p): its pool %p was finalized after the registration: null view", id, (unsigned long long)e->t.serial, (void *)e->t.mem, (const void *)e->t.pool); }
    return dead;
}
static void write_tex(int id, VkCommandBuffer cb, VkWriteDescriptorSet *w, VkDescriptorImageInfo *ii) {   // caller holds g_tmu; it is let go while the image is made
    treg_ent ent = g_treg[id], *e = &ent;
    if (e->img && VKA_ENV("DM_VK_IMG_UPGRADE", 1)) e->t.d.flags |= NVN_TEXTURE_FLAGS_IMAGE;   // an image registration is image intent (nvn_mtl.m image_prepare), with or without the texture's IMAGE flag
    pthread_mutex_unlock(&g_tmu);
    nvn_pool_use(1);
    g_first_sampled = g_look_sample = !e->img;   // (a texture ID in the sampled table: a first use here is a sample, nvn2_sampled_content)
    vka_img *im = treg_dead(id, e) ? NULL : vka_tex_get(&e->t, cb);
    g_first_sampled = g_look_sample = 0;
    nvn_pool_use(0);
    // (img_storage_upgrade runs without nvn_pool_use: it copies image to image and never reads the storage)
    if (im && e->img && im->mdrop) vka_tex_whole(im, cb);   // (shaders write it: a mip drop or coarse-first image whole first)
    if (im && e->img && im->m.z) { pthread_mutex_lock(&g_tmu); img_own(im, cb); pthread_mutex_unlock(&g_tmu); }   // (shaders write it: memory of its own first)
    if (im && e->img && !(im->usage & VK_IMAGE_USAGE_STORAGE_BIT)) { pthread_mutex_lock(&g_tmu); img_storage_upgrade(im, cb); pthread_mutex_unlock(&g_tmu); }
    VkImageView view = im ? view_of(e, im, -1, 0) : VK_NULL_HANDLE;
    if (!view) view = g_null_view;
    pthread_mutex_lock(&g_tmu);
    if (im && e->img) { pthread_mutex_unlock(&g_tmu); vka_img_written(im); vka_rtw(im, VKA_RTW_STORE); pthread_mutex_lock(&g_tmu); }   // an image registration: shaders write it (vka_tex_id_view)
    *ii = (VkDescriptorImageInfo){VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
    *w = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g_set, .dstBinding = e->img ? 2 : 0, .dstArrayElement = (uint32_t)id, .descriptorCount = 1,
                                .descriptorType = e->img ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = ii};
}

// ---------- Per-draw lookups (vka_draw.c) ----------
// A shader's binding names the shape it samples (dim: SPIR-V Dim, array, multisample, sampled type): the view is made in that shape when the image allows it, else none
static int dim_view_type(uint8_t dim) {
    int a = (dim & DMSH_VK_DIM_ARRAY) != 0;
    switch (dim & DMSH_VK_DIM_MASK) {
    case 0: return a ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
    case 1: return a ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    case 2: return VK_IMAGE_VIEW_TYPE_3D;
    case 3: return a ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
    default: return -1;
    }
}
static VkImageView id_view(int id, VkCommandBuffer cb, uint8_t dim, int storage, vka_img **imo);
VkImageView vka_tex_id_view(int id, VkCommandBuffer cb, uint8_t dim, int storage, VkImageLayout *layout) {
    if (layout) *layout = VK_IMAGE_LAYOUT_GENERAL;
    return id_view(id, cb, dim, storage, NULL);
}
static int in_fb(const vka_img *im) { for (int i = 0; i < g_nfb; i++) if (g_fb_im[i] == im) return 1; return 0; }   // (a view of a pass's own target is cut to its other levels: view_of)
// Texture ID id stands for an image the open pass draws into (any level): a draw sampling it after a game barrier inside the pass needs the pass ended first (vka_draw.c
// bar_hazard). An ID whose image is not made yet is not one (the pass's attachments were made when it opened; make_images ends the pass for a new one anyway)
int vka_tex_id_in_pass(int id) {
    if (!g_nfb || id <= 0) return 0;
    pthread_mutex_lock(&g_tmu);
    int r = 0;
    if (id < g_ntreg && g_treg[id].t.serial) { smap_ent *e = map_get(g_treg[id].t.serial); r = e && e->im && in_fb(e->im); }
    pthread_mutex_unlock(&g_tmu);
    return r;
}
typedef struct { int id; uint8_t dim; uint32_t ver, gen; vka_img *im; VkImageView v; } vmemo;
static vmemo g_vm[2048]; static struct { int id; uint32_t ver; VkSampler s; } g_sm[512];
static inline vmemo *vm_slot(int id, uint8_t dim) { return &g_vm[((unsigned)id * 7u + dim) & 2047]; }
static inline int vm_hit(const vmemo *m, int id, uint8_t dim) {
    return m->id == id && m->dim == dim && g_tver && m->ver == atomic_load_explicit(&g_tver[id], memory_order_acquire) && m->gen == atomic_load_explicit(&g_tgen, memory_order_acquire) && !in_fb(m->im);
}
// vka_tex_id_view for the recording thread (one thread at a time: the replay worker, or the submitting thread inline), with the memo; storage images take the full path
VkImageView vka_tex_id_view_rec(int id, VkCommandBuffer cb, uint8_t dim, int storage, VkImageLayout *layout) {
    if (layout) *layout = VK_IMAGE_LAYOUT_GENERAL;
    if (storage || id <= 0 || id >= (1 << 20) || !VKA_ENV("DM_VK_TEX_MEMO", 1)) return id_view(id, cb, dim, storage, NULL);
    vmemo *m = vm_slot(id, dim);
    if (vm_hit(m, id, dim)) { if (m->im) { img_used(m->im); m->im->rl_sf = m->im->used_f; if (m->im->prov && !m->im->cf_want) { m->im->cf_want = 1; atomic_store(&g_cf_any, 1); } } return m->v; }   // ("Coarse-first": a lookup)
    uint32_t gen = atomic_load_explicit(&g_tgen, memory_order_acquire), ver = g_tver ? atomic_load_explicit(&g_tver[id], memory_order_acquire) : 0;
    vka_img *im = NULL; VkImageView v = id_view(id, cb, dim, 0, &im);
    if (v && im && g_tver && !in_fb(im)) *m = (vmemo){id, dim, ver, gen, im, v};
    return v;
}
int vka_tex_id_pending_rec(int id, uint8_t dim) {   // 0 at once when the memo holds the ID (its image was made)
    if (id > 0 && id < (1 << 20) && VKA_ENV("DM_VK_TEX_MEMO", 1) && vm_hit(vm_slot(id, dim), id, dim)) return 0;
    return vka_tex_id_pending(id);
}
VkSampler vka_samp_id_rec(int id) {
    if (id <= 0 || id >= 4096 * 4 || !g_sver || !VKA_ENV("DM_VK_TEX_MEMO", 1)) return vka_samp_id(id);
    uint32_t ver = atomic_load_explicit(&g_sver[id], memory_order_acquire);
    typeof(g_sm[0]) *m = &g_sm[id & 511];
    if (m->id == id && m->ver == ver && m->s) return m->s;
    VkSampler sp = vka_samp_id(id);
    m->id = id; m->ver = ver; m->s = sp;
    return sp;
}
static VkImageView id_view(int id, VkCommandBuffer cb, uint8_t dim, int storage, vka_img **imo) {
    pthread_mutex_lock(&g_tmu);
    if (id <= 0 || id >= g_ntreg || !g_treg[id].t.serial) { pthread_mutex_unlock(&g_tmu); return VK_NULL_HANDLE; }
    treg_ent e = g_treg[id];
    if (storage && VKA_ENV("DM_VK_IMG_UPGRADE", 1)) e.t.d.flags |= NVN_TEXTURE_FLAGS_IMAGE;   // bound to an image slot: image intent (img_storage_upgrade)
    pthread_mutex_unlock(&g_tmu);
    nvn_pool_use(1); g_first_sampled = g_look_sample = !storage; g_mk_path = MK_BIND; vka_img *im = treg_dead(id, &e) ? NULL : vka_tex_get(&e.t, cb); g_mk_path = MK_OTHER; g_first_sampled = g_look_sample = 0; nvn_pool_use(0);
    if (!im) return VK_NULL_HANDLE;
    int vt = dim_view_type(dim);
    if (vt < 0) return VK_NULL_HANDLE;
    if (((dim & DMSH_VK_DIM_MS) != 0) != (im->samples > 1)) return VK_NULL_HANDLE;
    if (storage && im->mdrop) vka_tex_whole(im, cb);   // (a shader image a mip drop or coarse-first image: whole before the draw writes it; vka_tex_id_storage_pending ended the pass)
    if (storage && im->m.z) { pthread_mutex_lock(&g_tmu); img_own(im, cb); pthread_mutex_unlock(&g_tmu); }   // (a shader image: memory of its own before the draw writes it; vka_tex_id_storage_pending ended the pass)
    if (!storage && im->m.size >= big_min()) { pthread_mutex_lock(&g_tmu); im->nsamp++; pthread_mutex_unlock(&g_tmu); }   // (DM_VK_BIGIMG_MB: bound for sampling)
    vka_rtw(im, storage ? VKA_RTW_STORE : VKA_RTW_SAMPLE);   // (DM_VK_RTWATCH_MB)
    if (!storage) im->rl_sf = (uint32_t)vka_frame();   // (DM_VK_RTLOG_MB: its last binding for sampling)
    // (a format with no storage use of its own goes on: the alias step below may remake the image with a view format that has it, img_create's EXTENDED_USAGE)
    if (storage && !(im->usage & VK_IMAGE_USAGE_STORAGE_BIT)) { pthread_mutex_lock(&g_tmu); int ok = img_storage_upgrade(im, cb); pthread_mutex_unlock(&g_tmu); if (!ok && (feats(im->f.vk) & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) return VK_NULL_HANDLE; }
    // Bound as a shader image: the GPU writes it (nvn_mtl.m mtl_tex_gpu_owned: ShaderWrite usage), so its storage is no longer its content. Without this a CPU write near a
    // tracked image's pages re-uploaded stale storage over what a compute pass wrote (DKB's 1x1 R16F texture, re-uploaded in the level)
    if (storage) vka_img_written(im);
    if (!storage && !(im->usage & VK_IMAGE_USAGE_SAMPLED_BIT)) return VK_NULL_HANDLE;
    // The sampled type must be the class (float / sint / uint) of the format the view is made in: another class is undefined in Vulkan. That is the registration's view
    // format or the texture's (view_of) when the image's format list takes it, else the image's own (vka_view's fallback). An R32_UINT texture shares a B10G11R11 image
    // whose list takes R32_UINT (same_shape) and is bound through a uint view of it; the image's float class refused it. One the list lacks is learned here (the binding
    // reads as unbound for this image's life, as a float view of it would be the wrong class).
    // A binding of the image's own class stays bound as before, also through a listed view of another class: a float-declared shader image over an R32_UINT registration
    // view of a B10G11R11 image passed this check (the image's class) and got the R32_UINT view under DM_VK_UBWC=0, which shows Ocarina of Time's gameplay picture.
    // The sampled type then differs from the view's class (undefined in Vulkan, as it always was under DM_VK_UBWC=0); refusing it would unbind that write once the pair
    // is learned (open: the declared class is the translator's, which does not know the format)
    // A storage binding over an image whose own format has no storage use, registered with a view format that has it (RGB9E5 through R32_UINT, a335c1db6270c355 binding 3,
    // declared float by the translator so the class step below leaves it): that format is learned and the image remade with it in its list (img_create: EXTENDED_USAGE)
    if (storage && !(im->usage & VK_IMAGE_USAGE_STORAGE_BIT) && e.has_v && e.v.set_format && !(feats(im->f.vk) & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
        vka_fmt sf;
        if (vka_fmt_of(e.v.format, &sf) && sf.kind < 3 && sf.vk != im->f.vk && storage_able_as(im, sf.vk)) {
            pthread_mutex_lock(&g_tmu);
            if (!img_lists(im, sf.vk)) view_unlisted(im, sf.vk, "the image is remade with it in its format list (storage use through it)");
            VkFormat x[VL_PER_IMAGE]; pthread_mutex_lock(&g_vlmu); int nx = vl_formats(im->f.vk, im->w, im->h, x, VL_PER_IMAGE); pthread_mutex_unlock(&g_vlmu);
            int lt = 0; for (int k = 0; k < nx; k++) lt |= x[k] == sf.vk;
            if (lt && vka_rp_open()) im->relist = sf.vk;   // (copies outside the pass: the next draw that binds it ends the pass first)
            else if (lt) { im->relist = VK_FORMAT_UNDEFINED; img_remake(im, cb, sf.vk); }
            pthread_mutex_unlock(&g_tmu);
        }
    }
    int want = dim & DMSH_VK_DIM_SINT ? 2 : dim & DMSH_VK_DIM_UINT ? 1 : 0, have = im->f.kind == 1 ? 1 : im->f.kind == 2 ? 2 : 0;
    // Storage bindings (DM_VK_IMG_ALIAS, default on): a format the list lacks is learned and the image remade with it in its list (img_remake: MUTABLE_FORMAT), so the
    // first binding already writes it instead of reading as unbound for the image's life.
    // An integer storage binding over a colour image of 4-byte texels of another class, with no registration view of that class, is viewed as R32_UINT / R32_SINT: image
    // atomics (SUATOM / SURED) and raw 32-bit stores are typed R32 integer by the translator whatever the texture's format, and the console works on the texel's bits.
    // Ocarina of Time's AA pass d17eefaff4f0bc23 ORs its blend weights into an RGBA8 image that way (nxnative docs/reference/shaders.md "Image atomics on a 32-bit colour texture").
    // A float storage binding over an integer image is a formatted store or load (SUST.P / SULD.P) the translator types float for want of the format, while the console
    // converts by the bound texture's format: bound through the image's own view, as a float binding of a listed R32_UINT view of a B10G11R11 image already is (above).
    // f8965bffbdf51d78 binding 3 writes integer bits so into a 384x216 R16G16_UINT image; both read as unbound before
    int ali = storage && im->f.kind < 3 && !compressed(&im->f) && VKA_ENV("DM_VK_IMG_ALIAS", 1), r32 = 0;
    vka_fmt vf; VkFormat rf = VK_FORMAT_UNDEFINED;
    if (want != have && im->f.kind < 3 && vka_fmt_of(e.has_v && e.v.set_format ? e.v.format : e.t.d.format, &vf) && vf.kind == want) rf = e.img || storage ? vf.vk : vf.view;   // (as view_of asks for it; kinds 0-2 are the classes)
    else if (want && want != have && ali && vl_bytes(im->f.vk) == 4) { rf = want == 1 ? VK_FORMAT_R32_UINT : VK_FORMAT_R32_SINT; r32 = 1; }
    if (rf) {
        pthread_mutex_lock(&g_tmu);
        if (!img_lists(im, rf)) { view_unlisted(im, rf, ali ? "the image is remade with it in its format list" : "the binding reads as unbound");
            // (only a pair the table now holds: one vl_learn refused (another texel size, the table full, VL_PER_IMAGE) made and dropped an image at every binding, and its
            // relist ended the render pass at every draw)
            int lt = 0; if (ali && !VKA_ENV("DM_VK_UBWC", 1)) lt = 1;
            else if (ali) { VkFormat x[VL_PER_IMAGE]; pthread_mutex_lock(&g_vlmu); int nx = vl_formats(im->f.vk, im->w, im->h, x, VL_PER_IMAGE); pthread_mutex_unlock(&g_vlmu); for (int k = 0; k < nx; k++) lt |= x[k] == rf; }
            if (lt && vka_rp_open()) im->relist = rf;   // (copies outside the pass: the next draw that binds it ends the pass first, vka_tex_id_storage_pending)
            else if (lt) { im->relist = VK_FORMAT_UNDEFINED; img_remake(im, cb, rf); } }
        if (img_lists(im, rf)) have = want; else r32 = 0;
        pthread_mutex_unlock(&g_tmu);
    }
    if (want != have && want == 0 && ali) have = want;   // (float binding, integer image: its own view)
    if (want != have && !(im->f.kind >= 3 && want == 0)) return VK_NULL_HANDLE;
    if (imo) *imo = im;
    if (r32) { if (!e.has_v) { memset(&e.v, 0, sizeof e.v); e.has_v = 1; } e.v.set_format = 1; e.v.format = want == 1 ? 0x0B : 0x0C; }   // (NVN R32_UI / R32_I: view_of's format)
    if (storage && !(im->usage & VK_IMAGE_USAGE_STORAGE_BIT)) return VK_NULL_HANDLE;
    return view_of(&e, im, vt, storage);
}
vka_img *vka_tex_id_img(int id, VkCommandBuffer cb) {   // the image of texture ID id (DM_VK_NANSCAN), NULL = none
    pthread_mutex_lock(&g_tmu);
    if (id <= 0 || id >= g_ntreg || !g_treg[id].t.serial) { pthread_mutex_unlock(&g_tmu); return NULL; }
    treg_ent e = g_treg[id];
    pthread_mutex_unlock(&g_tmu);
    nvn_pool_use(1); g_look_sample = 1; vka_img *im = treg_dead(id, &e) ? NULL : vka_tex_get(&e.t, cb); g_look_sample = 0; nvn_pool_use(0);   // (a scan reads it: a sample)
    return im;
}
// The NVN format texture ID id's registration views its texture in, as view_of takes it (the registration view's format, else the texture's; DM_VK_NANSCAN), 0 = not registered
// *img: 1 = an image registration (view_of then views it in the storage format, not the sRGB twin)
int vka_tex_id_fmt(int id, int *img) {
    pthread_mutex_lock(&g_tmu);
    const treg_ent *e = id > 0 && id < g_ntreg && g_treg[id].t.serial ? &g_treg[id] : NULL;
    int f = !e ? 0 : e->has_v && e->v.set_format ? (int)e->v.format : (int)e->t.d.format;
    if (img) *img = e && e->img;
    pthread_mutex_unlock(&g_tmu); return f;
}
// 1 = vka_view makes a view of im in fmt in that format (its own, its sRGB twin, a learned one, any with DM_VK_UBWC=0); 0 = it would fall back to the image's own format
// (DM_VK_NANSCAN: asks without vka_view's learning)
int vka_view_listed(const vka_img *im, VkFormat fmt) {
    pthread_mutex_lock(&g_tmu); int listed = img_lists(im, fmt); pthread_mutex_unlock(&g_tmu); return listed;
}
// DM_VK_NANSCAN_DUMP: level 0 of texture ID id as its storage holds it now (deswizzled), written to DM_TEMP as nan-now-<serial>.blk
void vka_tex_id_dump(int id) {
    pthread_mutex_lock(&g_tmu);
    if (id <= 0 || id >= g_ntreg || !g_treg[id].t.serial) { pthread_mutex_unlock(&g_tmu); return; }
    treg_ent e = g_treg[id];
    pthread_mutex_unlock(&g_tmu);
    vka_fmt f; if (!e.t.mem || !vka_fmt_of(e.t.d.format, &f)) return;
    uint32_t bx = (e.t.d.width + f.bw - 1) / f.bw, by = (e.t.d.height + f.bh - 1) / f.bh; size_t pitch = (size_t)bx * f.bpb, n = pitch * by;
    uint8_t *buf = malloc(n); if (!buf) return;
    nvn_pool_use(1); int bad = treg_dead(id, &e) || (e.t.pool && !nvn_pool_live(e.t.pool, NULL, NULL)) || nvn_deswizzle(&e.t.d, 0, 0, e.t.mem, buf, pitch); nvn_pool_use(0);   // (a finalized pool's storage is not read)
    if (!bad) {
        char fn[512]; snprintf(fn, sizeof fn, "%s/nan-now-%llu.blk", getenv("DM_TEMP") ? getenv("DM_TEMP") : "/data/local/tmp", (unsigned long long)e.t.serial);
        FILE *fo = fopen(fn, "wb"); if (fo) { fwrite(buf, 1, n, fo); fclose(fo); VLOG("NaN scan: texture ID %d serial %llu storage %p (pool flags %#x, layout %d): level 0 as it is now written to %s", id, (unsigned long long)e.t.serial, (void *)e.t.mem, pool_flags(e.t.pool), e.t.has_layout, fn); }
    }
    free(buf);
}
// DM_VK_NANSCAN: texture ID id's storage now against its last upload: changed with the tracker's knowledge (re-uploaded at the next submission) or without it (a missed write)
void vka_tex_id_check(int id) {
    nvn_pool_use(1); pthread_mutex_lock(&g_tmu);
    if (id <= 0 || id >= g_ntreg || !g_treg[id].t.serial) { pthread_mutex_unlock(&g_tmu); nvn_pool_use(0); return; }
    smap_ent *e = map_get(g_treg[id].t.serial); vka_img *im = e ? e->im : NULL;
    if (im && im->track && !im->gpu_owned && im->mem && im->hash && !img_pool_dead(im)) {
        uint64_t h = mem_hash(im->mem, im->L.size); int dirty = nvn_dirty_since_span(im->span, im->mem, im->L.size, im->stamp);
        static _Atomic int n; if (h != im->hash && atomic_fetch_add(&n, 1) < 32) { char why[600]; nvn_dirty_debug(im->span, im->mem, im->L.size, im->stamp, why, sizeof why);
            {   // the mapping's protection as the kernel has it (diagnostic)
                FILE *f = fopen("/proc/self/maps", "re"); char ln[512]; uintptr_t a = (uintptr_t)im->mem;
                while (f && fgets(ln, sizeof ln, f)) { unsigned long lo, hi; if (sscanf(ln, "%lx-%lx", &lo, &hi) == 2 && a >= lo && a < hi) { size_t k = strlen(why); ln[strcspn(ln, "\n")] = 0; snprintf(why + k, sizeof why - k, "; maps: %s", ln); break; } }
                if (f) fclose(f);
            }
            VLOG("NaN scan: texture ID %d serial %llu storage %p+%#llx changed since its upload; the dirty tracker %s: %s", id, (unsigned long long)im->serial, (const void *)im->mem, (unsigned long long)im->L.size, dirty ? "knows (re-upload pending)" : "MISSED it", why); }
    }
    pthread_mutex_unlock(&g_tmu); nvn_pool_use(0);
}
// What texture ID id is, for a report: its texture, the registration's view and the image (or why there is none)
void vka_tex_id_desc(int id, char *b, size_t n) {
    nvn_pool_use(1); pthread_mutex_lock(&g_tmu);   // (the image's storage is read below: its pool held mapped, as everywhere storage is read)
    if (id <= 0 || id >= g_ntreg || !g_treg[id].t.serial) { snprintf(b, n, "not registered"); pthread_mutex_unlock(&g_tmu); nvn_pool_use(0); return; }
    treg_ent e = g_treg[id]; smap_ent *m = map_get(e.t.serial); vka_img *im = m ? m->im : NULL;
    // the image's content state: tracked (re-uploaded on CPU writes), GPU-owned (CPU writes no longer followed), storage written since its last upload, storage non-zero
    // (read under both locks, and not at all once its pool was finalized: its storage may be unmapped)
    char st[96] = ""; if (im && im->mem && !img_pool_dead(im)) { int dirty = im->span ? nvn_dirty_since_span(im->span, im->mem, im->L.size, im->stamp) : -1; int nz = 0; for (uint64_t i = 0; i < im->L.size && i < (1u << 20); i += 8) if (*(const uint64_t *)(im->mem + i)) { nz = 1; break; }
        snprintf(st, sizeof st, "; track %d gpu %d mode %d dirty %d storage %s", im->track, im->gpu_owned, im->mode, dirty, nz ? "data" : "zero"); }
    else if (im && im->mem) snprintf(st, sizeof st, "; its pool finalized");
    uint32_t iw = im ? im->w : 0, ih = im ? im->h : 0, idp = im ? im->depth : 0, il = im ? im->layers : 0, iu = im ? im->usage : 0;
    pthread_mutex_unlock(&g_tmu); nvn_pool_use(0);
    const view_t *v = e.has_v ? &e.v : NULL;
    // swizzle: NVN channel per component (0 zero 1 one 2 R 3 G 4 B 5 A), the view's (v) else the texture's (t), "-" none set
    char sw[16] = "-"; if (v && v->set_swizzle) snprintf(sw, sizeof sw, "v%d%d%d%d", v->swz[0], v->swz[1], v->swz[2], v->swz[3]);
    else if (e.t.set_swz) snprintf(sw, sizeof sw, "t%d%d%d%d", e.t.swz & 7, e.t.swz >> 3 & 7, e.t.swz >> 6 & 7, e.t.swz >> 9 & 7);
    snprintf(b, n, "serial %llu target %d format %#x %ux%ux%u levels %u flags %#x%s; view target %d layers %u+%u format %#x swizzle %s; image %s %ux%ux%u layers %u usage %#x",
             (unsigned long long)e.t.serial, e.t.d.target, e.t.d.format, e.t.d.width, e.t.d.height, e.t.d.depth, e.t.d.levels, e.t.d.flags, e.img ? " (image slot)" : "",
             v && v->set_target ? v->target : -1, v && v->set_layers ? v->min_layer : 0, v && v->set_layers ? v->nlayers : 0, v && v->set_format ? v->format : 0, sw,
             im ? "made" : m ? "refused" : "not made", iw, ih, idp, il, iu);
    if (st[0]) { size_t k = strlen(b); if (k + 1 < n) snprintf(b + k, n - k, "%s", st); }
}
int vka_tex_id_target(int id) {
    pthread_mutex_lock(&g_tmu);
    int t = id > 0 && id < g_ntreg && g_treg[id].t.serial ? (g_treg[id].has_v && g_treg[id].v.set_target ? g_treg[id].v.target : g_treg[id].t.d.target) : -1;
    pthread_mutex_unlock(&g_tmu);
    return t;
}
int vka_tex_id_storage_pending(int id) {
    pthread_mutex_lock(&g_tmu);
    smap_ent *e = id > 0 && id < g_ntreg && g_treg[id].t.serial ? map_get(g_treg[id].t.serial) : NULL;
    int r = e && e->im && (((!(e->im->usage & VK_IMAGE_USAGE_STORAGE_BIT) || (e->im->relist && !img_lists(e->im, e->im->relist))) && (storage_able(e->im) || (e->im->relist && storage_able_as(e->im, e->im->relist))) && !e->im->holds) || e->im->m.z || e->im->mdrop);   // (mdrop: made whole outside the pass)   // (a zero-shared image gets memory of its own: outside the pass too)
    pthread_mutex_unlock(&g_tmu);
    return r;
}
int vka_tex_id_pending(int id) {
    pthread_mutex_lock(&g_tmu);
    int r = id > 0 && id < g_ntreg && g_treg[id].t.serial && !map_get(g_treg[id].t.serial);
    pthread_mutex_unlock(&g_tmu);
    return r;
}
VkSampler vka_samp_id(int id) {
    pthread_mutex_lock(&g_tmu);
    VkSampler s = VK_NULL_HANDLE;
    if (id > 0 && id < g_nsreg) {
        if (id >= g_nsres) { int n = g_nsreg; g_sres = realloc(g_sres, (size_t)n * sizeof *g_sres); memset(g_sres + g_nsres, 0, (size_t)(n - g_nsres) * sizeof *g_sres); g_nsres = n; }
        if (!(s = g_sres[id])) s = g_sres[id] = sampler_of(&g_sreg[id]);
    }
    pthread_mutex_unlock(&g_tmu);
    return s;
}
VkSampler vka_samp_default(void) {
    static VkSampler s;
    if (!s) { VkSamplerCreateInfo ci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                         .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0.25f};
              vk.vkCreateSampler(V.dev, &ci, NULL, &s); }
    return s;
}

// ---------- Texture-use feedback: the bitmap, its readback and the gate (above, "Texture-use feedback") ----------
static VkBuffer g_tfb_buf, g_tfb_rbuf; static VkDeviceMemory g_tfb_mem, g_tfb_rmem; static uint8_t *g_tfb_rptr; static uint32_t g_tfb_rslot; static int g_tfb_clear;
static struct { uint64_t fence; uint32_t n; uint8_t busy; } g_tfb_r[TFB_RING];   // (under g_tmu) the ring's slots: the submission whose bits they hold, bytes
static VkImage g_tfb_phi[2]; static VkDeviceMemory g_tfb_phm[2]; static VkImageView g_tfb_phv[7]; static int g_tfb_ph;   // the placeholders by VkImageViewType (0-6); g_tfb_ph: 1 made, -1 none
static pthread_once_t g_tfb_once = PTHREAD_ONCE_INIT; static int g_tfb_made, g_tfb_gnow;   // g_tfb_gnow: tfb_gate's last answer (the log line)
// A buffer of n bytes in memory of the wanted properties (or, host visible, the nearest), mapped when host visible
static int tfb_buffer(VkDeviceSize n, VkBufferUsageFlags use, int host, VkBuffer *b, VkDeviceMemory *m, uint8_t **p) {
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = use};
    if (vk.vkCreateBuffer(V.dev, &bi, NULL, b)) { *b = VK_NULL_HANDLE; return 0; }
    VkMemoryRequirements mr; vk.vkGetBufferMemoryRequirements(V.dev, *b, &mr);
    uint32_t t = host ? vka_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0) : vka_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
    if (t == UINT32_MAX && host) t = vka_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    if (t == UINT32_MAX && !host) t = vka_mem_type(mr.memoryTypeBits, 0, 0);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = t}; void *q = NULL;
    if (t == UINT32_MAX || vk.vkAllocateMemory(V.dev, &ai, NULL, m)) { *m = VK_NULL_HANDLE; vk.vkDestroyBuffer(V.dev, *b, NULL); *b = VK_NULL_HANDLE; return 0; }
    if (vk.vkBindBufferMemory(V.dev, *b, *m, 0) || (host && vk.vkMapMemory(V.dev, *m, 0, VK_WHOLE_SIZE, 0, &q))) { vk.vkFreeMemory(V.dev, *m, NULL); vk.vkDestroyBuffer(V.dev, *b, NULL); *b = VK_NULL_HANDLE; *m = VK_NULL_HANDLE; return 0; }
    if (p) *p = q;
    return 1;
}
// The bitmap, made once (DM_VK_TEXFB on) at the first pipeline or the set, whichever comes first: the constant is true in every pipeline made after it, so it is never
// true without the bitmap. Then (bindless_init, the set made) written into the set, cleared at the first submission's start (vka_tex_prepare)
static void tfb_make(void) {
    if (!tfb_mode()) return;
    if (tfb_buffer(TFB_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, 0, &g_tfb_buf, &g_tfb_mem, NULL)) g_tfb_made = 1;
    else VRED("texture-use feedback: the %u KiB bitmap was not made: off, the constant false (DM_VK_TEXFB %d)", TFB_BYTES >> 10, tfb_mode());
}
int vka_tex_fb_spec(void) { if (!tfb_mode()) return 0; pthread_once(&g_tfb_once, tfb_make); return g_tfb_made; }
static void tfb_init(void) {
    if (!vka_tex_fb_spec()) return;
    VkDescriptorBufferInfo bi = {g_tfb_buf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g_set, .dstBinding = DMSH_VK_POOL_FB_BINDING, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &bi};
    vk.vkUpdateDescriptorSets(V.dev, 1, &w, 0, NULL);
    g_tfb_clear = 1; atomic_store(&g_tfb_ok, 1);
    int cpl = 0, lib = vka_shader_texfb(&cpl);
    VLOG("texture-use feedback: on, DM_VK_TEXFB %d (%s): a %u KiB bitmap at set 1 binding %d; the library %s", tfb_mode(),
         tfb_mode() == 1 ? "logged only" : tfb_mode() == 2 ? "scan-named images gated under memory pressure" : "scan-named images gated always",
         TFB_BYTES >> 10, DMSH_VK_POOL_FB_BINDING, lib ? (cpl ? "reports from every pool-reading shader" : "reports from some pool-reading shaders (the others' draws are blind)")
                                                   : "has no reporting shader (loaded later, or built before DMSH_VK_F_TEXFB): nothing reports, nothing is gated");
}
// The placeholders a held-back slot shows (DM_VK_TEXFB 2 or 3; caller holds g_tmu; cb outside a render pass): one 1x1 RGBA8 image of 6 layers (2D, 2D array, cube, cube
// array views) and one 1x1x1 3D image, cleared to grey with alpha 1. 1D targets are 2D images here (view_type); multisampled, integer and depth registrations get none (null)
static void tfb_ph_make(VkCommandBuffer cb) {
    g_tfb_ph = -1;
    if (!VKA_ENV("DM_VK_TEXFB_PLACEHOLDER", 1)) return;
    for (int i = 0; i < 2; i++) {
        VkImageCreateInfo ci = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = i ? 0 : VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, .imageType = i ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
                                .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {1, 1, 1}, .mipLevels = 1, .arrayLayers = i ? 1 : 6, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
        VkMemoryRequirements mr; uint32_t t;
        if (vk.vkCreateImage(V.dev, &ci, NULL, &g_tfb_phi[i])) { g_tfb_phi[i] = VK_NULL_HANDLE; continue; }
        vk.vkGetImageMemoryRequirements(V.dev, g_tfb_phi[i], &mr);
        if ((t = vka_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)) == UINT32_MAX) t = vka_mem_type(mr.memoryTypeBits, 0, 0);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = t};
        if (t == UINT32_MAX || vk.vkAllocateMemory(V.dev, &ai, NULL, &g_tfb_phm[i]) || vk.vkBindImageMemory(V.dev, g_tfb_phi[i], g_tfb_phm[i], 0)) {
            if (g_tfb_phm[i]) vk.vkFreeMemory(V.dev, g_tfb_phm[i], NULL);
            vk.vkDestroyImage(V.dev, g_tfb_phi[i], NULL); g_tfb_phi[i] = VK_NULL_HANDLE; g_tfb_phm[i] = VK_NULL_HANDLE; continue; }
        vka_img_barrier(cb, g_tfb_phi[i], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        VkClearColorValue grey = {{0.5f, 0.5f, 0.5f, 1.0f}}; VkImageSubresourceRange rg = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS};
        vk.vkCmdClearColorImage(cb, g_tfb_phi[i], VK_IMAGE_LAYOUT_GENERAL, &grey, 1, &rg);
        static const VkImageViewType vt[2][4] = {{VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, VK_IMAGE_VIEW_TYPE_CUBE, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY}, {VK_IMAGE_VIEW_TYPE_3D}};
        for (int k = 0; k < (i ? 1 : 4); k++) {
            VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g_tfb_phi[i], .viewType = vt[i][k], .format = VK_FORMAT_R8G8B8A8_UNORM,
                                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, vt[i][k] == VK_IMAGE_VIEW_TYPE_2D || vt[i][k] == VK_IMAGE_VIEW_TYPE_3D ? 1 : 6}};
            if (vk.vkCreateImageView(V.dev, &vi, NULL, &g_tfb_phv[vt[i][k]])) g_tfb_phv[vt[i][k]] = VK_NULL_HANDLE;   // (a cube array view needs imageCubeArray)
        }
    }
    vka_barrier(cb);
    g_tfb_ph = g_tfb_phv[VK_IMAGE_VIEW_TYPE_2D] ? 1 : -1;
    VLOG("texture-use feedback: placeholders for held-back slots (1x1 grey, alpha 1): 2D %s, 2D array %s, cube %s, cube array %s, 3D %s", g_tfb_phv[1] ? "yes" : "no", g_tfb_phv[5] ? "yes" : "no",
         g_tfb_phv[3] ? "yes" : "no", g_tfb_phv[6] ? "yes" : "no", g_tfb_phv[2] ? "yes" : "no");
}
static VkImageView tfb_placeholder(const treg_ent *e) {   // (caller holds g_tmu) the placeholder of registration e's view type, or none (the slot reads null)
    if (g_tfb_ph <= 0 || e->img || e->t.d.samples > 1) return VK_NULL_HANDLE;
    const view_t *v = e->has_v ? &e->v : NULL; vka_fmt f;
    if (!vka_fmt_of(v && v->set_format ? v->format : e->t.d.format, &f) || f.kind != 0) return VK_NULL_HANDLE;   // (a float-class colour view: the placeholder's class)
    int tgt = v && v->set_target ? v->target : e->t.d.target;
    if (tgt == NVN_TEXTURE_TARGET_2D_MULTISAMPLE || tgt == NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY || tgt == NVN_TEXTURE_TARGET_BUFFER) return VK_NULL_HANDLE;
    uint32_t na = v && v->set_layers && v->nlayers ? v->nlayers : e->t.L.layers ? e->t.L.layers : 1;
    VkImageViewType vt = view_type(tgt, na);
    return (unsigned)vt < 7 ? g_tfb_phv[vt] : VK_NULL_HANDLE;
}
// Gating on (caller holds g_tmu): DM_VK_TEXFB 3, or 2 under memory pressure; the bitmap made, the library reporting, the per-ID stamps sized, no blind draw within TFB_BLIND_S
static int tfb_gate_now(void) {
    int m = tfb_mode(); if (m < 2 || !atomic_load(&g_tfb_ok) || !g_tfb_last || !g_tfbs || !g_tfb_rbuf || !vka_shader_texfb(NULL)) return g_tfb_gnow = 0;   // (no ring: no readbacks, a held ID would wait forever)
    uint32_t b = atomic_load(&g_tfb_blind_s); if (b && g_now_s + 1 - b <= TFB_BLIND_S) return g_tfb_gnow = 0;
    return g_tfb_gnow = m >= 3 || mem_low();
}
// The gate's end spread over frames (DM_VK_PRESS_OFF_MB=<MiB>, live, default 0; Ocarina of Time 64): when the gate turns off (the pressure is over), the deferred IDs it held
// back were all wanted by the next draws' scans and readbacks and made in one frame: 4,446 and 4,356 textures (638-644 MB of uploads) in frames of 2.48 and 2.53 s, whose burst
// turned the pressure on again 0.7-0.8 s later (session-2026-11-07 aa/AA.md section 4). From that end on, the deferred IDs a scan or a readback wants take at most this many
// MiB (their whole size) a replay frame; the others stay deferred (their placeholder) for the next frame's scans and readbacks. The spread ends at the first frame that took
// them all, or when the gate turns on again. Logged at its start and end (`texture-use feedback: the gate's end spread …`). 0: all at once, as before
static struct { int on; uint64_t frame, used, n, b, put_off; uint32_t frames, capped; uint64_t t0; } g_poff;   // (under g_tmu)
static int tfb_gate(void) {
    int was = g_tfb_gnow, r = tfb_gate_now();
    static int nl;   // (log lines: a blind draw within TFB_BLIND_S also turns the gate off and on)
    if (r) { if (g_poff.on) { g_poff.on = 0; if (nl < 32 && ++nl) VLOG("texture-use feedback: the gate's end spread stops, the gate is on again (%u frames, %llu IDs made, %.0f MiB)", g_poff.frames, (unsigned long long)g_poff.n, g_poff.b / 1048576.0); } }
    else if (was && !g_poff.on && VKA_ENV("DM_VK_PRESS_OFF_MB", 0) > 0) {
        memset(&g_poff, 0, sizeof g_poff); g_poff.on = 1; g_poff.t0 = now_ns();
        if (nl < 32 && ++nl) VLOG("texture-use feedback: the gate is off; the IDs it held back are made at most %ld MiB a frame (DM_VK_PRESS_OFF_MB)", (long)VKA_ENV("DM_VK_PRESS_OFF_MB", 0));
    }
    return r;
}
// A deferred texture ID wanted while the gate is off (caller holds g_tmu): 1 = make it now, 0 = put off to a later frame (the gate's end spread)
static int poff_take(int id) {
    if (!g_poff.on) return 1;
    long mb = VKA_ENV("DM_VK_PRESS_OFF_MB", 0); uint64_t f = vka_frame() + 1;
    if (f != g_poff.frame) {
        if (g_poff.frame && !g_poff.capped) {   // the last frame took every ID wanted in it: the spread is over
            g_poff.on = 0; static int nl;
            if (nl < 32 && ++nl) VLOG("texture-use feedback: the gate's end spread is over: %llu IDs made (%.0f MiB) over %u frames in %.1f s, %llu wants put off a frame",
                 (unsigned long long)g_poff.n, g_poff.b / 1048576.0, g_poff.frames, (now_ns() - g_poff.t0) / 1e9, (unsigned long long)g_poff.put_off);
            return 1;
        }
        g_poff.frame = f; g_poff.used = 0; g_poff.capped = 0; g_poff.frames++;
    }
    if (mb <= 0) { g_poff.on = 0; return 1; }   // (the switch turned off live)
    uint64_t b = dense_bytes(&g_treg[id].t.d, &g_treg[id].t.L, 0), cap = (uint64_t)mb << 20;
    if (g_poff.used && g_poff.used + b > cap) { g_poff.capped = 1; g_poff.put_off++; return 0; }   // (at least one a frame, however large)
    g_poff.used += b; g_poff.n++; g_poff.b += b;
    return 1;
}
void vka_tex_fb_draw(int reporting) {
    if (!tfb_mode()) return;
    if (reporting) { atomic_store(&g_tfb_any, 1); atomic_fetch_add(&g_tfb_rep_n, 1); }
    else { atomic_store(&g_tfb_blind_s, g_now_s + 1); atomic_fetch_add(&g_tfb_blind_n, 1); }
}
// The ring (caller holds g_tmu): slots of at least need bytes; remade larger only when no slot is waiting for its submission (until then the bitmap keeps its bits)
static int tfb_ring(uint32_t need) {
    for (int i = 0; i < TFB_RING; i++) if (g_tfb_r[i].busy) return 0;
    uint32_t sz = 4096; while (sz < need) sz <<= 1;
    if (g_tfb_rbuf) { vk.vkUnmapMemory(V.dev, g_tfb_rmem); vk.vkDestroyBuffer(V.dev, g_tfb_rbuf, NULL); vk.vkFreeMemory(V.dev, g_tfb_rmem, NULL); g_tfb_rbuf = VK_NULL_HANDLE; g_tfb_rptr = NULL; g_tfb_rslot = 0; }
    if (!tfb_buffer((VkDeviceSize)sz * TFB_RING, VK_BUFFER_USAGE_TRANSFER_DST_BIT, 1, &g_tfb_rbuf, &g_tfb_rmem, &g_tfb_rptr)) {
        static int n; if (n++ < 4) VRED("texture-use feedback: the readback ring (%u x %u KiB) was not made: no reports", TFB_RING, sz >> 10); return 0; }
    g_tfb_rslot = sz;
    VLOG("texture-use feedback: readback ring %u x %u KiB (%d texture IDs registered)", TFB_RING, sz >> 10, g_ntreg);
    return 1;
}
void vka_tex_fb_commit(VkCommandBuffer cb, uint64_t fence) {
    if (!cb || !atomic_load(&g_tfb_ok) || !atomic_exchange(&g_tfb_any, 0)) return;
    pthread_mutex_lock(&g_tmu);
    uint32_t n = (uint32_t)(((uint64_t)g_ntreg + 31) / 32 * 4), k = TFB_RING; if (n > TFB_BYTES) n = TFB_BYTES;
    if (n && (g_tfb_rslot >= n || tfb_ring(n))) for (uint32_t i = 0; i < TFB_RING && k == TFB_RING; i++) if (!g_tfb_r[i].busy) k = i;
    if (k == TFB_RING) { g_tfb_skip++; atomic_store(&g_tfb_any, 1); pthread_mutex_unlock(&g_tmu); return; }   // (every slot waiting, or the ring waiting to be remade: the bits stay for the next submission's copy)
    g_tfb_r[k].fence = fence; g_tfb_r[k].n = n; g_tfb_r[k].busy = 1;
    VkBuffer rb = g_tfb_rbuf; VkDeviceSize off = (VkDeviceSize)k * g_tfb_rslot;
    pthread_mutex_unlock(&g_tmu);
    VkMemoryBarrier b0 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
    vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b0, 0, NULL, 0, NULL);
    VkBufferCopy c = {0, off, n}; vk.vkCmdCopyBuffer(cb, g_tfb_buf, rb, 1, &c);
    vk.vkCmdFillBuffer(cb, g_tfb_buf, 0, TFB_BYTES, 0);
    VkMemoryBarrier b1 = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_READ_BIT};
    vk.vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &b1, 0, NULL, 0, NULL);
}
// Texture ID id was indexed in a readback (caller holds g_tmu): stamps; with gating modes, a deferred ID wanted, a provisional image's promotion asked for
static void tfb_hit(int id, int act) {
    uint32_t now1 = g_now_s + 1;
    if (g_tfb_last[id] != now1) { g_tfb_sec_n++; g_tfb_last[id] = now1; }
    const treg_ent *e = &g_treg[id]; uint64_t ser = e->t.serial;
    if (!ser) return;
    g_tfb_ser[ser & 4095] = ser;
    if (act && !g_pool_read && g_twant && g_tdfflag && g_tdfflag[id] && !g_twant[id] && !e->img && poff_take(id)) {   // (deferred: its image and slot at this submission's write_tables; the gate's end spread may put it off)
        g_twant[id] = 1; g_tdfflag[id] = 0; g_tfbs[id] |= TFB_LATE; g_tfb_wn++;
        dirty(&g_tdirty, &g_ntd, &g_tdcap, &g_tdflag, 1 << 20, id);
    }
    smap_ent *m = map_get(ser); vka_img *im = m ? m->im : NULL;
    if (!im) return;
    im->tfb_s = now1; im->tfb_sub = g_tfb_nsub; im->tfb_f = (uint32_t)vka_frame();
    if (im->tfb_st == 1) { im->tfb_st = 2; g_tfb_s0n[0]++; g_tfb_s0b[0] += im->bytes; }
    if (act && im->prov) { if (!im->cf_want) { im->cf_want = 1; atomic_store(&g_cf_any, 1); g_tfb_cfw++; } im->cf_rep = 1; }   // ("Coarse-first": promoted at cf_promote, within its caps; "Reported textures first")
}
// The readbacks whose submissions are done (vka_tex_prepare, caller holds g_tmu; before image eviction, so its release rule sees them)
static void tfb_consume(void) {
    if (!tfb_mode() || !atomic_load(&g_tfb_ok) || !g_tfb_last || !g_tfbs) return;
    if (g_tfb_sec != g_now_s) { g_tfb_ids_ps = g_tfb_sec_n; g_tfb_sec_n = 0; g_tfb_sec = g_now_s; }
    uint64_t done = vka_done(); int act = tfb_mode() >= 2;
    for (int pass = 0; pass < TFB_RING; pass++) {   // in fence order (the oldest done slot first)
        int k = -1; for (int i = 0; i < TFB_RING; i++) if (g_tfb_r[i].busy && g_tfb_r[i].fence <= done && (k < 0 || g_tfb_r[i].fence < g_tfb_r[k].fence)) k = i;
        if (k < 0) break;
        g_tfb_nsub++; g_tfb_subt[g_tfb_nsub & (TFB_SUBT - 1)] = g_now_s;
        const uint32_t *w = (const uint32_t *)(g_tfb_rptr + (size_t)k * g_tfb_rslot); uint32_t nw = g_tfb_r[k].n / 4;
        for (uint32_t i = 0; i < nw; i++) for (uint32_t x = w[i]; x; x &= x - 1) { int id = (int)(i * 32 + (uint32_t)__builtin_ctz(x)); if (id > 0 && id < g_ntreg) tfb_hit(id, act); }
        g_tfb_r[k].busy = 0;
    }
    static uint32_t s0, lastlog; static uint32_t lastsub;
    if (s0 != g_now_s) {   // step 0: scan-named images not indexed within 2 s of their creation
        s0 = g_now_s;
        for (int i = 0; i < g_nlive; i++) { vka_img *im = g_live[i]; if (im->tfb_st == 1 && g_now_s - im->made_s > 2) { im->tfb_st = 3; g_tfb_s0n[1]++; g_tfb_s0b[1] += im->bytes; } }
    }
    if (g_now_s - lastlog >= TFB_LOG_S && g_tfb_nsub != lastsub) { lastlog = g_now_s; lastsub = g_tfb_nsub; char b[1600]; tfb_paths(b, sizeof b); VLOG("%s", b + 2); }
}
static void tfb_remade(uint64_t s) {   // (caller holds g_tmu) a serial made again: counted and pinned when the release rule let it go within 30 s
    if (g_tfbr[s & 4095].serial != s) return;
    if (g_now_s - g_tfbr[s & 4095].s <= 30) { g_tfb_rem++; tpin_add(s); }
    g_tfbr[s & 4095].serial = 0;
}
static void tfb_paths(char *b, size_t n) {   // the `memory:` line's segment (read without the lock: a log line)
    if (!n) return; b[0] = 0;
    if (!tfb_mode()) return;
    int cpl = 0, lib = vka_shader_texfb(&cpl); uint32_t bl = atomic_load(&g_tfb_blind_s);
    snprintf(b, n, "; texture-use feedback (DM_VK_TEXFB %d): %s, gate %s; readbacks %u (%llu submissions not copied), %u texture IDs indexed in the last second; "
             "scan-named images whose ID the GPU indexed within 2 s of creation %llu/%llu (%.0f/%.0f MiB); scan-named made by the gate's evidence %llu (%.0f MiB), held back %llu (%.0f MiB whole), "
             "made late by a report %llu (%.0f MiB; %llu IDs wanted); released unused %llu (%.0f MiB), remade within 30 s %llu (serials pinned %u; released pinned under critical pressure %llu, %.0f MiB; "
             "window %d readbacks, %d frames, %d s); provisional ones asked for by a report %llu; "
             "draws reporting %llu, blind %llu (the last %d s ago)",
             tfb_mode(), !atomic_load(&g_tfb_ok) ? "no bitmap" : !lib ? "no reporting shader in the library" : cpl ? "every pool shader reports" : "some pool shaders report",
             tfb_mode() < 2 ? "off (logged only)" : g_tfb_gnow ? "on" : "off now", g_tfb_nsub, (unsigned long long)g_tfb_skip, g_tfb_ids_ps,
             (unsigned long long)g_tfb_s0n[0], (unsigned long long)(g_tfb_s0n[0] + g_tfb_s0n[1]), g_tfb_s0b[0] / 1048576.0, (g_tfb_s0b[0] + g_tfb_s0b[1]) / 1048576.0,
             (unsigned long long)g_tfb_mkn, g_tfb_mkb / 1048576.0, (unsigned long long)g_tfb_hn, g_tfb_hb / 1048576.0, (unsigned long long)g_tfb_ln, g_tfb_lb / 1048576.0, (unsigned long long)g_tfb_wn,
             (unsigned long long)g_tfb_rn, g_tfb_rb / 1048576.0, (unsigned long long)g_tfb_rem, g_tpin_n, (unsigned long long)g_tfb_pcn, g_tfb_pcb / 1048576.0,
             VKA_ENV("DM_VK_TEXFB_IDLE_READBACKS", 8), VKA_ENV("DM_VK_TEXFB_IDLE_FRAMES", 8), (int)tfb_idle_s(0), (unsigned long long)g_tfb_cfw,
             (unsigned long long)atomic_load(&g_tfb_rep_n), (unsigned long long)atomic_load(&g_tfb_blind_n), bl ? (int)(g_now_s + 1 - bl) : -1);
}
// The release rule's candidate (tex_evict; caller holds g_tmu): no index of its IDs in nr readbacks, nf replay frames (fr: now) and idle_s seconds (tfb_idle_s), no CPU
// lookup in those nf frames (the caller checks its seconds), no blind draw in that window (from the earlier of idle_s ago and the nr-th last readback)
static int tfb_unindexed(const vka_img *im, uint32_t now, uint32_t nr, uint32_t nf, uint32_t fr, uint32_t idle_s) {
    if (now < idle_s || g_tfb_nsub < nr || g_tfb_nsub - im->tfb_sub < nr || fr - im->tfb_f < nf || fr - im->used_f < nf) return 0;
    uint32_t start = im->tfb_s ? im->tfb_s - 1 : im->made_s, tn = g_tfb_subt[(g_tfb_nsub - (nr - 1)) & (TFB_SUBT - 1)], ws = now - idle_s < tn ? now - idle_s : tn;
    uint32_t bl = atomic_load(&g_tfb_blind_s);
    return now - start >= idle_s && (!bl || bl - 1 < ws);
}

// ---------- Image eviction ----------
// An image goes only with its last texture object (mtl_tex_release at Finalize, or a texture object initialized again), and games keep texture objects across an area
// change while they load the next area into the same pool memory: Ocarina of Time went from 769 to 1330 images (+334 MiB) at one area change and nothing was given back.
// Once a second (vka_tex_prepare, before anything is recorded into the submission) an image is let go when all of these hold:
//   - no lookup for a draw, copy or clear found it for DM_VK_EVICT_S seconds of submitting (used_s, g_now_s; default 10);
//   - no registered texture ID names it: a pool-reading shader samples a table slot with no lookup on the CPU, so used_s says nothing about those; an ID registered again
//     with another texture no longer names it (its slot is written again at this submission, before any draw). Except a dead image, below;
//   - nothing outside recording holds it (holds: the present, frame generation, capture), no linear write-back is queued, it is no attachment of the open pass, and it is
//     not sparse (a residency image or a sparse pool's texture: their binds follow the page table);
//   - every reference is one of its serials in the table (refs == the map entries), so letting those go destroys it;
//   - its content can be had again: an image the GPU never wrote holds what its storage held (an upload or a clear by storage_mode), and the next lookup of one of its
//     serials makes a new image from the storage as it is then (vka_tex_get, as at a first use, through the upload batch). An image the GPU wrote holds content that
//     is nowhere else, so it goes only when the guest has reused its storage: a newer image lies over that storage (below). On the Switch that overwrote the texture's
//     bytes, so what the old texture would show is lost there too.
// A newer image over the storage: one made after the image's last lookup (made_s > used_s), over any byte of its storage, whose content is new bytes there: the GPU wrote
// it, or it was filled from the storage (from_mem: the guest stored a texture's bytes there and registered a texture over them). Before, only GPU-written ones counted
// (DM_VK_EVICT_OVER=0, live: as before). Such an image also makes a named one dead (EV_DEAD): the slot of an ID that names it would show the new bytes on the Switch
// (bytes read in its own format and shape: garbage, so a sensible game no longer samples it; an area change loads the next area's textures over the last one's). A dead
// named image goes like an unnamed one, and the IDs naming it go back to the lazy state they were registered in (write_tables: deferred, a null descriptor, as an ID no
// draw has sampled yet): a pool-reading draw whose storage buffer names one again marks it wanted (vka_tex_pool_mark) and its image is made from the storage as it is then
// before that draw (vka_tex_pool_write), which is what the Switch would sample. So only while lazy images hold (vka_tex_pool_lazy: a draw that may index any texture ends
// them), with the seen stamps sized, and not for an image an ID names in the image table (shaders write those) or one whose ID a scanned storage buffer named in the
// handle form within DM_VK_EVICT_S (g_tseen: alive, whatever lies over it). A dead image's serial made again later is pinned (g_evpin, ev_remade): the rule leaves its
// images alone from then on, so a wrong guess costs one re-upload, and null reads until the draw path's next scan of the storage buffer naming it (a frame; DM_VK_POOL_RESCAN_MS) per serial.
// Its serials leave the table (as at Finalize), and the image, its views and its memory are destroyed once the submission being recorded is done, after every earlier one.
// The recording memo is dropped (g_tgen). An image made again for a serial let go is counted (g_ev_remade): each is a re-upload, a hitch on the thread that records.
// The line (image eviction:) and a second one (image eviction kept:) come at most every 30 s while images go, else every 120 s: the kept named images split by seen /
// unseen and overlapped or not, the dead ones let go, registrations a second, and the 8 largest images kept with what keeps them.
// DM_VK_EVICT=2 (live) also lets a named image go as dead when no scanned storage buffer named any of its IDs within DM_VK_EVICT_S, neither in the handle form nor by
// the low 20 bits (g_tseen, g_tseen20: "unseen"), when no image registration names it, the GPU never wrote it (its content is its storage) and its serial is not pinned;
// at most EV_UNSEEN_MAX a second. Lazy images make every pool-read image from such a scan in the first place (vka_tex_pool_mark), so the same scan makes it again when a
// draw names it again; a CPU lookup makes it through vka_tex_get. Ocarina of Time at the Deku Tree cutscene: 130 MiB in 245 named unseen images, then the low-memory killer.
// DM_VK_EVICT=0 (live) keeps every image, as before; so does a device without null descriptors (tex_evict)
#define EV_UNSEEN_MAX (64ull << 20)
enum { EV_NONE, EV_CAND, EV_NAMED, EV_GPU, EV_HELD, EV_DEAD, EV_MASK = 7, EV_UNSEEN = 0x08,
       EV_IMGREG = 0x10, EV_SEEN = 0x20, EV_OVER = 0x40, EV_SEEN20 = 0x80 };   // (flags beside the state in ev_state: named by an image registration, seen in the handle form, a newer image over it, seen by the low 20 bits)
static const char *ev_why(uint8_t s) {
    switch (s & EV_MASK) {
    case EV_NAMED: return s & EV_IMGREG ? "named, image" : s & EV_SEEN ? (s & EV_OVER ? "named, seen, overlapped" : "named, seen") : s & EV_OVER ? "named, overlapped" : s & EV_SEEN20 ? "named, low bits seen" : "named, unseen";
    case EV_GPU: return "GPU-written"; case EV_HELD: return "held"; default: return "?";
    }
}
static void tex_evict(void) {   // caller holds g_tmu
    static uint32_t last, last_log; static int nlog; static uint64_t iv_n, iv_bytes, iv_dn, iv_dbytes, iv_un, iv_ubytes, reg0;
    uint32_t now = g_now_s; int mode = VKA_ENV("DM_VK_EVICT", 1);
    if (!mode || now == last || !g_nlive) return;
    // Only with null descriptors: without them a slot whose new registration has no view keeps the old texture's view (write_tables), which no ID names any more, so
    // letting that image go would leave a destroyed view in the table
    if (!V.null_desc) { static int said; if (!said) { said = 1; VLOG("image eviction: off (the device has no null descriptors)"); } return; }
    last = now;
    uint32_t idle = (uint32_t)VKA_ENV("DM_VK_EVICT_S", 10); if (idle < 2) idle = 2;
    int over_on = VKA_ENV("DM_VK_EVICT_OVER", 1) != 0, demote = over_on && tex_lazy() && !g_pool_read && g_tseen && g_twant;
    int rel = demote && tfb_gate();   // ("Texture-use feedback": named images no draw indexed let go; tfb_c 1 a candidate, 2 named by a texture ID and no image registration)
    int crit = rel && mem_crit();   // (pinned serials released too)
    uint32_t nr = 0, nf = 0, fr = (uint32_t)vka_frame(), tis = tfb_idle_s(rel && tfb_fast());
    if (rel) { int k = VKA_ENV("DM_VK_TEXFB_IDLE_READBACKS", 8), f = VKA_ENV("DM_VK_TEXFB_IDLE_FRAMES", 8); nr = k < 1 ? 1 : k > TFB_SUBT - 1 ? TFB_SUBT - 1 : (uint32_t)k; nf = f < 0 ? 0 : (uint32_t)f; }
    int ncand = 0, nrel = 0;
    for (int i = 0; i < g_nlive; i++) {
        vka_img *im = g_live[i]; im->ev_refs = 0; im->tfb_rel = 0;
        int base = im->mem && im->inited && !im->holds && !im->wb_pending && !im->rs && !im->vpool && !im->vbind && !in_fb(im);
        im->ev_state = now - im->used_s >= idle && base ? EV_CAND : EV_NONE;
        im->tfb_c = (uint8_t)(rel && base && !im->gpu_owned && now - im->used_s >= tis && g_evpin[im->serial & 4095] != im->serial && (crit || !tpin_has(im->serial))
                              && tfb_unindexed(im, now, nr, nf, fr, tis));
        ncand += im->ev_state == EV_CAND; nrel += im->tfb_c;
    }
    if (!ncand && !nrel) return;
    for (int id = 1; id < g_ntreg; id++) {
        if (!g_treg[id].t.serial) continue;
        smap_ent *e = map_get(g_treg[id].t.serial); vka_img *im = e ? e->im : NULL;
        if (im && im->tfb_c) im->tfb_c = g_treg[id].img ? 0 : 2;   // (once 0 it stays 0: an image registration names it, shaders write it)
        if (!im || ((im->ev_state & EV_MASK) != EV_CAND && (im->ev_state & EV_MASK) != EV_NAMED)) continue;
        im->ev_state = (uint8_t)((im->ev_state & ~EV_MASK) | EV_NAMED);
        if (g_treg[id].img) im->ev_state |= EV_IMGREG;
        if (g_tseen && g_tseen[id] && now - g_tseen[id] < idle) im->ev_state |= EV_SEEN;
        if (g_tseen20 && g_tseen20[id] && now - g_tseen20[id] < idle) im->ev_state |= EV_SEEN20;
    }
    // Candidates the GPU wrote, and named ones: a newer image over their storage (the live images by start address, as for the GPU writes into guest memory above)
    static vka_img **by; static int bycap; int nby = 0; uint64_t maxsz = 0;
    if (bycap < g_nlive) { bycap = g_nlive + 256; by = realloc(by, (size_t)bycap * sizeof *by); }
    for (int i = 0; i < g_nlive; i++) if (g_live[i]->mem && g_live[i]->L.size) { by[nby++] = g_live[i]; if (g_live[i]->L.size > maxsz) maxsz = g_live[i]->L.size; }
    qsort(by, (size_t)nby, sizeof *by, img_by_mem);
    for (int i = 0; i < g_nlive; i++) {
        vka_img *a = g_live[i]; int st = a->ev_state & EV_MASK;
        if (!(st == EV_CAND && a->gpu_owned) && !(st == EV_NAMED && over_on)) continue;
        int any = over_on && g_evpin[a->serial & 4095] != a->serial;   // (newer images filled from the storage count too; a pinned serial: GPU-written ones only, as before)
        uint64_t lo = (uint64_t)(uintptr_t)a->mem, hi = lo + a->L.size; int over = 0;
        int l = 0, h = nby;   // the first image starting at or after a's storage end
        while (l < h) { int m = (l + h) / 2; if ((uint64_t)(uintptr_t)by[m]->mem < hi) l = m + 1; else h = m; }
        for (int j = l - 1; j >= 0 && !over && (uint64_t)(uintptr_t)by[j]->mem + maxsz > lo; j--) {
            const vka_img *b = by[j];
            over = b != a && (b->gpu_owned || (any && b->from_mem)) && b->made_s > a->used_s && (uint64_t)(uintptr_t)b->mem + b->L.size > lo;
        }
        if (st == EV_CAND) { if (!over) a->ev_state = EV_GPU; continue; }
        if (!over) continue;
        a->ev_state |= EV_OVER;
        if (demote && any && !(a->ev_state & (EV_IMGREG | EV_SEEN))) a->ev_state = EV_DEAD;   // (any: not a pinned serial, which a GPU-written newer image does not make dead either)
    }
    if (demote && mode >= 2) {   // named and unseen (DM_VK_EVICT=2, above): dead too, in live-list order up to EV_UNSEEN_MAX a second
        uint64_t ub = 0;
        for (int i = 0; i < g_nlive; i++) {
            vka_img *a = g_live[i];
            if ((a->ev_state & EV_MASK) != EV_NAMED || (a->ev_state & (EV_IMGREG | EV_SEEN | EV_SEEN20)) || a->gpu_owned || g_evpin[a->serial & 4095] == a->serial
                || ub + a->bytes > EV_UNSEEN_MAX) continue;
            a->ev_state = EV_DEAD | EV_UNSEEN; ub += a->bytes;
        }
    }
    if (rel) {   // named, not GPU-written, no draw indexed it ("Texture-use feedback"): dead too, in live-list order up to TFB_REL_MAX a second
        uint64_t rb = 0;
        for (int i = 0; i < g_nlive; i++) {
            vka_img *a = g_live[i]; int st = a->ev_state & EV_MASK;
            if (a->tfb_c != 2 || (st != EV_NONE && st != EV_NAMED) || rb + a->bytes > TFB_REL_MAX) continue;
            a->ev_state = EV_DEAD; a->tfb_rel = 1; rb += a->bytes;
        }
    }
    for (size_t i = 0; i < g_mapcap; i++) { smap_ent *e = &g_map[i]; if (e->serial && e->serial != ~0ull && e->im && ((e->im->ev_state & EV_MASK) == EV_CAND || (e->im->ev_state & EV_MASK) == EV_DEAD)) e->im->ev_refs++; }
    uint64_t kb[6] = {0}; int kn[6] = {0}; uint64_t sb[3] = {0}, ob = 0; int sn[3] = {0}, on = 0;   // sb / sn: named kept seen, seen by the low bits only, unseen; ob / on: overlapped
    for (int i = 0; i < g_nlive; i++) {
        vka_img *im = g_live[i]; int st = im->ev_state & EV_MASK;
        if ((st == EV_CAND || st == EV_DEAD) && im->refs != (int)im->ev_refs) im->ev_state = (uint8_t)((im->ev_state & ~EV_MASK) | EV_HELD), st = EV_HELD;
        if (st > EV_CAND && st != EV_DEAD) { kb[st] += im->bytes; kn[st]++; }
        if (st == EV_NAMED) { int k = im->ev_state & EV_SEEN ? 0 : im->ev_state & EV_SEEN20 ? 1 : 2; sb[k] += im->bytes; sn[k]++; if (im->ev_state & EV_OVER) { ob += im->bytes; on++; } }
    }
    // Dead named images: their IDs back to the lazy state (write_tables at this submission writes their slots as null descriptors and defers them)
    int ndem = 0;
    if (demote) for (int id = 1; id < g_ntreg; id++) {
        if (!g_treg[id].t.serial) continue;
        smap_ent *e = map_get(g_treg[id].t.serial);
        if (!e || !e->im || (e->im->ev_state & EV_MASK) != EV_DEAD) continue;
        g_twant[id] = 0; dirty(&g_tdirty, &g_ntd, &g_tdcap, &g_tdflag, 1 << 20, id); ndem++;
        if (e->im->tfb_rel && g_tfb_last && g_tfbs) {   // (let go unindexed: no stale evidence remakes it at the next scan; its slot shows the placeholder)
            uint64_t s = g_treg[id].t.serial; g_tfb_last[id] = 0; g_tfbs[id] |= TFB_PH; if (g_tfb_ser[s & 4095] == s) g_tfb_ser[s & 4095] = 0; }
    }
    // Let go: each serial's entry becomes a tombstone (as map_del); the last one takes the image out of the live list (refs == ev_refs: nothing else names it)
    int n = 0; uint64_t bytes = 0, fence = vka_fence_last() + 1;   // (after the submission being recorded, as img_own)
    for (size_t i = 0; i < g_mapcap; i++) {
        smap_ent *e = &g_map[i]; vka_img *im = e->im;
        if (!e->serial || e->serial == ~0ull || !im || (im->ev_state != EV_CAND && (im->ev_state & EV_MASK) != EV_DEAD)) continue;
        g_evs[e->serial & 4095] = e->serial;
        if ((im->ev_state & EV_MASK) == EV_DEAD && !im->tfb_rel) g_evd[e->serial & 4095] = e->serial;   // (a feedback release: its own pin, tfb_remade)
        if (im->tfb_rel) { g_tfbr[e->serial & 4095].serial = e->serial; g_tfbr[e->serial & 4095].s = now; }
        e->serial = ~0ull; e->im = NULL;
        if (--im->refs) continue;
        for (int k = 0; k < g_nlive; k++) if (g_live[k] == im) { g_live[k] = g_live[--g_nlive]; break; }
        for (int k = 0; k < g_nbig; k++) if (g_big[k] == im) { g_big[k] = g_big[--g_nbig]; break; }
        rtw_forget(im); rl_note(im, "gone (evicted)");
        g_img_bytes -= im->bytes; bytes += im->bytes; n++;
        if ((im->ev_state & EV_MASK) == EV_DEAD) { g_evd_n++; g_evd_bytes += im->bytes; iv_dn++; iv_dbytes += im->bytes; }
        if (im->ev_state & EV_UNSEEN) { g_evu_n++; g_evu_bytes += im->bytes; iv_un++; iv_ubytes += im->bytes; }
        if (im->tfb_rel) { g_tfb_rn++; g_tfb_rb += im->bytes; if (tpin_has(im->serial)) { g_tfb_pcn++; g_tfb_pcb += im->bytes; } }
        vka_after(fence, img_destroy, im);   // (im is not read again in this loop: its last entry was this one)
    }
    if (n) atomic_fetch_add_explicit(&g_tgen, 1, memory_order_release);
    g_ev_n += (uint64_t)n; g_ev_bytes += bytes; iv_n += (uint64_t)n; iv_bytes += bytes;
    if ((iv_n && (nlog < 8 || now - last_log >= 30)) || now - last_log >= 120) {
        uint32_t dt = last_log && now > last_log ? now - last_log : 0;
        nlog++; last_log = now;
        VLOG("image eviction: %llu images (%.1f MiB) not used for %u s let go since the last line; %llu (%.0f MiB) in all, %llu made again; idle images kept: named by a texture ID "
             "%.0f MiB (%d), GPU-written with no newer image over their storage %.0f MiB (%d), referenced from elsewhere %.0f MiB (%d); %d images live (%.0f MiB)",
             (unsigned long long)iv_n, iv_bytes / 1048576.0, idle, (unsigned long long)g_ev_n, g_ev_bytes / 1048576.0, (unsigned long long)g_ev_remade, kb[EV_NAMED] / 1048576.0,
             kn[EV_NAMED], kb[EV_GPU] / 1048576.0, kn[EV_GPU], kb[EV_HELD] / 1048576.0, kn[EV_HELD], g_nlive, g_img_bytes / 1048576.0);
        // The 8 largest images kept (by what keeps them), and the registrations since the last line (an area change that registers its textures at new IDs keeps the old
        // area's named; one that registers them at the old IDs un-names them)
        vka_img *top[8]; int nt = 0;
        for (int i = 0; i < g_nlive; i++) { vka_img *im = g_live[i]; int st = im->ev_state & EV_MASK;
            if (st != EV_NAMED && st != EV_GPU && st != EV_HELD) continue;
            int k = nt < 8 ? nt++ : 8; if (k == 8) { if (im->bytes <= top[7]->bytes) continue; k = 7; }
            while (k > 0 && top[k - 1]->bytes < im->bytes) { top[k] = top[k - 1]; k--; }
            top[k] = im; }
        int nids = 0; for (int id = 1; id < g_ntreg; id++) nids += g_treg[id].t.serial != 0;
        char tb[1400]; size_t tl = 0; tb[0] = 0;
        for (int i = 0; i < nt && tl < sizeof tb; i++) { const vka_img *im = top[i];
            tl += (size_t)snprintf(tb + tl, sizeof tb - tl, "%s serial %llu %ux%u format %#x %.1f MiB idle %u s (%s%s%s)", i ? ";" : "", (unsigned long long)im->serial, im->w, im->h,
                                   im->d.format, im->bytes / 1048576.0, now - im->used_s, ev_why(im->ev_state), g_evpin[im->serial & 4095] == im->serial ? ", pinned" : "",
                                   tpin_has(im->serial) ? ", release-pinned" : ""); }
        VLOG("image eviction kept: named %.0f MiB (%d): seen in a scanned storage buffer within %u s as a handle %.0f MiB (%d), by the low 20 bits only %.0f MiB (%d), unseen %.0f MiB (%d); "
             "a newer image over their storage %.0f MiB (%d); dead (a newer texture over their storage, or unseen under DM_VK_EVICT=2) let go since the last line %llu (%.1f MiB), "
             "%llu (%.0f MiB) in all, %llu made again (their serials pinned); of them unseen %llu (%.1f MiB) since the last line, %llu (%.0f MiB) in all; %d texture IDs deferred again now; dead named images %s; registrations %.1f a second (%llu in all, %d IDs registered, table %d); largest kept:%s",
             kb[EV_NAMED] / 1048576.0, kn[EV_NAMED], idle, sb[0] / 1048576.0, sn[0], sb[1] / 1048576.0, sn[1], sb[2] / 1048576.0, sn[2], ob / 1048576.0, on,
             (unsigned long long)iv_dn, iv_dbytes / 1048576.0, (unsigned long long)g_evd_n, g_evd_bytes / 1048576.0, (unsigned long long)g_evd_remade,
             (unsigned long long)iv_un, iv_ubytes / 1048576.0, (unsigned long long)g_evu_n, g_evu_bytes / 1048576.0, ndem,
             !over_on ? "kept (DM_VK_EVICT_OVER=0)" : demote ? "let go" : !tex_lazy() ? "kept (DM_VK_TEX_LAZY=0)" : g_pool_read ? "kept (a draw may index any texture: lazy images ended)" : "kept (no seen stamps)",
             dt ? (double)(g_nreg - reg0) / dt : 0.0, (unsigned long long)g_nreg, nids, g_ntreg, nt ? tb : " none");
        reg0 = g_nreg;
        iv_n = iv_bytes = iv_dn = iv_dbytes = iv_un = iv_ubytes = 0;
    }
}

// Mip drop's shrink (above, "Mip drop"): idle tracked images of the kind md_levels takes made again without their finer levels, oldest in the live list first, at most
// DM_VK_MIPDROP_SHRINK_MB of their memory a second (caller holds g_tmu and nvn_pool_use; no render pass open in cb)
static void md_shrink(VkCommandBuffer cb) {
    static uint32_t last; static int nlog; uint32_t now = g_now_s;
    if (now == last || !g_nlive) return;
    last = now;
    double capmb = VKA_ENVF("DM_VK_MIPDROP_SHRINK_MB", 64.0);
    if (capmb <= 0 || !md_pressure()) return;
    uint64_t cap = (uint64_t)(capmb * 1048576.0), done = 0, saved = 0; uint32_t idle = (uint32_t)VKA_ENV("DM_VK_MIPDROP_IDLE_S", 10); int n = 0;
    for (int i = 0; i < g_nlive && done < cap; i++) {
        vka_img *im = g_live[i];
        if (im->base || im->mdrop || !im->inited || !im->track || im->gpu_owned || !im->mem || im->vpool || im->rs || im->vbind || im->holds || im->m.z || im->wb_pending
            || (im->usage & VK_IMAGE_USAGE_STORAGE_BIT) || now - im->used_s < idle || in_fb(im)) continue;
        uint32_t k = md_levels(&im->d, &im->f); uint64_t b0 = im->bytes;
        if (!k || !img_relevel(im, k, cb)) continue;
        im->mdrop = (uint8_t)k; done += b0; saved += b0 > im->bytes ? b0 - im->bytes : 0; n++;
    }
    g_mds_n += (uint64_t)n; g_mds_saved += saved;
    if (n && (nlog < 8 || !(nlog & 63))) VLOG("mip drop: %d idle images made without their finer levels (%.1f → %.1f MiB); %llu shrunk in all (-%.0f MiB), %llu made dropped (-%.0f MiB), MemAvailable %ld MiB",
                                              n, done / 1048576.0, (done - saved) / 1048576.0, (unsigned long long)g_mds_n, g_mds_saved / 1048576.0, (unsigned long long)g_md_n,
                                              g_md_saved / 1048576.0, g_md_avail);
    if (n) nlog++;
}
// Mip restore (DM_VK_MIPDROP_RESTORE_MB=<MiB>, live; default 0 = off, as before; Ocarina of Time 32): images mip drop made without their finer levels (or shrank) stayed so
// for the session, "0 made whole again" in the play run (session-2026-11-07 play/PLAY.md: 4,829 dropped, 974 shrunk), while textures made in a short gap of the pressure
// were sharp: mixed detail. Once the pressure (md_pressure) has been off for DM_VK_MIPDROP_RESTORE_CALM_S (15) seconds and MemAvailable is at least
// DM_VK_MIPDROP_RESTORE_AVAIL_MB (1536), dropped images in use (a lookup within DM_VK_MIPDROP_IDLE_S, or an index the GPU reported in that time) are made from level 0
// again (img_relevel, uploaded from their storage), at most this many MiB of new image memory a second. Pressure stops it at once; md_shrink drops them again under it.
// Provisional coarse-first images are coarse-first's own (cf_promote)
static void md_restore(VkCommandBuffer cb) {
    static uint32_t last, calm1; static int nlog; uint32_t now = g_now_s;
    if (now == last || !g_nlive) return;
    last = now;
    double capmb = VKA_ENVF("DM_VK_MIPDROP_RESTORE_MB", 0.0);
    if (capmb <= 0 || VKA_ENV("DM_VK_MIPDROP", 0) != 1) return;
    if (md_pressure()) { calm1 = 0; return; }
    if (!calm1) calm1 = now + 1;
    if (now + 1 - calm1 < (uint32_t)VKA_ENV("DM_VK_MIPDROP_RESTORE_CALM_S", 15) || g_md_avail < VKA_ENV("DM_VK_MIPDROP_RESTORE_AVAIL_MB", 1536)) return;
    uint64_t cap = (uint64_t)(capmb * 1048576.0), made = 0, b0s = 0; uint32_t idle = (uint32_t)VKA_ENV("DM_VK_MIPDROP_IDLE_S", 10); int n = 0;
    for (int i = 0; i < g_nlive && made < cap; i++) {
        vka_img *im = g_live[i];
        if (!im->mdrop || im->prov || im->base != im->mdrop || im->holds || in_fb(im) || img_pool_dead(im)) continue;
        if (now - im->used_s >= idle && !(im->tfb_s && now + 1 - im->tfb_s <= idle)) continue;   // (in use: looked up or indexed lately)
        uint64_t est = dense_bytes(&im->d, &im->L, 0);
        if (made && made + est > cap) continue;   // (one larger image when it is the second's first)
        uint64_t b0 = im->bytes;
        if (!img_relevel(im, 0, cb)) continue;
        im->mdrop = 0; made += im->bytes; b0s += b0; n++;
        g_mdw_n++; g_mrs_n++; g_mrs_b0 += b0; g_mrs_b1 += im->bytes;
    }
    if (n && (nlog < 8 || !(nlog & 63))) VLOG("mip drop: %d images in use made whole again after the pressure (%.1f → %.1f MiB); %llu in all, MemAvailable %ld MiB (DM_VK_MIPDROP_RESTORE_MB)",
                                              n, b0s / 1048576.0, made / 1048576.0, (unsigned long long)g_mrs_n, g_md_avail);
    if (n) nlog++;
}
void vka_tex_prepare(VkCommandBuffer cb) {
    { static time_t wall; struct timespec ts; clock_gettime(CLOCK_MONOTONIC_COARSE, &ts); if (ts.tv_sec != wall) { if (wall) g_now_s++; wall = ts.tv_sec; } }   // (g_now_s above)
    bindless_init();
    nvn_pool_use(1);   // the re-uploads below read guest storage: no pool is finalized (and unmapped) under them. Taken before g_tmu, as write_tex does
    pthread_mutex_lock(&g_tmu);
    if (atomic_load(&g_tfb_ok)) {   // ("Texture-use feedback": the bitmap cleared before its first use, the placeholders made; then the readbacks that are done)
        if (g_tfb_clear) { g_tfb_clear = 0; vk.vkCmdFillBuffer(cb, g_tfb_buf, 0, TFB_BYTES, 0); vka_barrier(cb); }
        if (!g_tfb_ph && tfb_mode() >= 2) tfb_ph_make(cb);
        uint64_t tf_ = VKA_RP_T0(); tfb_consume(); VKA_RP_ADD(VKA_RPX_TEXFB, tf_);
    }
    { uint64_t te_ = VKA_RP_T0(); tex_evict(); VKA_RP_ADD(VKA_RPX_EVICT, te_); }   // (first: an image let go is not re-uploaded below, and nothing in this submission names it)
    uint64_t tm_ = VKA_RP_T0();
    md_shrink(cb);   // (DM_VK_MIPDROP: idle images without their finer levels, uploaded again from their storage)
    md_restore(cb);   // (DM_VK_MIPDROP_RESTORE_MB: in-use dropped images whole again once the pressure is over)
    cf_promote(cb);   // (DM_VK_SCAN_COARSE: provisional images a lookup asked for, or in the background)
    VKA_RP_ADD(VKA_RPX_MIPDROP, tm_);
    rtw_tick();   // (DM_VK_RTWATCH_MB: a line for each watched big target every 10 s)
    rl_tick();   // (DM_VK_RTLOG_MB: the live large images and the overlap groups every 10 s)
    for (int i = 0; i < g_nredo; i++) {   // CPU writes to textures that already have an image (WriteTexels): from now on they follow their storage
        smap_ent *e = map_get(g_redo[i]); vka_img *im = e ? e->im : NULL;
        if (!im || !im->cpu_dirty) continue;
        im->cpu_dirty = 0;
        if (im->gpu_owned || img_pool_dead(im)) continue;
        tex_t t = {0}; t.serial = im->serial; t.d = im->d; t.L = im->L; t.mem = (uint8_t *)im->mem;
        nvn_tex_layout(&t.d, &t.L);
        if (im->vtrack) { sparse_refresh(im, cb); g_reup++; continue; }
        if (!im->track && (im->span = nvn_dirty_span(t.mem, t.L.size))) im->track = 1;
        if (im->track) upload_tracked(im, &t, cb); else upload_fill(im, &t, cb);
        g_reup++;
    }
    g_nredo = 0;
    // GPU writes whose submission is done: the images over them that did not get their bytes are uploaded again (see gw_t above)
    gml_poll();
    if (vka_nan_on()) watch_scan(cb);
    static uint64_t nprep, kinds[4], ntake; nprep++;
    if (vka_nan_on()) { for (int i = 0; i < g_ngw; i++) if (g_gw[i].fence && g_gw[i].fence <= vka_done()) kinds[g_gw[i].kind]++;
        if (nprep % 600 == 0) VLOG("NaN scan: GPU writes into guest memory done so far: %llu copies, %llu fills, %llu inline updates, %llu stores / texture → buffer copies (%llu pages from raw global memory logs, %llu logs full, %llu without a log); textures re-uploaded after them %llu", (unsigned long long)kinds[0], (unsigned long long)kinds[1], (unsigned long long)kinds[2], (unsigned long long)kinds[3], (unsigned long long)g_gml_pages, (unsigned long long)g_gml_full, (unsigned long long)g_gml_none, (unsigned long long)g_gw_late); }
    (void)ntake;
    if (g_ngw) {
        // The images whose content comes from their storage, by start address, searched for each done write (hundreds a frame, thousands of images)
        static vka_img **by; static int bycap; int nby = 0; uint64_t maxsz = 0, done = vka_done(); int k = 0, any = 0;
        for (int i = 0; i < g_ngw && !any; i++) any = g_gw[i].fence && g_gw[i].fence <= done;
        if (any) {
            if (bycap < g_nlive) { bycap = g_nlive + 256; by = realloc(by, (size_t)bycap * sizeof *by); }
            for (int j = 0; j < g_nlive; j++) { vka_img *im = g_live[j]; if (im->from_mem && !im->gpu_owned && im->mem && !im->vpool && !img_pool_dead(im)) { by[nby++] = im; if (im->L.size > maxsz) maxsz = im->L.size; } }
            qsort(by, (size_t)nby, sizeof *by, img_by_mem);
        }
        for (int i = 0; i < g_ngw; i++) {
            gw_t *w = &g_gw[i];
            if (!w->fence || w->fence > done) { g_gw[k++] = *w; continue; }
            int lo = 0, hi = nby;   // the first image starting at or after the write's end
            while (lo < hi) { int m = (lo + hi) / 2; if ((uint64_t)(uintptr_t)by[m]->mem < w->hi) lo = m + 1; else hi = m; }
            for (int j = lo - 1; j >= 0 && (uint64_t)(uintptr_t)by[j]->mem + maxsz > w->lo; j--) {
                vka_img *im = by[j]; uint64_t a = (uint64_t)(uintptr_t)im->mem;
                if (w->lo >= a + im->L.size || im->gw_seen >= w->seq || im->gpu_owned) continue;
                tex_t t = {0}; t.serial = im->serial; t.d = im->d; t.L = im->L; t.mem = (uint8_t *)im->mem;
                if (im->track) upload_tracked(im, &t, cb); else upload_fill(im, &t, cb);
                g_gw_late++; if (g_gw_late <= 8 || !(g_gw_late & (g_gw_late - 1))) VLOG("textures re-uploaded after a GPU write into their storage: %llu (uploads that applied pending GPU writes: %llu; pages raw global memory stores wrote: %llu)", (unsigned long long)g_gw_late, (unsigned long long)g_gw_over, (unsigned long long)g_gml_pages);
                if (g_gw_late <= 8) VLOG("texture serial %llu %ux%u format %#x: a GPU write into its storage (%s %#llx+%#llx) is done, re-uploaded", (unsigned long long)im->serial, im->w, im->h, im->d.format,
                                          w->kind == GW_COPY ? "copy" : w->kind == GW_FILL ? "fill" : w->kind == GW_UPDATE ? "inline update" : "store", (unsigned long long)w->lo, (unsigned long long)(w->hi - w->lo));
            }
            free(w->data);
        }
        g_ngw = k;
    }
    // Tracked images whose pages the CPU wrote since their last upload (a write to a protected page moves the dirty clock, so nothing is scanned while it stands still)
    // Sparse pools' textures also when a mapping request came since (the page table's generation moved)
    static uint64_t seen, vseen; uint64_t now = nvn_dirty_now(), vnow = nvn_sparse_gen();
    if (now != seen || vnow != vseen) {
        seen = now; vseen = vnow;
        for (int i = 0; i < g_nlive; i++) {
            vka_img *im = g_live[i];
            if (img_pool_dead(im)) continue;
            if (im->vbind && im->mem) { if (nvn_sparse_since(im->vpool, (uint64_t)(uintptr_t)im->mem, im->L.size, im->vgen, im->stamp)) sparse_bind(im, cb, 0); continue; }   // (GPU-written: binds only)
            if (im->vtrack && !im->gpu_owned && im->mem) {
                if (!nvn_sparse_since(im->vpool, (uint64_t)(uintptr_t)im->mem, im->L.size, im->vgen, im->stamp)) continue;
                sparse_refresh(im, cb);
                if (g_reup++ < 8) VLOG("sparse texture serial %llu %ux%u format %#x: its mappings or pages changed, re-uploaded", (unsigned long long)im->serial, im->w, im->h, im->d.format);
                continue;
            }
            if (!im->track || im->gpu_owned || !im->mem) continue;
            uint64_t size = im->L.size;
            size_t lo, hi;
            if (!nvn_dirty_bounds_since(im->span, im->mem, size, im->stamp, &lo, &hi)) continue;
            tex_t t = {0}; t.serial = im->serial; t.d = im->d; t.L = im->L; t.mem = (uint8_t *)im->mem;
            int rows = VKA_ENV("DM_VK_ROWS", 1);
            if (rows && upload_rows_tracked(im, &t, cb, lo, hi)) continue;
            upload_tracked(im, &t, cb);
            if (g_reup++ < 8) VLOG("texture serial %llu %ux%u format %#x flags %#x target %d: its storage changed, re-uploaded", (unsigned long long)im->serial, im->w, im->h, im->d.format, im->d.flags, im->d.target);
        }
    }
    nvn_pool_use(0);   // (write_tex takes it again for each ID, after letting g_tmu go: never twice on one thread)
    if (!g_set) { g_ntd = g_nsd = 0; if (g_tdflag) memset(g_tdflag, 0, 1 << 20); if (g_sdflag) memset(g_sdflag, 0, 4096 * 4); pthread_mutex_unlock(&g_tmu); return; }
    write_tables(cb);
    pthread_mutex_unlock(&g_tmu);
}
// The table slots registered since the last write (caller holds g_tmu; g_set made). A sampled texture ID waits in the deferred list while no draw or dispatch has read the
// texture pool (vka_tex_pool_read): its image is made at the first draw that binds it (id_view), so a registered texture nothing samples costs no image
static void write_tables(VkCommandBuffer cb) {
    enum { BATCH = 64 }; VkWriteDescriptorSet w[BATCH]; VkDescriptorImageInfo ii[BATCH]; int nw = 0;
    for (int k = 0; k < g_ntd; k++) {
        int id = g_tdirty[k]; g_tdflag[id] = 0;
        if (id >= (int)(g_treg[id].img ? g_nimg_slots : g_ntex_slots)) { VRED_ONCE("texture ID %d beyond the table", id); continue; }
        if (!g_pool_read && !g_treg[id].img && tex_lazy() && !(g_twant && g_twant[id])) {
            dirty(&g_tdefer, &g_ntdf, &g_tdfcap, &g_tdfflag, 1 << 20, id);
            // Its slot reads as a null descriptor meanwhile (zeros), not as the view of whatever texture held the ID before (destroyed since, perhaps); held back or let go
            // by the texture-use feedback, the placeholder of its view type when it has one
            VkImageView ph = g_tfbs && (g_tfbs[id] & TFB_PH) ? tfb_placeholder(&g_treg[id]) : VK_NULL_HANDLE;
            if (ph) { ii[nw] = (VkDescriptorImageInfo){VK_NULL_HANDLE, ph, VK_IMAGE_LAYOUT_GENERAL};
                      w[nw] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g_set, .dstBinding = 0, .dstArrayElement = (uint32_t)id, .descriptorCount = 1,
                                                     .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &ii[nw]};
                      if (++nw == BATCH) { vk.vkUpdateDescriptorSets(V.dev, (uint32_t)nw, w, 0, NULL); nw = 0; } }
            else if (V.null_desc) { ii[nw] = (VkDescriptorImageInfo){VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL};
                               w[nw] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g_set, .dstBinding = 0, .dstArrayElement = (uint32_t)id, .descriptorCount = 1,
                                                              .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &ii[nw]};
                               if (++nw == BATCH) { vk.vkUpdateDescriptorSets(V.dev, (uint32_t)nw, w, 0, NULL); nw = 0; } }
            else if (g_null_view) { ii[nw] = (VkDescriptorImageInfo){VK_NULL_HANDLE, g_null_view, VK_IMAGE_LAYOUT_GENERAL};   // (no null descriptors: the null view, as write_tex)
                                    w[nw] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g_set, .dstBinding = 0, .dstArrayElement = (uint32_t)id, .descriptorCount = 1,
                                                                   .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &ii[nw]};
                                    if (++nw == BATCH) { vk.vkUpdateDescriptorSets(V.dev, (uint32_t)nw, w, 0, NULL); nw = 0; } }
            continue;
        }
        g_mk_path = g_pool_read || g_treg[id].img || !tex_lazy() ? MK_ALL : g_twant && g_twant[id] == 2 ? MK_STICKY : MK_FRESH;   // (the `memory:` line's paths)
        g_mk_id = id;
        if (g_tfbs) { g_mk_tfb = g_tfbs[id] & (TFB_PASS | TFB_LATE); g_tfbs[id] &= (uint8_t)~(TFB_PASS | TFB_LATE | TFB_PH); }   // (written now: no placeholder from here on)
        write_tex(id, cb, &w[nw], &ii[nw]);
        g_mk_path = MK_OTHER; g_mk_id = 0; g_mk_tfb = 0;
        // No view (no image, a refused format or shape, a dead pool): the slot is written as a null descriptor when the device has them. Skipped, it kept the view of
        // whatever texture held the ID before, which may be destroyed since (a texture finalized, or an image let go by tex_evict)
        if (w[nw].pImageInfo->imageView || V.null_desc) nw++;
        if (nw == BATCH) { vk.vkUpdateDescriptorSets(V.dev, (uint32_t)nw, w, 0, NULL); nw = 0; }
    }
    g_ntd = 0;
    for (int k = 0; k < g_nsd; k++) {
        int id = g_sdirty[k]; g_sdflag[id] = 0;
        if (id >= (int)g_nsamp_slots) { VRED_ONCE("sampler ID %d beyond the table (%u)", id, g_nsamp_slots); continue; }
        VkSampler s = sampler_of(&g_sreg[id]); if (!s) continue;
        ii[nw] = (VkDescriptorImageInfo){s, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        w[nw] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g_set, .dstBinding = 1, .dstArrayElement = (uint32_t)id, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .pImageInfo = &ii[nw]};
        if (++nw == BATCH) { vk.vkUpdateDescriptorSets(V.dev, (uint32_t)nw, w, 0, NULL); nw = 0; }
    }
    g_nsd = 0;
    if (nw) vk.vkUpdateDescriptorSets(V.dev, (uint32_t)nw, w, 0, NULL);
}
// Lazy images still defer texture IDs: no draw or dispatch has read every registered texture yet (a hint, read unlocked: vka_draw.c pool_scan skips its scan once it is 0)
int vka_tex_pool_lazy(void) { return tex_lazy() && !g_pool_read; }
// A pool-reading draw's texture IDs, as far as the CPU can tell (vka_draw.c pool_scan): the low 20 bits of every word of its storage buffers (where NVN2 games keep the
// handles a shader indexes the pool with, such as a material table; other words name IDs that are not registered or cost an image that is not sampled). The deferred ones
// among them are marked wanted and queued for writing; returns how many. vka_tex_pool_write then makes their images and slots, outside any render pass.
// range: the size of the scanned binding range the words come from (0: unknown), for the `memory:` line's counters only (NM_WIN)
int vka_tex_pool_mark(const uint32_t *wd, size_t n, uint64_t range, int fb) {
    if (g_pool_read) return 0;
    pthread_mutex_lock(&g_tmu);
    int made = 0;
    if (!g_twant) g_twant = calloc(1 << 20, 1);
    if (!g_tnamed && (g_tnamed = calloc(1 << 20, 1)) && !(g_tnkib = calloc(1 << 20, sizeof *g_tnkib))) { free(g_tnamed); g_tnamed = NULL; }
    uint32_t now = g_now_s; uint8_t nb = range && range <= NM_WIN ? 1 : 2;
    int gate = fb && n && tfb_gate();   // ("Texture-use feedback": the GPU's report decides which named IDs get images)
    for (size_t i = 0; i < n && g_twant && g_set && !g_pool_read; i++) {
        uint32_t w = wd[i], sid = w >> 20; int id = (int)(w & 0xFFFFF);
        if (id <= 0 || id >= g_ntreg) continue;
        if (g_tnamed && !(g_tnamed[id] & nb)) nm_named(id, nb);
        if (g_tseen) { g_tseen20[id] = now; if (sid && g_sok[sid]) g_tseen[id] = now; }   // (sid < 4096 * 4: 12 bits)
        if (!g_tdfflag || !g_tdfflag[id] || g_twant[id]) continue;
        if (gate) {   // indexed within TFB_RECENT_S, or a handle whose serial the GPU indexed before, or a serial that already has its image (holding it saves nothing, and a
                      // render target sampled through the pool, a depth one by Dref, would read null or a colour placeholder); else held back (the placeholder from the next write)
            uint32_t ls = g_tfb_last[id]; uint64_t ser = g_treg[id].t.serial; const smap_ent *me = ser ? map_get(ser) : NULL;
            if (!(ls && now + 1 - ls <= TFB_RECENT_S) && !(sid && g_sok[sid] && ser && g_tfb_ser[ser & 4095] == ser) && !(me && me->im)) {
                if (!(g_tfbs[id] & TFB_HELD)) { g_tfbs[id] |= TFB_HELD | TFB_PH; g_tfb_hn++; g_tfb_hb += dense_bytes(&g_treg[id].t.d, &g_treg[id].t.L, 0); dirty(&g_tdirty, &g_ntd, &g_tdcap, &g_tdflag, 1 << 20, id); }
                continue;
            }
            g_tfbs[id] |= TFB_PASS;
        } else if (!poff_take(id)) continue;   // (the gate's end spread: wanted again by a later frame's scan)
        g_twant[id] = 1; g_tdfflag[id] = 0; made++;
        dirty(&g_tdirty, &g_ntd, &g_tdcap, &g_tdflag, 1 << 20, id);
    }
    pthread_mutex_unlock(&g_tmu);
    return made;
}
// Before a draw or dispatch whose shaders read the texture pool (vka_draw.c pool_scan; no render pass open): the IDs marked wanted (vka_tex_pool_mark) get their images and
// slots on the recording command buffer (their first uploads go through the upload batch, as a draw's lookups do). all: the shaders may index any registered texture, so
// every deferred ID gets its image, and registrations from now on are written at once
void vka_tex_pool_write(VkCommandBuffer cb, int all) {
    pthread_mutex_lock(&g_tmu);
    if (g_set) {
        int n = 0;
        if (all && !g_pool_read) {
            g_pool_read = 1; n = g_ntdf;
            for (int k = 0; k < g_ntdf; k++) { int id = g_tdefer[k]; if (!g_tdfflag[id]) continue; g_tdfflag[id] = 0; dirty(&g_tdirty, &g_ntd, &g_tdcap, &g_tdflag, 1 << 20, id); }
            g_ntdf = 0;
            VLOG("texture pool: a draw or dispatch may index any registered texture; the %d texture IDs registered without an image get theirs now", n);
        } else if (!all) { static _Atomic int nl; if (atomic_fetch_add(&nl, 1) < 8) VLOG("texture pool: %d texture IDs queued, those a pool-reading draw's storage buffers named among them, get their images and slots", g_ntd); }
        write_tables(cb);
    }
    if (all) g_pool_read = 1;
    pthread_mutex_unlock(&g_tmu);
}

// ---------- The frame capture (vka_capture.c, DM_NVN_CAPTURE) and its replayer (vka_replay.c) ----------
// Read under the texture lock: a registry entry (0 = none registered at id; *n = the table's size), a sampler entry, whether the CPU wrote a serial before its image existed,
// and an image a texture would share at its first use (same storage and shape: vka_tex_get), with a reference the caller drops (vka_tex_drop)
int vka_cap_treg(int id, tex_t *t, view_t *v, uint8_t *has_v, uint8_t *img, int *n) {
    pthread_mutex_lock(&g_tmu);
    if (n) *n = g_ntreg;
    int ok = id >= 0 && id < g_ntreg && g_treg[id].t.serial;
    if (ok) { *t = g_treg[id].t; *v = g_treg[id].v; *has_v = g_treg[id].has_v; *img = g_treg[id].img; }
    pthread_mutex_unlock(&g_tmu);
    return ok;
}
int vka_cap_sreg(int id, samp_par *p, int *n) {
    static const samp_par zero;
    pthread_mutex_lock(&g_tmu);
    if (n) *n = g_nsreg;
    int ok = id >= 0 && id < g_nsreg && memcmp(&g_sreg[id], &zero, sizeof zero);
    if (ok) *p = g_sreg[id];
    pthread_mutex_unlock(&g_tmu);
    return ok;
}
int vka_cap_cpuw(uint64_t serial) {
    pthread_mutex_lock(&g_tmu);
    int r = 0; if (g_cpuwcap) for (size_t i = hslot(serial, g_cpuwcap);; i = (i + 1) & (g_cpuwcap - 1)) { if (g_cpuw[i] == serial) { r = 1; break; } if (!g_cpuw[i]) break; }
    pthread_mutex_unlock(&g_tmu);
    return r;
}
vka_img *vka_cap_shared(const tex_t *t) {
    vka_fmt f; vka_img *im = NULL;
    if (!t || !t->mem || !vka_fmt_of(t->d.format, &f)) return NULL;
    pthread_mutex_lock(&g_tmu);
    for (int i = 0; i < g_nlive && !im; i++) if (same_shape(g_live[i], t, &f, 0)) { im = g_live[i]; im->refs++; }
    pthread_mutex_unlock(&g_tmu);
    return im;
}
#endif
