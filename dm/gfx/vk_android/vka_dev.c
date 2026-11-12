// The Vulkan device of the replay: instance, the one queue, the timeline that fences submissions, the completion thread that runs their CPU-side effects, and scratch memory.
// Feature and extension choices follow the probe of android/spike (results/honor-pnm-n49.md): placed memory maps, buffer device addresses, update-after-bind arrays, display timing
#if defined(__ANDROID__)
#include "vka.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>
#include <unistd.h>

// This file's lines made before the run log opened. The launcher's surfaceCreated (GameActivity → Native.setSurface → nxhost_set_window → vka_init) comes before
// Native.start points stderr at the run log (nxlaunch.c log_open); until then stderr is the app process's /dev/null, so the device's setup lines (driver, extensions,
// the GPU clock floor's kernel context and DCVS mode) reached neither the run log nor logcat: pf1 and play1 had none of them, only the floor's first line, which the first
// submission writes. While stderr is /dev/null the lines are kept here as well, and written once stderr is another file, at this file's next line or vka_init call after that
// (the first submission). Lines of the other files made in that time (vka_driver.c, vka_present.c) are still lost
static pthread_mutex_t g_early_mu = PTHREAD_MUTEX_INITIALIZER;
static _Atomic int g_early_st;   // 0 not looked at yet, 1 stderr is /dev/null: the lines go into g_early as well, 2 written (or stderr was a file from the start)
static char *g_early; static size_t g_early_n, g_early_cap;
static int stderr_is_null(void) { char to[32]; ssize_t n = readlink("/proc/self/fd/2", to, sizeof to - 1); if (n <= 0) return 1; to[n] = 0; return !strcmp(to, "/dev/null"); }
static void early_flush_locked(void) {   // (under g_early_mu)
    if (atomic_load(&g_early_st) == 0) atomic_store(&g_early_st, stderr_is_null() ? 1 : 2);
    if (atomic_load(&g_early_st) != 1 || stderr_is_null()) return;
    atomic_store(&g_early_st, 2);
    if (g_early_n) LOG("[vk] the device's lines from before the run log opened (stderr was /dev/null until then):");
    for (char *s = g_early, *e; s && s < g_early + g_early_n && (e = memchr(s, '\n', (size_t)(g_early + g_early_n - s))); s = e + 1) LOG("%.*s", (int)(e - s), s);
    free(g_early); g_early = NULL; g_early_n = g_early_cap = 0;
}
static void early_flush(void) { if (atomic_load_explicit(&g_early_st, memory_order_relaxed) == 2) return; pthread_mutex_lock(&g_early_mu); early_flush_locked(); pthread_mutex_unlock(&g_early_mu); }
static void dev_log(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void dev_log(const char *fmt, ...) {
    char sb[1024], *s = sb; va_list ap, aq; va_start(ap, fmt); va_copy(aq, ap);
    int n = vsnprintf(sb, sizeof sb, fmt, ap); va_end(ap);
    if (n >= (int)sizeof sb) { char *h = malloc((size_t)n + 1); if (h) { vsnprintf(h, (size_t)n + 1, fmt, aq); s = h; } }   // (else the line cut at 1023 bytes)
    va_end(aq);
    if (atomic_load_explicit(&g_early_st, memory_order_relaxed) != 2) {
        pthread_mutex_lock(&g_early_mu);
        early_flush_locked();
        if (atomic_load(&g_early_st) == 1) {   // at most 64 KiB of them (the setup writes a few)
            size_t l = strlen(s), need = g_early_n + l + 1;
            if (need > g_early_cap && need <= (64u << 10)) { size_t c = g_early_cap ? g_early_cap : 8192; while (c < need) c *= 2; char *p = realloc(g_early, c); if (p) { g_early = p; g_early_cap = c; } }
            if (need <= g_early_cap) { memcpy(g_early + g_early_n, s, l); g_early[g_early_n + l] = '\n'; g_early_n = need; }
        }
        pthread_mutex_unlock(&g_early_mu);
    }
    LOG("%s", s);
    if (s != sb) free(s);
}
#undef VLOG
#undef VRED
#define VLOG(...) dev_log("[vk] " __VA_ARGS__)
#define VRED(...) dev_log("[RED] vk: " __VA_ARGS__)

vka_fns vk;
vka_dev V = {.qmu = PTHREAD_MUTEX_INITIALIZER};
static int g_state;   // 0 not tried, 1 up, -1 failed
static pthread_mutex_t g_init_mu = PTHREAD_MUTEX_INITIALIZER;


uint32_t vka_mem_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < V.memp.memoryTypeCount; i++)
        if ((bits >> i & 1) && (V.memp.memoryTypes[i].propertyFlags & want) == want && !(V.memp.memoryTypes[i].propertyFlags & avoid)) return i;
    return UINT32_MAX;
}

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) { for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return 1; return 0; }

static void *completion_main(void *arg);
// DM_VK_MEMREPORT=1 (restart only: the device's extensions; default off; a diagnostic): the driver reports every allocation it makes for the device, its own ones too
// (VK_EXT_device_memory_report), and a thread logs every 10 s what is live by object type and heap: "memory report:" lines. Found with it: what the driver holds beyond the
// memory the host allocates itself (the game's note, "Galaxy S26 Ultra"). The callback runs on any thread, inside the driver: it only updates the table below under its lock
#define MR_SLOTS (1u << 19)
#define MR_TYPES 40   // VkObjectType 0..38 by value, 39 every other one
static struct { uint64_t id, size; uint8_t type, heap, used; } *g_mr;
static pthread_mutex_t g_mr_mu = PTHREAD_MUTEX_INITIALIZER;
static uint64_t g_mr_live[MR_TYPES], g_mr_n[MR_TYPES], g_mr_allocs[MR_TYPES], g_mr_frees[MR_TYPES], g_mr_heap[8], g_mr_failed, g_mr_lost, g_mr_unknown_free, g_mr_freed_b[MR_TYPES], g_mr_big_frees;
static uint32_t mr_slot(uint64_t id) { return (uint32_t)((id * 0x9e3779b97f4a7c15ull) >> 45) & (MR_SLOTS - 1); }
static void VKAPI_PTR mr_cb(const VkDeviceMemoryReportCallbackDataEXT *d, void *user) {
    (void)user; int t = (uint32_t)d->objectType < MR_TYPES - 1 ? (int)d->objectType : MR_TYPES - 1;
    pthread_mutex_lock(&g_mr_mu);
    if (d->type == VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_ALLOCATE_EXT || d->type == VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_IMPORT_EXT) {
        uint32_t i = mr_slot(d->memoryObjectId), k = 0;
        for (; k < 256 && g_mr[i].used == 1 && g_mr[i].id != d->memoryObjectId; k++) i = (i + 1) & (MR_SLOTS - 1);
        if (k == 256) g_mr_lost++;
        else { if (g_mr[i].used == 1) { g_mr_live[g_mr[i].type] -= g_mr[i].size; g_mr_n[g_mr[i].type]--; g_mr_heap[g_mr[i].heap] -= g_mr[i].size; }   // the id again: replaced
               g_mr[i].id = d->memoryObjectId; g_mr[i].size = d->size; g_mr[i].type = (uint8_t)t; g_mr[i].heap = (uint8_t)(d->heapIndex & 7); g_mr[i].used = 1;
               g_mr_live[t] += d->size; g_mr_n[t]++; g_mr_allocs[t]++; g_mr_heap[d->heapIndex & 7] += d->size; }
    } else if (d->type == VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_FREE_EXT || d->type == VK_DEVICE_MEMORY_REPORT_EVENT_TYPE_UNIMPORT_EXT) {
        uint32_t i = mr_slot(d->memoryObjectId), k = 0;
        for (; k < 256 && g_mr[i].used && !(g_mr[i].used == 1 && g_mr[i].id == d->memoryObjectId); k++) i = (i + 1) & (MR_SLOTS - 1);
        if (k == 256 || !g_mr[i].used) g_mr_unknown_free++;
        else { g_mr_live[g_mr[i].type] -= g_mr[i].size; g_mr_n[g_mr[i].type]--; g_mr_frees[g_mr[i].type]++; g_mr_freed_b[g_mr[i].type] += g_mr[i].size; g_mr_heap[g_mr[i].heap] -= g_mr[i].size; g_mr[i].used = 2;
               if (g_mr[i].size >= (4u << 20) && g_mr_big_frees++ < 60) LOG("[vk] memory report: freed %.1f MiB of object type %d (one of the first 60 frees of 4 MiB or more)", (double)g_mr[i].size / 1048576.0, (int)g_mr[i].type); }   // 2: a tombstone
    } else g_mr_failed++;
    pthread_mutex_unlock(&g_mr_mu);
}
static void *mr_main(void *arg) {
    (void)arg;
    static const char *const names[MR_TYPES] = {"unknown", "instance", "physical device", "device", "queue", "semaphore", "command buffer", "fence", "device memory", "buffer", "image", "event",
        "query pool", "buffer view", "image view", "shader module", "pipeline cache", "pipeline layout", "render pass", "pipeline", "set layout", "sampler", "descriptor pool", "descriptor set",
        "framebuffer", "command pool", [MR_TYPES - 1] = "other"};
    for (int tick = 1;; tick++) {
        sleep(10);
        uint64_t live[MR_TYPES], n[MR_TYPES], al[MR_TYPES], fr[MR_TYPES], fb[MR_TYPES], heap[8], failed, lost, unk, total = 0, freed = 0; char b[1400]; size_t bn = 0; b[0] = 0;
        pthread_mutex_lock(&g_mr_mu);
        memcpy(live, g_mr_live, sizeof live); memcpy(n, g_mr_n, sizeof n); memcpy(al, g_mr_allocs, sizeof al); memcpy(fr, g_mr_frees, sizeof fr); memcpy(fb, g_mr_freed_b, sizeof fb); memcpy(heap, g_mr_heap, sizeof heap);
        failed = g_mr_failed; lost = g_mr_lost; unk = g_mr_unknown_free;
        pthread_mutex_unlock(&g_mr_mu);
        for (int t = 0; t < MR_TYPES; t++) { total += live[t]; freed += fb[t];
            if ((al[t] || n[t]) && bn < sizeof b) bn += (size_t)snprintf(b + bn, sizeof b - bn, "; %s %.1f MiB in %llu (%llu made, %llu freed: %.1f MiB)", names[t] ? names[t] : "?", (double)live[t] / 1048576.0,
                                                                         (unsigned long long)n[t], (unsigned long long)al[t], (unsigned long long)fr[t], (double)fb[t] / 1048576.0); }
        VLOG("memory report: at %d s the driver holds %.1f MiB (heap 0: %.1f, heap 1: %.1f), freed so far %.1f MiB%s; %llu failed, %llu not tracked, %llu frees of unknown ids (DM_VK_MEMREPORT)", tick * 10,
             (double)total / 1048576.0, (double)heap[0] / 1048576.0, (double)heap[1] / 1048576.0, (double)freed / 1048576.0, b, (unsigned long long)failed, (unsigned long long)lost, (unsigned long long)unk);
    }
    return NULL;
}

static int init_once(void) {
    void *lib = vka_driver_open();   // DM_VK_DRIVER: a custom driver through libadrenotools (vka_driver.c); NULL = the system's
    if (!lib) lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib || !(vk.vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr"))) { VRED("no libvulkan.so: %s", dlerror()); return 0; }
#define G(f) vk.f = (PFN_##f)vk.vkGetInstanceProcAddr(NULL, #f);
    VKA_GLOBAL_FNS(G)
#undef G
    uint32_t api = VK_API_VERSION_1_0; if (vk.vkEnumerateInstanceVersion) vk.vkEnumerateInstanceVersion(&api);
    if (api < VK_API_VERSION_1_3) { VRED("Vulkan instance %u.%u: the replay needs 1.3", VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api)); return 0; }
    uint32_t ne = 0; vk.vkEnumerateInstanceExtensionProperties(NULL, &ne, NULL);
    VkExtensionProperties *ie = calloc(ne ? ne : 1, sizeof *ie); vk.vkEnumerateInstanceExtensionProperties(NULL, &ne, ie);
    const char *iext[4]; uint32_t niext = 0;
    if (has_ext(ie, ne, VK_KHR_SURFACE_EXTENSION_NAME) && has_ext(ie, ne, VK_KHR_ANDROID_SURFACE_EXTENSION_NAME)) { iext[niext++] = VK_KHR_SURFACE_EXTENSION_NAME; iext[niext++] = VK_KHR_ANDROID_SURFACE_EXTENSION_NAME; V.has_android_surface = 1; }
    free(ie);
    const char *layer = "VK_LAYER_KHRONOS_validation"; uint32_t nlayer = 0;
    const char *ve = getenv("DM_VK_VALIDATE");
    if (ve && atoi(ve)) {
        uint32_t nl = 0; vk.vkEnumerateInstanceLayerProperties(&nl, NULL); VkLayerProperties *lp = calloc(nl ? nl : 1, sizeof *lp); vk.vkEnumerateInstanceLayerProperties(&nl, lp);
        for (uint32_t i = 0; i < nl; i++) if (!strcmp(lp[i].layerName, layer)) nlayer = 1;
        free(lp); VLOG("DM_VK_VALIDATE: validation layer %s", nlayer ? "on" : "not installed");
    }
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "Anka Native", .pEngineName = "Anka Native", .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai, .enabledExtensionCount = niext, .ppEnabledExtensionNames = iext,
                                .enabledLayerCount = nlayer, .ppEnabledLayerNames = &layer};
    VkResult r = vk.vkCreateInstance(&ici, NULL, &V.inst);
    if (r) { VRED("vkCreateInstance %d", r); return 0; }
#define I(f) vk.f = (PFN_##f)vk.vkGetInstanceProcAddr(V.inst, #f);
    VKA_INSTANCE_FNS(I)
#undef I
    uint32_t npd = 1; VkPhysicalDevice pd[1];
    if (vk.vkEnumeratePhysicalDevices(V.inst, &npd, pd) < 0 || !npd) { VRED("no Vulkan physical device"); return 0; }
    V.pd = pd[0];
    vk.vkGetPhysicalDeviceProperties(V.pd, &V.props); vk.vkGetPhysicalDeviceMemoryProperties(V.pd, &V.memp);
    if (V.props.apiVersion < VK_API_VERSION_1_3) { VRED("device %s is Vulkan %u.%u: the replay needs 1.3", V.props.deviceName, VK_API_VERSION_MAJOR(V.props.apiVersion), VK_API_VERSION_MINOR(V.props.apiVersion)); return 0; }
    uint32_t nq = 0; vk.vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp); vk.vkGetPhysicalDeviceQueueFamilyProperties(V.pd, &nq, qp);
    V.qf = UINT32_MAX; for (uint32_t i = 0; i < nq && V.qf == UINT32_MAX; i++) if ((qp[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) V.qf = i;
    { VkPhysicalDeviceFeatures pf; vk.vkGetPhysicalDeviceFeatures(V.pd, &pf); const VkPhysicalDeviceSparseProperties *sp = &V.props.sparseProperties;   // sparse residency: for sparse NVN
      VLOG("sparse: binding %d, residency buffer %d image2D %d image3D %d aliased %d, standard 2D block shape %d, non-resident strict %d, address space %llu MiB, sparse binding on queue family %u: %s; "
           "buffer / image granularity %llu",   // pools' textures (docs/reference/nvn-objects.md "Sparse texture arrays made at their full size")
           pf.sparseBinding, pf.sparseResidencyBuffer, pf.sparseResidencyImage2D, pf.sparseResidencyImage3D, pf.sparseResidencyAliased, sp->residencyStandard2DBlockShape, sp->residencyNonResidentStrict,
           (unsigned long long)(V.props.limits.sparseAddressSpaceSize >> 20), V.qf, V.qf < nq && (qp[V.qf].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) ? "yes" : "no",
           (unsigned long long)V.props.limits.bufferImageGranularity);
      // Sparse residency images (vka_sparse.c) need it all: binding on the replay's queue, 2D residency, the standard block shape (NVN's 64 KB tiles) and strict non-resident reads
      // (an unbound tile reads 0, as an unmapped NVN page). The features are enabled with every other supported one (vkGetPhysicalDeviceFeatures2 below)
      V.sparse_res = pf.sparseBinding && pf.sparseResidencyImage2D && sp->residencyStandard2DBlockShape && sp->residencyNonResidentStrict && V.qf < nq && (qp[V.qf].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT);
      // Sparse residency buffers (vka_mem.c span buffers: a pool placed by parts as one buffer): binding on the replay's queue, buffer residency and strict non-resident reads
      V.sparse_buf = pf.sparseBinding && pf.sparseResidencyBuffer && sp->residencyNonResidentStrict && V.qf < nq && (qp[V.qf].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT); }
    free(qp);
    if (V.qf == UINT32_MAX) { VRED("no graphics + compute queue family"); return 0; }

    uint32_t nde = 0; vk.vkEnumerateDeviceExtensionProperties(V.pd, NULL, &nde, NULL);
    VkExtensionProperties *de = calloc(nde ? nde : 1, sizeof *de); vk.vkEnumerateDeviceExtensionProperties(V.pd, NULL, &nde, de);
    const char *dext[20]; uint32_t ndext = 0;
    { char q[700]; size_t qn = 0; q[0] = 0;   // the vendor's own extensions (tile properties, render pass transforms ...): what a tiler-specific path could use
      for (uint32_t i = 0; i < nde && qn < sizeof q; i++) if (strstr(de[i].extensionName, "_QCOM_")) qn += (size_t)snprintf(q + qn, sizeof q - qn, " %s", de[i].extensionName + 3);
      if (qn) VLOG("device extensions of the vendor:%s", q); }
    int swap = has_ext(de, nde, VK_KHR_SWAPCHAIN_EXTENSION_NAME), map2 = has_ext(de, nde, VK_KHR_MAP_MEMORY_2_EXTENSION_NAME), placed = map2 && has_ext(de, nde, VK_EXT_MAP_MEMORY_PLACED_EXTENSION_NAME);
    V.display_timing = swap && has_ext(de, nde, VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME);
    if (swap) dext[ndext++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    if (map2) dext[ndext++] = VK_KHR_MAP_MEMORY_2_EXTENSION_NAME;
    if (placed) dext[ndext++] = VK_EXT_MAP_MEMORY_PLACED_EXTENSION_NAME;
    if (V.display_timing) dext[ndext++] = VK_GOOGLE_DISPLAY_TIMING_EXTENSION_NAME;
    // The draw path (vka_draw.c): each one optional, with a fallback when absent
    int x_push = has_ext(de, nde, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME), x_vin = has_ext(de, nde, VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);
    const char *x_rob = has_ext(de, nde, VK_KHR_ROBUSTNESS_2_EXTENSION_NAME) ? VK_KHR_ROBUSTNESS_2_EXTENSION_NAME : has_ext(de, nde, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME) ? VK_EXT_ROBUSTNESS_2_EXTENSION_NAME : NULL;
    const char *x_i8 = has_ext(de, nde, VK_KHR_INDEX_TYPE_UINT8_EXTENSION_NAME) ? VK_KHR_INDEX_TYPE_UINT8_EXTENSION_NAME : has_ext(de, nde, VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME) ? VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME : NULL;
    const char *x_div = has_ext(de, nde, VK_KHR_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME) ? VK_KHR_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME : has_ext(de, nde, VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME) ? VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME : NULL;
    if (x_push) dext[ndext++] = VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
    if (x_vin) dext[ndext++] = VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME;
    if (x_rob) dext[ndext++] = x_rob;
    if (x_i8) dext[ndext++] = x_i8;
    if (x_div) dext[ndext++] = x_div;
    int x_gpl = has_ext(de, nde, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME) && has_ext(de, nde, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME);
    if (x_gpl) { dext[ndext++] = VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME; dext[ndext++] = VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME; }
    int mesh_ext = has_ext(de, nde, VK_EXT_MESH_SHADER_EXTENSION_NAME); const char *mesh_env = getenv("DM_VK_MESH");   // restart only: the device's extensions
    int x_mesh = mesh_ext && !(mesh_env && !strcmp(mesh_env, "0"));
    if (x_mesh) dext[ndext++] = VK_EXT_MESH_SHADER_EXTENSION_NAME;
    int x_dcc = has_ext(de, nde, VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME);   // TEMP diagnostic (DM_VK_MESH_SPEC=4)
    if (x_dcc) dext[ndext++] = VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME;
    int x_ph = has_ext(de, nde, "VK_QCOM_queue_perf_hint");
    if (x_ph) dext[ndext++] = "VK_QCOM_queue_perf_hint";
    const char *mr_env = getenv("DM_VK_MEMREPORT"); int x_mr = mr_env && !strcmp(mr_env, "1") && has_ext(de, nde, VK_EXT_DEVICE_MEMORY_REPORT_EXTENSION_NAME);
    if (x_mr && !(g_mr = calloc(MR_SLOTS, sizeof *g_mr))) x_mr = 0;
    if (x_mr) dext[ndext++] = VK_EXT_DEVICE_MEMORY_REPORT_EXTENSION_NAME;
    free(de);

    // Every feature the device has, but robustness and capture-replay (they cost and nothing needs them); the shader side takes what it needs from the same device
    VkPhysicalDeviceMapMemoryPlacedFeaturesEXT fmp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_FEATURES_EXT};
    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT fvin = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT};
    VkPhysicalDeviceRobustness2FeaturesKHR frob = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_KHR};
    VkPhysicalDeviceIndexTypeUint8Features fi8 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INDEX_TYPE_UINT8_FEATURES};
    VkPhysicalDeviceVertexAttributeDivisorFeatures fdiv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES};
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT fgpl = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT};
    VkPhysicalDeviceMeshShaderFeaturesEXT fmesh = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    void *chain = placed ? &fmp : NULL;
#define LINK(on, st) if (on) { (st).pNext = chain; chain = &(st); }
    struct { VkStructureType sType; void *pNext; VkBool32 queuePerfHint; } fph = {(VkStructureType)1000302001};   // VkPhysicalDeviceQueuePerfHintFeaturesQCOM (not in the NDK's headers yet)
    VkPhysicalDeviceDeviceMemoryReportFeaturesEXT fmr = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_MEMORY_REPORT_FEATURES_EXT};
    VkPhysicalDeviceDepthClipControlFeaturesEXT fdcc = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT};   // TEMP diagnostic
    LINK(x_mr, fmr)
    LINK(x_dcc, fdcc)
    LINK(x_vin, fvin) LINK(x_rob, frob) LINK(x_i8, fi8) LINK(x_div, fdiv) LINK(x_ph, fph) LINK(x_gpl, fgpl) LINK(x_mesh, fmesh)
#undef LINK
    VkPhysicalDeviceVulkan13Features f13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .pNext = chain};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .pNext = &f13};
    VkPhysicalDeviceVulkan11Features f11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &f12};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f11};
    vk.vkGetPhysicalDeviceFeatures2(V.pd, &f2);
    f2.features.robustBufferAccess = VK_FALSE; f13.robustImageAccess = VK_FALSE;
    f12.bufferDeviceAddressCaptureReplay = VK_FALSE; f12.bufferDeviceAddressMultiDevice = VK_FALSE;
    frob.robustBufferAccess2 = VK_FALSE; frob.robustImageAccess2 = VK_FALSE;   // only nullDescriptor: unbound slots read a null descriptor
    // Mesh shading: the task and mesh stages only (multiview, shading-rate and query support each need more of the device than the replay enables or uses)
    fmesh.multiviewMeshShader = VK_FALSE; fmesh.primitiveFragmentShadingRateMeshShader = VK_FALSE; fmesh.meshShaderQueries = VK_FALSE;
    if (x_mesh && !fmesh.meshShader) fmesh.taskShader = VK_FALSE;   // (task shading is of no use without mesh shading)
    V.push_desc = x_push; V.vin_dyn = x_vin && fvin.vertexInputDynamicState; V.null_desc = x_rob && frob.nullDescriptor; V.idx8 = x_i8 && fi8.indexTypeUint8;
    V.divisor = x_div && fdiv.vertexAttributeInstanceRateDivisor; V.logic_op = f2.features.logicOp; V.dual_src = f2.features.dualSrcBlend; V.geom = f2.features.geometryShader;
    V.tess = f2.features.tessellationShader; V.depth_clamp = f2.features.depthClamp; V.fill_lines = f2.features.fillModeNonSolid; V.depth_bias_clamp = f2.features.depthBiasClamp;
    V.multi_vp = f2.features.multiViewport; V.pstats = f2.features.pipelineStatisticsQuery; V.a2c = 1; V.mesh = x_mesh && fmesh.meshShader; V.task = V.mesh && fmesh.taskShader;
    V.depth_clip_control = x_dcc && fdcc.depthClipControl; VLOG("TEMP depth clip control %d", V.depth_clip_control);   // TEMP diagnostic
    if (!f12.timelineSemaphore || !f13.dynamicRendering) { VRED("the device lacks timeline semaphores (%d) or dynamic rendering (%d)", f12.timelineSemaphore, f13.dynamicRendering); return 0; }
    V.bda = f12.bufferDeviceAddress; V.uab = f12.descriptorBindingSampledImageUpdateAfterBind && f12.descriptorBindingPartiallyBound && f12.runtimeDescriptorArray;
    V.placed = placed && fmp.memoryMapPlaced; V.placed_range = placed && fmp.memoryMapRangePlaced; V.unmap_reserve = placed && fmp.memoryUnmapReserve;
    float prio = 1.f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = V.qf, .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci, .enabledExtensionCount = ndext, .ppEnabledExtensionNames = dext};
    VkDeviceDeviceMemoryReportCreateInfoEXT mrci = {VK_STRUCTURE_TYPE_DEVICE_DEVICE_MEMORY_REPORT_CREATE_INFO_EXT, .pNext = &f2, .pfnUserCallback = mr_cb};
    if (x_mr && fmr.deviceMemoryReport) { dci.pNext = &mrci; pthread_t th; if (!pthread_create(&th, NULL, mr_main, NULL)) pthread_detach(th); }
    VLOG("driver memory report (VK_EXT_device_memory_report): %s", x_mr && fmr.deviceMemoryReport ? "on (DM_VK_MEMREPORT=1)" : mr_env && !strcmp(mr_env, "1") ? "asked for, not on this driver" : "off");
    void vka_kgsl_before(int has_ext), vka_kgsl_after(int has_hint); vka_kgsl_before(x_ph);   // the kernel contexts the device's queue gets (the KGSL floor below)
    if ((r = vk.vkCreateDevice(V.pd, &dci, NULL, &V.dev))) { VRED("vkCreateDevice %d", r); return 0; }
#define D(f) vk.f = (PFN_##f)vk.vkGetDeviceProcAddr(V.dev, #f);
    VKA_DEVICE_FNS(D)
#undef D
    if (!vk.vkMapMemory2KHR) vk.vkMapMemory2KHR = (PFN_vkMapMemory2KHR)vk.vkGetDeviceProcAddr(V.dev, "vkMapMemory2");   // core in 1.4
    if (!vk.vkUnmapMemory2KHR) vk.vkUnmapMemory2KHR = (PFN_vkUnmapMemory2KHR)vk.vkGetDeviceProcAddr(V.dev, "vkUnmapMemory2");
    if (!vk.vkMapMemory2KHR || !vk.vkUnmapMemory2KHR) V.placed = V.placed_range = 0;
    vk.vkGetDeviceQueue(V.dev, V.qf, 0, &V.q);
    if (x_ph && fph.queuePerfHint) V.perf_hint = (void *)vk.vkGetDeviceProcAddr(V.dev, "vkQueueSetPerfHintQCOM");
    VLOG("GPU clock floor (VK_QCOM_queue_perf_hint): %s", V.perf_hint ? "available (DM_VK_GPU_FLOOR)" : x_ph ? "extension without the feature" : "not on this device");
    vka_kgsl_after(V.perf_hint != NULL);
#if defined(__ANDROID__)
    { int vka_floor_adpf(void), vka_floor_kgsl(void); if (!V.perf_hint && !vka_floor_kgsl()) VLOG("GPU clock floor through ADPF: %s", vka_floor_adpf() ? "each frame's span and GPU time reported against the game's shortest period (DM_VK_GPU_FLOOR_ADPF)" : "off (DM_VK_GPU_FLOOR=0 or DM_VK_GPU_FLOOR_ADPF=0)"); }
#endif

    VkPhysicalDeviceMapMemoryPlacedPropertiesEXT pmp = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAP_MEMORY_PLACED_PROPERTIES_EXT};
    VkPhysicalDeviceVulkan13Properties p13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES, .pNext = placed ? &pmp : NULL};
    VkPhysicalDeviceVulkan11Properties p11 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, .pNext = &p13};
    VkPhysicalDevicePushDescriptorPropertiesKHR ppd = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR, .pNext = &p11};
    VkPhysicalDeviceMaintenance3Properties pm3 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES, .pNext = V.push_desc ? (void *)&ppd : (void *)&p11};
    VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT pgpl = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT, .pNext = &pm3};
    VkPhysicalDeviceDriverProperties pdr = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, .pNext = x_gpl ? (void *)&pgpl : (void *)&pm3};
    VkPhysicalDeviceMeshShaderPropertiesEXT pmesh = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT, .pNext = &pdr};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = x_mesh ? (void *)&pmesh : (void *)&pdr};
    vk.vkGetPhysicalDeviceProperties2(V.pd, &p2);
    VLOG("driver: %s (%s), driver ID %d, conformance %u.%u.%u.%u, driver version %#x; %s", pdr.driverName, pdr.driverInfo, pdr.driverID, pdr.conformanceVersion.major,
         pdr.conformanceVersion.minor, pdr.conformanceVersion.subminor, pdr.conformanceVersion.patch, V.props.driverVersion, vka_driver_tag() ? "custom (DM_VK_DRIVER)" : "the system's");
    V.driver_id = (int)pdr.driverID;
    { const char *e = getenv("DM_VK_GPL"); int want = !(e && !strcmp(e, "0"));   // restart only: the libraries are made with the device's pipelines
      V.gpl = want && x_gpl && fgpl.graphicsPipelineLibrary && pgpl.graphicsPipelineLibraryFastLinking;
      VLOG("graphics pipeline libraries: %s", V.gpl ? "on (a pipeline the cache lacks is linked from stage libraries, the optimized one is made in the background; DM_VK_GPL=0 off)"
           : !x_gpl ? "not on this device" : !fgpl.graphicsPipelineLibrary ? "extension without the feature" : !pgpl.graphicsPipelineLibraryFastLinking ? "no fast linking" : "off (DM_VK_GPL=0)"); }
    V.placed_align = pmp.minPlacedMemoryMapAlignment ? pmp.minPlacedMemoryMapAlignment : 4096; V.max_alloc = pm3.maxMemoryAllocationSize; V.max_buf = p13.maxBufferSize;
    V.max_push = V.push_desc ? ppd.maxPushDescriptors : 0;
    if (V.mesh && !vk.vkCmdDrawMeshTasksEXT) V.mesh = V.task = 0;
    if (V.mesh) { V.mesh_inv = pmesh.maxMeshWorkGroupInvocations; V.task_inv = pmesh.maxTaskWorkGroupInvocations; V.mesh_out_v = pmesh.maxMeshOutputVertices;
                  V.mesh_out_p = pmesh.maxMeshOutputPrimitives; V.task_payload = pmesh.maxTaskPayloadSize;
                  V.task_groups = pmesh.maxTaskWorkGroupCount[0]; V.mesh_groups = pmesh.maxMeshWorkGroupCount[0]; }
    VLOG("mesh shading (NVN2 task and mesh programs): %s", V.task ? "on" : V.mesh ? "mesh shaders only (no taskShader)" : !mesh_ext ? "not on this device" : !x_mesh ? "off (DM_VK_MESH=0)" : "extension without the mesh stage");
    if (V.mesh) VLOG("mesh shading limits: workgroups mesh %u / task %u invocations, mesh output %u vertices / %u primitives / %u components / %u bytes, task payload %u bytes, mesh shared %u bytes",
                     V.mesh_inv, V.task_inv, V.mesh_out_v, V.mesh_out_p, pmesh.maxMeshOutputComponents, pmesh.maxMeshOutputMemorySize, V.task_payload, pmesh.maxMeshSharedMemorySize);
    VLOG("subgroups: size %u, stages %#x (geometry %s, fragment %s, vertex %s), operations %#x, quad operations in all stages %s, required size stages %#x", p11.subgroupSize, p11.subgroupSupportedStages,
         p11.subgroupSupportedStages & VK_SHADER_STAGE_GEOMETRY_BIT ? "yes" : "NO", p11.subgroupSupportedStages & VK_SHADER_STAGE_FRAGMENT_BIT ? "yes" : "NO", p11.subgroupSupportedStages & VK_SHADER_STAGE_VERTEX_BIT ? "yes" : "NO",
         p11.subgroupSupportedOperations, p11.subgroupQuadOperationsInAllStages ? "yes" : "no", p13.requiredSubgroupSizeStages);
    V.subgroup_stages = p11.subgroupSupportedStages;
    V.subgroup_req_stages = f13.subgroupSizeControl ? p13.requiredSubgroupSizeStages : 0;
    V.subgroup_size = p11.subgroupSize; V.subgroup_min = p13.minSubgroupSize ? p13.minSubgroupSize : p11.subgroupSize; V.subgroup_max = p13.maxSubgroupSize ? p13.maxSubgroupSize : p11.subgroupSize;
    if (!vk.vkCmdPushDescriptorSetKHR) vk.vkCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)vk.vkGetDeviceProcAddr(V.dev, "vkCmdPushDescriptorSet");   // core in 1.4
    if (!vk.vkCmdPushDescriptorSetKHR) V.push_desc = 0, V.max_push = 0;
    if (!vk.vkCmdSetVertexInputEXT) V.vin_dyn = 0;
    vka_enc_install();
    const VkPhysicalDeviceLimits *lim = &V.props.limits;
    V.ub_align = lim->minUniformBufferOffsetAlignment ? lim->minUniformBufferOffsetAlignment : 256; V.sb_align = lim->minStorageBufferOffsetAlignment ? lim->minStorageBufferOffsetAlignment : 256;
    V.ub_range = lim->maxUniformBufferRange; V.sb_range = lim->maxStorageBufferRange; V.max_vin_bindings = lim->maxVertexInputBindings;

    VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    V.guest_type = vka_mem_type(~0u, HV | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
    if (V.guest_type == UINT32_MAX) V.guest_type = vka_mem_type(~0u, HV | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
    if (V.guest_type == UINT32_MAX) V.guest_type = vka_mem_type(~0u, HV, 0);
    V.upload_type = V.guest_type;
    V.local_type = vka_mem_type(~0u, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);

    VkSemaphoreTypeCreateInfo st = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
    VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
    if ((r = vk.vkCreateSemaphore(V.dev, &sci, NULL, &V.tl))) { VRED("timeline semaphore %d", r); return 0; }
    pthread_t th; if (pthread_create(&th, NULL, completion_main, NULL)) { VRED("cannot start the completion thread"); return 0; }
    pthread_detach(th);
    VLOG("device %s, Vulkan %u.%u.%u, driver %#x; placed maps %s (range %s, alignment %llu), buffer device address %s, update-after-bind %s, display timing %s; guest memory type %u (%#x), max allocation %llu MiB, at most %u allocations",
         V.props.deviceName, VK_API_VERSION_MAJOR(V.props.apiVersion), VK_API_VERSION_MINOR(V.props.apiVersion), VK_API_VERSION_PATCH(V.props.apiVersion), V.props.driverVersion,
         V.placed ? "yes" : "no", V.placed_range ? "yes" : "no", (unsigned long long)V.placed_align, V.bda ? "yes" : "no", V.uab ? "yes" : "no", V.display_timing ? "yes" : "no",
         V.guest_type, V.guest_type < 32 ? V.memp.memoryTypes[V.guest_type].propertyFlags : 0, (unsigned long long)(V.max_alloc >> 20), V.props.limits.maxMemoryAllocationCount);   // (images take one allocation each: Mario Kart World ran out of them)
    VLOG("draw path: push descriptors %s (%u), vertex input dynamic state %s, null descriptors %s, 8-bit indices %s, instance divisors %s, logic op %s, dual source %s, geometry %s, tessellation %s; "
         "uniform alignment %llu (range %llu), storage alignment %llu, subgroups %u (%u-%u)",
         V.push_desc ? "yes" : "no", V.max_push, V.vin_dyn ? "yes" : "no", V.null_desc ? "yes" : "no", V.idx8 ? "yes" : "no", V.divisor ? "yes" : "no", V.logic_op ? "yes" : "no",
         V.dual_src ? "yes" : "no", V.geom ? "yes" : "no", V.tess ? "yes" : "no", (unsigned long long)V.ub_align, (unsigned long long)V.ub_range, (unsigned long long)V.sb_align, V.subgroup_size, V.subgroup_min, V.subgroup_max);
    if (!V.placed) VRED("no VK_EXT_map_memory_placed: guest memory stays anonymous and the GPU sees no memory pool");
    if (!swap) VRED("no VK_KHR_swapchain: nothing is presented");
    vka_present_init();
    return 1;
}
int vka_init(void) {
    early_flush();   // the setup's lines, once stderr is the run log
    if (__atomic_load_n(&g_state, __ATOMIC_ACQUIRE)) return g_state > 0;
    pthread_mutex_lock(&g_init_mu);
    if (!g_state) __atomic_store_n(&g_state, init_once() ? 1 : -1, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&g_init_mu);
    return g_state > 0;
}
int vka_ok(void) { return __atomic_load_n(&g_state, __ATOMIC_ACQUIRE) > 0; }

VkResult vka_submit(VkCommandBuffer cb, uint64_t signal, uint32_t nwait, const VkSemaphore *wait, const uint64_t *wait_vals, const VkPipelineStageFlags *stages, VkSemaphore bin_signal) {
    VkSemaphore sig[2]; uint64_t sv[2]; uint32_t ns = 0;
    if (signal) { sig[ns] = V.tl; sv[ns++] = signal; }
    if (bin_signal) { sig[ns] = bin_signal; sv[ns++] = 0; }
    // The waits, and one more for the sparse binds queued before it (vks_queue_locked)
    VkSemaphore ws[9]; uint64_t wv[9]; VkPipelineStageFlags wst[9];
    if (nwait > 8) { VRED_ONCE("vka_submit: %u waits, at most 8 taken: the rest are dropped", nwait); nwait = 8; }
    for (uint32_t i = 0; i < nwait; i++) { ws[i] = wait[i]; wv[i] = wait_vals[i]; wst[i] = stages[i]; }
    pthread_mutex_lock(&V.qmu);
    vks_queue_locked(&nwait, ws, wv, wst);
    VkTimelineSemaphoreSubmitInfo ti = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, .waitSemaphoreValueCount = nwait, .pWaitSemaphoreValues = wv, .signalSemaphoreValueCount = ns, .pSignalSemaphoreValues = sv};
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &ti, .waitSemaphoreCount = nwait, .pWaitSemaphores = ws, .pWaitDstStageMask = wst,
                       .commandBufferCount = cb ? 1 : 0, .pCommandBuffers = &cb, .signalSemaphoreCount = ns, .pSignalSemaphores = sig};
    VkResult r = vka_hung() ? VK_ERROR_DEVICE_LOST : vk.vkQueueSubmit(V.q, 1, &si, VK_NULL_HANDLE);
    if (!r && signal > V.tl_queued) __atomic_store_n(&V.tl_queued, signal, __ATOMIC_RELEASE);
    pthread_mutex_unlock(&V.qmu);
    return r;
}
// The queue idle, bounded by the hang time: the timeline up to the last value handed to it (the present's copies wait for no more), then vkQueueWaitIdle for the rest.
// An idle queue needs no turbo (DM_GPU_TURBO): it is taken back here, so at a suspend (mtl_drain: the app paused or in the background) and at a swapchain change, and the
// next submission sets it again (vka_gpu_floor)
static int queue_idle_(void);
void vka_turbo_set(int on, const char *why);
int vka_queue_idle(void) { int ok = queue_idle_(); vka_turbo_set(0, "the queue drained: a suspend or a swapchain change"); return ok; }
static int queue_idle_(void) {
    if (vka_hung()) return 0;
    uint64_t q = vka_fence_queued();
    VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &V.tl, .pValues = &q};
    VkResult r = q ? vk.vkWaitSemaphores(V.dev, &wi, vka_hang_ns()) : VK_SUCCESS;
    if (r) { VRED("queue idle: fence %llu not reached in %.1f s (%d): not waited for", (unsigned long long)q, vka_hang_ns() / 1e9, r); return 0; }
    pthread_mutex_lock(&V.qmu); vk.vkQueueWaitIdle(V.q); pthread_mutex_unlock(&V.qmu);
    return 1;
}
uint64_t vka_fence_queued(void) { return __atomic_load_n(&V.tl_queued, __ATOMIC_ACQUIRE); }

// ---------- Fences ----------
// Fence n is reserved by the submission that signals timeline value n. The completion thread waits for the timeline, then runs that submission's CPU-side effects in order
// and only then publishes n as done: a SyncWait that returns has seen the events and counters of everything before it, as nvn_mtl.m's post_complete does
enum { POST_RING = 1024 };
typedef struct { uint64_t fence; void *ops; int n; void (*run)(void *, int); } post_rec;
static post_rec g_post[POST_RING];
typedef struct after_rec { struct after_rec *next; uint64_t fence; void (*fn)(void *); void *arg; } after_rec;
static after_rec *g_after;
static uint64_t g_fence, g_submitted, g_done;
static pthread_mutex_t g_fmu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_fcv = PTHREAD_COND_INITIALIZER, g_scv = PTHREAD_COND_INITIALIZER;

uint64_t vka_fence_last(void) { pthread_mutex_lock(&g_fmu); uint64_t f = g_fence; pthread_mutex_unlock(&g_fmu); return f; }
uint64_t vka_fence_next(void) { pthread_mutex_lock(&g_fmu); uint64_t f = ++g_fence; pthread_mutex_unlock(&g_fmu); return f; }
void vka_fence_post(uint64_t fence, void *ops, int n, void (*run)(void *, int)) {
    pthread_mutex_lock(&g_fmu);
    while (g_post[fence % POST_RING].fence && g_post[fence % POST_RING].fence != fence) pthread_cond_wait(&g_fcv, &g_fmu);   // that slot's earlier fence has not completed yet
    g_post[fence % POST_RING] = (post_rec){fence, ops, n, run};
    if (fence > g_submitted) g_submitted = fence;
    pthread_cond_broadcast(&g_scv); pthread_mutex_unlock(&g_fmu);
}
void vka_fence_cpu(uint64_t fence, void *ops, int n, void (*run)(void *, int)) {
    if (run) run(ops, n); else free(ops);
    pthread_mutex_lock(&g_fmu);
    if (fence > g_submitted) g_submitted = fence;
    if (fence > g_done) __atomic_store_n(&g_done, fence, __ATOMIC_RELEASE);
    pthread_cond_broadcast(&g_fcv); pthread_mutex_unlock(&g_fmu);
}
uint64_t vka_done(void) { return __atomic_load_n(&g_done, __ATOMIC_ACQUIRE); }
int vka_wait_done(uint64_t fence, uint64_t timeout_ns) {
    if (vka_done() >= fence) return 1;
    if (!timeout_ns) return 0;
    struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl);
    uint64_t ns = (uint64_t)dl.tv_nsec + (timeout_ns > 3600000000000ull ? 3600000000000ull : timeout_ns);
    dl.tv_sec += (time_t)(ns / 1000000000ull); dl.tv_nsec = (long)(ns % 1000000000ull);
    pthread_mutex_lock(&g_fmu);
    while (g_done < fence) if (pthread_cond_timedwait(&g_fcv, &g_fmu, &dl)) break;
    int ok = g_done >= fence; pthread_mutex_unlock(&g_fmu);
    return ok;
}
void vka_after(uint64_t fence, void (*fn)(void *), void *arg) {
    pthread_mutex_lock(&g_fmu);
    if (g_done >= fence) { pthread_mutex_unlock(&g_fmu); fn(arg); return; }
    after_rec *a = malloc(sizeof *a); *a = (after_rec){g_after, fence, fn, arg}; g_after = a;
    pthread_mutex_unlock(&g_fmu);
}
// ---------- GPU hangs ----------
// A fence that makes no progress for DM_VK_HANG_S seconds (default 6) is a hang: the GPU, or the driver's queue, will not get there. Then nothing more is submitted (mtl_sub_begin
// records no command buffer, vka_submit refuses, the present drops its frames), every fence reads as done with its CPU-side effects run, pool memory the GPU may still read is
// kept instead of freed, and the host exits (dm_host_shutdown) unless DM_VK_HANG_EXIT=0. Before this, the completion thread logged the stall every 2 s and went on waiting, a pool
// given back was freed anyway after 2 s, and the present waited on the queue: on the Adreno 840 phone the system server restarted (a soft reboot) a few seconds later
static _Atomic int g_hung;
int vka_hung(void) { return atomic_load_explicit(&g_hung, memory_order_acquire); }
uint64_t vka_hang_ns(void) {
    double s = VKA_ENVF("DM_VK_HANG_S", 6.0); if (!(s > 0.5)) s = 6.0;
    return (uint64_t)(s * 1e9);
}
static void *hang_exit_main(void *arg) { (void)arg; pthread_setname_np(pthread_self(), "vk-hang"); dm_host_shutdown("GPU hang"); return NULL; }
void vka_exec_hang_report(uint64_t fence);   // vka_exec.c: what the stuck submission held
static void hang_declare(uint64_t want, uint64_t at, double s, const char *what) {
    if (atomic_exchange(&g_hung, 1)) return;
    VRED("GPU hang: %s; fence %llu not reached in %.1f s (the timeline at %llu, %llu handed to the queue, %llu reserved): nothing more is submitted, every fence reads as done",
         what, (unsigned long long)want, s, (unsigned long long)at, (unsigned long long)vka_fence_queued(), (unsigned long long)vka_fence_last());
    vka_exec_hang_report(at + 1);
    if (!VKA_ENV("DM_VK_HANG_EXIT", 1)) { VRED("GPU hang: DM_VK_HANG_EXIT=0, the game runs on without the GPU"); return; }
    VRED("GPU hang: the host exits (DM_VK_HANG_EXIT=0 runs on without the GPU)");
    pthread_t th; if (!pthread_create(&th, NULL, hang_exit_main, NULL)) pthread_detach(th);
}

static void *completion_main(void *arg) {
    (void)arg; pthread_setname_np(pthread_self(), "vk-complete"); void vka_perf_thread(int nice); vka_perf_thread(-8);   // vka_perf.c
    uint64_t stall_t0 = 0, stall_at = 0; int stall_logged = 0;
    for (;;) {
        pthread_mutex_lock(&g_fmu);
        while (g_done >= g_submitted) pthread_cond_wait(&g_scv, &g_fmu);
        // Wait for the next fence, not the last one handed to the queue: each fence is published (and its waiters woken) when the GPU reaches it. Waiting for the last one
        // held every earlier fence back until all work queued at that moment had run: Donkey Kong Bananza waits for its previous frame's last fence after queueing two
        // submissions of the next frame, so its wait ended only when those were done too. DM_VK_DONE_LAST=1 (live) waits for the last one, as before, for A/B runs
        uint64_t sub = g_submitted, done = g_done, want = VKA_ENV("DM_VK_DONE_LAST", 0) ? sub : done + 1; pthread_mutex_unlock(&g_fmu);
        uint64_t v = 0;
        if (!vka_hung()) {
            VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &V.tl, .pValues = &want};
            VkResult r = vk.vkWaitSemaphores(V.dev, &wi, 500000000ull);
            vk.vkGetSemaphoreCounterValue(V.dev, V.tl, &v);
            uint64_t t = now_ns();
            if (r == VK_TIMEOUT && v <= done) {   // no progress in this wait
                if (!stall_t0 || stall_at != done) { stall_t0 = t - 500000000ull; stall_at = done; stall_logged = 0; }
                double st = (t - stall_t0) / 1e9;
                if (st >= 2 && !stall_logged) { stall_logged = 1; VRED("GPU: fence %llu not reached after %.1f s (at %llu, %llu handed to the queue)", (unsigned long long)want, st, (unsigned long long)v, (unsigned long long)vka_fence_queued()); }
                if (t - stall_t0 < vka_hang_ns()) continue;
                hang_declare(want, v, st, "no progress");
            } else if (r && r != VK_TIMEOUT) {
                char w[64]; snprintf(w, sizeof w, "vkWaitSemaphores %d (the device is lost)", r);
                hang_declare(want, v, 0, w);
            } else stall_t0 = 0;
        }
        if (vka_hung()) v = sub;   // every fence submitted so far is taken as done: its CPU-side effects run, the guest's waits return
        if (v > sub) v = sub;
        for (uint64_t f = done + 1; f <= v; f++) {
            post_rec *p = &g_post[f % POST_RING];
            if (p->fence == f) { if (p->run) p->run(p->ops, p->n); else free(p->ops); }
        }
        pthread_mutex_lock(&g_fmu);
        for (uint64_t f = done + 1; f <= v; f++) if (g_post[f % POST_RING].fence == f) g_post[f % POST_RING].fence = 0;
        if (v > g_done) __atomic_store_n(&g_done, v, __ATOMIC_RELEASE);
        after_rec *run = NULL, **pp = &g_after;
        while (*pp) { after_rec *a = *pp; if (a->fence <= g_done) { *pp = a->next; a->next = run; run = a; } else pp = &a->next; }
        pthread_cond_broadcast(&g_fcv); pthread_mutex_unlock(&g_fmu);
        while (run) { after_rec *a = run; run = a->next; a->fn(a->arg); free(a); }
    }
    return NULL;
}

// ---------- Scratch memory ----------
// Blocks of host-visible memory handed out linearly. Two owners: the submission being recorded (class 0: the blocks retire at its fence, vka_scratch_retire) and the
// open upload batch (class 1, while vka_scr_up is set on the calling thread: the blocks retire at the batch's value on the upload timeline, vka_up_flush). A block goes
// back to use once that value is reached
typedef struct sblk { struct sblk *next; VkBuffer buf; VkDeviceMemory mem; uint8_t *ptr; VkDeviceSize size, used; uint64_t fence; int up; } sblk;
static sblk *g_open[2], *g_busy;
static _Atomic uint64_t g_scratch_bytes; static _Atomic uint32_t g_scratch_n;
static pthread_mutex_t g_smu = PTHREAD_MUTEX_INITIALIZER;
_Thread_local int vka_scr_up;
static VkSemaphore g_utl;   // the upload timeline: value n = upload batch n has completed
static VkDeviceSize g_ubytes;   // staging taken by the open upload batch (under g_smu)
static uint64_t up_done(void) {   // the upload timeline's value (every batch counts as done after a GPU hang)
    if (vka_hung()) return UINT64_MAX;
    uint64_t v = 0; if (g_utl) vk.vkGetSemaphoreCounterValue(V.dev, g_utl, &v); return v;
}
static int sblk_idle(const sblk *b, uint64_t done, uint64_t udone) { return b->fence <= (b->up ? udone : done); }
static sblk *sblk_new(VkDeviceSize n) {
    sblk *b = calloc(1, sizeof *b); b->size = n;
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    if (vk.vkCreateBuffer(V.dev, &bi, NULL, &b->buf)) { free(b); return NULL; }
    VkMemoryRequirements mr; vk.vkGetBufferMemoryRequirements(V.dev, b->buf, &mr);
    uint32_t t = vka_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
    if (t == UINT32_MAX) t = vka_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = t};
    void *p = NULL;
    if (t == UINT32_MAX || vk.vkAllocateMemory(V.dev, &ai, NULL, &b->mem) || vk.vkBindBufferMemory(V.dev, b->buf, b->mem, 0) || vk.vkMapMemory(V.dev, b->mem, 0, VK_WHOLE_SIZE, 0, &p)) {
        VRED("scratch block of %llu bytes not allocated", (unsigned long long)n); if (b->mem) vk.vkFreeMemory(V.dev, b->mem, NULL); vk.vkDestroyBuffer(V.dev, b->buf, NULL); free(b); return NULL; }
    b->ptr = p; atomic_fetch_add(&g_scratch_bytes, n); atomic_fetch_add(&g_scratch_n, 1); vka_mem_grew(n); return b;
}
uint64_t vka_scratch_bytes(uint32_t *blocks) { if (blocks) *blocks = atomic_load(&g_scratch_n); return atomic_load(&g_scratch_bytes); }
void vka_mem_grew(uint64_t bytes) {
    static _Atomic uint64_t acc, step = 1;
    uint64_t s = atomic_load_explicit(&step, memory_order_relaxed);
    if (s == 1) { const char *e = getenv("DM_VK_MEMLOG"); s = (e ? strtoull(e, NULL, 10) : 512) << 20; atomic_store_explicit(&step, s, memory_order_relaxed); }
    if (!s || atomic_fetch_add(&acc, bytes) + bytes < s) return;
    atomic_store(&acc, 0);
    int ni; uint32_t ns; uint64_t img = vka_tex_mem_nolock(&ni), scr = vka_scratch_bytes(&ns), pl = vka_mem_placed_nolock(), sp = vks_bytes_nolock();
    char tail[3600]; vka_tex_mem_paths(tail, sizeof tail);
    VLOG("memory: placed pools %.0f MiB, images %.0f MiB (%d), sparse tile blocks %.0f MiB, scratch %.0f MiB (%u blocks): %.0f MiB%s", pl / 1048576.0, img / 1048576.0, ni, sp / 1048576.0,
         scr / 1048576.0, ns, (pl + img + sp + scr) / 1048576.0, tail);
}
// Idle blocks (their value reached) larger than the standard 16 MiB, or beyond 64 MiB of idle blocks in all (DM_VK_SCRATCH_IDLE_MB), are freed (under g_smu): at every miss and at every retire.
// Kept, every block stayed for the process's life: Mario Kart World 1.8.0's boot uploads (levels of up to 144 MiB, each in a block of its size) left 1.4 GB of scratch
// before its first frame, and the phone's low-memory killer stopped it
// DM_VK_SCRATCH_IDLE_MB=<MiB> (live; default 64): how much of the idle standard blocks is kept (0: none). Each is host-visible device memory the phone's low-memory killer
// counts as the game's; a lower value gives it back sooner, at the cost of making a block again when the game takes more scratch (Ocarina of Time: 96 MiB of scratch at the kill)
static void sblk_trim(uint64_t done, uint64_t udone) {
    int keep_mb = VKA_ENV("DM_VK_SCRATCH_IDLE_MB", 64);
    VkDeviceSize idle = 0, keep = (VkDeviceSize)(keep_mb > 0 ? keep_mb : 0) << 20;
    for (sblk **pp = &g_busy; *pp;) {
        sblk *b = *pp;
        int id = sblk_idle(b, done, udone);
        if (id && (b->size > (16u << 20) || idle + b->size > keep)) {
            *pp = b->next; vk.vkUnmapMemory(V.dev, b->mem); vk.vkDestroyBuffer(V.dev, b->buf, NULL); vk.vkFreeMemory(V.dev, b->mem, NULL);
            atomic_fetch_sub(&g_scratch_bytes, b->size); atomic_fetch_sub(&g_scratch_n, 1); free(b);
        } else { if (id) idle += b->size; pp = &b->next; }
    }
}
int vka_scratch(VkDeviceSize n, VkDeviceSize align, vka_span *out) {
    if (!align) align = 16;
    int c = vka_scr_up ? 1 : 0;
    pthread_mutex_lock(&g_smu);
    sblk *b = g_open[c];
    VkDeviceSize o = b ? (b->used + align - 1) / align * align : 0;
    if (!b || o + n > b->size) {
        // Best fit among the idle blocks (a 144 MiB block kept for a 1 KB request stays alive), then the idle ones beyond the budget freed
        uint64_t done = vka_done(), udone = up_done(); sblk **best = NULL;
        for (sblk **pp = &g_busy; *pp; pp = &(*pp)->next) if ((*pp)->size >= n && sblk_idle(*pp, done, udone) && (!best || (*pp)->size < (*best)->size)) best = pp;
        b = NULL; if (best) { b = *best; *best = b->next; }
        sblk_trim(done, udone);
        if (!b) b = sblk_new(n > (16u << 20) ? (n + 0xFFFF) & ~(VkDeviceSize)0xFFFF : 16u << 20);
        if (!b) { pthread_mutex_unlock(&g_smu); return 0; }
        b->used = 0; b->up = c; b->next = g_open[c]; g_open[c] = b; o = 0;
    }
    b->used = o + n; *out = (vka_span){b->buf, b->ptr + o, o};
    if (c) g_ubytes += n;
    pthread_mutex_unlock(&g_smu);
    return 1;
}
static void retire(int c, uint64_t value) {
    pthread_mutex_lock(&g_smu);
    while (g_open[c]) { sblk *b = g_open[c]; g_open[c] = b->next; b->fence = value; b->next = g_busy; g_busy = b; }
    if (c) { g_ubytes = 0; sblk_trim(vka_done(), up_done()); }   // (the submission's blocks are trimmed at a miss only: a game that takes more than 16 MiB a frame would make and free them every frame)
    pthread_mutex_unlock(&g_smu);
}
void vka_scratch_retire(uint64_t fence) { retire(0, fence); }

// ---------- Upload batches ----------
// The first content of an image the replay makes (its upload from guest memory, or its clear) is recorded into a command buffer of its own, an upload batch, instead of
// the submission being recorded. The batch goes to the queue when its staging passes DM_VK_UPLOAD_BATCH_MB (live, default 64) and always before the replay's next part
// or submission goes (vka_exec.c: chunk, vka_rp_sub_commit), so the queue runs it before every command that uses the image; it ends with a full barrier, whose second
// scope covers every later submission. Its staging retires when the GPU reaches the batch's value on the upload timeline, not at the fence of the submission it was made
// in: one submission that makes hundreds of images (Mario Kart World 1.8.0 at boot: 29,000 images, 1.5 GB of staging in one submission) no longer holds all of it.
// When the batches in flight hold more than DM_VK_UPLOAD_INFLIGHT_MB (live, default 256) of staging, the replay thread waits for the oldest. Only a new image is safe here:
// nothing recorded earlier in the submission reads it (a re-upload of a live image stays in the submission, in order with the draws before it). DM_VK_UPLOAD_BATCH=0 (live)
// records first contents into the submission, as before
enum { NUP = 16 };
static struct { VkCommandBuffer cb; uint64_t val; } g_ub[NUP];
static VkCommandPool g_upool; static uint64_t g_uval; static int g_ui = -1; static unsigned g_unext;   // g_ui: the open batch's slot (-1: none); replay thread only
static uint64_t g_n_ubatch, g_n_uwait, g_uwait_ns;
static int up_wait(uint64_t v) {   // 1 = the upload timeline reached v
    if (up_done() >= v) return 1;
    VkSemaphoreWaitInfo wi = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &g_utl, .pValues = &v};
    uint64_t t0 = now_ns(); VkResult r = vk.vkWaitSemaphores(V.dev, &wi, vka_hang_ns());
    g_n_uwait++; g_uwait_ns += now_ns() - t0;
    if (r && !vka_hung()) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 4) VRED("upload batch %llu not done in %.1f s (%d)", (unsigned long long)v, vka_hang_ns() / 1e9, r); }
    return !r || vka_hung();
}
int vka_up_on(void) { return VKA_ENV("DM_VK_UPLOAD_BATCH", 1) != 0; }
VkCommandBuffer vka_up_begin(void) {
    if (g_ui < 0) {
        static int failed; if (failed) return VK_NULL_HANDLE;
        if (!g_upool) {
            VkCommandPoolCreateInfo pi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, .queueFamilyIndex = V.qf};
            VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = NUP};
            VkSemaphoreTypeCreateInfo st = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
            VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
            VkCommandBuffer cbs[NUP]; VkCommandPool p = VK_NULL_HANDLE;
            if (vk.vkCreateCommandPool(V.dev, &pi, NULL, &p) || ((ai.commandPool = p), vk.vkAllocateCommandBuffers(V.dev, &ai, cbs)) || vk.vkCreateSemaphore(V.dev, &sci, NULL, &g_utl)) {
                VRED_ONCE("upload batches not set up: first contents are recorded into the submission"); g_utl = VK_NULL_HANDLE; failed = 1; return VK_NULL_HANDLE; }
            for (int i = 0; i < NUP; i++) g_ub[i].cb = cbs[i];
            g_upool = p;
        }
        if (vka_hung()) return VK_NULL_HANDLE;
        unsigned k = g_unext % NUP;   // (16 slots: the backpressure below keeps them mostly done)
        if (g_ub[k].val && !up_wait(g_ub[k].val)) return VK_NULL_HANDLE;   // (its batch still runs: the caller records into the submission)
        g_unext++;
        vk.vkResetCommandBuffer(g_ub[k].cb, 0);
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        if (vk.vkBeginCommandBuffer(g_ub[k].cb, &bi)) return VK_NULL_HANDLE;
        g_ui = (int)k;
    }
    vka_scr_up = 1;
    return g_ub[g_ui].cb;
}
void vka_up_end(void) { vka_scr_up = 0; }
static int is_batch(VkCommandBuffer cb) { for (int i = 0; i < NUP; i++) if (cb && g_ub[i].cb == cb) return 1; return 0; }
// No batch can be begun (a slot's batch still runs after the hang time): the upload goes on in the submission being recorded, with its staging (the batch before it was sent,
// so its image's first barrier and copies come before; an upload stopped half-way would leave a tracked image with part of its content for good)
static VkCommandBuffer up_fallback(void) { vka_scr_up = 0; return vka_rp_cb(); }
VkCommandBuffer vka_up_cur(VkCommandBuffer cb) {   // the open batch when cb was one (vka_up_point may have sent it), else cb
    if (!is_batch(cb)) return cb;
    if (g_ui >= 0) return g_ub[g_ui].cb;
    VkCommandBuffer n = vka_up_begin(); return n ? n : up_fallback();
}
VkCommandBuffer vka_up_point(VkCommandBuffer cb) {
    if (g_ui < 0 || cb != g_ub[g_ui].cb) return cb;
    pthread_mutex_lock(&g_smu); VkDeviceSize n = g_ubytes; pthread_mutex_unlock(&g_smu);
    double mb = VKA_ENVF("DM_VK_UPLOAD_BATCH_MB", 64.0);
    if ((double)n < mb * 1048576.0) return cb;
    vka_up_flush();
    VkCommandBuffer nb = vka_up_begin(); return nb ? nb : up_fallback();
}
void vka_up_flush(void) {
    if (g_ui < 0) return;
    unsigned k = (unsigned)g_ui; g_ui = -1;
    VkCommandBuffer cb = g_ub[k].cb;
    vka_barrier(cb);   // everything the batch wrote, for every later submission on the queue
    VkResult r = vk.vkEndCommandBuffer(cb);
    uint64_t v = ++g_uval;
    VkSemaphore ws[1]; uint64_t wv[1]; VkPipelineStageFlags wst[1]; uint32_t nw = 0;   // the sparse binds queued before it (vks_queue_locked)
    VkTimelineSemaphoreSubmitInfo ti = {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO, .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &v};
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &ti, .commandBufferCount = 1, .pCommandBuffers = &cb, .signalSemaphoreCount = 1, .pSignalSemaphores = &g_utl};
    pthread_mutex_lock(&V.qmu);
    if (!r) {
        vks_queue_locked(&nw, ws, wv, wst);
        ti.waitSemaphoreValueCount = nw; ti.pWaitSemaphoreValues = wv; si.waitSemaphoreCount = nw; si.pWaitSemaphores = ws; si.pWaitDstStageMask = wst;
        r = vka_hung() ? VK_ERROR_DEVICE_LOST : vk.vkQueueSubmit(V.q, 1, &si, VK_NULL_HANDLE);
    }
    pthread_mutex_unlock(&V.qmu);
    // Nothing on the GPU will signal it: the host does, so later batches and the slot are not held; only once the batch before it is done (a host signal must be below
    // every value still pending on the queue)
    if (r && !vka_hung() && (v == 1 || up_wait(v - 1))) { VkSemaphoreSignalInfo ss = {VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = g_utl, .value = v}; vk.vkSignalSemaphore(V.dev, &ss); }
    if (r) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VRED("upload batch %llu: %d (its images keep what they held)", (unsigned long long)v, r); }
    g_ub[k].val = v; g_n_ubatch++;
    retire(1, v);
}
// Backpressure: the batches in flight hold at most DM_VK_UPLOAD_INFLIGHT_MB of staging (the oldest is waited for; the GPU runs them in queue order). Called where the
// replay thread holds no lock (vka_tex_get after the table lock, chunk, the commit): the present and other threads that take the image table do not wait for the GPU
void vka_up_throttle(void) {
    if (!g_utl) return;
    double cap = VKA_ENVF("DM_VK_UPLOAD_INFLIGHT_MB", 256.0) * 1048576.0;
    // Cap hits (the play run, session-2026-11-07 play/PLAY.md, had no line saying whether the 64 MiB cap ever held an area load back): counted, a line at most every 10 s
    static uint64_t hits, hit_ns, last_log; uint64_t t0 = 0;
    for (;;) {
        uint64_t udone = up_done(), oldest = UINT64_MAX; VkDeviceSize fl = 0;
        pthread_mutex_lock(&g_smu);
        for (sblk *b = g_busy; b; b = b->next) if (b->up && b->fence > udone) { fl += b->size; if (b->fence < oldest) oldest = b->fence; }
        pthread_mutex_unlock(&g_smu);
        if ((double)fl <= cap || oldest == UINT64_MAX) break;
        if (!t0) { t0 = now_ns(); hits++; }
        if (!up_wait(oldest)) break;
    }
    if (t0) {
        uint64_t t = now_ns(); hit_ns += t - t0;
        if (t - last_log >= 10000000000ull) { VLOG("upload staging: the in-flight cap (DM_VK_UPLOAD_INFLIGHT_MB %.0f) held the replay back %llu times, %.0f ms in all", cap / 1048576.0, (unsigned long long)hits, hit_ns / 1e6); last_log = t; }
    }
}
void vka_up_stats(uint64_t *batches, uint64_t *waits, double *wait_ms) { if (batches) *batches = g_n_ubatch; if (waits) *waits = g_n_uwait; if (wait_ms) *wait_ms = g_uwait_ns / 1e6; }
#endif

#if defined(__ANDROID__)
// DM_VK_GPU_FLOOR=s (live; default 100): the queue's GPU clock floor through VK_QCOM_queue_perf_hint, s % of the highest clock (100: the highest; 0: the driver's default
// governor). The GPU's governor follows its busy time, and the replay keeps the GPU idle while it records the start of a frame, so it clocks down: the same replayed level
// frame took 21-28 ms of GPU time with the clock up and 93-99 ms after two seconds of idle GPU (Adreno 840), and in the live level the GPU finished 23-33 ms after the
// recording at the default governor, 6-10 ms with the floor at 100 (with DM_VK_CHUNK_PASS). The game's frame-rate controller measures frame start to GPU end, so that tail
// decides between its 20 and 30 fps: in one live session (CPU capped at 2.23 GHz) 85 % of the frames at 30 fps with the floor at 100, 0-16 % at 70 or with the
// driver's governor. The cost is power while a frame's GPU work runs; a lower floor or 0 gives it back
// A hint is the queue's state until changed; the kernel takes the highest of all queues' floors
// DM_VK_GPU_FLOOR_AUTO=1 (live; default 0 = the fixed floor): the floor follows what the frame needs, between 0 and DM_VK_GPU_FLOOR. Each frame's span from the start of the replay's
// recording of its first submission to the GPU end of its last (vka_exec.c, vka_floor_span: recording plus the GPU tail) is held against the period the frame aims at: the game's
// shortest period (vka_floor_period_ns), not the period it presents at now. Over the last 30 frames: the 90th percentile within 2 ms of the period raises the floor a step (within
// 0.5 ms: to DM_VK_GPU_FLOOR at once); more than 5 ms below it (DM_VK_GPU_FLOOR_LO; raise: DM_VK_GPU_FLOOR_HI, 2 ms) for 60 frames since the last change lowers it a step. Steps of 25 %
enum { FSPAN = 64 };
static _Atomic uint64_t g_fspan[FSPAN], g_fspan_f[FSPAN];   // per frame (mod FSPAN): its span in ns, and which frame it is
static uint64_t g_floor_frames[5];                            // frames at each step (statistics)
static double g_p90_sum; static uint64_t g_p90_n;
// The period the floor aims at, for every floor (the queue hint's DM_VK_GPU_FLOOR_AUTO=1 and the ADPF floor's reports, vka_perf.c): the game's shortest period, the shortest
// present period it has asked for in this session and at most one 60 Hz vsync (16.7 ms), not the period it presents at now. A game with a dynamic present interval decides
// its interval from how long its frames take: DKB's controller (main+0x204a18) steps from 3 vsyncs (20 fps) back to 2 only when its measure of the frame is below 20 ms. At
// 20 fps the GPU idles a third of the time, its governor lowers the clock, the frame's GPU time grows to 33-36 ms (Adreno 840) and the game stays at 20 fps for good; a floor
// aiming at the current period (50 ms, or the 33.3 ms cap the auto floor had) sees that frame on target and never raises the clock. Aiming at the shortest period keeps the
// clock where the game can step back up. DM_VK_GPU_FLOOR_PERIOD=<ms> (live; default 0 = the shortest period) sets it instead (33.3: the cap before 2026-11-05)
_Atomic int64_t vka_period_min_ns;   // the shortest present period the game asked for (vka_perf.c vka_perf_frame); 0 = none yet
int64_t vka_floor_period_ns(void) {
    double ms = VKA_ENVF("DM_VK_GPU_FLOOR_PERIOD", 0.0);
    if (ms >= 4 && ms <= 200) return (int64_t)(ms * 1e6);
    int64_t m = atomic_load_explicit(&vka_period_min_ns, memory_order_relaxed), v = 16666667;
    return m >= 4000000 && m < v ? m : v;
}
void vka_floor_span(uint64_t frame, uint64_t ns) {           // the completion thread: a submission of frame ended ns after the frame began
    unsigned k = frame % FSPAN;
    if (atomic_load(&g_fspan_f[k]) != frame) { atomic_store(&g_fspan[k], 0); atomic_store(&g_fspan_f[k], frame); }
    if (ns > atomic_load(&g_fspan[k])) atomic_store(&g_fspan[k], ns);
}
uint64_t vka_floor_span_of(uint64_t frame) { unsigned k = frame % FSPAN; return atomic_load(&g_fspan_f[k]) == frame ? atomic_load(&g_fspan[k]) : 0; }   // 0 = not in (vka_perf.c DM_VK_ADPF_SPAN)
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static int floor_auto(int top, uint64_t f) {
    static int cur = -1; static uint64_t last_f, changed_f;
    if (cur < 0 || cur > top) cur = top;
    if (f == last_f) return cur;
    last_f = f;
    uint64_t v[30]; int n = 0;
    for (uint64_t g = f > 3 ? f - 3 : 0; g > 0 && g + 34 > f && n < 30; g--) { unsigned k = g % FSPAN; if (atomic_load(&g_fspan_f[k]) == g && atomic_load(&g_fspan[k])) v[n++] = atomic_load(&g_fspan[k]); }
    if (n < 20) return cur;
    qsort(v, n, sizeof *v, cmp_u64);
    double p90 = v[n * 9 / 10] / 1e6, per = vka_floor_period_ns() / 1e6;
    int was = cur;
    g_p90_sum += p90; g_p90_n++;
    double hi = VKA_ENVF("DM_VK_GPU_FLOOR_HI", 2.0), lo = VKA_ENVF("DM_VK_GPU_FLOOR_LO", 5.0);   // ms below the period: raise above per - hi, lower below per - lo
    // A step down that has to be taken back within 60 frames is not tried again for a while (300 frames, doubling up to 4,800 each time it fails again): on the Adreno
    // 840 a floor of 75 instead of 100 added 5-15 ms to the level's span at once, and the controller went up and down every few frames
    static uint64_t hold_until, backoff = 300, lowered_f;
    if (p90 > per - 0.5) cur = top;
    else if (p90 > per - hi) cur = cur + 25 > top ? top : cur + 25;
    else if (p90 < per - lo && f - changed_f >= 60 && f >= hold_until) { cur = cur - 25 < 0 ? 0 : cur - 25; lowered_f = f; }
    if (cur > was && lowered_f && f - lowered_f < 60) { hold_until = f + backoff; backoff = backoff * 2 > 4800 ? 4800 : backoff * 2; lowered_f = 0; }
    if (lowered_f && f - lowered_f > 600) { backoff = 300; lowered_f = 0; }   // a step down that held
    if (cur != was) { changed_f = f; if (VKA_ENV("DM_VK_GPU_FLOOR_LOG", 0)) VLOG("GPU clock floor %d -> %d %%: frame %llu, span p90 %.1f ms of %.1f", was, cur, (unsigned long long)f, p90, per); }
    g_floor_frames[cur / 25 < 5 ? cur / 25 : 4]++;
    return cur;
}
void vka_floor_stats(char *b, size_t n) {   // the replay statistics: frames at each floor step since the last call
    uint64_t t = 0; for (int i = 0; i < 5; i++) t += g_floor_frames[i];
    if (!t) { if (n) b[0] = 0; return; }
    snprintf(b, n, "GPU clock floor (auto): frames at 0/25/50/75/100 %%: %.0f/%.0f/%.0f/%.0f/%.0f %%; recording start to GPU end p90 %.1f ms on average", 100.0 * g_floor_frames[0] / t, 100.0 * g_floor_frames[1] / t, 100.0 * g_floor_frames[2] / t,
             100.0 * g_floor_frames[3] / t, 100.0 * g_floor_frames[4] / t, g_p90_n ? g_p90_sum / g_p90_n : 0.0);
    g_p90_sum = 0; g_p90_n = 0;
    memset(g_floor_frames, 0, sizeof g_floor_frames);
}
// The floor through the kernel driver (KGSL) for a Vulkan driver without VK_QCOM_queue_perf_hint on an Adreno (Turnip): the power constraint of the queue's own kernel context.
// Qualcomm's hint maps onto the same kernel call: IOCTL_KGSL_SETPROPERTY KGSL_PROP_PWR_CONSTRAINT {KGSL_CONSTRAINT_PWRLEVEL, context, level} (graphics-kernel adreno.c
// adreno_set_constraint), level 1 = the highest clock or 2-100 = a floor at that % of it, the hint's two kinds. On GPUs whose clock the GMU decides (gen8, the Adreno 840:
// not host_based_dcvs) the kernel hands it to the GMU at once as the context's rule (adreno_gen8_hwsched.c gen8_hwsched_set_pwrconstraint, H2F_MSG_CONTEXT_RULE), which holds
// the clock while that context's work runs; on host-DCVS GPUs (older) it applies at dispatch of submissions flagged KGSL_CONTEXT_PWR_CONSTRAINT and there only "highest" (1)
// is taken. The rule only reaches the GMU while the context is registered with it (after its first submission; again after the GPU slept), and Turnip does not create its
// context with that flag, so the level is sent again every 30 frames. Which contexts: the kernel lets a file descriptor set the constraint only on contexts it owns
// (kgsl_context_get_owner), so the contexts the driver made for the device's queue are the ids this process's /dev/kgsl-3d0 descriptors own after vkCreateDevice and did
// not own before (Turnip: one per VkQueue, tu_knl_kgsl.cc kgsl_submitqueue_new). The ownership test is the same ioctl with KGSL_CONSTRAINT_NONE, which changes nothing on a
// context without a constraint. Before 2026-11-05 Turnip's floor was the ADPF session (vka_perf.c), which did not raise the GPU clock: 282-646 MHz (mean 380) in DKB's
// level where the system driver ran 1,100-1,200 MHz. DM_VK_GPU_FLOOR_KGSL=0 (restart) leaves it off (the ADPF floor then); DM_VK_GPU_FLOOR and DM_VK_GPU_FLOOR_AUTO work as
// with the queue hint
#include <fcntl.h>
#include <sys/ioctl.h>
#include <dirent.h>
#include <unistd.h>
#include <errno.h>
struct kgsl_dc { unsigned int type, context_id; void *data; size_t size; };
struct kgsl_gp { unsigned int type; void *value; size_t sizebytes; };
#define KGSL_IOCTL_SETPROPERTY _IOW(0x09, 0x32, struct kgsl_gp)
enum { KG_FDS = 4, KG_IDS = 1024, KG_CTX = 8 };
static int g_kg_fd[KG_FDS], g_kg_nfd; static uint8_t g_kg_own[KG_FDS][KG_IDS / 8];
static struct { int fd; unsigned id; } g_kg[KG_CTX]; static int g_kg_n, g_kg_host;   // the queue's contexts; host DCVS: only "highest"
static char g_kg_desc[128];   // which contexts and which DCVS, for the floor's first line and the turbo's (the setup's own line was lost before 2026-11-07: see dev_log)
static int kg_set(int fd, unsigned id, int type, unsigned level) {
    unsigned lv = level; struct kgsl_dc c = {(unsigned)type, id, &lv, sizeof lv}; struct kgsl_gp g = {0x12, &c, sizeof c};   // KGSL_PROP_PWR_CONSTRAINT
    int r; do r = ioctl(fd, KGSL_IOCTL_SETPROPERTY, &g); while (r < 0 && (errno == EINTR || errno == EAGAIN));
    return r < 0 ? -errno : 0;
}
static void kg_scan(int keep_fds) {   // this process's descriptors on /dev/kgsl-3d0 and the context ids each owns
    if (!keep_fds) {
        g_kg_nfd = 0; DIR *d = opendir("/proc/self/fd"); if (!d) return;
        struct dirent *de; char path[64], to[64];
        while ((de = readdir(d)) && g_kg_nfd < KG_FDS) {
            int fd = atoi(de->d_name); if (fd <= 2) continue;
            snprintf(path, sizeof path, "/proc/self/fd/%d", fd); ssize_t n = readlink(path, to, sizeof to - 1); if (n <= 0) continue; to[n] = 0;
            if (!strncmp(to, "/dev/kgsl-3d", 12)) g_kg_fd[g_kg_nfd++] = fd;
        }
        closedir(d);
    }
    for (int i = 0; i < g_kg_nfd; i++) { memset(g_kg_own[i], 0, sizeof g_kg_own[i]); for (unsigned id = 1; id < KG_IDS; id++) if (!kg_set(g_kg_fd[i], id, 0, 0)) g_kg_own[i][id / 8] |= (uint8_t)(1u << (id % 8)); }
}
static uint8_t g_kg_was[KG_FDS][KG_IDS / 8]; static int g_kg_wasfd[KG_FDS], g_kg_wasn, g_kg_tried;
void vka_kgsl_before(int has_ext) {   // (not with the queue hint's extension: its driver's contexts are left alone)
    g_kg_tried = !has_ext && VKA_ENV("DM_VK_GPU_FLOOR_KGSL", 1) != 0;
    if (!g_kg_tried) return;
    kg_scan(0); g_kg_wasn = g_kg_nfd; memcpy(g_kg_wasfd, g_kg_fd, sizeof g_kg_fd); memcpy(g_kg_was, g_kg_own, sizeof g_kg_own);
}
void vka_kgsl_after(int has_hint) {
    if (has_hint) return;   // the queue hint does it
    if (!g_kg_tried) { VLOG("GPU clock floor through KGSL: off (DM_VK_GPU_FLOOR_KGSL=0, or the queue hint's extension without its feature)"); return; }
    kg_scan(0);
    g_kg_n = 0;
    for (int i = 0; i < g_kg_nfd; i++) {
        int w = -1; for (int j = 0; j < g_kg_wasn; j++) if (g_kg_wasfd[j] == g_kg_fd[i]) w = j;
        for (unsigned id = 1; id < KG_IDS && g_kg_n < KG_CTX; id++)
            if ((g_kg_own[i][id / 8] >> (id % 8) & 1) && !(w >= 0 && (g_kg_was[w][id / 8] >> (id % 8) & 1))) { g_kg[g_kg_n].fd = g_kg_fd[i]; g_kg[g_kg_n].id = id; g_kg_n++; }
    }
    // The level form the kernel takes: 2-100 % with GMU DCVS, else -EINVAL and only 1 (highest). Another error leaves the DCVS unknown (the floor is then sent as "highest")
    int probe = 0;
    if (g_kg_n) { probe = kg_set(g_kg[0].fd, g_kg[0].id, 1, 50); g_kg_host = probe != 0; kg_set(g_kg[0].fd, g_kg[0].id, 0, 0); }
    char b[96]; size_t o = 0; b[0] = 0; for (int k = 0; k < g_kg_n && o < sizeof b; k++) o += (size_t)snprintf(b + o, sizeof b - o, "%s%u", k ? "," : "", g_kg[k].id);
    char dcvs[64]; snprintf(dcvs, sizeof dcvs, !probe ? "GMU DCVS" : probe == -EINVAL ? "host DCVS" : "DCVS unknown: the level probe gave %d", probe);
    if (g_kg_n) snprintf(g_kg_desc, sizeof g_kg_desc, "context%s %s on descriptor %d, %s", g_kg_n > 1 ? "s" : "", b, g_kg[0].fd, dcvs);
    if (g_kg_n) VLOG("GPU clock floor through KGSL: the queue's kernel context%s %s (%d descriptor%s on /dev/kgsl-3d0; %s: %s), DM_VK_GPU_FLOOR (DM_VK_GPU_FLOOR_KGSL)",
                     g_kg_n > 1 ? "s" : "", b, g_kg_nfd, g_kg_nfd == 1 ? "" : "s", dcvs, g_kg_host ? "highest clock only" : "floor in % of the highest clock");
    else VLOG("GPU clock floor through KGSL: %s", g_kg_nfd ? "no kernel context of the device's own (the ADPF floor)" : "no /dev/kgsl-3d0 descriptor (not an Adreno with KGSL; the ADPF floor)");
}
int vka_floor_kgsl(void) { return !V.perf_hint && g_kg_n > 0; }
// The GPU clock now, for the GPU profile's lines (vka_gprof.c): KGSL's sysfs files, the first that reads of gpuclk (Hz), devfreq/cur_freq (Hz) and clock_mhz, with
// thermal_pwrlevel and max_gpuclk when they read. Perfetto's power/gpu_frequency is not reachable from the app, and neither Vulkan nor Turnip reports a clock. SELinux may
// refuse the files to an app (untrusted_app has no sysfs_kgsl read on some builds; the POCO F8 Pro: vendor_sysfs_kgsl): the answer then names each file and its error,
// nothing is guessed. When none reads, the probe is not repeated (each refused open is an avc line in logcat: 1,527 in fb10's profiled frames); its answer is kept.
// → 1 a clock was read, 0 not
static int sys_u64(const char *path, uint64_t *v, int *err) {
    int fd = open(path, O_RDONLY | O_CLOEXEC); if (fd < 0) { *err = errno; return 0; }
    char b[32]; ssize_t n = read(fd, b, sizeof b - 1); int e = errno; close(fd);
    if (n <= 0) { *err = n < 0 ? e : EIO; return 0; }
    b[n] = 0; char *c; *v = strtoull(b, &c, 10); if (c == b) { *err = EINVAL; return 0; }
    return 1;
}
int vka_gpu_clock(char *out, size_t n) {
    static const struct { const char *path, *name; double to_mhz; } src[3] = {{"/sys/class/kgsl/kgsl-3d0/gpuclk", "gpuclk", 1e-6}, {"/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq", "devfreq/cur_freq", 1e-6},
                                                                             {"/sys/class/kgsl/kgsl-3d0/clock_mhz", "clock_mhz", 1}};
    static char none[200]; static _Atomic int unreadable;   // (none written before unreadable is set)
    if (atomic_load(&unreadable)) { snprintf(out, n, "%s", none); return 0; }
    char why[160]; size_t w = 0; why[0] = 0; uint64_t v; int err;
    for (int i = 0; i < 3; i++) {
        if (sys_u64(src[i].path, &v, &err)) {
            size_t o = (size_t)snprintf(out, n, "GPU clock %.0f MHz (KGSL %s", v * src[i].to_mhz, src[i].name);
            uint64_t t, m; int e2;
            if (o < n && sys_u64("/sys/class/kgsl/kgsl-3d0/thermal_pwrlevel", &t, &e2)) o += (size_t)snprintf(out + o, n - o, ", thermal_pwrlevel %llu", (unsigned long long)t);
            if (o < n && sys_u64("/sys/class/kgsl/kgsl-3d0/max_gpuclk", &m, &e2)) o += (size_t)snprintf(out + o, n - o, ", max_gpuclk %.0f MHz", m * 1e-6);
            if (o < n) snprintf(out + o, n - o, ")");
            return 1;
        }
        if (w < sizeof why) w += (size_t)snprintf(why + w, sizeof why - w, "%s%s: %s", w ? ", " : "", src[i].name, strerror(err));
    }
    snprintf(none, sizeof none, "GPU clock not readable by the app (KGSL sysfs, probed once: %s)", why);
    atomic_store(&unreadable, 1);
    snprintf(out, n, "%s", none); return 0;
}
static VkResult kg_apply(int s) {   // the floor s % (0: none) on the queue's contexts
    int bad = 0;
    for (int k = 0; k < g_kg_n; k++) {
        int r = s <= 0 ? kg_set(g_kg[k].fd, g_kg[k].id, 0, 0) : kg_set(g_kg[k].fd, g_kg[k].id, 1, s >= 100 || g_kg_host ? 1u : (unsigned)(s < 2 ? 2 : s));
        if (r) { bad = r; static int n; if (n++ < 3) VRED("KGSL power constraint %d %% on context %u: %d", s, g_kg[k].id, r); }
    }
    return bad ? VK_ERROR_UNKNOWN : VK_SUCCESS;
}
static _Atomic int g_tail_n;   // DM_VK_GPU_FLOOR_AUTO=2: submissions with draws counted at their commit and not yet completed
// The queue's hint (any thread: the replay's, the completion thread's). tail: the floor is s while a counted submission is pending, else 0, decided under the queue lock, so
// a commit and a completion racing cannot apply their floors in the wrong order (before 2026-11-03 a completion's 0 could land after the next commit's floor)
static void floor_set_(int s, int tail, int again) {   // again: send it even if unchanged (the KGSL rule, see kg_apply)
    static _Atomic int cur = -1;
    if (!tail && !again && atomic_load(&cur) == s) return;
    pthread_mutex_lock(&V.qmu);
    if (tail) s = atomic_load(&g_tail_n) > 0 ? s : 0;
    if (!again && atomic_load(&cur) == s) { pthread_mutex_unlock(&V.qmu); return; }
    VkResult r;
    if (V.perf_hint) {
        struct { VkStructureType sType; void *pNext; uint32_t type, scale; } h = {(VkStructureType)1000302000, NULL, s >= 100 ? 2u : s > 0 ? 3u : 0u, s > 0 && s < 100 ? (uint32_t)s : 0u};
        r = ((VkResult (*)(VkQueue, const void *))V.perf_hint)(V.q, &h);
    } else r = kg_apply(s);
    int was = atomic_exchange(&cur, s);
    pthread_mutex_unlock(&V.qmu);
    if (was < 0 || (was != s && VKA_ENV("DM_VK_GPU_FLOOR_LOG", 0) > 1))
        VLOG("GPU clock floor%s%s%s: %d %% (%s): %d", V.perf_hint ? " (VK_QCOM_queue_perf_hint" : " (KGSL, ", V.perf_hint ? "" : g_kg_desc, ")", s, s >= 100 ? "highest" : s ? "scaled" : "default governor", r);
}
// DM_VK_GPU_FLOOR_AUTO=2: the floor only for the tail. The driver's governor keeps the clock while the frame is recorded (its parts reach the GPU as fast as they are
// recorded, so a faster GPU only waits longer); when a submission with draws is handed to the queue the floor goes to DM_VK_GPU_FLOOR, and back to the governor once
// every such submission handed over has completed. A submission is counted when its span op is posted (vka_floor_tail_add) and the count is given back by that op on
// completion, so the two always pair: before 2026-11-03 every completed submission with draws took one off, also those committed before a live switch to mode 2 or whose
// end failed, and the floor could drop with a counted submission still on the GPU
static int floor_mode(void) { return V.perf_hint || g_kg_n ? VKA_ENV("DM_VK_GPU_FLOOR_AUTO", 0) : -1; }
static int floor_top(void) { int s = VKA_ENV("DM_VK_GPU_FLOOR", 100); return s < 0 ? 0 : s > 100 ? 100 : s; }
// The floor without VK_QCOM_queue_perf_hint and without the KGSL constraint above (a driver that is not on KGSL, or DM_VK_GPU_FLOOR_KGSL=0): the ADPF session (vka_perf.c) reports each frame's span (recording start to the GPU end of
// its last submission) with its GPU time against the shortest period, so the system raises the clocks the frame's work runs on when the frame takes longer than the period
// the game could run at. Before 2026-11-05 there was no floor at all there: on Turnip DKB stayed at 20 fps with 33-36 ms of GPU time a frame, as the system driver did with
// DM_VK_GPU_FLOOR=0. It has no level: DM_VK_GPU_FLOOR > 0 turns it on, 0 off, as does DM_VK_GPU_FLOOR_ADPF=0 (live); DM_VK_ADPF_SPAN set explicitly decides the reports. On the Adreno 840 with Turnip it did not raise the GPU clock (282-646 MHz, mean 380; 2026-11-05)
int vka_floor_adpf(void) { return !V.perf_hint && !g_kg_n && floor_top() > 0 && VKA_ENV("DM_VK_GPU_FLOOR_ADPF", 1); }
int vka_floor_tail_add(void) { if (floor_mode() != 2) return 0; atomic_fetch_add(&g_tail_n, 1); return 1; }   // vka_exec.c, a submission with draws: 1 = counted
void vka_floor_commit(int counted) { if (counted) floor_set_(floor_top(), 1, 0); }   // before that submission is handed to the queue
void vka_floor_done(int counted) {     // the completion thread, that submission done
    if (!counted) return;
    atomic_fetch_sub(&g_tail_n, 1);
    if (floor_mode() == 2) floor_set_(floor_top(), 1, 0);
}
// DM_GPU_TURBO=1 (live; default 0): the device-wide KGSL turbo, KGSL_PROP_PWRCTRL (0xE) = 0 through IOCTL_KGSL_SETPROPERTY on a /dev/kgsl-3d0 descriptor of its own (the
// call libadrenotools' adrenotools_set_turbo makes). The kernel then turns its clock scaling off and holds the GPU at the highest level the thermal limits allow (graphics-
// kernel adreno_setproperty: kgsl_pwrscale_disable(device, true); kernels of that form also stop their long-IB fault detection with it), for every process, not only the
// queue's contexts as the floor above. Whether a GMU-clocked GPU (gen8) follows it is not known yet: the log line gives the kernel's answer and the floor's DCVS mode. Why: in Ocarina of Time (pf1, POCO F8 Pro, Adreno 830, Turnip) the clock never went above 900 MHz with the floor at 100 % and fell to 734 MHz at a battery of
// 44-46 °C. The turbo cannot lift a thermal cap, and it heats the phone sooner, which brings the thermal cap sooner. It is set at the next submission and taken back when
// the queue is drained (vka_queue_idle: a suspend, a swapchain change), at exit() and when the switch goes to 0; each change is logged with the kernel's answer. The setting
// outlives the process: a run killed with it on (the low-memory killer, a crash, _exit) leaves the GPU without clock scaling, so a marker file in TMPDIR (dm_gpu_turbo) says
// it is on, and the next run's first submission turns the scaling back on when it finds one (turbo_stale_locked). Not at the device setup: in the launcher that comes
// before Native.start sets the run's environment (TMPDIR among it), so it found no marker. A game that ends through nxlaunch.c's _exit skips exit(): it is taken back at
// the suspend before it (the screen closed), else by the next run
static pthread_mutex_t g_turbo_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_turbo_fd = -1; static _Atomic int g_turbo_on, g_turbo_failed, g_turbo_checked;   // failed: the last try to set it failed; not tried again until the switch is 0 once
static int kgsl_pwrctrl(int fd, unsigned enable) {   // enable 0: the turbo; 1: the kernel's clock scaling
    struct kgsl_gp g = {0xE, &enable, sizeof enable};   // KGSL_PROP_PWRCTRL
    int r; do r = ioctl(fd, KGSL_IOCTL_SETPROPERTY, &g); while (r < 0 && (errno == EINTR || errno == EAGAIN));
    return r < 0 ? -errno : 0;
}
static void turbo_marker(int on) {
    const char *t = getenv("TMPDIR");
    if (!t || !*t) { if (on) VRED("GPU turbo: no TMPDIR, so no marker: a run killed with the turbo on is not seen by the next"); return; }
    char path[512]; snprintf(path, sizeof path, "%s/dm_gpu_turbo", t);
    if (on) { int fd = open(path, O_WRONLY | O_CREAT | O_CLOEXEC, 0600); if (fd >= 0) close(fd); else VRED("GPU turbo: marker %s not made (%s): a run killed with the turbo on is not seen by the next", path, strerror(errno)); }
    else unlink(path);
}
static void turbo_stale_locked(void) {   // (under g_turbo_mu) once, before this run's first turbo: a marker left by a run that ended with the turbo on
    if (atomic_load(&g_turbo_checked)) return;
    atomic_store(&g_turbo_checked, 1);
    const char *t = getenv("TMPDIR"); if (!t || !*t) return;
    char path[512]; snprintf(path, sizeof path, "%s/dm_gpu_turbo", t);
    if (access(path, F_OK)) return;
    int fd = open("/dev/kgsl-3d0", O_RDWR | O_CLOEXEC), r = fd < 0 ? -errno : kgsl_pwrctrl(fd, 1);
    if (fd >= 0) close(fd);
    if (!r) { unlink(path); VLOG("GPU turbo: an earlier run ended with it on (killed, or a crash); the kernel's clock scaling is back on (KGSL_PROP_PWRCTRL 1)"); }
    else VRED("GPU turbo: an earlier run ended with it on; turning the clock scaling back on failed: %s (errno %d)", strerror(-r), -r);
}
static void turbo_exit(void) { vka_turbo_set(0, "exit"); }
void vka_turbo_set(int on, const char *why) {
    if (atomic_load_explicit(&g_turbo_on, memory_order_relaxed) == on || (on && atomic_load_explicit(&g_turbo_failed, memory_order_relaxed))) return;
    pthread_mutex_lock(&g_turbo_mu);
    turbo_stale_locked();
    if (atomic_load(&g_turbo_on) == on || (on && atomic_load(&g_turbo_failed))) { pthread_mutex_unlock(&g_turbo_mu); return; }
    int r = 0;
    if (g_turbo_fd < 0 && (g_turbo_fd = open("/dev/kgsl-3d0", O_RDWR | O_CLOEXEC)) < 0) r = -errno;
    if (!r) r = kgsl_pwrctrl(g_turbo_fd, on ? 0u : 1u);
    if (!r) { atomic_store(&g_turbo_on, on); turbo_marker(on); static int reg; if (on && !reg) reg = !atexit(turbo_exit); }
    else if (on) atomic_store(&g_turbo_failed, 1);
    pthread_mutex_unlock(&g_turbo_mu);
    if (!r) VLOG("GPU turbo %s (KGSL_PROP_PWRCTRL %u on /dev/kgsl-3d0; %s): %s", on ? "on" : "off", on ? 0u : 1u, on ? "the clock scaling off: the highest clock the thermal limits allow" : "the clock scaling back", why);
    else VRED("GPU turbo %s (KGSL_PROP_PWRCTRL %u on /dev/kgsl-3d0; %s): %s (errno %d)%s", on ? "on" : "off", on ? 0u : 1u, why, strerror(-r), -r,
              on ? ": the kernel's clock scaling stays (not tried again until DM_GPU_TURBO is 0 once)" : ": the GPU may stay without clock scaling until a later run or a reboot");
}
void vka_gpu_floor(void) {
    if (!atomic_load_explicit(&g_turbo_checked, memory_order_relaxed)) { pthread_mutex_lock(&g_turbo_mu); turbo_stale_locked(); pthread_mutex_unlock(&g_turbo_mu); }   // also with the switch off
    { int want = VKA_ENV("DM_GPU_TURBO", 0) > 0, on = want && !vka_hung();   // the turbo (above); off once the GPU hung
      if (!want) atomic_store_explicit(&g_turbo_failed, 0, memory_order_relaxed);
      if (on != atomic_load_explicit(&g_turbo_on, memory_order_relaxed) && !(on && atomic_load_explicit(&g_turbo_failed, memory_order_relaxed))) {
          char why[192]; snprintf(why, sizeof why, "%s%s%s", !want ? "DM_GPU_TURBO=0" : !on ? "the GPU hung" : "DM_GPU_TURBO=1", on && g_kg_desc[0] ? "; the floor's " : "", on ? g_kg_desc : "");
          vka_turbo_set(on, why); } }
    int mode = floor_mode(); if (mode < 0 || mode == 2) return;
    int s = VKA_ENV("DM_VK_GPU_FLOOR", 100); if (s < 0) s = 0; if (s > 100) s = 100;
    uint64_t vka_frame(void);
    uint64_t f = vka_frame();
    if (mode == 1) s = floor_auto(s, f);
    static uint64_t sent_f;   // the KGSL rule again every 30 frames (it is lost when the GMU registers the context again)
    int again = !V.perf_hint && (f >= sent_f + 30 || f < sent_f);
    if (again) sent_f = f;
    floor_set_(s, 0, again);
}
#endif
