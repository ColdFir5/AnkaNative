// Shaders of the Vulkan replay: the game's library (android/SHADERS.md), its entries as VkShaderModules, and NVN programs resolved to one shader per stage.
// The library is <pack folder>/shaders/v-<DMSH_VK_CHAIN>/dmsh_vk.lib beside its inputs.tsv (the pack folder is the one the modules came from, dm_module_dir), or the file
// DM_VK_SHLIB names. One whose chain, Vulkan chain or game is not this host's is refused with a [RED] line: the game runs on with no draws. While this version's library is
// not in the pack yet (the app's delta update makes it, android/SHADERS.md#delta-updates), another v-<V>/ of the pack whose header carries this host's DMSH_VK_ABI is played
// instead (the newest): what the replay reads of it is the same; the new library is taken from the next start. Modules are made at a program's first draw (the submit
// thread), not at nvnProgramSetShaders
#if defined(__ANDROID__)
#include "vka.h"
#include "../nvn/nvn_mtl.h"
#include "../nvn/dmsh_chain.h"
#include "../nvn/dmsh_chain_vk.h"
#include "game_profile.h"
#include "vka_spvpatch.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <time.h>

static vka_shlib g_lib; static int g_lib_state;   // 0 not tried, 1 loaded, -1 none
static char g_lib_path[1024];                       // the library loaded (vka_shader_lib_path)
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;

// How each stage runs the subgroup operations of a library built for sg lanes (DMSH_VK_F_SUBGROUP shaders). A module built for 64 lanes splits ballots and shuffles into
// 32-lane halves, so it is right on 32 or 64 lanes; one built for n lanes, on n. Per stage, in this order: no subgroup operations in the stage on this device = rewritten
// as one-invocation subgroups (single_lane); a size the library is right for that the device may vary and lets a pipeline pin (subgroupSizeControl, the stage in
// requiredSubgroupSizeStages) = pinned to it (VkPipelineShaderStageRequiredSubgroupSizeCreateInfo, vka_pso.c); the device's fixed size right for the library = as built;
// otherwise (a wrong or varying size that cannot be pinned) = one-invocation subgroups too. Mesa Turnip on the Adreno 840 reports 128 lanes (64-128) with fragment and
// compute pinnable: before this, every stage ran the 64-lane modules at up to 128 lanes (black cave walls in DKB); now fragment and compute are pinned to 64 and the
// vertex, tessellation and geometry stages run one-lane subgroups, as on Qualcomm's driver (which has none in those stages). DM_VK_SUBGROUP_PIN (restart): 0 = no pinning,
// 1 = every stage the device lets pin (default), any other value = a VkShaderStageFlags mask of the stages that may pin (0x30: fragment and compute)
static uint32_t g_sg_pin[8]; static uint8_t g_sg_one[8];   // by DMSH_VK_STAGE_*
static const uint32_t g_stage_bit[8] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, VK_SHADER_STAGE_GEOMETRY_BIT,
                                        VK_SHADER_STAGE_FRAGMENT_BIT, VK_SHADER_STAGE_COMPUTE_BIT, VK_SHADER_STAGE_TASK_BIT_EXT, VK_SHADER_STAGE_MESH_BIT_EXT};
static void sg_plan(unsigned sg) {
    static const char *const nm[8] = {"vertex", "tess control", "tess evaluation", "geometry", "fragment", "compute", "task", "mesh"};
#define SG_OK(n) (sg == 64 ? ((n) == 32 || (n) == 64) : (n) == sg)
    unsigned pin = 0; int pm = VKA_ENV("DM_VK_SUBGROUP_PIN", 1); uint32_t pin_stages = pm == 1 ? V.subgroup_req_stages : V.subgroup_req_stages & (uint32_t)pm;
    if (V.subgroup_min != V.subgroup_max && pin_stages)
        for (unsigned c = sg; c >= 32 && !pin; c /= 2) if (SG_OK(c) && c >= V.subgroup_min && c <= V.subgroup_max) pin = c;
    // Turnip's geometry stage computes subgroup operations wrong at a pinned 64 (T30, Mesa 26.3: DKB's terrain culling GS 5d537f7630ec3671 left the cave walls black; one-lane
    // subgroups there drew them right, with every other stage pinned). Eden leaves geometry out of Turnip's subgroup stages too. DM_VK_SUBGROUP_GS (restart): 0 = the geometry
    // stage runs one-lane subgroups, 1 = as the device reports; default 0 on VK_DRIVER_ID_MESA_TURNIP, 1 on other drivers
    uint32_t stages = V.subgroup_stages;
    if (!VKA_ENV("DM_VK_SUBGROUP_GS", V.driver_id != VK_DRIVER_ID_MESA_TURNIP)) stages &= ~(uint32_t)VK_SHADER_STAGE_GEOMETRY_BIT;
    char q[256]; size_t qn = 0; q[0] = 0; int one_new = 0;
    for (int i = 0; i < 8; i++) {
        if (i >= DMSH_VK_STAGE_TS && !V.mesh) continue;
        uint32_t b = g_stage_bit[i]; g_sg_pin[i] = 0; g_sg_one[i] = 0;
        if (!(stages & b)) g_sg_one[i] = 1;
        else if (pin && (pin_stages & b)) g_sg_pin[i] = pin;
        else if (!SG_OK(V.subgroup_size) || V.subgroup_min != V.subgroup_max) g_sg_one[i] = 1, one_new = 1;
        qn += (size_t)snprintf(q + qn, sizeof q - qn, "%s%s %s", i ? ", " : "", nm[i], g_sg_one[i] ? "one lane" : g_sg_pin[i] ? (pin == 64 ? "pinned 64" : "pinned 32") : "as built");
        if (qn >= sizeof q) qn = sizeof q - 1;
    }
#undef SG_OK
    VLOG("subgroup shaders (built for %u lanes; the device's %u, %u-%u, pinnable in %#x, pinned in %#x): %s", sg, V.subgroup_size, V.subgroup_min, V.subgroup_max, V.subgroup_req_stages, pin ? pin_stages : 0, q);
    if (one_new) VLOG("subgroup shaders: stages whose subgroup size is wrong for the library or varies unpinned run them as one-invocation subgroups");
}
uint32_t vka_shader_sg_pin(int stage) { return stage >= 0 && stage < 8 ? g_sg_pin[stage] : 0; }

// Compute shaders' subgroup shuffles through workgroup memory (the library's DMSH_VK_MOD_SHUFFLE_LDS modules, vk_android_spv.py shuffle_lds). Turnip has no
// arbitrary-index shuffle instruction: ir3 makes each OpGroupNonUniformShuffle a loop of rotates, one pass per distinct (index - lane) in the subgroup, and the translator's
// LDSM / MOVM gathers take up to 24 (agl's upscaler network in DKB, 14a8aaf3e0a9edbd: 10.7 ms a dispatch on the Adreno 840, 2.0 ms with the module through workgroup
// memory). Qualcomm's driver shuffles natively and runs the base module faster (1.7 ms against 1.9). DM_VK_SHUFFLE_LDS (restart): 1 = those modules, 0 = the base ones;
// default 1 on VK_DRIVER_ID_MESA_TURNIP, 0 on other drivers. Needs DMSH_VK_SHARED_MAX bytes of shared memory (the library keeps the modules within it)
static int g_shuffle_lds = -1;
static void shuffle_plan(void) {
    const char *ev = getenv("DM_VK_SHUFFLE_LDS");
    int want = ev && *ev ? atoi(ev) != 0 : V.driver_id == VK_DRIVER_ID_MESA_TURNIP;
    int fits = V.props.limits.maxComputeSharedMemorySize >= DMSH_VK_SHARED_MAX;
    g_shuffle_lds = want && fits;
    VLOG("compute subgroup shuffles: %s", g_shuffle_lds ? "through workgroup memory where the library has the module (DM_VK_SHUFFLE_LDS=0: the instruction)"
         : want ? "the instruction (the device's shared memory is under DMSH_VK_SHARED_MAX)" : "the instruction (DM_VK_SHUFFLE_LDS=1: through workgroup memory)");
}
VkShaderModule vka_shader_cs_module(vka_shader *s) {
    if (g_shuffle_lds < 0) shuffle_plan();
    VkShaderModule m = g_shuffle_lds && vka_shlib_module(s->lib, s->e, DMSH_VK_MOD_SHUFFLE_LDS) ? vka_shader_module(s, DMSH_VK_MOD_SHUFFLE_LDS) : VK_NULL_HANDLE;
    return m ? m : s->mods[DMSH_VK_MOD_BASE];   // (that module not made: logged by vka_shader_module; the base one)
}

const char *vka_shader_lib_path(void) { return g_lib_path; }
// GPU texture-use feedback (vka_tex.c "Texture-use feedback"): whether the library's entries report the texture IDs they index (DMSH_VK_F_TEXFB), counted once at its load.
// complete: every entry with a texture pool binding has the flag (a module the build's pass refused has not), so a draw that reads no texture of the pool (a dispatch over
// material tables) may be gated too: whatever samples its IDs reports them
static _Atomic int g_tfb_n, g_tfb_all;
int vka_shader_texfb(int *complete) { if (complete) *complete = atomic_load(&g_tfb_all); return atomic_load(&g_tfb_n) > 0; }
static void tfb_count(const vka_shlib *L) {
    uint32_t pool = 0, fb = 0, fbpool = 0;
    for (uint32_t i = 0; i < L->h->entry_count; i++) {
        const dmsh_vk_entry *e = &L->ent[i]; const dmsh_vk_bind *b = vka_shlib_binds(L, e); int tp = 0;
        for (unsigned j = 0; j < e->bind_count && !tp; j++) tp = b[j].kind == DMSH_VK_KIND_TEX_POOL;
        pool += tp; fb += (e->flags & DMSH_VK_F_TEXFB) != 0; fbpool += tp && (e->flags & DMSH_VK_F_TEXFB);
    }
    atomic_store(&g_tfb_all, fb && fbpool == pool); atomic_store(&g_tfb_n, (int)fb);
    if (fb) { VkPhysicalDeviceFeatures pf; vk.vkGetPhysicalDeviceFeatures(V.pd, &pf);   // (the module declares the bitmap's stores and atomics whatever the constant: every supported feature is enabled, vka_dev.c)
        if (!pf.fragmentStoresAndAtomics || !pf.vertexPipelineStoresAndAtomics)
            VRED("texture-use feedback: the device lacks %s, which the library's DMSH_VK_F_TEXFB modules need (their bitmap store is in the module even with the constant off): "
                 "their pipelines may be refused", !pf.fragmentStoresAndAtomics ? "fragmentStoresAndAtomics" : "vertexPipelineStoresAndAtomics"); }
    if (fb || VKA_ENV("DM_VK_TEXFB", 0)) VLOG("texture-use feedback: %u of the library's %u shaders report the texture IDs they index (DMSH_VK_F_TEXFB), %u of the %u with a texture pool binding%s",
                                             fb, L->h->entry_count, fbpool, pool, fb && fbpool < pool ? ": draws of the others keep the scan's images, dispatches that read no pool texture are not gated" : "");
}

static int lib_load(void) {
    if (g_lib_state) return g_lib_state > 0;
    g_lib_state = -1;
    char path[1024], err[256]; const char *e = getenv("DM_VK_SHLIB"); const char *dir = dm_module_dir(); dir = dir && *dir ? dir : ".";
    if (e && *e) snprintf(path, sizeof path, "%s", e);
    else snprintf(path, sizeof path, "%s/shaders/" DMSH_VK_DIR_PREFIX "%s/" DMSH_VK_LIB_FILE, dir, DMSH_VK_CHAIN);
    int have = !access(path, R_OK) && vka_shlib_open(&g_lib, path, DMSH_CHAIN, DMSH_VK_CHAIN, DM_GAME_ID, DMSH_VK_ABI, err, sizeof err);
    if (!have && !(e && *e)) {   // this version's library not in the pack (yet): the newest other v-<V>/ that this host reads the same (its ABI)
        char why[256]; snprintf(why, sizeof why, "%s", access(path, R_OK) ? "missing" : err);
        char sd[1024], best[1024] = "", old[1024] = ""; time_t bt = 0, ot = 0; snprintf(sd, sizeof sd, "%s/shaders", dir);
        DIR *d = opendir(sd); struct dirent *de;
        while (d && (de = readdir(d))) {
            if (strncmp(de->d_name, DMSH_VK_DIR_PREFIX, strlen(DMSH_VK_DIR_PREFIX))) continue;
            char p[1024]; struct stat st; vka_shlib t; char e2[256];
            if (snprintf(p, sizeof p, "%s/%s/" DMSH_VK_LIB_FILE, sd, de->d_name) >= (int)sizeof p || stat(p, &st) || !strcmp(p, path)) continue;
            if (vka_shlib_open(&t, p, DMSH_CHAIN, DMSH_VK_CHAIN, DM_GAME_ID, DMSH_VK_ABI, e2, sizeof e2)) {
                vka_shlib_close(&t);
                if (!*best || st.st_mtime > bt) { snprintf(best, sizeof best, "%s", p); bt = st.st_mtime; }
                continue;
            }
            // A library made before the abi field (all zero): its ABI is not known, so it is the last resort, played with a warning rather than nothing drawn
            // (the user's choice: never block Play on it; the app suggests one more shader update, after which updates are deltas of the same ABI)
            int legacy = vka_shlib_open(&t, p, NULL, NULL, DM_GAME_ID, NULL, e2, sizeof e2);
            if (legacy) { static const char z[12]; legacy = !memcmp(t.h->abi, z, sizeof z); vka_shlib_close(&t); if (!legacy) snprintf(e2, sizeof e2, "another ABI than this host's %s", DMSH_VK_ABI); }
            if (!legacy) { VLOG("shader library %s not usable here: %s", p, e2); continue; }
            if (!*old || st.st_mtime > ot) { snprintf(old, sizeof old, "%s", p); ot = st.st_mtime; }
        }
        if (d) closedir(d);
        if (*best && vka_shlib_open(&g_lib, best, DMSH_CHAIN, DMSH_VK_CHAIN, DM_GAME_ID, DMSH_VK_ABI, err, sizeof err)) {
            char v[17]; vka_shlib_str(g_lib.h->vk_chain, v);
            VLOG("shader library %s: this host's version %s is %s (the app's shader update makes it); playing version %s, the same ABI %s", path, DMSH_VK_CHAIN, why, v, DMSH_VK_ABI);
            snprintf(path, sizeof path, "%s", best); have = 1;
        } else if (!*best && *old && vka_shlib_open(&g_lib, old, NULL, NULL, DM_GAME_ID, NULL, err, sizeof err)) {
            char v[17]; vka_shlib_str(g_lib.h->vk_chain, v);
            VRED("shader library %s: this host's version %s is %s; playing version %s, made before versioned shader updates (no ABI in its header, this host's is %s): "
                 "it may draw wrong. Update the shaders once in the app; later updates are then partial", path, DMSH_VK_CHAIN, why, v, DMSH_VK_ABI);
            snprintf(path, sizeof path, "%s", old); have = 1;
        } else { VRED("no shader library at %s (the pack's shaders/" DMSH_VK_DIR_PREFIX "%s: %s) and none of another version with this host's ABI %s: nothing is drawn", path, DMSH_VK_CHAIN, why, DMSH_VK_ABI); return 0; }
    }
    if (!have) { VRED("shader library %s refused: %s; nothing is drawn", path, access(path, R_OK) ? "missing" : err); return 0; }
    snprintf(g_lib_path, sizeof g_lib_path, "%s", path);
    // The translator's answers in the header against this device: modules built for 64-lane subgroups split ballots and shuffles into 32-lane halves, so they are right on
    // 32 or 64 lanes; storage buffers are bound rounded down to ssbo_align, which the device's alignment must divide (android/SHADERS.md "Translator answers")
    unsigned sg = g_lib.h->subgroup_size ? g_lib.h->subgroup_size : 32, sa = g_lib.h->ssbo_align;
    sg_plan(sg); shuffle_plan();
    if (sa && (sa % V.sb_align)) { VRED("shader library %s refused: built for storage buffers at %u-byte boundaries, the device needs %llu; nothing is drawn", path, sa, (unsigned long long)V.sb_align); vka_shlib_close(&g_lib); return 0; }
    char inputs[17]; vka_shlib_str(g_lib.h->inputs, inputs);
    VLOG("shader library %s: %u shaders, %u modules, %.1f MB (chain %s, Vulkan chain %s, inputs %s; %u-lane subgroups, storage buffers at %u)", path, g_lib.h->entry_count, g_lib.h->mod_count,
         g_lib.size / 1e6, DMSH_CHAIN, DMSH_VK_CHAIN, inputs, sg, sa ? sa : 1);
    g_lib_state = 1;
    tfb_count(&g_lib);
    const char *ck = getenv("DM_VK_SHLIB_CHECK");
    if (ck && atoi(ck)) vka_shlib_check_all(&g_lib);
    return 1;
}

// ---------- Static memory: buffers that live as long as the process ----------
typedef struct { VkBuffer buf; VkDeviceMemory mem; uint8_t *ptr; VkDeviceSize size, used; } stblk;
static stblk g_st;
int vka_static(VkDeviceSize n, VkDeviceSize align, vka_span *out) {
    if (!align) align = 16;
    pthread_mutex_lock(&g_mu);
    VkDeviceSize o = (g_st.used + align - 1) / align * align;
    if (!g_st.buf || o + n > g_st.size) {
        stblk b = {0}; b.size = n > (4u << 20) ? (n + 0xFFFF) & ~(VkDeviceSize)0xFFFF : 4u << 20;
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = b.size, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT};
        VkMemoryRequirements mr; void *p = NULL; uint32_t t = UINT32_MAX;
        if (!vk.vkCreateBuffer(V.dev, &bi, NULL, &b.buf)) {
            vk.vkGetBufferMemoryRequirements(V.dev, b.buf, &mr);
            t = vka_mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 0);
            VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = t};
            if (t == UINT32_MAX || vk.vkAllocateMemory(V.dev, &ai, NULL, &b.mem) || vk.vkBindBufferMemory(V.dev, b.buf, b.mem, 0) || vk.vkMapMemory(V.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &p)) p = NULL;
        }
        if (!p) { pthread_mutex_unlock(&g_mu); VRED("static block of %llu bytes not allocated", (unsigned long long)b.size); return 0; }
        b.ptr = p; g_st = b; o = 0;   // the old block stays (its buffers are still bound by shaders)
    }
    g_st.used = o + n; *out = (vka_span){g_st.buf, g_st.ptr + o, o};
    pthread_mutex_unlock(&g_mu);
    return 1;
}

// ---------- Shaders by key ----------
typedef struct { uint64_t key; vka_shader *sh; const vka_shlib *lib; } shent;
static shent *g_sh; static size_t g_shcap, g_shn;
static size_t hslot(uint64_t k, size_t cap) { return (size_t)((k * 0x9E3779B97F4A7C15ull) >> 17) & (cap - 1); }
static shent *sh_find(uint64_t key, const vka_shlib *L) {
    if (!g_shcap) return NULL;
    for (size_t i = hslot(key, g_shcap);; i = (i + 1) & (g_shcap - 1)) { if (!g_sh[i].sh) return NULL; if (g_sh[i].key == key && g_sh[i].lib == L) return &g_sh[i]; }
}
static void sh_put(uint64_t key, const vka_shlib *L, vka_shader *s) {
    if (2 * (g_shn + 1) > g_shcap) {
        size_t nc = g_shcap ? 2 * g_shcap : 1024; shent *n = calloc(nc, sizeof *n);
        for (size_t i = 0; i < g_shcap; i++) if (g_sh[i].sh) { size_t j = hslot(g_sh[i].key, nc); while (n[j].sh) j = (j + 1) & (nc - 1); n[j] = g_sh[i]; }
        free(g_sh); g_sh = n; g_shcap = nc;
    }
    size_t j = hslot(key, g_shcap); while (g_sh[j].sh) j = (j + 1) & (g_shcap - 1);
    g_sh[j] = (shent){key, s, L}; g_shn++;
}

static const char *stage_name(int s) { static const char *n[] = {"vertex", "tess control", "tess evaluation", "geometry", "fragment", "compute", "task", "mesh"}; return s >= 0 && s < 8 ? n[s] : "?"; }
// What the replay cannot run yet, decided once per shader: refused shaders keep their entry, so the draws that use them are skipped and counted
static const char *refusal(const vka_shlib *L, const dmsh_vk_entry *e) {
    if (e->flags & (DMSH_VK_F_NV_PASSTHROUGH | DMSH_VK_F_NV_VIEWPORT)) return "NV-only geometry passthrough / viewport mask (a library built before the gs_passthrough answer)";
    if (e->flags & DMSH_VK_F_INTERLOCK) return "fragment shader interlock";
    if (e->stage == DMSH_VK_STAGE_GS && !V.geom) return "a geometry shader on a device without geometryShader";
    if ((e->stage == DMSH_VK_STAGE_TCS || e->stage == DMSH_VK_STAGE_TES) && !V.tess) return "a tessellation shader on a device without tessellationShader";
    // NVN2 task and mesh programs, and the pixel shaders that read their per-primitive outputs (valid only in a mesh pipeline): only with VK_EXT_mesh_shader (vka_dev.c)
    if ((e->stage == DMSH_VK_STAGE_TS || e->stage == DMSH_VK_STAGE_MS || (e->flags & DMSH_VK_F_MESH_PRIM)) && !V.mesh) return "a task / mesh shader (or a pixel shader of their per-primitive outputs) on a device without VK_EXT_mesh_shader (or DM_VK_MESH=0)";
    if (e->stage == DMSH_VK_STAGE_TS && !V.task) return "a task shader on a device with mesh shaders but without taskShader";
    if (e->stage == DMSH_VK_STAGE_TS || e->stage == DMSH_VK_STAGE_MS) {   // the workgroup (the entry's local) within the device's limit
        uint64_t inv = (uint64_t)e->local[0] * e->local[1] * e->local[2];
        if (!inv || inv > (e->stage == DMSH_VK_STAGE_TS ? V.task_inv : V.mesh_inv)) return "a task / mesh workgroup larger than the device's (maxTaskWorkGroupInvocations / maxMeshWorkGroupInvocations)";
    }
    const dmsh_vk_bind *b = vka_shlib_binds(L, e);
    for (unsigned i = 0; i < e->bind_count; i++) if (b[i].kind == DMSH_VK_KIND_UNKNOWN) return "a binding the build could not place";
    return NULL;
}
// A module of a stage sg_plan runs as one lane (no subgroup operations there on this device, or a subgroup size the library is not right for that cannot be pinned) has them
// rewritten as subgroups of one invocation (vka_spv_subgroup_single; Qualcomm's Adreno 840 driver has them in fragment, compute, task and mesh shaders, not in the vertex pipeline's). Logged once per shader key; a module that cannot be rewritten is used as it is, with a red line
static void single_lane(uint64_t key, int stage, uint32_t **w, size_t *nb) {
    if (stage < 0 || stage > 7 || !g_sg_one[stage]) return;   // sg_plan
    // A task or mesh workgroup uses its subgroup operations across the workgroup (output counts, compaction): one-lane subgroups give it wrong results. The Adreno 840's
    // driver has them in both stages at its fixed 64 (subgroupSupportedStages 0xf0), so this is a device the library was not made for
    if (stage >= DMSH_VK_STAGE_TS) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VRED("%s shader %016llx: its subgroup operations run as one lane (the device's subgroups in this stage do not fit the library): its output is likely wrong", stage == DMSH_VK_STAGE_TS ? "task" : "mesh", (unsigned long long)key); }
    size_t on = 0; int ch = 0; char e[256] = ""; uint32_t *o = vka_spv_subgroup_single(*w, *nb / 4, &on, &ch, e, sizeof e);
    if (!o) { if (e[0]) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VRED("shader %016llx: subgroup operations in a stage without them not rewritten: %s", (unsigned long long)key, e); } return; }
    static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VLOG("shader %016llx: subgroup operations rewritten as one-invocation subgroups (%s)", (unsigned long long)key,
                                                            stage == DMSH_VK_STAGE_GS && V.driver_id == VK_DRIVER_ID_MESA_TURNIP ? "Turnip's geometry stage (DM_VK_SUBGROUP_GS)" : V.subgroup_stages & g_stage_bit[stage] ? "the device's subgroup size in this stage is not the library's and cannot be pinned" : "the device has none in this stage");
    free(*w); *w = o; *nb = on * 4;
}
// A task or mesh module on Qualcomm's driver takes its lane and its subgroup masks from LocalInvocationIndex (vka_spv_subgroup_from_index: a driver workaround, the
// measurements with it). Only for a workgroup of at most 32 invocations, the case measured (one workgroup = one subgroup, ballot bits in index order).
// DM_VK_MESH_INDEX=0 leaves the modules as built (a diagnostic; restart only). Logged for the first shaders; a module that cannot be rewritten is used as it is, with a red line
static void mesh_index_lanes(uint64_t key, const dmsh_vk_entry *e, uint32_t **w, size_t *nb) {
    if ((e->stage != DMSH_VK_STAGE_TS && e->stage != DMSH_VK_STAGE_MS) || V.driver_id != VK_DRIVER_ID_QUALCOMM_PROPRIETARY || g_sg_one[e->stage]) return;
    if ((uint64_t)e->local[0] * e->local[1] * e->local[2] > 32 || !VKA_ENV("DM_VK_MESH_INDEX", 1)) return;
    size_t on = 0; int ch = 0; char er[256] = ""; uint32_t *o = vka_spv_subgroup_from_index(*w, *nb / 4, &on, &ch, er, sizeof er);
    if (!o) { if (er[0]) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VRED("%s shader %016llx: subgroup built-ins not taken from the invocation index: %s", stage_name(e->stage), (unsigned long long)key, er); } return; }
    static _Atomic int n; if (atomic_fetch_add(&n, 1) < 4) VLOG("%s shader %016llx: subgroup id and masks taken from LocalInvocationIndex (Qualcomm driver workaround)", stage_name(e->stage), (unsigned long long)key);
    free(*w); *w = o; *nb = on * 4;
}
// A mesh module's output stores go through the translator's mesh_plane_store, which tests the address against every output plane at each call; each call is sent to a
// copy made for its own plane (vka_spv_mesh_store_planes, with the measurements: Ocarina of Time's 35 drawn task / mesh modules compile in 5.1 s instead of 12.4 s on
// the Adreno 840, and in 22.6 s with real lane ids and the calls left as built, where the largest took 1.7 GB and the process aborted). Every driver: the module does
// the same with less code. DM_VK_MESH_PLANES=0 leaves the modules as built (a diagnostic; restart only). A module that cannot be rewritten is used as it is, with a red line
static void mesh_store_planes(uint64_t key, const dmsh_vk_entry *e, uint32_t **w, size_t *nb) {
    if (e->stage != DMSH_VK_STAGE_MS || !VKA_ENV("DM_VK_MESH_PLANES", 1)) return;
    size_t on = 0; int ch = 0; char er[256] = ""; uint32_t *o = vka_spv_mesh_store_planes(*w, *nb / 4, &on, &ch, er, sizeof er);
    if (!o) { if (er[0]) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VRED("mesh shader %016llx: output stores not specialised by plane: %s", (unsigned long long)key, er); } return; }
    static _Atomic int n; if (atomic_fetch_add(&n, 1) < 4) VLOG("mesh shader %016llx: %d output stores sent to per-plane copies of mesh_plane_store", (unsigned long long)key, ch);
    free(*w); *w = o; *nb = on * 4;
}
static VkShaderModule spv_module(const vka_shlib *L, const dmsh_vk_entry *e, const dmsh_vk_module *m, uint64_t key, const char **why, char *err, size_t errn) {
    size_t nb = 0; uint32_t *w = vka_shlib_spirv(L, m, &nb, err, errn);
    if (!w) { *why = err; return VK_NULL_HANDLE; }
    if (e->flags & (DMSH_VK_F_SUBGROUP | DMSH_VK_F_TEXFB)) single_lane(key, e->stage, &w, &nb);
    mesh_index_lanes(key, e, &w, &nb);
    mesh_store_planes(key, e, &w, &nb);
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = nb, .pCode = w};
    VkShaderModule mod = VK_NULL_HANDLE; VkResult r = vk.vkCreateShaderModule(V.dev, &ci, NULL, &mod);
    free(w);
    if (r) { snprintf(err, errn, "vkCreateShaderModule %d (variant %u of %016llx)", r, m->variant, (unsigned long long)key); *why = err; return VK_NULL_HANDLE; }
    return mod;
}
VkShaderModule vka_shader_module(vka_shader *s, int variant) {
    if (variant < 0 || variant > DMSH_VK_MOD_SHUFFLE_LDS) return VK_NULL_HANDLE;
    if (s->mods[variant]) return s->mods[variant];
    const dmsh_vk_module *m = vka_shlib_module(s->lib, s->e, variant);
    if (!m) return VK_NULL_HANDLE;
    const char *why = NULL; char err[256];
    VkShaderModule mod = spv_module(s->lib, s->e, m, s->key, &why, err, sizeof err);
    if (!mod) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VRED("shader %016llx variant %d: %s", (unsigned long long)s->key, variant, why); return VK_NULL_HANDLE; }
    pthread_mutex_lock(&g_mu);
    if (s->mods[variant]) { pthread_mutex_unlock(&g_mu); vk.vkDestroyShaderModule(V.dev, mod, NULL); return s->mods[variant]; }
    s->mods[variant] = mod;
    pthread_mutex_unlock(&g_mu);
    return mod;
}
// A task or mesh module on Qualcomm's driver without its specialization constants: frozen at the values the pipeline would give them (gl: DMSH_VK_SPEC_GL_DEPTH;
// texfb: DMSH_VK_SPEC_TEXFB, the same for the whole process, so the modules are kept by gl alone). That driver hands a mesh module that declares one a lane id and
// subgroup masks of 0 (vka_spv_spec_freeze, with the measurements), and every translated mesh module declares dm_gl_depth. NULL = not such a module, or one without
// any constant (remembered), or DM_VK_MESH_SPEC=0 (a diagnostic; restart only): the pipeline then takes the base module and specializes it as everywhere else.
// Made from the base module's words with the per-plane output stores (mesh_store_planes), not the index lanes (mesh_index_lanes: the lane id reads right here)
// TEMP diagnostic (DM_VK_MESH_SPEC=5): the wrapper's closing depth fix-up stores dropped from a mesh module, in place (the position stays as mesh_plane_store wrote it)
static int temp_drop_depth_fixup(uint32_t *w, size_t *nw) {
    uint32_t bound = w[3], gd = 0; int n = 0; uint8_t *sel = calloc(bound, 1); if (!sel) return 0;
    for (size_t i = 5; i < *nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len) break;
        if (op == 71 && len >= 4 && w[i + 2] == 1 && w[i + 3] == DMSH_VK_SPEC_GL_DEPTH) gd = w[i + 1];
        if (op == 169 && len == 6 && gd && w[i + 3] == gd && w[i + 2] < bound) sel[w[i + 2]] = 1;
        i += len; }
    size_t o = 5;
    for (size_t i = 5; i < *nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len) break;
        if (op == 62 && len >= 3 && w[i + 2] < bound && sel[w[i + 2]]) { n++; i += len; continue; }
        memmove(w + o, w + i, (size_t)len * 4); o += len; i += len; }
    *nw = o; free(sel); return n;
}
// TEMP diagnostic (DM_VK_MESH_SPEC=6): every float arithmetic result of the module decorated NoContraction and its Position built-in Invariant, so that two modules
// with the same position math give the same position (a depth pre-pass and a later pass of the same geometry). Replaces *w (malloc'd); the count of decorations added
static int temp_precise(uint32_t **pw, size_t *pnb) {
    uint32_t *w = *pw; size_t nw = *pnb / 4, first = 0, add = 0;
    for (size_t i = 5; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len) return 0;
        if (!first && (op == 71 || op == 72)) first = i;
        if (op == 72 && len >= 5 && w[i + 3] == 11 && w[i + 4] == 0) add += 4;
        if (op == 127 || op == 129 || op == 131 || op == 133 || op == 136 || (op >= 140 && op <= 148)) add += 3;
        i += len; }
    if (!first || !add) return 0;
    uint32_t *o = malloc((nw + add) * 4); if (!o) return 0;
    memcpy(o, w, first * 4); size_t n = first; int c = 0;
    for (size_t i = 5; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == 72 && len >= 5 && w[i + 3] == 11 && w[i + 4] == 0) { o[n++] = 4u << 16 | 72; o[n++] = w[i + 1]; o[n++] = w[i + 2]; o[n++] = 18; c++; }
        if (op == 127 || op == 129 || op == 131 || op == 133 || op == 136 || (op >= 140 && op <= 148)) { o[n++] = 3u << 16 | 71; o[n++] = w[i + 2]; o[n++] = 42; c++; }
        i += len; }
    memcpy(o + n, w + first, (nw - first) * 4); n += nw - first;
    free(w); *pw = o; *pnb = n * 4; return c;
}
// TEMP diagnostic (DM_VK_MESH_SPEC=7): every vertex's position x and y stored 0 by the wrapper before the game's code runs (it stores z 0 and w 1 there already), so
// that a primitive of vertices the game's code never wrote has no area. Replaces *pw (malloc'd); the vertices per invocation, 0 when the module is not of that shape
static int temp_init_xy(uint32_t **pw, size_t *pnb) {
    uint32_t *w = *pw; size_t nw = *pnb / 4, fn = 0, smo = 0; uint32_t bound = w[3], L = 0, N = 0, S = 0, tu = 0, tf = 0, A = 0, P = 0, V = 0, pf = 0, lii = 0;
    for (size_t i = 5; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len) return 0;
        if (op == 16 && w[i + 2] == 17) L = w[i + 3] * w[i + 4] * w[i + 5];
        if (op == 16 && w[i + 2] == 26) N = w[i + 3];
        if (op == 72 && len >= 5 && w[i + 2] == 0 && w[i + 3] == 11 && w[i + 4] == 0) S = w[i + 1];
        if (op == 71 && len >= 4 && w[i + 2] == 11 && w[i + 3] == 29) lii = w[i + 1];
        if (op == 21 && w[i + 2] == 32 && w[i + 3] == 0) tu = w[i + 1];
        if (op == 22 && w[i + 2] == 32) tf = w[i + 1];
        if (op == 28 && S && w[i + 2] == S) A = w[i + 1];
        if (op == 32 && w[i + 2] == 3 && A && w[i + 3] == A) P = w[i + 1];
        if (op == 32 && w[i + 2] == 3 && tf && w[i + 3] == tf) pf = w[i + 1];
        if (op == 59 && P && w[i + 1] == P) V = w[i + 2];
        if (op == 54 && !fn) fn = i;
        if (op == 5295 && !smo) smo = i + len;
        i += len; }
    uint32_t K = L ? N / L : 0;
    if (!L || !N || N % L || K > 8 || !tu || !tf || !V || !pf || !lii || !fn || !smo) return 0;
    uint32_t *o = malloc((nw + 16 + 4 * K + 4 + 25 * K) * 4); if (!o) return 0;
    uint32_t id = bound, f0 = id++, u0 = id++, u1 = id++, kl[8] = {0}; for (uint32_t k = 1; k < K; k++) kl[k] = id++;
    memcpy(o, w, fn * 4); size_t n = fn;
#define E4(a, b, c, d) (o[n++] = (a), o[n++] = (b), o[n++] = (c), o[n++] = (d))
    E4(4u << 16 | 43, tf, f0, 0); E4(4u << 16 | 43, tu, u0, 0); E4(4u << 16 | 43, tu, u1, 1);
    for (uint32_t k = 1; k < K; k++) E4(4u << 16 | 43, tu, kl[k], k * L);
    memcpy(o + n, w + fn, (smo - fn) * 4); n += smo - fn;
    uint32_t idx = id++; E4(4u << 16 | 61, tu, idx, lii);
    for (uint32_t k = 0; k < K; k++) { uint32_t v = idx;
        if (k) { v = id++; E4(5u << 16 | 128, tu, v, idx); o[n++] = kl[k]; }
        for (uint32_t c = 0; c < 2; c++) { uint32_t p = id++; E4(7u << 16 | 65, pf, p, V); o[n++] = v; o[n++] = u0; o[n++] = c ? u1 : u0; o[n++] = 3u << 16 | 62; o[n++] = p; o[n++] = f0; } }
#undef E4
    memcpy(o + n, w + smo, (nw - smo) * 4); n += nw - smo; o[3] = id;
    free(w); *pw = o; *pnb = n * 4; return (int)K;
}
// TEMP diagnostic (DM_VK_MESH_SPEC=8): the wrapper's w of a vertex starts as a marker instead of 1.0 in mesh_position_zw, and a primitive with a corner that still
// carries it after the game's code (a vertex that code never stored) gets the indices 0, 0, 0, so it has no area. Replaces *pw (malloc'd); the index stores rewritten
static int temp_cull_unwritten(uint32_t **pw, size_t *pnb) {
    uint32_t *w = *pw; size_t nw = *pnb / 4, fn = 0; uint32_t bound = w[3], zw = 0, pv = 0, tu = 0, tb = 0, tv3 = 0, pwu = 0, pov3 = 0, one[8], none = 0, stores = 0;
    for (size_t i = 5; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) return 0;
        if (op == 5 && len == 7 && !memcmp(&w[i + 2], "mesh_position_zw", 17)) zw = w[i + 1];
        if (op == 71 && len >= 4 && w[i + 2] == 11 && w[i + 3] == 5296) pv = w[i + 1];
        if (op == 21 && w[i + 2] == 32 && w[i + 3] == 0) tu = w[i + 1];
        if (op == 20) tb = w[i + 1];
        if (op == 23 && tu && w[i + 2] == tu && w[i + 3] == 3) tv3 = w[i + 1];
        if (op == 32 && w[i + 2] == 4 && tu && w[i + 3] == tu) pwu = w[i + 1];
        if (op == 32 && w[i + 2] == 3 && tv3 && w[i + 3] == tv3) pov3 = w[i + 1];
        if (op == 43 && len == 4 && tu && w[i + 1] == tu && w[i + 3] == 0x3f800000 && none < 8) one[none++] = w[i + 2];
        if (op == 62 && pv) stores++;
        if (op == 54 && !fn) fn = i;
        i += len; }
    if (!zw || !pv || !tu || !tb || !tv3 || !pwu || !pov3 || !none || !fn) return 0;
    uint8_t *kind = calloc(bound, 1); uint32_t *o = malloc((nw + 16 + (size_t)stores * 120) * 4); if (!kind || !o) { free(kind); free(o); return 0; }
    uint32_t id = bound, u0 = id++, u1 = id++, u2 = id++, mark = id++; int done = 0;
    memcpy(o, w, fn * 4); size_t n = fn;
#define E4(a, b, c, d) (o[n++] = (a), o[n++] = (b), o[n++] = (c), o[n++] = (d))
    E4(4u << 16 | 43, tu, u0, 0); E4(4u << 16 | 43, tu, u1, 1); E4(4u << 16 | 43, tu, u2, 2); E4(4u << 16 | 43, tu, mark, 0xdeadbeef);
    for (size_t i = fn; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == 65 && len >= 5 && w[i + 2] < bound) kind[w[i + 2]] = w[i + 3] == zw ? 1 : w[i + 3] == pv && w[i + 1] == pov3 && len == 5 ? 2 : 0;
        if (op == 62 && len == 3 && w[i + 1] < bound && kind[w[i + 1]] == 1) { int is1 = 0; for (uint32_t k = 0; k < none; k++) is1 |= one[k] == w[i + 2];
            if (is1) { o[n++] = w[i]; o[n++] = w[i + 1]; o[n++] = mark; i += len; continue; } }
        if (op == 62 && len == 3 && w[i + 1] < bound && kind[w[i + 1]] == 2) { uint32_t c[3], b[3], s[3], any, v;
            for (uint32_t k = 0; k < 3; k++) { uint32_t m = id + 1, a = id + 2, p = id + 3, x = id + 4; c[k] = id; b[k] = id + 5; id += 6;
                E4(5u << 16 | 81, tu, c[k], w[i + 2]); o[n++] = k;
                E4(5u << 16 | 132, tu, m, c[k]); o[n++] = u2;
                E4(5u << 16 | 128, tu, a, m); o[n++] = u1;
                E4(5u << 16 | 65, pwu, p, zw); o[n++] = a;
                E4(4u << 16 | 61, tu, x, p);
                E4(5u << 16 | 170, tb, b[k], x); o[n++] = mark; }
            any = id++; E4(5u << 16 | 166, tb, any, b[0]); o[n++] = b[1];
            v = id++; E4(5u << 16 | 166, tb, v, any); o[n++] = b[2]; any = v;
            for (uint32_t k = 0; k < 3; k++) { s[k] = id++; E4(6u << 16 | 169, tu, s[k], any); o[n++] = u0; o[n++] = c[k]; }
            v = id++; E4(6u << 16 | 80, tv3, v, s[0]); o[n++] = s[1]; o[n++] = s[2];
            o[n++] = w[i]; o[n++] = w[i + 1]; o[n++] = v; done++; i += len; continue; }
        memcpy(o + n, w + i, len * 4); n += len; i += len; }
#undef E4
    o[3] = id; free(kind); free(w); *pw = o; *pnb = n * 4; return done;
}
// TEMP diagnostic (DM_VK_MESH_SPEC=9): the wrapper's index array (the Workgroup uint array holding the primitive count at [3] and the index bytes from [4]) filled with
// a marker before the game's code runs, and a primitive whose index bytes still carry it after that code (bytes this workgroup never wrote) gets the indices 0, 0, 0.
// Replaces *pw (malloc'd); the index stores rewritten
static int temp_cull_stale(uint32_t **pw, size_t *pnb) {
    uint32_t *w = *pw; size_t nw = *pnb / 4, fn = 0, smo = 0; uint32_t bound = w[3], pv = 0, tu = 0, tb = 0, tv3 = 0, pwu = 0, pov3 = 0, ix = 0, lii = 0, stores = 0, zw = 0;
    uint32_t *cv = calloc(bound, 4), *alen = calloc(bound, 4), *pt = calloc(bound, 4), *vt = calloc(bound, 4), L = 0, N = 0;
    if (!cv || !alen || !pt || !vt) { free(cv); free(alen); free(pt); free(vt); return 0; }
    for (size_t i = 5; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) break;
        if (op == 71 && len >= 4 && w[i + 2] == 11 && w[i + 3] == 5296) pv = w[i + 1];
        if (op == 5 && len == 7 && !memcmp(&w[i + 2], "mesh_position_zw", 17)) zw = w[i + 1];
        if (op == 71 && len >= 4 && w[i + 2] == 11 && w[i + 3] == 29) lii = w[i + 1];
        if (op == 21 && w[i + 2] == 32 && w[i + 3] == 0) tu = w[i + 1];
        if (op == 20) tb = w[i + 1];
        if (op == 23 && tu && w[i + 2] == tu && w[i + 3] == 3) tv3 = w[i + 1];
        if (op == 32 && w[i + 2] == 4 && tu && w[i + 3] == tu) pwu = w[i + 1];
        if (op == 32 && w[i + 2] == 3 && tv3 && w[i + 3] == tv3) pov3 = w[i + 1];
        if (op == 43 && len == 4 && tu && w[i + 1] == tu && w[i + 2] < bound) cv[w[i + 2]] = w[i + 3] + 1;
        if (op == 28 && w[i + 1] < bound) alen[w[i + 1]] = w[i + 3];
        if (op == 32 && w[i + 1] < bound) pt[w[i + 1]] = w[i + 3];
        if (op == 59 && w[i + 2] < bound) vt[w[i + 2]] = w[i + 1];
        if (op == 16 && w[i + 2] == 17) L = w[i + 3] * w[i + 4] * w[i + 5];
        if (op == 65 && len == 5 && smo && !ix && w[i + 1] == pwu && w[i + 4] < bound && cv[w[i + 4]] == 4) ix = w[i + 3];
        if (op == 62) stores++;
        if (op == 54 && !fn) fn = i;
        if (op == 5295 && !smo) smo = i + len;
        i += len; }
    if (ix && vt[ix] && pt[vt[ix]] && alen[pt[vt[ix]]] && cv[alen[pt[vt[ix]]]]) N = cv[alen[pt[vt[ix]]]] - 1;
    free(cv); free(alen); free(pt); free(vt);
    if (!pv || !tu || !tb || !tv3 || !pwu || !pov3 || !ix || ix == zw || L != 32 || N < 64 || N > 1024 || (N & (N - 1)) || !lii || !fn || !smo) return 0;
    uint32_t *chain = calloc(bound, 4); uint32_t *o = malloc((nw + 64 + 9 * N / 32 + (size_t)stores * 60) * 4); if (!chain || !o) { free(chain); free(o); return 0; }
    uint32_t id = bound, u0 = id++, u3 = id++, u4 = id++, u127 = id++, mark = id++, u1 = id++, u2 = id++, kk[32]; int done = 0;
    for (uint32_t k = 0; k < N / 32; k++) kk[k] = id++;
    memcpy(o, w, fn * 4); size_t n = fn;
#define E4(a, b, c, d) (o[n++] = (a), o[n++] = (b), o[n++] = (c), o[n++] = (d))
    E4(4u << 16 | 43, tu, u0, 0); E4(4u << 16 | 43, tu, u3, 3); E4(4u << 16 | 43, tu, u4, 4); E4(4u << 16 | 43, tu, u127, N - 1); E4(4u << 16 | 43, tu, mark, 0xdeadbeef); E4(4u << 16 | 43, tu, u1, 1); E4(4u << 16 | 43, tu, u2, 2);
    for (uint32_t k = 0; k < N / 32; k++) E4(4u << 16 | 43, tu, kk[k], 32 * k);
    memcpy(o + n, w + fn, (smo - fn) * 4); n += smo - fn;
    { uint32_t li = id++; E4(4u << 16 | 61, tu, li, lii);
      for (uint32_t k = 0; k < N / 32; k++) { uint32_t m = id++, p = id++;     // entry i + 32k (all N for 32 invocations), entries below 4 sent to 4
          E4(5u << 16 | 128, tu, m, li); o[n++] = kk[k];
          uint32_t lo = id++, sel = id++; E4(5u << 16 | 176, tb, lo, m); o[n++] = u4;   // < 4
          E4(6u << 16 | 169, tu, sel, lo); o[n++] = u4; o[n++] = m;
          E4(5u << 16 | 65, pwu, p, ix); o[n++] = sel; o[n++] = 3u << 16 | 62; o[n++] = p; o[n++] = mark; } }
    for (size_t i = smo; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == 65 && len == 5 && w[i + 1] == pov3 && w[i + 3] == pv && w[i + 2] < bound) chain[w[i + 2]] = w[i + 4] + 1;
        if (op == 62 && len == 3 && w[i + 1] < bound && chain[w[i + 1]]) { uint32_t pi = chain[w[i + 1]] - 1, t = id++, b[3], any, v, s[3], c[3];
            E4(5u << 16 | 132, tu, t, pi); o[n++] = u3;
            for (uint32_t k = 0; k < 3; k++) { uint32_t a = id++, m = id++, p = id++, x = id++; b[k] = id++;   // byte k at (3i + 4 + k) & (N - 1)
                E4(5u << 16 | 128, tu, a, t); o[n++] = u4;
                if (k) { uint32_t a2 = id++; E4(5u << 16 | 128, tu, a2, a); o[n++] = k == 1 ? u1 : u2; a = a2; }
                E4(5u << 16 | 199, tu, m, a); o[n++] = u127;
                E4(5u << 16 | 65, pwu, p, ix); o[n++] = m; E4(4u << 16 | 61, tu, x, p);
                E4(5u << 16 | 170, tb, b[k], x); o[n++] = mark; }
            any = id++; E4(5u << 16 | 166, tb, any, b[0]); o[n++] = b[1]; v = id++; E4(5u << 16 | 166, tb, v, any); o[n++] = b[2]; any = v;
            for (uint32_t k = 0; k < 3; k++) { c[k] = id++; E4(5u << 16 | 81, tu, c[k], w[i + 2]); o[n++] = k; s[k] = id++; E4(6u << 16 | 169, tu, s[k], any); o[n++] = u0; o[n++] = c[k]; }
            v = id++; E4(6u << 16 | 80, tv3, v, s[0]); o[n++] = s[1]; o[n++] = s[2];
            o[n++] = w[i]; o[n++] = w[i + 1]; o[n++] = v; done++; i += len; continue; }
        memcpy(o + n, w + i, len * 4); n += len; i += len; }
#undef E4
    o[3] = id; free(chain); free(w); *pw = o; *pnb = n * 4; return done;
}
// TEMP diagnostic (DM_VK_MESH_SPEC=11): a mesh module's SubgroupLocalInvocationId and SubgroupLtMask taken from where the driver puts the invocation in its own
// ballots: at the start of main every invocation runs 32 ballots of "LocalInvocationIndex == j" and keeps the one for its own index, so its lane is the bit it has in
// any ballot and its Lt mask the bits below it. For a driver whose lane id and masks do not match its ballot bit order in real draws. Replaces *pw (malloc'd); the
// loads redirected, 0 when the module is not of that shape
static int temp_ballot_lanes(uint32_t **pw, size_t *pnb) {
    uint32_t *w = *pw; size_t nw = *pnb / 4, fn = 0, smo = 0, ep = 0; uint32_t bound = w[3], sv = 0, ltv = 0, lii = 0, tu = 0, tb = 0, tv4 = 0, ppu = 0, ppv = 0, svt = 0; size_t fd = 0;
    for (size_t i = 5; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) return 0;
        if (op == 71 && len >= 4 && w[i + 2] == 11 && w[i + 3] == 41) sv = w[i + 1];
        if (op == 71 && len >= 4 && w[i + 2] == 11 && w[i + 3] == 4420) ltv = w[i + 1];
        if (op == 71 && len >= 4 && w[i + 2] == 11 && (w[i + 3] >= 4416 && w[i + 3] <= 4419)) return 0;   // other masks: not handled
        if (op == 71 && len >= 4 && w[i + 2] == 11 && w[i + 3] == 29) lii = w[i + 1];
        if (op == 15 && !ep) ep = i;
        if (op == 71 && !fd) fd = i;
        if (op == 59 && len >= 4 && sv && w[i + 2] == sv) svt = w[i + 1];
        if (op == 21 && w[i + 2] == 32 && w[i + 3] == 0) tu = w[i + 1];
        if (op == 20) tb = w[i + 1];
        if (op == 23 && tu && w[i + 2] == tu && w[i + 3] == 4) tv4 = w[i + 1];
        if (op == 32 && w[i + 2] == 6 && tu && w[i + 3] == tu) ppu = w[i + 1];
        if (op == 32 && w[i + 2] == 6 && tv4 && w[i + 3] == tv4) ppv = w[i + 1];
        if (op == 54 && !fn) fn = i;
        if (op == 5295 && !smo) smo = i + len;
        i += len; }
    if ((!sv && !ltv) || (!lii && !svt) || !fd || fd < ep || !tu || !tb || !tv4 || !fn || !smo || !ep || w[1] < 0x10400) return 0;
    uint32_t *o = malloc((nw + 2048) * 4); if (!o) return 0;
    uint32_t id = bound, c[32], u0 = id++, u1 = id++, u3 = id++, u32 = id++, uff = id++, pl = id++, pm = id++, ppu_new = 0, ppv_new = 0, lii_new = 0; int done = 0;
    if (!lii) lii = lii_new = id++;
    for (int j = 0; j < 32; j++) c[j] = id++;
    if (!ppu) ppu = ppu_new = id++;
    if (!ppv) ppv = ppv_new = id++;
    size_t n = 0, el = w[ep] >> 16;
    memcpy(o, w, ep * 4); n = ep; o[n++] = (uint32_t)(el + 2) << 16 | 15; memcpy(o + n, w + ep + 1, (el - 1) * 4); n += el - 1; o[n++] = pl; o[n++] = pm; if (lii_new) { o[n++] = lii_new; o[ep] += 1u << 16; }
    memcpy(o + n, w + ep + el, (fd - ep - el) * 4); n += fd - ep - el;
    if (lii_new) { o[n++] = 4u << 16 | 71; o[n++] = lii_new; o[n++] = 11; o[n++] = 29; }
    memcpy(o + n, w + fd, (fn - fd) * 4); n += fn - fd;
#define E4(a, b, c_, d) (o[n++] = (a), o[n++] = (b), o[n++] = (c_), o[n++] = (d))
    if (ppu_new) E4(4u << 16 | 32, ppu, 6, tu);
    if (ppv_new) E4(4u << 16 | 32, ppv, 6, tv4);
    E4(4u << 16 | 43, tu, u0, 0); E4(4u << 16 | 43, tu, u1, 1); E4(4u << 16 | 43, tu, u3, 3); E4(4u << 16 | 43, tu, u32, 32); E4(4u << 16 | 43, tu, uff, 0xffffffffu);
    for (int j = 0; j < 32; j++) E4(4u << 16 | 43, tu, c[j], (uint32_t)j);
    E4(4u << 16 | 59, ppu, pl, 6); E4(4u << 16 | 59, ppv, pm, 6); if (lii_new) E4(4u << 16 | 59, svt, lii_new, 1);
    uint32_t cur = 0;
    for (size_t i = fn; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16;   // every read redirected; main must not read them before OpSetMeshOutputsEXT
        if (op == 54) cur = w[i + 2];
        if (i == smo) { size_t at = n; (void)at; break; }
        if ((op == 61 || op == 65) && len >= 4 && sv && w[i + 3] == sv) { if (cur == w[ep + 2]) { free(o); return 0; } memcpy(o + n, w + i, len * 4); o[n + 3] = pl; n += len; i += len; done++; continue; }
        if (op == 61 && len >= 4 && ltv && w[i + 3] == ltv) { if (cur == w[ep + 2]) { free(o); return 0; } memcpy(o + n, w + i, len * 4); o[n + 3] = pm; n += len; i += len; done++; continue; }
        if (op == 65 && len >= 4 && ltv && w[i + 3] == ltv) { if (cur == w[ep + 2]) { free(o); return 0; } memcpy(o + n, w + i, len * 4); o[n + 1] = ppu; o[n + 3] = pm; n += len; i += len; done++; continue; }
        memcpy(o + n, w + i, len * 4); n += len; i += len; }
    { uint32_t li = id++, ax = 0, ay = 0; E4(4u << 16 | 61, tu, li, lii);
      for (int j = 0; j < 32; j++) { uint32_t e = id++, b = id++, bx = id++, by = id++;
          E4(5u << 16 | 170, tb, e, li); o[n++] = c[j];
          E4(5u << 16 | 339, tv4, b, u3); o[n++] = e;
          E4(5u << 16 | 81, tu, bx, b); o[n++] = 0; E4(5u << 16 | 81, tu, by, b); o[n++] = 1;
          if (!j) { ax = bx; ay = by; continue; }
          uint32_t nx = id++, ny = id++; E4(6u << 16 | 169, tu, nx, e); o[n++] = bx; o[n++] = ax; E4(6u << 16 | 169, tu, ny, e); o[n++] = by; o[n++] = ay; ax = nx; ay = ny; }
      uint32_t nz = id++, xm = id++, ym = id++, px = id++, py = id++, py2 = id++, ln = id++, w0 = id++, w1 = id++, lt = id++;
      E4(5u << 16 | 171, tb, nz, ax); o[n++] = u0;
      E4(5u << 16 | 130, tu, xm, ax); o[n++] = u1; E4(5u << 16 | 130, tu, ym, ay); o[n++] = u1;
      E4(4u << 16 | 205, tu, px, xm); E4(4u << 16 | 205, tu, py, ym); E4(5u << 16 | 128, tu, py2, py); o[n++] = u32;
      E4(6u << 16 | 169, tu, ln, nz); o[n++] = px; o[n++] = py2;
      E4(6u << 16 | 169, tu, w0, nz); o[n++] = xm; o[n++] = uff;
      E4(6u << 16 | 169, tu, w1, nz); o[n++] = u0; o[n++] = ym;
      o[n++] = 7u << 16 | 80; o[n++] = tv4; o[n++] = lt; o[n++] = w0; o[n++] = w1; o[n++] = u0; o[n++] = u0;
      o[n++] = 3u << 16 | 62; o[n++] = pl; o[n++] = ln; o[n++] = 3u << 16 | 62; o[n++] = pm; o[n++] = lt; }
    for (size_t i = smo; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == 61 && len >= 4 && sv && w[i + 3] == sv) { memcpy(o + n, w + i, len * 4); o[n + 3] = pl; n += len; i += len; done++; continue; }
        if (op == 61 && len >= 4 && ltv && w[i + 3] == ltv) { memcpy(o + n, w + i, len * 4); o[n + 3] = pm; n += len; i += len; done++; continue; }
        if (op == 65 && len >= 4 && ltv && w[i + 3] == ltv) { memcpy(o + n, w + i, len * 4); o[n + 1] = ppu; o[n + 3] = pm; n += len; i += len; done++; continue; }
        memcpy(o + n, w + i, len * 4); n += len; i += len; }
#undef E4
    o[3] = id; free(w); *pw = o; *pnb = n * 4; return done;
}
// TEMP diagnostic (DM_VK_MESH_SPEC=12): a mesh module's subgroup ballots made in Workgroup memory instead of by the driver: each invocation ORs (pred ? 1 << index
// : 0) into a word of its own for the site, all wait at a workgroup barrier, then read the word (bits in LocalInvocationIndex order, word 0). Only for a module whose
// ballots are all in the function main calls between its barriers, outside every selection and before any return there (so every invocation reaches them); the lane
// id and masks then come from the index (mesh_index_lanes). Replaces *pw (malloc'd); the ballots rewritten, 0 when the module is not of that shape
static int temp_shared_ballot(uint32_t **pw, size_t *pnb) {
    uint32_t *w = *pw; size_t nw = *pnb / 4, fn = 0, smo = 0, ep = 0; uint32_t bound = w[3], lii = 0, tu = 0, tv4 = 0, pwu = 0, gf = 0, S = 0, sawbar = 0;
    for (size_t i = 5; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) return 0;
        if (op == 71 && len >= 4 && w[i + 2] == 11 && w[i + 3] == 29) lii = w[i + 1];
        if (op == 15 && !ep) ep = i;
        if (op == 21 && w[i + 2] == 32 && w[i + 3] == 0) tu = w[i + 1];
        if (op == 23 && tu && w[i + 2] == tu && w[i + 3] == 4) tv4 = w[i + 1];
        if (op == 32 && w[i + 2] == 4 && tu && w[i + 3] == tu) pwu = w[i + 1];
        if (op == 54 && !fn) fn = i;
        if (op == 5295 && !smo) smo = i + len;
        if (smo && !gf && op == 224) sawbar = 1;
        if (smo && sawbar && !gf && op == 57) gf = w[i + 3];
        i += len; }
    if (!lii || !tu || !tv4 || !pwu || !fn || !smo || !ep || !gf || w[1] < 0x10400) return 0;
    { uint32_t cur = 0, depth = 0, ret = 0, merges[64];   // every ballot in gf, at depth 0, before any return inside a selection
      for (size_t i = fn; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
          if (op == 54) { cur = w[i + 2]; depth = 0; ret = 0; }
          if (op == 248) { for (uint32_t d = 0; d < depth; d++) if (merges[d] == w[i + 1]) { merges[d] = merges[--depth]; d--; } }
          if (op == 247) { if (depth == 64) return 0; merges[depth++] = w[i + 1]; }
          if ((op == 253 || op == 254 || op == 252 || op == 4416) && depth) ret = 1;
          if (op == 339) { if (cur != gf || depth || ret || len != 5) return 0; S++; }
          i += len; } }
    if (!S || S > 64) return 0;
    uint32_t *o = malloc((nw + 64 + 4 * S + S * 40) * 4); if (!o) return 0;
    uint32_t id = bound, u0 = id++, u1 = id++, u2 = id++, u264 = id++, cS = id++, arr = id++, parr = id++, sb = id++, ck[64]; int done = 0;
    for (uint32_t k = 0; k < S; k++) ck[k] = id++;
    size_t n = 0, el = w[ep] >> 16;
    memcpy(o, w, ep * 4); n = ep; o[n++] = (uint32_t)(el + 1) << 16 | 15; memcpy(o + n, w + ep + 1, (el - 1) * 4); n += el - 1; o[n++] = sb;
    memcpy(o + n, w + ep + el, (fn - ep - el) * 4); n += fn - ep - el;
#define E4(a, b, c_, d) (o[n++] = (a), o[n++] = (b), o[n++] = (c_), o[n++] = (d))
    E4(4u << 16 | 43, tu, u0, 0); E4(4u << 16 | 43, tu, u1, 1); E4(4u << 16 | 43, tu, u2, 2); E4(4u << 16 | 43, tu, u264, 264); E4(4u << 16 | 43, tu, cS, S);
    for (uint32_t k = 0; k < S; k++) E4(4u << 16 | 43, tu, ck[k], k);
    E4(4u << 16 | 28, arr, tu, cS); E4(4u << 16 | 32, parr, 4, arr); E4(4u << 16 | 59, parr, sb, 4);
    uint32_t site = 0;
    for (size_t i = fn; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (i == smo) for (uint32_t k = 0; k < S; k++) { uint32_t p = id++; E4(5u << 16 | 65, pwu, p, sb); o[n++] = ck[k]; o[n++] = 3u << 16 | 62; o[n++] = p; o[n++] = u0; }
        if (op == 339 && len == 5) { uint32_t li = id++, bit = id++, v = id++, p = id++, a = id++, x = id++;
            E4(4u << 16 | 61, tu, li, lii);
            E4(5u << 16 | 196, tu, bit, u1); o[n++] = li;
            E4(6u << 16 | 169, tu, v, w[i + 4]); o[n++] = bit; o[n++] = u0;
            E4(5u << 16 | 65, pwu, p, sb); o[n++] = ck[site++];
            E4(7u << 16 | 242, tu, a, p); o[n++] = u2; o[n++] = u0; o[n++] = v;
            E4(4u << 16 | 224, u2, u2, u264);
            E4(6u << 16 | 227, tu, x, p); o[n++] = u2; o[n++] = u0;
            o[n++] = 7u << 16 | 80; o[n++] = w[i + 1]; o[n++] = w[i + 2]; o[n++] = x; o[n++] = u0; o[n++] = u0; o[n++] = u0;
            done++; i += len; continue; }
        memcpy(o + n, w + i, len * 4); n += len; i += len; }
#undef E4
    o[3] = id; free(w); *pw = o; *pnb = n * 4; return done;
}
// TEMP diagnostic (DM_VK_MESH_SPEC=13): every Workgroup array of a mesh module doubled, and every access to it moved into the half chosen by WorkGroupID.x & 1, so
// two workgroups with neighbouring IDs never share an element even if the driver gives them the same storage. Replaces *pw (malloc'd); the accesses moved, 0 when
// the module is not of that shape
static int temp_wg_split(uint32_t **pw, size_t *pnb) {
    uint32_t *w = *pw; size_t nw = *pnb / 4, fn = 0; uint32_t bound = w[3], tu = 0, wid = 0, nacc = 0;
    uint32_t *cv = calloc(bound, 4), *arrlen = calloc(bound, 4), *arrel = calloc(bound, 4), *pt = calloc(bound, 4), *psc = calloc(bound, 4), *vty = calloc(bound, 4), *isv = calloc(bound, 4);
    uint32_t *o = 0; int done = 0;
#define OUT(r) do { free(cv); free(arrlen); free(arrel); free(pt); free(psc); free(vty); free(isv); return (r); } while (0)
    if (!cv || !arrlen || !arrel || !pt || !psc || !vty || !isv) OUT(0);
    for (size_t i = 5; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) OUT(0);
        if (op == 21 && w[i + 2] == 32 && w[i + 3] == 0) tu = w[i + 1];
        if (op == 43 && len == 4 && w[i + 2] < bound) cv[w[i + 2]] = w[i + 3] + 1;
        if (op == 28 && w[i + 1] < bound) { arrel[w[i + 1]] = w[i + 2]; arrlen[w[i + 1]] = w[i + 3]; }
        if (op == 32 && w[i + 1] < bound) { pt[w[i + 1]] = w[i + 3]; psc[w[i + 1]] = w[i + 2]; }
        if (op == 71 && len >= 4 && w[i + 2] == 11 && w[i + 3] == 26) wid = w[i + 1];
        if (op == 59 && w[i + 2] < bound) vty[w[i + 2]] = w[i + 1];
        if (op == 54 && !fn) fn = i;
        i += len; }
    if (!tu || !wid || !fn || !vty[wid]) OUT(0);
    uint32_t v3 = pt[vty[wid]];
    // Workgroup array variables: isv[var] = 1 + length
    for (uint32_t v = 1; v < bound; v++) { uint32_t p = vty[v]; if (!p || psc[p] != 4) continue; uint32_t a = pt[p];
        if (!arrlen[a] || !cv[arrlen[a]]) OUT(0);   // a Workgroup variable not an array of constant length: not handled
        isv[v] = cv[arrlen[a]]; }
    for (size_t i = fn; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if ((op == 61 && len >= 4 && isv[w[i + 3]]) || (op == 62 && isv[w[i + 1]])) OUT(0);   // whole-array load or store
        if ((op == 65 || op == 66) && len >= 5 && w[i + 3] < bound && isv[w[i + 3]]) nacc++;
        i += len; }
    if (!nacc) OUT(0);
    o = malloc((nw + 64 * 12 + (size_t)nacc * 24) * 4); if (!o) OUT(0);
    uint32_t id = bound, u1 = id++, nt[64], np[64], nc[64], cl[64], nv = 0, vv[64], vl[64];
    for (uint32_t v = 1; v < bound; v++) if (isv[v]) { if (nv == 64) { free(o); OUT(0); } vv[nv] = v; vl[nv] = isv[v] - 1; nc[nv] = id++; cl[nv] = id++; nt[nv] = id++; np[nv] = id++; nv++; }
    size_t n = 0;
#define E4(a, b, c_, d) (o[n++] = (a), o[n++] = (b), o[n++] = (c_), o[n++] = (d))
    for (size_t i = 0; i < fn;) { uint32_t op = i < 5 ? 0 : w[i] & 0xffff, len = i < 5 ? 5 : w[i] >> 16;
        if (op == 59 && w[i + 2] < bound && isv[w[i + 2]]) {   // the doubled array's types and constant, then the variable with that type
            for (uint32_t k = 0; k < nv; k++) if (vv[k] == w[i + 2]) { uint32_t el = arrel[pt[vty[vv[k]]]];
                E4(4u << 16 | 43, tu, nc[k], 2 * vl[k]); E4(4u << 16 | 43, tu, cl[k], vl[k]); E4(4u << 16 | 28, nt[k], el, nc[k]); E4(4u << 16 | 32, np[k], 4, nt[k]); E4(4u << 16 | 59, np[k], vv[k], 4); }
            i += len; continue; }
        memcpy(o + n, w + i, len * 4); n += len; i += len; }
    E4(4u << 16 | 43, tu, u1, 1);
    for (size_t i = fn; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if ((op == 65 || op == 66) && len >= 5 && w[i + 3] < bound && isv[w[i + 3]]) { uint32_t k = 0; while (vv[k] != w[i + 3]) k++;
            uint32_t g = id++, x = id++, b = id++, off = id++, ni = id++;
            E4(4u << 16 | 61, v3, g, wid);
            E4(5u << 16 | 81, tu, x, g); o[n++] = 0;
            E4(5u << 16 | 199, tu, b, x); o[n++] = u1;
            E4(5u << 16 | 132, tu, off, b); o[n++] = cl[k];
            E4(5u << 16 | 128, tu, ni, w[i + 4]); o[n++] = off;
            memcpy(o + n, w + i, len * 4); o[n + 4] = ni; n += len; done++; i += len; continue; }
        memcpy(o + n, w + i, len * 4); n += len; i += len; }
#undef E4
    o[3] = id; free(w); *pw = o; *pnb = n * 4; OUT(done);
#undef OUT
}
// TEMP diagnostic (DM_VK_MESH_SPEC=14): a mesh module's per-vertex outputs (mesh_position, mesh_outN) written only by the invocation that owns the vertex
// (vertex % 32 == LocalInvocationIndex): the game's code stores them to Workgroup copies instead, main copies its own vertices' initial values in before the first
// barrier and the results out after the second. For a driver that loses a store to another invocation's vertex. Replaces *pw (malloc'd); the accesses moved, 0 when
// the module is not of that shape. cull (DM_VK_MESH_SPEC=16): position x starts as a marker, and a primitive with a corner that still has it after the game's code gets
// the indices 0, 0, 0 (a vertex no invocation wrote)
static int temp_owner_outputs(uint32_t **pw, size_t *pnb, int cull) {
    uint32_t *w = *pw; size_t nw = *pnb / 4, fn = 0, smo = 0, ep = 0, b1 = 0, b2 = 0; uint32_t bound = w[3], tu = 0, tf = 0, pof = 0, lii = 0, L = 0, N = 0, mainf = 0, piv = 0, tv3 = 0, pov3 = 0, tb = 0;
    uint8_t *kind = calloc(bound, 1); uint32_t vars[16], vpos[16], nv = 0; if (!kind) return 0;
#define BAIL do { free(kind); return 0; } while (0)
    for (size_t i = 5; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) BAIL;
        if (op == 5 && len >= 4 && w[i + 1] < bound) { const char *s = (const char *)&w[i + 2];
            if (!strcmp(s, "mesh_position")) kind[w[i + 1]] = 2;
            else if (!strncmp(s, "mesh_out", 8) && s[8] >= '0' && s[8] <= '9' && (!s[9] || (s[9] >= '0' && s[9] <= '9' && !s[10]))) kind[w[i + 1]] = 1; }
        if (op == 16 && w[i + 2] == 17) L = w[i + 3] * w[i + 4] * w[i + 5];
        if (op == 16 && w[i + 2] == 26) N = w[i + 3];
        if (op == 71 && len >= 4 && w[i + 2] == 11 && w[i + 3] == 29) lii = w[i + 1];
        if (op == 15 && !ep) { ep = i; mainf = w[i + 2]; }
        if (op == 21 && w[i + 2] == 32 && w[i + 3] == 0) tu = w[i + 1];
        if (op == 22 && w[i + 2] == 32) tf = w[i + 1];
        if (op == 71 && len >= 4 && w[i + 2] == 11 && w[i + 3] == 5296) piv = w[i + 1];
        if (op == 20) tb = w[i + 1];
        if (op == 23 && tu && w[i + 2] == tu && w[i + 3] == 3) tv3 = w[i + 1];
        if (op == 32 && w[i + 2] == 3 && tv3 && w[i + 3] == tv3) pov3 = w[i + 1];
        if (op == 32 && w[i + 2] == 3 && tf && w[i + 3] == tf) pof = w[i + 1];
        if (op == 54 && !fn) fn = i;
        if (op == 5295 && !smo) smo = i + len;
        if (smo && op == 224) { if (!b1) b1 = i; else if (!b2) b2 = i + len; }
        i += len; }
    if (L != 32 || !N || N % 32 || N > 256 || !lii || !tu || !tf || !pof || !fn || !smo || !b1 || !b2 || !ep || w[1] < 0x10400 || (cull && (!piv || !tv3 || !pov3 || !tb))) BAIL;
    for (size_t i = 5; i < fn;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16;   // the Output variables with those names
        if (op == 59 && w[i + 3] == 3 && kind[w[i + 2]]) { if (nv == 16) BAIL; vars[nv] = w[i + 2]; vpos[nv] = kind[w[i + 2]] == 2; nv++; }
        i += len; }
    if (!nv) BAIL;
    { uint32_t cur = 0; for (size_t i = fn; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16;   // outside main: only [v, c] / [v, 0, c] float chains
        if (op == 54) cur = w[i + 2];
        if (cur != mainf && ((op == 61 && len >= 4 && w[i + 3] < bound && kind[w[i + 3]]) || (op == 62 && w[i + 1] < bound && kind[w[i + 1]]))) BAIL;   // whole loads / stores
        if (cur != mainf && op == 65 && len >= 4 && w[i + 3] < bound && kind[w[i + 3]]) { int isvar = 0, pos = 0; for (uint32_t k = 0; k < nv; k++) if (vars[k] == w[i + 3]) { isvar = 1; pos = vpos[k]; }
            if (isvar && (w[i + 1] != pof || len != (pos ? 7u : 6u))) BAIL; }
        i += len; } }
    uint32_t *o = malloc((nw + 256 + (size_t)nv * (N / 32) * 4 * 40 + (size_t)nw) * 4); if (!o) BAIL;
    uint32_t id = bound, mk = id++, mku = id++, u0 = id++, u4 = id++, cn = id++, ta = id++, pa = id++, pwf = id++, uc[4], c32[8], st[16]; int done = 0;
    for (int c = 0; c < 4; c++) uc[c] = id++;
    for (uint32_t j = 0; j < N / 32; j++) c32[j] = id++;
    for (uint32_t k = 0; k < nv; k++) st[k] = id++;
    size_t n = 0, el = w[ep] >> 16;
    memcpy(o, w, ep * 4); n = ep; o[n++] = (uint32_t)(el + nv) << 16 | 15; memcpy(o + n, w + ep + 1, (el - 1) * 4); n += el - 1; for (uint32_t k = 0; k < nv; k++) o[n++] = st[k];
    memcpy(o + n, w + ep + el, (fn - ep - el) * 4); n += fn - ep - el;
#define E4(a, b, c_, d) (o[n++] = (a), o[n++] = (b), o[n++] = (c_), o[n++] = (d))
    E4(4u << 16 | 43, tf, mk, 0xf2cafe00u); E4(4u << 16 | 43, tu, mku, 0xf2cafe00u);
    E4(4u << 16 | 43, tu, u0, 0); E4(4u << 16 | 43, tu, u4, 4); E4(4u << 16 | 43, tu, cn, N * 4);
    for (int c = 0; c < 4; c++) E4(4u << 16 | 43, tu, uc[c], (uint32_t)c);
    for (uint32_t j = 0; j < N / 32; j++) E4(4u << 16 | 43, tu, c32[j], 32 * j);
    E4(4u << 16 | 28, ta, tf, cn); E4(4u << 16 | 32, pa, 4, ta); E4(4u << 16 | 32, pwf, 4, tf);
    for (uint32_t k = 0; k < nv; k++) E4(4u << 16 | 59, pa, st[k], 4);
    // copy: dir 0 = Output -> stage, 1 = stage -> Output, for vertices li + 32j
    #define COPY(dir) do { uint32_t li = id++; E4(4u << 16 | 61, tu, li, lii); \
        for (uint32_t j = 0; j < N / 32; j++) { uint32_t v = id++, v4 = id++; E4(5u << 16 | 128, tu, v, li); o[n++] = c32[j]; E4(5u << 16 | 132, tu, v4, v); o[n++] = u4; \
            for (uint32_t k = 0; k < nv; k++) for (int c = 0; c < 4; c++) { uint32_t po = id++, ix = id++, ps = id++, x = id++; \
                if (vpos[k]) { o[n++] = 7u << 16 | 65; o[n++] = pof; o[n++] = po; o[n++] = vars[k]; o[n++] = v; o[n++] = u0; o[n++] = uc[c]; } \
                else { o[n++] = 6u << 16 | 65; o[n++] = pof; o[n++] = po; o[n++] = vars[k]; o[n++] = v; o[n++] = uc[c]; } \
                E4(5u << 16 | 128, tu, ix, v4); o[n++] = uc[c]; E4(5u << 16 | 65, pwf, ps, st[k]); o[n++] = ix; \
                if (!(dir)) { E4(4u << 16 | 61, tf, x, po); o[n++] = 3u << 16 | 62; o[n++] = ps; o[n++] = cull && vpos[k] && c == 0 ? mk : x; } \
                else { E4(4u << 16 | 61, tf, x, ps); o[n++] = 3u << 16 | 62; o[n++] = po; o[n++] = x; } } } } while (0)
    uint32_t cur = 0, pchain[64], npc = 0, kpos = 0; for (uint32_t k = 0; k < nv; k++) if (vpos[k]) kpos = st[k];
    for (size_t i = fn; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == 54) cur = w[i + 2];
        if (cull && cur == mainf && op == 65 && len == 5 && w[i + 1] == pov3 && w[i + 3] == piv && npc < 64) pchain[npc++] = w[i + 2];
        if (cull && kpos && cur == mainf && op == 62 && len == 3) { int hit = 0; for (uint32_t q = 0; q < npc; q++) hit |= pchain[q] == w[i + 1];
            if (hit) { uint32_t b[3], any, v, sc[3];
                for (uint32_t c = 0; c < 3; c++) { uint32_t e = id++, e4 = id++, p = id++, x = id++; b[c] = id++;
                    E4(5u << 16 | 81, tu, e, w[i + 2]); o[n++] = c;
                    E4(5u << 16 | 132, tu, e4, e); o[n++] = u4;
                    E4(5u << 16 | 65, pwf, p, kpos); o[n++] = e4;
                    { uint32_t xf = id++; E4(4u << 16 | 61, tf, xf, p); E4(4u << 16 | 124, tu, x, xf); }
                    E4(5u << 16 | 170, tb, b[c], x); o[n++] = mku; }
                any = id++; E4(5u << 16 | 166, tb, any, b[0]); o[n++] = b[1]; v = id++; E4(5u << 16 | 166, tb, v, any); o[n++] = b[2]; any = v;
                for (uint32_t c = 0; c < 3; c++) { uint32_t e = id++; E4(5u << 16 | 81, tu, e, w[i + 2]); o[n++] = c; sc[c] = id++; E4(6u << 16 | 169, tu, sc[c], any); o[n++] = u0; o[n++] = e; }
                v = id++; E4(6u << 16 | 80, tv3, v, sc[0]); o[n++] = sc[1]; o[n++] = sc[2];
                o[n++] = w[i]; o[n++] = w[i + 1]; o[n++] = v; i += len; continue; } }
        if (i == b1) COPY(0);
        if (i == b2) COPY(1);
        if (cur != mainf && op == 65 && w[i + 3] < bound && kind[w[i + 3]]) { int isvar = 0, pos = 0, k = 0; for (uint32_t q = 0; q < nv; q++) if (vars[q] == w[i + 3]) { isvar = 1; pos = vpos[q]; k = (int)q; }
            if (isvar) { uint32_t v4 = id++, ix = id++, cc = w[i + (pos ? 6 : 5)];
                E4(5u << 16 | 132, tu, v4, w[i + 4]); o[n++] = u4; E4(5u << 16 | 128, tu, ix, v4); o[n++] = cc;
                E4(5u << 16 | 65, pwf, w[i + 2], st[k]); o[n++] = ix; done++; i += len; continue; } }
        memcpy(o + n, w + i, len * 4); n += len; i += len; }
#undef COPY
#undef E4
    free(kind); o[3] = id; free(w); *pw = o; *pnb = n * 4; return done;
#undef BAIL
}
// TEMP diagnostic (DM_VK_MESH_SPEC=15): every GLSL.std.450 Fma of a mesh module split into an FMul and an FAdd decorated NoContraction, so that two pipelines
// computing the same position (a depth pre-pass and the pass that tests depth EQUAL against it) round the same way whether the driver fuses an Fma or not.
// Replaces *pw (malloc'd); the Fmas split, 0 when there are none
static int temp_split_fma(uint32_t **pw, size_t *pnb) {
    uint32_t *w = *pw; size_t nw = *pnb / 4, fd = 0; uint32_t bound = w[3], glsl = 0, nf = 0;
    for (size_t i = 5; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16; if (!len || i + len > nw) return 0;
        if (op == 11 && len >= 3 && !strcmp((const char *)&w[i + 2], "GLSL.std.450")) glsl = w[i + 1];
        if (op == 71 && !fd) fd = i;
        if (op == 12 && len == 8 && glsl && w[i + 3] == glsl && w[i + 4] == 50) nf++;
        i += len; }
    if (!nf || !fd) return 0;
    uint32_t *o = malloc((nw + (size_t)nf * 12 + 16) * 4); if (!o) return 0;
    uint32_t id = bound; size_t n = 0;
    memcpy(o, w, fd * 4); n = fd;
    size_t dec = n; n += (size_t)nf * 6;   // two NoContraction decorations per Fma: the FMul's new id and the FAdd (the Fma's id)
    uint32_t k = 0;
    for (size_t i = fd; i < nw;) { uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == 12 && len == 8 && glsl && w[i + 3] == glsl && w[i + 4] == 50) { uint32_t t = id + k;
            o[n++] = 5u << 16 | 133; o[n++] = w[i + 1]; o[n++] = t; o[n++] = w[i + 5]; o[n++] = w[i + 6];          // OpFMul
            o[n++] = 5u << 16 | 129; o[n++] = w[i + 1]; o[n++] = w[i + 2]; o[n++] = t; o[n++] = w[i + 7];          // OpFAdd, the Fma's id
            o[dec++] = 3u << 16 | 71; o[dec++] = t; o[dec++] = 42; o[dec++] = 3u << 16 | 71; o[dec++] = w[i + 2]; o[dec++] = 42; k++; i += len; continue; }
        memcpy(o + n, w + i, len * 4); n += len; i += len; }
    o[3] = id + nf; free(w); *pw = o; *pnb = n * 4; return (int)nf;
}
VkShaderModule vka_shader_frozen_module(vka_shader *s, int gl, int texfb) {
    int st = s->e->stage; gl = gl != 0;
    if ((st != DMSH_VK_STAGE_TS && st != DMSH_VK_STAGE_MS) || V.driver_id != VK_DRIVER_ID_QUALCOMM_PROPRIETARY || g_sg_one[st] || !VKA_ENV("DM_VK_MESH_SPEC", 1)) return VK_NULL_HANDLE;
    pthread_mutex_lock(&g_mu);
    VkShaderModule had = s->frozen[gl]; int none = s->frozen_none;
    pthread_mutex_unlock(&g_mu);
    if (had || none) return had;
    VkShaderModule mod = VK_NULL_HANDLE; char err[256] = ""; int ch = 0;
    const dmsh_vk_module *m = vka_shlib_module(s->lib, s->e, DMSH_VK_MOD_BASE);
    size_t nb = 0; uint32_t *w = m ? vka_shlib_spirv(s->lib, m, &nb, err, sizeof err) : NULL;
    int dmode = VKA_ENV("DM_VK_MESH_SPEC", 1);   // TEMP diagnostic: 2 = with the index lanes too, 3 = GL depth frozen false
    const uint32_t ids[2] = {DMSH_VK_SPEC_GL_DEPTH, DMSH_VK_SPEC_TEXFB}, vals[2] = {dmode >= 3 ? 0 : (uint32_t)gl, texfb != 0};
    if (w && dmode == 5 && st == DMSH_VK_STAGE_MS) { size_t n = nb / 4; int d = temp_drop_depth_fixup(w, &n); nb = n * 4; static _Atomic int c; if (atomic_fetch_add(&c, 1) < 4) VLOG("TEMP mesh %016llx: %d depth fix-up stores dropped", (unsigned long long)s->key, d); }
    size_t on = 0; uint32_t *o = w ? vka_spv_spec_freeze(w, nb / 4, ids, vals, 2, &on, &ch, err, sizeof err) : NULL;
    free(w);
    if (o) {
        nb = on * 4; if (dmode == 2 || dmode == 10) mesh_index_lanes(s->key, s->e, &o, &nb);
        mesh_store_planes(s->key, s->e, &o, &nb);
        if (dmode == 7 && st == DMSH_VK_STAGE_MS) { int c = temp_init_xy(&o, &nb); static _Atomic int k; if (atomic_fetch_add(&k, 1) < 40) VLOG("TEMP mesh %016llx: position x, y initialised, %d vertices per invocation", (unsigned long long)s->key, c); }
        if (dmode == 8 && st == DMSH_VK_STAGE_MS) { int c = temp_cull_unwritten(&o, &nb); static _Atomic int k; if (atomic_fetch_add(&k, 1) < 40) VLOG("TEMP mesh %016llx: %d index stores test for unwritten corners", (unsigned long long)s->key, c); }
        if (dmode == 9 && st == DMSH_VK_STAGE_MS) { int c = temp_cull_stale(&o, &nb); static _Atomic int k; if (atomic_fetch_add(&k, 1) < 40) VLOG("TEMP mesh %016llx: %d index stores test for stale index bytes", (unsigned long long)s->key, c); }
        if (dmode == 11 && st == DMSH_VK_STAGE_MS) { int c = temp_ballot_lanes(&o, &nb); static _Atomic int k; if (atomic_fetch_add(&k, 1) < 40) VLOG("TEMP mesh %016llx: %d lane reads from the ballot order", (unsigned long long)s->key, c); }
        if (dmode == 12 && st == DMSH_VK_STAGE_MS) { int c = temp_shared_ballot(&o, &nb); if (c) mesh_index_lanes(s->key, s->e, &o, &nb); static _Atomic int k; if (atomic_fetch_add(&k, 1) < 60) VLOG("TEMP mesh %016llx: %d ballots made in workgroup memory", (unsigned long long)s->key, c); }
        if (dmode == 13 && st == DMSH_VK_STAGE_MS) { int c = temp_wg_split(&o, &nb); static _Atomic int k; if (atomic_fetch_add(&k, 1) < 60) VLOG("TEMP mesh %016llx: %d workgroup accesses split by workgroup parity", (unsigned long long)s->key, c); }
        if (dmode == 14 && st == DMSH_VK_STAGE_MS) { int c = temp_owner_outputs(&o, &nb, 0); static _Atomic int k; if (atomic_fetch_add(&k, 1) < 60) VLOG("TEMP mesh %016llx: %d output accesses staged for the owning invocation", (unsigned long long)s->key, c); }
        if (dmode == 15 && st == DMSH_VK_STAGE_MS) { int p = temp_precise(&o, &nb); int c = temp_split_fma(&o, &nb); static _Atomic int k; if (atomic_fetch_add(&k, 1) < 60) VLOG("TEMP mesh %016llx: %d precise, %d Fma split", (unsigned long long)s->key, p, c); }
        if (dmode == 16 && st == DMSH_VK_STAGE_MS) { int c = temp_owner_outputs(&o, &nb, 1); static _Atomic int k; if (atomic_fetch_add(&k, 1) < 60) VLOG("TEMP mesh %016llx: %d output accesses staged, unwritten corners culled", (unsigned long long)s->key, c); }
        if (dmode == 6) { int c = temp_precise(&o, &nb); static _Atomic int k; if (atomic_fetch_add(&k, 1) < 4) VLOG("TEMP %s %016llx: %d precise decorations", stage_name(st), (unsigned long long)s->key, c); }
        VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = nb, .pCode = o};
        VkResult r = vk.vkCreateShaderModule(V.dev, &ci, NULL, &mod); if (r) { snprintf(err, sizeof err, "vkCreateShaderModule %d", r); mod = VK_NULL_HANDLE; }
        free(o);
    }
    if (!mod && err[0]) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VRED("%s shader %016llx: specialization constants not frozen: %s; the base module, whose lane ids this driver may hand out as 0", stage_name(st), (unsigned long long)s->key, err); }
    if (mod) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 4) VLOG("%s shader %016llx: %d specialization constant%s frozen (GL depth %d; Qualcomm driver workaround)", stage_name(st), (unsigned long long)s->key, ch, ch == 1 ? "" : "s", gl); }
    pthread_mutex_lock(&g_mu);
    if (s->frozen[gl]) { had = s->frozen[gl]; pthread_mutex_unlock(&g_mu); if (mod) vk.vkDestroyShaderModule(V.dev, mod, NULL); return had; }   // another thread made it meanwhile
    if (mod) s->frozen[gl] = mod; else s->frozen_none = 1;
    pthread_mutex_unlock(&g_mu);
    return mod;
}
// A geometry shader that writes memory, with those writes made by one run per primitive (vka_spv_gs_claim: the Adreno 840 runs a geometry shader several times for each
// primitive, each run writing memory). out NULL: only counts them. Made from the base module after the one-invocation subgroup rewrite
static int gs_words(vka_shader *s, uint32_t **out, size_t *on, char *err, size_t errn) {
    const dmsh_vk_module *m = vka_shlib_module(s->lib, s->e, DMSH_VK_MOD_BASE);
    size_t nb = 0; uint32_t *w = m ? vka_shlib_spirv(s->lib, m, &nb, err, errn) : NULL;
    if (!w) return -1;
    if (s->e->flags & (DMSH_VK_F_SUBGROUP | DMSH_VK_F_TEXFB)) single_lane(s->key, s->e->stage, &w, &nb);
    int r = vka_spv_gs_claim(w, nb / 4, VKA_GS_CLAIM_BINDING, out, on, err, errn);
    free(w); return r;
}
VkShaderModule vka_shader_gs_module(vka_shader *s) {
    if (s->gs_fx <= 0) return VK_NULL_HANDLE;
    if (s->gs_claim) return s->gs_claim;
    uint32_t *o = NULL; size_t on = 0; char err[256] = ""; VkShaderModule mod = VK_NULL_HANDLE;
    if (gs_words(s, &o, &on, err, sizeof err) > 0 && o) {
        VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = on * 4, .pCode = o};
        VkResult r = vk.vkCreateShaderModule(V.dev, &ci, NULL, &mod); if (r) { snprintf(err, sizeof err, "vkCreateShaderModule %d", r); mod = VK_NULL_HANDLE; }
    }
    free(o);
    if (!mod) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VRED("geometry shader %016llx: module with the claim not made: %s", (unsigned long long)s->key, err); return VK_NULL_HANDLE; }
    pthread_mutex_lock(&g_mu);
    if (s->gs_claim) { pthread_mutex_unlock(&g_mu); vk.vkDestroyShaderModule(V.dev, mod, NULL); return s->gs_claim; }
    s->gs_claim = mod;
    pthread_mutex_unlock(&g_mu);
    return mod;
}
// The base vertex module with the inputs of umask / smask retyped (vka_spvpatch.c): made once per pair of masks, kept for the process like the base module
struct vka_vinmod { struct vka_vinmod *next; uint32_t u, s; VkShaderModule mod; };
VkShaderModule vka_shader_vin_module(vka_shader *s, uint32_t umask, uint32_t smask) {
    pthread_mutex_lock(&g_mu);
    for (struct vka_vinmod *v = s->vin; v; v = v->next) if (v->u == umask && v->s == smask) { VkShaderModule m = v->mod; pthread_mutex_unlock(&g_mu); return m; }
    pthread_mutex_unlock(&g_mu);
    VkShaderModule mod = VK_NULL_HANDLE; char err[256] = ""; int own = 0;   // own: made here (the base module of the fallback is the shader's, never destroyed)
    const dmsh_vk_module *m = vka_shlib_module(s->lib, s->e, DMSH_VK_MOD_BASE);
    size_t nb = 0; uint32_t *w = m ? vka_shlib_spirv(s->lib, m, &nb, err, sizeof err) : NULL;
    if (w && (s->e->flags & (DMSH_VK_F_SUBGROUP | DMSH_VK_F_TEXFB))) single_lane(s->key, s->e->stage, &w, &nb);
    size_t on = 0; uint32_t *o = w ? vka_spv_vin_retype(w, nb / 4, umask, smask, &on, err, sizeof err) : NULL;
    if (o) { VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = on * 4, .pCode = o};
             VkResult r = vk.vkCreateShaderModule(V.dev, &ci, NULL, &mod); if (r) { snprintf(err, sizeof err, "vkCreateShaderModule %d", r); mod = VK_NULL_HANDLE; } else own = 1; }
    free(w); free(o);
    if (!mod) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VRED("vertex shader %016llx: inputs %#x (uint) / %#x (int) not retyped: %s; the base module reads them as floats", (unsigned long long)s->key, umask, smask, err);
                mod = vka_shader_module(s, DMSH_VK_MOD_BASE); }
    // Two threads (the frame path and a prewarm or optimizer thread) may have made it at once: the first one in the list wins, the other's module is destroyed
    pthread_mutex_lock(&g_mu);
    for (struct vka_vinmod *o = s->vin; o; o = o->next) if (o->u == umask && o->s == smask) {
        VkShaderModule w = o->mod; pthread_mutex_unlock(&g_mu);
        if (own && mod != w) vk.vkDestroyShaderModule(V.dev, mod, NULL);   // (no pipeline was made from it: it never left this function)
        return w; }
    struct vka_vinmod *v = malloc(sizeof *v); *v = (struct vka_vinmod){s->vin, umask, smask, mod}; s->vin = v;
    pthread_mutex_unlock(&g_mu);
    return mod;
}
// The base pixel module with the float outputs of umask / smask retyped for integer attachments (vka_spvpatch.c vka_spv_fout_retype): made once per pair of masks, as the vertex ones
VkShaderModule vka_shader_fout_module(vka_shader *s, uint32_t umask, uint32_t smask) {
    pthread_mutex_lock(&g_mu);
    for (struct vka_vinmod *v = s->fout; v; v = v->next) if (v->u == umask && v->s == smask) { VkShaderModule m = v->mod; pthread_mutex_unlock(&g_mu); return m; }
    pthread_mutex_unlock(&g_mu);
    VkShaderModule mod = VK_NULL_HANDLE; char err[256] = ""; int own = 0;
    const dmsh_vk_module *m = vka_shlib_module(s->lib, s->e, DMSH_VK_MOD_BASE);
    size_t nb = 0; uint32_t *w = m ? vka_shlib_spirv(s->lib, m, &nb, err, sizeof err) : NULL;
    if (w && (s->e->flags & (DMSH_VK_F_SUBGROUP | DMSH_VK_F_TEXFB))) single_lane(s->key, s->e->stage, &w, &nb);
    size_t on = 0; uint32_t *o = w ? vka_spv_fout_retype(w, nb / 4, umask, smask, &on, err, sizeof err) : NULL;
    if (o) { VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = on * 4, .pCode = o};
             VkResult r = vk.vkCreateShaderModule(V.dev, &ci, NULL, &mod); if (r) { snprintf(err, sizeof err, "vkCreateShaderModule %d", r); mod = VK_NULL_HANDLE; } else own = 1; }
    free(w); free(o);
    if (!mod) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VRED("pixel shader %016llx: outputs %#x (uint) / %#x (int) not retyped for their integer targets: %s; the base module, whose float outputs those targets do not take", (unsigned long long)s->key, umask, smask, err);
                mod = vka_shader_module(s, DMSH_VK_MOD_BASE); }
    else { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VLOG("pixel shader %016llx: float outputs %#x (uint) / %#x (int) retyped for their integer targets (their bits written as they are)", (unsigned long long)s->key, umask, smask); }
    pthread_mutex_lock(&g_mu);
    for (struct vka_vinmod *x = s->fout; x; x = x->next) if (x->u == umask && x->s == smask) {
        VkShaderModule had = x->mod; pthread_mutex_unlock(&g_mu);
        if (own && mod != had) vk.vkDestroyShaderModule(V.dev, mod, NULL);
        return had; }
    struct vka_vinmod *v = malloc(sizeof *v);
    if (v) { *v = (struct vka_vinmod){s->fout, umask, smask, mod}; s->fout = v; }
    pthread_mutex_unlock(&g_mu);
    return mod;
}
int vka_shader_fout_retyped(vka_shader *s, uint32_t umask, uint32_t smask) {
    VkShaderModule m = vka_shader_fout_module(s, umask, smask);
    return m && m != vka_shader_module(s, DMSH_VK_MOD_BASE);
}
vka_shader *vka_shader_make(const vka_shlib *L, uint64_t key) {
    pthread_mutex_lock(&g_mu);
    shent *h = sh_find(key, L);
    if (h) { vka_shader *s = h->sh; pthread_mutex_unlock(&g_mu); return s->bad ? NULL : s; }
    pthread_mutex_unlock(&g_mu);
    vka_shader *s = calloc(1, sizeof *s); s->key = key; s->lib = L;
    const dmsh_vk_entry *e = vka_shlib_find(L, key);
    const char *why = NULL; char err[256] = "";
    if (!e) why = "not in the shader library";
    else {
        s->e = e; s->b = vka_shlib_binds(L, e);
        why = refusal(L, e);
        const dmsh_vk_module *m = why ? NULL : vka_shlib_module(L, e, DMSH_VK_MOD_BASE);
        if (!why && !m) why = "no base module";
        if (m) s->mods[DMSH_VK_MOD_BASE] = spv_module(L, e, m, key, &why, err, sizeof err);
        if (!why && e->stage == DMSH_VK_STAGE_GS && VKA_ENV("DM_VK_GS_CLAIM", 1)) {   // memory writes: made once per primitive (vka_spv_gs_claim)
            char e2[256] = ""; s->gs_fx = gs_words(s, NULL, NULL, e2, sizeof e2);
            if (s->gs_fx < 0) VRED("geometry shader %016llx: memory writes not put under a claim (%s): drawn as it is, its writes repeated by the device's runs of each primitive", (unsigned long long)key, e2);
            else if (s->gs_fx) VLOG("geometry shader %016llx: %d memory writes, made by one run per primitive (a claim word per primitive at set 0 binding %d)", (unsigned long long)key, s->gs_fx, VKA_GS_CLAIM_BINDING);
        }
        if (!why && e->c1_len) {   // the c1 bytes: a uniform buffer of their own, made once
            if (vka_static(e->c1_len, V.ub_align, &s->c1)) { memcpy(s->c1.ptr, vka_shlib_c1(L, e), e->c1_len); s->c1_len = e->c1_len; }
            else why = "no static memory for its c1 bytes";
        }
    }
    if (why) {
        s->bad = 1;
        static _Atomic int n; if (atomic_fetch_add(&n, 1) < 64) VRED("shader %016llx (%s): %s; its draws are skipped", (unsigned long long)key, e ? stage_name(e->stage) : "?", why);
    }
    pthread_mutex_lock(&g_mu);
    h = sh_find(key, L);
    if (h) { if (s->mods[0]) vk.vkDestroyShaderModule(V.dev, s->mods[0], NULL); free(s); s = h->sh; }   // another thread made it meanwhile
    else sh_put(key, L, s);
    pthread_mutex_unlock(&g_mu);
    return s->bad ? NULL : s;
}

// A shader of the game's library by key, the library loaded if it is not yet (the prewarm threads, vka_pso.c); NULL = no library, not in it, or refused
vka_shader *vka_shader_by_key(uint64_t key) {
    pthread_mutex_lock(&g_mu); int ok = lib_load(); pthread_mutex_unlock(&g_mu);
    return ok ? vka_shader_make(&g_lib, key) : NULL;
}

// Every shader of a library through the device's compiler: each module made, each stage's set-0 layout made, each compute shader's pipeline built (graphics pipelines need
// a pair and state, so vertex and pixel shaders stop at the module and layout). DM_VK_SHLIB_CHECK=1 runs it when the game's library loads; the selftest runs it on DM_VK_SHLIB.
// Returns the number of failures; one log line per failure kind and a summary
int vka_shlib_check_all(const vka_shlib *L) {
    unsigned n = L->h->entry_count, made = 0, refused = 0, nolayout = 0, cs = 0, csfail = 0; uint64_t t0 = now_ns();
    for (unsigned i = 0; i < n; i++) {
        const dmsh_vk_entry *e = &L->ent[i];
        vka_shader *s = vka_shader_make(L, e->key);
        if (!s) { refused++; continue; }
        made++;
        for (int v = 1; v <= DMSH_VK_MOD_SHUFFLE_LDS; v++) if (vka_shlib_module(L, e, v) && !vka_shader_module(s, v)) nolayout++;   // the variants through the compiler too (counted as failures)
        vka_shader *sh[VKA_NSTAGE] = {0}; int st = vka_nvn_stage(e->stage); if (st < 0) { nolayout++; continue; } sh[st] = s;
        vka_layout lay;
        if (!vka_layout_of(sh, st == 5, &lay)) { nolayout++; continue; }
        if (st == 5) { uint64_t t = now_ns(); cs++; if (!vka_cs_pipeline(s, &lay)) csfail++;
                       if (cs <= 3 || cs % 25 == 0) VLOG("shader library check: compute pipeline %u (%016llx) %.0f ms; %u / %u shaders", cs, (unsigned long long)e->key, (now_ns() - t) / 1e6, i + 1, n); }
        if ((i + 1) % 2000 == 0) VLOG("shader library check: %u / %u shaders, %.1f s", i + 1, n, (now_ns() - t0) / 1e9);
    }
    VLOG("shader library check: %u shaders, %u modules made, %u refused, %u without a layout, %u compute pipelines (%u failed), %.1f s",
         n, made, refused, nolayout, cs, csfail, (now_ns() - t0) / 1e9);
    vka_pso_save(1);
    return (int)(nolayout + csfail);
}

// ---------- Programs ----------
// By serial (a new serial at every nvnProgramSetShaders; mtl_prog_release drops it): the program's stage keys resolved to shaders, each at the NVN stage its entry names
typedef struct { uint64_t serial; vka_progsh ps; } prent;
static prent *g_pr; static size_t g_prcap, g_prn;
static prent *pr_find(uint64_t s) {
    if (!g_prcap) return NULL;
    for (size_t i = hslot(s, g_prcap);; i = (i + 1) & (g_prcap - 1)) { if (!g_pr[i].serial) return NULL; if (g_pr[i].serial == s) return &g_pr[i]; }
}
static void pr_put(uint64_t s, const vka_progsh *ps) {
    prent *e = pr_find(s); if (e) { e->ps = *ps; return; }
    if (2 * (g_prn + 1) > g_prcap) {
        size_t nc = g_prcap ? 2 * g_prcap : 1024; prent *n = calloc(nc, sizeof *n);
        for (size_t i = 0; i < g_prcap; i++) if (g_pr[i].serial && g_pr[i].serial != ~0ull) { size_t j = hslot(g_pr[i].serial, nc); while (n[j].serial) j = (j + 1) & (nc - 1); n[j] = g_pr[i]; }
        free(g_pr); g_pr = n; g_prcap = nc; size_t live = 0; for (size_t i = 0; i < nc; i++) live += n[i].serial != 0; g_prn = live;
    }
    size_t j = hslot(s, g_prcap); while (g_pr[j].serial && g_pr[j].serial != ~0ull) j = (j + 1) & (g_prcap - 1);
    if (!g_pr[j].serial) g_prn++;
    g_pr[j] = (prent){s, *ps};
}
void vka_prog_put(uint64_t serial, const vka_progsh *ps) { pthread_mutex_lock(&g_mu); pr_put(serial, ps); pthread_mutex_unlock(&g_mu); }
void mtl_prog_release(prog_t *p) {
    if (!p || !p->serial) return;
    pthread_mutex_lock(&g_mu); prent *e = pr_find(p->serial); if (e) { e->serial = ~0ull; memset(&e->ps, 0, sizeof e->ps); } pthread_mutex_unlock(&g_mu);   // a tombstone
}
void mtl_prog_prefetch(const prog_t *p) { (void)p; }   // modules are made at the program's first draw (vka_prog_shaders)

// DM_NVN_SHDUMP=directory (debugging): a stage missing from the library is saved as <key>.ctl + <key>.code, the Mac's form (nvn_mtl.m; shader_split.py nvn_dumps, runtime_shaders.py --dump),
// once per key. The control section is hle_nvn.c's copy (nvnProgramSetShaders keeps one while SHDUMP is set), the code area is read in the game's memory pool at its address;
// a captured program (vka_replay.c) has no control section and is skipped
static void shdump(const prog_t *p, unsigned i, uint64_t key) {
    static int on = -1; if (on < 0) on = getenv("DM_NVN_SHDUMP") != NULL;
    if (!on || !p->sd[i].control) return;
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER; static const char *dir; static uint64_t *done; static size_t ndone, cap;
    pthread_mutex_lock(&mu);
    if (!dir) { dir = getenv("DM_NVN_SHDUMP"); if (*dir) mkdir(dir, 0777); }   // create the dump folder: a missing one made every fopen fail silently
    for (size_t j = 0; j < ndone; j++) if (done[j] == key) { pthread_mutex_unlock(&mu); return; }
    if (ndone == cap) { size_t nc = cap ? 2 * cap : 64; uint64_t *n = realloc(done, nc * sizeof *n); if (!n) { pthread_mutex_unlock(&mu); return; } done = n; cap = nc; }
    done[ndone++] = key;
    pthread_mutex_unlock(&mu);
    uint32_t ncode, nctl; nvn_ctl_lens(p->sd[i].control, &nctl, &ncode);
    if (nctl == 0x48) nctl = 0xd8;   // NVN2 (nvn_ctl_lens gives its key's 0x48 bytes): the whole 0xd8-byte section, as the split needs its stage byte (+0x56) and thread group (+0x58); hle_nvn.c's copy holds it
    int nvn_shader_code_readable(uint64_t a, uint64_t n);   // (hle_nvn.c: a shader-code pool not placed, DM_VK_PLACE_SHADER_POOLS=0)
    uint64_t code = p->sd[i].data; int okc = code && ncode && ncode < (1u << 24) && (mtl_addr(code, ncode, NULL, NULL) || nvn_shader_code_readable(code, ncode));
    char path[1024]; FILE *f; int okf = 0;
    snprintf(path, sizeof path, "%s/%016llx.ctl", dir, (unsigned long long)key); if ((f = fopen(path, "wb"))) { okf = fwrite(p->sd[i].control, 1, nctl, f) == nctl; okf &= fclose(f) == 0; }
    snprintf(path, sizeof path, "%s/%016llx.code", dir, (unsigned long long)key);
    if (okf && okc && (f = fopen(path, "wb"))) { okc = fwrite((const void *)(uintptr_t)code, 1, ncode, f) == ncode; okc &= fclose(f) == 0; } else okc = 0;
    if (okf && okc) VLOG("DM_NVN_SHDUMP: saved %s/%016llx.ctl (%#x bytes) and .code (%#x bytes)", dir, (unsigned long long)key, nctl, ncode);
    else VRED("DM_NVN_SHDUMP: shader %016llx not saved in %s (%s)", (unsigned long long)key, dir, !okf ? "the control section could not be written" : "its code area is not in a memory pool or could not be written");
}

int vka_prog_shaders(const prog_t *p, vka_progsh *out) {
    memset(out, 0, sizeof *out);
    if (!p || !p->serial) return 0;
    pthread_mutex_lock(&g_mu); prent *e = pr_find(p->serial); if (e) *out = e->ps; pthread_mutex_unlock(&g_mu);
    if (e) return 1;
    pthread_mutex_lock(&g_mu); int ok = lib_load(); pthread_mutex_unlock(&g_mu);
    if (!ok) return 0;
    for (unsigned i = 0; i < p->nstage && i < 6; i++) {   // (a program has at most 6 shaders: hle_nvn.c nvnProgramSetShaders)
        uint8_t md[8];
        if (!prog_key(p, i, md)) continue;
        uint64_t key = vka_shlib_key(md);
        const dmsh_vk_entry *en = vka_shlib_find(&g_lib, key);
        if (!en) { out->bad |= VKA_BAD_MISSING; static _Atomic int n; if (atomic_fetch_add(&n, 1) < 32) VRED("shader %016llx (program stage %u) is not in the shader library", (unsigned long long)key, i); shdump(p, i, key); continue; }
        int st = vka_nvn_stage(en->stage);
        if (st < 0) { out->bad |= VKA_BAD_MISSING; continue; }   // (vka_shlib_open refuses a stage past DMSH_VK_STAGE_MS)
        if (out->sh[st]) { static _Atomic int n; if (atomic_fetch_add(&n, 1) < 8) VRED("program serial %llu has two %s shaders (%016llx, %016llx): the second is used", (unsigned long long)p->serial, stage_name(en->stage), (unsigned long long)out->sh[st]->key, (unsigned long long)key); }
        out->sh[st] = vka_shader_make(&g_lib, key);
        if (!out->sh[st]) out->bad |= 1u << st;
    }
    vka_prog_put(p->serial, out);
    return 1;
}
#endif
