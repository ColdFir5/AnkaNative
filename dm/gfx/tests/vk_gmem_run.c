// test_vk_gmem_flatten.py's GPU run: one compute module of that test (raw global memory helpers: a pool table, a write log, accesses through physical pointers) run on the
// first Vulkan device with buffer device addresses and 64-bit integers, over 64 cases; prints what a run produced, so two modules can be compared line for line:
// each case's loaded value, a digest of each pool's memory, the write log's pages (sorted: the slot a page lands in depends on the order the threads ran) and its overflow
// word, and whether the sink (dmsh_abi_vk.h DMSH_VK_GMEM_SINK_WORD) holds only values of stores to addresses in no pool.
// A module that reads other cases first (test_vk_gmem_flatten.py warm: 4 loads before its own case, gmem_cache's hits and misses) writes them past the 64 results:
// their digest is printed too. "full" as the second argument fills the table to its last entry (4095): entries 4 to 4095 are 4 KiB ranges from 0x1ff0005000 up, the
// last one at 0x2000000000 (where the "above every pool" cases lie), all over the first pool's memory.
// After the table the page index (vk_android_spv.py gmem_pidx) built by the host's code (dm/gfx/vk_android/vka_pidx.h, as vka_draw.c gmem_pidx): at word 32768 the
// window's first page and its page count (the entries' pages rounded out to the environment's PIDX_GRAN pages, default the host's VKA_PIDX_GRAN; 1: from the lowest
// entry start to the highest end exactly), 2 words of 0, then an 8-byte record a page and one past the window: the page's device address with bit 0 set for a page in
// an entry, else 0. Modules without gmem_pidx read only the table.
// Usage: vk_gmem_run <module.spv> [full]; exit 77 when there is no such device (the test is then skipped)
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../vk_android/vka_pidx.h"

#define CK(x) do { VkResult r_ = (x); if (r_) { printf("%s: %d\n", #x, r_); exit(1); } } while (0)
enum { PIDX_WORD = 4096 * 8, PIDX_SHIFT = 12, MAP_ENTRIES = 4096, MAP_ENTRY = 32, LOG_WORDS = 1 + 1024, CASES = 64, WARM = 4, POOLS = 3, POOL_BYTES = 3 * 4096, SINK_WORD = 6, SINK_BYTES = 256 };
static const uint64_t POOL_AT[POOLS] = {0x1000000000ull, 0x1000100000ull, 0x10ffffe000ull};   // guest addresses (the last one ends at a 4 GiB boundary + 0x1000: carry into the high word)

static PFN_vkGetInstanceProcAddr gipa; static VkInstance inst; static VkDevice dev; static VkPhysicalDevice pd;
#define IP(n) PFN_##n n = (PFN_##n)gipa(inst, #n)
#define DP(n) PFN_##n n = (PFN_##n)gdpa(dev, #n)
static PFN_vkGetDeviceProcAddr gdpa;
typedef struct { VkBuffer buf; VkDeviceMemory mem; void *ptr; uint64_t addr; VkDeviceSize size; } buf_t;

static uint32_t mem_type(uint32_t bits) {
    IP(vkGetPhysicalDeviceMemoryProperties); VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    printf("no host-visible coherent memory\n"); exit(1);
}
static buf_t make(VkDeviceSize n) {
    DP(vkCreateBuffer); DP(vkGetBufferMemoryRequirements); DP(vkAllocateMemory); DP(vkBindBufferMemory); DP(vkMapMemory); DP(vkGetBufferDeviceAddress);
    buf_t b = {.size = n};
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
    CK(vkCreateBuffer(dev, &bi, NULL, &b.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    VkMemoryAllocateFlagsInfo af = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &af, .allocationSize = mr.size, .memoryTypeIndex = mem_type(mr.memoryTypeBits)};
    CK(vkAllocateMemory(dev, &ai, NULL, &b.mem)); CK(vkBindBufferMemory(dev, b.buf, b.mem, 0)); CK(vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.ptr));
    VkBufferDeviceAddressInfo da = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b.buf}; b.addr = vkGetBufferDeviceAddress(dev, &da);
    memset(b.ptr, 0, n);
    return b;
}
static int cmp_u32(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return x < y ? -1 : x > y; }

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: vk_gmem_run module.spv\n"); return 2; }
    void *lib = dlopen("libvulkan.so.1", RTLD_NOW); if (!lib) { printf("no Vulkan loader\n"); return 77; }
    gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    PFN_vkCreateInstance ci_ = (PFN_vkCreateInstance)gipa(NULL, "vkCreateInstance");
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai};
    if (ci_(&ici, NULL, &inst)) { printf("no Vulkan 1.3 instance\n"); return 77; }
    IP(vkEnumeratePhysicalDevices); IP(vkGetPhysicalDeviceFeatures2); IP(vkCreateDevice); IP(vkGetPhysicalDeviceProperties);
    gdpa = (PFN_vkGetDeviceProcAddr)gipa(inst, "vkGetDeviceProcAddr");
    uint32_t n = 8; VkPhysicalDevice pds[8]; vkEnumeratePhysicalDevices(inst, &n, pds);
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    for (uint32_t i = 0; i < n && !pd; i++) {
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pds[i], &pp);
        if (pp.apiVersion < VK_API_VERSION_1_3) continue;
        f12 = (VkPhysicalDeviceVulkan12Features){VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; vkGetPhysicalDeviceFeatures2(pds[i], &f2);
        if (f12.bufferDeviceAddress && f2.features.shaderInt64) pd = pds[i];
    }
    if (!pd) { printf("no device with buffer device addresses and 64-bit integers\n"); return 77; }
    VkPhysicalDeviceVulkan12Features e12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .bufferDeviceAddress = 1};
    VkPhysicalDeviceFeatures2 e2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &e12, .features = {.shaderInt64 = 1}};
    float prio = 1; VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &e2, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
    CK(vkCreateDevice(pd, &dci, NULL, &dev));

    int full = argc > 2 && !strcmp(argv[2], "full");
    uint64_t gran = getenv("PIDX_GRAN") ? strtoull(getenv("PIDX_GRAN"), NULL, 0) : VKA_PIDX_GRAN;
    uint64_t win_lo = POOL_AT[0] >> PIDX_SHIFT & ~(gran - 1), win_hi = (((full ? 0x2000001000ull : POOL_AT[POOLS - 1] + POOL_BYTES) >> PIDX_SHIFT) + gran - 1) & ~(gran - 1);   // the window's pages
    if (PIDX_WORD != DMSH_VK_GMEM_PIDX_WORD || PIDX_SHIFT != DMSH_VK_GMEM_PIDX_SHIFT || !gran || (gran & (gran - 1))) { printf("page index layout or PIDX_GRAN\n"); return 1; }
    buf_t map = make((VkDeviceSize)MAP_ENTRIES * MAP_ENTRY + vka_pidx_bytes(win_hi - win_lo)), log = make(LOG_WORDS * 4), cases = make(CASES * 16), out = make(CASES * 4 * (1 + WARM)), sink = make(SINK_BYTES), pool[POOLS];
    for (int p = 0; p < POOLS; p++) { pool[p] = make(POOL_BYTES); for (uint32_t w = 0; w < POOL_BYTES / 4; w++) ((uint32_t *)pool[p].ptr)[w] = 0x9e3779b1u * (w + 1000u * p) ^ 0x5bd1e995u; }
    // the table as the host builds it (vka_draw.c gmem_map): entry 0 empty with the sink in its padding, the pools by address, unused entries at ~0
    memset(map.ptr, 0xff, (size_t)MAP_ENTRIES * MAP_ENTRY); memset(map.ptr, 0, MAP_ENTRY);
    memcpy((uint32_t *)map.ptr + SINK_WORD, &sink.addr, 8);
    for (int p = 0; p < POOLS; p++) { uint64_t e[4] = {POOL_AT[p], POOL_AT[p] + POOL_BYTES, pool[p].addr - POOL_AT[p], 0}; memcpy((uint8_t *)map.ptr + (size_t)(p + 1) * MAP_ENTRY, e, sizeof e); }
    for (int k = POOLS + 1; full && k < MAP_ENTRIES; k++) {
        uint64_t at = 0x2000000000ull - (uint64_t)(MAP_ENTRIES - 1 - k) * 4096, e[4] = {at, at + 4096, pool[0].addr - at, 0};
        memcpy((uint8_t *)map.ptr + (size_t)k * MAP_ENTRY, e, sizeof e);
    }
    {   // the page index from the table's entries, by the host's code (vka_draw.c gmem_pidx: vka_pidx_take, vka_pidx_update)
        static uint64_t lo[MAP_ENTRIES], hi[MAP_ENTRIES], bda[MAP_ENTRIES], dl[MAP_ENTRIES], slo[MAP_ENTRIES], shi[MAP_ENTRIES], sdl[MAP_ENTRIES]; int k = 0;
        for (int i = 1; i < MAP_ENTRIES; i++) {
            const uint64_t *e = (const uint64_t *)((const uint8_t *)map.ptr + (size_t)i * MAP_ENTRY);
            if (e[0] == ~0ull) break;
            lo[k] = e[0]; hi[k] = e[1]; bda[k] = e[0] + e[2]; k++;
        }
        vka_pidx s = {.max = MAP_ENTRIES, .lo = slo, .hi = shi, .dl = sdl}; vka_pidx_stats st = {0};
        int n = vka_pidx_take(lo, hi, bda, dl, k, gran, win_hi - win_lo, &st);
        if (n != k) { printf("entries left out of the page index: %d (off the page %d, past 2^44 %d, past the window %d)\n", k - n, st.unaligned, st.high, st.over); return 1; }
        if (vka_pidx_update((uint8_t *)map.ptr + PIDX_WORD * 4, win_hi - win_lo + 1, &s, lo, hi, dl, n, gran, &st) != 2 || s.base != win_lo || s.pages != win_hi - win_lo) { printf("page index not built\n"); return 1; }
    }
    // cases: (address, value, kind 0 load / 1 store)
    uint32_t (*c)[4] = cases.ptr; uint32_t outside_values[CASES]; int no = 0;
    for (int i = 0; i < CASES; i++) {
        int p = i % POOLS; uint64_t a; int kind = (i / 3) % 2;
        switch (i % 9) {
            case 0: a = POOL_AT[p] + 4u * (uint64_t)(i * 37 % (POOL_BYTES / 4)); break;   // inside
            case 1: a = POOL_AT[p] + POOL_BYTES - 4; break;                                  // the last word
            case 2: a = POOL_AT[p] + POOL_BYTES; break;                                      // the end (exclusive): outside unless the next pool starts there
            case 3: a = POOL_AT[0] - 4096; break;                                            // below every pool
            case 4: a = 0x2000000000ull + 4u * i; break;                                     // above every pool
            case 5: a = (POOL_AT[0] + POOL_BYTES + POOL_AT[1]) / 2 & ~3ull; break;          // between two pools
            case 6: a = POOL_AT[p] + 4096 * (uint64_t)(i % 3) + 4u * i; break;               // the pool's three pages
            case 7: a = POOL_AT[2] + 0x1ffc - 4u * (i % 5); break;                           // across the 4 GiB boundary of the third pool
            default: a = POOL_AT[p] + (1ull << 44) + 4u * i; break;                          // a pool's page + 2^44: outside (the page index's high word clamp)
        }
        c[i][0] = (uint32_t)a; c[i][1] = (uint32_t)(a >> 32); c[i][2] = 0xc0de0000u + i; c[i][3] = kind;
        int in = 0; for (int q = 0; q < POOLS; q++) in |= a >= POOL_AT[q] && a < POOL_AT[q] + POOL_BYTES;
        if (full && a >= 0x2000000000ull - (uint64_t)(MAP_ENTRIES - 1 - POOLS - 1) * 4096 && a < 0x2000001000ull) in = 1;
        if (kind == 1 && !in) outside_values[no++] = c[i][2];
    }

    DP(vkCreateShaderModule); DP(vkCreateDescriptorSetLayout); DP(vkCreatePipelineLayout); DP(vkCreateComputePipelines); DP(vkCreateDescriptorPool); DP(vkAllocateDescriptorSets);
    DP(vkUpdateDescriptorSets); DP(vkCreateCommandPool); DP(vkAllocateCommandBuffers); DP(vkBeginCommandBuffer); DP(vkCmdBindPipeline); DP(vkCmdBindDescriptorSets);
    DP(vkCmdDispatch); DP(vkEndCommandBuffer); DP(vkGetDeviceQueue); DP(vkQueueSubmit); DP(vkQueueWaitIdle);
    FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET); uint32_t *code = malloc(sz); if (fread(code, 1, sz, f) != (size_t)sz) return 1; fclose(f);
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = code}; VkShaderModule sm; CK(vkCreateShaderModule(dev, &smi, NULL, &sm));
    VkDescriptorSetLayoutBinding lb[4]; for (int i = 0; i < 4; i++) lb[i] = (VkDescriptorSetLayoutBinding){i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 4, .pBindings = lb}; VkDescriptorSetLayout dl; CK(vkCreateDescriptorSetLayout(dev, &dli, NULL, &dl));
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dl}; VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
    VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = pl, .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main"}};
    VkPipeline pipe; CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}; VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &ps};
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dl}; VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev, &dai, &ds));
    buf_t *bs[4] = {&map, &log, &cases, &out}; VkDescriptorBufferInfo dbi[4]; VkWriteDescriptorSet wr[4];
    for (int i = 0; i < 4; i++) { dbi[i] = (VkDescriptorBufferInfo){bs[i]->buf, 0, VK_WHOLE_SIZE}; wr[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = i, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi[i]}; }
    vkUpdateDescriptorSets(dev, 4, wr, 0, NULL);
    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0}; VkCommandPool cp; CK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1}; VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(cb, &bi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe); vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cb, 1, 1, 1); CK(vkEndCommandBuffer(cb));
    VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q); VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); CK(vkQueueWaitIdle(q));

    for (int i = 0; i < CASES; i++) printf("case %2d %s %08x%08x -> %08x\n", i, c[i][3] ? "store" : "load ", c[i][1], c[i][0], ((uint32_t *)out.ptr)[i]);
    { uint64_t h = 1469598103934665603ull; const uint32_t *o = out.ptr;
      for (int k = CASES; k < CASES * (1 + WARM); k++) h = (h ^ o[k]) * 1099511628211ull;
      printf("warm-up loads digest %016llx\n", (unsigned long long)h); }
    for (int p = 0; p < POOLS; p++) {
        uint64_t h = 1469598103934665603ull; const uint8_t *b = pool[p].ptr;
        for (int k = 0; k < POOL_BYTES; k++) h = (h ^ b[k]) * 1099511628211ull;
        printf("pool %d digest %016llx\n", p, (unsigned long long)h);
    }
    uint32_t pages[LOG_WORDS]; int np = 0; const uint32_t *lw = log.ptr;
    for (int k = 1; k < LOG_WORDS; k++) if (lw[k]) pages[np++] = lw[k];
    qsort(pages, np, 4, cmp_u32);
    printf("log overflow %u, %d pages:", lw[0], np); for (int k = 0; k < np; k++) printf(" %x", pages[k]); printf("\n");
    int ok = 1; const uint32_t *sw = sink.ptr;
    for (int k = 0; k < SINK_BYTES / 4; k++) if (sw[k]) { int found = 0; for (int j = 0; j < no; j++) found |= sw[k] == outside_values[j]; ok &= found; }
    printf(ok ? "sink untouched by in-pool accesses\n" : "SINK HOLDS A VALUE OF AN IN-POOL ACCESS\n");
    return 0;
}
