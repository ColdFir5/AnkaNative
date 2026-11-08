// test_vk_ssbo_base.py's GPU run: one compute module of that test (a storage buffer read at addr + offset through c0's address, the translator's shape) run on the first
// Vulkan 1.1 device with robustBufferAccess, over 64 offsets around a slot address, some of them negative (below the slot, as Ocarina of Time's UI text reads). The
// "guest memory" is one buffer standing for guest addresses [G, G + MEM), across a 4 GiB boundary so the low words the shader subtracts wrap; each word holds a value
// derived from its guest address, so a read is right only when it reached that exact address. Offsets -40 and -4 stay above the address rounded down to
// DMSH_VK_SSBO_ALIGN, which the descriptor already reached before the fix.
// Usage: vk_ssbo_base_run <module.spv> <window|aligned>. window: the descriptor starts where the host starts it for a DMSH_VK_F_SSBO_BASE entry (vka_bind.h vka_ssbo_back:
// as far back as the buffer goes, at most half the range) and the slot's c0 base word (dmsh_abi_vk.h DMSH_VK_C0_SSBO_BASE) holds its low word; aligned: the descriptor
// starts at the address rounded down to DMSH_VK_SSBO_ALIGN and reaches to the end of the buffer, the base word holds that address (the host before the fix, and
// DM_VK_SSBO_EXACT). Prints each case, then "below the rounded address right a/b, others right c/d"; exit 77 when there is no such device (the test is then skipped)
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../vk_android/vka_bind.h"

#define CK(x) do { VkResult r_ = (x); if (r_) { printf("%s: %d\n", #x, r_); exit(1); } } while (0)
enum { CASES = 64, MEM = 0x10000, C0_ROW = DMSH_C0_SSBO_CS / 16 };   // slot 0 of the compute stage's c0 storage table
static const uint64_t G = 0x2fffff000ull;                            // guest address of the buffer's first byte: the slot's low word wraps past 2^32 below it
static const uint64_t A = 0x2fffff000ull + 0x2028;                   // the slot's address (0x3_0000_1028): not on DMSH_VK_SSBO_ALIGN, 8 KiB into the buffer
static uint32_t word_at(uint64_t g) { return (uint32_t)(g * 0x9e3779b97f4a7c15ull >> 32) ^ (uint32_t)g; }

static PFN_vkGetInstanceProcAddr gipa; static VkInstance inst; static VkDevice dev; static VkPhysicalDevice pd; static PFN_vkGetDeviceProcAddr gdpa;
#define IP(n) PFN_##n n = (PFN_##n)gipa(inst, #n)
#define DP(n) PFN_##n n = (PFN_##n)gdpa(dev, #n)
typedef struct { VkBuffer buf; VkDeviceMemory mem; void *ptr; VkDeviceSize size; } buf_t;

static uint32_t mem_type(uint32_t bits) {
    IP(vkGetPhysicalDeviceMemoryProperties); VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    printf("no host-visible coherent memory\n"); exit(1);
}
static buf_t make(VkDeviceSize n, VkBufferUsageFlags use) {
    DP(vkCreateBuffer); DP(vkGetBufferMemoryRequirements); DP(vkAllocateMemory); DP(vkBindBufferMemory); DP(vkMapMemory);
    buf_t b = {.size = n};
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = use};
    CK(vkCreateBuffer(dev, &bi, NULL, &b.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = mem_type(mr.memoryTypeBits)};
    CK(vkAllocateMemory(dev, &ai, NULL, &b.mem)); CK(vkBindBufferMemory(dev, b.buf, b.mem, 0)); CK(vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.ptr));
    memset(b.ptr, 0, n);
    return b;
}

int main(int argc, char **argv) {
    if (argc < 3 || (strcmp(argv[2], "window") && strcmp(argv[2], "aligned"))) { printf("usage: vk_ssbo_base_run module.spv window|aligned\n"); return 2; }
    int window = !strcmp(argv[2], "window");
    void *lib = dlopen("libvulkan.so.1", RTLD_NOW); if (!lib) { printf("no Vulkan loader\n"); return 77; }
    gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    PFN_vkCreateInstance ci_ = (PFN_vkCreateInstance)gipa(NULL, "vkCreateInstance");
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai};
    if (ci_(&ici, NULL, &inst)) { printf("no Vulkan 1.1 instance\n"); return 77; }
    IP(vkEnumeratePhysicalDevices); IP(vkGetPhysicalDeviceFeatures); IP(vkCreateDevice); IP(vkGetPhysicalDeviceProperties);
    gdpa = (PFN_vkGetDeviceProcAddr)gipa(inst, "vkGetDeviceProcAddr");
    uint32_t n = 8; VkPhysicalDevice pds[8]; vkEnumeratePhysicalDevices(inst, &n, pds);
    VkPhysicalDeviceProperties pp = {0};
    for (uint32_t i = 0; i < n && !pd; i++) {
        VkPhysicalDeviceFeatures f; vkGetPhysicalDeviceFeatures(pds[i], &f); vkGetPhysicalDeviceProperties(pds[i], &pp);
        if (pp.apiVersion >= VK_API_VERSION_1_1 && f.robustBufferAccess && DMSH_VK_SSBO_ALIGN % pp.limits.minStorageBufferOffsetAlignment == 0) pd = pds[i];
    }
    if (!pd) { printf("no Vulkan 1.1 device with robustBufferAccess and a storage buffer alignment dividing %d\n", DMSH_VK_SSBO_ALIGN); return 77; }
    VkPhysicalDeviceFeatures en = {.robustBufferAccess = 1};   // as the replay's device: a read outside the descriptor gives zeros (or a value inside it), never the address it meant
    float prio = 1; VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci, .pEnabledFeatures = &en};
    CK(vkCreateDevice(pd, &dci, NULL, &dev));

    buf_t mem = make(MEM, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT), c0 = make(DMSH_DRV_SIZE, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT), cases = make(CASES * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
          out = make(CASES * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    for (uint32_t w = 0; w < MEM / 4; w++) ((uint32_t *)mem.ptr)[w] = word_at(G + 4ull * w);
    // the slot as the host fills it (vka_drv_fill: address, size; vka_c0_bases: the base word) and its descriptor (vka_draw.c ssbo_desc)
    uint64_t al = A & ~(uint64_t)(DMSH_VK_SSBO_ALIGN - 1), off = al - G;
    uint64_t back = window ? vka_ssbo_back(off, pp.limits.minStorageBufferOffsetAlignment, pp.limits.maxStorageBufferRange) : 0;
    uint32_t row[4] = {(uint32_t)A, (uint32_t)(A >> 32), 0x700, (uint32_t)(al - back)};
    memcpy((uint8_t *)c0.ptr + 16 * C0_ROW, row, 16);
    VkDeviceSize dofs = off - back, drange = MEM - dofs;
    int32_t *cs = cases.ptr;
    static const int32_t fixed[] = {-0x2028, -28 * 224, -224, -152, -64, -40, -4, 0, 4, 152, 0x6fc, 0x700, 0x700 + 3 * 224, MEM - 0x2028 - 4};
    int nf = (int)(sizeof fixed / sizeof *fixed);
    for (int i = 0; i < CASES; i++) cs[i] = i < nf ? fixed[i] : -0x2028 + (i - nf) * 0x400;

    DP(vkCreateShaderModule); DP(vkCreateDescriptorSetLayout); DP(vkCreatePipelineLayout); DP(vkCreateComputePipelines); DP(vkCreateDescriptorPool); DP(vkAllocateDescriptorSets);
    DP(vkUpdateDescriptorSets); DP(vkCreateCommandPool); DP(vkAllocateCommandBuffers); DP(vkBeginCommandBuffer); DP(vkCmdBindPipeline); DP(vkCmdBindDescriptorSets);
    DP(vkCmdDispatch); DP(vkEndCommandBuffer); DP(vkGetDeviceQueue); DP(vkQueueSubmit); DP(vkQueueWaitIdle);
    FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET); uint32_t *code = malloc(sz); if (fread(code, 1, sz, f) != (size_t)sz) return 1; fclose(f);
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = code}; VkShaderModule sm; CK(vkCreateShaderModule(dev, &smi, NULL, &sm));
    VkDescriptorSetLayoutBinding lb[4];
    for (int i = 0; i < 4; i++) lb[i] = (VkDescriptorSetLayoutBinding){i, i ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 4, .pBindings = lb}; VkDescriptorSetLayout dl; CK(vkCreateDescriptorSetLayout(dev, &dli, NULL, &dl));
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dl}; VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
    VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = pl, .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main"}};
    VkPipeline pipe; CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = ps};
    VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dl}; VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev, &dai, &ds));
    VkDescriptorBufferInfo dbi[4] = {{c0.buf, 0, DMSH_DRV_SIZE}, {mem.buf, dofs, drange}, {cases.buf, 0, VK_WHOLE_SIZE}, {out.buf, 0, VK_WHOLE_SIZE}}; VkWriteDescriptorSet wr[4];
    for (int i = 0; i < 4; i++) wr[i] = (VkWriteDescriptorSet){VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = i, .descriptorCount = 1, .descriptorType = lb[i].descriptorType, .pBufferInfo = &dbi[i]};
    vkUpdateDescriptorSets(dev, 4, wr, 0, NULL);
    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0}; VkCommandPool cp; CK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1}; VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(cb, &bi));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe); vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cb, 1, 1, 1); CK(vkEndCommandBuffer(cb));
    VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q); VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); CK(vkQueueWaitIdle(q));

    int neg = 0, negok = 0, pos = 0, posok = 0;
    printf("descriptor at guest %#llx + %#llx, base word %08x, slot %#llx\n", (unsigned long long)(G + dofs), (unsigned long long)drange, row[3], (unsigned long long)A);
    for (int i = 0; i < CASES; i++) {
        uint32_t want = word_at(A + (int64_t)cs[i]), got = ((uint32_t *)out.ptr)[i]; int ok = got == want;
        printf("case %2d offset %7d -> %08x (want %08x)%s\n", i, cs[i], got, want, ok ? "" : " WRONG");
        if (A + (int64_t)cs[i] < al) { neg++; negok += ok; } else { pos++; posok += ok; }
    }
    printf("below the rounded address right %d/%d, others right %d/%d\n", negok, neg, posok, pos);
    return 0;
}
