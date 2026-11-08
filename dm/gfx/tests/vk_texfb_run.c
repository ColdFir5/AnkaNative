// test_vk_texfb.py's GPU run: a compute module of that test with texture-use feedback (vk_android_spv.py tex_feedback) on the first Vulkan 1.3 device with non-uniform
// sampled image indexing. 8 workgroups of 64 invocations each fetch texel (0, 0) of texture ID ids[g] of a 64-entry pool (set 1 binding 0; ~0 = no fetch), every entry a
// 1x1 image of its own colour; the bitmap (set 1 binding 3) starts with PRESET's bits. Prints the bitmap's two words and whether every fetch returned its entry's colour.
// Usage: vk_texfb_run <module.spv> <0|1: DMSH_VK_SPEC_TEXFB>; exit 77 when there is no such device (the test is then skipped)
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do { VkResult r_ = (x); if (r_) { printf("%s: %d\n", #x, r_); exit(1); } } while (0)
enum { GROUPS = 8, LOCAL = 64, N = GROUPS * LOCAL, POOL = 64, FB_WORDS = 32768, SPEC_ID = 2 };
static const uint32_t PRESET[2] = {1u << 9 | 1u << 30, 1u << (40 - 32)};   // bits already set: IDs 9, 30, 40 (40 is also fetched: test before set)

static PFN_vkGetInstanceProcAddr gipa; static VkInstance inst; static VkDevice dev; static VkPhysicalDevice pd; static PFN_vkGetDeviceProcAddr gdpa;
#define IP(n) PFN_##n n = (PFN_##n)gipa(inst, #n)
#define DP(n) PFN_##n n = (PFN_##n)gdpa(dev, #n)
typedef struct { VkBuffer buf; VkDeviceMemory mem; void *ptr; } buf_t;

static uint32_t mem_type(uint32_t bits, VkMemoryPropertyFlags want) {
    IP(vkGetPhysicalDeviceMemoryProperties); VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    printf("no memory type %x\n", want); exit(1);
}
static buf_t make(VkDeviceSize n, VkBufferUsageFlags usage) {
    DP(vkCreateBuffer); DP(vkGetBufferMemoryRequirements); DP(vkAllocateMemory); DP(vkBindBufferMemory); DP(vkMapMemory);
    buf_t b = {0};
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n, .usage = usage};
    CK(vkCreateBuffer(dev, &bi, NULL, &b.buf));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b.buf, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                               .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    CK(vkAllocateMemory(dev, &ai, NULL, &b.mem)); CK(vkBindBufferMemory(dev, b.buf, b.mem, 0)); CK(vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.ptr));
    memset(b.ptr, 0, n);
    return b;
}
static uint32_t colour(uint32_t id) { return 0xff000000u | (id * 0x2f1u + 0x10203u) & 0xffffffu; }

int main(int argc, char **argv) {
    if (argc < 3) { printf("usage: vk_texfb_run module.spv 0|1\n"); return 2; }
    void *lib = dlopen("libvulkan.so.1", RTLD_NOW); if (!lib) { printf("no Vulkan loader\n"); return 77; }
    gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
    PFN_vkCreateInstance ci_ = (PFN_vkCreateInstance)gipa(NULL, "vkCreateInstance");
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai};
    if (ci_(&ici, NULL, &inst)) { printf("no Vulkan 1.3 instance\n"); return 77; }
    IP(vkEnumeratePhysicalDevices); IP(vkGetPhysicalDeviceFeatures2); IP(vkCreateDevice); IP(vkGetPhysicalDeviceProperties); IP(vkGetPhysicalDeviceProperties2);
    gdpa = (PFN_vkGetDeviceProcAddr)gipa(inst, "vkGetDeviceProcAddr");
    uint32_t n = 8; VkPhysicalDevice pds[8]; vkEnumeratePhysicalDevices(inst, &n, pds);
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    VkPhysicalDeviceSubgroupProperties sg = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    for (uint32_t i = 0; i < n && !pd; i++) {
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pds[i], &pp);
        if (pp.apiVersion < VK_API_VERSION_1_3) continue;
        f12 = (VkPhysicalDeviceVulkan12Features){VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; vkGetPhysicalDeviceFeatures2(pds[i], &f2);
        VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sg}; vkGetPhysicalDeviceProperties2(pds[i], &p2);
        int ops = (sg.supportedOperations & (VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT)) == (VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT);
        if (f12.shaderSampledImageArrayNonUniformIndexing && ops && (sg.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT)) pd = pds[i];
    }
    if (!pd) { printf("no device with non-uniform sampled image indexing and compute subgroup ballots\n"); return 77; }
    printf("subgroup size %u\n", sg.subgroupSize);
    VkPhysicalDeviceVulkan12Features e12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .shaderSampledImageArrayNonUniformIndexing = 1, .runtimeDescriptorArray = f12.runtimeDescriptorArray};
    VkPhysicalDeviceFeatures2 e2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &e12};
    float prio = 1; VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &e2, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
    CK(vkCreateDevice(pd, &dci, NULL, &dev));
    DP(vkCreateShaderModule); DP(vkCreateDescriptorSetLayout); DP(vkCreatePipelineLayout); DP(vkCreateComputePipelines); DP(vkCreateDescriptorPool); DP(vkAllocateDescriptorSets);
    DP(vkUpdateDescriptorSets); DP(vkCreateCommandPool); DP(vkAllocateCommandBuffers); DP(vkBeginCommandBuffer); DP(vkCmdBindPipeline); DP(vkCmdBindDescriptorSets);
    DP(vkCmdDispatch); DP(vkEndCommandBuffer); DP(vkGetDeviceQueue); DP(vkQueueSubmit); DP(vkQueueWaitIdle); DP(vkCreateImage); DP(vkGetImageMemoryRequirements);
    DP(vkAllocateMemory); DP(vkBindImageMemory); DP(vkCreateImageView); DP(vkCreateSampler); DP(vkCmdPipelineBarrier); DP(vkCmdCopyBufferToImage);

    // the cases: per workgroup (a wave of 64 on a 64-lane device, two of 32 elsewhere) one pattern
    buf_t ids = make(N * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT), out = make(N * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT), fb = make(FB_WORDS * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    buf_t stage = make(POOL * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    uint32_t *id = ids.ptr, want[2] = {PRESET[0], PRESET[1]};
    for (int g = 0; g < N; g++) {
        int w = g / LOCAL, l = g % LOCAL; uint32_t v;
        switch (w) {
            case 0: v = 5; break;                                    // one ID in the whole wave
            case 1: v = (uint32_t)(l * 37 % 32) * 2 + 1; break;      // 32 distinct (odd) IDs, each in two lanes
            case 2: v = l % 3 == 0 ? 7 : l % 3 == 1 ? 33 : 62; break; // three
            case 3: v = l & 1 ? ~0u : 12; break;                     // half the lanes fetch nothing
            case 4: v = l == 63 ? 40 : ~0u; break;                   // one lane, a preset bit
            case 5: v = ~0u; break;                                  // none
            case 6: v = l < 32 ? 20 : 51; break;                     // two halves
            default: v = (uint32_t)(63 - l / 8); break;              // eight runs
        }
        id[g] = v;
        if (v != ~0u) want[v >> 5] |= 1u << (v & 31);
    }
    memcpy(fb.ptr, PRESET, sizeof PRESET);
    for (uint32_t i = 0; i < POOL; i++) ((uint32_t *)stage.ptr)[i] = colour(i);
    VkImage img[POOL]; VkImageView view[POOL];
    for (uint32_t i = 0; i < POOL; i++) {
        VkImageCreateInfo ii = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {1, 1, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        CK(vkCreateImage(dev, &ii, NULL, &img[i]));
        VkMemoryRequirements mr; vkGetImageMemoryRequirements(dev, img[i], &mr); VkDeviceMemory mem;
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = mem_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        CK(vkAllocateMemory(dev, &mai, NULL, &mem)); CK(vkBindImageMemory(dev, img[i], mem, 0));
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img[i], .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        CK(vkCreateImageView(dev, &vi, NULL, &view[i]));
    }
    VkSamplerCreateInfo sci = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; VkSampler smp; CK(vkCreateSampler(dev, &sci, NULL, &smp));

    FILE *f = fopen(argv[1], "rb"); if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET); uint32_t *code = malloc(sz); if (fread(code, 1, sz, f) != (size_t)sz) return 1; fclose(f);
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sz, .pCode = code}; VkShaderModule sm; CK(vkCreateShaderModule(dev, &smi, NULL, &sm));
    VkDescriptorSetLayoutBinding l0[2] = {{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutBinding l1[3] = {{0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, POOL, VK_SHADER_STAGE_COMPUTE_BIT, NULL}, {1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
                                          {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayout dl[2];
    VkDescriptorSetLayoutCreateInfo dli0 = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = l0}; CK(vkCreateDescriptorSetLayout(dev, &dli0, NULL, &dl[0]));
    VkDescriptorSetLayoutCreateInfo dli1 = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = l1}; CK(vkCreateDescriptorSetLayout(dev, &dli1, NULL, &dl[1]));
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 2, .pSetLayouts = dl}; VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));
    VkBool32 on = atoi(argv[2]) ? VK_TRUE : VK_FALSE;
    VkSpecializationMapEntry me = {SPEC_ID, 0, sizeof(VkBool32)}; VkSpecializationInfo spi = {1, &me, sizeof on, &on};
    VkComputePipelineCreateInfo cpi = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = pl,
                                       .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main", .pSpecializationInfo = &spi}};
    VkPipeline pipe; CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));
    VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, POOL}, {VK_DESCRIPTOR_TYPE_SAMPLER, 1}};
    VkDescriptorPoolCreateInfo dpi = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 2, .poolSizeCount = 3, .pPoolSizes = ps}; VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpi, NULL, &dp));
    VkDescriptorSetAllocateInfo dai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 2, .pSetLayouts = dl}; VkDescriptorSet ds[2]; CK(vkAllocateDescriptorSets(dev, &dai, ds));
    VkDescriptorBufferInfo b0 = {ids.buf, 0, VK_WHOLE_SIZE}, b1 = {out.buf, 0, VK_WHOLE_SIZE}, b3 = {fb.buf, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo ii[POOL], si_ = {.sampler = smp};
    for (int i = 0; i < POOL; i++) ii[i] = (VkDescriptorImageInfo){.imageView = view[i], .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet wr[5] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds[0], .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &b0},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds[0], .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &b1},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds[1], .dstBinding = 0, .descriptorCount = POOL, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = ii},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds[1], .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .pImageInfo = &si_},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds[1], .dstBinding = 3, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &b3}};
    vkUpdateDescriptorSets(dev, 5, wr, 0, NULL);
    VkCommandPoolCreateInfo cpci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = 0}; VkCommandPool cp; CK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1}; VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &cai, &cb));
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; CK(vkBeginCommandBuffer(cb, &bi));
    for (uint32_t i = 0; i < POOL; i++) {
        VkImageMemoryBarrier to = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img[i], .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &to);
        VkBufferImageCopy bic = {.bufferOffset = 4 * i, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {1, 1, 1}};
        vkCmdCopyBufferToImage(cb, stage.buf, img[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
        VkImageMemoryBarrier rd = to; rd.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; rd.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        rd.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; rd.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &rd);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe); vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 2, ds, 0, NULL);
    vkCmdDispatch(cb, GROUPS, 1, 1); CK(vkEndCommandBuffer(cb));
    VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q); VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); CK(vkQueueWaitIdle(q));

    const uint32_t *w = fb.ptr, *o = out.ptr; int bad = 0, rest = 0;
    for (int g = 0; g < N; g++) bad += o[g] != (id[g] == ~0u ? 0u : colour(id[g]));
    for (int k = 2; k < FB_WORDS; k++) rest |= w[k] != 0;
    printf("bitmap %08x %08x, words past them %s\n", w[0], w[1], rest ? "WRITTEN" : "clear");
    printf("expected on %08x %08x, off %08x %08x\n", want[0], want[1], PRESET[0], PRESET[1]);
    printf("fetches %s (%d wrong)\n", bad ? "WRONG" : "right", bad);
    return 0;
}
