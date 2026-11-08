// Image memory: small images share 64 MiB blocks instead of a VkDeviceMemory each (android/PORTING.md, NVN → Vulkan replay, Textures).
// Every image had a dedicated allocation of its own. Beside the placed pool segments (vka_mem.c) that runs into the device's maxMemoryAllocationCount: Mario Kart World's
// image allocations all failed from about 2,300 images on (beside 1,732 segments, "-2; 262 MiB in images", small images too), and Donkey Kong Bananza keeps about 7,200.
// The count problem is thousands of small images, so only an image under VKA_IMEM_MAX (4 MiB) whose memory requirements neither require nor prefer a dedicated allocation
// (VkMemoryDedicatedRequirements) takes space in a block of its memory type: exactly the 256-byte units its size needs, at its alignment, first fit (vka_extents.h) in the
// fullest block it fits (place). Larger
// images, the ones the driver wants dedicated, every image with DM_VK_IMGMEM=0, and an image whose block could not be allocated keep a dedicated allocation as before.
// Only optimal-tiling images live in the blocks, so bufferImageGranularity (which separates linear from optimal resources) does not apply between them.
// Memory overhead: block space no image uses is committed memory too (the "free in blocks" of the stats line), which matters to a game near the low-memory killer (Donkey Kong
// Bananza: 8.07 GB RSS, killed at about 9.2 GB); an earlier version rounded every image up to a power of two (a third more for mipped textures, up to half for 8-16 MiB ones).
// Lifetime: as before, an image's memory is given back in img_destroy, which runs once the GPU is done with the image (vka_after), after vkDestroyImage. A block with no image
// left is freed, except one empty block a memory type keeps while its other blocks have under KEEP_FREE free or no 4 MiB free run (trim). The 64 MiB vkAllocateMemory runs outside the lock (other
// threads' allocations and frees go on).
// Fullest block first: first fit from the newest block spread long-lived textures over every block, so none ever emptied (Ocarina of Time: 6-7 blocks, 63-76 MiB free in
// them, "0 freed", until the low-memory killer). New images now fill the fullest block they fit, so the emptiest block gets only what fits nowhere else and empties as its
// images go; an empty block is given back once the others have room
// Large never-written GPU-content images instead alias one zeroed allocation per create info (vka_imem_bind_zero; vka_tex.c "Zero-shared images"), and large GPU-content
// images over one guest storage start one allocation per key (vka_imem_bind_alias; vka_tex.c "Guest-aliased targets")
#if defined(__ANDROID__)
#include "vka.h"
#include "vka_extents.h"
#include "vka_zshare.h"
#include "vka_rtalias.h"
#include <stdlib.h>

#define UNIT 256ull
#define BLOCK (64ull << 20)
#define VKA_IMEM_MAX (4ull << 20)            // images of at least 4 MiB keep their own allocation
typedef struct iblock { struct iblock *next; VkDeviceMemory dm; uint32_t type; vke_list free; } iblock;
static iblock *g_blocks[VK_MAX_MEMORY_TYPES];
static pthread_mutex_t g_imu = PTHREAD_MUTEX_INITIALIZER;
// Stats (under g_imu): blocks, images in blocks (count, units' bytes, the images' own sizes), dedicated allocations (count, bytes), blocks freed, block allocations that failed
static struct { uint64_t blocks, runs, run_bytes, req_bytes, ded, ded_bytes, freed, fail; } S;

// Memory of these requirements fits a block (size, alignment, type): 1, or 0
static int block_fit(uint32_t type, const VkMemoryRequirements *mr) {
    return VKA_ENV("DM_VK_IMGMEM", 1) && mr->size < VKA_IMEM_MAX && mr->alignment <= VKA_IMEM_MAX && type < VK_MAX_MEMORY_TYPES && BLOCK <= V.max_alloc;
}
// Shared memory for the image (it fits a block): 1, or 0 (a dedicated allocation)
static int sharable(VkImage img, uint32_t type, const VkMemoryRequirements *mr) {
    if (!block_fit(type, mr) || !vk.vkGetImageMemoryRequirements2) return 0;
    VkMemoryDedicatedRequirements dr = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 r2 = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &dr};
    VkImageMemoryRequirementsInfo2 ri = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, NULL, img};
    vk.vkGetImageMemoryRequirements2(V.dev, &ri, &r2);
    return !dr.prefersDedicatedAllocation && !dr.requiresDedicatedAllocation;
}
static VkResult dedicated(VkImage img, uint32_t type, const VkMemoryRequirements *mr, vka_imem *m) {   // as before: one allocation for the image
    VkMemoryDedicatedAllocateInfo dai = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .image = img};
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &dai, .allocationSize = mr->size, .memoryTypeIndex = type};
    VkResult r;
    if ((r = vk.vkAllocateMemory(V.dev, &ai, NULL, &m->dm))) { m->dm = VK_NULL_HANDLE; return r; }
    if ((r = vk.vkBindImageMemory(V.dev, img, m->dm, 0))) { vk.vkFreeMemory(V.dev, m->dm, NULL); m->dm = VK_NULL_HANDLE; return r; }
    m->size = mr->size;
    pthread_mutex_lock(&g_imu); S.ded++; S.ded_bytes += mr->size; pthread_mutex_unlock(&g_imu);
    return VK_SUCCESS;
}
#define KEEP_FREE (16ull << 20)   // an empty block is kept only while the memory type's other blocks have less free in all
static int take(iblock *b, uint32_t n, uint32_t al, const VkMemoryRequirements *mr, vka_imem *m) {   // n units of b (caller holds g_imu): 1, or 0 when they do not fit
    int64_t u = vke_alloc(&b->free, n, al);
    if (u < 0) return 0;
    S.runs++; S.run_bytes += n * UNIT; S.req_bytes += mr->size;
    m->dm = b->dm; m->off = (uint64_t)u * UNIT; m->blk = b; m->units = n; m->size = mr->size;
    return 1;
}
// n units in the fullest block of the type they fit (caller holds g_imu): 1, or 0. Blocks are few (tens at most), so a pass over them per image
static int place(uint32_t type, uint32_t n, uint32_t al, const VkMemoryRequirements *mr, vka_imem *m) {
    iblock *o[64]; int k = 0;
    for (iblock *b = g_blocks[type]; b; b = b->next) {
        if (k == 64) { if (take(b, n, al, mr, m)) return 1; continue; }   // (beyond 64 blocks: list order)
        int i = k++; while (i > 0 && o[i - 1]->free.used < b->free.used) { o[i] = o[i - 1]; i--; }
        o[i] = b;
    }
    for (int i = 0; i < k; i++) if (take(o[i], n, al, mr, m)) return 1;
    return 0;
}
// Empty blocks of the type off the list, chained on *gone for freeing outside the lock (caller holds g_imu): every one, except one while the others have under KEEP_FREE free
// or no free extent of VKA_IMEM_MAX (fragmented free space: the next large image would allocate a block again, and a short-lived one would free it, a 64 MiB churn)
static void trim(uint32_t type, iblock **gone) {
    uint64_t fr = 0; int empties = 0, run = 0;
    for (iblock *x = g_blocks[type]; x; x = x->next) {
        if (vke_empty(&x->free)) { empties++; continue; }
        fr += (uint64_t)(x->free.units - x->free.used) * UNIT;
        for (uint32_t j = 0; j < x->free.nf && !run; j++) run = x->free.f[j].n >= VKA_IMEM_MAX / UNIT;
    }
    int keep = fr < KEEP_FREE || !run;
    if (empties <= keep) return;
    for (iblock **p = &g_blocks[type]; *p;) {
        iblock *b = *p;
        if (!vke_empty(&b->free) || (keep && keep--)) { p = &b->next; continue; }
        *p = b->next; b->next = *gone; *gone = b; S.blocks--; S.freed++;
    }
}

// A run of a block for memory of mr (block_fit), into m (nothing bound): 1, or 0 (no block could be allocated; nothing taken)
static int run_get(uint32_t type, const VkMemoryRequirements *mr, vka_imem *m) {
    uint32_t n = (uint32_t)((mr->size + UNIT - 1) / UNIT), al = mr->alignment > UNIT ? (uint32_t)(mr->alignment / UNIT) : 1;
    if (al & (al - 1)) return 0;   // (Vulkan alignments are powers of two)
    pthread_mutex_lock(&g_imu);
    int got = 0;
    got = place(type, n, al, mr, m);
    pthread_mutex_unlock(&g_imu);
    if (!got) {   // a new block, allocated without the lock; another thread may have added one meanwhile, which is tried first
        iblock *nb = calloc(1, sizeof *nb);
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = BLOCK, .memoryTypeIndex = type};
        int ok = nb && vke_init(&nb->free, (uint32_t)(BLOCK / UNIT)) && !vk.vkAllocateMemory(V.dev, &ai, NULL, &nb->dm);
        if (nb && !ok) nb->dm = VK_NULL_HANDLE;
        pthread_mutex_lock(&g_imu);
        got = place(type, n, al, mr, m);
        if (!got && ok) { nb->type = type; nb->next = g_blocks[type]; g_blocks[type] = nb; S.blocks++; got = take(nb, n, al, mr, m); nb = NULL; }
        if (!ok) S.fail++;
        pthread_mutex_unlock(&g_imu);
        if (nb) { if (nb->dm) vk.vkFreeMemory(V.dev, nb->dm, NULL); vke_fini(&nb->free); free(nb); }   // not needed (another thread's block had room) or not allocated
    }
    return got;
}
// The image's memory (type: the memory type the caller picked from mr's memoryTypeBits): space in a block, or a dedicated allocation as before; bound to the image.
// VK_SUCCESS, or the Vulkan error with nothing allocated
VkResult vka_imem_bind(VkImage img, uint32_t type, const VkMemoryRequirements *mr, vka_imem *m) {
    memset(m, 0, sizeof *m);
    if (!sharable(img, type, mr) || !run_get(type, mr, m)) return dedicated(img, type, mr, m);   // (no block could be allocated: this image as before)
    VkResult r;
    if ((r = vk.vkBindImageMemory(V.dev, img, m->dm, m->off))) { vka_imem_free(m); return r; }
    return VK_SUCCESS;
}

// Zero-shared memory (vka_zshare.h; vka_tex.c "Zero-shared images"): the image is bound at offset 0 of the allocation its key names, made (size mr->size, type) at the key's
// first image. *fresh: 1 when this call made it (the caller counts its bytes once). VK_SUCCESS, or the Vulkan error with nothing bound (the caller binds the image as before).
// An image the driver requires a dedicated allocation for is refused (VK_ERROR_FEATURE_NOT_PRESENT): a dedicated allocation names one image
static vkz_tab Z;
VkResult vka_imem_bind_zero(VkImage img, uint32_t type, const VkMemoryRequirements *mr, const void *key, uint32_t klen, vka_imem *m, int *fresh) {
    memset(m, 0, sizeof *m); *fresh = 0;
    if (klen > VKZ_KEY) return VK_ERROR_FEATURE_NOT_PRESENT;
    if (vk.vkGetImageMemoryRequirements2) {
        VkMemoryDedicatedRequirements dr = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
        VkMemoryRequirements2 r2 = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &dr};
        VkImageMemoryRequirementsInfo2 ri = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, NULL, img};
        vk.vkGetImageMemoryRequirements2(V.dev, &ri, &r2);
        if (dr.requiresDedicatedAllocation) return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    pthread_mutex_lock(&g_imu);
    int i = vkz_ref(&Z, key, klen);
    VkDeviceMemory dm = i >= 0 ? (VkDeviceMemory)(uintptr_t)Z.e[i].h : VK_NULL_HANDLE;
    pthread_mutex_unlock(&g_imu);
    VkResult r;
    if (i < 0) {   // the key's first image: its allocation (callers hold vka_tex.c's table lock, so no other image of the key comes meanwhile)
        VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr->size, .memoryTypeIndex = type};
        if ((r = vk.vkAllocateMemory(V.dev, &ai, NULL, &dm))) return r;
        if ((r = vk.vkBindImageMemory(V.dev, img, dm, 0))) { vk.vkFreeMemory(V.dev, dm, NULL); return r; }
        pthread_mutex_lock(&g_imu); int k = vkz_add(&Z, key, klen, (uint64_t)(uintptr_t)dm, mr->size); pthread_mutex_unlock(&g_imu);
        if (k < 0) { m->dm = dm; return VK_SUCCESS; }   // no room in the table: the allocation is this image's own (size 0: freed outside the stats)
        *fresh = 1;
    } else if ((r = vk.vkBindImageMemory(V.dev, img, dm, 0))) {
        pthread_mutex_lock(&g_imu); vkz_unref(&Z, (uint64_t)(uintptr_t)dm); pthread_mutex_unlock(&g_imu);   // (not the last: the entry's other images hold it)
        return r;
    }
    m->dm = dm; m->size = mr->size; m->z = 1;
    return VK_SUCCESS;
}
uint64_t vka_imem_zbytes(void) { uint64_t b; pthread_mutex_lock(&g_imu); vkz_stats(&Z, NULL, NULL, &b, NULL); pthread_mutex_unlock(&g_imu); return b; }
uint64_t vka_imem_zbytes_nolock(void) { uint64_t b = 0; for (int i = 0; i < Z.n; i++) b += Z.e[i].size; return b; }   // (a log line's value, read without the lock)
void vka_imem_zowned(void) { pthread_mutex_lock(&g_imu); vkz_note_own(&Z); pthread_mutex_unlock(&g_imu); }
int vka_imem_zadopt(vka_imem *m) {   // (callers hold vka_tex.c's table lock, as for vka_imem_bind_zero: no image of the key comes meanwhile)
    if (!m->z || !m->dm) return 0;
    pthread_mutex_lock(&g_imu);
    int ok = vkz_adopt(&Z, (uint64_t)(uintptr_t)m->dm);
    if (ok) { m->z = 0; m->blk = NULL; m->off = 0; S.ded++; S.ded_bytes += m->size; }   // (a dedicated allocation from now on: vka_imem_free frees it)
    pthread_mutex_unlock(&g_imu);
    return ok;
}

// Guest-aliased memory (vka_rtalias.h; vka_tex.c "Guest-aliased targets"): the image is bound at offset 0 of its key's current allocation, made (mr->size, or grown) when
// that one is too small or there is none. An image the driver requires a dedicated allocation for is refused (VK_ERROR_FEATURE_NOT_PRESENT), as for zero-shared memory.
// An allocation is a record (vka_imem, the table entry's handle): a dedicated VkDeviceMemory, or, under VKA_IMEM_MAX when the driver does not prefer a dedicated one, a run of
// a 64 MiB block as an image of that size alone takes. With DM_VK_RTALIAS_MB under 4 the 1-4 MiB targets come here, one allocation per storage start: as VkDeviceMemory objects
// of their own they would add to the device's allocation count what the blocks were made to save (maxMemoryAllocationCount; vka_imem_allocations counts the dedicated ones)
static vkra_tab A;
static void arec_free(vka_imem *b) { vka_imem_free(b); free(b); }   // an allocation record whose last image went (its run back to the block, or its memory freed)
VkResult vka_imem_bind_alias(VkImage img, uint32_t type, const VkMemoryRequirements *mr, const void *key, uint32_t klen, vka_imem *m, uint64_t *fresh) {
    memset(m, 0, sizeof *m); *fresh = 0;
    if (klen > VKRA_KEY) return VK_ERROR_FEATURE_NOT_PRESENT;
    int pref = 0;   // the driver prefers a dedicated allocation: no run of a block
    if (vk.vkGetImageMemoryRequirements2) {
        VkMemoryDedicatedRequirements dr = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
        VkMemoryRequirements2 r2 = {VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &dr};
        VkImageMemoryRequirementsInfo2 ri = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2, NULL, img};
        vk.vkGetImageMemoryRequirements2(V.dev, &ri, &r2);
        if (dr.requiresDedicatedAllocation) return VK_ERROR_FEATURE_NOT_PRESENT;
        pref = dr.prefersDedicatedAllocation != 0;
    } else pref = 1;
    uint64_t size = 0;
    pthread_mutex_lock(&g_imu);
    int i = vkra_ref(&A, key, klen, mr->size, &size);
    vka_imem *b = i >= 0 ? (vka_imem *)(uintptr_t)A.e[i].h : NULL; uint64_t asz = i >= 0 ? A.e[i].size : 0;
    pthread_mutex_unlock(&g_imu);
    VkResult r;
    if (i < 0) {   // none of the key holds it: a new current allocation (callers hold vka_tex.c's table lock, so no other image of the key comes meanwhile)
        if (V.max_alloc && size > V.max_alloc) size = mr->size;   // (the grown size past the device's limit: the image's own size)
        if (!(b = calloc(1, sizeof *b))) return VK_ERROR_OUT_OF_HOST_MEMORY;
        VkMemoryRequirements rr = *mr; rr.size = size;
        if (pref || !block_fit(type, &rr) || !run_get(type, &rr, b)) {   // a dedicated allocation (size 0 in its record: outside the dedicated stats)
            VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = size, .memoryTypeIndex = type};
            if ((r = vk.vkAllocateMemory(V.dev, &ai, NULL, &b->dm))) { free(b); return r; }
        }
        if ((r = vk.vkBindImageMemory(V.dev, img, b->dm, b->off))) { arec_free(b); return r; }
        pthread_mutex_lock(&g_imu); int k = vkra_add(&A, key, klen, (uint64_t)(uintptr_t)b, size, mr->size); pthread_mutex_unlock(&g_imu);
        if (k < 0) { *m = *b; free(b); return VK_SUCCESS; }   // no room in the table: the memory is this image's own (a run, or an allocation of size 0: freed outside the stats; the caller counts it as its own)
        *fresh = size; asz = size;
    } else if ((r = vk.vkBindImageMemory(V.dev, img, b->dm, b->off))) {
        pthread_mutex_lock(&g_imu); int last = vkra_unref(&A, (uint64_t)(uintptr_t)b, mr->size); pthread_mutex_unlock(&g_imu);
        if (last == 1) arec_free(b);   // (the entry's other images went meanwhile: vka_imem_free runs on the fence thread)
        return r;
    }
    m->dm = b->dm; m->off = b->off; m->size = mr->size; m->a = 1; m->asz = asz; m->ab = b;
    return VK_SUCCESS;
}
uint64_t vka_imem_abytes(void) { uint64_t b; pthread_mutex_lock(&g_imu); vkra_stats(&A, NULL, NULL, &b, NULL); pthread_mutex_unlock(&g_imu); return b; }
// (the `memory:` line's values: g_imu is a leaf lock, taken under the texture lock and the others vka_mem_grew's callers hold, never the other way, so these take it; the table
// may be reallocated by an image made on another thread meanwhile)
uint64_t vka_imem_abytes_nolock(void) { return vka_imem_abytes(); }
static void astats_locked(char *o, size_t n) {   // caller holds g_imu
    if (!n) return; o[0] = 0;
    if (!A.made) return;
    uint64_t ai, aa, ab, as, ar = 0; vkra_stats(&A, &ai, &aa, &ab, &as);
    for (int i = 0; i < A.n; i++) ar += ((const vka_imem *)(uintptr_t)A.e[i].h)->blk != NULL;
    snprintf(o, n, "; guest-aliased: %llu images in %llu allocations (%.0f MiB, %.0f MiB saved; %llu of them runs of blocks), %llu made, %llu of them grown", (unsigned long long)ai,
             (unsigned long long)aa, ab / 1048576.0, as / 1048576.0, (unsigned long long)ar, (unsigned long long)A.made, (unsigned long long)A.grown);
}
void vka_imem_astats(char *o, size_t n) { pthread_mutex_lock(&g_imu); astats_locked(o, n); pthread_mutex_unlock(&g_imu); }

// The memory back (after vkDestroyImage of the image bound to it, once the GPU is done with it)
void vka_imem_free(vka_imem *m) {
    if (!m->dm) return;
    if (m->a) {   // a guest-aliased allocation: freed with its last image
        pthread_mutex_lock(&g_imu); int last = vkra_unref(&A, (uint64_t)(uintptr_t)m->ab, m->size); pthread_mutex_unlock(&g_imu);
        if (last < 0) VRED_ONCE("guest-aliased image memory given back twice or never taken (left as it is)");
        if (last == 1) arec_free(m->ab);
        memset(m, 0, sizeof *m); return;
    }
    if (m->z) {   // a zero-shared allocation: freed with its last image
        pthread_mutex_lock(&g_imu); int last = vkz_unref(&Z, (uint64_t)(uintptr_t)m->dm); pthread_mutex_unlock(&g_imu);
        if (last < 0) VRED_ONCE("zero-shared image memory given back twice or never taken (left as it is)");
        if (last == 1) vk.vkFreeMemory(V.dev, m->dm, NULL);
        memset(m, 0, sizeof *m); return;
    }
    if (!m->blk) {
        vk.vkFreeMemory(V.dev, m->dm, NULL);
        if (m->size) { pthread_mutex_lock(&g_imu); S.ded--; S.ded_bytes -= m->size; pthread_mutex_unlock(&g_imu); }   // (size 0: an allocation made elsewhere, vka_framegen.c)
        memset(m, 0, sizeof *m); return;
    }
    iblock *b = m->blk, *gone = NULL;
    pthread_mutex_lock(&g_imu);
    int fr = vke_free(&b->free, (uint32_t)(m->off / UNIT), m->units);
    if (fr == -2) VRED_ONCE("image memory block: units %llu+%u given back twice or never given (left as they are)", (unsigned long long)(m->off / UNIT), m->units);
    else if (fr) { VRED_ONCE("image memory block: free extent list not grown, %u units lost to reuse until the block empties", m->units); b->free.used -= m->units; }   // still counted free: the block can be released
    if (fr != -2) { S.runs--; S.run_bytes -= m->units * UNIT; S.req_bytes -= m->size; }   // a bad free was never a run
    if (fr != -2) trim(b->type, &gone);   // (this block emptied, or the others gained the room that lets a kept empty one go)
    pthread_mutex_unlock(&g_imu);
    while (gone) { iblock *x = gone; gone = x->next; vk.vkFreeMemory(V.dev, x->dm, NULL); vke_fini(&x->free); free(x); }
    memset(m, 0, sizeof *m);
}

// For the stats line: the blocks, the bytes the images in them take, what rounding to units and the blocks' free space leave unused, the dedicated allocations
void vka_imem_stats(char *o, size_t n) {
    pthread_mutex_lock(&g_imu);
    snprintf(o, n, "image memory: %llu blocks (%.0f MiB, %llu freed), used %.1f MiB by %llu images (wasted %.2f MiB rounding to units, %.1f MiB free in blocks), %llu dedicated (%.0f MiB)",
             (unsigned long long)S.blocks, S.blocks * (double)BLOCK / 1048576.0, (unsigned long long)S.freed, S.req_bytes / 1048576.0, (unsigned long long)S.runs,
             (S.run_bytes - S.req_bytes) / 1048576.0, (S.blocks * (double)BLOCK - S.run_bytes) / 1048576.0, (unsigned long long)S.ded, S.ded_bytes / 1048576.0);
    if (S.fail) { size_t l = strlen(o); snprintf(o + l, n - l, "; %llu block allocations failed (those images dedicated)", (unsigned long long)S.fail); }
    if (Z.made) { uint64_t zi, za, zb, zs; vkz_stats(&Z, &zi, &za, &zb, &zs); size_t l = strlen(o);
        snprintf(o + l, n - l, "; zero-shared: %llu images in %llu allocations (%.0f MiB, %.0f MiB saved), %llu made their own", (unsigned long long)zi, (unsigned long long)za, zb / 1048576.0, zs / 1048576.0, (unsigned long long)Z.owned); }
    if (A.made) { size_t l = strlen(o); astats_locked(o + l, n - l); }
    pthread_mutex_unlock(&g_imu);
}
uint64_t vka_imem_allocations(void) {   // (a guest-aliased allocation in a block's run is not one of its own)
    pthread_mutex_lock(&g_imu); uint64_t c = S.blocks + S.ded + (uint64_t)Z.n;
    for (int i = 0; i < A.n; i++) c += ((const vka_imem *)(uintptr_t)A.e[i].h)->blk == NULL;
    pthread_mutex_unlock(&g_imu); return c;
}
#endif
