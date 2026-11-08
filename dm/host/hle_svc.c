// Horizon kernel system calls (triggered by the SDK runtime once the SDK is loaded). Only what the P1 boot chain needs is implemented: memory layout + heap + query + debug output + timing.
// Register convention: inputs w1.. / outputs w0=Result, x1=value (GetInfo / SetHeapSize), consistent with the public Horizon ABI.
#include "dm.h"
#include <sys/mman.h>
#include <errno.h>
#include <unistd.h>
#include <unistd.h>
#include <sched.h>
#include <time.h>
#include <stdatomic.h>
#include <sys/uio.h>
#if defined(__APPLE__)
#include <mach/mach.h>
#endif

// Reserve one whole block of guest address space, partitioned the Horizon way (address values are the real addresses returned by the host's mmap).
// It only takes address space, not memory.
// Size 5 GB: the game needs the first 1 GB (alias region, stacks) + one whole heap block (application memory pool of 3285 MiB, see APP_POOL below),
// about 4.2 GB in total; on iOS without the "extended virtual addressing" entitlement a single reservation can be at most 6 GB
// (read on a real device: 6 GB succeeds, 7 GB fails), so it cannot be made any wider --
// with an 8 GB reservation, builds from developer accounts that cannot sign that entitlement (free accounts) would exit at this step.
#define RESERVE DM_GAME_MEM_RESERVE   // SMBW: the heap needs only 1.47 GB, so the block stays at 5 GB (the total proven on a real device; a single 6 GB block can be reserved, but Metal etc. still need address space)
static uint8_t *g_as;              // ASLR region base = the whole block
static size_t g_reserve;           // its size: RESERVE, or more with a DM_ALIAS_SIZE alias region
static uint8_t *g_heap; static _Atomic size_t g_heap_cur; static size_t g_heap_cap;
// The address range lies entirely within the currently opened (readable and writable) part of the guest heap: such reads need no kernel trap to check, memcpy directly (used where guest memory is read every tick, such as diag.c tk_rd)
int dm_guest_heap_readable(uint64_t a, size_t n) { uint64_t b = (uint64_t)(uintptr_t)g_heap; size_t cur = g_heap_cur; return g_heap && a >= b && n <= cur && a - b <= cur - n; }
// The opened heap span for callers that address guest memory by region (hle_cheats.c region 1, like dmnt's heap extents): the
// reservation's base (guest address) and the length the game has actually opened with SetHeapSize right now, which grows and
// shrinks with it; both 0 until the game's first SetHeapSize. There is no lock: the caller uses the pair only to bounds-check and
// restarts the check on each access, so a concurrent resize can at worst give it an answer about the span a moment ago
void dm_guest_heap_range(uint64_t *base, size_t *open_len) {
    if (base) *base = (uint64_t)(uintptr_t)g_heap;
    if (open_len) *open_len = g_heap ? atomic_load(&g_heap_cur) : 0;
}
static uint8_t *g_alias; static size_t g_alias_cap;
static _Atomic uint64_t *g_alias_bits;   // alias region page bitmap (alias_pages)
static uint64_t *g_alias_exposed;   // one bit per host page ever opened; protected by g_map_lock
static uint8_t *g_stack; static size_t g_stack_cap;
// The address lies in the guest address block (every guest thread stack is here: the game's own heap or the stack region); the big-stack entry (hle_bigstack.c) uses it to decide whether to switch stacks
int dm_guest_as_contains(uintptr_t a) { uintptr_t b = (uintptr_t)g_as; return g_as && a >= b && a - b < g_reserve; }

// Guest RAM the graphics layer may back with GPU memory (dm/gfx/vk_android, placed Vulkan memory at the guest addresses): the regions whose pages are opened and given back here,
// each page range before it is opened, and the give-back (1 = done there). No-ops on the Apple builds
__attribute__((weak)) void dm_gfx_guest_region(void *base, size_t size, const char *name) { (void)base; (void)size; (void)name; }
__attribute__((weak)) void dm_gfx_guest_open(void *lo, size_t n) { (void)lo; (void)n; }
__attribute__((weak)) int dm_gfx_guest_release(uintptr_t lo, uintptr_t hi) { (void)lo; (void)hi; return 0; }
static void *alias_windows_thread(void *arg);
static void *regions_thread(void *arg);
static void *touched_thread(void *arg);
static void mem_init_once(void) {
    size_t pg = (size_t)getpagesize();
    if (RESERVE < 0x10000000 || DM_GAME_MEM_ALIAS > RESERVE - 0x10000000 ||
        DM_GAME_MEM_STACK > RESERVE - 0x10000000 - DM_GAME_MEM_ALIAS ||
        ((RESERVE | DM_GAME_MEM_ALIAS | DM_GAME_MEM_STACK) & (pg - 1)))
        die("guest address-space profile has invalid native boundaries");
    // DM_ALIAS_SIZE=<bytes>: an alias region of that size instead of the profile's, the heap and stacks as they are; the block grows with it. A Switch-sized one (64 GB) holds what
    // a game reserves unchanged, for measuring its address regions on the Mac (the NSP script's heap check); an iPhone has no room for more than the profile's
    size_t alias = DM_GAME_MEM_ALIAS; const char *as = getenv("DM_ALIAS_SIZE");
    if (as && *as) { alias = (size_t)strtoull(as, NULL, 0); if (alias < 0x10000000 || (alias & (pg - 1))) die("DM_ALIAS_SIZE %s: at least 256 MB, in whole host pages", as); }
    g_reserve = RESERVE - DM_GAME_MEM_ALIAS + alias;
    // On Android at a fixed address on a 64 GiB boundary, at least 2 MiB-aligned, so the regions start where placed Vulkan memory can be mapped (plat_android.c, dm/gfx/vk_android)
    g_as = plat_mem_reserve(g_reserve);
    if (!g_as) die("reserve guest AS (%#llx bytes, errno %d)", (unsigned long long)g_reserve, errno);
    // Fixed partition (all inside the reserved block): alias 2.875 GB, stack 256 MB, heap the remaining 1.625 GB.
    // In newer SDKs the AddressRegionManager always takes 0x3fe00000 (~1 GB) at the start of the alias region as its node area (sdk 0x134728) before handing out regions; with a 512 MB alias region the very first AllocateAddressRegion fails (0x1803).
    // SMBW also defines a 1792000 KB (1.75 GB) VirtualAddressMemory in HeapDef (System/HeapSize): it is subtracted from the memory pool before the heap is opened (SetHeapSize asks for only 0x5be00000),
    //   and resource loading then MapPhysicalMemory's into the alias region after the node area on demand. With a 1.25 GB alias region only about 256 MB was left there, and 3 MB-class layout archives (WipeCourseIn / WipeWarp)
    //   got no address, were skipped, and crashed on a null pointer when used (main +0x373e04). So alias = node area + 1.75 GB + slack; the heap needs only 1.47 GB, and 2.625 GB is plenty.
    // The alias and stack sizes come from the game profile (mem_alias / mem_stack), the heap takes the rest; Mario: alias 0xB8000000, stack 0x10000000 → stack at +0xC8000000, heap at +0xD8000000
    g_alias = g_as + 0x10000000; g_alias_cap = alias;
    g_stack = g_alias + g_alias_cap; g_stack_cap = DM_GAME_MEM_STACK;
    g_heap  = g_stack + g_stack_cap; g_heap_cap  = g_reserve - (size_t)(g_heap - g_as);
    if (!(g_alias_bits = calloc((g_alias_cap >> 18) + 1, sizeof *g_alias_bits))) die("alias region page bitmap calloc");
    if (!(g_alias_exposed = calloc(g_alias_cap / pg / 64 + 1, sizeof *g_alias_exposed))) die("alias exposure bitmap calloc");
    LOG("guest AS %p +%#llx  heap %p alias %p stack %p", g_as, (unsigned long long)g_reserve, g_heap, g_alias, g_stack);
    dm_gfx_guest_region(g_heap, g_heap_cap, "heap"); dm_gfx_guest_region(g_alias, g_alias_cap, "alias");
    const char *aw = getenv("DM_ALIAS_WINDOWS");
    if (aw && atoi(aw) > 0) { pthread_t th; if (!pthread_create(&th, NULL, alias_windows_thread, (void *)(uintptr_t)atoi(aw))) pthread_detach(th); }
    const char *rf = getenv("DM_REGIONS");
    if (rf && *rf) { pthread_t th; if (!pthread_create(&th, NULL, regions_thread, (void *)rf)) pthread_detach(th); }
    const char *ts = getenv("DM_MEM_TOUCHED_S"); int tper = ts && *ts ? atoi(ts) : 30;
    if (tper > 0) { pthread_t th; if (!pthread_create(&th, NULL, touched_thread, (void *)(intptr_t)tper)) pthread_detach(th); }
}
static void mem_init(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, mem_init_once);
}

// Application memory pool = the 3285 MB the Switch (4 GB model) gives applications. At startup (main+0x1b66160) the game sets the whole total reported by QueryMemoryInfo as the heap, retrying with 2 MB less on failure,
// and hands the whole block to its own StandardAllocator (main+0x32137a0) -- so heap size = pool − image footprint, it has only as much as the host reports.
// Previously 3 GB was reported here and SetHeapSize never failed: the heap was about 150 MB smaller than on the real device, at the run entry peak the main heap had only about 18 MB left, and under high local load a timing shift made a 128 KB request return null,
// which the guest writes to without a null check (CRASH main+0x1b36138, re-verified after merge with inv_tap).
#define APP_POOL DM_GAME_APP_POOL   // game profile app_pool (Mario 0xCD500000)
static size_t image_bytes(void) { size_t n = 0; for (int i = 0; i < g_nmod; i++) if (!g_mod[i].nro) n += g_mod[i].size;   // an NRO lives in memory the game allocated (its image and bss buffers)
    return (n + 0x1fffff) & ~(size_t)0x1fffff; }
// Giant bss (modules with the patch area before the base, patch_nso.py FAR): the image bytes beyond the file part. Used memory was reported as "heap + 64 MB", where the 64 MB covers ordinary modules' code and data (Mario's two modules are about 74 MB together,
// and the heap it opens = pool − 64 MB − 1.75 GB virtual address memory − 1 MB = 0x5be00000, as on hardware); Metroid Dread's main has about 1.8 GB of bss, the engine's static heap, which the Switch counts as used;
// without it the game asks for pool − 64 MB = 3.15 GB of heap
// Like image_bytes, exclude NROs: their buffers already belong to the game's heap. Counting every offline-mapped NRO's bss made FFX/X-2's used memory exceed its pool before any LoadModule.
static size_t big_bss_bytes(void) { size_t n = 0; for (int i = 0; i < g_nmod; i++) if (!g_mod[i].nro && g_mod[i].patch_off < 0) n += g_mod[i].size - g_mod[i].back_len - g_mod[i].file_image_len; return n; }

// For memory accounting (memstat.m): the range of the whole guest address space block (base = NULL when not initialized)
void svc_guest_range(uint8_t **base, size_t *size) { *base = g_as; *size = g_reserve; }

enum { I_CoreMask=0, I_PriorityMask=1, I_AliasAddr=2, I_AliasSize=3, I_HeapAddr=4, I_HeapSize=5,
       I_TotalMem=6, I_UsedMem=7, I_DebuggerAttached=8, I_ResourceLimit=9, I_IdleTick=10, I_RandomEntropy=11,
       I_AslrAddr=12, I_AslrSize=13, I_StackAddr=14, I_StackSize=15, I_SysResTotal=16, I_SysResUsed=17,
       I_ProgramId=18, I_InitProcIdRange=19, I_UserExcAddr=20, I_TotalNonSys=21, I_UsedNonSys=22, I_IsApplication=23, I_FreeThreadCount=24,
       I_ThreadTickCount=25, I_IsSvcPermitted=26, I_IoRegionHint=27, I_AliasExtraSize=28,
       I_RemoteAddr=29, I_RemoteSize=30, I_RemoteUsage0=31, I_RemoteUsage1=32, I_PageSize=33 };

static uint8_t g_seen[256];

// Stack destinations and their borrowed sources. Physical alias commits live in
// g_alias_bits. One lock covers memory validation, protections, copying, budgets
// and publication; QueryMemory and used-memory queries take the same snapshot.
// g_map_lock is held over a whole lookup + check + change (and over QueryMemory's reads): map_find's index is only valid under it, since map_del moves the last entry into the freed slot.
// The lock used to cover only the add and the delete, so two threads unmapping at once (stacks of exiting threads) could delete through a stale index and drop another live mapping
#define MAX_MAP 4096
static struct { uint64_t base, size, src; uint32_t state; } g_map[MAX_MAP];
static int g_nmap;
static pthread_mutex_t g_map_lock = PTHREAD_MUTEX_INITIALIZER;
// A hole-free read / write of guest heap bytes for the cheat VM (hle_cheats.c): under g_map_lock no concurrent SetHeapSize
// can mprotect the span away between the bounds check and the copy (both hold this same lock), so a live heap address cannot
// turn into a fault while the cheat VM touches it. Both return 0 when every byte was inside the currently opened (readable
// and writable) span, 1 otherwise. The address is the guest heap address the cheat VM computed (heap base + rel + offset register)
int dm_guest_heap_write(uint64_t addr, const void *src, size_t len) {
    pthread_mutex_lock(&g_map_lock);
    int out = !dm_guest_heap_readable(addr, len);
    if (!out) { dm_guest_write_prepare((void *)(uintptr_t)addr, len); memcpy((void *)(uintptr_t)addr, src, len); }
    pthread_mutex_unlock(&g_map_lock);
    return out;
}
int dm_guest_heap_read(uint64_t addr, void *out, size_t len) {
    pthread_mutex_lock(&g_map_lock);
    int bad = !dm_guest_heap_readable(addr, len);
    if (!bad) memcpy(out, (void *)(uintptr_t)addr, len);
    pthread_mutex_unlock(&g_map_lock);
    return bad;
}
static int map_find(uint64_t a, uint64_t n) {       // returns a 1-based index, 0 = no overlap; caller holds g_map_lock
    for (int i = 0; i < g_nmap; i++) if (a < g_map[i].base + g_map[i].size && g_map[i].base < a + n) return i + 1;
    return 0;
}
static void map_add(uint64_t b, uint64_t n, uint64_t src, uint32_t st) {   // caller holds g_map_lock
    if (g_nmap >= MAX_MAP) die("mapping table full");
    g_map[g_nmap++] = (typeof(g_map[0])){b, n, src, st};
}
static void map_del(int i) { g_map[i] = g_map[--g_nmap]; }   // caller holds g_map_lock

// One bit per 4 KB page of the alias region (mapped = 1): Map sets bits, Unmap clears them; returns the number of pages whose state really changed (signed: positive for Map, negative for Unmap)
static int64_t alias_pages(uint64_t off, uint64_t n, int map) {
    int64_t ch = 0;
    for (uint64_t p = off >> 12, e = (off + n + 0xFFF) >> 12; p < e; ) {
        uint64_t sh = p & 63, k = 64 - sh < e - p ? 64 - sh : e - p, m = (k == 64 ? ~0ull : (1ull << k) - 1) << sh;
        uint64_t old = map ? atomic_fetch_or(&g_alias_bits[p >> 6], m) : atomic_fetch_and(&g_alias_bits[p >> 6], ~m);
        ch += __builtin_popcountll((map ? ~old : old) & m);
        p += k;
    }
    return map ? ch : -ch;
}
// Pages of the range not mapped yet (the bitmap is left alone)
static int64_t alias_unmapped(uint64_t off, uint64_t n) {
    int64_t c = 0;
    for (uint64_t p = off >> 12, e = (off + n + 0xFFF) >> 12; p < e; ) {
        uint64_t sh = p & 63, k = 64 - sh < e - p ? 64 - sh : e - p, m = (k == 64 ? ~0ull : (1ull << k) - 1) << sh;
        c += __builtin_popcountll(~atomic_load(&g_alias_bits[p >> 6]) & m);
        p += k;
    }
    return c;
}
static _Atomic int64_t g_alias_mapped;   // bytes mapped in the alias region (page bitmap)
// Readers of guest memory on the host's behalf that the guest names by address (the Vulkan replay's texture uploads from pool storage): a page the guest has not mapped
// is inaccessible here (PROT_NONE), so it is read as zeros instead. No lock: UnmapPhysicalMemory holds g_map_lock while vka_mem_release waits for GPU fences the replay thread
// completes, so a reader taking it could deadlock. The bitmap is read atomically, Unmap clears a range's bits before it releases the pages, and the copy itself goes through
// guest_copy, which reports a page that became inaccessible after the check instead of faulting on it (Mario Kart World 1.8.0 docked: SEGV_ACCERR in this memcpy on vk-replay)
static int guest_page_mapped(uintptr_t p) {   // p: a guest address (page aligned)
    uintptr_t al = (uintptr_t)g_alias, hp = (uintptr_t)g_heap;
    if (g_alias && p >= al && p - al < g_alias_cap) { uint64_t i = (p - al) >> 12; return (int)(atomic_load(&g_alias_bits[i >> 6]) >> (i & 63) & 1); }
    if (g_heap && p >= hp && p - hp < g_heap_cap) return p - hp < atomic_load(&g_heap_cur);
    return 1;
}
uint64_t dm_guest_unmapped(uintptr_t a, size_t n) {
    uint64_t c = 0; if (!n) return 0;
    for (uintptr_t p = a & ~(uintptr_t)0xFFF; p < a + n; p += 0x1000) c += !guest_page_mapped(p);
    return c;
}
// Copy n bytes of our own memory that may become inaccessible meanwhile: the bytes copied before the first inaccessible one (a system call reads them, so an inaccessible page
// is an error, not a signal). If the system refuses the call itself (a seccomp filter: not seen on Android), a plain copy, with a red line once
static size_t guest_copy(void *d, const void *s, size_t n) {
#if defined(__APPLE__)
    vm_size_t got = 0;
    if (vm_read_overwrite(mach_task_self(), (vm_address_t)s, n, (vm_address_t)d, &got) == KERN_SUCCESS) return got;
    // whole-range call: on failure, page by page to keep the readable prefix
    size_t k = 0;
    for (uintptr_t p = (uintptr_t)s, e = p + n; p < e; ) {
        uintptr_t q = (p & ~(uintptr_t)0xFFF) + 0x1000; if (q > e) q = e; got = 0;
        if (vm_read_overwrite(mach_task_self(), (vm_address_t)p, q - p, (vm_address_t)((uint8_t *)d + (p - (uintptr_t)s)), &got) != KERN_SUCCESS || got != q - p) break;
        k += q - p; p = q;
    }
    return k;
#else
    struct iovec l = {d, n}, r = {(void *)s, n};
    ssize_t got = process_vm_readv(getpid(), &l, 1, &r, 1, 0);
    if (got >= 0) return (size_t)got;
    if (errno == EFAULT) return 0;
    static _Atomic int nred; if (!atomic_fetch_add(&nred, 1)) LOG("[RED] process_vm_readv refused (errno %d): guest pages are copied without fault protection", errno);
    memcpy(d, s, n); return n;
#endif
}
uint64_t dm_guest_read_mapped(void *dst, uintptr_t a, size_t n) {
    uint64_t c = 0; uint8_t *d = dst;
    for (uintptr_t p = a; p < a + n; ) {
        uintptr_t e = (p & ~(uintptr_t)0xFFF) + 0x1000; if (e > a + n) e = a + n;
        if (!guest_page_mapped(p & ~(uintptr_t)0xFFF)) { memset(d + (p - a), 0, e - p); c++; p = e; continue; }
        while (e < a + n && guest_page_mapped(e)) e = e + 0x1000 < a + n ? e + 0x1000 : a + n;   // a run of mapped pages: one copy
        size_t got = guest_copy(d + (p - a), (const void *)p, e - p);
        if (got == e - p) { p = e; continue; }
        // unmapped during the copy: the readable prefix stays, the page it stopped in reads as zeros, and the rest of the run is looked at again
        uintptr_t f = ((p + got) & ~(uintptr_t)0xFFF) + 0x1000; if (f > a + n) f = a + n;
        memset(d + (p + got - a), 0, f - (p + got)); c++; p = f;
    }
    return c;
}

// Word-at-a-time boundaries of the bitmap run containing p (guest page index).
static uint64_t alias_mask(unsigned n) { return n == 64 ? UINT64_MAX : (UINT64_C(1) << n) - 1; }
static uint64_t alias_run_end(uint64_t p, uint64_t end, int mapped) {
    for (uint64_t first = p + 1; first < end;) {
        unsigned shift = first & 63, n = (unsigned)(end - first < 64 - shift ? end - first : 64 - shift);
        uint64_t bits = atomic_load(&g_alias_bits[first >> 6]);
        uint64_t different = (mapped ? ~bits : bits) & (alias_mask(n) << shift);
        if (different) return (first & ~UINT64_C(63)) + __builtin_ctzll(different);
        first += n;
    }
    return end;
}
static void alias_query(uint64_t addr, uint64_t *base, uint64_t *size, int *mapped) {
    uint64_t p = (addr - (uintptr_t)g_alias) >> 12, lo = p;
    *mapped = (atomic_load(&g_alias_bits[p >> 6]) >> (p & 63)) & 1;
    while (lo) {
        uint64_t first = (lo - 1) & ~UINT64_C(63);
        uint64_t bits = atomic_load(&g_alias_bits[first >> 6]);
        uint64_t different = (*mapped ? ~bits : bits) & alias_mask((unsigned)(lo - first));
        if (different) { lo = first + 64 - __builtin_clzll(different); break; }
        lo = first;
    }
    uint64_t hi = alias_run_end(p, g_alias_cap >> 12, *mapped);
    *base = (uintptr_t)g_alias + (lo << 12); *size = (hi - lo) << 12;
}
// Give guest memory back to the system: a fresh inaccessible mapping over [lo, hi), whole host pages. mprotect(PROT_NONE) alone keeps every page resident and dirty, only
// out of phys_footprint, and madvise(MADV_FREE) does nothing to pages that are not writable, though it returns 0 (1 GiB on the Mac: 1,024 MiB still resident and dirty; this: 0).
// A memory pool's no-copy buffer over the range keeps the old pages for the GPU work still using it until the buffer goes. Returns 1 when the range now holds zero pages;
// on failure the range is only inaccessible, with its old bytes
static int guest_release(uintptr_t lo, uintptr_t hi) {
    if (dm_gfx_guest_release(lo, hi)) return 1;
    if (mmap((void *)lo, hi - lo, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0) != MAP_FAILED) return 1;
    static _Atomic int nred; if (atomic_fetch_add(&nred, 1) < 8) LOG("[RED] guest memory %#lx+%#lx not given back (errno %d): it stays resident", (unsigned long)lo, (unsigned long)(hi - lo), errno);
    if (mprotect((void *)lo, hi - lo, PROT_NONE)) die("guest_release mprotect %#lx+%#lx errno=%d", (unsigned long)lo, (unsigned long)(hi - lo), errno);
    return 0;
}
// Initialize holes using the OLD commit bitmap. Fresh anonymous host pages stay
// lazy; after any exposure, even an untouched 4 KiB sibling may hold stale bytes.
static void alias_initialize(uint64_t off, size_t size) {
    size_t pg = (size_t)getpagesize(), ratio = pg >> 12;
    uint64_t end = (off + size) >> 12;
    for (uint64_t p = off >> 12; p < end;) {
        int mapped = (atomic_load(&g_alias_bits[p >> 6]) >> (p & 63)) & 1;
        uint64_t next = alias_run_end(p, end, mapped);
        if (!mapped) {
            uint64_t clear = p;
            for (uint64_t q = p; q < next;) {
                uint64_t host = q / ratio, stop = (host + 1) * ratio;
                if (stop > next) stop = next;
                if (!(g_alias_exposed[host >> 6] & (UINT64_C(1) << (host & 63)))) {
                    if (clear < q) memset(g_alias + (clear << 12), 0, (q - clear) << 12);
                    clear = stop;
                }
                q = stop;
            }
            if (clear < next) memset(g_alias + (clear << 12), 0, (next - clear) << 12);
        }
        p = next;
    }
    for (uint64_t h = off / pg, last = (off + size - 1) / pg; h <= last; ++h)
        g_alias_exposed[h >> 6] |= UINT64_C(1) << (h & 63);
}

static int memory_contains(uint64_t base, uint64_t capacity, uint64_t addr, uint64_t size) {
    return addr >= base && size <= capacity && addr - base <= capacity - size;
}
static uint32_t map_arguments(uint64_t dst, uint64_t src, uint64_t size) {
    if ((dst | src) & 0xfff) return 0xCC01;
    if (!size || (size & 0xfff)) return 0xCA01;
    if (size > UINT64_MAX - src || size > UINT64_MAX - dst) return 0xD401;
    return memory_contains((uintptr_t)g_stack, g_stack_cap, dst, size) ? 0 : 0xDC01;
}
static int map_source_busy(uint64_t addr, uint64_t size) { // g_map_lock held; nonwrapping ranges
    for (int i = 0; i < g_nmap; ++i)
        if (addr < g_map[i].src + g_map[i].size && g_map[i].src < addr + size) return 1;
    return 0;
}
// Loader text/patch seals use 16 KiB boundaries, independently of host page size.
static void module_readonly_run(uint64_t off, uint64_t start, uint64_t end,
                                uint64_t *lo, uint64_t *hi, uint64_t *perm) {
    if (start >= end) return;
    if (off < start) { if (start < *hi) *hi = start; }
    else if (off >= end) { if (end > *lo) *lo = end; }
    else { if (start > *lo) *lo = start; if (end < *hi) *hi = end; *perm &= ~UINT64_C(2); }
}
static void module_data_query(const module_t *m, uint64_t off, uint64_t *lo, uint64_t *hi, uint64_t *perm) {
    module_readonly_run(off, 0, (m->text_end + 0x3fff) & ~UINT64_C(0x3fff), lo, hi, perm);
    if (m->patch_off > 0) module_readonly_run(off, (uint64_t)m->patch_off & ~UINT64_C(0x3fff), m->size, lo, hi, perm);
    if (m->back_len) module_readonly_run(off, m->back_off, m->back_off + m->back_len, lo, hi, perm);
    module_readonly_run(off, m->key_page_off, m->key_page_off + m->key_page_size, lo, hi, perm);
}
static int map_source_backed(uint64_t src, uint64_t size) {
    if (memory_contains((uintptr_t)g_heap, g_heap_cur, src, size)) return 1;
    int dm_ue_heap_owns(uint64_t a, uint64_t n);   // Unreal's heap on a host malloc zone outside the guest block (hle_uemalloc.c): Unreal allocates its threads' stacks there
    if (dm_ue_heap_owns(src, size)) return 1;
    if (memory_contains((uintptr_t)g_alias, g_alias_cap, src, size))
        return !alias_unmapped(src - (uintptr_t)g_alias, size);
    module_t *m = module_of((void *)(uintptr_t)src);
    // Active NRO data is valid backing too (FFX/X-2's launcher allocates thread stacks from its bss). Deactivation below checks borrowers under this same lock before reload can restore it.
    if (!m || !m->loaded || !memory_contains((uintptr_t)m->base, m->size, src, size)) return 0;
    uint64_t lo = m->data_off ? m->data_off : (m->ro_off ? m->ro_off : m->text_end);
    uint64_t hi = m->patch_off > 0 ? (uint64_t)m->patch_off : m->back_len ? m->back_off : m->size;
    uint64_t off = src - (uintptr_t)m->base, perm = 3;
    module_data_query(m, off, &lo, &hi, &perm);
    return perm == 3 && hi >= lo && memory_contains(lo, hi - lo, off, size);
}
int svc_nro_deactivate(module_t *m) {
    pthread_mutex_lock(&g_map_lock);
    uint64_t base = (uintptr_t)m->base;
    for (int i = 0; i < g_nmap; i++)
        if (g_map[i].src < base + m->size && base < g_map[i].src + g_map[i].size) {
            pthread_mutex_unlock(&g_map_lock); return 0;
        }
    m->loaded = 0;
    pthread_mutex_unlock(&g_map_lock); return 1;
}
static void map_source_query(uint64_t addr, uint64_t *base, uint64_t *size, uint32_t *attr, uint64_t *perm) {
    uint64_t lo = *base, remaining = *size; // the final free page can end at 2^64
    for (int i = 0; i < g_nmap; ++i) {
        uint64_t src = g_map[i].src, end = src + g_map[i].size;
        uint64_t lower = lo, upper = remaining;
        if (addr >= src && addr < end) {
            if (src > lower) lower = src;
            if (end - lo < upper) upper = end - lo;
            *attr = 1; *perm = 0; // Locked / IsBorrowed, not PermissionLocked (bit 4)
        } else if (end <= addr && end > lower) lower = end;
        else if (src > addr && src - lo < upper) upper = src - lo;
        remaining = upper - (lower - lo); lo = lower;
    }
    *base = lo; *size = remaining;
}

// Diagnostic DM_ALIAS_WINDOWS=<window MB>: every 30 s, the mapped memory of each window of that size in the alias region (windows aligned to the size in absolute addresses, counted from the page
// bitmap; windows with nothing mapped are left out). For sizing an allocator that reserves fixed, aligned windows and commits pages from each window's start: Unreal's FMallocBinned3 takes
// 74 pools of 512 MB (Octopath Traveler 0, main+0x9f748)
static void *alias_windows_thread(void *arg) {
    uint64_t win = (uint64_t)(uintptr_t)arg << 20, a0 = (uint64_t)(uintptr_t)g_alias, a1 = a0 + g_alias_cap;
    for (;;) {
        sleep(30);
        char line[8192] = {0}; size_t o = 0; uint64_t tot = 0;
        pthread_mutex_lock(&g_map_lock);
        for (uint64_t w = a0 & ~(win - 1); w < a1 && o < sizeof line - 64; w += win) {
            uint64_t n = 0, p0 = (w > a0 ? w - a0 : 0) >> 12, p1 = ((w + win < a1 ? w + win : a1) - a0) >> 12;
            for (uint64_t p = p0; p < p1; p++) n += atomic_load(&g_alias_bits[p >> 6]) >> (p & 63) & 1;
            if (n) { o += snprintf(line + o, sizeof line - o, " +%#llx:%.1f", (unsigned long long)(w - (a0 & ~(win - 1))), n * 4096 / 1048576.0); tot += n; }
        }
        pthread_mutex_unlock(&g_map_lock);
        LOG("[mem] alias windows of %llu MB (window offset:MB mapped), %.1f MB in all:%s", (unsigned long long)(win >> 20), tot * 4096 / 1048576.0, line);
    }
    return NULL;
}

// ---- Guest memory touched (Linux / Android; counters only) ----
// Every DM_MEM_TOUCHED_S seconds (default 30, 0 off) and at exit (atexit, dm_host_shutdown): a `[mem] touched` line of the guest heap [0, current size), the alias
// region's mapped pages (its page bitmap, read in 256 KiB runs: "never" counts the runs' unmapped pages too) and the memory pools not placed yet (dm_gfx_unplaced_pools;
// inside the other two), from /proc/self/pagemap (its flags need no privilege, only the frame numbers do): resident anonymous (bit 63), resident shared or file
// (bit 63 and 61: placed Vulkan memory over guest addresses, or a file mapping: "file" when not placed), swapped (bit 62: zram on the phone), never touched (none of them).
// Without pagemap, mincore gives resident only ("swapped" then reads 0, "never" holds both). Then /sys/block/zram0/mm_stat (orig / compr / mem_used) when readable,
// /proc/meminfo's swap in use, and the scan's cost.
// One pass serves this line and the footprint lines below (DM_MEM_FOOT): the ranges both ask for (the heap holds most pools, the touched line's pools not placed are pools
// too) are cut at every range boundary (mem_foot.h dmf_sweep) and each piece's pages are read once. Placed memory (dm_gfx_placed_ranges: the driver's memory over guest
// addresses, committed when allocated) is not read: its pages count as resident shared (fb8: 1.4 GiB placed, read every 30 s before). The tables are sized from the live
// counts (no cap); a table still short after it grew is said ("n of m").
// For guest-memory.md §4: heap touched ~6.2 GB makes a smaller app_pool worth an A/B, ~3.3 GB closes the guest side
__attribute__((weak)) int dm_gfx_unplaced_pools(uint64_t *r, int max) { (void)r; (void)max; return 0; }   // (addr, size) pairs of pools not placed yet; returns how many there are (vka_mem.c)
__attribute__((weak)) int dm_gfx_placed_ranges(uint64_t *r, int max) { (void)r; (void)max; return 0; }    // (lo, hi) pairs of placed memory; returns how many there are (vka_mem.c)
#if defined(__linux__)
#include <fcntl.h>
#include <sys/resource.h>
#include "mem_foot.h"
typedef struct { uint64_t res, shr, swp, all, fil; } tch_t;   // bytes; fil: the part of shr read from pagemap (shared or file pages outside placed memory)
static pthread_mutex_t g_tch_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_pm_fd = -2, g_pm_err;   // /proc/self/pagemap (-2 not opened yet, -1 refused: g_pm_err)
static uint64_t g_pm_read, g_pm_skip;   // bytes of pagemap read so far, bytes of placed memory counted without a read (the lines' cost)
static void tch_range(uint64_t a, uint64_t len, tch_t *t) {   // caller holds g_tch_lock
    static uint64_t e[8192]; static unsigned char v[8192];
    uint64_t pg = (uint64_t)getpagesize(), lo = a / pg, hi = (a + len + pg - 1) / pg;
    t->all += (hi - lo) * pg;
    for (uint64_t p = lo; p < hi; ) {
        size_t c = hi - p < 8192 ? (size_t)(hi - p) : 8192;
        if (g_pm_fd >= 0) {
            ssize_t r = pread(g_pm_fd, e, c * 8, (off_t)(p * 8)); if (r < 8) return;
            c = (size_t)r / 8; g_pm_read += (uint64_t)r;
            for (size_t i = 0; i < c; i++) { uint64_t x = e[i]; if (x >> 63) { if (x >> 61 & 1) t->shr += pg; else t->res += pg; } else if (x >> 62 & 1) t->swp += pg; }
        } else {
            if (mincore((void *)(uintptr_t)(p * pg), c * pg, v)) return;
            for (size_t i = 0; i < c; i++) t->res += (v[i] & 1) * pg;
        }
        p += c;
    }
}
static int tch_str(char *b, size_t n, const char *name, const tch_t *t, uint64_t size) {
    double G = 1073741824.0; uint64_t used = t->res + t->shr + t->swp;
    int o = snprintf(b, n, "%s %.2f: resident %.2f, shared %.2f, swapped %.2f, never %.2f", name, size / G, t->res / G, t->shr / G, t->swp / G, (t->all > used ? t->all - used : 0) / G);
    if (t->fil && o >= 0 && (size_t)o < n) o += snprintf(b + o, n - (size_t)o, " (file %.2f)", t->fil / G);
    return o;
}
// The scan plan (under g_tch_lock): the ranges asked for, each with the counter it adds to (to[i] for q[i]); kept between lines (sized by the largest plan so far)
typedef struct { dmf_q *q; tch_t **to; int n, cap, fail; } scan_plan;
static void sp_add(scan_plan *p, uint64_t lo, uint64_t len, tch_t *to) {
    if (!len) return;
    uint64_t pg = (uint64_t)getpagesize(), a = lo / pg * pg, e = (lo + len + pg - 1) / pg * pg;
    if (p->n == p->cap) {
        int nc = p->cap ? 2 * p->cap : 1024; dmf_q *nq = realloc(p->q, (size_t)nc * sizeof *nq); if (!nq) { p->fail = 1; return; }
        p->q = nq; tch_t **nt = realloc(p->to, (size_t)nc * sizeof *nt); if (!nt) { p->fail = 1; return; }
        p->to = nt; p->cap = nc;
    }
    p->q[p->n] = (dmf_q){a, e, p->n}; p->to[p->n++] = to;
}
typedef struct { scan_plan *p; tch_t *to; } sp_ctx;
static void sp_gap(uint64_t a, uint64_t e, void *c) { const sp_ctx *x = c; sp_add(x->p, a, e - a, x->to); }
typedef struct { const scan_plan *p; const dmf_iv *pl; int npl; } sp_run_ctx;
static void sp_read(uint64_t a, uint64_t e, void *t) { tch_range(a, e - a, (tch_t *)t); }
static void sp_piece(uint64_t x, uint64_t y, const int *act, int na, void *ctx) {   // one piece: read once (outside placed memory), added to every range over it
    const sp_run_ctx *c = ctx; tch_t t = {0};
    dmf_gaps(x, y, c->pl, c->npl, sp_read, &t);
    uint64_t placed = y - x > t.all ? y - x - t.all : 0;   // (pieces and placed ranges are whole pages: what the gaps leave is placed)
    t.fil = t.shr; t.shr += placed; t.all += placed; g_pm_skip += placed;
    for (int k = 0; k < na; k++) { tch_t *o = c->p->to[act[k]]; o->res += t.res; o->shr += t.shr; o->swp += t.swp; o->all += t.all; o->fil += t.fil; }
}
static int sp_run(const scan_plan *p, const dmf_iv *pl, int npl) {   // 0: no memory for the sweep's tables
    static dmf_ev *ev; static int *act; static int cap;
    if (p->n > cap) {
        int nc = p->n + p->n / 4; dmf_ev *ne = realloc(ev, (size_t)nc * 2 * sizeof *ne); if (!ne) return 0;
        ev = ne; int *na = realloc(act, (size_t)nc * sizeof *na); if (!na) return 0;
        act = na; cap = nc;
    }
    sp_run_ctx c = {p, pl, npl};
    dmf_sweep(p->q, p->n, ev, act, sp_piece, &c);
    return 1;
}
// A table sized from a live count: grown to the count (and an eighth) when the last call said there are more; how many it holds after the call (*total: how many there are)
static int grow(void **t, int *cap, size_t each, int want) {
    if (want <= *cap) return 1;
    int nc = want + want / 8 + 64; void *n = realloc(*t, (size_t)nc * each); if (!n) return 0;
    *t = n; *cap = nc; return 1;
}
static int ranges_get(int (*get)(uint64_t *, int), uint64_t **r, int *cap, int *total) {   // (pairs of uint64_t)
    int n = get(*r, *cap);
    for (int t = 0; t < 2 && n > *cap; t++) { if (!grow((void **)r, cap, 16, n)) break; n = get(*r, *cap); }
    *total = n; return n < *cap ? n : *cap;
}

// ---- Guest memory footprint (DM_MEM_FOOT=1; Linux / Android; counters only) ----
// With the `[mem] touched` line (same thread, period and pagemap pass, DM_MEM_TOUCHED_S): `[mem] foot` lines that split the guest's pages by region, each region's resident
// anonymous, resident shared (placed Vulkan memory: the driver's pages over guest addresses; "file": shared or file pages outside placed memory), swapped (zram) and touched
// (the three together) MiB, and in brackets the change since the last line. Regions: every memory pool (vka_mem.c dm_gfx_pool_foot: its NVN flags, how much of it is
// placed, the images over its storage by content; scanned on its own range, so pools that overlap count in each), the heap outside pools ([0, current size)), the alias
// region's mapped pages outside pools, the modules (code and data; NROs lie in the heap or alias region), the stack region, and the rest of the block below the alias region.
// Then the pools grouped by flags, placement and what the host still reads of them (mem_foot.h dmf_use), the pools of at least DM_MEM_FOOT_MB (8) touched one line each
// (32 at most, the largest first), and the 10 pools with the most swapped out. The process's major and minor page faults (getrusage) since the last line, and the cost of
// the pass both lines share: wall and CPU time, pagemap bytes read (64 KiB a read; 8 bytes a 4 KiB page, so about 2 MiB a GiB of address scanned) and placed memory
// counted without a read. The pool table is sized from the live pool count (and NVN's list from its own); nothing is kept beyond the last line's values per live pool
__attribute__((weak)) int dm_gfx_pool_foot(dm_pool_foot *o, int max, dmf_counts *c) { (void)o; (void)max; *c = (dmf_counts){0}; return 0; }   // (vka_mem.c; none without the Vulkan replay)
enum { FOOT_NG = 48, FOOT_NREG = 5 };
static uint64_t ftch(const tch_t *t) { return t->res + t->shr + t->swp; }
static int foot_vals(char *b, size_t n, uint64_t res, uint64_t shr, uint64_t swp, uint64_t fil, int have, uint64_t ptch, uint64_t pswp) {   // MiB, and the change since the last line
    double M = 1048576.0; uint64_t t = res + shr + swp;
    int o = snprintf(b, n, "res %.0f shr %.0f swp %.0f touched %.0f", res / M, shr / M, swp / M, t / M);
    if (fil && o >= 0 && (size_t)o < n) o += snprintf(b + o, n - (size_t)o, " (file %.0f)", fil / M);
    if (have && o >= 0 && (size_t)o < n) o += snprintf(b + o, n - (size_t)o, " [%+.0f, swp %+.0f]", ((double)t - (double)ptch) / M, ((double)swp - (double)pswp) / M);
    return o;
}
static void foot_imgs(char *b, size_t n, const dm_pool_foot *p) {
    double M = 1048576.0;
    snprintf(b, n, "images fill %u (%.0f), track %u (%.0f), gpu %u (%.0f), levels left out %u (%.0f) MiB of storage, %.0f MiB image memory", p->n_fill, p->s_fill / M, p->n_track,
             p->s_track / M, p->n_gpu, p->s_gpu / M, p->n_drop, p->s_drop / M, p->img / M);
}
static int foot_pst(const dm_pool_foot *p) { return p->placed >= p->size && p->size ? 2 : p->placed ? 1 : 0; }   // placed: 0 none, 1 parts, 2 whole
static const char *foot_pname(int s) { return s == 2 ? "placed" : s == 1 ? "parts placed" : "not placed"; }
static void foot_pool_line(const char *what, const dm_pool_foot *p, const dmf_prev *pv) {
    char fl[64], v[192], im[256]; double M = 1048576.0;
    dmf_flags_str(p->flags, (int)p->fl_known, fl, sizeof fl); foot_vals(v, sizeof v, p->res, p->shr, p->swp, 0, pv != NULL, pv ? pv->tch : 0, pv ? pv->swp : 0); foot_imgs(im, sizeof im, p);
    LOG("[mem] foot %s: pool %llu (Initialize %llu) %#llx+%.1f MiB %s, %s %.0f MiB: %s; %s; host: %s", what, (unsigned long long)p->id, (unsigned long long)p->gen,
        (unsigned long long)p->lo, p->size / M, fl, foot_pname(foot_pst(p)), p->placed / M, v, im, dmf_use_name(dmf_use(p)));
}
typedef struct { double wall_ms, cpu_ms; uint64_t pm, skip; } scan_cost;
static void foot_print(dm_pool_foot *P, int np, const dmf_counts *fc, const tch_t *R, const scan_cost *sc, int pm_ok) {
    static dmf_prev *prev; static int nprev;
    // faults
    static struct rusage ru0; static struct timespec t0; static int have; struct rusage ru; getrusage(RUSAGE_SELF, &ru);
    struct timespec w1; clock_gettime(CLOCK_MONOTONIC, &w1);
    double dt = have ? (w1.tv_sec - t0.tv_sec) + (w1.tv_nsec - t0.tv_nsec) / 1e9 : 0, M = 1048576.0;
    long dmaj = have ? ru.ru_majflt - ru0.ru_majflt : ru.ru_majflt, dmin = have ? ru.ru_minflt - ru0.ru_minflt : ru.ru_minflt;
    // the pools by flags, placement and host use
    struct grp { uint32_t flags, known; int pst, use, n; uint64_t size, res, shr, swp, placed; dm_pool_foot sum; };
    static struct grp G[FOOT_NG], GP[FOOT_NG]; static int ngp; int ng = 0, gover = 0, nunk = 0;
    dm_pool_foot all = {0};
    for (int i = 0; i < np; i++) { const dm_pool_foot *p = &P[i]; int ps = foot_pst(p), u = dmf_use(p), k = 0;
        all.res += p->res; all.shr += p->shr; all.swp += p->swp; all.size += p->size; all.placed += p->placed; nunk += !p->fl_known;
        while (k < ng && !(G[k].flags == p->flags && G[k].known == p->fl_known && G[k].pst == ps && G[k].use == u)) k++;
        if (k == ng) { if (ng == FOOT_NG) { k = FOOT_NG - 1; gover = 1; } else { memset(&G[ng], 0, sizeof G[ng]); G[ng].flags = p->flags; G[ng].known = p->fl_known; G[ng].pst = ps; G[ng].use = u; ng++; } }
        struct grp *g = &G[k]; g->n++; g->size += p->size; g->res += p->res; g->shr += p->shr; g->swp += p->swp; g->placed += p->placed;
        g->sum.n_fill += p->n_fill; g->sum.n_track += p->n_track; g->sum.n_gpu += p->n_gpu; g->sum.n_drop += p->n_drop;
        g->sum.s_fill += p->s_fill; g->sum.s_track += p->s_track; g->sum.s_gpu += p->s_gpu; g->sum.s_drop += p->s_drop; g->sum.img += p->img; }
    static tch_t RP[FOOT_NREG], AP; static int rhave;
    static const char *rn[FOOT_NREG] = {"heap outside pools", "alias mapped outside pools", "modules", "stack", "other below alias"};
    char line[1600]; size_t o = 0;
    for (int r = 0; r < FOOT_NREG && o < sizeof line; r++) {
        o += (size_t)snprintf(line + o, sizeof line - o, "%s%s ", r ? " | " : "", rn[r]);
        if (o < sizeof line) o += (size_t)foot_vals(line + o, sizeof line - o, R[r].res, R[r].shr, R[r].swp, R[r].fil, rhave, ftch(&RP[r]), RP[r].swp); }
    if (o < sizeof line) {
        if (np < fc->pools) o += (size_t)snprintf(line + o, sizeof line - o, " | %d of %d pools (table cut) %.0f MiB (%.0f placed) ", np, fc->pools, all.size / M, all.placed / M);
        else o += (size_t)snprintf(line + o, sizeof line - o, " | %d pools %.0f MiB (%.0f placed) ", np, all.size / M, all.placed / M); }
    if (o < sizeof line) o += (size_t)foot_vals(line + o, sizeof line - o, all.res, all.shr, all.swp, 0, rhave, AP.res + AP.shr + AP.swp, AP.swp);
    LOG("[mem] foot (MiB; resident anonymous, resident shared = placed, swapped; [change since the last line]%s): %s", pm_ok ? "" : "; pagemap not readable: resident only", line);
    LOG("[mem] foot faults: %ld major, %ld minor %s %.0f s (%.0f major/s); scan (with the touched line) %.1f ms, %.1f ms CPU, %.1f MiB of pagemap read, %.0f MiB placed not read; "
        "NVN flags for %d of %d live pools listed, %d pools without flags", dmaj, dmin, have ? "in" : "since the start, first line", dt, dt > 0 ? dmaj / dt : 0.0,
        sc->wall_ms, sc->cpu_ms, sc->pm / M, sc->skip / M, fc->nvn_listed, fc->nvn_live, nunk);
    for (int k = 0; k < ng; k++) { const struct grp *g = &G[k]; const struct grp *q = NULL; char fl[64], v[192], im[256];
        for (int j = 0; j < ngp && !q; j++) if (GP[j].flags == g->flags && GP[j].known == g->known && GP[j].pst == g->pst && GP[j].use == g->use) q = &GP[j];
        dmf_flags_str(g->flags, (int)g->known, fl, sizeof fl); foot_vals(v, sizeof v, g->res, g->shr, g->swp, 0, q != NULL, q ? q->res + q->shr + q->swp : 0, q ? q->swp : 0); foot_imgs(im, sizeof im, &g->sum);
        LOG("[mem] foot group %s, %s, host: %s: %d pools %.0f MiB (%.0f placed): %s; %s", fl, foot_pname(g->pst), dmf_use_name(g->use), g->n, g->size / M, g->placed / M, v, im); }
    if (gover) LOG("[mem] foot group: more than %d kinds: the last group holds the rest", FOOT_NG);
    int ix[32], nt;
    long mb = 8; { const char *e = getenv("DM_MEM_FOOT_MB"); if (e && *e) mb = atol(e); }
    if (mb > 0) {
        nt = dmf_top(P, np, 1, ix, 32);
        int shown = 0; for (int t = 0; t < nt; t++) if (dmf_touched(&P[ix[t]]) >= (uint64_t)mb << 20) { foot_pool_line("pool", &P[ix[t]], dmf_prev_find(prev, nprev, P[ix[t]].id)); shown++; }
        int big = 0; for (int i = 0; i < np; i++) big += dmf_touched(&P[i]) >= (uint64_t)mb << 20;
        if (big > shown) LOG("[mem] foot pool: %d more pools of %ld MiB touched and more (32 shown)", big - shown, mb);
    }
    nt = dmf_top(P, np, 0, ix, 10);
    for (int t = 0; t < nt; t++) { char w[32]; snprintf(w, sizeof w, "swapped #%d", t + 1); foot_pool_line(w, &P[ix[t]], dmf_prev_find(prev, nprev, P[ix[t]].id)); }
    // the values the next line compares with
    dmf_prev *nv = np ? malloc((size_t)np * sizeof *nv) : NULL;
    if (nv || !np) { for (int i = 0; i < np; i++) nv[i] = (dmf_prev){P[i].id, dmf_touched(&P[i]), P[i].res, P[i].swp};
        if (np) qsort(nv, (size_t)np, sizeof *nv, dmf_prev_cmp); free(prev); prev = nv; nprev = np; }
    memcpy(RP, R, sizeof RP); AP = (tch_t){all.res, all.shr, all.swp, 0, 0}; rhave = 1;
    memcpy(GP, G, sizeof GP); ngp = ng;
    ru0 = ru; t0 = w1; have = 1;
}
static void foot_add_iv(dmf_iv **iv, int *cap, int *n, uint64_t lo, uint64_t hi) {
    if (*n == *cap && !grow((void **)iv, cap, sizeof **iv, *n + 1)) return;
    (*iv)[(*n)++] = (dmf_iv){lo, hi};
}
// The touched line, and with foot the footprint lines, from one pagemap pass
static void mem_scan_locked(const char *why, int foot);
static void mem_scan(const char *why, int foot) {   // one scan at a time: the exit line's (another thread) would race the periodic one over the static, realloc'd tables
    static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_lock(&mu); mem_scan_locked(why, foot); pthread_mutex_unlock(&mu);
}
static void mem_scan_locked(const char *why, int foot) {
    struct timespec w0, c0, w1, c1; clock_gettime(CLOCK_MONOTONIC, &w0); clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c0);
    static uint64_t *PLr, *Ur; static int plcap, ucap;
    static dm_pool_foot *P; static tch_t *PT; static dmf_iv *IV; static int pcap, ptcap, ivcap;
    // the tables (each under its own lock, before the scan's)
    int npl_all, nu_all, npl, nu, np = 0; dmf_counts fc = {0};
    npl = ranges_get(dm_gfx_placed_ranges, &PLr, &plcap, &npl_all);
    nu = ranges_get(dm_gfx_unplaced_pools, &Ur, &ucap, &nu_all);
    if (foot) {
        np = dm_gfx_pool_foot(P, pcap, &fc);
        for (int t = 0; t < 2 && fc.pools > pcap; t++) { if (!grow((void **)&P, &pcap, sizeof *P, fc.pools)) break; np = dm_gfx_pool_foot(P, pcap, &fc); }
        if (np > 0 && !grow((void **)&PT, &ptcap, sizeof *PT, np)) { LOG("[RED] [mem] foot: no memory for the pool counters: no foot lines"); foot = 0; }
    }
    pthread_mutex_lock(&g_tch_lock);
    if (g_pm_fd == -2) { g_pm_fd = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC); if (g_pm_fd < 0) g_pm_err = errno; }
    uint64_t pm0 = g_pm_read, sk0 = g_pm_skip;
    static scan_plan S; S.n = 0; S.fail = 0;
    // the touched line's ranges
    tch_t h = {0}, al = {0}, pl = {0}; uint64_t hcur = atomic_load(&g_heap_cur), amap = 0, ppl = 0; double G = 1073741824.0;
    if (g_heap && hcur) sp_add(&S, (uint64_t)(uintptr_t)g_heap, hcur, &h);
    if (g_alias && g_alias_bits) {   // runs of non-empty bitmap words (256 KiB each)
        uint64_t nw = (g_alias_cap >> 18) + 1, run = 0; int in = 0;
        for (uint64_t w = 0; w <= nw; w++) {
            uint64_t x = w < nw ? atomic_load(&g_alias_bits[w]) : 0;
            amap += (uint64_t)__builtin_popcountll(x) * 4096;
            if (x && !in) { in = 1; run = w; }
            else if (!x && in) { in = 0; uint64_t a = run << 18, e = w << 18 < g_alias_cap ? w << 18 : g_alias_cap; if (e > a) sp_add(&S, (uint64_t)(uintptr_t)g_alias + a, e - a, &al); }
        }
    }
    for (int i = 0; i < nu; i++) { sp_add(&S, Ur[2 * i], Ur[2 * i + 1], &pl); ppl += Ur[2 * i + 1]; }
    // the footprint's: every pool on its own range, then the regions outside the pools' union
    tch_t R[FOOT_NREG]; memset(R, 0, sizeof R); int ivn = 0;
    if (foot) {
        for (int i = 0; i < np; i++) { PT[i] = (tch_t){0}; sp_add(&S, P[i].lo, P[i].size, &PT[i]); foot_add_iv(&IV, &ivcap, &ivn, P[i].lo, P[i].lo + P[i].size); }
        int npiv = dmf_merge(IV, ivn);   // the pools' union: the heap and alias remainders leave it out
        sp_ctx c0x = {&S, &R[0]}, c1x = {&S, &R[1]}, c4x = {&S, &R[4]};
        if (g_heap && hcur) dmf_gaps((uint64_t)(uintptr_t)g_heap, (uint64_t)(uintptr_t)g_heap + hcur, IV, npiv, sp_gap, &c0x);
        if (g_alias && g_alias_bits) {   // the alias runs again, outside pools
            uint64_t nw = (g_alias_cap >> 18) + 1, run = 0; int in = 0;
            for (uint64_t w = 0; w <= nw; w++) {
                uint64_t x = w < nw ? atomic_load(&g_alias_bits[w]) : 0;
                if (x && !in) { in = 1; run = w; }
                else if (!x && in) { in = 0; uint64_t a = run << 18, e = w << 18 < g_alias_cap ? w << 18 : g_alias_cap;
                    if (e > a) dmf_gaps((uint64_t)(uintptr_t)g_alias + a, (uint64_t)(uintptr_t)g_alias + e, IV, npiv, sp_gap, &c1x); }
            }
        }
        ivn = npiv;
        for (int i = 0; i < g_nmod; i++) if (g_mod[i].base && g_mod[i].size && !g_mod[i].nro) {   // (an NRO lives in memory the game allocated: heap or alias)
            sp_add(&S, (uint64_t)(uintptr_t)g_mod[i].base, g_mod[i].size, &R[2]);
            foot_add_iv(&IV, &ivcap, &ivn, (uint64_t)(uintptr_t)g_mod[i].base, (uint64_t)(uintptr_t)g_mod[i].base + g_mod[i].size); }
        if (g_stack) sp_add(&S, (uint64_t)(uintptr_t)g_stack, g_stack_cap, &R[3]);
        if (g_as && g_alias > g_as) { int nx = dmf_merge(IV, ivn); dmf_gaps((uint64_t)(uintptr_t)g_as, (uint64_t)(uintptr_t)g_alias, IV, nx, sp_gap, &c4x); }
    }
    int ok = !S.fail && sp_run(&S, (const dmf_iv *)PLr, npl);
    scan_cost sc = {0, 0, g_pm_read - pm0, g_pm_skip - sk0}; int pm_ok = g_pm_fd >= 0, pm_err = g_pm_err;
    pthread_mutex_unlock(&g_tch_lock);
    clock_gettime(CLOCK_MONOTONIC, &w1); clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c1);
    sc.wall_ms = ((w1.tv_sec - w0.tv_sec) * 1e9 + (w1.tv_nsec - w0.tv_nsec)) / 1e6; sc.cpu_ms = ((c1.tv_sec - c0.tv_sec) * 1e9 + (c1.tv_nsec - c0.tv_nsec)) / 1e6;
    if (!ok) { LOG("[RED] [mem] touched (%s): no memory for the scan's tables (%d ranges): no line", why, S.n); return; }
    // the touched line
    char line[1280]; size_t o = 0;
    o += (size_t)tch_str(line + o, sizeof line - o, "heap", &h, hcur);
    o += (size_t)snprintf(line + o, sizeof line - o, " | alias mapped %.2f, runs", amap / G); o += (size_t)tch_str(line + o, sizeof line - o, "", &al, al.all);
    if (nu < nu_all) o += (size_t)snprintf(line + o, sizeof line - o, " | pools not placed (%d of %d ranges: table cut)", nu, nu_all);
    else o += (size_t)snprintf(line + o, sizeof line - o, " | pools not placed (%d)", nu);
    o += (size_t)tch_str(line + o, sizeof line - o, "", &pl, ppl);
    FILE *f = fopen("/sys/block/zram0/mm_stat", "r"); unsigned long long zo = 0, zc = 0, zm = 0;
    if (f && fscanf(f, "%llu %llu %llu", &zo, &zc, &zm) == 3) o += (size_t)snprintf(line + o, sizeof line - o, " | zram orig %.2f, compr %.2f, mem_used %.2f (ratio %.2f)", zo / G, zc / G, zm / G, zc ? (double)zo / zc : 0.0);
    else o += (size_t)snprintf(line + o, sizeof line - o, " | zram mm_stat not readable (errno %d)", f ? 0 : errno);
    if (f) fclose(f);
    if ((f = fopen("/proc/meminfo", "r"))) { char ln[128]; long st = -1, sf = -1;
        while (fgets(ln, sizeof ln, f)) { sscanf(ln, "SwapTotal: %ld kB", &st); sscanf(ln, "SwapFree: %ld kB", &sf); }
        fclose(f); if (st >= 0 && sf >= 0 && o < sizeof line) o += (size_t)snprintf(line + o, sizeof line - o, " | swap used %.2f of %.2f", (st - sf) / 1048576.0, st / 1048576.0); }
    if (o < sizeof line) o += (size_t)snprintf(line + o, sizeof line - o, " | scan%s %.1f ms, %.1f ms CPU, %.1f MiB of pagemap read, placed %.2f not read (%d%s ranges)",
                                               foot ? " (with the foot lines)" : "", sc.wall_ms, sc.cpu_ms, sc.pm / 1048576.0, sc.skip / G, npl, npl < npl_all ? " listed, more not" : "");
    if (!pm_ok) LOG("[mem] touched (GiB, %s; pagemap not readable, errno %d: mincore, swapped not told from never): %s", why, pm_err, line);
    else LOG("[mem] touched (GiB, %s): %s", why, line);
    if (!foot) return;
    for (int i = 0; i < np; i++) { P[i].res = PT[i].res; P[i].shr = PT[i].shr; P[i].swp = PT[i].swp; P[i].all = PT[i].all; }
    foot_print(P, np, &fc, R, &sc, pm_ok);
}
static int g_foot_on;
static int g_tch_on;
void dm_mem_touched_exit(void) { static _Atomic int once; if (g_tch_on && !atomic_exchange(&once, 1)) mem_scan("exit", 0); }
static void touched_atexit(void) { dm_mem_touched_exit(); }
static void *touched_thread(void *arg) {
    int per = (int)(intptr_t)arg; g_tch_on = 1; atexit(touched_atexit);
    const char *fe = getenv("DM_MEM_FOOT"); g_foot_on = fe && atoi(fe) > 0;
    for (;;) { sleep((unsigned)per); mem_scan("periodic", g_foot_on); }
    return NULL;
}
#else
static void *touched_thread(void *arg) { (void)arg; return NULL; }
void dm_mem_touched_exit(void) {}
#endif

// ---- Address regions ----
// The SDK's VammManager hands out the alias region as address regions. The rewriter's observation gates in the sdk (patch_nso.py "Address regions", placed by its symbols in any
// build) show each one: its size, where the request returns to in the game (the first return address outside the sdk on the frame chain: the function that asked, or its caller
// when it tail-called the SDK API, whose frame is gone by then), the SDK API it came through (such as StandardAllocator) and its address. With the page bitmaps that gives how much of each region a game uses, which decides how its reservations can fit an iPhone's alias region.
// A region of 256 MB or more is logged when made; DM_REGIONS=<file> writes the live table every 2 s (the NSP script's heap check reads it, with a Switch-sized DM_ALIAS_SIZE)
typedef struct { uint64_t off, size, via; char who[40]; } region_t;   // via: the outermost sdk return address on the way (the SDK API the request came through)
static region_t *g_rgn; static int g_nrgn, g_rgn_cap; static pthread_mutex_t g_rgn_lock = PTHREAD_MUTEX_INITIALIZER;   // grows as needed: Unity 6 takes all its memory as regions
static _Thread_local uint64_t t_rgn_size, t_rgn_via; static _Thread_local char t_rgn_who[40];
// Where a request at an sdk function's entry gate returns to in the game: buf gets "main+0x…", *via the outermost sdk return address on the way (0 = none)
static void guest_caller(const uint64_t *r, char *buf, size_t n, uint64_t *via) {
    // at the function's entry the return address is the x30 the gate block pushed (at the block's sp, regs[99]: x29, x30; entry.S), x30 itself holds the gate's own
    uint64_t lr = ((const uint64_t *)(uintptr_t)r[99])[1], fp = r[29]; module_t *m = module_of((void *)(uintptr_t)lr);
    *via = 0;
    for (int i = 0; i < 32 && m && !strcmp(m->name, "sdk"); i++) {
        *via = lr - (uint64_t)(uintptr_t)m->base;   // AArch64 frame records: [fp] = the caller's fp, [fp + 8] = its return address
        uint64_t rec[2];   // read safely: static constructors run on the host's main thread, whose stack is outside the guest block
        if (!fp || (fp & 15) || plat_mem_read(fp, rec, sizeof rec)) break;
        lr = rec[1]; fp = rec[0]; m = module_of((void *)(uintptr_t)lr);
    }
    if (m) snprintf(buf, n, "%s+%#llx", m->name, (unsigned long long)(lr - (uint64_t)(uintptr_t)m->base));
    else snprintf(buf, n, "%#llx", (unsigned long long)lr);
}
static void region_ask(uint64_t *r, int param) {
    (void)param;
    t_rgn_size = r[2]; guest_caller(r, t_rgn_who, sizeof t_rgn_who, &t_rgn_via);
}
static void region_got(uint64_t *r, int param) {
    mem_init();
    uint64_t a = param >= 0 && param < 31 ? r[param] : 0, ab = (uint64_t)(uintptr_t)g_alias;
    if (a < ab || a - ab >= g_alias_cap) return;
    pthread_mutex_lock(&g_rgn_lock);
    if (g_nrgn == g_rgn_cap) {
        region_t *t = realloc(g_rgn, sizeof *t * (size_t)(g_rgn_cap ? 2 * g_rgn_cap : 256));
        if (t) { g_rgn = t; g_rgn_cap = g_rgn_cap ? 2 * g_rgn_cap : 256; }
    }
    int kept = g_nrgn < g_rgn_cap;
    if (kept) { g_rgn[g_nrgn].off = a - ab; g_rgn[g_nrgn].size = t_rgn_size; g_rgn[g_nrgn].via = t_rgn_via; memcpy(g_rgn[g_nrgn].who, t_rgn_who, sizeof t_rgn_who); g_nrgn++; }
    pthread_mutex_unlock(&g_rgn_lock);
    if (!kept) { static _Atomic int once; if (!atomic_fetch_add(&once, 1)) LOG("[RED] address region table: out of memory at %d regions, later ones are not in it", g_nrgn); }
    if (t_rgn_size >= 0x10000000) LOG("[mem] address region %.2f GB at alias +%.2f GB, returning to %s", t_rgn_size / 1073741824.0, (a - ab) / 1073741824.0, t_rgn_who);
}
static void region_free(uint64_t *r, int param) {
    (void)param;
    uint64_t off = r[1] - (uint64_t)(uintptr_t)g_alias;
    pthread_mutex_lock(&g_rgn_lock);
    for (int i = 0; i < g_nrgn; i++) if (g_rgn[i].off == off) { g_rgn[i] = g_rgn[--g_nrgn]; break; }
    pthread_mutex_unlock(&g_rgn_lock);
}
// ---- StandardAllocator's address space ----
// With virtual address memory, StandardAllocator::Initialize(nullptr, size) takes a region of size from the alias region and lays its heap out over all of it (CentralHeap), and
// games pass sizes made for the Switch's 64 GB alias region: Pokémon Legends: Z-A 12 GB, Pokémon Scarlet 12 GB, Paper Mario: The Thousand-Year Door 62 GB, Absolum 40 GB. Where the
// size is more than the alias region has room for, the request would fail (an nn::diag abort in nninitStartup), so the gate at the function's entry (patch_nso.py) gives it the room
// instead: the alias region less the regions already made (at least the SDK's 62 MB node area) and 256 MB for regions made later, in 32 MB steps; the profile key stdalloc_cap sets
// it instead (Absolum's .NET garbage collector takes 1.5 GB after it). The SDK lays the region out from that size, so it holds; the pool, not the region, bounds what the heap
// can map. A workaround for the iPhone's address space, not the game: with room for the request (a Switch-sized DM_ALIAS_SIZE) nothing changes. Found by symbol in every sdk
#define STDALLOC_LATER 0x10000000ull
// What the alias region has room for in one more region: past the end of the regions made so far, less STDALLOC_LATER for those made later, in 32 MB steps (0 = none)
static uint64_t alias_room(void) {
    mem_init();
    uint64_t reach = 0x3e00000;   // the node area with the SDK-wide cut (sdk_patches.py); a build without it keeps 1 GB, and a 12 GB request fails there as before
    pthread_mutex_lock(&g_rgn_lock);
    for (int i = 0; i < g_nrgn; i++) if (g_rgn[i].off + g_rgn[i].size > reach) reach = g_rgn[i].off + g_rgn[i].size;
    pthread_mutex_unlock(&g_rgn_lock);
    return g_alias_cap > reach + STDALLOC_LATER ? (g_alias_cap - reach - STDALLOC_LATER) & ~0x1ffffffull : 0;
}
static void stdalloc_init(uint64_t *r, int param) {
    (void)param;
    if (r[1]) return;   // a heap over memory the game passes in: no address region
    uint64_t cap = DM_GAME_STDALLOC_CAP ? DM_GAME_STDALLOC_CAP : alias_room();
    if (!cap || r[2] <= cap) return;
    char who[40]; uint64_t via; guest_caller(r, who, sizeof who, &via);
    LOG("[mem] StandardAllocator::Initialize(nullptr, %.2f GB) returning to %s: %.2f GB of address space, %s", r[2] / 1073741824.0, who, cap / 1073741824.0,
        DM_GAME_STDALLOC_CAP ? "the profile's stdalloc_cap" : "what the alias region has room for (the profile key stdalloc_cap sets it)");
    r[2] = cap;
}
// ---- Unreal's engine region ----
// Unreal Engine 4 / 5's nninitStartup reserves one address region for the engine, which the engine hands out itself: 62 GB by default (16 GB in Unreal Engine 4.18), made for the
// Switch's 64 GB alias region. The rewriter's gate on the store of its size right before the reservation (dm/tools/ue_region.py, in any Unreal main) gives it what the alias region
// has room for, as StandardAllocator's gate above: the size variable (the register the gate number names) and the request (x1) both take it, so the engine lays the region out from
// the size it got. -SwitchVirtualAddressSpaceSize= gives a size too, but UE 4.26 then skips the reservation (Pikmin 4); a smaller size given that way stays. A workaround for the
// iPhone's address space, not the game: with room for the request (a Switch-sized DM_ALIAS_SIZE) nothing changes
static void ue_region(uint64_t *r, int param) {
    if (param < 1 || param > 28 || param == 18) return;
    uint64_t cap = alias_room();
    if (!cap || r[param] <= cap) return;
    LOG("[mem] Unreal's engine region: %.2f GB asked, %.2f GB of address space, what the alias region has room for", r[param] / 1073741824.0, cap / 1073741824.0);
    r[param] = r[1] = cap;
}
__attribute__((constructor)) static void regions_attach(void) {
    gate_obs_set(OBS_REGION_ASK, region_ask); gate_obs_set(OBS_REGION_GOT, region_got); gate_obs_set(OBS_REGION_FREE, region_free);
    gate_obs_set(OBS_STDALLOC_INIT, stdalloc_init); gate_obs_set(OBS_UE_REGION, ue_region);
}
// Pages set in a bitmap's bits [lo, hi), a word at a time: a 62 GB region is 16 million pages, counted under g_map_lock, which every MapPhysicalMemory takes
static uint64_t bits_set(_Atomic uint64_t *b, uint64_t lo, uint64_t hi) {
    uint64_t n = 0;
    for (; lo < hi && (lo & 63); lo++) n += atomic_load(&b[lo >> 6]) >> (lo & 63) & 1;
    for (; lo + 64 <= hi; lo += 64) n += (uint64_t)__builtin_popcountll(atomic_load(&b[lo >> 6]));
    for (; lo < hi; lo++) n += atomic_load(&b[lo >> 6]) >> (lo & 63) & 1;
    return n;
}
// The first and last bit set in [lo, hi) (-1 when none), skipping empty words
static void bits_span(const uint64_t *b, uint64_t lo, uint64_t hi, int64_t *first, int64_t *last) {
    *first = *last = -1;
    for (uint64_t i = lo; i < hi; ) {
        uint64_t w = b[i >> 6] >> (i & 63);
        if (!w) { i = (i | 63) + 1; continue; }
        i += (uint64_t)__builtin_ctzll(w);
        if (i >= hi) break;
        if (*first < 0) *first = (int64_t)i;
        *last = (int64_t)i; i++;
    }
}
// The table: alias offset, size, where the request returns to, bytes mapped now (4 KB pages), the part ever opened (host pages: from its first to its last, region-relative),
// the sdk offset it came through
static void *regions_thread(void *arg) {
    const char *path = arg; char tmp[1024]; snprintf(tmp, sizeof tmp, "%s.partial", path);
    uint64_t hpg = (uint64_t)getpagesize();
    for (;;) {
        FILE *f = fopen(tmp, "w");
        if (f) {
            fprintf(f, "# alias region %#llx; offset\tsize\treturns to\tmapped\tfirst opened\tend of the last opened (region-relative; - = none)\tvia (sdk offset)\n", (unsigned long long)g_alias_cap);
            pthread_mutex_lock(&g_rgn_lock); int n = g_nrgn; region_t *t = malloc(sizeof *t * (size_t)(n ? n : 1)); if (t && n) memcpy(t, g_rgn, sizeof *t * (size_t)n); pthread_mutex_unlock(&g_rgn_lock);
            pthread_mutex_lock(&g_map_lock);
            for (int i = 0; t && i < n; i++) {
                uint64_t lo = t[i].off, hi = t[i].off + t[i].size > g_alias_cap ? g_alias_cap : t[i].off + t[i].size, mapped = bits_set(g_alias_bits, lo >> 12, hi >> 12);
                int64_t first, last; bits_span(g_alias_exposed, lo / hpg, (hi + hpg - 1) / hpg, &first, &last);
                char fo[24] = "-", lo2[24] = "-";
                if (first >= 0) { snprintf(fo, sizeof fo, "%#llx", (unsigned long long)((uint64_t)first * hpg > lo ? (uint64_t)first * hpg - lo : 0));
                                  snprintf(lo2, sizeof lo2, "%#llx", (unsigned long long)((uint64_t)(last + 1) * hpg - lo)); }
                fprintf(f, "%#llx\t%#llx\t%s\t%#llx\t%s\t%s\t%#llx\n", (unsigned long long)t[i].off, (unsigned long long)t[i].size, t[i].who, (unsigned long long)mapped * 4096, fo, lo2,
                        (unsigned long long)t[i].via);
            }
            pthread_mutex_unlock(&g_map_lock); free(t);
            fclose(f); rename(tmp, path);
        }
        sleep(2);
    }
    return NULL;
}

int svc_thread(int num, uint64_t *regs);   // hle_thread.c: thread and synchronization kind

uint64_t svc_dispatch(int num, uint64_t *regs, uint64_t orig_addr) {
    { void hlecount_svc(int); hlecount_svc(num); }   // DM_HLECOUNT: svc counts by number
    mem_init();
    if (svc_thread(num, regs)) return regs[0];
    switch (num) {
    case 0x01: {  // SetHeapSize(out_addr*, size)  → x1 = heap base
        size_t size = regs[1];
        // Larger than the heap region: ResultOutOfMemory (0xD001), as Horizon's KPageTable::SetHeapSize returns. nn::os::SetMemoryHeapSize turns only that, 103 and LimitReached into
        // os::ResultOutOfMemory and aborts on anything else; Donkey Kong Country: Tropical Freeze finds its heap size by a binary search between 3 and 6 GB, which needs the error back
        if (size > g_heap_cap) { LOG("SetHeapSize OOM: req %#zx > cap %#zx", size, g_heap_cap); regs[0] = 0xD001; return 0; }
        pthread_mutex_lock(&g_map_lock);
        size_t previous = g_heap_cur, image = image_bytes();
        if (size < previous && map_source_busy((uintptr_t)g_heap + size, previous - size)) {
            pthread_mutex_unlock(&g_map_lock); regs[0] = 0xD401; return 0;
        }
        if (image > APP_POOL || size > APP_POOL - image) {   // ResultLimitReached; games retry in 2 MB decrements
            // Logged once (Mario's retries would repeat it): a game that sizes its heap as pool − used memory and does not retry stops here when its modules are larger than the used memory reported
            // (profile mem_used; the NSP script's probe boot reads this line, and sets mem_used when no SetHeapSize succeeds afterwards)
            static int once; if (!once++) LOG("[mem] SetHeapSize %#zx + modules %#zx > app pool %#llx (used memory reported %#llx + bss %#zx)", size, image_bytes() - big_bss_bytes(),
                                              (unsigned long long)APP_POOL, (unsigned long long)DM_GAME_MEM_USED, big_bss_bytes());
            pthread_mutex_unlock(&g_map_lock); regs[0] = 0x10801; return 0;
        }
        uint64_t charged = (uint64_t)DM_GAME_MEM_USED + big_bss_bytes() + atomic_load(&g_alias_mapped);
        if (DM_GAME_MEM_MAP_LIMIT && (charged > APP_POOL || size > APP_POOL - charged)) {
            pthread_mutex_unlock(&g_map_lock); regs[0] = 0x10801; return 0;
        }
        int rc = 0;
        if (size > g_heap_cur) { dm_gfx_guest_open(g_heap + g_heap_cur, size - g_heap_cur); rc = mprotect(g_heap + g_heap_cur, size - g_heap_cur, PROT_READ | PROT_WRITE); }
        else if (size < g_heap_cur) (void)guest_release((uintptr_t)g_heap + size, (uintptr_t)g_heap + g_heap_cur);
        if (rc) die("SetHeapSize mprotect failed: req=%#zx errno=%d", size, errno);
        if (g_flag.svc) LOG("SetHeapSize %#zx -> base %p mprotect=%d errno=%d", size, g_heap, rc, errno);
        else if (size > g_heap_cur) LOG("[mem] heap %.2f GB (heap region %.2f GB)", size / 1073741824.0, g_heap_cap / 1073741824.0);
        g_heap_cur = size;
        pthread_mutex_unlock(&g_map_lock);
        regs[0] = 0; regs[1] = (uint64_t)g_heap; return 0;
    }
    case 0x02: case 0x03: { // Permission/attribute stubs must not modify a borrowed source.
        uint64_t addr = regs[0], size = regs[1];
        pthread_mutex_lock(&g_map_lock);
        int busy = size > UINT64_MAX - addr || (size && map_source_busy(addr, size));
        pthread_mutex_unlock(&g_map_lock);
        regs[0] = busy ? 0xD401 : 0; return 0;
    }
    case 0x04: {  // MapMemory(dst, src, size): when creating a thread the SDK moves the stack on the heap into the stack region
        uint64_t dst = regs[0], src = regs[1], size = regs[2];
        uint32_t result = map_arguments(dst, src, size);
        if (result) { regs[0] = result; return 0; }
        pthread_mutex_lock(&g_map_lock);
        if (map_find(dst, size) || map_source_busy(src, size) || !map_source_backed(src, size)) result = 0xD401;
        else if (g_nmap == MAX_MAP) result = 0xCE01;
        if (result) { pthread_mutex_unlock(&g_map_lock); regs[0] = result; LOG("[RED] MapMemory refused (%#x): %#llx <- %#llx +%#llx", result, (unsigned long long)dst, (unsigned long long)src, (unsigned long long)size); return 0; }
        // Apple pages are 16K, Horizon pages 4K: the accessible region is extended to a 16K boundary (the extra pages opened are unused pages of the same region, semantics unaffected)
        uint64_t mask = (size_t)getpagesize() - 1;
        uint64_t lo = dst & ~mask, hi = (dst + size + mask) & ~mask;
        if (mprotect((void *)lo, hi - lo, PROT_READ | PROT_WRITE)) die("MapMemory mprotect errno=%d", errno);
        dm_guest_write_prepare((void *)lo, hi - lo);
        memcpy((void *)dst, (void *)src, size);   // move semantics: the source region is not accessed while mapped (Horizon makes it inaccessible)
        map_add(dst, size, src, 0x0B /*MappedMemory*/);
        pthread_mutex_unlock(&g_map_lock);
        if (g_flag.svc) LOG("MapMemory %#llx <- %#llx +%#llx", dst, src, size);
        regs[0] = 0; return 0;
    }
    case 0x05: {  // UnmapMemory(dst, src, size): contents are moved back to the source region; a 16K page may be shared with an adjacent mapping, permissions are not revoked
        uint64_t dst = regs[0], src = regs[1], size = regs[2];
        uint32_t result = map_arguments(dst, src, size);
        if (result) { regs[0] = result; return 0; }
        pthread_mutex_lock(&g_map_lock);
        int k = map_find(dst, size);
        if (!k || g_map[k - 1].base != dst || g_map[k - 1].size != size || g_map[k - 1].src != src) {
            pthread_mutex_unlock(&g_map_lock); LOG("UnmapMemory does not match a mapping %#llx+%#llx", dst, size); regs[0] = 0xD401; return 0; }
        dm_guest_write_prepare((void *)src, size);
        memcpy((void *)src, (void *)dst, size);
        map_del(k - 1);
        pthread_mutex_unlock(&g_map_lock);
        regs[0] = 0; return 0;
    }
    case 0x06: {  // QueryMemory(MemoryInfo* out, PageInfo* pout, addr) → x1 = pageinfo
        uint64_t mi = regs[0], addr = regs[2];
        // MemoryInfo: base(0) size(8) state(0x10) attr(0x14) perm(0x18) ...
        uint8_t *a = (uint8_t *)addr;
        uint64_t base, size, state, perm;
        uint32_t attr = 0;
        int k;
        pthread_mutex_lock(&g_map_lock);   // the registry branches read g_map; the other branches are cheap, so the lock simply spans the whole chain
        uint64_t ab = (uintptr_t)g_alias, ae = ab + g_alias_cap, rb = 0, rn = 0;
        int mapped = 0, in_alias = addr >= ab && addr < ae;
        if (in_alias) alias_query(addr, &rb, &rn, &mapped);
        if (a >= g_heap && a < g_heap + g_heap_cur) { base=(uint64_t)g_heap; size=g_heap_cur; state=5/*Heap*/; perm=3; }
        else if ((k = map_find(addr, 1))) { base=g_map[k-1].base; size=g_map[k-1].size; state=g_map[k-1].state; perm=3; }
        else if (in_alias && mapped) { base=rb; size=rn; state=5; perm=3; }
        else if (a >= g_as && a < g_as + g_reserve) {   // free: up to the adjacent mapping/heap boundary
            uint64_t lo = (uint64_t)g_as, hi = (uint64_t)g_as + g_reserve;
            if (a >= g_heap) lo = (uint64_t)g_heap + g_heap_cur;
            else if (hi > (uint64_t)g_heap) hi = (uint64_t)g_heap;
            if (in_alias) {
                if (rb > ab && rb > lo) lo = rb;
                if (rb + rn < ae && rb + rn < hi) hi = rb + rn;
            } else if (g_alias_cap && addr < ab) {
                alias_query(ab, &rb, &rn, &mapped);
                uint64_t edge = mapped ? ab : rb + rn;
                if ((mapped || edge < ae) && edge < hi) hi = edge;
            } else if (g_alias_cap) {
                alias_query(ae - 1, &rb, &rn, &mapped);
                uint64_t edge = mapped ? ae : rb;
                if ((mapped || edge > ab) && edge > lo) lo = edge;
            }
            for (int i = 0; i < g_nmap; i++) {
                uint64_t e = g_map[i].base + g_map[i].size;
                if (e <= addr && e > lo) lo = e;
                if (g_map[i].base > addr && g_map[i].base < hi) hi = g_map[i].base;
            }
            base=lo; size=hi-lo; state=0; perm=0; }
        else if (module_of(a)) {
            // Loaded module: report the regions in Horizon's layout -- code [0, ro) Code+RX, read-only [ro, data) Code+R, data and bss [data, patch area) CodeData+RW.
            // The SDK's C++ exception unwinding (nn::ro::detail::GetExceptionInfo) walks back from the PC over "Code+RX" regions to find the module start, forward over Code then CodeData to find its end, then reads MOD0 at the module start for .eh_frame_hdr.
            // Previously "the 4 KiB page around it, Code+RX" was reported: the backward walk ran on into other memory, found no MOD0 and aborted (Engage's IL2CPP throws a C++ exception for every managed exception).
            // The rewriter's patch area is reported as CodeData+RX: modules sit back to back in the host, and an executable patch area reported as Code+RX would merge the previous module into the backward walk
            // Modules with the patch area before the base (patch_off < 0): the patch area is reported alone as CodeData+RX, and data and bss run to the end of the image, or to a back patch area
            // (patch_nso.py TWO), which is reported as CodeData+RX like a patch area after the image
            module_t *mm = module_of(a); uint64_t off = (uint64_t)(a - mm->base), mb = (uint64_t)mm->base;
            uint64_t ro = mm->ro_off ? mm->ro_off : mm->text_end, da = mm->data_off ? mm->data_off : ro, pa = mm->patch_off > 0 ? (uint64_t)mm->patch_off : mm->back_len ? mm->back_off : mm->size;
            if (a < mm->base)  { base = mb + mm->patch_off; size = mm->patch_len; state = 4/*CodeData*/; perm = 5; }
            else if (off < ro) { base = mb;      size = ro;            state = 3/*Code*/;     perm = 5; }
            else if (off < da) { base = mb + ro; size = da - ro;       state = 3/*Code*/;     perm = 1; }
            else if (off < pa) { base = mb + da; size = pa - da;       state = 4/*CodeData*/; perm = 3; }
            else               { base = mb + pa; size = mm->size - pa; state = 4/*CodeData*/; perm = 5; }
            if (state == 4 && perm == 3) {
                uint64_t lo = base - mb, hi = lo + size;
                module_data_query(mm, off, &lo, &hi, &perm);
                base = mb + lo; size = hi - lo;
            }
        }
        else {
            // The address lies in "the current host thread's stack": report the whole stack (state 0xB = stack). When the SDK initializes the main thread record it queries here with the sp at that moment and takes the reported range as the main thread stack;
            // later nn::os::GetCurrentStackInfo asserts that sp is still inside that range and aborts if not. Previously every address outside the guest address space was reported as "the 4 KiB page containing it",
            // so the main thread stack was recorded as one page. Whether the sp at init and a later sp share a page depends on the absolute position of the host stack top, which moves with the total length of environment variables / launch arguments (a different environment occasionally aborts at startup).
            uint64_t top, ssz; plat_thread_stack(&top, &ssz);
            if (addr < top && addr >= top - ssz) { base = top - ssz; size = ssz; state = 0xB/*Stack*/; perm = 3; }
            // Other host memory: report "free, no permissions" (the 4 KiB page around it) as Horizon does for unmapped addresses. It used to report Code+RX, and GetExceptionInfo walking back from main merged on through host memory down to 0
            else { base=addr & ~0xFFFull; size=0x1000; state=0/*Free*/; perm=0; } }
        map_source_query(addr, &base, &size, &attr, &perm);
        pthread_mutex_unlock(&g_map_lock);
        if (mi) {
            struct { uint64_t base, size; uint32_t state, attr, perm, ipc, device, reserved; }
                out = {base, size, (uint32_t)state, attr, (uint32_t)perm, 0, 0, 0};
            _Static_assert(sizeof out == 40, "MemoryInfo ABI size");
            memcpy((void *)(uintptr_t)mi, &out, sizeof out);
        }
        regs[0] = 0; regs[1] = 0; return 0;
    }
    case 0x2C: case 0x2D: {  // MapPhysicalMemory / UnmapPhysicalMemory(addr, size): only allowed inside the alias region (used by the SDK's VammManager)
        uintptr_t a = regs[0]; size_t size = regs[1];
        if (!size || ((a | size) & 0xfff) || !memory_contains((uintptr_t)g_alias, g_alias_cap, a, size)) {
            LOG("SVC %#x outside the alias region %p+%#zx", num, (void *)a, size); regs[0] = 0xD401; return 0;
        }
        pthread_mutex_lock(&g_map_lock);
        // The SDK extends committed windows across borrowed stack sources
        // (Raging Loop, StandardAllocator). Recommit preserves every live byte;
        // only decommit can invalidate backing required for eventual copyback.
        if (num == 0x2D && map_source_busy(a, size)) { pthread_mutex_unlock(&g_map_lock); regs[0] = 0xD401; return 0; }
        uint64_t off = a - (uintptr_t)g_alias;
        int64_t need = alias_unmapped(off, size) * 4096;
        // Profile mem_map_limit: a map that would take the process past the application pool fails with ResultLimitReached, as on Horizon (the pool less the used memory GetInfo reports).
        // Pokémon Legends: Z-A's nninitStartup allocates and frees everything allocatable (pool − 64 MB, mapped at the top of its StandardAllocator region: 12 GB, 4 GB after the cut in its patches.py); the SDK keeps the freed pages mapped
        // and maps new ones for later allocations, until a map fails: then it unmaps the cached block and retries (measured on the Mac: the refused 4 MB map at +0x43e2c000 was followed by
        // UnmapPhysicalMemory of the 3.15 GB block and the same map again). Without the limit the mapped total went past the pool in the first minute and the freed block stayed mapped.
        // Off by default: Ace Attorney Investigations Collection maps a little more than the 1 GB pool its profile reports
        if (DM_GAME_MEM_MAP_LIMIT && num == 0x2C) {
            int64_t room = (int64_t)APP_POOL - (int64_t)(g_heap_cur + DM_GAME_MEM_USED + big_bss_bytes()) - atomic_load(&g_alias_mapped);
            if (need && need > room) {
                pthread_mutex_unlock(&g_map_lock);
                static _Atomic int nref; if (atomic_fetch_add(&nref, 1) < 8) LOG("[mem] MapPhysicalMemory +%#zx refused: %.2f GB mapped, the application pool is full (ResultLimitReached, the SDK frees cached pages and retries)",
                                                                       size, atomic_load(&g_alias_mapped) / 1073741824.0);
                regs[0] = 0x10801; return 0;
            }
        }
        // Host pages are 16 KB, Horizon pages 4 KB: a map is rounded out to host pages (the extra neighboring 4K pages are inside the alias region and merely become writable early, harmless);
        // An edge host page is reclaimed only when no live logical neighbor remains.
        uintptr_t pg = (uintptr_t)getpagesize(), lo = (uintptr_t)a, hi = (uintptr_t)a + size;
        uintptr_t mlo = lo & ~(pg - 1), mhi = (hi + pg - 1) & ~(pg - 1);
        int commit = num == 0x2C;
        int64_t changed = commit ? need : (int64_t)size - need;
        if (!commit) {
            if ((lo - mlo) / 4096 != (uint64_t)alias_unmapped(mlo - (uintptr_t)g_alias, lo - mlo)) mlo += pg;
            if ((mhi - hi) / 4096 != (uint64_t)alias_unmapped(hi - (uintptr_t)g_alias, mhi - hi)) mhi -= pg;
        }
        int64_t unmap_d = 0; int unmapped_bits = 0;   // Unmap: the bitmap change, made before the pages are released
        if (changed && mhi > mlo) {
            if (commit) {
                dm_gfx_guest_open((void *)mlo, mhi - mlo);
                if (mprotect((void *)mlo, mhi - mlo, PROT_READ | PROT_WRITE))
                    die("SVC %#x mprotect %p+%#zx errno=%d", num, (void *)a, size, errno);
                dm_guest_write_prepare((void *)mlo, mhi - mlo);
            } else {
                // the bits go first: a host reader (dm_guest_read_mapped) that finds a page mapped must not find it inaccessible
                unmap_d = alias_pages(off, size, 0) * 4096; unmapped_bits = 1;
                if (guest_release(mlo, mhi)) {   // fresh zero pages: alias_initialize has nothing to clear when they are mapped again
                    for (uint64_t h = (mlo - (uintptr_t)g_alias) / pg, last = (mhi - 1 - (uintptr_t)g_alias) / pg; h <= last; ++h)
                        g_alias_exposed[h >> 6] &= ~(UINT64_C(1) << (h & 63));
                }
            }
        }
        if (commit && changed) alias_initialize(off, size);
        if (g_flag.svc) LOG("%sPhysicalMemory %p+%#zx", commit ? "Map" : "Unmap", (void *)a, size);
        // how deep the alias region is used (one line per 256 MB crossed): for sizing the reservation and alias region (iOS caps a single reservation at 6 GB)
        if (num == 0x2C) { static _Atomic uint64_t hw; uint64_t end = off + size, old = atomic_load(&hw);
            while (end > old && !atomic_compare_exchange_weak(&hw, &old, end)) {}
            if (end > old && end >> 28 != old >> 28) LOG("[mem] alias region mapped up to +%.2f GB (alias region %.2f GB)", end / 1073741824.0, g_alias_cap / 1073741824.0); }
        // Total actually mapped in the alias region (one line per 256 MB crossed, up or down): how much physical memory the game uses and whether it keeps growing. Counted with a page bitmap, only pages whose state really changed --
        // the SDK maps pages that are already mapped and unmaps ranges larger than what it mapped (Engage unmaps 0x1000000 and then 0x103f000 at the same address), so counting requested sizes went negative
        { int64_t d = unmapped_bits ? unmap_d : alias_pages(off, size, commit) * 4096, before = atomic_fetch_add(&g_alias_mapped, d), after = before + d;
          pthread_mutex_unlock(&g_map_lock);
          if (before >> 28 != after >> 28) LOG("[mem] alias region mapped %.2f GB (application pool %.2f GB)", after / 1073741824.0, APP_POOL / 1073741824.0); }
        regs[0] = 0; return 0;
    }
    case 0x26: {  // Break(reason, ptr, size): SDK fatal abort
        uint32_t res = (regs[1] && regs[2] >= 4) ? *(uint32_t *)regs[1] : 0;
        module_t *mm = module_of((void *)regs[30]);
        static int nb;
        LOG("Break#%d reason=%#llx Result=%#x svc at %s+%#llx", ++nb, regs[0], res, mm ? mm->name : "?", orig_addr);
        dm_backtrace(regs[29]);
        if (regs[0] & 0x80000000ull) return 0;   // notification only, can continue
        _exit(42);
    }
    case 0x1E: guest_clock_read_wait(); regs[0] = 0; regs[1] = guest_ticks(); return 0;   // GetSystemTick
    case 0x27: {  // OutputDebugString(str, len)
        if (g_flag.dbg) fprintf(stderr, "[guest] %.*s\n", (int)regs[1], (char *)regs[0]);
        regs[0] = 0; return 0;
    }
    case 0x29: {  // GetInfo(out*, id, handle, subid) → x1 = value
        uint64_t id = regs[1], v = 0;
        switch (id) {
        case I_CoreMask: v = 7; break;
        case I_PriorityMask: v = 0; break;
        case I_AliasAddr: v = (uint64_t)g_alias; break;   case I_AliasSize: v = g_alias_cap; break;
        case I_HeapAddr:  v = (uint64_t)g_heap;  break;   case I_HeapSize:  v = g_heap_cap;  break;
        case I_AslrAddr:  v = (uint64_t)g_as;    break;   case I_AslrSize:  v = g_reserve;   break;
        case I_StackAddr: v = (uint64_t)g_stack; break;   case I_StackSize: v = g_stack_cap; break;
        case I_TotalMem: case I_TotalNonSys: v = APP_POOL; break;
        case I_UsedMem:  case I_UsedNonSys:  pthread_mutex_lock(&g_map_lock);
                                             v = g_heap_cur + DM_GAME_MEM_USED + big_bss_bytes();   // profile mem_used, 64 MB unless the game needs more (see mem_used in dm/games/README.md)
                                             // with mem_map_limit, the pages mapped in the alias region count too, as on Horizon (the limit in MapPhysicalMemory is the pool less this)
                                             if (DM_GAME_MEM_MAP_LIMIT) v += (uint64_t)atomic_load(&g_alias_mapped);
                                             pthread_mutex_unlock(&g_map_lock); break;
        // the system resource the NPDM declares (profile sys_resource; 32 MB without the key). The SDK's IsVirtualAddressMemoryEnabled is "total != 0": Big Brain Academy's nninitStartup
        // asks StandardAllocator for 60 GB of virtual address memory when it is, and sets a heap only when it is not (its NPDM declares none)
        case I_SysResTotal: v = DM_GAME_SYS_RESOURCE; break; case I_SysResUsed: v = DM_GAME_SYS_RESOURCE ? 0x400000 : 0; break;
        case I_DebuggerAttached: v = 0; break;
        case I_RandomEntropy: { uint64_t dm_entropy64(void); v = dm_entropy64(); break; }   // a deterministic stream when DM_DETCLOCK is on (hle_os.c)
        case I_ProgramId: { static uint64_t pid; if (!pid) { const char *e = getenv("DM_PROGRAM_ID"); pid = e ? strtoull(e, NULL, 16) : DM_GAME_PROGRAM_ID; } v = pid; break; }   // default from the game profile's program_id; DM_PROGRAM_ID=hex overrides
        case I_IsApplication: v = 1; break;
        // nn::os::GetFreeThreadCount (SDK 11+): the thread resource limit less the threads alive, as the kernel answers it (Atmosphère kern_svc_info.cpp). pm gives an application a limit of
        // 96 threads (pm_spec.cpp BaseApplicationThreads; a boost raises it to 256). Monster Train 2's Unity 2022.3 player creates a thread only when this is at least 1: the 0 it got here
        // left its main thread waiting at boot for a thread it never created (main+0x3e1ac4)
        case I_FreeThreadCount: { int thr_limit_used(void); int u = thr_limit_used(); v = u < 96 ? 96 - u : 0; break; }
        case I_UserExcAddr: { static _Alignas(0x1000) uint8_t ctx[0x1000]; v = (uint64_t)ctx; break; }   // user exception context area: newer SDKs' SetDyingMessageRegion writes to +0x1c0 (0 would write through a null pointer)
        // The kernel's page size (Switch 2 SDKs): nn::ro::ProtectRelro takes it as the alignment of the RELRO bounds (value − 1 as a mask, 0 aborts), the VammManager maps that much up front.
        // Horizon's pages are 4 KB, the alignment of every module's segments (Donkey Kong Bananza)
        case I_PageSize: v = 0x1000; break;
        // Horizon's other types (Atmosphère kern_svc_info.cpp), answered 0 as before they were named: no thread tick count, no SVC permission bits, no I/O region, and no extra alias
        // region size (SDK 18 and later ask for it; only a process created with the extra-size flag has one)
        case I_ThreadTickCount: case I_IsSvcPermitted: case I_IoRegionHint: case I_AliasExtraSize: v = 0; break;
        // The Switch 2's remote region: nnosInitialize builds nn::os::detail::RemoteRegionManager's address space allocator over [type 29, type 29 + type 30), and
        // nn::os::GetRemoteMemoryUsage returns types 31 and 32. The SDK's only user is the GPU driver's boot (nv::InitializeGraphics → nn::gs::sdk::BootGraphicsSystem →
        // BootWithRemoteMemory), which the host's own nv::InitializeGraphics replaces (dm/gfx/hle_nv.c), so the process has no remote region and no remote memory (Donkey Kong Bananza)
        case I_RemoteAddr: case I_RemoteSize: case I_RemoteUsage0: case I_RemoteUsage1: v = 0; break;
        default: { static uint8_t seen[64]; if (id < 64 && !seen[id]) { seen[id] = 1; LOG("[RED] svc GetInfo: unknown info type %llu (answered 0)", (unsigned long long)id); } v = 0; break; }
        }
        regs[0] = 0; regs[1] = v; return 0;
    }
    default:
        if (!g_seen[num & 0xff]) { g_seen[num & 0xff] = 1; LOG("SVC %#x x0=%#llx x1=%#llx x2=%#llx (placeholder returns success)", num, regs[0], regs[1], regs[2]); }
        regs[0] = 0; return 0;
    }
}
