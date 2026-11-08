// Guest-aliased image memory (vka_imgmem.c vka_imem_bind_alias, vka_tex.c "Guest-aliased targets"): the bookkeeping, free of Vulkan so the unit test
// (dm/host/tests/t_android_rtalias.c) can check it. GPU-content images whose guest storage starts at one address (one pool, one Initialize serial) and whose create infos
// differ only in extent are bound at offset 0 of one allocation: on the Switch they are the same bytes, so only the newest one written holds content there too.
// An entry is an allocation (handle, size), its key, the images bound to it (refs) and the bytes they would have had alone (isum). The newest allocation of a key is its
// current one: an image that fits it is bound there; a larger one makes a new current allocation of max(its size, VKRA_GROW_NUM / VKRA_GROW_DEN of the current one's), so
// a target that grows a little every frame makes a few allocations, not one per size. An older allocation stays with the images bound to it and goes with its last image
#ifndef VKA_RTALIAS_H
#define VKA_RTALIAS_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { VKRA_KEY = 128, VKRA_GROW_NUM = 3, VKRA_GROW_DEN = 2 };
typedef struct { uint8_t key[VKRA_KEY]; uint32_t klen; uint64_t h, size, isum; int refs, cur; } vkra_ent;
typedef struct { vkra_ent *e; int n, cap; uint64_t made, grown; } vkra_tab;   // made: allocations ever made; grown: of them made because an image did not fit the current one

static inline int vkra_find_cur(const vkra_tab *t, const void *key, uint32_t klen) {
    for (int i = 0; i < t->n; i++) if (t->e[i].cur && t->e[i].klen == klen && !memcmp(t->e[i].key, key, klen)) return i;
    return -1;
}
static inline int vkra_find_h(const vkra_tab *t, uint64_t h) { for (int i = 0; i < t->n; i++) if (t->e[i].h == h) return i; return -1; }
// One more image of key that needs need bytes: the index of the key's current allocation when it holds need (refs counted), else -1 with *size the allocation to make
// (need, or the grown size of a current allocation too small for it)
static inline int vkra_ref(vkra_tab *t, const void *key, uint32_t klen, uint64_t need, uint64_t *size) {
    *size = need;
    if (klen > VKRA_KEY || !need) return -1;
    int i = vkra_find_cur(t, key, klen);
    if (i >= 0 && t->e[i].size >= need) { t->e[i].refs++; t->e[i].isum += need; return i; }
    if (i >= 0) { uint64_t g = t->e[i].size / VKRA_GROW_DEN * VKRA_GROW_NUM; if (g > *size) *size = g; }
    return -1;
}
// A new allocation h of size bytes for key, with its first image (need bytes): its index, the key's current allocation from now on; -1 = no room (the caller keeps h as
// that image's own memory)
static inline int vkra_add(vkra_tab *t, const void *key, uint32_t klen, uint64_t h, uint64_t size, uint64_t need) {
    if (klen > VKRA_KEY || !h || size < need) return -1;
    if (t->n == t->cap) { int c = t->cap ? 2 * t->cap : 16; vkra_ent *e = (vkra_ent *)realloc(t->e, (size_t)c * sizeof *e); if (!e) return -1; t->e = e; t->cap = c; }
    int o = vkra_find_cur(t, key, klen);
    if (o >= 0) { t->e[o].cur = 0; t->grown++; }
    vkra_ent *e = &t->e[t->n]; memset(e, 0, sizeof *e); memcpy(e->key, key, klen); e->klen = klen; e->h = h; e->size = size; e->refs = 1; e->isum = need; e->cur = 1; t->made++;
    return t->n++;
}
// An image of allocation h (need bytes) let go: 1 when it was the last (the entry is gone; the caller frees h), 0 when others remain, -1 when h is no entry's
static inline int vkra_unref(vkra_tab *t, uint64_t h, uint64_t need) {
    int i = vkra_find_h(t, h); if (i < 0) return -1;
    t->e[i].isum = t->e[i].isum > need ? t->e[i].isum - need : 0;
    if (--t->e[i].refs > 0) return 0;
    t->e[i] = t->e[--t->n];
    return 1;
}
// The table now: images bound, allocations, their bytes, and the bytes they save (what the images would have had alone, beyond each allocation's size)
static inline void vkra_stats(const vkra_tab *t, uint64_t *images, uint64_t *allocs, uint64_t *bytes, uint64_t *saved) {
    uint64_t im = 0, by = 0, sv = 0;
    for (int i = 0; i < t->n; i++) { im += (uint64_t)t->e[i].refs; by += t->e[i].size; if (t->e[i].isum > t->e[i].size) sv += t->e[i].isum - t->e[i].size; }
    if (images) *images = im; if (allocs) *allocs = (uint64_t)t->n; if (bytes) *bytes = by; if (saved) *saved = sv;
}
#endif
