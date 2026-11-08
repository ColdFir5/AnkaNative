// Guest-aliased image memory (dm/gfx/vk_android/vka_rtalias.h): images of one key share the key's current allocation while they fit it, a larger one makes a grown current
// allocation and the older one stays with its images, other keys do not share, an allocation goes with its last image (in any order), and the stats count each allocation
// once and the bytes the sharing saves
#include <stdio.h>
#include "../../gfx/vk_android/vka_rtalias.h"

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)
#define MIB (1ull << 20)

typedef struct { uint64_t pool, pgen, mem; uint32_t fmt, levels, usage; } key;   // a stand-in for vka_tex.c's key (storage start, create info without the extent)
static int bind(vkra_tab *t, const key *k, uint64_t need, uint64_t *next_h, uint64_t *h_out, uint64_t *made) {   // vka_imem_bind_alias's bookkeeping: 1 = a new allocation
    uint64_t size = 0;
    int i = vkra_ref(t, k, sizeof *k, need, &size);
    if (i >= 0) { *h_out = t->e[i].h; if (made) *made = 0; return 0; }
    uint64_t h = (*next_h)++;
    CHECK(size >= need, "the allocation to make holds the image");
    CHECK(vkra_add(t, k, sizeof *k, h, size, need) >= 0, "add");
    *h_out = h; if (made) *made = size; return 1;
}
static uint64_t d32(uint32_t w) { return (uint64_t)w * w * 4 + 64 * 1024; }   // a square DEPTH32F target's memory, roughly (texels and a little metadata)

int main(void) {
    vkra_tab t = {0}; uint64_t nh = 100, h[16], made;
    key k = {0x4140e33288ull, 14, 0x412f25d600ull, 126, 1, 0x37}, other = k, gen = k, fmt = k;
    other.mem += 0x1646600; gen.pgen = 15; fmt.fmt = 122;
    // standing: five sizes around 1150², the first allocation grows once (1.5 times) and holds the rest
    CHECK(bind(&t, &k, d32(1142), &nh, &h[0], &made) == 1 && made == d32(1142), "the first size: an allocation of its size");
    CHECK(bind(&t, &k, d32(1140), &nh, &h[1], NULL) == 0 && h[1] == h[0], "a smaller size: the same allocation");
    CHECK(bind(&t, &k, d32(1156), &nh, &h[2], &made) == 1 && h[2] != h[0] && made == d32(1142) / 2 * 3, "a larger size: a grown allocation (%llu)", (unsigned long long)made);
    CHECK(bind(&t, &k, d32(1145), &nh, &h[3], NULL) == 0 && h[3] == h[2], "from then on the grown one is the key's current allocation");
    CHECK(bind(&t, &k, d32(1153), &nh, &h[4], NULL) == 0 && h[4] == h[2], "and holds the other sizes");
    // walking: 1112² to 1482², about one size a frame, all over the same storage
    int fresh = 0; uint64_t hw;
    for (uint32_t w = 1112; w <= 1482; w += 3) fresh += bind(&t, &k, d32(w), &nh, &hw, NULL);
    CHECK(fresh == 1, "124 sizes from 1112² to 1482²: %d more allocations (one)", fresh);
    uint64_t im, al, by, sv; vkra_stats(&t, &im, &al, &by, &sv);
    CHECK(im == 5 + 124 && al == 3, "stats: %llu images in %llu allocations", (unsigned long long)im, (unsigned long long)al);
    CHECK(by == d32(1142) + d32(1142) / 2 * 3 + d32(1142) / 2 * 3 / 2 * 3, "stats: %.1f MiB of allocations", by / (double)MIB);
    CHECK(by < 25 * MIB && sv > 600 * MIB, "under 25 MiB for 129 images that would have taken %.0f MiB alone", (by + sv) / (double)MIB);
    CHECK(t.made == 3 && t.grown == 2, "made %llu, grown %llu", (unsigned long long)t.made, (unsigned long long)t.grown);
    // other keys: another storage start, a new pool at the same address (another Initialize serial), another format
    CHECK(bind(&t, &other, d32(1142), &nh, &h[5], NULL) == 1 && h[5] != h[0] && h[5] != h[2], "another storage address: its own allocation");
    CHECK(bind(&t, &gen, d32(1142), &nh, &h[6], NULL) == 1 && h[6] != h[5], "the same address in a pool initialized again: its own allocation");
    CHECK(bind(&t, &fmt, d32(1142), &nh, &h[7], NULL) == 1 && h[7] != h[6], "another format: its own allocation");
    // let go in any order: the older allocation goes with its last image, the current one stays the key's
    CHECK(vkra_unref(&t, h[1], d32(1140)) == 0, "one of two images of the first allocation let go: kept");
    CHECK(vkra_unref(&t, h[0], d32(1142)) == 1, "the other: the first allocation is freed");
    CHECK(vkra_unref(&t, h[0], d32(1142)) == -1, "and is no entry's any more");
    CHECK(bind(&t, &k, d32(1150), &nh, &hw, NULL) == 0 && hw != h[0], "a new image of the key goes to the current allocation");
    // an allocation whose images all go: the key makes a new one of the image's size (not grown)
    { key s = k; s.mem = 0x4130e35600ull; uint64_t a, b;
      CHECK(bind(&t, &s, d32(1920), &nh, &a, &made) == 1 && made == d32(1920), "1920²: its own size");
      CHECK(vkra_unref(&t, a, d32(1920)) == 1, "let go: freed");
      CHECK(bind(&t, &s, d32(2048), &nh, &b, &made) == 1 && made == d32(2048), "a larger one after it: its size, not grown from a freed allocation"); }
    // refused: a key longer than VKRA_KEY, an empty image, an allocation smaller than its image
    { uint64_t sz; CHECK(vkra_ref(&t, &k, VKRA_KEY + 1, MIB, &sz) < 0 && vkra_add(&t, &k, VKRA_KEY + 1, 999, MIB, MIB) < 0, "a key longer than VKRA_KEY is refused");
      CHECK(vkra_ref(&t, &k, sizeof k, 0, &sz) < 0, "an image of no bytes is refused");
      key z = k; z.mem = 1; CHECK(vkra_add(&t, &z, sizeof z, 998, MIB, 2 * MIB) < 0, "an allocation smaller than its image is refused"); }
    // many keys: the table grows
    int n0 = t.n;
    for (uint32_t i = 0; i < 1000; i++) { key q = k; q.mem = 0x500000000ull + (uint64_t)i * 0x1000000; uint64_t hh; bind(&t, &q, 4 * MIB, &nh, &hh, NULL); }
    CHECK(t.n == n0 + 1000, "1000 more keys: %d allocations", t.n);
    free(t.e);
    printf("t_android_rtalias: %s\n", fails ? "FAILED" : "ok");
    return fails != 0;
}
