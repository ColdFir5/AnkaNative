// The Android replay's SPIR-V rewrite of vertex inputs fed integer formats (dm/gfx/vk_android/vka_spvpatch.c), on the Mac: the graphics selftest's vertex shader
// (a component read through an access chain at location 0, whole loads at location 1) retyped as uint / int, checked by structure and, when spirv-val is on the PATH,
// by validation. DM_VK_SPV_DIR=<folder of .spv vertex modules> also rewrites each of those (every vec4 input as uint) and validates it. DM_VK_GS_SPV=<a geometry module
// that writes memory> puts its writes under a per-primitive claim (vka_spv_gs_claim, after the one-invocation subgroup rewrite) and validates it
#include "../../gfx/vk_android/vka_spvpatch.h"
#include "../../gfx/vk_android/vka_selftest_spv.h"
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { g_fail++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)
static int have_val(void) { return system("spirv-val --version >/dev/null 2>&1") == 0; }
static int validate(const uint32_t *w, size_t n, const char *tag) {
    char path[256]; snprintf(path, sizeof path, "/tmp/t_spvpatch_%d.spv", (int)getpid());
    FILE *f = fopen(path, "wb"); if (!f) return 0; fwrite(w, 4, n, f); fclose(f);
    char cmd[400]; snprintf(cmd, sizeof cmd, "spirv-val --target-env vulkan1.3 %s", path);
    int r = system(cmd); remove(path);
    if (r) printf("  spirv-val failed on %s\n", tag);
    return r == 0;
}
static int count_op(const uint32_t *w, size_t n, uint32_t op) { int k = 0; for (size_t i = 5; i < n; i += w[i] >> 16) { if (!(w[i] >> 16)) break; k += (w[i] & 0xffff) == op; } return k; }
int main(void) {
    int val = have_val();
    size_t n = sizeof SPV_TRI_VERT / 4, on = 0; char err[200] = "";
    uint32_t *o = vka_spv_vin_retype(SPV_TRI_VERT, n, 1, 2, &on, err, sizeof err);
    CHECK(o, "selftest vertex shader not rewritten: %s", err);
    if (o) {
        CHECK(o[3] > SPV_TRI_VERT[3], "bound grows (%u → %u)", SPV_TRI_VERT[3], o[3]);
        CHECK(count_op(o, on, 124) >= count_op(SPV_TRI_VERT, n, 124) + 2, "a bitcast after each load of the two inputs");
        CHECK(count_op(o, on, 21) >= 2, "an int type of each signedness");
        if (val) CHECK(validate(o, on, "selftest vertex"), "valid SPIR-V");
        free(o);
    }
    CHECK(!vka_spv_vin_retype(SPV_TRI_VERT, n, 4, 0, &on, err, sizeof err), "a location the module does not read is refused");
    CHECK(!vka_spv_vin_retype(SPV_TRI_VERT, n, 1, 1, &on, err, sizeof err), "a location both uint and int is refused");
    {   // A float colour output into an integer target (vka_spv_fout_retype): the selftest's fragment shader, its vec4 output at location 0 as uvec4 / ivec4
        size_t fn = sizeof SPV_TRI_FRAG / 4;
        for (int c = 0; c < 2; c++) {
            uint32_t *f = vka_spv_fout_retype(SPV_TRI_FRAG, fn, c ? 0 : 1, c ? 1 : 0, &on, err, sizeof err);
            CHECK(f, "selftest fragment shader output not retyped (%s): %s", c ? "int" : "uint", err);
            if (!f) continue;
            CHECK(count_op(f, on, 124) >= count_op(SPV_TRI_FRAG, fn, 124) + 1, "a bitcast before each store of the output");
            CHECK(count_op(f, on, 21) >= 1, "an integer type");
            if (val) CHECK(validate(f, on, c ? "selftest fragment (int)" : "selftest fragment (uint)"), "valid SPIR-V");
            free(f);
        }
        CHECK(!vka_spv_fout_retype(SPV_TRI_FRAG, fn, 2, 0, &on, err, sizeof err), "a location the module does not write is refused");
        CHECK(!vka_spv_fout_retype(SPV_TRI_FRAG, fn, 1, 1, &on, err, sizeof err), "a location both uint and int is refused");
    }
    {   // An integer texture read through a float-typed sampled image binding (vka_spv_tex_retype): a small compute module with two bindings sharing one float 2D
        // OpTypeSampledImage (binding 2: OpImageFetch %v4float + extract + OpBitcast %int; binding 3: OpImageFetch %v4float untouched)
        static const uint32_t spv_tex[] = {
            0x07230203u, 0x00010600u, 0x00070000u, 0x00000018u, 0x00000000u, 0x00020011u, 0x00000001u, 0x0003000eu, 0x00000000u, 0x00000001u,
            0x0007000fu, 0x00000005u, 0x00000001u, 0x6e69616du, 0x00000000u, 0x00000002u, 0x00000003u, 0x00060010u, 0x00000001u, 0x00000011u,
            0x00000001u, 0x00000001u, 0x00000001u, 0x00040047u, 0x00000002u, 0x00000022u, 0x00000000u, 0x00040047u, 0x00000002u, 0x00000021u,
            0x00000002u, 0x00040047u, 0x00000003u, 0x00000022u, 0x00000000u, 0x00040047u, 0x00000003u, 0x00000021u, 0x00000003u, 0x00020013u,
            0x00000004u, 0x00030021u, 0x00000005u, 0x00000004u, 0x00030016u, 0x00000006u, 0x00000020u, 0x00040017u, 0x00000007u, 0x00000006u,
            0x00000004u, 0x00040015u, 0x00000008u, 0x00000020u, 0x00000001u, 0x00040017u, 0x00000009u, 0x00000008u, 0x00000002u, 0x0004002bu,
            0x00000008u, 0x0000000au, 0x00000000u, 0x0005002cu, 0x00000009u, 0x0000000bu, 0x0000000au, 0x0000000au, 0x00090019u, 0x0000000cu,
            0x00000006u, 0x00000001u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000001u, 0x00000000u, 0x0003001bu, 0x0000000du, 0x0000000cu,
            0x00040020u, 0x0000000eu, 0x00000000u, 0x0000000du, 0x0004003bu, 0x0000000eu, 0x00000002u, 0x00000000u, 0x0004003bu, 0x0000000eu,
            0x00000003u, 0x00000000u, 0x00050036u, 0x00000004u, 0x00000001u, 0x00000000u, 0x00000005u, 0x000200f8u, 0x0000000fu, 0x0004003du,
            0x0000000du, 0x00000010u, 0x00000002u, 0x00040064u, 0x0000000cu, 0x00000011u, 0x00000010u, 0x0007005fu, 0x00000007u, 0x00000012u,
            0x00000011u, 0x0000000bu, 0x00000002u, 0x0000000au, 0x00050051u, 0x00000006u, 0x00000013u, 0x00000012u, 0x00000000u, 0x0004007cu,
            0x00000008u, 0x00000014u, 0x00000013u, 0x0004003du, 0x0000000du, 0x00000015u, 0x00000003u, 0x00040064u, 0x0000000cu, 0x00000016u,
            0x00000015u, 0x0007005fu, 0x00000007u, 0x00000017u, 0x00000016u, 0x0000000bu, 0x00000002u, 0x0000000au, 0x000100fdu, 0x00010038u,
        };
        size_t tn = sizeof spv_tex / 4;
        uint32_t *tu = vka_spv_tex_retype(spv_tex, tn, 1ull << 2, 0, &on, err, sizeof err);
        CHECK(tu, "compute texture binding 2 not retyped to uint: %s", err);
        if (tu) {
            CHECK(tu[3] > spv_tex[3], "tex retype bound grows (%u -> %u)", spv_tex[3], tu[3]);
            CHECK(count_op(tu, on, 124) == count_op(spv_tex, tn, 124) + 1, "one bitcast added (binding 2 only)");
            uint32_t u32 = 0, v4u = 0; int saw_f2 = 0, saw_f3 = 0;
            for (size_t i = 5; i < on;) {
                uint32_t op = tu[i] & 0xffff, len = tu[i] >> 16; if (!len) break;
                if (op == 21 && len >= 4 && tu[i + 2] == 32 && tu[i + 3] == 0) u32 = tu[i + 1];
                else if (op == 23 && len >= 4 && tu[i + 2] == u32 && u32 && tu[i + 3] == 4) v4u = tu[i + 1];
                else if (op == 95 && len >= 5) {
                    if (tu[i + 3] == 0x11) {   // binding 2 fetch -> %v4uint, followed by OpBitcast %v4float %18 %tmp
                        size_t j = i + len;
                        saw_f2 = v4u && tu[i + 1] == v4u && j + 4 <= on && (tu[j] & 0xffff) == 124 && tu[j + 1] == 7 && tu[j + 2] == 0x12 && tu[j + 3] == tu[i + 2];
                    } else if (tu[i + 3] == 0x16) {   // binding 3 fetch -> untouched %v4float %23
                        saw_f3 = tu[i + 1] == 7 && tu[i + 2] == 0x17;
                    }
                }
                i += len;
            }
            CHECK(saw_f2, "binding 2 OpImageFetch is now %%v4uint followed by OpBitcast to %%v4float");
            CHECK(saw_f3, "binding 3 OpImageFetch is untouched (%%v4float)");
            if (val) CHECK(validate(tu, on, "compute tex retype (uint)"), "valid SPIR-V");
            free(tu);
        }
        uint32_t *ts = vka_spv_tex_retype(spv_tex, tn, 0, 1ull << 2, &on, err, sizeof err);
        CHECK(ts, "compute texture binding 2 not retyped to int: %s", err);
        if (ts) {
            if (val) CHECK(validate(ts, on, "compute tex retype (int)"), "valid SPIR-V");
            free(ts);
        }
        CHECK(!vka_spv_tex_retype(spv_tex, tn, 1ull << 4, 0, &on, err, sizeof err), "a binding not in the module is refused");
        CHECK(!vka_spv_tex_retype(spv_tex, tn, 1ull << 2, 1ull << 2, &on, err, sizeof err), "a binding both uint and int is refused");
        CHECK(!vka_spv_tex_retype(spv_tex, tn, 0, 0, &on, err, sizeof err), "empty masks are refused");
    }
    const char *dir = getenv("DM_VK_SPV_DIR");
    if (dir) {
        DIR *d = opendir(dir); struct dirent *e; int done = 0, refused = 0;
        while (d && (e = readdir(d))) {
            size_t l = strlen(e->d_name); if (l < 4 || strcmp(e->d_name + l - 4, ".spv")) continue;
            char p[1024]; snprintf(p, sizeof p, "%s/%s", dir, e->d_name);
            FILE *f = fopen(p, "rb"); if (!f) continue; fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
            uint32_t *w = malloc((size_t)sz); if (fread(w, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(w); continue; } fclose(f);
            // every Location-decorated input: find them by a first rewrite of all 32 locations, narrowing on refusal
            uint32_t mask = 0;
            for (size_t i = 5; i < (size_t)sz / 4; i += w[i] >> 16) { if (!(w[i] >> 16)) break; if ((w[i] & 0xffff) == 71 && (w[i] >> 16) >= 4 && w[i + 2] == 30 && w[i + 3] < 32) {
                uint32_t id = w[i + 1]; for (size_t j = 5; j < (size_t)sz / 4; j += w[j] >> 16) { if (!(w[j] >> 16)) break; if ((w[j] & 0xffff) == 59 && w[j + 2] == id && w[j + 3] == 1) mask |= 1u << w[i + 3]; } } }
            uint32_t *x = mask ? vka_spv_vin_retype(w, (size_t)sz / 4, mask, 0, &on, err, sizeof err) : NULL;
            if (!x) { refused++; if (refused <= 4) printf("  %s: %s\n", e->d_name, mask ? err : "no input"); }
            else { done++; if (val) CHECK(validate(x, on, e->d_name), "%s valid", e->d_name); free(x); }
            free(w);
        }
        if (d) closedir(d);
        printf("t_android_spvpatch: %d modules rewritten, %d refused in %s\n", done, refused, dir);
    }
    const char *gs = getenv("DM_VK_GS_SPV");
    if (gs) {
        FILE *f = fopen(gs, "rb"); CHECK(f, "%s not readable", gs);
        if (f) { fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f); uint32_t *w = malloc((size_t)sz); size_t nw = fread(w, 1, (size_t)sz, f) / 4; fclose(f);
            int ch = 0; size_t sn = 0; uint32_t *sg = vka_spv_subgroup_single(w, nw, &sn, &ch, err, sizeof err); if (sg) { free(w); w = sg; nw = sn; }
            int found = vka_spv_gs_claim(w, nw, 255, NULL, NULL, err, sizeof err);
            CHECK(found >= 0, "%s: memory writes counted (%s)", gs, err);
            uint32_t *x = NULL; size_t xn = 0; int r = found > 0 ? vka_spv_gs_claim(w, nw, 255, &x, &xn, err, sizeof err) : -1;
            if (found > 0) CHECK(r == found && x, "claim module made (%s)", err);
            if (x) {
                CHECK(count_op(x, xn, 218) == count_op(w, nw, 218), "emits as before");
                CHECK(count_op(x, xn, 247) == count_op(w, nw, 247) + found, "a selection around each write");
                CHECK(count_op(x, xn, 229) == count_op(w, nw, 229) + 1, "one exchange for the claim");
                if (val) CHECK(validate(x, xn, "gs claim"), "claim module valid");
                free(x);
            }
            printf("t_android_spvpatch: %s: %d memory writes claimed\n", gs, found);
            free(w); }
    }
    printf("t_android_spvpatch: %s%s\n", g_fail ? "FAILED" : "ok", val ? "" : " (no spirv-val: structure only)");
    return g_fail != 0;
}
