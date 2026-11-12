// SPIR-V rewrites the replay makes at pipeline time (plain C, no Vulkan: unit-tested on the Mac by dm/host/tests/t_android_spvpatch.c).
//
// vka_spv_vin_retype: vertex inputs fed integer formats. The translator declares every vertex input a vec4 of floats (the library's vin_types say float for all of
// Donkey Kong Bananza's 65,846 inputs) and a shader that uses one as integer bits bit-casts it. On Maxwell an attribute of an integer format (UINT8 … SINT32) puts the
// integer, widened by its own sign, into the 32-bit register the shader reads, so those bits are what the bit-cast must see (bone indices, packed data). Vulkan gives a float
// input only a float, normalized or scaled format, and a USCALED stand-in hands the shader the bits of 5.0 instead of 5. So for a pipeline whose integer attributes feed float
// inputs, the vertex module is rewritten: those inputs become uvec4 / ivec4 (Input variables of the integer vector type), and each load of one, or of a component through an
// access chain, is followed by an OpBitcast back to the float type the body expects. The bits are then Maxwell's, and the attribute keeps its own integer format.
// nvn_mtl.m reaches the same bits by converting such attributes on the CPU into float32 bits (vconv_int); the Apple chain's dmsh_vin is the reverse case (integer inputs fed float formats).
#include "vka_spvpatch.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

enum { OpExtInst = 12, OpName = 5, OpMemberName = 6, OpEntryPoint = 15, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23, OpTypePointer = 32, OpFunction = 54,
       OpVariable = 59, OpLoad = 61, OpAccessChain = 65, OpInBoundsAccessChain = 66, OpDecorate = 71, OpMemberDecorate = 72, OpBitcast = 124,
       OpDecorateId = 332, OpDecorateString = 5632, OpMemberDecorateString = 5633 };
enum { DecLocation = 30, SCInput = 1 };

typedef struct { uint32_t *w; size_t n, cap; } vec_t;
static int put(vec_t *v, uint32_t x) {
    if (v->n == v->cap) { size_t c = v->cap ? 2 * v->cap : 4096; uint32_t *p = realloc(v->w, c * 4); if (!p) return 0; v->w = p; v->cap = c; }
    v->w[v->n++] = x; return 1;
}
static int put_n(vec_t *v, const uint32_t *x, size_t n) { for (size_t i = 0; i < n; i++) if (!put(v, x[i])) return 0; return 1; }

uint32_t *vka_spv_vin_retype(const uint32_t *w, size_t nw, uint32_t umask, uint32_t smask, size_t *out_nw, char *err, size_t errn) {
#define FAIL(...) do { snprintf(err, errn, __VA_ARGS__); goto fail; } while (0)
    uint32_t *cls = NULL; vec_t o = {0};   // cls[id]: 0 untouched, 1 a retyped input or a chain into one, as uint, 2 as int (bit 4: the id is a pointer to a vec4)
    if (nw < 5 || w[0] != 0x07230203) { snprintf(err, errn, "not SPIR-V"); return NULL; }
    uint32_t bound = w[3];
    if (!(cls = calloc(bound, 4))) { snprintf(err, errn, "out of memory"); return NULL; }
    if ((umask | smask) == 0 || (umask & smask)) FAIL("no input, or an input both uint and int");
    // Pass 1: locations, the types, the Input variables
    uint32_t *loc = calloc(bound, 4); if (!loc) FAIL("out of memory");
    for (uint32_t i = 0; i < bound; i++) loc[i] = ~0u;
    uint32_t t_f32 = 0, t_u32 = 0, t_i32 = 0, t_v4f = 0, t_v4u = 0, t_v4i = 0, p_in_v4f = 0;
    uint32_t p_in[2][2] = {{0}};   // [uint/int][scalar/vec4] Input pointers that exist
    size_t first_fn = nw;
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > nw) { free(loc); FAIL("bad instruction at word %zu", i); }
        if (op == OpDecorate && len >= 4 && w[i + 2] == DecLocation && w[i + 1] < bound) loc[w[i + 1]] = w[i + 3];
        else if (op == OpTypeFloat && len >= 3 && w[i + 2] == 32) t_f32 = w[i + 1];
        else if (op == OpTypeInt && len >= 4 && w[i + 2] == 32) { if (w[i + 3]) t_i32 = w[i + 1]; else t_u32 = w[i + 1]; }
        else if (op == OpTypeVector && len >= 4 && w[i + 3] == 4) {
            if (w[i + 2] == t_f32 && t_f32) t_v4f = w[i + 1]; else if (w[i + 2] == t_u32 && t_u32) t_v4u = w[i + 1]; else if (w[i + 2] == t_i32 && t_i32) t_v4i = w[i + 1]; }
        else if (op == OpTypePointer && len >= 4 && w[i + 2] == SCInput) {
            uint32_t t = w[i + 3];
            if (t == t_v4f && t) p_in_v4f = w[i + 1];
            for (int c = 0; c < 2; c++) { if (t == (c ? t_i32 : t_u32) && t) p_in[c][0] = w[i + 1]; if (t == (c ? t_v4i : t_v4u) && t) p_in[c][1] = w[i + 1]; }
        }
        else if (op == OpFunction && first_fn == nw) first_fn = i;
        i += len;
    }
    if (!t_f32 || !t_v4f || !p_in_v4f) { free(loc); FAIL("no vec4 float input type"); }
    // New ids for the types the module lacks
    uint32_t nb = bound;
    int need_c[2] = {(umask != 0), (smask != 0)};
    uint32_t t_s[2] = {t_u32, t_i32}, t_v[2] = {t_v4u, t_v4i};
    uint32_t mk_s[2] = {0}, mk_v[2] = {0}, mk_p[2][2] = {{0}};
    for (int c = 0; c < 2; c++) if (need_c[c]) {
        if (!t_s[c]) t_s[c] = mk_s[c] = nb++;
        if (!t_v[c]) t_v[c] = mk_v[c] = nb++;
        if (!p_in[c][0]) p_in[c][0] = mk_p[c][0] = nb++;
        if (!p_in[c][1]) p_in[c][1] = mk_p[c][1] = nb++;
    }
    // The input variables to retype: Location in a mask, type pointer-to-vec4-float
    uint32_t moved[32]; int nmoved = 0, found = 0;
    for (size_t i = 5; i < first_fn;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == OpVariable && len >= 4 && w[i + 3] == SCInput && w[i + 2] < bound && w[i + 1] == p_in_v4f) {
            uint32_t id = w[i + 2], l = loc[id];
            if (l < 32 && ((umask | smask) >> l & 1)) { cls[id] = (umask >> l & 1 ? 1 : 2) | 16; if (nmoved < 32) moved[nmoved++] = id; found |= 1u << l; }
        }
        i += len;
    }
    free(loc);
    if ((uint32_t)found != (umask | smask)) FAIL("inputs %#x are not vec4 float Input variables", (umask | smask) & ~(uint32_t)found);
    // Pass 2: copy, retyping. The globals up to the first function, minus the moved variables; then the new types and the variables; then the functions
    if (!put_n(&o, w, 5)) FAIL("out of memory");
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; const uint32_t *in = w + i;
        if (i == first_fn) {
            for (int c = 0; c < 2; c++) if (need_c[c]) {
                if (mk_s[c]) { uint32_t x[4] = {4u << 16 | OpTypeInt, mk_s[c], 32, (uint32_t)c}; if (!put_n(&o, x, 4)) FAIL("out of memory"); }
                if (mk_v[c]) { uint32_t x[4] = {4u << 16 | OpTypeVector, mk_v[c], t_s[c], 4}; if (!put_n(&o, x, 4)) FAIL("out of memory"); }
                if (mk_p[c][0]) { uint32_t x[4] = {4u << 16 | OpTypePointer, mk_p[c][0], SCInput, t_s[c]}; if (!put_n(&o, x, 4)) FAIL("out of memory"); }
                if (mk_p[c][1]) { uint32_t x[4] = {4u << 16 | OpTypePointer, mk_p[c][1], SCInput, t_v[c]}; if (!put_n(&o, x, 4)) FAIL("out of memory"); }
            }
            for (int k = 0; k < nmoved; k++) { uint32_t x[4] = {4u << 16 | OpVariable, p_in[(cls[moved[k]] & 3) - 1][1], moved[k], SCInput}; if (!put_n(&o, x, 4)) FAIL("out of memory"); }
        }
        if (op == OpVariable && len >= 4 && w[i + 2] < bound && cls[w[i + 2]] && i < first_fn) { i += len; continue; }   // re-emitted above
        if ((op == OpAccessChain || op == OpInBoundsAccessChain) && len >= 4 && w[i + 3] < bound && cls[w[i + 3]]) {
            int c = (cls[w[i + 3]] & 3) - 1; uint32_t rt = w[i + 1], r = w[i + 2];
            // The chain's result: a component (pointer to float) or the vector itself (no index)
            if (r >= bound) FAIL("bad id");
            int vec = len == 4;
            if (!vec && len != 5) FAIL("an access chain of %u indexes into a retyped input", len - 4);
            (void)rt;
            cls[r] = (uint32_t)(c + 1) | (vec ? 16 : 0);
            if (!put_n(&o, in, len)) FAIL("out of memory");
            o.w[o.n - len + 1] = p_in[c][vec];
            i += len; continue;
        }
        if (op == OpLoad && len >= 4 && w[i + 3] < bound && cls[w[i + 3]]) {
            int c = (cls[w[i + 3]] & 3) - 1, vec = (cls[w[i + 3]] & 16) != 0;
            uint32_t ft = w[i + 1], r = w[i + 2], tmp = nb++;
            if (!put_n(&o, in, len)) FAIL("out of memory");
            o.w[o.n - len + 1] = vec ? t_v[c] : t_s[c]; o.w[o.n - len + 2] = tmp;
            uint32_t x[4] = {4u << 16 | OpBitcast, ft, r, tmp}; if (!put_n(&o, x, 4)) FAIL("out of memory");
            i += len; continue;
        }
        // Any other use of a retyped variable or chain (a call argument, a copy): not understood, refused. Debug names, decorations and the entry point's interface are fine
        if (op != OpName && op != OpMemberName && op != OpEntryPoint && op != OpDecorate && op != OpMemberDecorate && op != OpDecorateId && op != OpDecorateString && op != OpMemberDecorateString
            && i >= first_fn)
            for (uint32_t k = 1; k < len; k++) if (!(op == OpExtInst && k == 4) && w[i + k] < bound && cls[w[i + k]] && op != OpVariable) FAIL("opcode %u uses a retyped input (id %u)", op, w[i + k]);
        if (!put_n(&o, in, len)) FAIL("out of memory");
        i += len;
    }
    o.w[3] = nb;
    free(cls);
    *out_nw = o.n; return o.w;
fail:
    free(cls); free(o.w);
    return NULL;
#undef FAIL
}

// vka_spv_subgroup_single: subgroup operations in a stage the device has none for. The Adreno 840 lists subgroup operations for the fragment and compute stages only
// (VkPhysicalDeviceSubgroupProperties::supportedStages 0xf0), and the translator emits them wherever the Maxwell code votes or shuffles: in Donkey Kong Bananza four geometry
// shaders (the terrain's culling shader 5d537f7630ec3671 among them) count each warp's surviving triangles with a ballot, let one lane add them to the chunk's indirect draw
// record and hand the base to the others with a shuffle. The driver takes the module and gives no meaningful lanes, so triangles were dropped or written over each other:
// black holes in the cave walls. Rewritten as subgroups of one invocation, which every warp-level algorithm must already handle (a warp with one active lane): the
// invocation's id is 0, the subgroup size 1, a ballot is its predicate in bit 0, the equal / less-or-equal / greater-or-equal masks are bit 0, the less / greater masks 0,
// a shuffle or broadcast returns its own value, all / any return the predicate, all-equal and elect true. Ballot counts, finds and inverse ballots are refused (none met)
enum { OpTypeBool = 20, OpConstantTrue = 41, OpConstant = 43, OpConstantComposite = 44, OpCompositeConstruct = 80, OpCopyObject = 83, OpSelect = 169,
       OpGNUElect = 333, OpGNUAll = 334, OpGNUAny = 335, OpGNUAllEqual = 336, OpGNUBroadcast = 337, OpGNUBroadcastFirst = 338, OpGNUBallot = 339,
       OpGNUShuffle = 345, OpGNUShuffleXor = 346, OpGNUShuffleUp = 347, OpGNUShuffleDown = 348 };
enum { DecBuiltIn = 11, BISubgroupSize = 36, BISubgroupLocalInvocationId = 41, BIEq = 4416, BIGe = 4417, BIGt = 4418, BILe = 4419, BILt = 4420 };
static int bi_kind(uint32_t b) {   // 1 = value 0 (id, lt, gt), 2 = value 1 (size, eq, le, ge), 0 = not a subgroup built-in
    switch (b) { case BISubgroupLocalInvocationId: case BILt: case BIGt: return 1; case BISubgroupSize: case BIEq: case BILe: case BIGe: return 2; default: return 0; }
}
uint32_t *vka_spv_subgroup_single(const uint32_t *w, size_t nw, size_t *out_nw, int *changed, char *err, size_t errn) {
#define FAIL(...) do { snprintf(err, errn, __VA_ARGS__); goto fail; } while (0)
    uint8_t *bi = NULL, *chain = NULL; int32_t *cval = NULL; vec_t o = {0}; *changed = 0;
    if (nw < 5 || w[0] != 0x07230203) { snprintf(err, errn, "not SPIR-V"); return NULL; }
    uint32_t bound = w[3], nb = bound;
    bi = calloc(bound, 1); chain = calloc(bound, 1); cval = malloc((size_t)bound * 4);
    if (!bi || !chain || !cval) FAIL("out of memory");
    for (uint32_t i = 0; i < bound; i++) cval[i] = -1;
    uint32_t t_u32 = 0, t_v4u = 0, t_bool = 0; size_t first_fn = nw; int any = 0;
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > nw) FAIL("bad instruction at word %zu", i);
        if (op == OpDecorate && len >= 4 && w[i + 2] == DecBuiltIn && w[i + 1] < bound && bi_kind(w[i + 3])) bi[w[i + 1]] = (uint8_t)bi_kind(w[i + 3]) | (w[i + 3] >= BIEq ? 4 : 0);   // bit 2: a vec4 mask
        else if (op == OpTypeInt && len >= 4 && w[i + 2] == 32 && !w[i + 3]) t_u32 = w[i + 1];
        else if (op == OpTypeBool && len >= 2) t_bool = w[i + 1];
        else if (op == OpTypeVector && len >= 4 && w[i + 3] == 4 && w[i + 2] == t_u32 && t_u32) t_v4u = w[i + 1];
        else if (op == OpConstant && len >= 4 && w[i + 2] < bound) cval[w[i + 2]] = (int32_t)w[i + 3];
        else if (op == OpFunction && first_fn == nw) first_fn = i;
        else if (op >= OpGNUElect && op <= OpGNUShuffleDown) any = 1;
        i += len;
    }
    for (uint32_t i = 0; i < bound && !any; i++) any = bi[i] != 0;
    if (!any) { free(bi); free(chain); free(cval); return NULL; }   // nothing to do
    if (!t_u32) FAIL("no 32-bit unsigned type");
    uint32_t mk_v4 = t_v4u ? 0 : nb++, mk_bool = t_bool ? 0 : nb++; if (!t_v4u) t_v4u = mk_v4; if (!t_bool) t_bool = mk_bool;
    uint32_t c0 = nb++, c1 = nb++, cv0 = nb++, cv1 = nb++, ctrue = nb++;   // uint 0, uint 1, uvec4(0), uvec4(1, 0, 0, 0), true
    if (!put_n(&o, w, 5)) FAIL("out of memory");
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; const uint32_t *in = w + i;
        if (i == first_fn) {
            if (mk_bool) { uint32_t x[2] = {2u << 16 | OpTypeBool, mk_bool}; if (!put_n(&o, x, 2)) FAIL("out of memory"); }
            if (mk_v4) { uint32_t x[4] = {4u << 16 | OpTypeVector, mk_v4, t_u32, 4}; if (!put_n(&o, x, 4)) FAIL("out of memory"); }
            uint32_t x[] = {4u << 16 | OpConstant, t_u32, c0, 0, 4u << 16 | OpConstant, t_u32, c1, 1, 7u << 16 | OpConstantComposite, t_v4u, cv0, c0, c0, c0, c0,
                            7u << 16 | OpConstantComposite, t_v4u, cv1, c1, c0, c0, c0, 3u << 16 | OpConstantTrue, t_bool, ctrue};
            if (!put_n(&o, x, sizeof x / 4)) FAIL("out of memory");
        }
        if (i >= first_fn) {
            if ((op == OpAccessChain || op == OpInBoundsAccessChain) && len >= 4 && w[i + 3] < bound && bi[w[i + 3]]) {   // a component of a mask
                if (len != 5 || w[i + 4] >= bound || cval[w[i + 4]] < 0) FAIL("an access chain into a subgroup mask with a non-constant index");
                int k = bi[w[i + 3]] & 3; chain[w[i + 2]] = (uint8_t)(k == 2 && cval[w[i + 4]] == 0 ? 2 : 1);   // bit 0 only for eq / le / ge
                i += len; continue;   // dropped: its loads become constants
            }
            if (op == OpLoad && len >= 4 && w[i + 3] < bound && (bi[w[i + 3]] || chain[w[i + 3]])) {
                uint32_t v = chain[w[i + 3]] ? (chain[w[i + 3]] == 2 ? c1 : c0) : (bi[w[i + 3]] & 4) ? ((bi[w[i + 3]] & 3) == 2 ? cv1 : cv0) : ((bi[w[i + 3]] & 3) == 2 ? c1 : c0);
                uint32_t x[4] = {4u << 16 | OpCopyObject, w[i + 1], w[i + 2], v}; if (!put_n(&o, x, 4)) FAIL("out of memory");
                i += len; *changed = 1; continue;
            }
            if (op >= OpGNUElect && op <= OpGNUShuffleDown) {
                uint32_t rt = w[i + 1], r = w[i + 2], v;
                switch (op) {
                case OpGNUElect: case OpGNUAllEqual: v = ctrue; break;
                case OpGNUAll: case OpGNUAny: case OpGNUBroadcast: case OpGNUBroadcastFirst: case OpGNUShuffle: case OpGNUShuffleXor: case OpGNUShuffleUp: case OpGNUShuffleDown:
                    if (len < 5) FAIL("short subgroup instruction %u", op);
                    v = w[i + 4]; break;
                case OpGNUBallot: {
                    if (len < 5 || rt != t_v4u) FAIL("a ballot of an unexpected type");
                    uint32_t tmp = nb++;
                    uint32_t x[] = {6u << 16 | OpSelect, t_u32, tmp, w[i + 4], c1, c0, 7u << 16 | OpCompositeConstruct, t_v4u, r, tmp, c0, c0, c0};
                    if (!put_n(&o, x, sizeof x / 4)) FAIL("out of memory");
                    i += len; *changed = 1; continue; }
                default: FAIL("subgroup instruction %u not rewritten", op);
                }
                uint32_t x[4] = {4u << 16 | OpCopyObject, rt, r, v}; if (!put_n(&o, x, 4)) FAIL("out of memory");
                i += len; *changed = 1; continue;
            }
        }
        if (!put_n(&o, in, len)) FAIL("out of memory");
        i += len;
    }
    o.w[3] = nb;
    free(bi); free(chain); free(cval);
    *out_nw = o.n; return o.w;
fail:
    free(bi); free(chain); free(cval); free(o.w);
    return NULL;
#undef FAIL
}

// vka_spv_subgroup_from_index: a task or mesh module's subgroup built-ins taken from LocalInvocationIndex (a driver workaround, Qualcomm Adreno 840, driver 0x8034a013).
// The translator's task and mesh modules number a workgroup's 32 invocations (Maxwell's warp) by SubgroupLocalInvocationId and compact their output with
// SubgroupLtMask and ballots. Measured on the phone with the game's own modules in a standalone mesh pipeline (Ocarina of Time, mesh 25c4a740753e7d7a): in these modules
// the driver hands every invocation SubgroupLocalInvocationId 0 and SubgroupLtMask 0, while OpGroupNonUniformBallot is right, with the whole workgroup in one subgroup
// and its bits in LocalInvocationIndex order (a ballot of true 0xffffffff, of "index odd" 0xaaaaaaaa, of "index < 3" 7). So every invocation wrote the outputs of lane 0:
// garbage triangles. The same built-ins read correctly in modules built by glslc; what in the translated module's form sets the driver off is open (a reduced module
// that still fails has only the two loads, a ballot and three stores). The rewrite: every load of SubgroupLocalInvocationId loads LocalInvocationIndex, and every load
// of a subgroup mask (or of a component through an access chain with a constant index) becomes that mask of the index: bit = 1 << (index & 31); eq = bit, lt = bit - 1,
// le = lt | eq, ge = ~lt, gt = ~le in word 0, and word 1 empty (full for ge and gt). Right only where a workgroup is one subgroup in index order and has at most 32
// invocations (the index below 32): the caller's to check, which is why vka_shader.c applies it on this driver alone. The masks are kept this plain on purpose: the form
// that also chose the word by index >> 5 (two selects for each mask) made the driver's shader compiler run out of memory and abort the process in vkCreateGraphicsPipelines
// on Ocarina of Time's mesh eed1add7c89f5d34; that module's pipeline compiles in 1.4 s untouched and 3.3 s rewritten (compile only, on the phone), why slower is open.
// A module without the LocalInvocationIndex variable has its SubgroupLocalInvocationId variable decorated as that instead. NULL with err empty when the module reads
// none of these built-ins
enum { OpIdxISub = 130, OpIdxShl = 196, OpIdxOr = 197, OpIdxAnd = 199, OpIdxNot = 200, BILocalInvocationIndex = 29 };
uint32_t *vka_spv_subgroup_from_index(const uint32_t *w, size_t nw, size_t *out_nw, int *changed, char *err, size_t errn) {
#define FAIL(...) do { snprintf(err, errn, __VA_ARGS__); goto fail; } while (0)
#define EMIT(...) do { uint32_t x_[] = {__VA_ARGS__}; if (!put_n(&o, x_, sizeof x_ / 4)) FAIL("out of memory"); } while (0)
    uint8_t *kind = NULL, *chain = NULL; int32_t *cval = NULL; vec_t o = {0}; *changed = 0;   // kind[id]: 1 the id, 2 eq, 3 ge, 4 gt, 5 le, 6 lt; chain[id]: kind | component << 4
    if (nw < 5 || w[0] != 0x07230203) { snprintf(err, errn, "not SPIR-V"); return NULL; }
    uint32_t bound = w[3], nb = bound;
    kind = calloc(bound, 1); chain = calloc(bound, 1); cval = malloc((size_t)bound * 4);
    if (!kind || !chain || !cval) FAIL("out of memory");
    for (uint32_t i = 0; i < bound; i++) cval[i] = -1;
    uint32_t t_u32 = 0, t_v4u = 0, l_var = 0, s_var = 0; size_t first_fn = nw, s_dec = 0; int any = 0, masks = 0;
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > nw) FAIL("bad instruction at word %zu", i);
        if (op == OpDecorate && len >= 4 && w[i + 2] == DecBuiltIn && w[i + 1] < bound) {
            uint32_t b = w[i + 3], id = w[i + 1];
            if (b == BILocalInvocationIndex) l_var = id;
            else if (b == BISubgroupLocalInvocationId) { kind[id] = 1; s_var = id; s_dec = i; any = 1; }
            else if (b >= BIEq && b <= BILt) { kind[id] = (uint8_t)(2 + b - BIEq); any = masks = 1; }
        } else if (op == OpTypeInt && len >= 4 && w[i + 2] == 32 && !w[i + 3]) t_u32 = w[i + 1];
        else if (op == OpTypeVector && len >= 4 && w[i + 3] == 4 && w[i + 2] == t_u32 && t_u32) t_v4u = w[i + 1];
        else if (op == OpConstant && len >= 4 && w[i + 2] < bound) cval[w[i + 2]] = (int32_t)w[i + 3];
        else if (op == OpFunction && first_fn == nw) first_fn = i;
        i += len;
    }
    if (!any) { free(kind); free(chain); free(cval); return NULL; }   // nothing to do
    if (!t_u32) FAIL("no 32-bit unsigned type");
    if (masks && !t_v4u) FAIL("a subgroup mask without a uvec4 type");
    int redecorate = 0;
    if (!l_var) { if (!s_var) FAIL("subgroup masks in a module without LocalInvocationIndex or SubgroupLocalInvocationId"); l_var = s_var; redecorate = 1; kind[s_var] = 0; }
    uint32_t c0 = nb++, c1 = nb++, c31 = nb++, cff = nb++;
    if (!put_n(&o, w, 5)) FAIL("out of memory");
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; const uint32_t *in = w + i;
        if (i == first_fn) {
            EMIT(4u << 16 | OpConstant, t_u32, c0, 0, 4u << 16 | OpConstant, t_u32, c1, 1, 4u << 16 | OpConstant, t_u32, c31, 31, 4u << 16 | OpConstant, t_u32, cff, 0xffffffffu);
        }
        if (redecorate && i == s_dec) { EMIT(in[0], in[1], in[2], BILocalInvocationIndex); i += len; *changed = 1; continue; }
        if (i >= first_fn) {
            if ((op == OpAccessChain || op == OpInBoundsAccessChain) && len >= 4 && w[i + 3] < bound && kind[w[i + 3]] >= 2) {   // a component of a mask
                if (len != 5 || w[i + 4] >= bound || cval[w[i + 4]] < 0 || cval[w[i + 4]] > 3) FAIL("an access chain into a subgroup mask with a non-constant index");
                chain[w[i + 2]] = (uint8_t)(kind[w[i + 3]] | cval[w[i + 4]] << 4);
                i += len; continue;   // dropped: its loads are computed
            }
            if (op == OpLoad && len >= 4 && w[i + 3] < bound && kind[w[i + 3]] == 1) {   // the invocation's id: the index
                EMIT(4u << 16 | OpLoad, w[i + 1], w[i + 2], l_var);
                i += len; *changed = 1; continue;
            }
            if (op == OpLoad && len >= 4 && w[i + 3] < bound && (kind[w[i + 3]] >= 2 || chain[w[i + 3]])) {
                int k = chain[w[i + 3]] ? chain[w[i + 3]] & 15 : kind[w[i + 3]], comp = chain[w[i + 3]] ? chain[w[i + 3]] >> 4 : -1;
                uint32_t L = nb++, sh = nb++, bit = nb++, ltw = nb++, x = nb++, y = k >= 3 && k <= 4 ? cff : c0;   // the index is below 32: word 1 is empty, or full for ge / gt
                EMIT(4u << 16 | OpLoad, t_u32, L, l_var, 5u << 16 | OpIdxAnd, t_u32, sh, L, c31, 5u << 16 | OpIdxShl, t_u32, bit, c1, sh);
                if (k == 2) EMIT(4u << 16 | OpCopyObject, t_u32, x, bit);   // eq
                else {
                    EMIT(5u << 16 | OpIdxISub, t_u32, ltw, bit, c1);
                    if (k == 6) EMIT(4u << 16 | OpCopyObject, t_u32, x, ltw);   // lt
                    else if (k == 3) EMIT(4u << 16 | OpIdxNot, t_u32, x, ltw);   // ge
                    else { uint32_t le = k == 5 ? x : nb++; EMIT(5u << 16 | OpIdxOr, t_u32, le, ltw, bit); if (k == 4) EMIT(4u << 16 | OpIdxNot, t_u32, x, le); }   // le, gt
                }
                if (comp < 0) EMIT(7u << 16 | OpCompositeConstruct, w[i + 1], w[i + 2], x, y, c0, c0);
                else EMIT(4u << 16 | OpCopyObject, w[i + 1], w[i + 2], comp == 0 ? x : comp == 1 ? y : c0);
                i += len; *changed = 1; continue;
            }
        }
        if (!put_n(&o, in, len)) FAIL("out of memory");
        i += len;
    }
    o.w[3] = nb;
    free(kind); free(chain); free(cval);
    *out_nw = o.n; return o.w;
fail:
    free(kind); free(chain); free(cval); free(o.w);
    return NULL;
#undef EMIT
#undef FAIL
}

// vka_spv_gs_claim: a geometry shader that writes memory, made to write it once per primitive. The Adreno 840 runs a geometry shader several times for each primitive (three
// times for a triangle in and three vertices out, counted with an atomic add per gl_PrimitiveIDIn), each run executing the whole body: whatever the shader writes to memory
// is written that many times. Donkey Kong Bananza's terrain culling shader 5d537f7630ec3671 added each surviving triangle 3 times to its chunk's indirect draw record and wrote
// its indices 3 times, past the chunk's index buffer into the next chunk's: black holes in the cave walls. Vulkan says nothing of how many times a geometry shader runs, so the
// module claims its primitive first: one word per primitive in a storage buffer the replay binds at set 0 binding `binding` (VKA_GS_CLAIM_BINDING, zeroed for each draw), set by an atomic exchange
// at gl_PrimitiveIDIn; only the run that finds it 0 writes memory. Every store and copy into Uniform / StorageBuffer / PhysicalStorageBuffer / Image memory, image write and
// atomic is put under that condition (its block split around it); the other runs see an atomic's result as the memory's value (a compare-exchange its comparator, so a
// compare-exchange loop ends). What the shader emits is left alone: every run emits as before. Returns the count of memory writes (0: nothing to do, *out NULL), -1 with err
// on failure; with out and a count above 0, the rewritten module (malloc'd, *out_nw words)
enum { OpUndef = 1, OpExecutionMode = 16, OpTypeRuntimeArray = 29, OpTypeStruct = 30, OpTypeForwardPointer = 39,
       OpFunctionEnd = 56, OpFunctionParameter = 55, OpImageTexelPointer = 60, OpStore = 62, OpCopyMemory = 63, OpCopyMemorySized = 64, OpPtrAccessChain = 67,
       OpInBoundsPtrAccessChain = 70, OpImageWrite = 99, OpConvertUToPtr = 120, OpIEqual = 170, OpPhi = 245, OpLoopMerge = 246, OpSelectionMerge = 247, OpLabel = 248,
       OpBranch = 249, OpBranchConditional = 250, OpSwitch = 251, OpKill = 252, OpReturn = 253, OpReturnValue = 254, OpUnreachable = 255, OpTerminateInvocation = 4416,
       OpAtomicStore = 228, OpAtomicExchange = 229, OpAtomicCompareExchange = 230, OpAtomicCompareExchangeWeak = 231, OpAtomicXor = 242,
       OpAtomicFMinEXT = 5614, OpAtomicFMaxEXT = 5615, OpAtomicFAddEXT = 6035, OpDecorationGroup = 73, OpGroupDecorate = 74, OpGroupMemberDecorate = 75 };
enum { SCUniform = 2, SCPrivate = 6, SCImage = 11, SCStorageBuffer = 12, SCPhysicalStorageBuffer = 5349, DecBufferBlock = 3, DecArrayStride = 6, DecOffset = 35,
       DecBinding = 33, DecDescriptorSet = 34, BIPrimitiveId = 7 };
static int sc_mem(uint32_t sc) { return sc == SCUniform || sc == SCImage || sc == SCStorageBuffer || sc == SCPhysicalStorageBuffer; }
static int is_term(uint32_t op) { return op == OpBranch || op == OpBranchConditional || op == OpSwitch || op == OpKill || op == OpReturn || op == OpReturnValue || op == OpUnreachable || op == OpTerminateInvocation; }
static int is_annot(uint32_t op) { return op == OpDecorate || op == OpMemberDecorate || op == OpDecorationGroup || op == OpGroupDecorate || op == OpGroupMemberDecorate || op == OpDecorateId || op == OpDecorateString || op == OpMemberDecorateString; }
// One pass of the rewrite, run twice: the first records in rec where each split block of the input ends, the second gives that to the phis (last) that name such a block
// as their parent (the new ids are the same in both passes)
static int claim_pass(const uint32_t *w, size_t nw, uint32_t binding, const uint32_t *last, uint32_t *rec, vec_t *o, int *found, char *err, size_t errn) {
#define FAIL(...) do { snprintf(err, errn, __VA_ARGS__); goto fail; } while (0)
    uint32_t bound = w[3], nb = bound;
    uint32_t *psc = calloc(bound, 4), *ty = calloc(bound, 4);   // psc[pointer type]: storage class + 1; ty[id]: the type of a pointer-valued id
    if (!psc || !ty) FAIL("out of memory");
    uint32_t t_bool = 0, t_u32 = 0, t_i32 = 0, p_in_i32 = 0, p_priv_bool = 0, p_uni_u32 = 0, main_fn = 0, prim_var = 0;
    size_t first_fn = nw, first_annot = nw, ep_at = nw;
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > nw) FAIL("bad instruction at word %zu", i);
        if ((op == OpTypePointer || op == OpTypeForwardPointer) && len >= 3 && w[i + 1] < bound) psc[w[i + 1]] = w[i + 2] + 1;
        if (op == OpTypePointer && len >= 4) {
            if (w[i + 2] == SCInput && w[i + 3] && w[i + 3] == t_i32) p_in_i32 = w[i + 1];
            if (w[i + 2] == SCPrivate && w[i + 3] && w[i + 3] == t_bool) p_priv_bool = w[i + 1];
            if (w[i + 2] == SCUniform && w[i + 3] && w[i + 3] == t_u32) p_uni_u32 = w[i + 1];
        }
        else if ((op == OpVariable || op == OpAccessChain || op == OpInBoundsAccessChain || op == OpPtrAccessChain || op == OpInBoundsPtrAccessChain || op == OpConvertUToPtr ||
                  op == OpBitcast || op == OpCopyObject || op == OpImageTexelPointer || op == OpFunctionParameter) && len >= 3 && w[i + 2] < bound) ty[w[i + 2]] = w[i + 1];
        if (op == OpTypeBool) t_bool = w[i + 1];
        else if (op == OpTypeInt && len >= 4 && w[i + 2] == 32) { if (w[i + 3]) t_i32 = w[i + 1]; else t_u32 = w[i + 1]; }
        else if (op == OpEntryPoint && len >= 3) { if (main_fn) FAIL("more than one entry point"); main_fn = w[i + 2]; ep_at = i; }
        else if (op == OpDecorate && len >= 4 && w[i + 2] == DecBuiltIn && w[i + 3] == BIPrimitiveId) prim_var = w[i + 1];
        else if (op == OpExecutionMode && len >= 4 && w[i + 2] == 0 && w[i + 3] > 1) FAIL("%u invocations a primitive: they would share its claim", w[i + 3]);   // Invocations
        if (is_annot(op) && first_annot == nw) first_annot = i;
        if (op == OpFunction && first_fn == nw) first_fn = i;
        i += len;
    }
    if (!main_fn || first_fn == nw || !t_u32) FAIL("no entry point, function or 32-bit unsigned type");
    if (first_annot > first_fn) first_annot = first_fn;
    if (prim_var && (!p_in_i32 || ty[prim_var] != p_in_i32)) FAIL("a PrimitiveId input that is not a 32-bit int");
    // New ids: types missing, the claim buffer's types and variable, the constants, the guard variable
    uint32_t mk_bool = t_bool ? 0 : nb++, mk_i32 = t_i32 ? 0 : nb++; if (!t_bool) t_bool = mk_bool; if (!t_i32) t_i32 = mk_i32;
    uint32_t mk_pin = p_in_i32 && !mk_i32 ? 0 : nb++, mk_ppb = p_priv_bool && !mk_bool ? 0 : nb++, mk_pu = p_uni_u32 ? 0 : nb++;
    if (mk_pin) p_in_i32 = mk_pin; if (mk_ppb) p_priv_bool = mk_ppb; if (mk_pu) p_uni_u32 = mk_pu;
    uint32_t mk_prim = prim_var ? 0 : nb++; if (!prim_var) prim_var = mk_prim;
    uint32_t rta = nb++, st = nb++, pst = nb++, cvar = nb++, c0 = nb++, c1 = nb++, gvar = nb++;
    if (!put_n(o, w, 5)) FAIL("out of memory");
    uint32_t cur = 0, cur_orig = 0; int in_fn = 0, in_main_entry = 0;
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; const uint32_t *in = w + i;
        if (i == ep_at && !mk_prim) { if (!put_n(o, in, len)) FAIL("out of memory"); i += len; continue; }
        if (i == ep_at) {   // the PrimitiveId input joins the interface
            if (!put(o, (len + 1) << 16 | OpEntryPoint) || !put_n(o, in + 1, len - 1) || !put(o, prim_var)) FAIL("out of memory");
            i += len; continue;
        }
        if (i == first_annot) {
            uint32_t x[] = {4u << 16 | OpDecorate, rta, DecArrayStride, 4, 3u << 16 | OpDecorate, st, DecBufferBlock, 5u << 16 | OpMemberDecorate, st, 0, DecOffset, 0,
                            4u << 16 | OpDecorate, cvar, DecDescriptorSet, 0, 4u << 16 | OpDecorate, cvar, DecBinding, binding};
            if (!put_n(o, x, sizeof x / 4)) FAIL("out of memory");
            if (mk_prim) { uint32_t y[] = {4u << 16 | OpDecorate, mk_prim, DecBuiltIn, BIPrimitiveId}; if (!put_n(o, y, 4)) FAIL("out of memory"); }
        }
        if (i == first_fn) {
            if (mk_bool) { uint32_t x[] = {2u << 16 | OpTypeBool, mk_bool}; if (!put_n(o, x, 2)) FAIL("out of memory"); }
            if (mk_i32) { uint32_t x[] = {4u << 16 | OpTypeInt, mk_i32, 32, 1}; if (!put_n(o, x, 4)) FAIL("out of memory"); }
            if (mk_pin) { uint32_t x[] = {4u << 16 | OpTypePointer, mk_pin, SCInput, t_i32}; if (!put_n(o, x, 4)) FAIL("out of memory"); }
            if (mk_ppb) { uint32_t x[] = {4u << 16 | OpTypePointer, mk_ppb, SCPrivate, t_bool}; if (!put_n(o, x, 4)) FAIL("out of memory"); }
            if (mk_pu) { uint32_t x[] = {4u << 16 | OpTypePointer, mk_pu, SCUniform, t_u32}; if (!put_n(o, x, 4)) FAIL("out of memory"); }
            if (mk_prim) { uint32_t x[] = {4u << 16 | OpVariable, p_in_i32, mk_prim, SCInput}; if (!put_n(o, x, 4)) FAIL("out of memory"); }
            uint32_t x[] = {3u << 16 | OpTypeRuntimeArray, rta, t_u32, 3u << 16 | OpTypeStruct, st, rta, 4u << 16 | OpTypePointer, pst, SCUniform, st,
                            4u << 16 | OpVariable, pst, cvar, SCUniform, 4u << 16 | OpConstant, t_u32, c0, 0, 4u << 16 | OpConstant, t_u32, c1, 1,
                            4u << 16 | OpVariable, p_priv_bool, gvar, SCPrivate};
            if (!put_n(o, x, sizeof x / 4)) FAIL("out of memory");
        }
        if (op == OpFunction) { in_fn = 1; in_main_entry = w[i + 2] == main_fn ? 1 : 0; }
        else if (op == OpFunctionEnd) in_fn = 0;
        if (op == OpLabel) { cur = cur_orig = w[i + 1]; if (!put_n(o, in, len)) FAIL("out of memory"); i += len; continue; }
        if (in_main_entry == 1 && op != OpVariable && cur) {   // the claim, after the entry block's variables
            uint32_t pid = nb++, ptr = nb++, old = nb++, first = nb++;
            uint32_t x[] = {4u << 16 | OpLoad, t_i32, pid, prim_var, 6u << 16 | OpAccessChain, p_uni_u32, ptr, cvar, c0, pid,
                            7u << 16 | OpAtomicExchange, t_u32, old, ptr, c1, c0, c1, 5u << 16 | OpIEqual, t_bool, first, old, c0, 3u << 16 | OpStore, gvar, first};
            if (!put_n(o, x, sizeof x / 4)) FAIL("out of memory");
            in_main_entry = 2;
        }
        if (op == OpPhi && last) {   // a parent block that was split ends in its last piece
            if (!put_n(o, in, 3)) FAIL("out of memory");
            for (uint32_t k = 3; k + 1 < len; k += 2) { uint32_t par = in[k + 1]; if (!put(o, in[k]) || !put(o, par < bound && last[par] ? last[par] : par)) FAIL("out of memory"); }
            i += len; continue;
        }
        #define SC(id) ((id) < bound && ty[id] < bound && psc[ty[id]] ? psc[ty[id]] - 1 : 0u)
        int drop = 0, atomic = 0;
        if (in_fn && (op == OpStore || op == OpCopyMemory || op == OpCopyMemorySized || op == OpAtomicStore) && len >= 2 && sc_mem(SC(in[1]))) drop = 1;
        else if (in_fn && op == OpImageWrite) drop = 1;
        else if (in_fn && ((op >= OpAtomicExchange && op <= OpAtomicXor) || op == OpAtomicFMinEXT || op == OpAtomicFMaxEXT || op == OpAtomicFAddEXT) && len >= 4 && sc_mem(SC(in[3]))) atomic = 1;
        if (drop || atomic) {
            (*found)++;
            for (size_t j = i; j < nw;) { uint32_t oj = w[j] & 0xffff; if (oj == OpLoopMerge) FAIL("a memory write in a loop header block"); if (is_term(oj)) break; j += w[j] >> 16; }
            uint32_t g = nb++, a = nb++, l = nb++, fb = 0;
            if (atomic) {
                uint32_t sc = SC(in[3]);
                if ((op == OpAtomicCompareExchange || op == OpAtomicCompareExchangeWeak) && len >= 9) fb = in[8];
                else { fb = nb++;
                    if (sc == SCImage) { uint32_t x[] = {3u << 16 | OpUndef, in[1], fb}; if (!put_n(o, x, 3)) FAIL("out of memory"); }
                    else if (sc == SCPhysicalStorageBuffer) { uint32_t x[] = {6u << 16 | OpLoad, in[1], fb, in[3], 2, 4}; if (!put_n(o, x, 6)) FAIL("out of memory"); }   // Aligned 4
                    else { uint32_t x[] = {4u << 16 | OpLoad, in[1], fb, in[3]}; if (!put_n(o, x, 4)) FAIL("out of memory"); } }
            }
            uint32_t x[] = {4u << 16 | OpLoad, t_bool, g, gvar, 3u << 16 | OpSelectionMerge, l, 0, 4u << 16 | OpBranchConditional, g, a, l, 2u << 16 | OpLabel, a};
            if (!put_n(o, x, sizeof x / 4)) FAIL("out of memory");
            if (atomic) { uint32_t r = in[2]; if (!put(o, in[0]) || !put(o, in[1]) || !put(o, nb) || !put_n(o, in + 3, len - 3)) FAIL("out of memory");
                uint32_t ra = nb++, y[] = {2u << 16 | OpBranch, l, 2u << 16 | OpLabel, l, 7u << 16 | OpPhi, in[1], r, ra, a, fb, cur};
                if (!put_n(o, y, sizeof y / 4)) FAIL("out of memory"); }
            else { if (!put_n(o, in, len)) FAIL("out of memory"); uint32_t y[] = {2u << 16 | OpBranch, l, 2u << 16 | OpLabel, l}; if (!put_n(o, y, 4)) FAIL("out of memory"); }
            cur = l; if (rec && cur_orig < bound) rec[cur_orig] = l;
            i += len; continue;
        }
        #undef SC
        if (!put_n(o, in, len)) FAIL("out of memory");
        i += len;
    }
    o->w[3] = nb;
    free(psc); free(ty);
    return 0;
fail:
    free(psc); free(ty);
    return -1;
#undef FAIL
}
int vka_spv_gs_claim(const uint32_t *w, size_t nw, uint32_t binding, uint32_t **out, size_t *out_nw, char *err, size_t errn) {
    if (out) *out = NULL;
    if (nw < 5 || w[0] != 0x07230203) { snprintf(err, errn, "not SPIR-V"); return -1; }
    uint32_t *rec = calloc(w[3], 4); vec_t o = {0}; int found = 0;
    if (!rec) { snprintf(err, errn, "out of memory"); return -1; }
    if (claim_pass(w, nw, binding, NULL, rec, &o, &found, err, errn)) { free(rec); free(o.w); return -1; }
    if (found && out) {
        free(o.w); o = (vec_t){0}; int f2 = 0;
        if (claim_pass(w, nw, binding, rec, NULL, &o, &f2, err, errn)) { free(rec); free(o.w); return -1; }
        *out = o.w; *out_nw = o.n; o.w = NULL;
    }
    free(rec); free(o.w);
    return found;
}

// vka_spv_fout_retype: float colour outputs drawn into integer targets. On Maxwell / Ampere a colour output is a 32-bit register per component and the target's format decides
// how its bits are read: a pixel shader the translator types as float (its outputs computed with float instructions, or moved through float registers) still writes the bits
// it holds into a UINT / SINT target. Vulkan leaves a float output to an integer attachment undefined, and the library's MOD_UINT / MOD_UINT_MIXED variants exist only for
// shaders whose outputs come from integer bits or constants (vk_android_build.py ps_variants). So for a pipeline where a float output meets an integer attachment, the module's
// Output variables at those locations (a float or a float vector) become the unsigned / signed integer type of as many components, each store into one (whole, or a component
// through an access chain) stores the OpBitcast of its value, and each load of one is bit-cast back. The bits reach the target unchanged, as on the Switch.
// Mario Kart World's fragment dcb0504c844f1ecc (DrivePassShaderDefault, paired with task-less mesh programs: by their shape ShadingRateWrite) writes its UINT target 0 this way
enum { SCOutput = 3 };
uint32_t *vka_spv_fout_retype(const uint32_t *w, size_t nw, uint32_t umask, uint32_t smask, size_t *out_nw, char *err, size_t errn) {
#define FAIL(...) do { snprintf(err, errn, __VA_ARGS__); goto fail; } while (0)
    uint32_t *cls = NULL, *loc = NULL, *ty = NULL; vec_t o = {0};   // cls[id]: 1 uint / 2 int, bit 4 the whole variable (else a component chain); ty: for each float type id its components (1..4), 0 none
    if (nw < 5 || w[0] != 0x07230203) { snprintf(err, errn, "not SPIR-V"); return NULL; }
    uint32_t bound = w[3];
    if ((umask | smask) == 0 || (umask & smask)) { snprintf(err, errn, "no output, or an output both uint and int"); return NULL; }
    if (!(cls = calloc(bound, 4)) || !(loc = calloc(bound, 4)) || !(ty = calloc(bound, 4))) FAIL("out of memory");
    for (uint32_t i = 0; i < bound; i++) loc[i] = ~0u;
    uint32_t t_f32 = 0, t_int[2] = {0}, t_ivec[2][5] = {{0}}, p_out_int[2][5] = {{0}};   // [uint/int][components]; the existing types
    uint32_t *ptee = calloc(bound, 4); if (!ptee) FAIL("out of memory");                // Output pointer type → its pointee
    size_t first_fn = nw;
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > nw) { free(ptee); FAIL("bad instruction at word %zu", i); }
        if (op == OpDecorate && len >= 4 && w[i + 2] == DecLocation && w[i + 1] < bound) loc[w[i + 1]] = w[i + 3];
        else if (op == OpTypeFloat && len >= 3 && w[i + 2] == 32 && w[i + 1] < bound) { t_f32 = w[i + 1]; ty[t_f32] = 1; }
        else if (op == OpTypeInt && len >= 4 && w[i + 2] == 32) t_int[w[i + 3] ? 1 : 0] = w[i + 1];
        else if (op == OpTypeVector && len >= 4 && w[i + 3] >= 2 && w[i + 3] <= 4 && w[i + 1] < bound) {
            if (w[i + 2] == t_f32 && t_f32) ty[w[i + 1]] = w[i + 3];
            for (int c = 0; c < 2; c++) if (w[i + 2] == t_int[c] && t_int[c]) t_ivec[c][w[i + 3]] = w[i + 1];
        }
        else if (op == OpTypePointer && len >= 4 && w[i + 2] == SCOutput && w[i + 1] < bound) {
            ptee[w[i + 1]] = w[i + 3];
            for (int c = 0; c < 2; c++) { if (w[i + 3] == t_int[c] && t_int[c]) p_out_int[c][1] = w[i + 1]; for (int n = 2; n <= 4; n++) if (w[i + 3] == t_ivec[c][n] && t_ivec[c][n]) p_out_int[c][n] = w[i + 1]; }
        }
        else if (op == OpFunction && first_fn == nw) first_fn = i;
        i += len;
    }
    for (int c = 0; c < 2; c++) t_ivec[c][1] = t_int[c];
    // The output variables to retype: Location in a mask, a pointer to a float or float vector
    uint32_t moved[32], mcomp[32]; int nmoved = 0; uint32_t found = 0, need[2] = {0};   // need[c]: bit n = the integer type of n components (and its pointers) is used
    for (size_t i = 5; i < first_fn;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == OpVariable && len >= 4 && w[i + 3] == SCOutput && w[i + 1] < bound && w[i + 2] < bound) {
            uint32_t id = w[i + 2], l = loc[id], pt = ptee[w[i + 1]], n = pt < bound ? ty[pt] : 0;
            if (l < 32 && ((umask | smask) >> l & 1)) {
                if (!n) { free(ptee); FAIL("output %u is not a float or float vector", l); }
                if (len > 4) { free(ptee); FAIL("output %u has an initializer", l); }   // (the translator emits none; re-emitted without it, unwritten components would change)
                if (nmoved == 32) { free(ptee); FAIL("too many outputs"); }
                int c = umask >> l & 1 ? 0 : 1;
                cls[id] = (uint32_t)(c + 1) | 16; mcomp[nmoved] = n; moved[nmoved++] = id;
                found |= 1u << l; need[c] |= 1u << n | 2u;   // (the scalar for component chains)
            }
        }
        i += len;
    }
    free(ptee);
    if (found != (umask | smask)) FAIL("outputs %#x are not float Output variables", (umask | smask) & ~found);
    // New ids for the types the module lacks
    uint32_t nb = bound, mk_t[2][5] = {{0}}, mk_p[2][5] = {{0}};
    for (int c = 0; c < 2; c++) for (int n = 1; n <= 4; n++) if (need[c] >> n & 1) {
        if (!t_ivec[c][n]) t_ivec[c][n] = mk_t[c][n] = nb++;
        if (!p_out_int[c][n]) p_out_int[c][n] = mk_p[c][n] = nb++;
    }
    if ((need[0] && !t_ivec[0][1]) || (need[1] && !t_ivec[1][1])) FAIL("no scalar type");
    // Pass 2: copy, retyping (the new types and the moved variables go right before the first function, as in vka_spv_vin_retype)
    if (!put_n(&o, w, 5)) FAIL("out of memory");
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; const uint32_t *in = w + i;
        if (i == first_fn) {
            for (int c = 0; c < 2; c++) {
                if (mk_t[c][1]) { uint32_t x[4] = {4u << 16 | OpTypeInt, mk_t[c][1], 32, (uint32_t)c}; if (!put_n(&o, x, 4)) FAIL("out of memory"); }
                for (int n = 2; n <= 4; n++) if (mk_t[c][n]) { uint32_t x[4] = {4u << 16 | OpTypeVector, mk_t[c][n], t_ivec[c][1], (uint32_t)n}; if (!put_n(&o, x, 4)) FAIL("out of memory"); }
                for (int n = 1; n <= 4; n++) if (mk_p[c][n]) { uint32_t x[4] = {4u << 16 | OpTypePointer, mk_p[c][n], SCOutput, t_ivec[c][n]}; if (!put_n(&o, x, 4)) FAIL("out of memory"); }
            }
            for (int k = 0; k < nmoved; k++) { uint32_t x[4] = {4u << 16 | OpVariable, p_out_int[(cls[moved[k]] & 3) - 1][mcomp[k]], moved[k], SCOutput}; if (!put_n(&o, x, 4)) FAIL("out of memory"); }
        }
        if (op == OpVariable && len >= 4 && w[i + 2] < bound && cls[w[i + 2]] && i < first_fn) { i += len; continue; }   // re-emitted above
        if ((op == OpAccessChain || op == OpInBoundsAccessChain) && len >= 4 && w[i + 3] < bound && cls[w[i + 3]]) {
            int c = (cls[w[i + 3]] & 3) - 1; uint32_t r = w[i + 2];
            if (r >= bound) FAIL("bad id");
            if (len != 5 || !(cls[w[i + 3]] & 16)) FAIL("an access chain of %u indexes into a retyped output", len - 4);
            cls[r] = (uint32_t)(c + 1);   // a component: pointer to the integer scalar
            if (!put_n(&o, in, len)) FAIL("out of memory");
            o.w[o.n - len + 1] = p_out_int[c][1];
            i += len; continue;
        }
        if (op == OpStore && len >= 3 && w[i + 1] < bound && cls[w[i + 1]]) {
            uint32_t p = w[i + 1], c = (cls[p] & 3) - 1, n = 1;
            if (cls[p] & 16) { for (int k = 0; k < nmoved; k++) if (moved[k] == p) n = mcomp[k]; }
            uint32_t tmp = nb++;
            uint32_t x[4] = {4u << 16 | OpBitcast, t_ivec[c][n], tmp, w[i + 2]}; if (!put_n(&o, x, 4)) FAIL("out of memory");
            if (!put_n(&o, in, len)) FAIL("out of memory");
            o.w[o.n - len + 2] = tmp;
            i += len; continue;
        }
        if (op == OpLoad && len >= 4 && w[i + 3] < bound && cls[w[i + 3]]) {
            uint32_t p = w[i + 3], c = (cls[p] & 3) - 1, n = 1, ft = w[i + 1], r = w[i + 2], tmp = nb++;
            if (cls[p] & 16) { for (int k = 0; k < nmoved; k++) if (moved[k] == p) n = mcomp[k]; }
            if (!put_n(&o, in, len)) FAIL("out of memory");
            o.w[o.n - len + 1] = t_ivec[c][n]; o.w[o.n - len + 2] = tmp;
            uint32_t x[4] = {4u << 16 | OpBitcast, ft, r, tmp}; if (!put_n(&o, x, 4)) FAIL("out of memory");
            i += len; continue;
        }
        if (op != OpName && op != OpMemberName && op != OpEntryPoint && op != OpDecorate && op != OpMemberDecorate && op != OpDecorateId && op != OpDecorateString && op != OpMemberDecorateString
            && i >= first_fn)
            for (uint32_t k = 1; k < len; k++) if (!(op == OpExtInst && k == 4) && w[i + k] < bound && cls[w[i + k]]) FAIL("opcode %u uses a retyped output (id %u)", op, w[i + k]);
        if (!put_n(&o, in, len)) FAIL("out of memory");
        i += len;
    }
    o.w[3] = nb;
    free(cls); free(loc); free(ty);
    *out_nw = o.n; return o.w;
fail:
    free(cls); free(loc); free(ty); free(o.w);
    return NULL;
#undef FAIL
}

// vka_spv_mesh_store_planes: a mesh module's calls of the translator's mesh_plane_store, each sent to a copy of it made for the one plane the call can reach.
// The translator (SpirvMeshEntry.cs) turns each guest store to output memory into mesh_plane_store(word, value): the word address is split into a chunk (32 lanes
// times the planes), rem = word % chunk, plane = rem >> 5 and a vertex index, followed by one "if (plane == p) output[index].component = value" for every plane (24
// of them in Ocarina of Time's mesh 4c7d6be5a7717c8d, which calls it 216 times). The address a call passes is (lane part * 4 * chunk + (x & 124) + C) >> 2 with C a
// constant, so its plane is fixed, but only arithmetic on the lane shows it. Qualcomm's compiler (Adreno 840, driver 0x8034a013) does not see it once the lane is a
// real value (vka_spv_subgroup_from_index: before it the lane was the constant 0 and every call folded to one store): it keeps the whole chain of variable-index
// output stores at each call. Measured compile-only on the phone for that module: 7.1 s and 1.7 GB of peak memory, 0.9 s and 126 MB with the calls specialised; in the
// game the 1.7 GB went through the allocator's large-block path as about 13,000 separate mappings (each with a guard mapping), which with the game's own took the
// process to the kernel's mapping limit (66,283 lines in /proc/pid/maps): mmap failed, std::bad_alloc on the replay thread in vkCreateGraphicsPipelines, abort.
// The rewrite finds, for each call, the address as g * t + v with v in lo .. hi (lin_of: constants, add, multiply and shift by a constant, and with a constant mask,
// and loads of a function's own variable that is only loaded and stored whole, as any of the values stored into it). The plane is known when g is a multiple of the
// chunk and lo .. hi lies in one plane of it; the call then goes to a copy of the function in which each comparison of the plane with a constant is that constant
// result. Each scatter of the function (vertex: word % chunk; primitive: (word - base) % chunk) is settled on its own, and a call with none known is left alone. The
// result differs from the module only for an address past 32 bits (g * t wrapping), which no output has. *changed counts the calls redirected; NULL with err empty
// when the module has no such function or no call to settle. The better home for this is the translator (a helper per plane, or the stores written in line), whose
// binary is pinned: open
enum { OpConstantFalse = 42, OpFunctionCall = 57, OpIAdd = 128, OpIMul = 132, OpUDiv = 134, OpUMod = 137, OpLogicalNot = 168, OpULessThan = 176, OpShiftRightLogical = 194,
       SCFunction = 7 };
enum { F_T32 = 1, F_C32 = 2, F_VAR = 4, F_DIRTY = 8 };   // fl[id]: a 32-bit integer type, a constant of one, a Function variable, one used other than by whole loads and stores
typedef struct { int64_t g, lo, hi; } lin_t;   // the value is g * t + v for some integer t, v in lo .. hi; g = 1 says nothing
typedef struct { const uint32_t *w; uint32_t bound; const uint32_t *def; const uint8_t *fl; const uint32_t *st; size_t nst; lin_t *memo; uint8_t *state; } lin_ctx;
static int64_t gcd64(int64_t a, int64_t b) { while (b) { int64_t t = a % b; a = b; b = t; } return a; }
static lin_t lin_of(lin_ctx *c, uint32_t id, int depth) {
    const lin_t any = {1, 0, 0}; const int64_t lim = (int64_t)1 << 31;
    if (id >= c->bound || !c->def[id] || depth > 48 || c->state[id] == 1) return any;   // state 1: a cycle through a variable
    if (c->state[id] == 2) return c->memo[id];
    const uint32_t *in = c->w + c->def[id]; uint32_t op = in[0] & 0xffff, len = in[0] >> 16; lin_t r = any;
#define C32(x) ((x) < c->bound && (c->fl[x] & F_C32) && c->w[c->def[x] + 3] < 0x80000000u)
#define CV(x) ((int64_t)c->w[c->def[x] + 3])
    c->state[id] = 1;
    if (op == OpConstant) { if (C32(id)) r = (lin_t){0, CV(id), CV(id)}; }
    else if ((op == OpBitcast || op == OpCopyObject) && len == 4) r = lin_of(c, in[3], depth + 1);
    else if (op == OpLoad && len >= 4) { if (in[3] < c->bound && (c->fl[in[3]] & (F_VAR | F_DIRTY)) == F_VAR) r = lin_of(c, in[3], depth + 1); }
    else if (op == OpVariable) {   // any of the values stored into it
        int n = 0;
        for (size_t k = 0; k < c->nst; k += 2) if (c->st[k] == id) {
            lin_t p = lin_of(c, c->st[k + 1], depth + 1);
            if (!n++) r = p; else { r.g = gcd64(r.g, p.g); if (p.lo < r.lo) r.lo = p.lo; if (p.hi > r.hi) r.hi = p.hi; }
        }
    }
    else if (op == OpIAdd && len == 5) { lin_t p = lin_of(c, in[3], depth + 1), q = lin_of(c, in[4], depth + 1); r = (lin_t){gcd64(p.g, q.g), p.lo + q.lo, p.hi + q.hi}; }
    else if (op == OpIMul && len == 5) {
        uint32_t k = C32(in[4]) ? in[4] : C32(in[3]) ? in[3] : 0, x = k == in[4] ? in[3] : in[4];
        if (k) { lin_t p = lin_of(c, x, depth + 1); int64_t m = CV(k); r = (lin_t){p.g * m, p.lo * m, p.hi * m}; }
    }
    else if (op == OpIdxShl && len == 5) { if (C32(in[4]) && CV(in[4]) < 31) { lin_t p = lin_of(c, in[3], depth + 1); int64_t m = (int64_t)1 << CV(in[4]); r = (lin_t){p.g * m, p.lo * m, p.hi * m}; } }
    else if (op == OpIdxAnd && len == 5) { uint32_t k = C32(in[4]) ? in[4] : C32(in[3]) ? in[3] : 0; if (k) r = (lin_t){0, 0, CV(k)}; }
    else if (op == OpShiftRightLogical && len == 5) {
        if (C32(in[4]) && CV(in[4]) < 32) { lin_t p = lin_of(c, in[3], depth + 1); int s = (int)CV(in[4]); if (!(p.g & (((int64_t)1 << s) - 1))) r = (lin_t){p.g >> s, p.lo >> s, p.hi >> s}; }
    }
    if (r.g > lim || r.lo > lim || r.hi > lim) r = any;
    c->memo[id] = r; c->state[id] = 2;
    return r;
}
// The plane (rem >> 5) of every address of f in a scatter of chunk ch that starts at word base, -1 when they do not share one
static int lin_plane(lin_t f, int64_t ch, int64_t base) {
    if (f.g % ch || f.hi - f.lo >= 32) return -1;
    int64_t a = ((f.lo - base) % ch + ch) % ch, b = a + (f.hi - f.lo);
    return b < ch && a >> 5 == b >> 5 ? (int)(a >> 5) : -1;
}
// An instruction of mesh_plane_store: 1 with the words that are ids as bits of *ids and the word of its result id (0 none) in *res, 0 for one the copy does not know
static int plane_fn_op(uint32_t op, uint32_t len, uint32_t *ids, uint32_t *res) {
    *ids = 0; *res = 0;
    switch (op) {
    case OpFunction: if (len != 5) return 0; *ids = 1u << 1 | 1u << 2 | 1u << 4; *res = 2; return 1;
    case OpLabel: if (len != 2) return 0; *ids = 1u << 1; *res = 1; return 1;
    case OpSelectionMerge: if (len != 3) return 0; *ids = 1u << 1; return 1;
    case OpBranch: if (len != 2) return 0; *ids = 1u << 1; return 1;
    case OpBranchConditional: if (len < 4) return 0; *ids = 7u << 1; return 1;
    case OpStore: if (len < 3) return 0; *ids = 3u << 1; return 1;
    case OpReturn: case OpFunctionEnd: return len == 1;
    case OpFunctionParameter: case OpBitcast: case OpCopyObject: case OpAccessChain: case OpInBoundsAccessChain: case OpIAdd: case OpIdxISub: case OpIMul: case OpUDiv: case OpUMod:
    case OpLogicalNot: case OpIEqual: case OpULessThan: case OpShiftRightLogical: case OpIdxShl: case OpIdxAnd: case OpIdxOr: case OpSelect:
        if (len < 3 || len > 31) return 0;
        *ids = ((1u << len) - 1) & ~1u; *res = 2; return 1;
    }
    return 0;
}
enum { PLANE_SC = 4 };
typedef struct { int p[PLANE_SC]; uint32_t id; } plane_var;
uint32_t *vka_spv_mesh_store_planes(const uint32_t *w, size_t nw, size_t *out_nw, int *changed, char *err, size_t errn) {
#define FAIL(...) do { snprintf(err, errn, __VA_ARGS__); goto fail; } while (0)
#define EMIT(...) do { uint32_t x_[] = {__VA_ARGS__}; if (!put_n(&o, x_, sizeof x_ / 4)) FAIL("out of memory"); } while (0)
    uint32_t *def = NULL, *rm = NULL, *target = NULL; uint8_t *fl = NULL, *state = NULL; lin_t *memo = NULL; plane_var *var = NULL; size_t nvar = 0, cvar = 0;
    vec_t o = {0}, st = {0}, calls = {0}, defs = {0}; *changed = 0;
    if (nw < 5 || w[0] != 0x07230203) { snprintf(err, errn, "not SPIR-V"); return NULL; }
    uint32_t bound = w[3], nb = bound;
    def = calloc(bound, 4); rm = calloc(bound, 4); fl = calloc(bound, 1); state = calloc(bound, 1); memo = malloc((size_t)bound * sizeof *memo);
    if (!def || !rm || !fl || !state || !memo) FAIL("out of memory");
    // Pass 1: the function, the constants, what each id the address arithmetic can use is made by, the stores, the calls
    uint32_t fn = 0, t_bool = 0, cur = 0; size_t first_fn = nw, fs = 0, fe = 0;
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > nw) FAIL("bad instruction at word %zu", i);
        if (op == OpName && len >= 7 && !fn && !memcmp(w + i + 2, "mesh_plane_store", 17)) fn = w[i + 1];
        else if (op == OpTypeInt && len >= 4 && w[i + 2] == 32 && w[i + 1] < bound) fl[w[i + 1]] |= F_T32;
        else if (op == OpTypeBool && len >= 2) t_bool = w[i + 1];
        else if (op == OpConstant && len == 4 && w[i + 1] < bound && w[i + 2] < bound && (fl[w[i + 1]] & F_T32)) { fl[w[i + 2]] |= F_C32; def[w[i + 2]] = (uint32_t)i; }
        else if (op == OpFunction && len >= 5) { if (first_fn == nw) first_fn = i; cur = w[i + 2]; if (fn && cur == fn) fs = i; }
        else if (op == OpFunctionEnd) { if (fn && cur == fn) fe = i + len; cur = 0; }
        else if (i >= first_fn) {
            uint32_t scan = 1;   // the first word that makes a variable it names "used otherwise"
            if (op == OpVariable && len >= 4 && w[i + 2] < bound) { fl[w[i + 2]] |= len == 4 && w[i + 3] == SCFunction ? F_VAR : F_VAR | F_DIRTY; def[w[i + 2]] = (uint32_t)i; scan = len; }
            else if (op == OpLoad && len >= 4 && w[i + 2] < bound) { def[w[i + 2]] = (uint32_t)i; scan = len; }
            else if (op == OpStore && len >= 3) { if (!put(&st, w[i + 1]) || !put(&st, w[i + 2])) FAIL("out of memory"); scan = 2; }
            else if ((op == OpBitcast || op == OpCopyObject || op == OpIAdd || op == OpIMul || op == OpIdxShl || op == OpIdxAnd || op == OpShiftRightLogical) && len >= 4 && w[i + 2] < bound)
                def[w[i + 2]] = (uint32_t)i;
            else if (op == OpFunctionCall && len >= 5 && fn && w[i + 3] == fn && cur != fn) { if (!put(&calls, (uint32_t)i)) FAIL("out of memory"); }
            for (uint32_t k = scan; k < len; k++) if (w[i + k] < bound && (fl[w[i + k]] & F_VAR)) fl[w[i + k]] |= F_DIRTY;   // a literal that happens to be the id only costs the variable
        }
        i += len;
    }
    if (!fn || !fs || !fe || !calls.n) goto none;
    if (!t_bool) FAIL("mesh_plane_store without a bool type");
    // The function: its scatters (the chunk, the first word, the id of the plane) and the ids it makes
    struct { int64_t ch, base; uint32_t rem, plane; } sc[PLANE_SC]; int nsc = 0, planes = 0;
    struct { uint32_t id; int64_t base; } sub[PLANE_SC]; int nsub = 0;
    uint32_t param0 = 0;
#define K32(x) ((x) < bound && (fl[x] & F_C32))
    for (size_t i = fs; i < fe;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16, ids, res; const uint32_t *in = w + i;
        if (!plane_fn_op(op, len, &ids, &res)) FAIL("mesh_plane_store has an instruction the copy does not know (opcode %u)", op);
        if (res) { if (in[res] >= bound) FAIL("an id past the bound"); if (!put(&defs, in[res])) FAIL("out of memory"); }
        if (op == OpFunctionParameter && !param0) param0 = in[2];
        else if (op == OpIdxISub && len == 5 && in[3] == param0 && K32(in[4]) && nsub < PLANE_SC) { sub[nsub].id = in[2]; sub[nsub++].base = w[def[in[4]] + 3]; }
        else if (op == OpUMod && len == 5 && K32(in[4]) && w[def[in[4]] + 3] >= 32 && w[def[in[4]] + 3] < 0x80000000u && nsc < PLANE_SC) {
            int64_t base = -1;
            if (in[3] == param0) base = 0; else for (int k = 0; k < nsub; k++) if (sub[k].id == in[3]) base = sub[k].base;
            if (base >= 0) { sc[nsc].ch = w[def[in[4]] + 3]; sc[nsc].base = base; sc[nsc].rem = in[2]; sc[nsc++].plane = 0; }
        }
        else if (op == OpShiftRightLogical && len == 5 && K32(in[4]) && w[def[in[4]] + 3] == 5)
            for (int k = 0; k < nsc; k++) if (sc[k].rem == in[3] && !sc[k].plane) { sc[k].plane = in[2]; planes++; }
        i += len;
    }
    if (!param0 || !planes) goto none;
    // The calls: the planes each can reach, and the copy for them
    lin_ctx c = {w, bound, def, fl, st.w, st.n, memo, state};
    if (!(target = calloc(calls.n, 4))) FAIL("out of memory");
    for (size_t k = 0; k < calls.n; k++) {
        lin_t f = lin_of(&c, w[calls.w[k] + 4], 0); plane_var v; int known = 0;
        for (int s = 0; s < PLANE_SC; s++) { v.p[s] = s < nsc && sc[s].plane ? lin_plane(f, sc[s].ch, sc[s].base) : -1; known |= v.p[s] >= 0; }
        if (!known) continue;
        size_t j = 0;
        for (; j < nvar; j++) if (!memcmp(var[j].p, v.p, sizeof v.p)) break;
        if (j == nvar) {
            if (nvar == cvar) { size_t n = cvar ? 2 * cvar : 32; plane_var *p = realloc(var, n * sizeof *p); if (!p) FAIL("out of memory"); var = p; cvar = n; }
            v.id = nb++; var[nvar++] = v;
        }
        target[k] = var[j].id; (*changed)++;
    }
    if (!nvar) goto none;
    uint32_t c_true = nb++, c_false = nb++;
    // Pass 2: the module with the calls redirected and the copies after the function
    if (!put_n(&o, w, 5)) FAIL("out of memory");
    size_t ci = 0;
    for (size_t i = 5; i < nw;) {
        uint32_t len = w[i] >> 16;
        if (i == first_fn) EMIT(3u << 16 | OpConstantTrue, t_bool, c_true, 3u << 16 | OpConstantFalse, t_bool, c_false);
        if (!put_n(&o, w + i, len)) FAIL("out of memory");
        if (ci < calls.n && i == calls.w[ci]) { if (target[ci]) o.w[o.n - len + 3] = target[ci]; ci++; }
        i += len;
        if (i != fe) continue;
        for (size_t v = 0; v < nvar; v++) {
            for (size_t k = 0; k < defs.n; k++) rm[defs.w[k]] = defs.w[k] == fn ? var[v].id : nb++;
            for (size_t j = fs; j < fe;) {
                uint32_t op = w[j] & 0xffff, n = w[j] >> 16, ids, res; const uint32_t *in = w + j;
                plane_fn_op(op, n, &ids, &res);
                int fold = -1;   // a comparison of a known plane with a constant: its result
                if (op == OpIEqual && n == 5)
                    for (int s = 0; s < nsc; s++) if (sc[s].plane && var[v].p[s] >= 0) {
                        uint32_t k = in[3] == sc[s].plane ? in[4] : in[4] == sc[s].plane ? in[3] : 0;
                        if (k && K32(k)) fold = w[def[k] + 3] == (uint32_t)var[v].p[s];
                    }
                if (fold >= 0) EMIT(4u << 16 | OpCopyObject, in[1], rm[in[2]], fold ? c_true : c_false);
                else {
                    if (!put_n(&o, in, n)) FAIL("out of memory");
                    for (uint32_t k = 1; k < n && k < 32; k++) if ((ids >> k & 1) && in[k] < bound && rm[in[k]]) o.w[o.n - n + k] = rm[in[k]];
                }
                j += n;
            }
        }
    }
    o.w[3] = nb;
    free(def); free(rm); free(fl); free(state); free(memo); free(var); free(target); free(st.w); free(calls.w); free(defs.w);
    *out_nw = o.n; return o.w;
none:
    *changed = 0;
fail:
    free(def); free(rm); free(fl); free(state); free(memo); free(var); free(target); free(st.w); free(calls.w); free(defs.w); free(o.w);
    return NULL;
#undef K32
#undef C32
#undef CV
#undef EMIT
#undef FAIL
}

// vka_spv_spec_freeze: a module's specialization constants made plain constants of the values a pipeline would give them (a driver workaround, Qualcomm Adreno 840,
// driver 0x8034a013). Measured on the phone in a standalone mesh pipeline: a mesh module that declares a specialization constant, used or not, specialized by the
// pipeline or not (a bool or a uint, with or without a SpecId), has SubgroupLocalInvocationId read 0 and SubgroupLtMask read 0 in every invocation, while
// OpGroupNonUniformBallot and OpGroupNonUniformShuffle stay right. The same module without the constant reads both correctly (lane = LocalInvocationIndex, lt = the bits
// below it): shown with glslc's modules (one unused "layout(constant_id = 0) const bool" added breaks them) and with Ocarina of Time's mesh 25c4a740753e7d7a, whose only
// one is the translator's dm_gl_depth (DMSH_VK_SPEC_GL_DEPTH; every one of the game's 495 mesh modules has it, its task modules have none). With the id 0 every
// invocation of a translated mesh workgroup took itself for lane 0 and wrote lane 0's outputs: garbage triangles. Why the constant does this in the driver is open.
// The rewrite: each OpSpecConstantTrue / OpSpecConstantFalse / OpSpecConstant (32-bit) whose SpecId is in ids takes vals of the same index (a bool: non-zero is true),
// the others keep their default; all become OpConstantTrue / OpConstantFalse / OpConstant, OpSpecConstantComposite becomes OpConstantComposite, the SpecId decorations
// go. An OpSpecConstantOp is not folded here: the module is refused. *changed counts the constants frozen; NULL with err empty when the module has none
enum { OpSpecConstantTrue = 48, OpSpecConstantFalse = 49, OpSpecConstant = 50, OpSpecConstantComposite = 51, OpSpecConstantOp = 52, DecSpecId = 1 };
uint32_t *vka_spv_spec_freeze(const uint32_t *w, size_t nw, const uint32_t *ids, const uint32_t *vals, uint32_t n, size_t *out_nw, int *changed, char *err, size_t errn) {
#define FAIL(...) do { snprintf(err, errn, __VA_ARGS__); goto fail; } while (0)
    int32_t *sid = NULL; vec_t o = {0}; *changed = 0;   // sid[id]: the id's SpecId, -1 without one
    if (nw < 5 || w[0] != 0x07230203) { snprintf(err, errn, "not SPIR-V"); return NULL; }
    uint32_t bound = w[3]; int any = 0;
    if (!(sid = malloc((size_t)bound * 4))) { snprintf(err, errn, "out of memory"); return NULL; }
    for (uint32_t i = 0; i < bound; i++) sid[i] = -1;
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > nw) FAIL("bad instruction at word %zu", i);
        if (op == OpDecorate && len >= 4 && w[i + 2] == DecSpecId && w[i + 1] < bound) sid[w[i + 1]] = (int32_t)w[i + 3];
        else if (op >= OpSpecConstantTrue && op <= OpSpecConstantComposite) any = 1;
        else if (op == OpSpecConstantOp) FAIL("an OpSpecConstantOp (not folded)");
        else if (op == OpFunction) break;
        i += len;
    }
    if (!any) { free(sid); return NULL; }   // nothing to do
    if (!put_n(&o, w, 5)) FAIL("out of memory");
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; const uint32_t *in = w + i;
        if (op == OpDecorate && len >= 4 && w[i + 2] == DecSpecId) { i += len; continue; }   // dropped: a plain constant takes none
        if (op >= OpSpecConstantTrue && op <= OpSpecConstantComposite) {
            if (len < 3 || w[i + 2] >= bound) FAIL("bad specialization constant at word %zu", i);
            int given = 0; uint32_t v = 0;
            for (uint32_t k = 0; k < n; k++) if (sid[w[i + 2]] == (int32_t)ids[k]) { given = 1; v = vals[k]; }
            if (op == OpSpecConstantComposite) { if (!put(&o, len << 16 | OpConstantComposite) || !put_n(&o, in + 1, len - 1)) FAIL("out of memory"); }
            else if (op == OpSpecConstant) {
                if (given && len != 4) FAIL("a specialization constant wider than 32 bits given a value");
                if (!put(&o, len << 16 | OpConstant) || !put_n(&o, in + 1, len - 1)) FAIL("out of memory");
                if (given) o.w[o.n - 1] = v;
            } else {
                int t = given ? v != 0 : op == OpSpecConstantTrue;
                if (!put(&o, len << 16 | (uint32_t)(t ? OpConstantTrue : OpConstantFalse)) || !put_n(&o, in + 1, len - 1)) FAIL("out of memory");
            }
            ++*changed; i += len; continue;
        }
        if (!put_n(&o, in, len)) FAIL("out of memory");
        i += len;
    }
    free(sid);
    *out_nw = o.n; return o.w;
fail:
    free(sid); free(o.w);
    return NULL;
#undef FAIL
}
