// SPIR-V rewrites made by the replay at pipeline time (vka_spvpatch.c). Plain C, no Vulkan
#pragma once
#include <stddef.h>
#include <stdint.h>
// The vertex module w (nw words) with the vec4 float Input variables at the locations of umask retyped as uvec4 and those of smask as ivec4, every load of them bit-cast back
// to float: the shader then sees the integer register bits Maxwell's fetch gives an integer format. A malloc'd copy (*out_nw words), or NULL with the reason in err
uint32_t *vka_spv_vin_retype(const uint32_t *w, size_t nw, uint32_t umask, uint32_t smask, size_t *out_nw, char *err, size_t errn);
// Subgroup operations and built-ins rewritten as subgroups of one invocation, for a stage the device runs none in (vka_spvpatch.c). A malloc'd copy (*out_nw words,
// *changed = 1), NULL with err empty when the module has none, NULL with the reason in err when it cannot be rewritten
uint32_t *vka_spv_subgroup_single(const uint32_t *w, size_t nw, size_t *out_nw, int *changed, char *err, size_t errn);
// A task or mesh module's SubgroupLocalInvocationId and subgroup masks taken from LocalInvocationIndex, for a driver that hands such modules zeros (vka_spvpatch.c):
// only for a workgroup of at most 32 invocations that is one subgroup in index order. A malloc'd copy (*out_nw words, *changed = 1), NULL with err empty when the
// module reads none, NULL with the reason in err when it cannot be rewritten
uint32_t *vka_spv_subgroup_from_index(const uint32_t *w, size_t nw, size_t *out_nw, int *changed, char *err, size_t errn);
// A mesh module's calls of the translator's mesh_plane_store sent to copies of it made for the one output plane each call can reach, so that a shader compiler need
// not carry the whole chain of plane tests at every call (vka_spvpatch.c). A malloc'd copy (*out_nw words, *changed = the calls redirected), NULL with err empty when
// the module has no such function or no call whose plane is known, NULL with the reason in err when it cannot be rewritten
uint32_t *vka_spv_mesh_store_planes(const uint32_t *w, size_t nw, size_t *out_nw, int *changed, char *err, size_t errn);
// A module's specialization constants made plain constants: those with a SpecId in ids (n of them) take vals of the same index, the others their default, for a
// driver that mishandles a mesh module declaring any (vka_spvpatch.c). A malloc'd copy (*out_nw words, *changed = the constants frozen), NULL with err empty when
// the module has none, NULL with the reason in err when it cannot be rewritten
uint32_t *vka_spv_spec_freeze(const uint32_t *w, size_t nw, const uint32_t *ids, const uint32_t *vals, uint32_t n, size_t *out_nw, int *changed, char *err, size_t errn);
// A geometry shader's memory writes, counted (the return; -1 with err on failure) and, with out and a count above 0, the module with each of them made only by the run of
// the shader that first claims its primitive in the storage buffer at set 0 binding (zeroed for each draw, a word per primitive), malloc'd (*out_nw words): vka_spvpatch.c
int vka_spv_gs_claim(const uint32_t *w, size_t nw, uint32_t binding, uint32_t **out, size_t *out_nw, char *err, size_t errn);
// A pixel shader's float colour outputs at the locations of umask / smask as unsigned / signed integers of their bits (each store bit-cast), for integer attachments: malloc'd
// (*out_nw words), NULL with the reason in err (vka_spvpatch.c)
uint32_t *vka_spv_fout_retype(const uint32_t *w, size_t nw, uint32_t umask, uint32_t smask, size_t *out_nw, char *err, size_t errn);
