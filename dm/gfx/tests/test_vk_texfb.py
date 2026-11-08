"""Texture-use feedback in the Vulkan library's modules (vk_android_spv.py tex_feedback; dmsh_abi_vk.h DMSH_VK_POOL_FB_BINDING, DMSH_VK_SPEC_TEXFB, DMSH_VK_F_TEXFB): small pixel
and vertex shaders that index the texture pool as the translator's bindless sites do (set 1 binding 0, an access chain by the handle's low 20 bits), in SPIR-V 1.3 (the
translator's) and 1.6. Checked: spirv-val of the result; the bitmap binding, its type and the specialization constant; one site per distinct ID value in a block (the same
handle sampled twice gets one), a site per block otherwise, in a branch and in a loop body; the structure of each site (the gate, the waterfall loop, elect, test before the
atomicOr); helper invocations skipped in pixel shaders only; with the constant false spirv-opt folds every atomic away, with it true they stay; a module without a pool
access is left byte for byte; a pool access in a loop header block is refused (ValueError) without a broken module. Optional: DMSH_TEXFB_SAMPLE=<dir of .spv> runs the pass and
spirv-val over every module there that reads the pool (the full-library sample of fb/TOOLS.md)."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[0] / 'tools'
sys.path.insert(0, str(TOOLS))
import vk_android_spv as spv  # noqa: E402

GLSLANG, SPIRV_VAL, SPIRV_OPT, SPIRV_AS = (shutil.which(t) for t in ('glslangValidator', 'spirv-val', 'spirv-opt', 'spirv-as'))
POOL_SET, POOL_TEX, FB_BINDING, SPEC = 1, 0, 3, 2

PS = '''#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout(set = 1, binding = 0) uniform texture2D pool[1048576];
layout(set = 1, binding = 1) uniform sampler samp[4096];
layout(location = 0) flat in uvec2 h;
layout(location = 1) in vec2 uv;
layout(location = 0) out vec4 o;
void main()
{
    uint a = min(h.x & 0xFFFFFu, 0xFFFFFu);
    vec4 c = texture(sampler2D(pool[nonuniformEXT(a)], samp[0]), uv);
    c += texture(sampler2D(pool[nonuniformEXT(a)], samp[0]), uv + 0.25);      // the same ID in the same block: no second site
    if (uv.x > 0.5) c += texture(sampler2D(pool[nonuniformEXT(h.y & 0xFFFFFu)], samp[1]), uv);
    for (uint i = 0u; i < (h.y >> 28); i++) c += texture(sampler2D(pool[nonuniformEXT(a + i)], samp[0]), uv * float(i));
    o = c;
}
'''
VS = '''#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout(set = 1, binding = 0) uniform texture2D pool[1048576];
layout(set = 1, binding = 1) uniform sampler samp[4096];
layout(location = 0) in vec4 pos;
layout(location = 1) in uint hnd;
void main()
{
    float d = textureLod(sampler2D(pool[nonuniformEXT(hnd & 0xFFFFFu)], samp[0]), pos.xy, 0.0).x;
    gl_Position = pos + vec4(0.0, 0.0, d, 0.0);
}
'''
CS = '''#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std430) readonly buffer Ids { uint id[]; };
layout(set = 0, binding = 1, std430) writeonly buffer Out { uint o[]; };
layout(set = 1, binding = 0) uniform texture2D pool[64];
layout(set = 1, binding = 1) uniform sampler samp[1];
void main()
{
    uint g = gl_GlobalInvocationID.x, i = id[g];
    vec4 c = vec4(0.0);
    if (i != 0xFFFFFFFFu) c = texelFetch(nonuniformEXT(sampler2D(pool[nonuniformEXT(i)], samp[0])), ivec2(0), 0);   // (the combined value non-uniform too)
    o[g] = packUnorm4x8(c);
}
'''
PLAIN = '''#version 450
layout(set = 0, binding = 0) uniform sampler2D t;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 o;
void main() { o = texture(t, uv); }
'''
# a pool access in a loop header block (the translator puts none there; the pass must refuse it rather than split a header)
HEADER = '''OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %o %h
OpExecutionMode %main OriginUpperLeft
OpDecorate %pool DescriptorSet 1
OpDecorate %pool Binding 0
OpDecorate %o Location 0
OpDecorate %h Location 0
OpDecorate %h Flat
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4 = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%bool = OpTypeBool
%img = OpTypeImage %float 2D 0 0 0 1 Unknown
%n = OpConstant %uint 1048576
%arr = OpTypeArray %img %n
%parr = OpTypePointer UniformConstant %arr
%pimg = OpTypePointer UniformConstant %img
%pool = OpVariable %parr UniformConstant
%pout = OpTypePointer Output %v4
%o = OpVariable %pout Output
%pin = OpTypePointer Input %uint
%h = OpVariable %pin Input
%u0 = OpConstant %uint 0
%u1 = OpConstant %uint 1
%u4 = OpConstant %uint 4
%f0 = OpConstant %float 0
%zero = OpConstantComposite %v4 %f0 %f0 %f0 %f0
%main = OpFunction %void None %fn
%entry = OpLabel
%hv = OpLoad %uint %h
OpBranch %head
%head = OpLabel
%i = OpPhi %uint %u0 %entry %inext %cont
%ac = OpAccessChain %pimg %pool %hv
%im = OpLoad %img %ac
OpLoopMerge %merge %cont None
OpBranch %body
%body = OpLabel
%inext = OpIAdd %uint %i %u1
OpBranch %cont
%cont = OpLabel
%more = OpULessThan %bool %inext %u4
OpBranchConditional %more %head %merge
%merge = OpLabel
OpStore %o %zero
OpReturn
OpFunctionEnd
'''


def run(*a, **k): return subprocess.run(list(a), capture_output=True, text=True, **k)


@unittest.skipUnless(GLSLANG and SPIRV_VAL and SPIRV_OPT and SPIRV_AS, 'needs glslangValidator, spirv-val, spirv-opt, spirv-as')
class TexFeedback(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='texfb_')); self.addCleanup(shutil.rmtree, self.tmp, True)

    def compile(self, src, stage, env='vulkan1.1'):
        s = self.tmp / f'x.{stage}'; s.write_text(src); out = self.tmp / f'x_{stage}_{env}.spv'
        r = run(GLSLANG, '-V', '--target-env', env, '-S', stage, str(s), '-o', str(out))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return out.read_bytes()

    def val(self, data, name='m.spv'):
        p = self.tmp / name; p.write_bytes(data)
        r = run(SPIRV_VAL, '--target-env', 'vulkan1.3', str(p))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return p

    def ops(self, m, op): return [x for x in m.ins if x[0] & 0xffff == op]

    def check_decls(self, m):
        fb = [v for v, d in m.decos.items() if d.get(spv.DEC_SET) == [POOL_SET] and d.get(spv.DEC_BINDING) == [FB_BINDING]]
        self.assertEqual(len(fb), 1)
        ptr, sc = m.vars[fb[0]]
        self.assertEqual(sc, spv.SC_STORAGE_BUFFER)
        st = m.pointee(ptr)[1]
        self.assertIn(spv.DEC_BLOCK, m.decos[st])
        self.assertEqual(m.op(st), spv.OP_TYPE_STRUCT); self.assertEqual(len(m.types[st]), 3)
        rta = m.types[st][2]
        self.assertEqual(m.op(rta), spv.OP_TYPE_RUNTIME_ARRAY); self.assertEqual(m.int_type(m.types[rta][2]), (32, 0))
        self.assertEqual(m.decos[rta].get(spv.DEC_ARRAY_STRIDE), [4])
        spec = [v for v, d in m.decos.items() if d.get(spv.DEC_SPEC_ID) == [SPEC]]
        self.assertEqual(len(spec), 1)
        self.assertTrue(any(x[0] & 0xffff == spv.OP_SPEC_CONSTANT_FALSE and x[2] == spec[0] for x in m.ins))
        self.assertTrue({spv.CAP_GROUP_NON_UNIFORM, spv.CAP_GROUP_NON_UNIFORM_BALLOT} <= set(m.caps))
        return fb[0], spec[0]

    def fold(self, path, value):
        """spirv-opt with the constant set to value and the usual optimisations → number of OpAtomicOr left"""
        out = self.tmp / f'opt_{value}.spv'
        r = run(SPIRV_OPT, '--target-env=vulkan1.3', f'--set-spec-const-default-value={SPEC}:{value}', '--freeze-spec-const', '-O', str(path), '-o', str(out))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return sum(1 for x in spv.Module(out.read_bytes()).ins if x[0] & 0xffff == spv.OP_ATOMIC_OR)

    def test_pixel(self):
        for env in ('vulkan1.1', 'vulkan1.3'):   # SPIR-V 1.3 (the translator's) and 1.6
            with self.subTest(env=env):
                m = spv.Module(self.compile(PS, 'frag', env))
                marked, sites = spv.tex_feedback(m, POOL_SET, POOL_TEX, FB_BINDING, SPEC)
                self.assertEqual((marked, sites), (3, 4))   # the second sample of ID a in the first block shares the first's site
                p = self.val(m.data(), f'ps_{env}.spv')
                fb, spec = self.check_decls(m)
                self.assertEqual(len(self.ops(m, spv.OP_ATOMIC_OR)), 3)
                self.assertEqual(len(self.ops(m, spv.OP_GNU_BROADCAST_FIRST)), 3)
                self.assertEqual(len(self.ops(m, spv.OP_GNU_ELECT)), 3)
                self.assertEqual(len(self.ops(m, spv.OP_LOOP_MERGE)), 3 + 1)   # the shader's own loop
                helper, _ = m.builtin_var(spv.SC_INPUT, spv.BUILTIN_HELPER_INVOCATION)
                self.assertIsNotNone(helper)
                if env == 'vulkan1.3': self.assertIn(fb, m.interface()); self.assertIn(spv.DEC_VOLATILE, m.decos[helper])
                # each site: (spec && !helper) gates it; the atomic only behind elect && the bit clear; its word index is the clamped ID >> 5
                ands = self.ops(m, spv.OP_LOGICAL_AND)
                self.assertEqual(sum(1 for x in ands if x[3] == spec), 3)
                umin = [x for x in self.ops(m, spv.OP_EXT_INST) if x[4] == spv.GLSL_UMIN and m.consts.get(x[6], (0, [0]))[1] == [spv.TEXFB_MAX_ID]]
                self.assertGreaterEqual(len(umin), 3)
                for a in self.ops(m, spv.OP_ATOMIC_OR):
                    ac = next(x for x in m.ins if x[0] & 0xffff == spv.OP_ACCESS_CHAIN and x[2] == a[3])
                    self.assertEqual(ac[3], fb)
                    ld = [x for x in m.ins if x[0] & 0xffff == spv.OP_LOAD and x[3] == a[3]]
                    self.assertEqual(len(ld), 1)   # the test before the set
                self.assertEqual(self.fold(p, 'false'), 0)
                self.assertEqual(self.fold(p, 'true'), 3)

    def test_vertex(self):
        m = spv.Module(self.compile(VS, 'vert'))
        self.assertEqual(spv.tex_feedback(m, POOL_SET, POOL_TEX, FB_BINDING, SPEC), (1, 1))
        self.val(m.data(), 'vs.spv')
        self.check_decls(m)
        self.assertIsNone(m.builtin_var(spv.SC_INPUT, spv.BUILTIN_HELPER_INVOCATION)[0])   # no helper invocations outside fragment shaders
        self.assertEqual(sum(1 for x in self.ops(m, spv.OP_BRANCH_CONDITIONAL) if m.decos.get(x[1], {}).get(spv.DEC_SPEC_ID) == [SPEC]), 1)

    def test_no_pool(self):
        data = self.compile(PLAIN, 'frag')
        m = spv.Module(data)
        self.assertEqual(spv.tex_feedback(m, POOL_SET, POOL_TEX, FB_BINDING, SPEC), (0, 0))
        self.assertEqual(m.data(), data)

    def test_loop_header_refused(self):
        src = self.tmp / 'h.spvasm'; src.write_text(HEADER); out = self.tmp / 'h.spv'
        r = run(SPIRV_AS, '--target-env', 'vulkan1.1', str(src), '-o', str(out)); self.assertEqual(r.returncode, 0, r.stderr)
        self.val(out.read_bytes(), 'h_in.spv')
        with self.assertRaises(ValueError): spv.tex_feedback(spv.Module(out.read_bytes()), POOL_SET, POOL_TEX, FB_BINDING, SPEC)

    @unittest.skipUnless(shutil.which('cc'), 'needs a C compiler')
    def test_gpu(self):
        """On this machine's Vulkan device (vk_texfb_run.c): the bitmap holds exactly the preset bits and the IDs fetched with the constant true, only the preset bits with
        it false; every fetch still returns its entry's colour (uniform, 32 distinct, 3 distinct, half the lanes, one lane, none, two halves, eight runs a wave)"""
        m = spv.Module(self.compile(CS, 'comp'))
        self.assertEqual(spv.tex_feedback(m, POOL_SET, POOL_TEX, FB_BINDING, SPEC), (1, 1))
        p = self.val(m.data(), 'cs.spv')
        runner = self.tmp / 'vk_texfb_run'
        r = run('cc', '-O1', '-o', str(runner), str(HERE / 'vk_texfb_run.c'), '-ldl'); self.assertEqual(r.returncode, 0, r.stderr)
        for on in (1, 0):
            r = run(str(runner), str(p), str(on))
            if r.returncode == 77: self.skipTest(r.stdout.strip())
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            got = dict(ln.split(' ', 1) for ln in r.stdout.splitlines())
            want = got['expected'].split(', off ')[1 - on].split()[-2:]
            self.assertEqual(got['bitmap'], f'{want[0]} {want[1]}, words past them clear', r.stdout)
            self.assertEqual(got['fetches'], 'right (0 wrong)', r.stdout)

    @unittest.skipUnless(os.environ.get('DMSH_TEXFB_SAMPLE'), 'DMSH_TEXFB_SAMPLE=<dir of .spv> not set')
    def test_sample(self):
        """Every module of the folder that reads the pool: the pass, then spirv-val; prints the counts"""
        bad, n, marked, sites = [], 0, 0, 0
        for p in sorted(Path(os.environ['DMSH_TEXFB_SAMPLE']).glob('*.spv')):
            m = spv.Module(p.read_bytes())
            try: k = spv.tex_feedback(m, POOL_SET, POOL_TEX, FB_BINDING, SPEC)
            except ValueError as e: bad.append(f'{p.name}: {e}'); continue
            if not k[1]: continue
            n += 1; marked += k[0]; sites += k[1]
            q = self.tmp / 'sample.spv'; q.write_bytes(m.data())
            r = run(SPIRV_VAL, '--target-env', 'vulkan1.3', str(q))
            if r.returncode: bad.append(f'{p.name}: {(r.stdout + r.stderr).strip().splitlines()[0]}')
        print(f'\ntexfb sample: {n} pool-reading modules, {marked} sites given code of {sites}, {len(bad)} failures', file=sys.stderr)
        self.assertGreater(n, 0)
        self.assertEqual(bad, [])


if __name__ == '__main__':
    unittest.main()
