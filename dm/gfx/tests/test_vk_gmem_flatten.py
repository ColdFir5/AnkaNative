"""The Vulkan library's raw global memory helpers made one block (vk_android_spv.py gmem_flatten) and the translator's temporaries forwarded within a block (local_forward).
A compute shader whose helpers have the translator's shape (GlobalToStorage.TryGenerateRawStorageOp written as GLSL, compiled with glslang: the unrolled binary search of the
pool table, the pool check, the write log's 8 probes with an early exit each, the access through a physical storage buffer pointer) is rewritten: one block per helper,
the probes compare with ~0 where they are skipped, the access goes to the sink of table entry 0, spirv-val passes; a helper of another shape (a loop) is left alone. Where
a Vulkan device and a C compiler are present (vk_gmem_run.c), the shader runs before and after on 64 cases (addresses in the pool, below, above and between pools; loads,
stores) and the results must be the same: every value loaded, every byte of the pool's memory, the sink untouched by in-pool accesses, and the write log's pages.
The pool searches made loops (gmem_loop) match only after local_forward (each step's select read directly, not through a temporary: the order vk_android_build.py
finish() runs them in); the loop module is valid and, on a device, gives the same results too. The searches behind the invocation's cached entry (gmem_cache, the
graphics stages) are valid in vertex and fragment shaders (helper lanes, demote, non-uniform control flow) and, on a device, give the same results with every
invocation reading 4 other cases first (hits, misses into another pool, the end of a pool, entry 0, the sink), also with the table filled to its last entry.
The searches replaced by the page index (gmem_pidx) are straight-line (no loop, no new branch), valid in vertex and fragment shaders, refused where the helper's pool
check or device address is not the translator's formula, and, on a device, give the same results (the runner builds the index after the table)."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[0] / 'tools'
sys.path.insert(0, str(TOOLS))
import vk_android_spv as spv  # noqa: E402

SINK_WORD = 6   # dmsh_abi_vk.h DMSH_VK_GMEM_SINK_WORD
MAP_LOG2, PROBES, LOG_LOG2 = 12, 8, 10   # GlobalToStorage.GmemMapLog2, GmemLogProbes, GmemLogLog2


def search_glsl():
    """The translator's binary search: the last entry whose start is at or below the address (entry 0 starts at 0)"""
    L = ['    uint index = 0u;']
    step = 1 << (MAP_LOG2 - 1)
    while step:
        L += [f'    {{ uint cand = index + {step}u; uint e = cand * 8u; uint sl = map_.w[e]; uint sh = map_.w[e + 1u];',
              '      bool below = any(bvec2(sh < hi, all(bvec2(sh == hi, sl <= lo)))); index = below ? cand : index; }']
        step >>= 1
    L += ['    uint f = index * 8u; uint el = map_.w[f + 2u]; uint eh = map_.w[f + 3u];',
          '    bool inPool = any(bvec2(hi < eh, all(bvec2(hi == eh, lo < el))));']   # (no short circuit: the translator's operations, no branch)
    return L


def host_glsl():
    return ['        uint hl = lo + map_.w[f + 4u]; uint carry = hl < lo ? 1u : 0u; uint hh = hi + map_.w[f + 5u] + carry;',
            ]


def shader_glsl(loop_helper=False, warm=False):
    L = ['#version 460', '#extension GL_EXT_buffer_reference : require', '#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require',
         'layout(local_size_x = 64) in;',
         'layout(buffer_reference, std430, buffer_reference_align = 4) buffer PU { uint v; };',
         'layout(set = 0, binding = 0, std430) buffer Map { uint w[]; } map_;',
         'layout(set = 0, binding = 1, std430) buffer Log { uint w[]; } log_;',
         'layout(set = 0, binding = 2, std430) buffer Cases { uvec4 c[]; } cases;   // (address low, high, value, kind: 0 load, 1 store)',
         'layout(set = 0, binding = 3, std430) buffer Out { uint o[]; } out_;',
         '', 'uint gload(uint lo, uint hi) {'] + search_glsl()
    L += ['    if (inPool) {'] + host_glsl() + ['        return PU(packUint2x32(uvec2(hl, hh))).v;', '    }', '    return 0u;', '}', '',
          'void gstore(uint lo, uint hi, uint value) {'] + search_glsl()
    L += ['    if (inPool) {',
          '        uint page = (hi << 20) | (lo >> 12); uint hash = (page * 0x9e3779b1u) >> ' + str(32 - LOG_LOG2) + ';']
    if loop_helper:   # another shape: the probes in a loop (gmem_flatten leaves it alone)
        L += ['        for (uint pr = 0u; pr < 8u; pr++) { uint slot = ((hash + pr) & 1023u) + 1u; uint old = atomicCompSwap(log_.w[slot], 0u, page);',
              '            if (any(bvec2(old == 0u, old == page))) break; }']
    else:
        ind = '        '
        for pr in range(PROBES):
            L += [ind + f'uint slot{pr} = ((hash + {pr}u) & {(1 << LOG_LOG2) - 1}u) + 1u; uint old{pr} = atomicCompSwap(log_.w[slot{pr}], 0u, page);',
                  ind + f'if (!any(bvec2(old{pr} == 0u, old{pr} == page))) {{']
            ind += '    '
        L += [ind + 'log_.w[0] = 1u;'] + ['        ' + '    ' * (PROBES - 1 - k) + '}' for k in range(PROBES)]
    L += host_glsl() + ['        PU(packUint2x32(uvec2(hl, hh))).v = value;', '        return;', '    }', '}', '',
          'void main() {', '    uint i = gl_GlobalInvocationID.x; uvec4 c = cases.c[i];']
    if warm:   # other cases' addresses loaded first, the last one the case's own (gmem_cache: misses into other pools, hits, the entry kept across calls)
        for j, k in enumerate(('(i * 5u + 1u) & 63u', '(i * 7u + 3u) & 63u', '(i + 1u) & 63u', 'i')):
            L += [f'    {{ uvec4 w = cases.c[{k}]; out_.o[64u + i * 4u + {j}u] = gload(w.x, w.y); }}']
    L += [
          '    if (c.w == 0u) out_.o[i] = gload(c.x, c.y); else { gstore(c.x, c.y, c.z); out_.o[i] = 0xdeadu; }', '}']
    return '\n'.join(L)


def compile_glsl(src, out):
    p = out.with_suffix('.comp'); p.write_text(src)
    r = subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', '-o', str(out), str(p)], capture_output=True, text=True)
    if r.returncode: raise AssertionError(r.stdout + r.stderr)


def validate(path):
    r = subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', str(path)], capture_output=True, text=True)
    return None if r.returncode == 0 else (r.stdout + r.stderr)


def buffer_var(m, binding):
    return next(v for v, d in m.decos.items() if d.get(spv.DEC_BINDING) == [binding] and v in m.vars)


def function_shapes(m):
    out = {}
    for f in [x[2] for x in m.ins if x[0] & 0xffff == spv.OP_FUNCTION]:
        a, e = m.function_range(f)
        body = m.ins[a:e + 1]
        out[f] = {'labels': sum(1 for x in body if x[0] & 0xffff == spv.OP_LABEL), 'branches': sum(1 for x in body if x[0] & 0xffff == spv.OP_BRANCH_CONDITIONAL),
                  'cas': [x for x in body if x[0] & 0xffff == spv.OP_ATOMIC_COMPARE_EXCHANGE]}
    return out


@unittest.skipUnless(shutil.which('glslangValidator') and shutil.which('spirv-val'), 'glslangValidator and spirv-val needed')
class GmemFlatten(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='gmemflat'))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, loop_helper=False):
        src = self.tmp / ('loop.spv' if loop_helper else 'base.spv')
        compile_glsl(shader_glsl(loop_helper), src)
        m = spv.Module(src.read_bytes())
        return src, m

    def test_one_block_valid(self):
        src, m = self.build()
        before = function_shapes(m)
        n = spv.gmem_flatten(m, buffer_var(m, 0), buffer_var(m, 1), SINK_WORD)
        self.assertEqual(n, 2)   # the load and the store helper
        out = self.tmp / 'flat.spv'; out.write_bytes(m.data())
        self.assertIsNone(validate(out))
        after = function_shapes(m)
        entry = m.entry_function()
        for f, s in after.items():
            if f == entry: continue
            self.assertEqual((s['labels'], s['branches']), (1, 0))
        self.assertTrue(any(s['branches'] >= PROBES for f, s in before.items() if f != entry))
        # every probe compares with a select (the written page where the probe runs, ~0 where it is skipped)
        cas = [x for s in after.values() for x in s['cas']]
        self.assertEqual(len(cas), PROBES)
        selects = {x[2]: x for x in m.ins if x[0] & 0xffff == spv.OP_SELECT}
        ffff = next(i for i, (t, v) in m.consts.items() if v == [0xFFFFFFFF])
        for x in cas: self.assertEqual(selects[x[8]][5], ffff)
        # the sink: table words 6 and 7 read
        idx = {m.consts[x[5]][1][0] for x in m.ins if x[0] & 0xffff == spv.OP_ACCESS_CHAIN and x[3] == buffer_var(m, 0) and len(x) == 6 and x[5] in m.consts}
        self.assertTrue({SINK_WORD, SINK_WORD + 1} <= idx)
        self.flat = out; self.src = src

    def test_loop_helper_left(self):
        src, m = self.build(loop_helper=True)
        n = spv.gmem_flatten(m, buffer_var(m, 0), buffer_var(m, 1), SINK_WORD)
        self.assertEqual(n, 1)   # the load helper only: the store's probes are a loop
        self.assertEqual(sum(1 for x in m.ins if x[0] & 0xffff == spv.OP_LOOP_MERGE), 1)

    def test_forward(self):
        src, m = self.build()
        spv.gmem_flatten(m, buffer_var(m, 0), buffer_var(m, 1), SINK_WORD)
        loads = sum(1 for x in m.ins if x[0] & 0xffff == spv.OP_LOAD)
        n = spv.local_forward(m)
        out = self.tmp / 'fwd.spv'; out.write_bytes(m.data())
        self.assertIsNone(validate(out))
        self.assertGreater(n, 0)
        self.assertEqual(sum(1 for x in m.ins if x[0] & 0xffff == spv.OP_LOAD), loads - n)

    def test_search_loop(self):
        src, m = self.build()
        spv.gmem_flatten(m, buffer_var(m, 0), buffer_var(m, 1), SINK_WORD)
        self.assertEqual(spv.gmem_loop(spv.Module(m.data()), buffer_var(m, 0)), 0)   # before local_forward: the steps go through temporaries
        spv.local_forward(m)
        self.assertEqual(spv.gmem_loop(m, buffer_var(m, 0)), 2)   # the load and the store helper's search
        out = self.tmp / 'loop.spv'; out.write_bytes(m.data())
        self.assertIsNone(validate(out))
        merges = [x for x in m.ins if x[0] & 0xffff == spv.OP_LOOP_MERGE]
        self.assertEqual(len(merges), 2)
        self.assertTrue(all(x[3] & 2 for x in merges))   # DontUnroll
        self.assertEqual(spv.gmem_loop(m, buffer_var(m, 0)), 0)   # nothing left to take

    def test_search_loop_compute_only(self):   # gmem_loop_stage: loops in compute; in a graphics stage gmem_cache, or with it off the module stays as it is and the searches are counted
        src, m = self.build()
        spv.gmem_flatten(m, buffer_var(m, 0), buffer_var(m, 1), SINK_WORD)
        spv.local_forward(m)
        self.assertEqual(spv.GMEM_LOOP_STAGES, {'cs'})
        for stage in ('vs', 'ps', 'gs', 'ds', 'hs', 'ts', 'ms'):
            x = spv.Module(m.data())
            self.assertEqual(spv.gmem_loop_stage(x, buffer_var(m, 0), stage, cache=False, pidx=False), (0, 0, 2), stage)
            self.assertEqual(x.data(), m.data(), stage)
            x = spv.Module(m.data())
            self.assertEqual(spv.gmem_loop_stage(x, buffer_var(m, 0), stage), (0, 2, 0) if spv.GMEM_PIDX or spv.GMEM_CACHE else (0, 0, 2), stage)
        self.assertEqual(spv.gmem_loop_stage(spv.Module(m.data()), None, 'ps'), (0, 0, 0))   # no map table: nothing to count
        x = spv.Module(m.data())
        self.assertEqual(spv.gmem_loop_stage(x, buffer_var(m, 0), 'cs'), (2, 0, 0))
        y = spv.Module(m.data()); spv.gmem_loop(y, buffer_var(m, 0))
        self.assertEqual(x.data(), y.data())   # compute: exactly gmem_loop
        self.assertEqual(sum(1 for i in x.ins if i[0] & 0xffff == spv.OP_LOOP_MERGE), 2)

    def cached(self, warm=False, words=None):   # the module through gmem_flatten, local_forward and gmem_cache → (source, flattened, cached) paths
        src = self.tmp / ('warm.spv' if warm else 'base.spv')
        compile_glsl(shader_glsl(warm=warm), src)
        m = spv.Module(src.read_bytes())
        flat = spv.Module(m.data()); spv.gmem_flatten(flat, buffer_var(m, 0), buffer_var(m, 1), SINK_WORD); spv.local_forward(flat)
        fp = self.tmp / f'flat{int(warm)}.spv'; fp.write_bytes(flat.data())
        c = spv.Module(flat.data()); n = spv.gmem_cache(c, buffer_var(m, 0), words)
        cp = self.tmp / f'cache{int(warm)}_{words}.spv'; cp.write_bytes(c.data())
        return src, fp, cp, c, n

    def test_cache_valid(self):
        for words in (None, 0, 2, 6):
            src, fp, cp, c, n = self.cached(warm=True, words=words)
            self.assertEqual(n, 2, words)   # the load and the store helper's search
            self.assertIsNone(validate(cp), words)
            merges = [x for x in c.ins if x[0] & 0xffff == spv.OP_LOOP_MERGE]
            self.assertEqual(len(merges), 2)
            self.assertTrue(all(x[3] & 2 for x in merges))   # DontUnroll
            priv = [x for x in c.ins if x[0] & 0xffff == spv.OP_VARIABLE and x[3] == spv.SC_PRIVATE]
            self.assertEqual(len(priv), 1 + ((words if words is not None else spv.GMEM_CACHE_WORDS) + 2 if words != 0 else 0))
            self.assertTrue(all(len(x) == 5 for x in priv))   # each with its initializer (entry 0, words 0: an empty range, never a hit)
            self.assertEqual(spv.gmem_cache(c, buffer_var(c, 0), words), 0)   # nothing left to take
        x = spv.Module(fp.read_bytes())
        self.assertEqual(spv.gmem_loop_stage(x, buffer_var(x, 0), 'vs', cache=True, pidx=False)[:2], (0, 2))
        y = spv.Module(fp.read_bytes()); spv.gmem_cache(y, buffer_var(y, 0))
        self.assertEqual(x.data(), y.data())
        x = spv.Module(fp.read_bytes())
        self.assertEqual(spv.gmem_loop_stage(x, buffer_var(x, 0), 'vs', cache=False, pidx=False), (0, 0, 2))
        self.assertEqual(x.data(), fp.read_bytes())   # off: left as it is (counted)

    def test_cache_graphics_stages(self):   # a vertex shader and a fragment shader with helper lanes, demote and the loads in non-uniform control flow
        head = ['#version 460', '#extension GL_EXT_buffer_reference : require', '#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require',
                'layout(buffer_reference, std430, buffer_reference_align = 4) buffer PU { uint v; };',
                'layout(set = 0, binding = 0, std430) readonly buffer Map { uint w[]; } map_;', '', 'uint gload(uint lo, uint hi) {'] + search_glsl() + [
                '    if (inPool) {'] + host_glsl() + ['        return PU(packUint2x32(uvec2(hl, hh))).v;', '    }', '    return 0u;', '}', '']
        vs = head + ['layout(location = 0) in uvec4 a;', 'layout(location = 0) flat out uint o;', 'void main() {',
                     '    uint s = 0u; for (int k = 0; k < 4; k++) s += gload(a.x + uint(k) * 16u, a.y);',
                     '    if (a.z != 0u) s ^= gload(a.z, a.w);', '    o = s; gl_Position = vec4(float(s & 1u)); }']
        fs = ['#extension GL_EXT_demote_to_helper_invocation : require'] + head + ['layout(location = 0) flat in uvec4 a;', 'layout(location = 0) out vec4 col;',
              'void main() {', '    if ((a.w & 1u) != 0u) demote;', '    uint s = gload(a.x, a.y);',
              '    if (gl_FragCoord.x > 3.0) s += gload(a.x + 16u, a.y) + gload(a.z, a.y);',
              '    col = vec4(float(s), dFdx(float(s)), float(helperInvocationEXT()), 1.0); }']
        fs = [fs[1], fs[0]] + fs[2:]
        for stage, src, ext in (('vs', vs, 'vert'), ('ps', fs, 'frag')):
            p = self.tmp / f'g.{ext}'; p.write_text('\n'.join(src))
            out = self.tmp / f'g_{ext}.spv'
            r = subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', '-o', str(out), str(p)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            m = spv.Module(out.read_bytes())
            self.assertEqual(spv.gmem_flatten(m, buffer_var(m, 0), None, SINK_WORD), 1, stage)
            spv.local_forward(m)
            loops, cached, left = spv.gmem_loop_stage(m, buffer_var(m, 0), stage, cache=True)
            self.assertEqual((loops, cached, left), (0, 1, 0), stage)
            cp = self.tmp / f'g_{ext}_cache.spv'; cp.write_bytes(m.data())
            self.assertIsNone(validate(cp), stage)
            r = subprocess.run(['spirv-opt', '-O', '--target-env=vulkan1.3', str(cp), '-o', str(self.tmp / 'o.spv')], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)   # (the optimizer takes it: structured, inlinable)

    def gpu_runner(self):
        runner = self.tmp / 'vk_gmem_run'
        if not runner.exists():
            r = subprocess.run(['cc', '-O1', '-o', str(runner), str(HERE / 'vk_gmem_run.c'), '-ldl'], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        return runner

    def run_gpu(self, path, *args, gran=None):   # gran: the page index window's rounding (vk_gmem_run.c PIDX_GRAN; None: the host's)
        env = dict(os.environ, **({'PIDX_GRAN': str(gran)} if gran else {}))
        r = subprocess.run([str(self.gpu_runner()), str(path), *args], capture_output=True, text=True, timeout=120, env=env)
        if r.returncode == 77: self.skipTest(r.stdout.strip() or 'no Vulkan device')
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        return r.stdout

    @unittest.skipUnless(shutil.which('cc') and os.path.exists('/usr/include/vulkan/vulkan.h'), 'a C compiler and the Vulkan headers needed for the GPU run')
    def test_cache_same_results_on_gpu(self):
        for full in ((), ('full',)):
            src, fp, _, _, _ = self.cached(warm=True)
            base = self.run_gpu(src, *full)
            self.assertIn('sink untouched', base)
            self.assertNotIn('digest cbf29ce484222325', base.split('warm-up')[1].split('\n')[0])   # (the warm-up loads ran: not the empty digest)
            self.assertEqual(self.run_gpu(fp, *full), base, full)
            for words in (0, 2, 6):
                _, _, cp, _, n = self.cached(warm=True, words=words)
                self.assertEqual(n, 2)
                self.assertEqual(self.run_gpu(cp, *full), base, (full, words))

    def indexed(self, warm=False):   # the module through gmem_flatten, local_forward and gmem_pidx → (source, flattened, page-indexed) paths, the indexed module, count
        src = self.tmp / ('warm.spv' if warm else 'base.spv')
        compile_glsl(shader_glsl(warm=warm), src)
        m = spv.Module(src.read_bytes())
        flat = spv.Module(m.data()); spv.gmem_flatten(flat, buffer_var(m, 0), buffer_var(m, 1), SINK_WORD); spv.local_forward(flat)
        fp = self.tmp / f'flat{int(warm)}.spv'; fp.write_bytes(flat.data())
        x = spv.Module(flat.data()); n = spv.gmem_pidx(x, buffer_var(m, 0))
        xp = self.tmp / f'pidx{int(warm)}.spv'; xp.write_bytes(x.data())
        return src, fp, xp, x, n

    def test_pidx_valid(self):
        src, fp, xp, x, n = self.indexed(warm=True)
        self.assertEqual(n, 2)   # the load and the store helper's search
        self.assertIsNone(validate(xp))
        flat = spv.Module(fp.read_bytes())
        self.assertEqual(sum(1 for i in x.ins if i[0] & 0xffff == spv.OP_LOOP_MERGE), 0)
        self.assertEqual([v['branches'] for v in function_shapes(x).values()], [v['branches'] for v in function_shapes(flat).values()])   # straight-line: no branch added
        self.assertEqual(spv.gmem_pidx(x, buffer_var(x, 0)), 0)   # nothing left to take
        y = spv.Module(fp.read_bytes())
        self.assertEqual(spv.gmem_loop_stage(y, buffer_var(y, 0), 'vs', pidx=True), (0, 2, 0))
        self.assertEqual(y.data(), x.data())
        y = spv.Module(fp.read_bytes())
        self.assertEqual(spv.gmem_loop_stage(y, buffer_var(y, 0), 'vs', cache=False, pidx=False), (0, 0, 2))
        self.assertEqual(y.data(), fp.read_bytes())   # off: left as it is (counted)

    def test_pidx_refuses_other_formulas(self):   # the pool check or the device address not the translator's: the search stays
        for old, new in (('lo < el', 'lo <= el'), ('hl < lo ? 1u : 0u', 'hl <= lo ? 1u : 0u'), ('hi < eh', 'hi <= eh')):
            src = self.tmp / 'mut.spv'
            text = shader_glsl()
            self.assertIn(old, text)
            compile_glsl(text.replace(old, new), src)
            m = spv.Module(src.read_bytes())
            spv.gmem_flatten(m, buffer_var(m, 0), buffer_var(m, 1), SINK_WORD); spv.local_forward(m)
            self.assertEqual(spv.gmem_pidx(m, buffer_var(m, 0)), 0, new)

    def test_pidx_graphics_stages(self):   # a vertex shader and a fragment shader with helper lanes, demote and the loads in non-uniform control flow
        head = ['#version 460', '#extension GL_EXT_buffer_reference : require', '#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require',
                'layout(buffer_reference, std430, buffer_reference_align = 4) buffer PU { uint v; };',
                'layout(set = 0, binding = 0, std430) readonly buffer Map { uint w[]; } map_;', '', 'uint gload(uint lo, uint hi) {'] + search_glsl() + [
                '    if (inPool) {'] + host_glsl() + ['        return PU(packUint2x32(uvec2(hl, hh))).v;', '    }', '    return 0u;', '}', '']
        vs = head + ['layout(location = 0) in uvec4 a;', 'layout(location = 0) flat out uint o;', 'void main() {',
                     '    uint s = 0u; for (int k = 0; k < 4; k++) s += gload(a.x + uint(k) * 16u, a.y);',
                     '    if (a.z != 0u) s ^= gload(a.z, a.w);', '    o = s; gl_Position = vec4(float(s & 1u)); }']
        fs = ['#extension GL_EXT_demote_to_helper_invocation : require'] + head + ['layout(location = 0) flat in uvec4 a;', 'layout(location = 0) out vec4 col;',
              'void main() {', '    if ((a.w & 1u) != 0u) demote;', '    uint s = gload(a.x, a.y);',
              '    if (gl_FragCoord.x > 3.0) s += gload(a.x + 16u, a.y) + gload(a.z, a.y);',
              '    col = vec4(float(s), dFdx(float(s)), float(helperInvocationEXT()), 1.0); }']
        fs = [fs[1], fs[0]] + fs[2:]
        for stage, src, ext in (('vs', vs, 'vert'), ('ps', fs, 'frag')):
            p = self.tmp / f'g.{ext}'; p.write_text('\n'.join(src))
            out = self.tmp / f'g_{ext}.spv'
            r = subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', '-o', str(out), str(p)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            m = spv.Module(out.read_bytes())
            self.assertEqual(spv.gmem_flatten(m, buffer_var(m, 0), None, SINK_WORD), 1, stage)
            spv.local_forward(m)
            self.assertEqual(spv.gmem_loop_stage(m, buffer_var(m, 0), stage, pidx=True), (0, 1, 0), stage)
            xp = self.tmp / f'g_{ext}_pidx.spv'; xp.write_bytes(m.data())
            self.assertIsNone(validate(xp), stage)
            r = subprocess.run(['spirv-opt', '-O', '--target-env=vulkan1.3', str(xp), '-o', str(self.tmp / 'o.spv')], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    @unittest.skipUnless(shutil.which('cc') and os.path.exists('/usr/include/vulkan/vulkan.h'), 'a C compiler and the Vulkan headers needed for the GPU run')
    def test_pidx_same_results_on_gpu(self):
        for full in ((), ('full',)):
            for warm in (False, True):
                src, fp, xp, _, n = self.indexed(warm=warm)
                self.assertEqual(n, 2)
                base = self.run_gpu(src, *full)
                self.assertIn('sink untouched', base)
                for gran in (None, 1):   # the host's window (256 MiB rounding: records of no entry inside it) and the exact one (the sentinel right past the last pool)
                    self.assertEqual(self.run_gpu(xp, *full, gran=gran), base, (full, warm, gran))

    @unittest.skipUnless(shutil.which('cc') and os.path.exists('/usr/include/vulkan/vulkan.h'), 'a C compiler and the Vulkan headers needed for the GPU run')
    def test_same_results_on_gpu(self):
        src, m = self.build()
        flat = spv.Module(m.data()); spv.gmem_flatten(flat, buffer_var(m, 0), buffer_var(m, 1), SINK_WORD); spv.local_forward(flat)
        fp = self.tmp / 'flat.spv'; fp.write_bytes(flat.data())
        loop = spv.Module(flat.data()); self.assertEqual(spv.gmem_loop(loop, buffer_var(m, 0)), 2)
        lp = self.tmp / 'loop.spv'; lp.write_bytes(loop.data())
        runner = self.tmp / 'vk_gmem_run'
        r = subprocess.run(['cc', '-O1', '-o', str(runner), str(HERE / 'vk_gmem_run.c'), '-ldl'], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        res = {}
        for name, path in (('base', src), ('flat', fp), ('loop', lp)):
            r = subprocess.run([str(runner), str(path)], capture_output=True, text=True, timeout=120)
            if r.returncode == 77: self.skipTest(r.stdout.strip() or 'no Vulkan device')
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            res[name] = r.stdout
        self.assertEqual(res['base'], res['flat'])
        self.assertEqual(res['base'], res['loop'])
        self.assertIn('sink untouched', res['flat'])


if __name__ == '__main__':
    unittest.main()
