"""Storage buffers read from a base the host writes (vk_android_spv.py ssbo_base; dmsh_abi_vk.h DMSH_VK_F_SSBO_BASE, DMSH_VK_C0_SSBO_BASE; vka_bind.c vka_c0_bases,
vka_draw.c ssbo_desc). A compute shader in the translator's shape (GlobalToStorage with the ssbo_align answer, written as GLSL and compiled with glslang: the slot's address
loaded from its c0 table entry, index ((addr_lo + offset) - (addr_lo & -64)) >> 2) is rewritten: the mask becomes a load of the entry's base word, spirv-val passes; a mask of
another c0 word leaves the module as it is (-1), one of another uniform buffer's word (SBUF_UBO) stays. Where a Vulkan device and a C compiler are present
(vk_ssbo_base_run.c), the shader reads 64 offsets around a slot address, some below it (Ocarina of Time's UI text reads glyph records there) and the guest addresses
cross a 4 GiB boundary: as built, with the descriptor at the rounded address (the host before the fix), every read below that address is wrong; rewritten, with the
descriptor and base word the host gives a DMSH_VK_F_SSBO_BASE entry, every read is right; rewritten with the window off (DM_VK_SSBO_EXACT) it reads as before."""
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[0] / 'tools'
sys.path.insert(0, str(TOOLS))
import vk_android_spv as spv  # noqa: E402
import vk_android_delta as delta  # noqa: E402

VK = delta.vk_values()
ALIGN, WORD = VK['SSBO_ALIGN'], VK['C0_SSBO_BASE'] // 4
ROW = 0x310 // 16   # dmsh_abi.h DMSH_C0_SSBO_CS: slot 0 of the compute stage's c0 storage table

SHADER = '''#version 460
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std140) uniform C0 { vec4 c[160]; } vp_c0;   // the driver segment c0, as the translator declares it (vec4 rows)
layout(set = 0, binding = 1, std430) buffer S0 { uint w[]; } vp_s0;
layout(set = 0, binding = 2, std430) buffer Cases { int off[]; } cases;
layout(set = 0, binding = 3, std430) buffer Out { uint o[]; } out_;
%s
void main() {
    uint i = gl_GlobalInvocationID.x;
    int a = floatBitsToInt(vp_c0.c[%d].x) + cases.off[i];
    int b = floatBitsToInt(%s) & -64;
    out_.o[i] = vp_s0.w[uint(a - b) >> 2];
}
'''


def compile_glsl(src, out):
    p = out.with_suffix('.comp'); p.write_text(src)
    r = subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', '-o', str(out), str(p)], capture_output=True, text=True)
    if r.returncode: raise AssertionError(r.stdout + r.stderr)


def validate(path):
    r = subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', str(path)], capture_output=True, text=True)
    return None if r.returncode == 0 else (r.stdout + r.stderr)


def var_at(m, binding):
    return next(v for v, d in m.decos.items() if d.get(spv.DEC_BINDING) == [binding] and v in m.vars)


def masks(m):
    """OpBitwiseAnd instructions with the -ssbo_align constant"""
    neg = -ALIGN & 0xffffffff
    return [x for x in m.ins if x[0] & 0xffff == spv.OP_BITWISE_AND and any(m.consts.get(o, (0, None))[1] == [neg] for o in x[3:5])]


@unittest.skipUnless(shutil.which('glslangValidator') and shutil.which('spirv-val'), 'glslangValidator and spirv-val needed')
class SsboBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='ssbobase'))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def build(self, name, base_expr=f'vp_c0.c[{ROW}].x', extra=''):
        src = self.tmp / (name + '.spv')
        compile_glsl(SHADER % (extra, ROW, base_expr), src)
        return src, spv.Module(src.read_bytes())

    def test_rewrite_valid(self):
        src, m = self.build('base')
        self.assertEqual(len(masks(m)), 1)
        n = spv.ssbo_base(m, var_at(m, 0), {ROW}, ALIGN, WORD)
        self.assertEqual(n, 1)
        out = self.tmp / 'fixed.spv'; out.write_bytes(m.data())
        self.assertIsNone(validate(out))
        self.assertEqual(masks(m), [])
        rows = {tuple(spv.const_int(m, i) for i in x[4:]) for x in m.ins if x[0] & 0xffff == spv.OP_ACCESS_CHAIN and x[3] == var_at(m, 0)}
        self.assertIn((0, ROW, WORD), rows)   # the base word of the same entry

    def test_other_shapes(self):
        # a mask of another c0 row than the shader's storage slots: not a base the host writes, the module stays as it is
        src, m = self.build('other_row', base_expr=f'vp_c0.c[{ROW + 1}].x')
        before = m.data()
        self.assertEqual(spv.ssbo_base(m, var_at(m, 0), {ROW}, ALIGN, WORD), -1)
        self.assertEqual(m.data(), before)
        # another component (the size word): the same
        src, m = self.build('other_comp', base_expr=f'vp_c0.c[{ROW}].z')
        self.assertEqual(spv.ssbo_base(m, var_at(m, 0), {ROW}, ALIGN, WORD), -1)
        # an address kept in the game's own uniform buffer (SBUF_UBO): no base word there, the mask stays and the host binds it at the rounded address
        src, m = self.build('ubo', base_expr='g.u[2].x', extra='layout(set = 0, binding = 4, std140) uniform G { vec4 u[4]; } g;')
        before = m.data()
        self.assertEqual(spv.ssbo_base(m, var_at(m, 0), {ROW}, ALIGN, WORD), 0)
        self.assertEqual(m.data(), before)
        # no storage slot read through c0: nothing to do
        src, m = self.build('none')
        self.assertEqual(spv.ssbo_base(m, var_at(m, 0), set(), ALIGN, WORD), 0)

    @unittest.skipUnless(shutil.which('cc') and os.path.exists('/usr/include/vulkan/vulkan.h'), 'a C compiler and the Vulkan headers needed for the GPU run')
    def test_reads_below_the_address_on_gpu(self):
        src, m = self.build('base')
        self.assertEqual(spv.ssbo_base(m, var_at(m, 0), {ROW}, ALIGN, WORD), 1)
        fixed = self.tmp / 'fixed.spv'; fixed.write_bytes(m.data())
        runner = self.tmp / 'vk_ssbo_base_run'
        r = subprocess.run(['cc', '-O1', '-Wall', '-o', str(runner), str(HERE / 'vk_ssbo_base_run.c'), '-ldl'], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        res = {}
        for name, path, mode in (('before', src, 'aligned'), ('fixed', fixed, 'window'), ('exact', fixed, 'aligned')):
            r = subprocess.run([str(runner), str(path), mode], capture_output=True, text=True, timeout=120)
            if r.returncode == 77: self.skipTest(r.stdout.strip() or 'no Vulkan device')
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            g = re.search(r'below the rounded address right (\d+)/(\d+), others right (\d+)/(\d+)', r.stdout)
            self.assertIsNotNone(g, r.stdout)
            res[name] = tuple(int(v) for v in g.groups())
        below = res['fixed'][1]
        self.assertGreater(below, 10)
        self.assertEqual(res['before'][0], 0, res)                              # the bug: no read below the rounded address reached its word
        self.assertEqual(res['before'][2], res['before'][3], res)               # the reads above it did
        self.assertEqual(res['fixed'], (below, below, res['fixed'][3], res['fixed'][3]), res)   # the fix: every read right, across the 4 GiB boundary
        self.assertEqual(res['exact'], res['before'], res)                       # the window off: as before


if __name__ == '__main__':
    unittest.main()
