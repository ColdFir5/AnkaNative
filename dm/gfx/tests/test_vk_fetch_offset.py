"""Texel fetches with a register offset (vk_android_spv.py fetch_offset): Vulkan takes the Offset image operand only on gathers without maintenance8, so spirv-val refused
Ocarina of Time's AA pass d17eefaff4f0bc23 (10 such fetches) and it was not in the library. A compute module in the translator's shape (OpImageFetch Lod|Offset with an
ivec2 offset computed at run time) is rewritten to fetch at coordinate + offset with Lod alone and then passes spirv-val; a constant offset and a fetch whose coordinate
and offset types differ are left as they are."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / 'tools'))
import vk_android_spv as spv  # noqa: E402

ASM = '''
OpCapability Shader
OpCapability ImageGatherExtended
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %gid %tex %buf
OpExecutionMode %main LocalSize 8 8 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %tex DescriptorSet 0
OpDecorate %tex Binding 0
OpDecorate %buf DescriptorSet 0
OpDecorate %buf Binding 1
OpDecorate %rt ArrayStride 16
OpMemberDecorate %blk 0 Offset 0
OpDecorate %blk Block
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4 = OpTypeVector %float 4
%int = OpTypeInt 32 1
%uint = OpTypeInt 32 0
%v2i = OpTypeVector %int 2
%v3u = OpTypeVector %uint 3
%v3i = OpTypeVector %int 3
%img = OpTypeImage %float 2D 0 0 0 1 Unknown
%pimg = OpTypePointer UniformConstant %img
%tex = OpVariable %pimg UniformConstant
%rt = OpTypeRuntimeArray %v4
%blk = OpTypeStruct %rt
%pblk = OpTypePointer StorageBuffer %blk
%buf = OpVariable %pblk StorageBuffer
%pv4 = OpTypePointer StorageBuffer %v4
%pin = OpTypePointer Input %v3u
%gid = OpVariable %pin Input
%i0 = OpConstant %int 0
%i1 = OpConstant %int 1
%u0 = OpConstant %uint 0
%c11 = OpConstantComposite %v2i %i1 %i1
%main = OpFunction %void None %fn
%l = OpLabel
%g = OpLoad %v3u %gid
%gx = OpCompositeExtract %uint %g 0
%gy = OpCompositeExtract %uint %g 1
%x = OpBitcast %int %gx
%y = OpBitcast %int %gy
%c = OpCompositeConstruct %v2i %x %y
%ox = OpBitwiseAnd %int %x %i1
%o = OpCompositeConstruct %v2i %ox %i1
%im = OpLoad %img %tex
%f = OpImageFetch %v4 %im %c Lod|Offset %i0 %o
%fc = OpImageFetch %v4 %im %c Lod|ConstOffset %i0 %c11
%s = OpFAdd %v4 %f %fc
%p = OpAccessChain %pv4 %buf %i0 %gx
OpStore %p %s
OpReturn
OpFunctionEnd
'''


@unittest.skipUnless(shutil.which('spirv-as') and shutil.which('spirv-val') and shutil.which('spirv-dis'), 'needs spirv-as, spirv-val and spirv-dis')
class FetchOffset(unittest.TestCase):
    def build(self, asm):
        with tempfile.TemporaryDirectory() as d:
            s, o = Path(d) / 'a.spvasm', Path(d) / 'a.spv'
            s.write_text(asm)
            subprocess.run(['spirv-as', '--target-env', 'vulkan1.3', str(s), '-o', str(o)], check=True)
            return o.read_bytes()

    def val(self, data):
        with tempfile.NamedTemporaryFile(suffix='.spv') as f:
            f.write(data); f.flush()
            return subprocess.run(['spirv-val', '--target-env', 'vulkan1.3', f.name], capture_output=True, text=True)

    def dis(self, data):
        with tempfile.NamedTemporaryFile(suffix='.spv') as f:
            f.write(data); f.flush()
            return subprocess.run(['spirv-dis', '--raw-id', f.name], capture_output=True, text=True, check=True).stdout

    def test_register_offset_becomes_add(self):
        data = self.build(ASM)
        self.assertIn('Offset-10213', self.val(data).stderr)   # as the translator emits it: refused
        m = spv.Module(data)
        self.assertEqual(spv.fetch_offset(m), 1)
        out = m.data()
        self.assertEqual(self.val(out).returncode, 0, self.val(out).stderr)
        text = self.dis(out)
        self.assertNotIn('Lod|Offset', text)
        self.assertIn('Lod|ConstOffset', text)   # (a constant offset is valid: left)
        self.assertEqual(text.count('OpIAdd %'), 1)
        self.assertEqual(spv.fetch_offset(spv.Module(out)), 0)

    def test_mismatched_types_left(self):
        asm = ASM.replace('%o = OpCompositeConstruct %v2i %ox %i1', '%o = OpCompositeConstruct %v3i %ox %i1 %i1')
        m = spv.Module(self.build(asm))   # (not valid SPIR-V: spirv-as does not check it; the pass must leave such a fetch)
        before = m.data()
        self.assertEqual(spv.fetch_offset(m), 0)
        self.assertEqual(m.data(), before)


if __name__ == '__main__':
    unittest.main()
