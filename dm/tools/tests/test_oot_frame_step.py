"""Ocarina of Time's frame-ratio stub (dm/games/0400486024DEC000/patches.py FRAME_STEP, ported from seedds/nxnative 7d8dcdf8): the emitted words run on a small
interpreter of exactly the instructions the stub uses (no unicorn needed), for each state of the frame step state page; and, when OOT_MAIN_IMG names the
decompressed v131072 main (session-2026-11-07 analysis/main.img), the site's original word there. No game files needed otherwise."""
import importlib.util
import os
from pathlib import Path
import struct
import sys
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location('oot_patches', Path(__file__).resolve().parents[2] / 'games/0400486024DEC000/patches.py')
patch = importlib.util.module_from_spec(spec); spec.loader.exec_module(patch)
BUILD, SITE, ORIG = 'e0f3ccfaf50065ae4ea01eb5aeb13b1a', 0x6e6cc4, 0x1e201828   # fdiv s8,s1,s0


def f32(bits): return struct.unpack('<f', struct.pack('<I', bits & 0xffffffff))[0]
def bits(x): return struct.unpack('<I', struct.pack('<f', x))[0]
def sx(v, n): return v - (1 << n) if v >> (n - 1) & 1 else v


class Cpu:
    """Runs the stub's words from pc until a branch leaves the patch area; the state page is a dict of offsets."""
    def __init__(self, words, area, mem):
        self.w, self.area, self.mem, self.x, self.s = words, area, mem, [0] * 32, [0.0] * 32

    def run(self, pc, limit=64):
        for _ in range(limit):
            i = (pc - self.area) // 4
            if not 0 <= i < len(self.w): return pc
            w, nxt = self.w[i], pc + 4; d, n = w & 31, w >> 5 & 31
            if w & 0x9f000000 == 0x90000000:   # adrp
                imm = sx(((w >> 5 & 0x7ffff) << 2) | (w >> 29 & 3), 21); self.x[d] = (pc & ~0xfff) + (imm << 12)
            elif w & 0xffc00000 == 0xb9400000: self.x[d] = self.mem.get(self.x[n] + (w >> 10 & 0xfff) * 4, 0) & 0xffffffff   # ldr w
            elif w & 0xffc00000 == 0xf9400000: self.x[d] = self.mem.get(self.x[n] + (w >> 10 & 0xfff) * 8, 0)   # ldr x
            elif w & 0xffc00000 == 0xf9000000: self.mem[self.x[n] + (w >> 10 & 0xfff) * 8] = self.x[d]   # str x
            elif w & 0xff000000 == 0x34000000:   # cbz w
                if not self.x[d] & 0xffffffff: nxt = pc + sx(w >> 5 & 0x7ffff, 19) * 4
            elif w & 0xfffffc00 == 0x1e230000: self.s[d] = float(self.x[n] & 0xffffffff)   # ucvtf s, w
            elif w & 0xffe0fc00 == 0x8b000000: self.x[d] = (self.x[n] + self.x[w >> 16 & 31]) & (1 << 64) - 1   # add x
            elif w & 0xffe0fc00 == 0x1e200800: self.s[d] = f32(bits(self.s[n] * self.s[w >> 16 & 31]))   # fmul s
            elif w & 0xffe0fc00 == 0x1e201800: self.s[d] = f32(bits(self.s[n] / self.s[w >> 16 & 31]))   # fdiv s
            elif w & 0xff800000 == 0x52800000: self.x[d] = (w >> 5 & 0xffff) << (16 * (w >> 21 & 3))   # movz w
            elif w & 0xfffffc00 == 0x1e270000: self.s[d] = f32(self.x[n])   # fmov s, w
            elif w & 0xfc000000 == 0x14000000: nxt = pc + sx(w & 0x3ffffff, 26) * 4   # b
            else: raise AssertionError(f'unexpected word {w:08x} at {pc:#x}')
            pc = nxt
        raise AssertionError('no exit')


class FrameRatio(unittest.TestCase):
    area, state = 0x5000000, 0x5100000

    def build(self):
        words, puts = [], []
        ctx = types.SimpleNamespace(here=lambda: self.area + 4 * len(words), emit=words.append, site=lambda off, orig: self.assertEqual((off, orig), (SITE, '2818201e')),
                                    put=lambda off, ws, why: puts.append((off, ws, why)), state=self.state, render_scale=0)
        patch.FRAME_STEP[BUILD](ctx)
        self.assertEqual(len(puts), 1); off, ws, _ = puts[0]
        self.assertEqual((off, len(ws)), (SITE, 1))
        self.assertEqual(ws[0] & 0xfc000000, 0x14000000); self.assertEqual(SITE + sx(ws[0] & 0x3ffffff, 26) * 4, self.area)   # b to the stub
        return words

    def run_stub(self, target, step, vsyncs, base, rate):
        mem = {self.state: bits(float(target)) if target else 0, self.state + 0x20: step, self.state + 0x30: vsyncs}
        cpu = Cpu(self.build(), self.area, mem)
        cpu.s[1], cpu.s[0], cpu.x[8], cpu.x[9] = base, float(rate), 0x1234, 0x5678
        end = cpu.run(self.area)
        self.assertEqual(end, SITE + 4)
        self.assertEqual((cpu.x[8], cpu.x[9], cpu.s[1]), (0x1234, 0x5678, base))   # w8 / w9 (read right after the site) and the base rate kept
        return cpu.s[8], mem[self.state + 0x30]

    def test_registered_for_the_profiles_main(self):
        self.assertIn(BUILD, patch.FRAME_STEP)

    def test_off_runs_the_games_division(self):
        for target, step in ((0, 2), (60, 0)):   # fps_target 0, or no step decided yet
            ratio, vs = self.run_stub(target, step, 100, 60.0, 60)
            self.assertEqual((ratio, vs), (1.0, 100))
            ratio, _ = self.run_stub(target, step, 100, 30.0, 30)
            self.assertEqual(ratio, 1.0)

    def test_ratio_is_the_hosts_step(self):
        for step in (1, 2, 3, 4):
            ratio, vs = self.run_stub(60, step, 1000, 60.0, 60)
            self.assertAlmostEqual(ratio, float(step)); self.assertEqual(vs, 1000 + step)
            ratio, _ = self.run_stub(60, step, 0, 30.0, 30)   # the game's 30 fps mode: base 30
            self.assertAlmostEqual(ratio, step / 2)

    @unittest.skipUnless(os.environ.get('OOT_MAIN_IMG'), 'OOT_MAIN_IMG: the decompressed v131072 main (session-2026-11-07 analysis/main.img)')
    def test_site_in_the_games_main(self):
        img = Path(os.environ['OOT_MAIN_IMG']).read_bytes()
        self.assertEqual(struct.unpack_from('<I', img, SITE)[0], ORIG)
        self.assertEqual(img[0x6e6be0:0x6e6be4].hex(), '7f2303d5')   # the function's pacibsp


if __name__ == '__main__':
    unittest.main()
