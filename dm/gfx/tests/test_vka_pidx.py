"""The host's page index for raw global memory (dm/gfx/vk_android/vka_pidx.h, vka_draw.c gmem_pidx; dmsh_abi_vk.h DMSH_VK_GMEM_PIDX_*): vka_pidx_test.c keeps one
index across random sequences of table entries (segments placed, given back, re-placed, merged, windows that move, entries off the page or past 2^44), with the
host's window rounding and with others, and checks every record and gmem_pidx's lookup against the table after each step. The GPU run of the shaders reading such an
index (test_vk_gmem_flatten.py test_pidx_same_results_on_gpu) builds it with the same header."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


@unittest.skipUnless(shutil.which('cc'), 'a C compiler')
class PidxHost(unittest.TestCase):
    def test_random_sequences(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / 'vka_pidx_test'
            r = subprocess.run(['cc', '-O2', '-Wall', '-Wextra', '-Werror', '-o', str(exe), str(HERE / 'vka_pidx_test.c')], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            for gran in (65536, 4096, 1):   # the host's 256 MiB, then finer windows (more full builds)
                for seed in (1, 2, 3):
                    r = subprocess.run([str(exe), '500', str(seed), str(gran)], capture_output=True, text=True, timeout=600)
                    self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
                    ok, steps, full, inc, written = r.stdout.split()
                    self.assertEqual((ok, steps), ('ok', '500'))
                    self.assertGreater(int(full), 1); self.assertGreater(int(inc), 100); self.assertGreater(int(written), 0)


if __name__ == '__main__':
    unittest.main()
