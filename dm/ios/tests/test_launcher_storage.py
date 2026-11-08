"""Compile and run the launcher's real settings model on macOS; no simulator or UI required."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'dm/games'))
import settings

# The launcher's lists are made for a test profile derived from Donkey Kong Bananza's (DM_GAMES_DIR): a 60 fps target, a render base of 1600x900 docked
# and 1280x720 handheld, present_snap and pso_mtl4 on (the defaults launcher_storage_check.swift expects)
TUNED, BASE = '0100000000001000', '040074A01BF12000'
TUNED_KEYS = {'name': 'Tuned test profile', 'fps_target': '60', 'render_base': '1600x900 1280x720', 'present_snap': 'on', 'pso_mtl4': 'on'}


def tuned_profile(root):
    d = Path(root) / 'profiles' / TUNED; d.mkdir(parents=True)
    rows = [r for r in (REPO / 'dm/games' / BASE / 'game.tsv').read_text(encoding='utf-8').split('\n')
            if r and not r.startswith('#') and r.split('\t')[0] not in TUNED_KEYS and r.split('\t')[0] not in ('title_id', 'build')]
    (d / 'game.tsv').write_text('\n'.join([f'title_id\t{TUNED}', *rows, *(f'{k}\t{v}' for k, v in TUNED_KEYS.items())]) + '\n', encoding='utf-8')
    return str(d.parent)


@unittest.skipUnless(sys.platform == 'darwin' and shutil.which('xcrun'), 'needs the macOS Swift toolchain')
class LauncherStorageTests(unittest.TestCase):
    def test_only_changed_choices_are_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'patched').mkdir()
            (tmp / 'patched/main.fixups.tsv').write_text('# RENDER_SCALE\t1\n')   # the render-scale row, a range
            with patch.dict(os.environ, {'GFXAPI': 'nvn', 'DM_GAMES_DIR': tuned_profile(tmp)}), patch.object(settings.nacp, 'languages', return_value=['en-US', 'ja']):
                (tmp / 'launcher.json').write_text(json.dumps(settings.build(TUNED, tmp / 'patched', tmp / 'control.nacp')))
                with patch.dict(os.environ, {'SETTINGS_PLATFORM': 'mac'}):   # the Mac window's list: Display mode, the render scale's base size
                    (tmp / 'launcher_mac.json').write_text(json.dumps(settings.build(TUNED, tmp / 'patched', tmp / 'control.nacp')))
            exe = tmp / 'launcher_storage_check'
            compile_result = subprocess.run(['xcrun', '-sdk', 'macosx', 'swiftc', '-swift-version', '5', '-parse-as-library', '-module-cache-path', str(tmp / 'module-cache'),
                            '-o', str(exe), str(REPO / 'dm/ios/Sources/LauncherSettings.swift'), str(REPO / 'dm/ios/Sources/LauncherRow.swift'), str(REPO / 'dm/mac/Launcher.swift'),
                            str(Path(__file__).with_name('launcher_layout_check.swift')), str(Path(__file__).with_name('launcher_storage_check.swift'))], capture_output=True, text=True, timeout=300)
            self.assertEqual(compile_result.returncode, 0, compile_result.stdout + compile_result.stderr)
            r = subprocess.run([str(exe), str(tmp / 'launcher.json'), str(tmp / 'launcher_mac.json')], capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn('launcher storage: 0 failed', r.stdout)


if __name__ == '__main__':
    unittest.main()
