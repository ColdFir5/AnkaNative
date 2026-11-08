"""Pack tiny shader fixtures through build_ipa.sh's archive step, without Xcode or game files."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'dm/games'))
import game


class ProfilePackTests(unittest.TestCase):
    def test_pulling_libraries_follow_compiled_profile_and_explicit_override(self):
        # (pso_mtl4 on, IOS_PULL, with a pipeline archive, _bp libraries packed): an archive's bases are made of the pulling libraries, so an IPA with one carries them whatever IOS_PULL says.
        # The profile is Donkey Kong Bananza's (pso_mtl4 off); the 'on' cases take a derived copy of it with pso_mtl4 on (DM_GAMES_DIR, DM_GAMES_FIRST)
        gid = '040074A01BF12000'
        cases = [(True, None, False, True), (False, None, False, False),
                 (True, '0', False, False), (False, '1', False, True),
                 (False, None, True, True), (True, '0', True, True)]
        for mtl4, override, with_pso, included in cases:
            with self.subTest(pso_mtl4=mtl4, IOS_PULL=override, archive=with_pso), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dm = root / 'dm'
                profile = REPO / 'dm/games' / gid
                if mtl4:
                    derived = root / 'profiles' / gid; derived.mkdir(parents=True)
                    tsv = re.sub(r'(?m)^pso_mtl4\t.*\n', '', (profile / 'game.tsv').read_text())
                    (derived / 'game.tsv').write_text(tsv.rstrip('\n') + '\npso_mtl4\ton\n'); profile = derived
                header = dm / 'host/gen' / gid / 'game_profile.h'
                header.parent.mkdir(parents=True)
                header.write_text(game.header(game.Game(profile)))
                app, shaders, romfs = [root / name for name in ('DmIOS.app', 'shaders', 'romfs')]
                for folder in (app, shaders / 'b-fixture', romfs): folder.mkdir(parents=True)
                for name in ('fixture_b.metallib', 'fixture_bp.metallib', 'fixture_bv.metallib', 'fixture_b.json'):
                    (shaders / 'b-fixture' / name).write_text(name)
                (app / 'executable').write_text('app')
                (romfs / 'marker').write_text('romfs')
                (root / 'tools').symlink_to(REPO / 'dm/ios/tools', target_is_directory=True)
                out = root / 'fixture.ipa'
                env = {k: v for k, v in os.environ.items() if k not in ('IOS_PULL', 'DM_GAMES_DIR', 'DM_GAMES_FIRST')}
                env.update(GAME=gid, DM=str(dm), APP=str(app), MSL_IOS=str(shaders), MSLDIR='b-fixture',
                           AOC='', ROMFS=str(romfs), OUT_IPA=str(out), PROFILE_TOOL=str(REPO / 'dm/games/game.py'))
                if mtl4: env.update(DM_GAMES_DIR=str(root / 'profiles'), DM_GAMES_FIRST='1')
                if override is not None: env['IOS_PULL'] = override
                if with_pso:
                    pso = root / 'pso'; pso.mkdir(); (pso / 'prebuilt.tsv').write_text('fixture')
                    env['PSO'] = str(pso)
                # Exercise the actual archive step; only the preceding Xcode build is bypassed.
                script = (REPO / 'dm/ios/tools/build_ipa.sh').read_text()
                packing = script[script.index('\nP=Payload/DmIOS.app\n'):]
                result = subprocess.run(['/bin/bash', '-c', 'set -u\nt() { :; }; die() { echo "$*" >&2; exit 1; }\n'
                                         'gp() { python3 "$PROFILE_TOOL" "$GAME" "$1"; }\n' + packing],
                                        cwd=root, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                with zipfile.ZipFile(out) as archive:
                    names = set(archive.namelist())
                    prefix = 'Payload/DmIOS.app/msl/b-fixture/'
                    self.assertEqual(prefix + 'fixture_bp.metallib' in names, included)
                    for name in ('fixture_b.metallib', 'fixture_bv.metallib', 'fixture_b.json'):
                        self.assertIn(prefix + name, names)
                    self.assertEqual(archive.read('Payload/DmIOS.app/romfs/marker'), b'romfs')
                    self.assertEqual('Payload/DmIOS.app/pso/prebuilt.tsv' in names, with_pso)


if __name__ == '__main__':
    unittest.main()
