"""The Android library build's caches (vk_android_delta.py BuildCache, used by vk_android_build.py; android/SHADERS.md#build-speed): the translator's output per job and
each finished module's stored form. An entry is used only when its stamp and bytes match: another translator, chain, profile inputs or job (its code bytes, answers,
attributes) makes it stale; another validator, zstd level or library makes a module entry stale; a cut, flipped or misnamed entry is corrupt; each such miss is counted and
the entry is made again. The code file's name (a batch folder's) does not count. A full build's prune keeps what it and the full build before it used."""
from pathlib import Path
import os
import shutil
import sys
import tempfile
import time
import unittest
import collections

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[0] / 'tools'
sys.path.insert(0, str(TOOLS))
import vk_android_delta as delta  # noqa: E402

try: from compression import zstd
except ImportError:
    try: from backports import zstd  # pyright: ignore[reportMissingImports]
    except ImportError: zstd = None

VK = delta.VK
SPV = b'\x03\x02\x23\x07' + bytes(range(256)) * 9


class Cache(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='vkcache_'))
        self.w = self.tmp / 'w'; self.w.mkdir()
        self.counts = collections.Counter()
        self.c = self.cache()

    def tearDown(self): shutil.rmtree(self.tmp, ignore_errors=True)

    def cache(self, xstamp='x1', mstamp='m1'):
        return delta.BuildCache(str(self.tmp / 'cache'), xstamp, mstamp, zstd, lambda what: self.counts.update([what]))

    def job(self, code=b'code', folder='b1', **kw):
        d = self.tmp / folder; d.mkdir(exist_ok=True)
        (d / 'k.bin').write_bytes(code)
        return {'key': '00000000000000aa', 'stage': 'ps', 'code': str(d / 'k.bin'), 'attr': {}, 'subgroup_size': 64, **kw}

    def translated(self, key='00000000000000aa', spv=SPV, res=b'{"notes": []}'):
        (self.w / f'{key}.spv').write_bytes(spv); (self.w / f'{key}.res.json').write_bytes(res)

    def get(self, digest, key='00000000000000aa', tag='', c=None):
        for f in self.w.iterdir(): f.unlink()
        return (c or self.c).xlate_get(key, tag, digest, str(self.w))

    def test_job_digest(self):
        d = delta.job_digest(self.job(), 'x1')
        self.assertEqual(delta.job_digest(self.job(folder='b2'), 'x1'), d, 'the code file\'s folder does not count')
        self.assertNotEqual(delta.job_digest(self.job(code=b'code2'), 'x1'), d, 'the code bytes count')
        self.assertNotEqual(delta.job_digest(self.job(subgroup_size=32), 'x1'), d, 'an answer counts')
        self.assertNotEqual(delta.job_digest(self.job(attr={'3': 'uint'}), 'x1'), d, 'the integer-input attributes count')
        self.assertNotEqual(delta.job_digest(self.job(debug=False), 'x1'), d, 'XLATE_DEBUG counts')
        self.assertNotEqual(delta.job_digest(self.job(), 'x2'), d, 'the translator stamp counts')

    def test_translator_round_trip_and_stale(self):
        d = delta.job_digest(self.job(), 'x1')
        self.assertFalse(self.get(d)); self.assertEqual(self.counts['translator miss'], 1)
        self.translated(); self.c.xlate_put('00000000000000aa', '', d, str(self.w))
        self.assertTrue(self.get(d)); self.assertEqual(self.counts['translator hit'], 1)
        self.assertEqual((self.w / '00000000000000aa.spv').read_bytes(), SPV)
        self.assertEqual((self.w / '00000000000000aa.res.json').read_bytes(), b'{"notes": []}')
        self.assertFalse(self.get(delta.job_digest(self.job(code=b'other'), 'x1'))); self.assertEqual(self.counts['translator stale'], 1)
        self.assertFalse(self.get(d, tag='.int'), 'the integer-input translation is its own entry'); self.assertEqual(self.counts['translator miss'], 2)
        self.assertFalse(self.get(d, key='00000000000000bb'))

    def test_translator_corrupt(self):
        d = delta.job_digest(self.job(), 'x1')
        self.translated(); self.c.xlate_put('00000000000000aa', '', d, str(self.w))
        p = Path(self.c.xlate_path('00000000000000aa')); good = p.read_bytes()
        for bad in (good[:-1], good[:-5] + bytes([good[-5] ^ 1]) + good[-4:], good[:20], b'DMVKXC01', b'not an entry at all', good.replace(b'00000000000000aa', b'00000000000000bb', 1)):
            p.write_bytes(bad); n = self.counts['translator corrupt']
            self.assertFalse(self.get(d)); self.assertEqual(self.counts['translator corrupt'], n + 1, bad[:40])
            self.assertFalse((self.w / '00000000000000aa.spv').exists(), 'nothing written from a corrupt entry')
        p.write_bytes(good); self.assertTrue(self.get(d))

    def test_module_round_trip_stale_corrupt(self):
        import hashlib
        h = hashlib.sha256(SPV).hexdigest()
        self.assertIsNone(self.c.module_get(h, len(SPV))); self.assertEqual(self.counts['module miss'], 1)
        stored, codec = (zstd.compress(SPV, level=19), VK['CODEC_ZSTD']) if zstd else (SPV, VK['CODEC_RAW'])
        self.c.module_put(h, stored, codec)
        self.assertEqual(self.c.module_get(h, len(SPV)), (stored, codec)); self.assertEqual(self.counts['module hit'], 1)
        self.assertIsNone(self.cache(mstamp='m2').module_get(h, len(SPV)), 'another validator / zstd: stale'); self.assertEqual(self.counts['module stale'], 1)
        self.assertIsNone(self.c.module_get(h, len(SPV) + 4)); self.assertEqual(self.counts['module corrupt'], 1)
        p = Path(self.c.module_path(h)); good = p.read_bytes()
        p.write_bytes(good[:-1]); self.assertIsNone(self.c.module_get(h, len(SPV))); self.assertEqual(self.counts['module corrupt'], 2)
        # an entry under another module's name (decodes, but to other bytes)
        other = SPV[:-4] + b'\0\0\0\1'; ho = hashlib.sha256(other).hexdigest()
        os.makedirs(os.path.dirname(self.c.module_path(ho)), exist_ok=True); Path(self.c.module_path(ho)).write_bytes(good)
        self.assertIsNone(self.c.module_get(ho, len(other))); self.assertEqual(self.counts['module corrupt'], 3)
        p.write_bytes(good); self.assertEqual(self.c.module_get(h, len(SPV)), (stored, codec))

    def test_prune(self):
        import hashlib
        hs = [hashlib.sha256(bytes([i]) * 8).hexdigest() for i in range(3)]
        for h in hs: self.c.module_put(h, b'x', VK['CODEC_RAW'])
        old = time.time() - 1000
        for h in hs: os.utime(self.c.module_path(h), (old, old))
        self.assertEqual(self.c.prune(old + 10), 0, 'the first full build removes nothing (no full build before it)')
        os.utime(self.c.module_path(hs[0]), (old + 20, old + 20))   # used by that build
        self.assertEqual(self.c.prune(time.time()), 2, 'the second removes what neither used')
        self.assertTrue(os.path.exists(self.c.module_path(hs[0])))
        (n, _), (m, b) = self.c.size()
        self.assertEqual((n, m, b), (0, 1, os.path.getsize(self.c.module_path(hs[0]))))

    def test_translator_id(self):
        d = self.tmp / 'xl'; d.mkdir()
        (d / 'dmsh_xlate').write_text('#!/bin/sh\n'); (d / 'a.dll').write_bytes(b'1'); (d / 'sub').mkdir()
        i = delta.translator_id(str(d / 'dmsh_xlate'), False, b'rt 9')
        self.assertNotEqual(delta.translator_id(str(d / 'dmsh_xlate'), False, b'rt 10'), i, 'the dotnet runtimes count')
        (d / 'a.dll').write_bytes(b'2')
        self.assertNotEqual(delta.translator_id(str(d / 'dmsh_xlate'), False, b'rt 9'), i, 'a dll beside the program counts')
        lib = d / 'libdmsh_xlate.so'; lib.write_bytes(b'lib')
        j = delta.translator_id(str(lib), True)
        (d / 'a.dll').write_bytes(b'3')
        self.assertEqual(delta.translator_id(str(lib), True), j, 'the library form is its one file')


if __name__ == '__main__':
    unittest.main()
