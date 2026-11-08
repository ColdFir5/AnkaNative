"""The launcher's official-build stamp (android/README.md, Signer.java with android/app/src/jvmtest's SignerJvmTest): the signing certificate's SHA-256 as apksigner
prints it, the build tag's mark (official / unofficial build for the play and release apps, none for the test app) and the short signer id the logs and readiness
report carry. The release certificate's fingerprint is read from its one place, Signer.OFFICIAL: "not set" (NOT_SET) until the owner's release key exists
(docs/RELEASING.md), then 64 lower-case hex digits; the marks are checked against a sample certificate either way. The JVM part is skipped without a JDK."""
import hashlib
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / 'android/app/src/main/java/dev/nxnative/android/Signer.java'
TEST = REPO / 'android/app/src/jvmtest/java/dev/nxnative/android/SignerJvmTest.java'

def official_in_source():
    """Signer.OFFICIAL as the source has it: None while it is NOT_SET, else the fingerprint"""
    m = re.search(r'^    static final String OFFICIAL = (NOT_SET|"([0-9a-f]{64})");', SRC.read_text(encoding='utf-8'), re.M)
    assert m, 'Signer.java: OFFICIAL must be NOT_SET or "<64 lower-case hex digits>" (docs/RELEASING.md)'
    return m.group(2)


OFFICIAL = official_in_source()   # None until the release key exists
SAMPLE = hashlib.sha256(b'a sample release certificate').hexdigest()   # the marks against a given certificate
TAG = 'Anka Native 0.1 · 6a6448d'


@unittest.skipUnless(shutil.which('javac') and shutil.which('java'), 'no JDK')
class Stamp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix='signer_'))
        subprocess.run(['javac', '-Xlint:all', '-Werror', '-encoding', 'UTF-8', '-d', str(cls.tmp / 'classes'), str(SRC), str(TEST)], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls): shutil.rmtree(cls.tmp, ignore_errors=True)

    def java(self, *a):
        r = subprocess.run(['java', '-Dstdout.encoding=UTF-8', '-cp', str(self.tmp / 'classes'), 'dev.nxnative.android.SignerJvmTest', *map(str, a)],
                           capture_output=True, text=True, encoding='utf-8')
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout.rstrip('\n').split(' ', 1)[1]

    def test_official_constant(self):
        self.assertEqual(self.java('official'), OFFICIAL or 'not set')

    def test_release_certificate_in_effect(self):
        other = hashlib.sha256(b'someone else').hexdigest()
        if OFFICIAL is None:   # before the release key exists, nothing is official
            for signers in ((other,), (SAMPLE,), ('not set',), ()):
                self.assertEqual(self.java('tag', TAG, 1, *signers), TAG + ' · unofficial build')
        else:
            self.assertEqual(self.java('tag', TAG, 1, OFFICIAL), TAG + ' · official')
            self.assertEqual(self.java('tag', TAG, 1, other), TAG + ' · unofficial build')

    def test_sha256_as_hashlib(self):
        p = self.tmp / 'cert.der'
        p.write_bytes(bytes(range(256)) * 3)
        self.assertEqual(self.java('sha256', p), hashlib.sha256(p.read_bytes()).hexdigest())

    def test_marks(self):
        other = hashlib.sha256(b'someone else').hexdigest()
        self.assertEqual(self.java('tagwith', SAMPLE, TAG, 1, SAMPLE), TAG + ' · official')
        self.assertEqual(self.java('tagwith', SAMPLE, TAG, 1, SAMPLE.upper()), TAG + ' · official')
        self.assertEqual(self.java('tagwith', SAMPLE, TAG, 1, other), TAG + ' · unofficial build')
        self.assertEqual(self.java('tagwith', SAMPLE, TAG, 1, SAMPLE, other), TAG + ' · unofficial build')   # a second signer: not ours
        self.assertEqual(self.java('tagwith', SAMPLE, TAG, 1), TAG + ' · unofficial build')                  # nothing readable
        self.assertEqual(self.java('tagwith', SAMPLE, TAG, 0, SAMPLE), TAG)                                   # the test app makes no claim
        self.assertEqual(self.java('tagwith', SAMPLE, TAG, 0, other), TAG)
        self.assertEqual(self.java('tagwith', 'not set', TAG, 1, 'not set'), TAG + ' · unofficial build')     # an unset certificate matches nothing

    def test_build_id(self):
        """Home's build identifier and the overlay's build tag come from one function: the tag is the app's name and the identifier"""
        self.assertEqual(self.java('build', 0, '0.1.1-test', '4a150670aa', '-', 'android-launcher-2'),
                         'launcher-2@4a15067|Anka Native test · launcher-2@4a15067')
        self.assertEqual(self.java('build', 0, '0.1.1', 'unknown', '+dirty', 'int'), 'int@unknown+dirty|Anka Native test · int@unknown+dirty')
        self.assertEqual(self.java('build', 1, '0.1-release', '6a6448d123', '-', 'main'), '0.1 · 6a6448d|' + TAG)

    def test_short_id(self):
        other = hashlib.sha256(b'x').hexdigest()
        self.assertEqual(self.java('id', SAMPLE), SAMPLE[:16])
        self.assertEqual(self.java('id', SAMPLE, other), SAMPLE[:16] + '+1')
        self.assertEqual(self.java('id'), 'unknown')


if __name__ == '__main__':
    unittest.main()
