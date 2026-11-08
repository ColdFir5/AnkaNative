#!/usr/bin/env python3
# Text files written with LF line ends on Windows, as on Linux and the Mac. Python on Windows writes '\r\n' for every '\n' of a file opened in text mode;
# the phone's host and launcher read the tables the packing tools write (fixups.tsv, pack.tsv, the keys, shader_map.tsv ...) with LF, and the tools hash
# some of them byte for byte. install() makes open() (builtins.open, io.open, and so pathlib's open, read_text and write_text) default to newline='\n' for a
# file opened for writing in text mode; a caller that names newline keeps it, and reading is not changed. Elsewhere it does nothing.
# Usage: python lf_newlines.py <script> [args...]   runs script as its __main__ with install() done (pack_android.py runs its tools so on Windows)
import builtins, io, os, runpy, sys

_open = io.open


def _lf_open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True, opener=None):
    if newline is None and 'b' not in mode and any(c in mode for c in 'wax'): newline = '\n'
    return _open(file, mode, buffering, encoding, errors, newline, closefd, opener)


def install():
    if os.name == 'nt': builtins.open = io.open = _lf_open


if __name__ == '__main__':
    install()
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name='__main__')
