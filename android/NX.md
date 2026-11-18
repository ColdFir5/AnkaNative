# NX archive support

The `.nx` format (`NXAR` magic, wire versions `(0,2)` and `(0,3)`) is supported as a second self-contained container alongside [NUP](NUP.md). Each NX snapshot stores a complete tree (`exefs/` and `romfs/`) over a shared fixed-block store where individual blocks are either `RAW` or `ZSTD` compressed with 16 KiB CRC-32C integrity slices.

## Naming

`<file>.nx[:selector][/path]` mirrors NUP syntax:

- Split at the last `.nx:`. Only lowercase `.nx` is accepted (`.NX` is refused).
- Empty selector selects the highest-version update snapshot (`snapshot_type` 1 or 3), or else the base snapshot (`0`).
- `base` selects the base snapshot (`0`).
- `update/<decimal>` selects an update (`1` or `3`) with that `numeric_content_version`.
- `addon/<16-hex>` selects a DLC snapshot (`2`) matching lowercase `ADDON_TITLE_ID`.
- `aoc/<index>` selects a DLC snapshot (`2`) whose `ADDON_TITLE_ID & 0xfff` equals `<index>` (`DM_AOC_DIR/<index>.nx`).
- `/<path>` restricts the view to that subtree (the host appends `/romfs` for `DM_ROMFS` and add-ons).

## Host mount and reads

[dm/host/nx_image.h](../dm/host/nx_image.h) and [dm/host/hle_fs.c](../dm/host/hle_fs.c) mount an NX snapshot's `romfs` subtree onto the same `romfs_img` directory and lookup table as NUP, with `IMG_ENT_NX` entries pointing directly at the NX file record:

- **Mount validation**: validates dual header slots A/B at offsets `0` and `4096` (magic, header CRC-32C, generation tie-breaking), falls back to the older valid slot if the newer metadata is damaged, checks Catalog/Manifest/BlockIndex/Integrity SHA-256 hashes via `plat_sha256`, uses overflow-checked bounds arithmetic, skips required Catalog extensions `0x00010004`, `0x00010005`, and `0x00010008`, and rejects unknown required extensions. Logs one summary line (`[dm] fs NX <spec>: <n> entries, <nfiles> files, <nblocks> blocks (<raw> raw, <zstd> zstd), cache <mb> MiB`).
- **RAW block reads**: mapped slices are `pread` straight into the guest buffer through `nx_pread_guest` (with `dm_guest_write_prepare` and `EFAULT` retry handling).
- **ZSTD block reads**: whole blocks are decoded on demand with `ZSTD_decompress` (when built with `DM_NUP_ZSTD`) into a mutex-protected LRU block cache (`DM_NX_CACHE_MB`, default `64` MiB) and copied out. Without `DM_NUP_ZSTD`, headers still compile cleanly and compressed reads fail with a `[RED]` log line.
- **Verification**: every block slice touched by a read is CRC-32C checked by default (`DM_NX_VERIFY=0` disables slice checks); ARMv8 CRC32C hardware instructions are used when `__ARM_FEATURE_CRC32` is defined, with a lookup table fallback. Any mismatch logs `[RED]` naming the file and fails the read with `EIO`.
- **Accounting**: NX reads increment `fs_stat_note` bytes.

## Android launcher

[Nup.java](app/src/main/java/dev/nxnative/android/Nup.java) dispatches `open` / `peek` calls on `NXAR` magic or `.nx` extension to [Nx.java](app/src/main/java/dev/nxnative/android/Nx.java), keeping the same package, node, `stream`, and `bytes` API for callers (`Pack`, `PackFind`, `DumpLayout`, `BuildCheck`, `RomfsCheck`, `DumpCheck`, `MainActivity`, `GameActivity`, `Folders`, and `ReadinessActivity`). Both pure-Java launchers and APK code decode compressed blocks with [Zstd.java](app/src/main/java/dev/nxnative/android/Zstd.java) and verify CRC-32C slices and file SHA-256 digests when reading.

## Tests

- Host unit tests: `make -C dm/host/tests GAME=<id> build/t_nx` runs synthetic NX `(0,2)`/`(0,3)` format, concurrency, cache, slice CRC-32C, and add-on checks (`t_fs` covers NUP).
- Host comparison tool: `dm/host/tests/build/t_nx_compare <a.nx:sel/romfs> <b.nup:sel/romfs>` mounts both containers through `hle_fs.c` and checks file sizes, sequential reads, and random cross-block reads for every file present in both.
- JVM launcher tests: `python3 -m unittest dm/tools/tests/test_nup.py` runs `NupJvmTest` and [NxJvmTest](app/src/jvmtest/java/dev/nxnative/android/NxJvmTest.java); `NxJvmTest real <a.nx> <b.nup>` compares packages, RomFS file lists/sizes, `exefs/main.npdm`, and sampled RomFS SHA-256 hashes on real containers.

## Limits

- `dm/tools` Python preparation and packing scripts (`romfs_image.py`, `pack_android.py`, `prepare.py`) do not read `.nx` containers yet.
- Mac and iOS hosts are only verified to compile without `DM_NUP_ZSTD`; runtime testing and `.nx` app UI live on Android only.
