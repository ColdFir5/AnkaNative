# Anka Native changelog

Changes in Anka Native, grouped by topic. Most of the new behaviour is behind `DM_*` switches (every one is listed in
[docs/env_switches.tsv](docs/env_switches.tsv)); the ones Ocarina of Time needs are set by its game profile
([dm/games/0400486024DEC000/game.tsv](dm/games/0400486024DEC000/game.tsv)), so other games keep their old behaviour unless a switch is set.
The credits for the code it is built on are in [NOTICE.md](NOTICE.md).

## Anka Native 0.1.1

- **Prepare on the phone looked done but made nothing to play.** 0.1.0 was built without the on-phone packer (`-PnxPackEngine=fake`), so its
  Prepare ran the simulated engine: the progress finished, but the pack it made is a placeholder with no game code and no shaders, and Play could
  not start it. A play or release build without the packer now offers no Prepare (`Ondevice.available()`): Home and the game's card say to prepare
  the game on a computer (README "Install") and to pick that pack, and a placeholder pack left by 0.1.0 is shown as "No pack" and Play says why.
  Test builds keep the simulated engine for UI work.

## Anka Native 0.1.0

The first release: Ocarina of Time (the Switch 2 remake, `0400486024DEC000`) brought up on Android. Measured on one phone only, a POCO F8 Pro
(Snapdragon, Adreno 830, 12 GB of RAM) with a Turnip driver. Donkey Kong Bananza and Mario Kart World were not tested again after these changes.

### The name and the app

- The app is now called **Anka Native**: Anka is the Zumruduanka of Turkish myth, the phoenix that rises from its ashes. New app ids: the release app
  `dev.ankanative.android`, a per-game app `dev.ankanative.android.<suffix>` (`-PnxApp=oot` makes `dev.ankanative.android.oot`), the test app
  `dev.ankanative.android.test`. Each has its own data folder (`Android/data/<app id>/files`), so it installs beside older builds and does not touch their
  saves.
- The Anka mark (a phoenix rising, with flame wings) as the launcher icon, the in-app mark and the notification icon; the launcher's colours follow it
  (fire gold and orange on warm ivory, or on a warm night in dark mode). `android/tools/anka_icons.py` writes the icons from `anka_logo.py`.
- The committed translator `dm/gfx/tools/dmsh_xlate_bin/dmsh_xlate.dll` (and its `.pdb`) was rebuilt from the same source with the build folder mapped
  to `/_/` (`-p:PathMap`), so it names no local folder. Only the dll's fingerprint changed, so the chain versions moved (`DMSH_CHAIN` `d4f17b9bdef3`,
  library `v-0dd4341ae6ce`); a library of the previous version has the same `DMSH_VK_ABI` and is still played.
- Game profiles: only Ocarina of Time (`0400486024DEC000`), Donkey Kong Bananza (`040074A01BF12000`) and Mario Kart World (`0400C3F00006E000`) are
  kept; nxnative's 127 other profiles, their rows in the games table and the tests and tool that needed them (`test_astral_fps.py`,
  `check_thread_cuts.py`) were removed. `dm/tools/sdk_patches.py` is unchanged (it is part of the rewrite key, so packs stay current).
- About: the app names its license, GPL-3.0-or-later as a whole, and points to NOTICE.md; logs and the Vulkan application name say "Anka Native".
- Release signing: `-PnxLive=true` reads the key from `ANKA_KEYSTORE`, `ANKA_KEY_ALIAS`, `ANKA_KEYSTORE_PASSWORD` and `ANKA_KEY_PASSWORD` (environment
  or `~/.gradle/gradle.properties`); `android/tools/release.sh` builds, signs, verifies and checksums a release ([docs/RELEASING.md](docs/RELEASING.md)).
  `Signer.OFFICIAL` holds the release certificate's SHA-256 (`4d7443354f53…3bb0`, CN=EddizEge), so only builds signed with that key call
  themselves official.
- APK names `anka-native-<suffix>.apk`; exported packs `anka-pack-<title ID>-<version>.zip`; the in-app updater looks at this project's releases.
- Game settings > **RAM saver (uses storage)**, per game, off by default: the game's large write-once texture pools live in a file in the app's cache
  folder instead of in RAM (`DM_MEM_FILEBACK=1`, `DM_MEM_FILEBACK_KINDS=shader,texture`). For phones short of RAM with free storage: in Ocarina of Time
  it writes about 2 GB to storage per area load.

### Ocarina of Time (`0400486024DEC000`, update v131072)

- New game profile: boots from a `.nup` dump to the title screen, the opening movie and into the game; the code rewrite and the shader library were made
  on Linux.
- **Game speed at a low frame rate.** The game's frame-ratio division is stepped by the host (a frame-step patch, ported from upstream): at 15 fps the
  game ran at about a quarter of its speed, now at about 0.8-1.0.
- **Anti-aliasing shaders restored.** `shader_split.py` steps 8 bytes past a record that is not a control section (ported from upstream), which finds
  the game's SMAA shaders again (library `v-04fb6ae9ea34`). The pass runs, but its blend weights are still all zero, so there is no visible change yet.
- **A stand-in for agl's NDSR upscaler network** (`DM_VK_NDSR_STANDIN=1`): the network is not run; its output image gets its input sampled bilinearly.

### Memory (the game did not survive its first forest load on 12 GB before)

- **GPU texture-use feedback** (`DM_VK_TEXFB`): the shader library's modules set a bit for each texture-pool ID they index, so the host knows which named
  textures the GPU really uses; under memory pressure the ones it has not used are released (`DM_VK_TEXFB=2`). Its GPU cost was about zero (74.5 against
  75.2 ms). With it the forest load survived for the first time (peak 3,007 MiB).
- **Coarse-first textures** (`DM_VK_SCAN_COARSE`): under memory pressure a texture only a pool scan named is made at a coarse mip first (-157 MiB at the
  forest load).
- **Rate-based memory pressure** (`DM_VK_PRESS_RATE`) with hysteresis (`DM_VK_PRESS_HYST_MB`), and mip drop for more texture kinds (`DM_VK_MIPDROP`):
  pressure starts when available memory falls fast, not only when it is already low, and stays on long enough. Textures remade within 30 s fell from 66 %
  to 44 %, then to 6.6 % with the hysteresis; the remake after pressure ends is spread over frames (a 2.5 s stall became 0.36-0.44 s in 8 of 9 cases).
- **Guest-aliased render targets** (`DM_VK_RTALIAS`, `DM_VK_RTALIAS_MB`): a render target that starts at the same guest address as an earlier one shares
  its allocation. One depth target took a new size almost every frame and got new memory each time (+345 MiB in 110 frames); with aliasing, no extra
  render target memory, and the game left Link's house into Kokiri Forest.
- **File-backed memory pools** (`DM_MEM_FILEBACK`): shader-code pools (and, with the RAM saver, texture pools) as clean file pages the kernel can drop
  instead of compressing them into zram (page faults 2.7k/s to 0.9k/s). Shader-code pools are no longer placed (`DM_VK_PLACE_SHADER_POOLS=0`).
- Fewer page index buffers for Ocarina of Time (`DM_VK_GMEM_PIDX_BUFS=2`, -41 MiB) and a per-pool memory footprint log (`DM_MEM_FOOT`).

### Speed

- **Per-page pool index in the shaders** (library `v-fc7819d73925`): a shader that reads guest memory through a pointer found its pool with an unrolled
  search (253 searches, 82k instructions, 1,236 spills in the forest's terrain shaders); it now reads one 8-byte record per 4 KiB page. Vertex shaders
  became 3-4x faster (Link's house: 2,570 to 695 ns per invocation), GPU time at Link's house 121.6 to 59.1 ms, and the frame rate at the start 4.8-5.4 to
  12.7-16.2 fps. The index costs about 83 MiB.
- **Pipeline-cache saver capped** (`DM_VK_PCACHE_MIN_S`, `DM_VK_PCACHE_QUIET_S`): the pipeline cache was written 36.7 GB in 3 minutes; now 0.88 GB in
  8.6 minutes, and frames over 1 s while walking fell from 13 to 0.
- A sampler LOD bias floor (`DM_VK_LOD_BIAS_MIN=0`, off by default): softens texture shimmer by 10-25 % when on.

### Diagnostics

- Render-target, light-list and per-pipeline GPU time logs, pipeline statistics, NaN scans (`DM_VK_NANSCAN`) and memory footprint lines, used to find
  the causes above; all off by default.
