<p align="center"><img src="android/images/logo.png" alt="Anka Native logo" width="128"></p>

# Anka Native

**Anka Native runs Switch 2 games' own code natively on Android, with their graphics rebuilt on Vulkan.** Released as
**Anka Native by EddizEge**; this release is **Anka Native 0.1.0**.

It is not an emulator: the game's own ARM64 code runs directly on the phone's CPU. A small host answers the system calls the game
makes, and the game's NVN graphics (including the Switch 2's task and mesh shaders) are replayed on Vulkan, with the game's shaders
translated once, ahead of time, from your own dump.

Not affiliated with Nintendo, any game publisher, or the projects credited in [NOTICE.md](NOTICE.md); please report problems in this
repository's Issues, not to them. https://github.com/EddizEge/AnkaNative/issues

> [!CAUTION]
> Nothing from any game is in this repository or in the app: no game code, no game data, no keys, no firmware, no shader libraries.
> Use only your own, legally owned game dumps. Do not ask for games, keys or dumps, and do not share them in Issues.

## The name

Anka is the Zumruduanka (Zümrüdüanka) of Turkish myth, the phoenix that rises from its ashes. Anka Native shares its name and its mark
with the Anka emulator by the same author: here the games rise again on a phone, from their own code.

## Games

| Game | Title ID | Version | Status |
|---|---|---|---|
| The Legend of Zelda: Ocarina of Time (Switch 2) | `0400486024DEC000` | update v131072 (`.nup` dump) | **Experimental, playable with problems.** Measured on one phone (below) |
| Donkey Kong Bananza | `040074A01BF12000` | update 458752 | Played on Android before Anka Native's changes; **not tested again since** |
| Mario Kart World | `0400C3F00006E000` | 1.8.0 (base game) | Experimental before Anka Native's changes (in game and driving, with graphics problems); **not tested again since** |

Other versions are refused with a message that says which version the app needs. Each game is built into its own app
(for example **Anka Native** with the id `dev.ankanative.android.oot` for Ocarina of Time).

**Measured in Ocarina of Time, on one phone.** Every number here comes from a POCO F8 Pro (Snapdragon, Adreno 830, 12 GB of RAM) with a
Turnip driver, playing the game from a `.nup` dump. No other phone was tried, and Donkey Kong Bananza and Mario Kart World were not run
with these changes.

- From the save it loads into Kokiri Forest and Link's house; it stayed in the forest for 18 minutes without being closed, and saving
  from the pause menu (System > Save) worked.
- Frame rate at the start of the game: 12.7-16.2 fps (it was 4.8-5.4 before the shader page index). GPU time at Link's house 59.1 ms a
  frame (from 121.6 ms). Not 30 fps: the forest's terrain shaders are still the main cost.
- At a low frame rate the game now runs at about 0.8-1.0 of its normal speed (it was about a quarter at 15 fps).
- Memory is the limit on 12 GB: about 10 of the phone's 10.9 GiB were in use while it played, and Android closed background apps. With
  the RAM saver on, total RAM use was 9.6 GB.

What changed, with more numbers: [CHANGELOG_ANKA_NATIVE.md](CHANGELOG_ANKA_NATIVE.md). The game's notes:
[dm/games/0400486024DEC000/README.md](dm/games/0400486024DEC000/README.md).

## What you need

- Android 13 or newer, a 64-bit phone with a **Snapdragon (Adreno) GPU** and **12-16 GB of RAM**. Only a 12 GB POCO F8 Pro (Adreno 830)
  was tried with this release; with 12 GB, memory is tight.
- **Less than 16 GB of RAM: turn the RAM saver on** (the game's ⋯ › Game settings › RAM saver). Without it Android may close the
  game in heavy areas; with it, total RAM use in Ocarina of Time dropped to 9.6 GB on the 12 GB test phone. It costs storage writes
  (about 2 GB per area load) and needs that much free space.
- A Vulkan driver with `VK_EXT_map_memory_placed`. Phone drivers often lack it (the POCO F8 Pro's own Qualcomm driver does): install a
  Turnip driver, such as the [Anka Turnip driver](#the-anka-turnip-driver). The app's readiness check (Settings > Readiness check)
  says whether yours has it.
- **Your own game dump**, of a game you legally own: a `.nup` file or a dump folder (`base/` with `exefs/` and `romfs/`, and
  `updates/<version>/` if any). NSP and XCI files do not work for Switch 2 games. No keys or firmware are needed.
- A computer (Linux or Mac) to prepare the game once (see "Install"). Preparing on the phone is not part of this release's Ocarina of
  Time app.
- Free storage on the phone: the prepared game (the pack, with its shader library) is a few hundred MB to a few GB; the RAM saver writes
  about 2 GB per area load.
- Optional: your own `Lossless.dll` from Lossless Scaling for Windows, for frame generation (never included).

## Download and check

The release has the APK (`anka-native-oot.apk` for Ocarina of Time) and `SHA256SUMS.txt`. To check a download on a computer:
`sha256sum -c SHA256SUMS.txt --ignore-missing` (on a Mac: `shasum -a 256 -c SHA256SUMS.txt`). Only download the app from this
repository's Releases: a copy from anywhere else may be modified.

## Install

This is an experimental build with no warranty ([LICENSE](LICENSE)).

1. **Prepare the game on a computer**, once: make the environment and run `scripts/pack_android.py` on your dump, as
   [android/README.md](android/README.md#the-environment) describes ([Pack a game](android/README.md#pack-a-game)). It rewrites the
   game's code for Android and translates its shaders for Vulkan into a pack (`<title ID>-work/android-pack`, next to the dump).
   Ocarina of Time has about 42,000 shaders: this takes a long time and needs .NET and several GB of free disk.
2. Copy the dump (or the `.nup`) to the phone, for example to `Switch2/`, and the pack next to it as
   `Switch2/<title ID>-work/android-pack`. The pack is made from your dump for your phone: never share it.
3. Install the APK, open **Anka Native** and allow **All files access**.
4. Install a GPU driver if the readiness check asks for one ([below](#the-anka-turnip-driver)).
5. Tap **Add game** and pick the dump folder or the `.nup`. The app finds the pack next to it; tap **Play**.

Saves are kept in the app's own storage (`Android/data/<app id>/files/saves`). When you uninstall, Android asks whether to keep that
data. A game's ⋯ menu backs saves up to `Switch2/save-backups/<title ID>/` (kept when the app is uninstalled) and restores them.
Anka Native has its own app id, so it installs beside other builds and does not see their saves; move a save with the game's
⋯ menu (back up in one app, restore in the other).

## The Anka Turnip driver

The games run on Turnip, Mesa's open Vulkan driver for Adreno GPUs. The Anka Turnip driver is a Turnip build prepared for Anka Native:
https://github.com/EddizEge/AnkaTurnip (download its release zip). In the app: Settings > **GPU driver** > **Install...** > **Pick a
zip...**, then start the game again. Settings > GPU driver also offers to download community Turnip builds. A driver is never part of
this repository or of the APK.

## Settings

The gear on the main screen holds the app's settings; a game's ⋯ menu > **Game settings** holds that game's own.

| Setting | What it does |
|---|---|
| **RAM saver (uses storage)** (Game settings) | Off by default; **turn it on if your phone has less than 16 GB of RAM**. The game's large textures live in a file in the app's cache folder instead of in RAM, so Android can drop and reload them instead of closing the game. For phones short of RAM with free storage; it writes about 2 GB to storage per area load in Ocarina of Time |
| Frame-rate cap | the game's own, 60, 30, 20 or 15 fps |
| Game resolution | the phone's 1080p or the game's own |
| GPU driver | the Vulkan driver the game runs on (the system's, or an installed Turnip zip) |
| Frame generation | Lossless Scaling's frame generation (2x, 3x, 4x or a target frame rate); needs your own `Lossless.dll` |
| Performance overlay | fps, GPU time, throttling and battery over the game |
| Readiness check | what the phone is missing for a game, with a button to fix most items |

On-screen controls have a layout editor; gamepads work. Every switch the host reads is listed in
[docs/env_switches.tsv](docs/env_switches.tsv); a game's tested settings are in its profile.

## Known issues

- **Ocarina of Time is slow**: about 12-16 fps at the start, lower in open areas. The forest's terrain shaders are the main cost.
- **Memory on 12 GB phones**: Android may close the game after a long session or in a heavy area; it closes background apps while the
  game runs. Turn the RAM saver on (recommended below 16 GB of RAM). Back up your saves often.
- **Textures** may pop in while walking (textures released under memory pressure are made again) and **shimmer**; some foliage is
  **black** (mesh shaders).
- **Short stalls while walking** come from pipelines built late and from memory pressure.
- **Anti-aliasing** is not visible yet: the game's SMAA pass runs, but its blend weights are all zero.
- Donkey Kong Bananza and Mario Kart World were **not tested** with Anka Native's changes. Only one phone was tried.
- The phone gets hot, and the frame rate drops when it throttles.
- This release's app is a per-game test build: its icon has a small "T" badge, and it reads extra switches from `env.txt` in its files
  folder.

## Building

How to set up the environment, prepare a game and build the APK: [android/README.md](android/README.md) (and
[android/TESTING.md](android/TESTING.md) for testing on a phone). This release's APK was built with:

```sh
cd android/app
./gradlew --no-daemon -q -PnxGames=0400486024DEC000 -PnxApp=oot -PnxPackEngine=fake apk   # -> out/anka-native-oot.apk
```

`-PnxApp=<suffix>` sets the app id `dev.ankanative.android.<suffix>` and `-PnxLabel` its name; `-PnxPackEngine=fake` leaves out the
on-phone packer. Signed releases are built with `android/tools/release.sh` ([docs/RELEASING.md](docs/RELEASING.md)). The icons are made by `android/tools/anka_icons.py`. The Mac and iPhone builds this repository also holds are
described in [docs/mac-and-iphone.md](docs/mac-and-iphone.md).

## AI disclosure

Most of Anka Native's code, its commit messages and these documents were written by an AI coding assistant (Claude Opus 5.5, by
Anthropic) working under the direction of the human maintainer, who decided what to try, played and tested the builds on his phone, and
is responsible for the release. Many of the measurements were taken by automated runs on that phone. It is experimental and has bugs.

## License

Most of the code is MIT ([LICENSE](LICENSE)); the frame generation port (`dm/gfx/vk_android/lsfg`), one shader translator file
(`InstEmitAmpereMesh.cs`) and the Anka mark's script (`android/tools/anka_logo.py`) are GPL-3.0-or-later. **The Android app is GPL-3.0-or-later as a whole**, because it links GPL-3.0-or-later parts (frame
generation, and the shader translator in an APK that carries the on-phone packer); its source is this repository at the commit in the
app's build tag. Credits, third-party code and their licenses: [NOTICE.md](NOTICE.md). The Anka mark and its icons are EddizEge's own
design, GPL-3.0-or-later. The app's About section names the license and points to NOTICE.md.

Use only games you own. Do not ask for games, keys or dumps.
