# The Mac and iPhone builds

This repository also holds the Mac, iPhone and iPad builds of the host it started from (MIT). They are not part of Anka Native and were not changed or tested for it; their original README follows, unchanged except for its links and without its screenshot gallery (game screenshots are not kept in this repository). Credits: [NOTICE.md](../NOTICE.md).

## nxnative

Play your own Nintendo Switch games natively on iPhone and iPad. Bring your game (its NSP files) and your `prod.keys`; one script turns them into an IPA that you sideload.

It is not an emulator: the game's own code runs directly on the Apple chip, with Switch graphics rebuilt on Metal. Nothing from any game is in this repository, and the project is not affiliated with Nintendo or any game publisher.

Switch 2 games have started too: [Donkey Kong Bananza](../dm/games/040074A01BF12000/README.md) plays on the Mac, packed from a dump of the game; iPhone and iPad are untested.

## Acknowledgements

This project is built on other people's work ([licenses](../dm/third_party/README.md)):

- **[d2r-mobile](https://github.com/jhaizhou-ops/d2r-mobile)** (MIT) developed this native-execution approach, for Diablo II: Resurrected. nxnative started as a copy of its generic parts: the NSO rewriter, the host that stands in for Horizon, the NVN-on-Metal layer and the iOS app shell. Fixes go both ways.
- **Ryujinx** (MIT): its shader translator (`Ryujinx.Graphics.Shader`, `Ryujinx.Common` and `Spv.Generator` 1.3.2, in `dm/third_party/ryujinx` with a few fixes listed in [its README](../dm/third_party/ryujinx/README.md)) turns the games' shaders into SPIR-V; its Switch 2 task / mesh support is after [switch2mac](https://github.com/azmasm/switch2mac) (azmasm), with one GPL-3.0-or-later file from [Eglantine](https://github.com/djihads80/eglantine) (Eden Emulator Project), which makes the built translator GPL-3.0-or-later as a whole ([its README](../dm/third_party/ryujinx/README.md)). Its emulation of the kernel and system services is also a reference when the host's behaviour is in doubt.
- **[Atmosphère](https://github.com/Atmosphere-NX/Atmosphere)**: its kernel, mesosphère, is the reference for Horizon's synchronization semantics (address arbiter, condition variables).
- **[LibHac](https://github.com/Thealexbarney/LibHac)**: reads NSPs and layers updates over the base game in `dm/tools/nx_extract`.
- **[SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross)**: SPIR-V to Metal Shading Language.
- **[glslang](https://github.com/KhronosGroup/glslang)** compiles the GLSL of OpenGL games to SPIR-V for it.
- **[MacNativeGames](https://github.com/ly05010419/MacNativeGames)** (MIT): the GPU widening of 8-bit indices and in-pass clears in `dm/gfx/nvn/nvn_mtl.m`, and the visibility-counter lowering in the shader translator.

## How to play

Switch 2 games are built from a folder of their files, not from NSPs, and are tested on the Mac only: see [Switch 2 games](#switch-2-games).

### With a Mac app

Community projects build on nxnative with Mac apps that turn your games into apps for your iPhone, iPad or Mac, with no command line, and keep them in a library. They are separate projects: questions and bugs about them go to their own pages.

- [NX2Apple](https://github.com/LNPANDY/NX2Apple)
- [NX Play](https://github.com/varyform/nx-native-wrapper)
- [NxNative Launcher](https://github.com/rennan101/nxnative_Launcher)
- [LumaNX](https://github.com/Xc1Ym/LumaNX)

### Build from the command line

You need a Mac with macOS 26.6 or later, [Xcode](https://developer.apple.com/download/all/?q=Xcode%2027.2) 27.2, [Homebrew](https://brew.sh) and Python 3.12 or newer (an arm64 one, not an Intel Homebrew's). An iPhone build also needs Xcode's iOS platform (Xcode → Settings → Components).

The shaders are built with Xcode 27.2's Metal compiler. Select it before step 1: `sudo xcode-select -s /Applications/Xcode.app`. With several Xcode versions installed, name this one `Xcode-27.2.app` and select it with `sudo xcode-select -s /Applications/Xcode-27.2.app`.

**1. Install the tools** (only once), in Terminal:

```
xcodebuild -downloadComponent MetalToolchain
git clone https://github.com/seedds/nxnative.git && cd nxnative
brew install xcodegen zstd glslang
brew install --cask dotnet-sdk
python3 -m pip install lz4 "capstone>=5,<6" fonttools nsz zbic "backports.zstd; python_version < '3.14'"
```

**2. Put your game in a folder**: the game, plus its update and DLC if you have them (`.nsp`, `.xci`, `.nsz` or `.xcz`):

```
~/Games/MarioWonder/
  Super Mario Bros. Wonder [010015100B514000][v0].nsp
  Super Mario Bros. Wonder [010015100B514800][v327680].nsp
```

**3. Build**, with your game folder and your `prod.keys`:

```
python3 scripts/build_ipa_from_nsp.py ~/Games/MarioWonder -k ~/keys/prod.keys
```

The IPA appears in the game's folder. The first build takes 30–60 minutes and needs free disk space of about twice the game's size plus 10 GB; building again is much faster.

To play on the Mac instead, add `--platform mac-app`: you get a Mac app of the game to open from Finder. All options: [The packing script in detail](../docs/development.md#the-packing-script-in-detail).

### Install

Install the IPA on your iPhone or iPad with a sideloading tool such as AltStore, Sideloadly or LiveContainer. If your tool offers **Increased Memory Limit**, turn it on: the bigger games need it.

Game too big to install? Build again with `--romfs separate`, and run the command the script prints once to copy the game files to your phone ([Quick reinstalls](../docs/development.md#quick-reinstalls)).

Controls, saves and settings: [Playing](../docs/playing.md).

### Switch 2 games

A Switch 2 game is built from a folder of its files, not from NSPs, and needs no `prod.keys`. The script expects this folder (`updates` and `dlc` are optional):

```
~/Games/DonkeyKongBananza/
  base/exefs/                     main, main.npdm, sdk, rtld
  base/romfs/
  updates/458752/exefs/           the update's version, as a decimal number
  updates/458752/romfs/           the patched RomFS, not a patch
  dlc/040074a01bf13001/0/romfs/   the DLC's title ID
```

```
python3 scripts/build_ipa_from_nsp.py ~/Games/DonkeyKongBananza --platform mac-app
```

You get a Mac app, with no icon because the folder holds none. iPhone and iPad are untested.

## Games

Games tested on Mac or iPhone. See the [per-game notes](../dm/games/README.md#games) for tested platforms, progress and known issues:

Needs iOS / iPadOS 26 or later on an iPhone 15 Pro or newer (A17 Pro chip or later) or an iPad with an M1 chip or later: the games' textures are BC-compressed, and older iPhones can't read that format. Other games aren't supported yet. For them the script writes a survey and still tries its best to pack an IPA, guessing the game's settings; such an IPA may not start.


### For developers

[How it works](../docs/development.md) · [Game notes](../dm/games/README.md#games) · [Bring-up and debugging](../docs/agents/README.md) · [Front ends](../docs/frontends.md) · [Contributing](../CONTRIBUTING.md)

### License

MIT ([LICENSE](../LICENSE)). Third-party code keeps its own license ([dm/third_party](../dm/third_party/README.md)); two parts are GPL-3.0-or-later, and so is what is built with them: the shader translator ([dm/third_party/ryujinx](../dm/third_party/ryujinx/README.md)) and Android's frame generation ([android/NOTICE.md](../android/NOTICE.md)).
