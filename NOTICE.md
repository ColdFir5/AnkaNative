# Anka Native: credits and licenses

Anka Native is built on other people's work. This file is where they are credited; their copyrights and licenses are kept.

## The projects it is built on

- **[nxnative](https://github.com/seedds/nxnative)** by seedds (MIT). The host that runs a Switch game's own ARM64 code natively, the NSO rewriter, the
  game profiles, the shader tools and the Mac / iPhone builds all come from it. Its MIT license and copyright line are kept in [LICENSE](LICENSE). Later
  upstream fixes ported into Anka Native name the nxnative commit they come from in the code and in the docs (for example the Ocarina of Time frame step,
  `7d8dcdf8`, and the 8-byte record step of `shader_split.py`, `b1d7c707`).
- **[Banana Native](https://github.com/moceure/BananaNative)** by moceure (MIT, with the GPL parts named below), including the work in the Raphix91 fork:
  the Android port of nxnative that Anka Native continues: the Android launcher (`android/app`), the Vulkan replay of NVN (`dm/gfx/vk_android`), the
  on-phone packer, frame generation on Android, the custom GPU driver loader and the Donkey Kong Bananza and Mario Kart World work on Android. Its own
  notice is kept in [android/NOTICE.md](android/NOTICE.md).
- **[d2r-mobile](https://github.com/jhaizhou-ops/d2r-mobile)** (MIT): the native-execution approach nxnative started from.

## Code and findings from other projects

- **Ryujinx** (MIT): the shader translator (`Ryujinx.Graphics.Shader`, `Ryujinx.Common`, `Spv.Generator` 1.3.2) in `dm/third_party/ryujinx`, with local
  changes listed in its README.
- **[Eglantine](https://github.com/djihads80/eglantine)** via **[switch2mac](https://github.com/azmasm/switch2mac)** by azmasm: the Switch 2 task / mesh
  shader lowering (`InstEmitAmpereMesh.cs`, GPL-3.0-or-later); switch2mac's SPIR-V mesh entry points (MIT outside `nx2_lower`).
- **[kartnative](https://github.com/pahorema/kartnative)** by pahorema (MIT): findings for GPU-written sparse textures and the mesh programs'
  per-primitive header (followed without copying code).
- **[Eden](https://git.eden-emu.dev/eden-emu/eden)** and **[lsfg-vk](https://github.com/PancakeTAS/lsfg-vk)**: Lossless Scaling frame generation, ported
  to C in `dm/gfx/vk_android/lsfg` (GPL-3.0-or-later). `Lossless.dll` itself belongs to Lossless Scaling, is sold separately and is never included.
- **[libadrenotools](https://github.com/bylaws/libadrenotools)** (with liblinkernsbypass), as kept in Eden: loading a custom Adreno GPU driver.
- **Mesa's Turnip** Vulkan driver (MIT): the open Adreno driver the phone runs the games on. It is not in this repository or in the APK; the user installs a
  driver build, such as the Anka Turnip build or the community builds the launcher's Download... lists (each keeps its own license; Qualcomm's own driver
  binaries are proprietary).
- **Lichii** (Android emulator): Ocarina of Time findings. The game's 4-byte cut at `main+0x805bd0` uses the same offset and bytes as Lichii's built-in
  patch, and the frame-rate division the frame step replaces was found there first (as nxnative credits). Ideas only: no Lichii code is in this repository.
- **[Atmosphere](https://github.com/Atmosphere-NX/Atmosphere)** (the reference for Horizon's synchronization), **[LibHac](https://github.com/Thealexbarney/LibHac)**,
  **[SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross)**, **[glslang](https://github.com/KhronosGroup/glslang)**,
  **[MacNativeGames](https://github.com/ly05010419/MacNativeGames)** (MIT) and deko3d (zlib): used by the Mac and iPhone builds and the shared tools, as
  listed in [docs/mac-and-iphone.md](docs/mac-and-iphone.md#acknowledgements).
- Third-party code shipped in `dm/third_party` (AMD FidelityFX FSR 1, NVIDIA FLIP, libadrenotools, libvpx, stb_vorbis, Ryujinx, Vulkan-Headers, zstd and
  MacNativeGames' adapted parts): each keeps its license file in its own directory, listed in [dm/third_party/README.md](dm/third_party/README.md).
- Prebuilt tools committed for the Mac shader chain: `dm/gfx/tools/spirv_cross_bin/spirv-cross` (SPIRV-Cross, Apache-2.0, license texts beside it) and
  the translator `dm/gfx/tools/dmsh_xlate_bin` (a GPL-3.0-or-later combined work, see above). The translator folder also carries the .NET packages it
  was built with, unmodified from NuGet: Gommon 2.8.1.2 (MIT), Humanizer 3.0.10 with its language resources (MIT), MsgPack.Cli 1.0.1 (Apache-2.0),
  Microsoft.IO.RecyclableMemoryStream 3.0.1 (MIT) and System.CodeDom 4.4.0 (MIT, .NET Foundation); each package's license is in its NuGet package.
  Beside it, `dmsh_xlate-<linux|win>-<x64|arm64>` are .NET's application hosts (MIT) that start the same translator on Linux and Windows.

- The **on-phone packer** in the APK (`android/ondevice/build.sh` builds it, unmodified from each project's release): **[CPython](https://www.python.org)**
  3.14.6 for Android (PSF License), **[Capstone](https://github.com/capstone-engine/capstone)** 5.0.7 (BSD-3-Clause),
  **[python-lz4](https://github.com/python-lz4/python-lz4)** 4.4.5 with LZ4 (BSD-3-Clause, BSD-2-Clause),
  **[zbic](https://github.com/kinnay/zbic)** 1.0.1.post1 by Yannik Marchand (GPLv2; it carries Zstandard, BSD-3-Clause or GPLv2) for the
  zstd-compressed code of newer games, **[SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools)** (Apache-2.0) and the
  **[.NET](https://github.com/dotnet/runtime)** runtime the translator is compiled with (NativeAOT, MIT).

## Licenses

- The repository's own code is MIT ([LICENSE](LICENSE): seedds' copyright and Anka Native's), except the files marked GPL-3.0-or-later: the frame
  generation port (`dm/gfx/vk_android/lsfg`), the shader translator's `InstEmitAmpereMesh.cs` and the Anka mark's script (`android/tools/anka_logo.py`).
- **The Android app is GPL-3.0-or-later as a whole.** The APK links the frame generation port (`dm/gfx/vk_android/lsfg`, GPL-3.0-or-later) unless built
  with `-PnxLsfg=false`, and an APK that carries the on-phone packer also carries the GPL shader translator. Whoever passes such an APK on must offer its
  complete source: this repository at the commit in the app's build tag. Details: [android/NOTICE.md](android/NOTICE.md).
- The Anka mark (the phoenix) is EddizEge's own design, the same mark as the Anka emulator's. Its script `android/tools/anka_logo.py` is
  Copyright 2026 EddizEge, GPL-3.0-or-later, as in the Anka project (the license text: [dm/gfx/vk_android/lsfg/COPYING](dm/gfx/vk_android/lsfg/COPYING)).
  The icons it makes (`android/images/logo.png` and the `anka_*.png` and `ic_launcher*.png` files under `android/app/src/main/res`) are EddizEge's
  artwork and are licensed GPL-3.0-or-later too. `android/tools/anka_icons.py`, which writes them from the script, is MIT.
- The game profiles (`dm/games/<title ID>`) kept in this repository are Ocarina of Time (`0400486024DEC000`), Donkey Kong Bananza (`040074A01BF12000`) and
  Mario Kart World (`0400C3F00006E000`); nxnative's profiles of other games are not carried over.
- Nothing from any game is in this repository or in the app: no game code, no game data, no keys, no firmware and no shader libraries made from game
  shaders. Game profiles hold only identifiers, hashes, offsets and short instruction words used to check a patch's place.

Not affiliated with Nintendo, any game publisher, or any of the projects above.
