# The Legend of Zelda: Ocarina of Time (0400486024DEC000): the rewriter's patches of this game's own builds. dm/tools/patch_nso.py loads every profile's patches.py and applies those of a
# module's NSO build ID; CUTS and FRAME_STEP as described there ("Registered patches"). Patches of Nintendo SDK module builds are in dm/tools/sdk_patches.py
import os
from nso_asm import ADRP, B

# A diagnostic, off by default: DM_OOT_MAX_RES=1 in the rewriter's environment (patch_nso.py, or scripts/pack_android.py, which passes its environment on) registers _max_res_cuts
# below as CUTS; unset, empty or 0 registers no build at all (CUTS = {}, so the build is unregistered and keeps the profile's and the SDK's function-matched cuts, as without this
# file). Any other value raises here. This is the only patches.py that reads the environment, and patch_nso.py and docs_patches.py load every game's patches.py, so a bad value stops
# the rewrite and the docs of every game, not only this one's: fail closed, a typo never yields a silent plain rewrite. docs/agents/game-patches.md is generated without the
# switch (with it, docs_patches.py --check reports the file out of date); the cut is named there through BINDING's 'kept' line, whatever the switch.
# pack_android.py's rewrite key (<module>.android.key) covers this file but not the environment: rewrite with the switch into a pack folder of its own (--out), never over a pack made
# without it, or the module is kept as it was ("unchanged, kept").
# Fail closed: patch_nso.py asserts the original bytes before applying a cut and stops the rewrite on a mismatch ("cut original bytes mismatch 0x805bd0: ..."); the module is rewritten
# into a staging folder and swapped in only when complete, so a mismatch changes nothing.
_max_res = os.environ.get('DM_OOT_MAX_RES', '')
if _max_res not in ('', '0', '1'): raise ValueError(f'DM_OOT_MAX_RES={_max_res!r}: 1 turns the max-resolution diagnostic on, 0 or unset leaves it off')

_max_res_cuts = {
  # v131072 main (build ID e0f3ccfaf50065ae4ea01eb5aeb13b1a45da3348). main+0x805aa0 is the dynamic-resolution controller's per-frame update; it returns the render scale
  # (a float in w0) and a state, 1 or 2, in the high word. At +0x805bcc it reads the controller's atomic flag word (controller+0x188, through x8 = x0+0xf4, +0x94);
  # bit 0 is "dynamic resolution on", set and cleared at main+0x31d3c98 (from bit 11 of the system object's +0xe20 word, under its 0x440 mask) and cleared at main+0xdbb1e0.
  # With bit 0 clear the tbz goes to main+0x80640c, which returns scale 1.0 (0x3f800000) and skips the GPU-time history, the floor and the step logic (its state is 2 only when
  # [x19+0x178] is 0xfffffffe or 0xffffffff, and +0x80601c stores a byte there (from +0x174), so this path returns 1 every frame: what that state means to the caller is not known); with it set the
  # function follows GPU time and settles at its floor (683x384 for 2048x1152 on the phone). The cut makes the branch unconditional: scale 1.0 whatever the flag. Bits 1-3 of
  # the word (read at +0x805d34, +0x805ee4, +0x805fe4) and the function's other paths are unchanged. Offset and bytes as the Lichii-Android emulator's built-in patch for this
  # build ("Max Resolution"); disassembled here, 2026-11-07. A diagnostic of the black gameplay (is it the dynamic-resolution floor or the upscaler?), not a fix.
  'e0f3ccfaf50065ae4ea01eb5aeb13b1a': [
    (0x805bd0, 'e8410036', '0f020014', 'dynamic resolution update: tbz w8,#0,+0x83c (flag bit 0 clear: scale 1.0) -> b +0x83c (DM_OOT_MAX_RES=1 diagnostic: always scale 1.0)'),
  ],
}
CUTS = _max_res_cuts if _max_res == '1' else {}

# The game's frame ratio (ported from seedds/nxnative 7d8dcdf8, MIT; nxnative credits Lichii for finding the division). Once a frame the function at main+0x6e6be0 (v131072)
# divides its base rate (60.0, or 30.0 while bit 0 of [x21+0x24f] is set: fcsel at +0xe0) by the int frame rate at [x21+0x2c8] (scvtf s0 at +0xc8): fdiv s8,s1,s0 at
# main+0x6e6cc4 (+0xe4). The game steps its time by that ratio whatever the frame took, so a frame of two vsyncs at rate 60 plays at half speed: game speed = fps / 60 on the
# phone (session-2026-11-07 play/PLAY.md section 3). The stub gives the ratio the host's step instead: step × base / 60 (the division's value when the host keeps frames 60 / rate
# vsyncs) and counts the vsyncs stepped (+0x30). Until the host has decided a step (+0x20 = 0), and at fps_target 0 (+0 = 0), the original division runs.
# Free there (disassembled on our main.img, 2026-11-08): x16 / x17 are used nowhere in the function; s0 is written again on every path before it is read (ldr d0 at +0x124,
# fmov s0,s8 at +0x1ec); s1 (the base rate, read again at +0x224) and w8 / w9 (tbnz / tbz right after) are kept; the stub sets no flags
def frame_step(ctx):
    here, emit, fs_site, fs_put, FS_STATE_OFF = ctx.here, ctx.emit, ctx.site, ctx.put, ctx.state
    site = 0x6e6cc4
    fs_site(site, '2818201e')   # fdiv s8,s1,s0
    start = here(); emit(ADRP(16, here(), FS_STATE_OFF))
    for w in (0xb9400211,   # ldr w17,[x16]          target fps (float bits; 0 = off)
              0x340001b1,   # cbz w17,orig (+13)
              0xb9402211,   # ldr w17,[x16,#0x20]    step in vsyncs
              0x34000171,   # cbz w17,orig (+11)     not decided yet
              0x1e230220,   # ucvtf s0,w17
              0xf9401a10,   # ldr x16,[x16,#0x30]
              0x8b110210):  # add x16,x16,x17
        emit(w)
    emit(ADRP(17, here(), FS_STATE_OFF))
    for w in (0xf9001a30,   # str x16,[x17,#0x30]    vsyncs stepped
              0x1e210808,   # fmul s8,s0,s1          step × base rate
              0x52a84e10,   # mov w16,#0x42700000    60.0
              0x1e270200,   # fmov s0,w16
              0x1e201908):  # fdiv s8,s8,s0          / 60
        emit(w)
    emit(B(here(), site + 4))
    emit(0x1e201828); emit(B(here(), site + 4))   # orig: fdiv s8,s1,s0; back
    fs_put(site, [B(site, start)], "frame ratio: the host's step × base rate / 60 instead of base rate / the game's frame rate (state page target fps; 0 = the game's ratio)")

FRAME_STEP = {'e0f3ccfaf50065ae4ea01eb5aeb13b1a': frame_step}

# How general each patch is and could be (dm/tools/docs_patches.py "Binding", listed in docs/agents/game-patches.md): what, its binding now, the loosest it could have (goal),
# purpose, why; blocked or kept: why the gap stays open (cannot close yet, or left open on purpose); checked: what was checked on game files (else a judgement)
BINDING = [
    {'what': "frame ratio from the host's frame step", 'now': 'build', 'goal': 'game', 'purpose': 'frame rate',
     'blocked': 'frame-step stubs have no function-matched form here yet (nxnative has MATCH_FRAME_STEP)',
     'checked': 'v131072 main: the word 2818201e (fdiv s8,s1,s0) at +0x6e6cc4 in the function at +0x6e6be0, and the free registers, read by disassembly (2026-11-08)',
     'why': "the game's own frame-ratio division; seedds/nxnative 7d8dcdf8 (MIT) binds it by the function's fingerprint"},
    {'what': 'dynamic-resolution update held at scale 1.0 (diagnostic, off unless DM_OOT_MAX_RES=1)', 'now': 'build', 'goal': 'game', 'purpose': 'rendering',
     'kept': "a diagnostic, not a fix: main+0x805bd0 e8410036 -> 0f020014 (_max_res_cuts), registered only with DM_OOT_MAX_RES=1, so the default rewrite has no cut",
     'checked': 'v131072 main: build ID and the original word e8410036 at +0x805bd0, the function main+0x805aa0 and the writers of its flag bit (+0x31d3c98, +0xdbb1e0) disassembled; '
                'rewrites with the switch on and off differ only in that word, 2026-11-07',
     'why': "the game's own controller (agl / sead dynamic resolution); a function-matched rule could carry it over the game's updates if it is ever more than a diagnostic"},
]
