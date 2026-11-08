#!/bin/sh
# Live control of a running game on the phone (android/TESTING.md "Live control"): appends one control line to the app's ctl.txt, which the host reads every 100 ms
# (DM_CTL, hle_hid.c; the replay's commands in dm/gfx/vk_android/vka_ctl.c), then prints the log lines that answer it.
#   nxctl.sh env DM_VK_DRAW=0       a switch, applied at the next frame boundary ("env KEY=" clears it); several KEY=VALUE arguments go as one env line each
#   nxctl.sh capture 30             one log line per frame for the next 30 frames
#   nxctl.sh screenshot [file.png]  the next game frame as a PNG; pulled to file.png (default ./ctl-shot-<frame>.png)
#   nxctl.sh texshot <serial> [file.png]  one texture (a serial from the log) at the next present, as a PNG; pulled like a screenshot
#   nxctl.sh bufshot <dispatch index> <slot> [device path]  the compute storage slot as that dispatch of a frame left it (DM_VK_DISPATCH_LOG indices), raw
#   nxctl.sh stats                  averages of the last 60 frames, the switches set live
#   nxctl.sh skip on|off|toggle     the intro skip (DM_SKIP: A pressed, movies ended early, off by itself at gameplay)
#   nxctl.sh dump                   any other DM_CTL line as it is (key presses "A~0.2", "dump", ...)
#   nxctl.sh log [n]                the last n (40) lines of the run's log, nothing sent
# NX_SERIAL (default: adb's, ANDROID_SERIAL or the only phone) picks the phone, NX_PKG (default dev.ankanative.android.test; the OoT app: dev.ankanative.android.oot) the app, NX_WAIT (default 3) the seconds to wait for the answer.
set -eu
PKG=${NX_PKG:-dev.ankanative.android.test}
WAIT=${NX_WAIT:-3}
FILES=/sdcard/Android/data/$PKG/files
A="adb${NX_SERIAL:+ -s $NX_SERIAL}"
[ $# -ge 1 ] || { sed -n '2,13p' "$0"; exit 2; }

newest_log() { $A shell "ls -t $FILES/logs/*.log 2>/dev/null | head -1" | tr -d '\r'; }
LOG=$(newest_log)
if [ "$1" = log ]; then [ -n "$LOG" ] || { echo "no log in $FILES/logs" >&2; exit 1; }; $A shell "tail -n ${2:-40} '$LOG'"; exit 0; fi
$A shell pidof "$PKG:game" >/dev/null 2>&1 || { echo "the game ($PKG:game) is not running" >&2; exit 1; }

shot_local=
if [ "$1" = screenshot ] && [ $# -ge 2 ]; then shot_local=$2; set -- screenshot; fi
if [ "$1" = texshot ] && [ $# -ge 3 ]; then shot_local=$3; set -- texshot "$2"; fi
line=$*
case "$line" in *"'"*) echo "a control line cannot hold a single quote" >&2; exit 2;; esac
# env: one control line a KEY=VALUE (the replay takes one switch a line and rejects a value with white space: "env A=1 B=2" set A to "1 B=2" before)
lines=$line; nenv=0
if [ "$1" = env ]; then
    shift; [ $# -ge 1 ] || { echo "env wants KEY=VALUE" >&2; exit 2; }
    lines=; for kv in "$@"; do case "$kv" in *=*) ;; *) echo "env: '$kv' is not KEY=VALUE" >&2; exit 2;; esac; lines="$lines${lines:+\\n}env $kv"; nenv=$((nenv + 1)); done
    set -- env "$@"
fi
n0=$($A shell "wc -l < '$LOG'" | tr -d '\r ')
$A shell "f=$FILES/ctl.txt; printf '%b\\n' '$lines' >> \$f; chmod 666 \$f 2>/dev/null; true"

# The answer: the log lines after the ones there before, from the replay's control ("[vk] ctl:") and the host's reader ("DM_CTL"), and any [RED] line
want=1; case "$1" in capture) want=$(( ${2:-60} + 2 ));; env) want=$nenv;; esac
t=0; got=0
while [ "$t" -lt $((WAIT * 4 + want / 6)) ]; do
    sleep 0.25; t=$((t + 1))
    out=$($A shell "tail -n +$((n0 + 1)) '$LOG'" | tr -d '\r' | grep -E '\[vk\] ctl:|DM_CTL|DM_SKIP|\[RED\]|new swapchain' || true)
    got=$(printf '%s' "$out" | grep -c 'ctl: \(env .* applied\|env .* rejected\|capture done\|stats at\|screenshot of frame\|screenshot:\|bufshot of program\|bufshot:\)\|DM_CTL: press\|DM_CTL: env\|DM_SKIP: \(on\|off\|already\)' || true)
    [ "$got" -ge "$want" ] && { [ "$1" != capture ] || printf '%s' "$out" | grep -q 'capture done'; } && break
done
printf '%s\n' "${out:-(no answer in the log yet: nxctl.sh log)}"
if [ "$1" = screenshot ] || [ "$1" = texshot ]; then
    remote=$(printf '%s' "$out" | sed -n 's/.*screenshot of frame [0-9]*, [0-9x]* → //p' | tail -1)
    [ -n "$remote" ] || { echo "no screenshot written" >&2; exit 1; }
    $A pull "$remote" "${shot_local:-$(basename "$remote")}" >/dev/null && echo "pulled ${shot_local:-$(basename "$remote")}"
fi
