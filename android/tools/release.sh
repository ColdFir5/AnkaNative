#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Builds, signs and checks an Anka Native release (docs/RELEASING.md). The release key never lives in this repository: Gradle reads its keystore path,
# alias and passwords from the environment or from ~/.gradle/gradle.properties (ANKA_KEYSTORE, ANKA_KEY_ALIAS, ANKA_KEYSTORE_PASSWORD, ANKA_KEY_PASSWORD;
# android/app/build.gradle). This script never creates a key and never asks for, stores or prints a password.
#
#   android/tools/release.sh --fingerprint   prints the release certificate's SHA-256 (for Signer.OFFICIAL), from the keystore; keytool asks for the
#                                            store password unless ANKA_KEYSTORE_PASSWORD is set
#   android/tools/release.sh                 builds the release APK (-PnxLive=true: dev.ankanative.android, "Anka Native"), signed with the release key,
#                                            verifies it with apksigner, prints the certificate's SHA-256, and writes the APK and SHA256SUMS.txt
#                                            to the release folder outside the repository
#
# Settings (environment):
#   ANKA_GAMES          the games the APK runs (-PnxGames), default 0400486024DEC000 (Ocarina of Time)
#   ANKA_VERSION        the version name (-PnxVersion), default 0.1.0; ANKA_VERSION_CODE the version code (-PnxVersionCode), default 1
#   ANKA_PACK_ENGINE    native (default: the on-phone packer, from the stage NX_PACK_STAGE names) or fake (no packer: the app offers no Prepare)
#   ANKA_RELEASE_DIR    where the APK and SHA256SUMS.txt go, default ~/anka-native-releases/<version>; must be outside this repository
#   JAVA_HOME           a JDK 17-21 for Gradle (default /usr/lib/jvm/java-21-openjdk when it exists)
#   ANDROID_HOME        the Android SDK (default: sdk.dir of android/app/local.properties) for apksigner
#   ANKA_ALLOW_DIRTY=1  build from a tree with uncommitted changes (a release's source is the commit in its build tag, so the default refuses)
#   ANKA_ALLOW_UNOFFICIAL=1  build while Signer.OFFICIAL is not set (the app then calls itself "unofficial build")
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
die() { echo "release.sh: $*" >&2; exit 1; }

# A setting from the environment, else from ~/.gradle/gradle.properties (as build.gradle reads it)
setting() {
    local v=${!1:-}
    if [ -z "$v" ]; then
        local f="${GRADLE_USER_HOME:-$HOME/.gradle}/gradle.properties"
        [ -f "$f" ] && v=$(sed -n "s/^[[:space:]]*$1[[:space:]]*[=:][[:space:]]*//p" "$f" | tail -n 1)
    fi
    printf '%s' "$v"
}

inside_repo() { case "$(realpath -m "$1")/" in "$REPO"/*) return 0;; *) return 1;; esac; }

KEYSTORE=$(setting ANKA_KEYSTORE); ALIAS=$(setting ANKA_KEY_ALIAS)
[ -n "$KEYSTORE" ] && [ -n "$ALIAS" ] || die "set ANKA_KEYSTORE and ANKA_KEY_ALIAS (environment or ~/.gradle/gradle.properties; docs/RELEASING.md)"
[ -f "$KEYSTORE" ] || die "ANKA_KEYSTORE is not a file: $KEYSTORE"
inside_repo "$KEYSTORE" && die "the keystore must live outside this repository: $KEYSTORE"

if [ "${1:-}" = "--fingerprint" ]; then
    command -v keytool >/dev/null || die "keytool not found (a JDK's bin)"
    pass=()
    [ -n "${ANKA_KEYSTORE_PASSWORD:-}" ] && pass=(-storepass:env ANKA_KEYSTORE_PASSWORD)
    # The certificate's DER bytes (keytool -exportcert without -rfc), hashed as apksigner hashes them
    fp=$(keytool -exportcert -keystore "$KEYSTORE" -alias "$ALIAS" "${pass[@]}" | sha256sum | cut -c1-64)
    [[ $fp =~ ^[0-9a-f]{64}$ ]] && [ "$fp" != "$(printf '' | sha256sum | cut -c1-64)" ] || die "could not read the certificate of alias $ALIAS"
    echo "$fp"
    exit 0
fi
[ $# -eq 0 ] || die "usage: release.sh [--fingerprint]"

GAMES=${ANKA_GAMES:-0400486024DEC000}
VERSION=${ANKA_VERSION:-0.2.0}; CODE=${ANKA_VERSION_CODE:-3}
ENGINE=${ANKA_PACK_ENGINE:-native}
if [ "$ENGINE" = native ]; then
    [ -n "${NX_PACK_STAGE:-}" ] && [ -f "$NX_PACK_STAGE/lib/libnxpack.so" ] || die "set NX_PACK_STAGE to the on-phone packer's stage (android/ondevice/build.sh <work dir> makes <work dir>/stage), or ANKA_PACK_ENGINE=fake"
fi
OUT=${ANKA_RELEASE_DIR:-$HOME/anka-native-releases/$VERSION}
inside_repo "$OUT" && die "the release folder must be outside this repository: $OUT"
[ -n "$(setting ANKA_KEYSTORE_PASSWORD)" ] || die "set ANKA_KEYSTORE_PASSWORD (environment or ~/.gradle/gradle.properties) for Gradle's signing"

if [ -z "${JAVA_HOME:-}" ] && [ -d /usr/lib/jvm/java-21-openjdk ]; then export JAVA_HOME=/usr/lib/jvm/java-21-openjdk; fi
SDK=${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}
[ -n "$SDK" ] || SDK=$(sed -n 's/^sdk\.dir=//p' "$REPO/android/app/local.properties" 2>/dev/null || true)
[ -d "$SDK/build-tools" ] || die "no Android SDK build-tools (set ANDROID_HOME)"
APKSIGNER=$(ls -d "$SDK"/build-tools/*/apksigner 2>/dev/null | sort -V | tail -n 1)
[ -x "$APKSIGNER" ] || die "apksigner not found under $SDK/build-tools"

# The source of a release is the commit in its build tag: refuse uncommitted changes (tracked files)
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] && [ "${ANKA_ALLOW_DIRTY:-}" != 1 ]; then
    die "the tree has uncommitted changes: commit them first (ANKA_ALLOW_DIRTY=1 overrides)"
fi

# Signer.OFFICIAL: the certificate the app calls official (android/app/src/main/java/dev/nxnative/android/Signer.java)
SIGNER_JAVA="$REPO/android/app/src/main/java/dev/nxnative/android/Signer.java"
OFFICIAL=$(sed -n 's/^    static final String OFFICIAL = "\([0-9a-f]\{64\}\)";.*/\1/p' "$SIGNER_JAVA")
if [ -z "$OFFICIAL" ] && [ "${ANKA_ALLOW_UNOFFICIAL:-}" != 1 ]; then
    die "Signer.OFFICIAL is not set: run 'android/tools/release.sh --fingerprint' and fill it in first (docs/RELEASING.md); ANKA_ALLOW_UNOFFICIAL=1 overrides"
fi

echo "Building Anka Native $VERSION ($CODE) for $GAMES, pack engine $ENGINE, at $(git -C "$REPO" rev-parse --short=12 HEAD)"
(cd "$REPO/android/app" && ./gradlew --no-daemon -q -PnxLive=true -PnxGames="$GAMES" -PnxPackEngine="$ENGINE" \
    -PnxVersion="$VERSION" -PnxVersionCode="$CODE" apk)
APK="$REPO/android/app/out/anka-native-$VERSION.apk"
[ -f "$APK" ] || die "the build made no $APK"

# Verify the signature (v1 to v3 as apksigner checks them) and read the certificate
VERIFY=$("$APKSIGNER" verify --verbose --print-certs "$APK") || die "apksigner verify failed"
SIGNERS=$(grep -c '^Signer #[0-9]* certificate SHA-256 digest:' <<<"$VERIFY" || true)
[ "$SIGNERS" = 1 ] || die "expected one signer, apksigner reports $SIGNERS"
CERT=$(sed -n 's/^Signer #1 certificate SHA-256 digest: *//p' <<<"$VERIFY")
grep -E '^Verified using v[0-9]+ scheme' <<<"$VERIFY" || true
echo "Certificate SHA-256: $CERT"
if [ -n "$OFFICIAL" ] && [ "$CERT" != "$OFFICIAL" ]; then
    die "the APK's certificate is not Signer.OFFICIAL ($OFFICIAL): wrong key, or Signer.OFFICIAL is out of date"
fi

mkdir -p "$OUT"
cp "$APK" "$OUT/"
(cd "$OUT" && sha256sum "anka-native-$VERSION.apk" > SHA256SUMS.txt)
echo "Release folder: $OUT"
cat "$OUT/SHA256SUMS.txt"
