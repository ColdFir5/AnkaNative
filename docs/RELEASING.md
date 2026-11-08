# Releasing Anka Native

How the owner builds and signs a public release. The release key is the owner's alone: it is never in this repository, never in a pull request and never
in CI. Nothing here creates it for you or chooses its passwords: you run the commands and type the passwords yourself.

The release APK is the release app (`-PnxLive=true`): app id `dev.ankanative.android`, named "Anka Native", not debuggable, with the in-app updater on.
By default it runs Ocarina of Time (`0400486024DEC000`) and has no on-phone packer (`-PnxPackEngine=fake`); `android/tools/release.sh` builds it.

## 1. Create the release key (once)

Pick a folder outside this repository, for example `~/keys/anka-native`, and run (it asks for the keystore password twice; for a PKCS12 keystore, the
default, the key's password is the same as the store's):

```sh
mkdir -p ~/keys/anka-native && chmod 700 ~/keys/anka-native
keytool -genkeypair -v \
  -keystore ~/keys/anka-native/anka-native-release.jks -storetype PKCS12 \
  -alias anka-native -keyalg RSA -keysize 4096 -validity 10950 \
  -dname "CN=EddizEge, O=Anka Native"
```

- `-validity 10950` is 30 years: Android wants a release key valid well past the app's life, and a key cannot be extended later.
- The `-dname` names are public: anyone can read them from the APK. Leave out `-dname` and keytool asks for each field.
- Use a long password and keep it in a password manager.

**Keep the keystore safe and backed up.** Android installs an update only when it is signed by the same key as the installed app: if the keystore or its
password is lost, nobody can update the app any more, and every player has to uninstall it (losing its data folder) to install a new one. Keep at least two
copies outside this repository (for example an encrypted USB stick and an encrypted cloud backup), never in a Git repository, a shared folder or a chat.
`.gitignore` refuses `*.jks` and `*.keystore` files, and the Gradle build and `release.sh` refuse a keystore inside the repository.

## 2. Tell the build where the key is

The Gradle release build reads four settings from the environment, or else from your own `~/.gradle/gradle.properties` (never from the project's
`gradle.properties` or a `-P` option):

| Setting | Value |
|---|---|
| `ANKA_KEYSTORE` | the keystore's path, e.g. `$HOME/keys/anka-native/anka-native-release.jks` |
| `ANKA_KEY_ALIAS` | `anka-native` |
| `ANKA_KEYSTORE_PASSWORD` | the keystore's password |
| `ANKA_KEY_PASSWORD` | the key's password (the store's when unset; the same for PKCS12) |

For one shell session, typing the password without it going into the shell's history:

```sh
export ANKA_KEYSTORE="$HOME/keys/anka-native/anka-native-release.jks"
export ANKA_KEY_ALIAS=anka-native
read -rs -p 'Keystore password: ' ANKA_KEYSTORE_PASSWORD; echo
export ANKA_KEYSTORE_PASSWORD ANKA_KEY_PASSWORD="$ANKA_KEYSTORE_PASSWORD"
```

Or keep the path and alias (and, if you accept a password in a file, the passwords) in `~/.gradle/gradle.properties`, readable only by you
(`chmod 600 ~/.gradle/gradle.properties`):

```properties
ANKA_KEYSTORE=/home/<you>/keys/anka-native/anka-native-release.jks
ANKA_KEY_ALIAS=anka-native
```

## 3. Fill in the certificate (once, after the key exists)

The app ends its build tag with "official" only when its signing certificate is the one in `Signer.OFFICIAL`
(`android/app/src/main/java/dev/nxnative/android/Signer.java`, the block marked THE RELEASE CERTIFICATE). Until it is filled in it is `NOT_SET`, and every
build says "unofficial build". To fill it in:

```sh
android/tools/release.sh --fingerprint     # prints 64 lower-case hex digits (keytool asks for the password unless ANKA_KEYSTORE_PASSWORD is set)
```

Put them in place of `NOT_SET`, as `static final String OFFICIAL = "<the 64 hex digits>";`, then check and commit:

```sh
python3 -m unittest dm/tools/tests/test_android_signer.py
git commit -am "Signer: the release certificate"
```

The fingerprint is public data (anyone can read it from the APK); publish it in the release notes too, so players can check the APK they download
(`apksigner verify --print-certs`).

## 4. Build the release

From a clean checkout of the commit you release (`release.sh` refuses uncommitted changes, since the release's source is that commit):

```sh
android/tools/release.sh
```

It builds with a JDK 21 (`JAVA_HOME`, by default `/usr/lib/jvm/java-21-openjdk`; Gradle fails with JDK 27), lets Gradle sign the APK with the release key,
verifies it with `apksigner verify --print-certs` (one signer), prints the certificate's SHA-256 and stops when it is not `Signer.OFFICIAL`. The APK
(`anka-native-<version>.apk`) and `SHA256SUMS.txt` go to `~/anka-native-releases/<version>/`, outside the repository.

Options (environment): `ANKA_VERSION` (default `0.1.0`) and `ANKA_VERSION_CODE` (default `1`; raise it for every release, Android updates only to a higher
code), `ANKA_GAMES` (default `0400486024DEC000`; several are comma-separated), `ANKA_PACK_ENGINE` (`native` by default, which needs the on-phone packer's stage in
`NX_PACK_STAGE`, see [android/README.md](../android/README.md); `fake` builds without the packer, and the app then offers no Prepare), `ANKA_RELEASE_DIR` (another folder outside the repository).

## 5. Publish

1. Push the released commit (and its tag) to `github.com/EddizEge/AnkaNative` **before or together with** the APK: the APK is GPL-3.0-or-later as a whole,
   so its complete source (this repository at the commit in the app's build tag) must be available to everyone who gets it ([NOTICE.md](../NOTICE.md)).
2. Create a GitHub release (tag `v<version>`, marked as a pre-release) and upload only the APK and `SHA256SUMS.txt`. The in-app updater takes the
   release's first `.apk` asset and checks it against GitHub's SHA-256 digest of it.
3. Put the certificate's SHA-256 in the release notes.
