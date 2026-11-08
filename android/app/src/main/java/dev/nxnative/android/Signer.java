package dev.nxnative.android;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * The official-build stamp: which certificate signed this APK, and whether it is the release key's. The build tag's version, commit and date are text anyone can
 * rebuild with, so the play and release apps add "official" only when the APK's own signing certificate is the release one, and "unofficial build" otherwise;
 * the test app makes no claim. The first 16 hex digits of the signer's SHA-256 go into the first line of every game log and the readiness report, so a bug report
 * shows which key signed the build. It is a convenience, not a proof: a modified app can print any text it likes; what a user verifies is the certificate itself
 * (apksigner verify --print-certs). Pure Java (no Android): Prefs reads the certificates (PackageManager) and hands them here; SignerJvmTest drives it on the JVM.
 */
final class Signer {
    /** The release certificate's fingerprint before the release key exists: matches no signer */
    static final String NOT_SET = "not set";

    // ======================================================================================================================================================
    // THE RELEASE CERTIFICATE: the one place its fingerprint lives (dm/tools/tests/test_android_signer.py reads it from here). NOT SET until the owner's release
    // key exists. Then: docs/RELEASING.md, "Fill in the certificate": run `android/tools/release.sh --fingerprint`, put the 64 lower-case hex digits it prints here
    // in quotes in place of NOT_SET (OFFICIAL = "<64 hex digits>";), commit, and build the release with android/tools/release.sh.
    // While it is NOT_SET no APK counts as official: the play and release apps end their build tag with "unofficial build".
    // ======================================================================================================================================================
    /** SHA-256 of the release signing certificate (DER), lower-case hex, as apksigner verify --print-certs prints it (public data); {@link #NOT_SET} until it exists */
    static final String OFFICIAL = NOT_SET;

    private Signer() {}

    /** Whether a fingerprint is filled in: 64 hex digits (anything else, NOT_SET included, matches no signer) */
    static boolean isSet(String cert) { return cert != null && cert.matches("[0-9a-fA-F]{64}"); }

    /** The SHA-256 of a certificate's DER bytes, lower-case hex, as apksigner prints it */
    static String sha256(byte[] der) {
        try {
            byte[] h = MessageDigest.getInstance("SHA-256").digest(der);
            StringBuilder s = new StringBuilder(64);
            for (byte b : h) s.append(Character.forDigit((b >> 4) & 15, 16)).append(Character.forDigit(b & 15, 16));
            return s.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);   // every JVM and Android has SHA-256
        }
    }

    /** Official only with exactly one signer and that signer the release certificate (an APK signed by several keys is not ours, whatever else signed it) */
    static boolean official(String[] signers) { return official(signers, OFFICIAL); }

    /** {@link #official(String[])} against a given release certificate (the tests drive it with a sample one); never official while it is not set */
    static boolean official(String[] signers, String cert) {
        return isSet(cert) && signers != null && signers.length == 1 && cert.equalsIgnoreCase(signers[0]);
    }

    /** The build tag's mark: "official" or "unofficial build" for the play and release apps, null (no claim) for the test app */
    static String mark(boolean play, String[] signers) { return mark(play, signers, OFFICIAL); }

    static String mark(boolean play, String[] signers, String cert) {
        if (!play) return null;
        return official(signers, cert) ? "official" : "unofficial build";
    }

    /** For logs and reports: the first signer's first 16 hex digits ("+<n>" when more keys signed it), "unknown" when none could be read */
    static String shortId(String[] signers) {
        if (signers == null || signers.length == 0 || signers[0] == null || signers[0].length() < 16) return "unknown";
        return signers[0].substring(0, 16).toLowerCase(java.util.Locale.ROOT) + (signers.length > 1 ? "+" + (signers.length - 1) : "");
    }

    /**
     * Which build this is, without the app's name: play app "0.1 · 6a6448d" (the version up to its "-", the commit); test app "int@6a6448d" (the branch without
     * "android-", the commit). "+dirty" follows the commit when the tree had changes. No build time: the commit and the signer's fingerprint identify a build. Home
     * shows it under the title; {@link #buildTag} is it with the app's name, so the two never disagree
     */
    static String buildId(boolean play, String versionName, String commit, String dirty, String branch) {
        String c = commit;
        if (c.length() > 7 && !c.equals("unknown")) c = c.substring(0, 7);
        c += dirty;
        if (play) {
            int d = versionName.indexOf('-');
            return (d > 0 ? versionName.substring(0, d) : versionName) + " · " + c;
        }
        String b = branch.startsWith("android-") ? branch.substring(8) : branch;
        return b + "@" + c;
    }

    /** The build tag without its mark: "Anka Native &lt;buildId&gt;" (play app), "Anka Native test · &lt;buildId&gt;" (test app) */
    static String buildTag(boolean play, String buildId) { return "Anka Native " + (play ? "" : "test · ") + buildId; }

    /** The build tag with its mark:"<tag> · official", "<tag> · unofficial build", or the tag alone (test app) */
    static String tag(String tag, boolean play, String[] signers) { return tag(tag, play, signers, OFFICIAL); }

    static String tag(String tag, boolean play, String[] signers, String cert) {
        String m = mark(play, signers, cert);
        return m == null ? tag : tag + " · " + m;
    }
}
