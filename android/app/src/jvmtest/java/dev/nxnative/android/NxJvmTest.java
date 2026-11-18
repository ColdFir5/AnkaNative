package dev.nxnative.android;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

/**
 * Pure-Java NX archive reader tests on the JVM (no Android): Nx + Nup dispatch, SPEC (0,2)/(0,3) header/catalog/manifest/index validation,
 * slice CRC-32C + file SHA-256 verification, Zstd block decoding, selector parsing, Pack/DumpLayout/BuildCheck integration, and real OoT .nx vs .nup comparison.
 */
public final class NxJvmTest {
    private static int n;
    static final String TID = "0400486024DEC000", AOC = "0400486024dec001";

    static void check(boolean ok, String what) {
        if (!ok) throw new AssertionError(what);
        n++;
    }

    static void write(File f, byte[] b) throws IOException {
        f.getParentFile().mkdirs();
        Files.write(f.toPath(), b);
    }

    static byte[] npdm(long id) {
        byte[] b = new byte[0x100];
        b[0] = 'M'; b[1] = 'E'; b[2] = 'T'; b[3] = 'A';
        b[0x70] = (byte) 0x80;
        b[0x80] = 'A'; b[0x81] = 'C'; b[0x82] = 'I'; b[0x83] = '0';
        for (int i = 0; i < 8; i++) b[0x90 + i] = (byte) (id >>> (8 * i));
        return b;
    }

    static byte[] nso(int seed, int size) {
        byte[] b = new byte[size];
        b[0] = 'N'; b[1] = 'S'; b[2] = 'O'; b[3] = '0';
        for (int i = 0; i < 32; i++) b[0x40 + i] = (byte) (seed * 31 + i);
        for (int i = 0x100; i < size; i++) b[i] = (byte) (seed + i * 17);
        return b;
    }

    /** Valid single-frame RLE Zstandard block of `len` bytes equal to `val` */
    static byte[] zstRle(int len, byte val) {
        return new byte[] {
            0x28, (byte) 0xB5, 0x2F, (byte) 0xFD,
            (byte) 0xA0,
            (byte) len, (byte) (len >>> 8), (byte) (len >>> 16), (byte) (len >>> 24),
            (byte) ((len << 3) | 3), (byte) (len >>> 5), (byte) (len >>> 13),
            val
        };
    }

    private static void w16(byte[] b, int o, int v) { b[o] = (byte) v; b[o + 1] = (byte) (v >>> 8); }
    private static void w32(byte[] b, int o, long v) { for (int i = 0; i < 4; i++) b[o + i] = (byte) (v >>> (8 * i)); }
    private static void w64(byte[] b, int o, long v) { for (int i = 0; i < 8; i++) b[o + i] = (byte) (v >>> (8 * i)); }

    /** Minimal clean-room NX (0,2)/(0,3) writer for synthetic tests */
    static byte[] buildFixture(int minorVer, int badReqExt) throws Exception {
        long tid = Long.parseUnsignedLong(TID, 16);
        long aocTid = Long.parseUnsignedLong(AOC, 16);
        byte[] mainBase = nso(1, 0x200);
        byte[] mainUpd = nso(2, 0x200);
        byte[] npdmBytes = npdm(tid);
        byte[] empty = new byte[0];
        byte[] rawTail = new byte[4096];
        new Random(42).nextBytes(rawTail);
        byte[] multi = new byte[65536 + 4096];
        Arrays.fill(multi, 0, 65536, (byte) 'Z');
        System.arraycopy(rawTail, 0, multi, 65536, 4096);
        byte[] hello = "hello-nx\n".getBytes(StandardCharsets.UTF_8);
        byte[] dlcData = "dlc-bytes".getBytes(StandardCharsets.UTF_8);

        // Blocks: 0=mainBase(RAW), 1=mainUpd(RAW), 2=npdm(RAW), 3=zst64k(ZSTD), 4=rawTail(RAW), 5=hello(RAW), 6=dlcData(RAW)
        byte[][] stored = {mainBase, mainUpd, npdmBytes, zstRle(65536, (byte) 'Z'), rawTail, hello, dlcData};
        byte[][] logical = {mainBase, mainUpd, npdmBytes, Arrays.copyOf(multi, 65536), rawTail, hello, dlcData};
        int[] codec = {0, 0, 0, 1, 0, 0, 0};

        ByteArrayOutputStream data = new ByteArrayOutputStream();
        long[] blkOff = new long[stored.length];
        int[] blkFchk = new int[stored.length];
        List<Integer> checks = new ArrayList<>();
        long at = 8192;
        for (int i = 0; i < stored.length; i++) {
            blkOff[i] = at;
            data.write(stored[i]);
            at += stored[i].length;
            while ((at & 15) != 0) { data.write(0); at++; }
            blkFchk[i] = checks.size();
            int nc = (logical[i].length + 16383) / 16384;
            for (int s = 0; s < nc; s++) {
                int so = s * 16384, sl = Math.min(16384, logical[i].length - so);
                checks.add(Nx.crc32c(logical[i], so, sl));
            }
        }
        long dataEnd = blkOff[stored.length - 1] + stored[stored.length - 1].length;
        long dataSize = dataEnd - 8192;

        // Dirs (sorted entries):
        // 0: base root -> exefs(1), romfs(2)
        // 1: base exefs -> main(f0), main.npdm(f2)
        // 2: base romfs -> empty.bin(f3), multi.bin(f4), sub(3)
        // 3: base romfs/sub -> a.txt(f5)
        // 4: upd root -> exefs(5), romfs(6)
        // 5: upd exefs -> main(f1), main.npdm(f2)
        // 6: upd romfs -> multi.bin(f4)
        // 7: dlc root -> romfs(8)
        // 8: dlc romfs -> d.txt(f6)
        int[][] dirs = {{0, 2}, {2, 2}, {4, 3}, {7, 1}, {8, 2}, {10, 2}, {12, 1}, {13, 1}, {14, 1}};
        String[] enames = {"exefs", "romfs", "main", "main.npdm", "empty.bin", "multi.bin", "sub", "a.txt", "exefs", "romfs", "main", "main.npdm", "multi.bin", "romfs", "d.txt"};
        int[] ekind = {1, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2};
        int[] etarg = {1, 2, 0, 2, 3, 4, 3, 5, 5, 6, 1, 2, 4, 8, 6};
        // Files: 0=mainBase, 1=mainUpd, 2=npdm, 3=empty, 4=multi(blocks 3,4), 5=hello, 6=dlcData
        byte[][] fdata = {mainBase, mainUpd, npdmBytes, empty, multi, hello, dlcData};
        int[] fref = {0, 1, 2, 3, 3, 5, 6};
        int[] fbc = {1, 1, 1, 0, 2, 1, 1};
        int[] refs = {0, 1, 2, 3, 4, 5, 6};

        int nd = dirs.length, nf = fdata.length, ne = enames.length, nref = refs.length;
        long mDir = 96;
        long mFile = (mDir + nd * 16L + 7) & ~7L;
        long mEnt = (mFile + nf * 64L + 7) & ~7L;
        long mRef = (mEnt + ne * 24L + 7) & ~7L;
        long mPool = (mRef + nref * 4L + 7) & ~7L;
        ByteArrayOutputStream pool = new ByteArrayOutputStream();
        long[] eno = new long[ne];
        for (int i = 0; i < ne; i++) {
            eno[i] = mPool + pool.size();
            pool.write(enames[i].getBytes(StandardCharsets.UTF_8));
        }
        int manSz = (int) (mPool + pool.size());
        byte[] man = new byte[manSz];
        System.arraycopy("NXMF".getBytes(StandardCharsets.US_ASCII), 0, man, 0, 4);
        w16(man, 4, 1); w16(man, 6, 96); w64(man, 8, manSz);
        w32(man, 16, nd); w32(man, 20, nf); w32(man, 24, ne); w32(man, 28, nref);
        w64(man, 32, mDir); w64(man, 40, mFile); w64(man, 48, mEnt); w64(man, 56, mRef);
        w64(man, 64, mPool); w64(man, 72, pool.size());
        for (int i = 0; i < nd; i++) { w32(man, (int) mDir + i * 16, dirs[i][0]); w32(man, (int) mDir + i * 16 + 4, dirs[i][1]); }
        MessageDigest md = Nup.sha256();
        for (int i = 0; i < nf; i++) {
            int o = (int) mFile + i * 64;
            w64(man, o, fdata[i].length); w32(man, o + 8, fref[i]); w32(man, o + 12, fbc[i]); man[o + 16] = 16;
            System.arraycopy(md.digest(fdata[i]), 0, man, o + 20, 32);
        }
        for (int i = 0; i < ne; i++) {
            int o = (int) mEnt + i * 24;
            w64(man, o, eno[i]); w32(man, o + 8, enames[i].getBytes(StandardCharsets.UTF_8).length);
            w32(man, o + 12, etarg[i]); man[o + 16] = (byte) ekind[i];
        }
        for (int i = 0; i < nref; i++) w32(man, (int) mRef + i * 4, refs[i]);
        System.arraycopy(pool.toByteArray(), 0, man, (int) mPool, pool.size());

        // Catalog: 1 app, 1 prog, 3 snaps (base, update/458752, addon), extensions
        int extCnt = 2 + (minorVer == 3 ? 1 : 0) + (badReqExt != 0 ? 1 : 0);
        long cApp = 128;
        long cProg = (cApp + 80 + 7) & ~7L;
        long cSnap = (cProg + 64 + 7) & ~7L;
        long cExt = (cSnap + 3 * 120L + 7) & ~7L;
        long cPool = (cExt + extCnt * 32L + 7) & ~7L;
        long cExtp = (cPool + 7) & ~7L;
        int catSz = (int) (cExtp + extCnt * 16L);
        byte[] cat = new byte[catSz];
        System.arraycopy("NXCT".getBytes(StandardCharsets.US_ASCII), 0, cat, 0, 4);
        w16(cat, 4, 1); w16(cat, 6, 128); w64(cat, 8, catSz);
        w32(cat, 20, 1); w32(cat, 24, 1); w32(cat, 28, 3); w32(cat, 40, extCnt);
        w64(cat, 48, cApp); w64(cat, 56, cProg); w64(cat, 64, cSnap); w64(cat, 88, cExt); w64(cat, 96, cPool);
        w64(cat, (int) cApp, tid); cat[(int) cApp + 8] = 2; cat[(int) cApp + 9] = 3;
        w32(cat, (int) cApp + 16, 1); w32(cat, (int) cApp + 24, 3);
        w32(cat, (int) cApp + 28, (int) Nup.NONE); w32(cat, (int) cApp + 36, (int) Nup.NONE); w32(cat, (int) cApp + 44, (int) Nup.NONE);
        w32(cat, (int) cApp + 56, (int) Nup.NONE); w32(cat, (int) cApp + 60, (int) Nup.NONE); w32(cat, (int) cApp + 68, (int) Nup.NONE);
        w64(cat, (int) cProg, tid); w32(cat, (int) cProg + 36, 2); w32(cat, (int) cProg + 40, (int) Nup.NONE);

        int[] spidx = {0, 0, (int) Nup.NONE};
        int[] stype = {0, 1, 2};
        int[] sflag = {0, 0, 2};
        int[] sver = {0, 458752, 0};
        int[] sroot = {0, 4, 7};
        for (int i = 0; i < 3; i++) {
            int o = (int) cSnap + i * 120;
            w32(cat, o, spidx[i]); cat[o + 4] = (byte) stype[i]; cat[o + 5] = (byte) sflag[i];
            w32(cat, o + 8, sver[i]); w32(cat, o + 12, sroot[i]); w32(cat, o + 24, (int) Nup.NONE);
            if (i == 2) w64(cat, o + 32, aocTid);
        }
        int[] etypes = {0x00010004, 0x00010005, 0x00010008, badReqExt};
        int ei = 0;
        long ep = cExtp;
        for (int k = 0; k < 4; k++) {
            if (k == 2 && minorVer != 3) continue;
            if (k == 3 && badReqExt == 0) continue;
            int o = (int) cExt + ei * 32;
            w32(cat, o, etypes[k]); w32(cat, o + 4, 1); w16(cat, o + 8, 1); w32(cat, o + 12, 0);
            w64(cat, o + 16, ep); w64(cat, o + 24, 16);
            ep += 16; ei++;
        }

        byte[] idxBytes = new byte[stored.length * 24];
        for (int i = 0; i < stored.length; i++) {
            int o = i * 24;
            w64(idxBytes, o, blkOff[i]); w32(idxBytes, o + 8, stored[i].length); w32(idxBytes, o + 12, logical[i].length);
            idxBytes[o + 16] = (byte) codec[i]; w32(idxBytes, o + 20, blkFchk[i]);
        }
        byte[] intBytes = new byte[checks.size() * 4];
        for (int i = 0; i < checks.size(); i++) w32(intBytes, i * 4, checks.get(i));

        long catOff = (dataEnd + 15) & ~15L;
        long manOff = (catOff + catSz + 15) & ~15L;
        long biOff = (manOff + manSz + 15) & ~15L;
        long intOff = (biOff + idxBytes.length + 15) & ~15L;
        long archSz = intOff + intBytes.length;

        byte[] out = new byte[(int) archSz];
        byte[] db = data.toByteArray();
        System.arraycopy(db, 0, out, 8192, (int) dataSize);
        System.arraycopy(cat, 0, out, (int) catOff, catSz);
        System.arraycopy(man, 0, out, (int) manOff, manSz);
        System.arraycopy(idxBytes, 0, out, (int) biOff, idxBytes.length);
        System.arraycopy(intBytes, 0, out, (int) intOff, intBytes.length);

        byte[] hdr = new byte[256];
        System.arraycopy("NXAR".getBytes(StandardCharsets.US_ASCII), 0, hdr, 0, 4);
        w16(hdr, 4, 0); w16(hdr, 6, minorVer); w32(hdr, 8, 256); w16(hdr, 12, 1);
        w64(hdr, 16, archSz); w64(hdr, 24, catOff); w64(hdr, 32, catSz);
        w64(hdr, 56, 8192); w64(hdr, 64, dataSize);
        w64(hdr, 72, manOff); w64(hdr, 80, manSz);
        w64(hdr, 88, biOff); w64(hdr, 96, stored.length);
        w64(hdr, 104, intOff); w64(hdr, 112, intBytes.length);
        System.arraycopy(md.digest(cat), 0, hdr, 120, 32);
        System.arraycopy(md.digest(man), 0, hdr, 152, 32);
        md.update(idxBytes); md.update(intBytes);
        System.arraycopy(md.digest(), 0, hdr, 184, 32);
        w32(hdr, 224, 16384);
        w32(hdr, 252, Nx.crc32c(hdr, 0, 252));
        System.arraycopy(hdr, 0, out, 0, 256);
        System.arraycopy(hdr, 0, out, 4096, 256);
        return out;
    }

    public static void main(String[] args) throws Exception {
        if (args.length >= 3 && args[0].equals("real")) {
            real(new File(args[1]), new File(args[2]));
            System.out.println("ok " + n);
            return;
        }
        File tmp = new File(args[0]);
        names();
        synthetic(tmp);
        damage(tmp);
        System.out.println("ok " + n);
    }

    static void names() {
        Nup.Ref r = Nup.parse("/s/Game.nx:update/131072/romfs");
        check(r.path.equals("/s/Game.nx") && r.selector.equals("update/131072") && r.sub.equals("romfs"), "parse .nx update/romfs");
        r = Nup.parse("Game.nx:base");
        check(r.path.equals("Game.nx") && r.selector.equals("base") && r.sub.isEmpty(), "parse .nx base");
        r = Nup.parse("Game.nx:addon/" + AOC.toUpperCase(Locale.ROOT) + "/romfs");
        check(r.selector.equals("addon/" + AOC) && r.sub.equals("romfs"), "parse .nx addon lower-cased");
        r = Nup.parse("/a.nx:dir/b.nx:update/007/romfs");
        check(r.path.equals("/a.nx:dir/b.nx") && r.selector.equals("update/7"), "split at last .nx:");
        r = Nup.parse("/s/Game.nx");
        check(r.selector == null && r.path.equals("/s/Game.nx"), "plain .nx");
        check(Nup.parse(".nx") == null, "bare .nx refused");
        check(Nup.isNup(new File("/x/G.nx")) && Nup.isNx(new File("/x/G.nx")) && !Nup.isNup(new File("/x/G.NX")), "isNup / isNx");
        check(Nup.spec(new File("/s/G.nx"), "base", "romfs").equals("/s/G.nx:base/romfs"), "spec .nx");
    }

    static void synthetic(File tmp) throws Exception {
        byte[] fix2 = buildFixture(2, 0);
        File f2 = new File(tmp, "Game.nx");
        write(f2, fix2);
        Nup x = Nup.open(f2);
        check(x instanceof Nx, "Nup.open dispatches NXAR to Nx");
        check(x.packages.size() == 3, "3 snapshots");
        check(x.programId().equals(TID), "programId " + x.programId());
        Nup.Pkg base = x.base(), upd = x.find("update/458752"), dlc = x.find("addon/" + AOC);
        check(base != null && upd != null && dlc != null && x.newest() == upd, "base/update/addon/newest");

        byte[] multiWant = new byte[65536 + 4096];
        Arrays.fill(multiWant, 0, 65536, (byte) 'Z');
        byte[] tail = new byte[4096]; new Random(42).nextBytes(tail);
        System.arraycopy(tail, 0, multiWant, 65536, 4096);

        check(Arrays.equals(x.read(x.lookup(base, "romfs/empty.bin"), -1), new byte[0]), "empty file");
        check(Arrays.equals(x.read(x.lookup(base, "romfs/sub/a.txt"), -1), "hello-nx\n".getBytes(StandardCharsets.UTF_8)), "nested raw file");
        check(Arrays.equals(x.read(x.lookup(base, "romfs/multi.bin"), -1), multiWant), "multi-block ZSTD+RAW file");
        byte[] head = x.read(x.lookup(base, "romfs/multi.bin"), 700);
        check(head.length == 700 && Arrays.equals(head, Arrays.copyOf(multiWant, 700)), "partial head read");
        check(Arrays.equals(x.read(x.lookup(dlc, "romfs/d.txt"), -1), "dlc-bytes".getBytes(StandardCharsets.UTF_8)), "addon file");

        // DumpLayout + Pack integration
        check(DumpLayout.versions(f2).equals(List.of(0L, 458752L)), "DumpLayout.versions");
        check(DumpLayout.preparable(f2), "DumpLayout.preparable");
        check(DumpLayout.path(f2, 458752, "romfs").equals(f2.getPath() + ":update/458752/romfs"), "DumpLayout.path");
        File pdir = new File(tmp, "pack");
        write(new File(pdir, "main.patched.bin"), new byte[1]);
        write(new File(pdir, "pack.tsv"), ("format\t1\ntarget\tandroid\ntitle_id\t" + TID + "\nname\tGame\nversion\t458752\nromfs\t"
                + f2.getName() + ":update/458752/romfs\nmodules\tmain\naoc\t1\t" + f2.getName() + ":addon/" + AOC + "/romfs\n").getBytes(StandardCharsets.UTF_8));
        Pack p = Pack.read(pdir);
        check(p.romfsPath(f2).equals(f2.getAbsolutePath() + ":update/458752/romfs"), "Pack.romfsPath .nx");
        check(f2.getAbsolutePath().equals(p.aocLinks(f2).get("1.nx")), "Pack.aocLinks uses 1.nx");

        // Version (0,3) with 0x00010008
        File f3 = new File(tmp, "Game03.nx");
        write(f3, buildFixture(3, 0));
        check(Nup.open(f3).packages.size() == 3, "NX (0,3) opens");

        // Uppercase .NX refused by wrongCase
        File upper = new File(tmp, "UPPER.NX");
        write(upper, fix2);
        check(Nup.wrongCase(upper) && !Nup.isNup(upper), "uppercase .NX flagged as wrongCase");
    }

    static void expectBad(File f, String what) {
        try {
            Nup.open(f);
        } catch (IOException e) {
            n++;
            return;
        }
        throw new AssertionError("expected failure: " + what);
    }

    static void damage(File tmp) throws Exception {
        byte[] good = buildFixture(2, 0);
        File bad = new File(tmp, "bad.nx");
        write(bad, buildFixture(2, 0x00019999));
        expectBad(bad, "unknown required extension");

        // Corrupt catalog SHA in both headers
        byte[] b = good.clone();
        b[120] ^= 1; w32(b, 252, Nx.crc32c(b, 0, 252));
        System.arraycopy(b, 0, b, 4096, 256);
        write(bad, b);
        expectBad(bad, "Catalog SHA-256 mismatch");

        // Identical generation differing slots
        b = good.clone();
        b[4096 + 152] ^= 1; w32(b, 4096 + 252, Nx.crc32c(b, 4096, 252));
        write(bad, b);
        expectBad(bad, "identical generation differing slots");

        // Dual generation fallback: gen 1 damaged -> falls back to gen 0
        b = good.clone();
        w64(b, 4096 + 216, 1); b[4096 + 120] ^= 1; w32(b, 4096 + 252, Nx.crc32c(b, 4096, 252));
        write(bad, b);
        check(Nup.open(bad).packages.size() == 3, "dual-generation fallback to gen 0");

        // Corrupt block payload fails CRC-32C on read
        b = good.clone();
        b[8192] ^= 0x55;
        File badBlk = new File(tmp, "badblk.nx");
        write(badBlk, b);
        Nup dn = Nup.open(badBlk);
        try {
            dn.read(dn.lookup(dn.base(), "exefs/main"), -1);
            throw new AssertionError("expected CRC-32C failure");
        } catch (IOException e) {
            check(e.getMessage().contains("CRC-32C") || e.getMessage().contains("SHA-256"), "corrupt block rejected: " + e.getMessage());
        }
    }

    static void real(File nxFile, File nupFile) throws Exception {
        long t0 = System.nanoTime();
        Nup nx = Nup.open(nxFile);
        Nup nup = Nup.open(nupFile);
        System.out.printf("opened real OoT in %.2f s: nx=%d pkgs, nup=%d pkgs%n",
                (System.nanoTime() - t0) / 1e9, nx.packages.size(), nup.packages.size());
        check(nx.packages.size() == nup.packages.size(), "package count matches: " + nx.packages.size());
        check(nx.programId().equals(nup.programId()), "programId matches: " + nx.programId());

        MessageDigest md = Nup.sha256();
        for (Nup.Pkg np : nup.packages) {
            String sel = np.selector();
            Nup.Pkg xp = nx.find(sel);
            check(xp != null, "nx has package " + sel);
            System.out.println("  package " + sel);
            Nup.Node nex = nup.dir(np.exefsDir), xex = nx.dir(xp.exefsDir);
            List<Nup.Node> nExFiles = nup.children(nex), xExFiles = nx.children(xex);
            check(nExFiles.size() == xExFiles.size(), sel + " exefs entry count");
            for (int i = 0; i < nExFiles.size(); i++) {
                Nup.Node nf = nExFiles.get(i), xf = xExFiles.get(i);
                check(nf.name.equals(xf.name) && nf.size == xf.size, sel + "/exefs/" + nf.name);
                if (nf.name.equals("main.npdm") || nf.size <= (4 << 20)) {
                    byte[] h1 = md.digest(nup.read(nf, -1));
                    byte[] h2 = md.digest(nx.read(xf, -1));
                    check(Arrays.equals(h1, h2), sel + "/exefs/" + nf.name + " SHA-256 matches");
                }
            }

            Map<String, Nup.Node> nMap = walkFiles(nup, nup.dir(np.romfsDir), "");
            Map<String, Nup.Node> xMap = walkFiles(nx, nx.dir(xp.romfsDir), "");
            System.out.printf("    romfs files: nx=%d nup=%d%n", xMap.size(), nMap.size());
            int checked = 0;
            for (Map.Entry<String, Nup.Node> e : nMap.entrySet()) {
                Nup.Node xf = xMap.get(e.getKey());
                check(xf != null && xf.size == e.getValue().size, sel + "/romfs/" + e.getKey() + " present with equal size");
                if (checked < 10 && e.getValue().size <= (8 << 20)) {
                    byte[] h1 = md.digest(nup.read(e.getValue(), -1));
                    byte[] h2 = md.digest(nx.read(xf, -1));
                    check(Arrays.equals(h1, h2), sel + "/romfs/" + e.getKey() + " SHA-256 matches");
                    checked++;
                }
            }
        }
    }

    private static Map<String, Nup.Node> walkFiles(Nup arc, Nup.Node root, String prefix) throws IOException {
        Map<String, Nup.Node> out = new LinkedHashMap<>();
        List<Nup.Node> q = new ArrayList<>();
        List<String> paths = new ArrayList<>();
        for (Nup.Node c : arc.children(root)) { q.add(c); paths.add(prefix.isEmpty() ? c.name : prefix + "/" + c.name); }
        for (int i = 0; i < q.size(); i++) {
            Nup.Node c = q.get(i);
            String p = paths.get(i);
            if (c.dir) {
                for (Nup.Node cc : arc.children(c)) { q.add(cc); paths.add(p + "/" + cc.name); }
            } else {
                out.put(p, c);
            }
        }
        return out;
    }
}
