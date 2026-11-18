package dev.nxnative.android;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Pure-Java reader for NX archives (.nx, magic NXAR, versions (0,2) and (0,3); nxformat SPEC.md).
 * Dispatched from {@link Nup#open} / {@link Nup#peek} so all launcher callers share one {@link Nup} API.
 */
final class Nx extends Nup {
    private static final int CHECK_SIZE = 16384;
    private static final long MAX_CATALOG = 16L << 20;
    private static final long MAX_MANIFEST = 4L << 30;
    private static final int MAX_BLOCK_LOGICAL = 1 << 20;
    private static final int MAX_BLOCK_STORED = (1 << 20) + (64 << 10);

    private static final int[] CRC32C_TAB = new int[256];
    static {
        for (int i = 0; i < 256; i++) {
            int c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? (c >>> 1) ^ 0x82F63B78 : (c >>> 1);
            CRC32C_TAB[i] = c;
        }
    }

    static int crc32c(byte[] b, int off, int len) {
        int c = 0xFFFFFFFF;
        for (int i = 0; i < len; i++) c = (c >>> 8) ^ CRC32C_TAB[(c ^ b[off + i]) & 0xFF];
        return c ^ 0xFFFFFFFF;
    }

    static final class NxNode extends Nup.Node {
        final long firstBlockRef;
        final int blockCount, chunkShift;

        NxNode(long index, String name, long size, byte[] sha256, long firstBlockRef, int blockCount, int chunkShift) {
            super(false, index, name, size, 0, 0, 0, sha256, 0, 0);
            this.firstBlockRef = firstBlockRef;
            this.blockCount = blockCount;
            this.chunkShift = chunkShift;
        }
    }

    private static final class Header {
        final long archiveSize, catalogOff, catalogSize, mediaOff, mediaSize;
        final long dataOff, dataSize, manifestOff, manifestSize;
        final long blockIndexOff, blockCount, integrityOff, integritySize, generation;
        final byte[] catalogSha, manifestSha, indexSha;

        private Header(byte[] h) {
            archiveSize = u64(h, 16);
            catalogOff = u64(h, 24);
            catalogSize = u64(h, 32);
            mediaOff = u64(h, 40);
            mediaSize = u64(h, 48);
            dataOff = u64(h, 56);
            dataSize = u64(h, 64);
            manifestOff = u64(h, 72);
            manifestSize = u64(h, 80);
            blockIndexOff = u64(h, 88);
            blockCount = u64(h, 96);
            integrityOff = u64(h, 104);
            integritySize = u64(h, 112);
            catalogSha = Arrays.copyOfRange(h, 120, 152);
            manifestSha = Arrays.copyOfRange(h, 152, 184);
            indexSha = Arrays.copyOfRange(h, 184, 216);
            generation = u64(h, 216);
        }

        static Header parse(byte[] h, long physSize) {
            if (h.length != 256 || h[0] != 'N' || h[1] != 'X' || h[2] != 'A' || h[3] != 'R') return null;
            int maj = u16(h, 4), min = u16(h, 6);
            if (maj != 0 || (min != 2 && min != 3)) return null;
            if (u32(h, 8) != 256 || u16(h, 12) != 1 || u16(h, 14) != 0) return null;
            if (u32(h, 224) != CHECK_SIZE) return null;
            for (int i = 228; i < 252; i++) if (h[i] != 0) return null;
            if ((int) u32(h, 252) != crc32c(h, 0, 252)) return null;
            Header o = new Header(h);
            if (o.archiveSize <= 0 || o.archiveSize > physSize) return null;
            if ((o.catalogOff & 15) != 0 || o.catalogSize < 128 || o.catalogSize > MAX_CATALOG) return null;
            if ((o.manifestOff & 15) != 0 || o.manifestSize < 96 || o.manifestSize > MAX_MANIFEST) return null;
            if ((o.mediaOff == 0) != (o.mediaSize == 0)) return null;
            if (o.mediaSize != 0 && ((o.mediaOff & 15) != 0 || o.mediaSize < 32)) return null;
            if (o.blockCount < 0 || o.blockCount > 0x100000000L) return null;
            if (o.blockCount == 0) {
                if (o.dataOff != 0 || o.dataSize != 0 || o.blockIndexOff != 0 || o.integrityOff != 0 || o.integritySize != 0) return null;
            } else {
                if (o.dataOff != 8192 || o.dataSize <= 0) return null;
                if ((o.blockIndexOff & 15) != 0 || (o.integrityOff & 15) != 0 || (o.integritySize & 3) != 0) return null;
                if (o.integritySize / 4 < o.blockCount) return null;
            }
            long cur = o.blockCount > 0 ? addChecked(o.dataOff, o.dataSize) : 8192;
            if (cur < 0) return null;
            if (o.mediaSize != 0) {
                if (o.mediaOff < align16(cur)) return null;
                long mend = addChecked(o.mediaOff, o.mediaSize);
                if (mend < 0 || o.catalogOff != align16(mend)) return null;
            } else if (o.catalogOff < align16(cur)) return null;
            long cend = addChecked(o.catalogOff, o.catalogSize);
            if (cend < 0 || o.manifestOff != align16(cend)) return null;
            long manEnd = addChecked(o.manifestOff, o.manifestSize);
            if (manEnd < 0) return null;
            if (o.blockCount == 0) return manEnd == o.archiveSize ? o : null;
            if (o.blockIndexOff != align16(manEnd)) return null;
            long biSz = mulChecked(o.blockCount, 24);
            long biEnd = biSz < 0 ? -1 : addChecked(o.blockIndexOff, biSz);
            if (biEnd < 0 || o.integrityOff != align16(biEnd)) return null;
            long intEnd = addChecked(o.integrityOff, o.integritySize);
            return intEnd == o.archiveSize ? o : null;
        }
    }

    private final Header hdr;
    private final long entryCount, blockRefCount, dirTableOff, fileTableOff, entryTableOff, refTableOff, pathPoolOff, pathPoolSize;

    private Nx(File file, long fileSize, Header hdr, long dirCount, long fileCount, long entryCount, long blockRefCount,
               long dirTableOff, long fileTableOff, long entryTableOff, long refTableOff, long pathPoolOff, long pathPoolSize, List<Pkg> packages) {
        super(file, fileSize, hdr.catalogOff, hdr.catalogSize, hdr.dataOff, dirCount, fileCount, pathPoolOff, pathPoolSize, packages);
        this.hdr = hdr;
        this.entryCount = entryCount;
        this.blockRefCount = blockRefCount;
        this.dirTableOff = dirTableOff;
        this.fileTableOff = fileTableOff;
        this.entryTableOff = entryTableOff;
        this.refTableOff = refTableOff;
        this.pathPoolOff = pathPoolOff;
        this.pathPoolSize = pathPoolSize;
    }

    static Nx read(File f, boolean verify) throws IOException {
        if (!f.isFile()) throw new IOException(f + " is not a file");
        try (RandomAccessFile r = new RandomAccessFile(f, "r")) {
            long len = r.length();
            if (len < 256) throw new IOException(f.getName() + " is too short for a .nx");
            byte[] ha = new byte[256];
            r.seek(0);
            r.readFully(ha);
            Header a = Header.parse(ha, len);
            byte[] hb = null;
            Header b = null;
            if (len >= 4096 + 256) {
                hb = new byte[256];
                r.seek(4096);
                r.readFully(hb);
                b = Header.parse(hb, len);
            }
            if (a == null && b == null) {
                if (ha[0] != 'N' || ha[1] != 'X' || ha[2] != 'A' || ha[3] != 'R')
                    throw new IOException(f.getName() + " is not a .nx (no NXAR magic)");
                throw new IOException(f.getName() + " has no valid NX header slot");
            }
            if (a != null && b != null && a.generation == b.generation && !Arrays.equals(ha, hb))
                throw new IOException(f.getName() + ": header slots have identical generation but differ");

            List<Header> order = new ArrayList<>(2);
            if (a != null && b != null) {
                if (a.generation == b.generation) order.add(a);
                else if (Long.compareUnsigned(a.generation, b.generation) > 0) { order.add(a); order.add(b); }
                else { order.add(b); order.add(a); }
            } else order.add(a != null ? a : b);

            IOException last = null;
            for (Header h : order) {
                try {
                    return mountHeader(f, r, len, h, verify);
                } catch (IOException e) {
                    last = e;
                }
            }
            throw last;
        }
    }

    private static Nx mountHeader(File f, RandomAccessFile r, long len, Header h, boolean verify) throws IOException {
        byte[] cat = new byte[(int) h.catalogSize];
        r.seek(h.catalogOff);
        r.readFully(cat);
        if (verify && !Arrays.equals(sha256().digest(cat), h.catalogSha))
            throw new IOException(f.getName() + ": Catalog SHA-256 does not match (a damaged file)");

        if (cat[0] != 'N' || cat[1] != 'X' || cat[2] != 'C' || cat[3] != 'T' || u16(cat, 4) != 1 || u16(cat, 6) != 128
                || u64(cat, 8) != h.catalogSize || u32(cat, 16) != 0 || u32(cat, 44) != 0 || !isZero(cat, 112, 16))
            throw new IOException(f.getName() + ": bad Catalog header");

        long appCnt = u32(cat, 20), progCnt = u32(cat, 24), snapCnt = u32(cat, 28);
        long locCnt = u32(cat, 32), depCnt = u32(cat, 36), extCnt = u32(cat, 40);
        if (appCnt < 1 || appCnt > 4096 || progCnt > 16384 || snapCnt < 1 || snapCnt > 65536
                || locCnt > 16384 || depCnt > 32768 || extCnt > 4096)
            throw new IOException(f.getName() + ": Catalog counts out of bounds");

        long appOff = u64(cat, 48), progOff = u64(cat, 56), snapOff = u64(cat, 64);
        long locOff = u64(cat, 72), depOff = u64(cat, 80), extOff = u64(cat, 88);
        long cpoolOff = u64(cat, 96), cpoolSz = u64(cat, 104);
        long cur = 128;
        if (appOff != cur) throw new IOException(f.getName() + ": bad Catalog application table");
        cur = align8(cur + appCnt * 80);
        if ((progCnt == 0 ? progOff != 0 : progOff != cur)) throw new IOException(f.getName() + ": bad Catalog program table");
        if (progCnt > 0) cur = align8(cur + progCnt * 64);
        if (snapOff != cur) throw new IOException(f.getName() + ": bad Catalog snapshot table");
        cur = align8(cur + snapCnt * 120);
        if ((locCnt == 0 ? locOff != 0 : locOff != cur)) throw new IOException(f.getName() + ": bad Catalog localized table");
        if (locCnt > 0) cur = align8(cur + locCnt * 24);
        if ((depCnt == 0 ? depOff != 0 : depOff != cur)) throw new IOException(f.getName() + ": bad Catalog dependency table");
        if (depCnt > 0) cur = align8(cur + depCnt * 56);
        if ((extCnt == 0 ? extOff != 0 : extOff != cur)) throw new IOException(f.getName() + ": bad Catalog extension table");
        if (extCnt > 0) cur = align8(cur + extCnt * 32);
        long cpoolEnd = addChecked(cpoolOff, cpoolSz);
        if (cpoolOff != cur || cpoolEnd < 0 || cpoolEnd > h.catalogSize)
            throw new IOException(f.getName() + ": Catalog string pool out of bounds");

        long extCur = align8(cpoolEnd);
        for (int i = 0; i < extCnt; i++) {
            int o = (int) (extOff + i * 32L);
            long tid = u32(cat, o), fl = u32(cat, o + 4);
            int okind = u16(cat, o + 8), res0 = u16(cat, o + 10);
            long oidx = u32(cat, o + 12), doff = u64(cat, o + 16), dsz = u64(cat, o + 24);
            long dend = addChecked(doff, dsz);
            if (res0 != 0 || (fl & ~1L) != 0 || okind > 3) throw new IOException(f.getName() + ": invalid Catalog extension");
            boolean ownerOk = okind == 0 ? oidx == NONE : okind == 1 ? oidx < appCnt : okind == 2 ? oidx < progCnt : oidx < snapCnt;
            if (!ownerOk || doff < extCur || (doff & 7) != 0 || dend < 0 || dend > h.catalogSize)
                throw new IOException(f.getName() + ": Catalog extension payload out of bounds");
            extCur = align8(dend);
            if ((fl & 1L) != 0 && tid != 0x00010004L && tid != 0x00010005L && tid != 0x00010008L)
                throw new IOException(f.getName() + ": unknown required Catalog extension 0x" + Long.toHexString(tid));
        }

        if (h.mediaSize != 0) {
            byte[] mh = new byte[32];
            r.seek(h.mediaOff);
            r.readFully(mh);
            long recCnt = u32(mh, 16);
            if (mh[0] != 'N' || mh[1] != 'X' || mh[2] != 'M' || mh[3] != 'D' || u16(mh, 4) != 1 || u16(mh, 6) != 32
                    || u64(mh, 8) != h.mediaSize || recCnt < 1 || recCnt > 65535 || u16(mh, 20) != 64 || u16(mh, 22) != 0
                    || u32(mh, 24) != 32 || u32(mh, 28) != align16(32 + recCnt * 64))
                throw new IOException(f.getName() + ": invalid Media header");
        }

        if (verify) {
            if (!Arrays.equals(streamSha(r, h.manifestOff, h.manifestSize), h.manifestSha))
                throw new IOException(f.getName() + ": Manifest SHA-256 does not match (a damaged file)");
            MessageDigest ih = sha256();
            if (h.blockCount > 0) {
                streamInto(r, ih, h.blockIndexOff, h.blockCount * 24);
                streamInto(r, ih, h.integrityOff, h.integritySize);
            }
            if (!Arrays.equals(ih.digest(), h.indexSha))
                throw new IOException(f.getName() + ": block index SHA-256 does not match (a damaged file)");
        }

        byte[] mh = new byte[96];
        r.seek(h.manifestOff);
        r.readFully(mh);
        if (mh[0] != 'N' || mh[1] != 'X' || mh[2] != 'M' || mh[3] != 'F' || u16(mh, 4) != 1 || u16(mh, 6) != 96
                || u64(mh, 8) != h.manifestSize || !isZero(mh, 80, 16))
            throw new IOException(f.getName() + ": invalid Manifest header");
        long nd = u32(mh, 16), nf = u32(mh, 20), ne = u32(mh, 24), nref = u32(mh, 28);
        if (nd < 1 || nd > 16777216L || nf > 33554432L || ne > 50331648L)
            throw new IOException(f.getName() + ": Manifest counts out of bounds");
        long dirOff = u64(mh, 32), fileOff = u64(mh, 40), entOff = u64(mh, 48), refOff = u64(mh, 56);
        long mpoolOff = u64(mh, 64), mpoolSz = u64(mh, 72);
        if (mpoolSz > (2L << 30)) throw new IOException(f.getName() + ": Manifest path pool too large");
        long mc = 96;
        if (dirOff != mc) throw new IOException(f.getName() + ": bad Manifest directory table");
        mc = align8(mc + nd * 16);
        if ((nf == 0 ? fileOff != 0 : fileOff != mc)) throw new IOException(f.getName() + ": bad Manifest file table");
        if (nf > 0) mc = align8(mc + nf * 64);
        if ((ne == 0 ? entOff != 0 : entOff != mc)) throw new IOException(f.getName() + ": bad Manifest entry table");
        if (ne > 0) mc = align8(mc + ne * 24);
        if ((nref == 0 ? refOff != 0 : refOff != mc)) throw new IOException(f.getName() + ": bad Manifest block reference table");
        if (nref > 0) mc = align4(mc + nref * 4);
        long mpoolEnd = addChecked(mpoolOff, mpoolSz);
        if (mpoolOff != align8(mc) || mpoolEnd < 0 || mpoolEnd > h.manifestSize)
            throw new IOException(f.getName() + ": Manifest path pool out of bounds");

        long[] appIds = new long[(int) appCnt];
        for (int i = 0; i < appCnt; i++) appIds[i] = u64(cat, (int) (appOff + i * 80L));
        int[] progApp = new int[(int) progCnt];
        for (int i = 0; i < progCnt; i++) {
            int o = (int) (progOff + i * 64L);
            long ai = u32(cat, o + 8);
            if (ai >= appCnt) throw new IOException(f.getName() + ": program application index out of range");
            progApp[i] = (int) ai;
        }

        List<Pkg> pkgs = new ArrayList<>((int) snapCnt);
        for (int i = 0; i < snapCnt; i++) {
            int o = (int) (snapOff + i * 120L);
            long pidx = u32(cat, o);
            int stype = cat[o + 4] & 0xff, sflags = cat[o + 5] & 0xff;
            if (u16(cat, o + 6) != 0 || stype > 4 || (sflags & ~3) != 0)
                throw new IOException(f.getName() + ": invalid snapshot " + i);
            if (pidx != NONE && pidx >= progCnt)
                throw new IOException(f.getName() + ": snapshot program index out of range");
            long ver = u32(cat, o + 8), rootDir = u32(cat, o + 12);
            if (rootDir >= nd) throw new IOException(f.getName() + ": snapshot root directory out of range");

            int kind;
            long tid;
            if (stype == 0) {
                kind = BASE;
                tid = pidx != NONE ? appIds[progApp[(int) pidx]] : appIds[0];
            } else if (stype == 1 || stype == 3) {
                kind = UPDATE;
                tid = pidx != NONE ? appIds[progApp[(int) pidx]] : appIds[0];
            } else if (stype == 2 && (sflags & 2) != 0) {
                if (pidx != NONE || u64(cat, o + 40) != 0) throw new IOException(f.getName() + ": invalid ADDON_TITLE_ID snapshot");
                tid = u64(cat, o + 32);
                if (tid == 0) throw new IOException(f.getName() + ": zero ADDON_TITLE_ID");
                kind = ADDON;
            } else {
                kind = stype + 10;
                tid = pidx != NONE ? appIds[progApp[(int) pidx]] : appIds[0];
            }

            long exefsDir = NONE, romfsDir = NONE;
            byte[] rd = new byte[16];
            r.seek(h.manifestOff + dirOff + rootDir * 16L);
            r.readFully(rd);
            long rfirst = u32(rd, 0), rcnt = u32(rd, 4);
            if (u32(rd, 8) != 0 || u32(rd, 12) != 0 || rfirst + rcnt > ne)
                throw new IOException(f.getName() + ": snapshot root directory out of bounds");
            if (rcnt > 0 && rcnt <= 4096) {
                byte[] ents = new byte[(int) (rcnt * 24)];
                r.seek(h.manifestOff + entOff + rfirst * 24L);
                r.readFully(ents);
                for (int j = 0; j < rcnt; j++) {
                    int eo = j * 24;
                    if ((ents[eo + 16] & 0xff) != 1) continue;
                    long no = u64(ents, eo), nl = u32(ents, eo + 8), tidx = u32(ents, eo + 12);
                    if (no < mpoolOff || no + nl > mpoolEnd || nl > 16) continue;
                    byte[] nb = new byte[(int) nl];
                    r.seek(h.manifestOff + no);
                    r.readFully(nb);
                    String nm = new String(nb, StandardCharsets.UTF_8);
                    if (nm.equals("exefs")) exefsDir = tidx;
                    else if (nm.equals("romfs")) romfsDir = tidx;
                }
            }
            pkgs.add(new Pkg(i, tid, rootDir, exefsDir, romfsDir, ver, kind, sflags));
        }

        return new Nx(f, len, h, nd, nf, ne, nref, dirOff, fileOff, entOff, refOff, mpoolOff, mpoolSizeCompat(mpoolSz), pkgs);
    }

    private static long mpoolSizeCompat(long sz) { return sz; }

    @Override Node dir(long index) throws IOException {
        if (index == NONE) return null;
        if (index < 0 || index >= dirCount) throw new IOException(file.getName() + ": directory " + index + " past the table");
        try (RandomAccessFile r = new RandomAccessFile(file, "r")) {
            byte[] d = new byte[16];
            r.seek(hdr.manifestOff + dirTableOff + index * 16L);
            r.readFully(d);
            if (u32(d, 8) != 0 || u32(d, 12) != 0) throw new IOException(file.getName() + ": bad directory flags");
            return new Node(true, index, "", 0, 0, 0, 0, null, u32(d, 0), u32(d, 4));
        }
    }

    @Override List<Node> children(Node dir) throws IOException {
        if (!dir.dir || dir.childCount == 0) return Collections.emptyList();
        if (dir.firstChild + dir.childCount > entryCount || dir.childCount > Integer.MAX_VALUE / 24)
            throw new IOException(file.getName() + ": directory children past the table");
        List<Node> out = new ArrayList<>((int) dir.childCount);
        try (RandomAccessFile r = new RandomAccessFile(file, "r")) {
            byte[] ents = new byte[(int) (dir.childCount * 24)];
            r.seek(hdr.manifestOff + entryTableOff + dir.firstChild * 24L);
            r.readFully(ents);
            byte[] prevName = null;
            for (int i = 0; i < dir.childCount; i++) {
                int eo = i * 24;
                long no = u64(ents, eo), nl = u32(ents, eo + 8), tidx = u32(ents, eo + 12);
                int kind = ents[eo + 16] & 0xff, fl = ents[eo + 17] & 0xff;
                if (fl != 0 || u16(ents, eo + 18) != 0 || u32(ents, eo + 20) != 0 || (kind != 1 && kind != 2))
                    throw new IOException(file.getName() + ": invalid directory entry");
                if (nl < 1 || nl > 4096 || no < pathPoolOff || no + nl > pathPoolOff + pathPoolSize)
                    throw new IOException(file.getName() + ": entry name out of path pool");
                byte[] nb = new byte[(int) nl];
                r.seek(hdr.manifestOff + no);
                r.readFully(nb);
                if (prevName != null && Arrays.compareUnsigned(prevName, nb) >= 0)
                    throw new IOException(file.getName() + ": directory entries not sorted");
                prevName = nb;
                String name = new String(nb, StandardCharsets.UTF_8);
                if (name.indexOf('\0') >= 0 || name.indexOf('/') >= 0 || name.indexOf('\\') >= 0 || name.equals(".") || name.equals(".."))
                    throw new IOException(file.getName() + ": invalid entry name");
                if (kind == 1) {
                    if (tidx >= dirCount) throw new IOException(file.getName() + ": dir " + tidx + " out of range");
                    byte[] d = new byte[16];
                    r.seek(hdr.manifestOff + dirTableOff + tidx * 16L);
                    r.readFully(d);
                    if (u32(d, 8) != 0 || u32(d, 12) != 0) throw new IOException(file.getName() + ": bad directory flags");
                    out.add(new Node(true, tidx, name, 0, 0, 0, 0, null, u32(d, 0), u32(d, 4)));
                } else {
                    if (tidx >= fileCount) throw new IOException(file.getName() + ": file " + tidx + " out of range");
                    byte[] fb = new byte[64];
                    r.seek(hdr.manifestOff + fileTableOff + tidx * 64L);
                    r.readFully(fb);
                    long lsz = u64(fb, 0), fref = u32(fb, 8), bc = u32(fb, 12);
                    int cshift = fb[16] & 0xff, ffl = fb[17] & 0xff;
                    if (lsz < 0 || ffl != 0 || u16(fb, 18) != 0 || !isZero(fb, 52, 12) || cshift < 16 || cshift > 20)
                        throw new IOException(file.getName() + ": invalid file record");
                    long csz = 1L << cshift;
                    long expBc = lsz == 0 ? 0 : (lsz + csz - 1) >>> cshift;
                    if (bc != expBc || fref + bc > blockRefCount)
                        throw new IOException(file.getName() + ": file block references out of range");
                    out.add(new NxNode(tidx, name, lsz, Arrays.copyOfRange(fb, 20, 52), fref, (int) bc, cshift));
                }
            }
        }
        return out;
    }

    @Override byte[] read(Node f, long max) throws IOException {
        if (f.dir) throw new IOException(f.name + " is a directory");
        if (f.size == 0 || max == 0) return new byte[0];
        if (!(f instanceof NxNode)) throw new IOException(f.name + ": unexpected node type");
        NxNode nx = (NxNode) f;
        long want = max < 0 ? f.size : Math.min(f.size, max);
        if (want > Integer.MAX_VALUE - 16) throw new IOException(f.name + " is too large to read into memory");
        byte[] out = new byte[(int) want];
        int csz = 1 << nx.chunkShift;
        int needBlocks = (int) ((want + csz - 1) >>> nx.chunkShift);
        if (needBlocks > nx.blockCount) throw new IOException(f.name + ": block count mismatch");

        try (RandomAccessFile r = new RandomAccessFile(file, "r")) {
            byte[] refs = new byte[needBlocks * 4];
            r.seek(hdr.manifestOff + refTableOff + nx.firstBlockRef * 4L);
            r.readFully(refs);
            byte[] br = new byte[24];
            int outPos = 0;
            for (int b = 0; b < needBlocks; b++) {
                long bid = u32(refs, b * 4);
                if (bid >= hdr.blockCount) throw new IOException(file.getName() + ": block ID out of range");
                r.seek(hdr.blockIndexOff + bid * 24L);
                r.readFully(br);
                long boff = u64(br, 0);
                int bstored = (int) u32(br, 8), blogical = (int) u32(br, 12);
                int codec = br[16] & 0xff, bflags = br[17] & 0xff;
                long firstCheck = u32(br, 20);
                if (bflags != 0 || u16(br, 18) != 0 || codec > 1 || bstored <= 0 || bstored > MAX_BLOCK_STORED
                        || blogical <= 0 || blogical > MAX_BLOCK_LOGICAL || boff < hdr.dataOff
                        || (boff & 15) != 0 || boff + bstored > hdr.dataOff + hdr.dataSize)
                    throw new IOException(file.getName() + ": " + f.name + ": invalid block descriptor " + bid);

                int take = Math.min(blogical, (int) want - outPos);
                if (codec == 0) {
                    if (bstored != blogical) throw new IOException(file.getName() + ": " + f.name + ": RAW size mismatch");
                    int nchk = (take + CHECK_SIZE - 1) / CHECK_SIZE;
                    int rlen = Math.min(blogical, nchk * CHECK_SIZE);
                    byte[] raw = new byte[rlen];
                    r.seek(boff);
                    r.readFully(raw);
                    verifySlices(r, f.name, bid, firstCheck, raw, nchk);
                    System.arraycopy(raw, 0, out, outPos, take);
                } else {
                    byte[] z = new byte[bstored];
                    r.seek(boff);
                    r.readFully(z);
                    byte[] dec = Zstd.decode(z, 0, z.length, -1);
                    if (dec.length != blogical)
                        throw new IOException(file.getName() + ": " + f.name + ": decoded to " + dec.length + " bytes, expected " + blogical);
                    int nchk = (blogical + CHECK_SIZE - 1) / CHECK_SIZE;
                    verifySlices(r, f.name, bid, firstCheck, dec, nchk);
                    System.arraycopy(dec, 0, out, outPos, take);
                }
                outPos += take;
            }
        } catch (IOException e) {
            throw new IOException(file.getName() + ": " + f.name + ": " + e.getMessage(), e);
        }
        if (max < 0 && !Arrays.equals(sha256().digest(out), f.sha256))
            throw new IOException(file.getName() + ": " + f.name + " does not match its SHA-256: the file is damaged");
        return out;
    }

    private void verifySlices(RandomAccessFile r, String fname, long bid, long firstCheck, byte[] buf, int nchk) throws IOException {
        if ((firstCheck + nchk) * 4 > hdr.integritySize) throw new IOException(fname + ": integrity index out of bounds");
        byte[] cb = new byte[nchk * 4];
        r.seek(hdr.integrityOff + firstCheck * 4L);
        r.readFully(cb);
        for (int s = 0; s < nchk; s++) {
            int off = s * CHECK_SIZE;
            int len = Math.min(CHECK_SIZE, buf.length - off);
            int got = crc32c(buf, off, len), want = (int) u32(cb, s * 4);
            if (got != want)
                throw new IOException(fname + ": block " + bid + " slice " + s + " CRC-32C mismatch");
        }
    }

    private static byte[] streamSha(RandomAccessFile r, long off, long size) throws IOException {
        MessageDigest h = sha256();
        streamInto(r, h, off, size);
        return h.digest();
    }

    private static void streamInto(RandomAccessFile r, MessageDigest h, long off, long size) throws IOException {
        r.seek(off);
        byte[] buf = new byte[1 << 16];
        long left = size;
        while (left > 0) {
            int n = r.read(buf, 0, (int) Math.min(buf.length, left));
            if (n < 0) throw new IOException("unexpected EOF while hashing archive region");
            h.update(buf, 0, n);
            left -= n;
        }
    }

    private static boolean isZero(byte[] b, int off, int len) {
        for (int i = 0; i < len; i++) if (b[off + i] != 0) return false;
        return true;
    }

    private static long align4(long v) { return (v + 3L) & ~3L; }
    private static long align8(long v) { return (v + 7L) & ~7L; }
    private static long align16(long v) { return (v + 15L) & ~15L; }
    private static long addChecked(long a, long b) { return a < 0 || b < 0 || a > Long.MAX_VALUE - b ? -1 : a + b; }
    private static long mulChecked(long a, long b) { return a < 0 || b < 0 || (a > 0 && b > Long.MAX_VALUE / a) ? -1 : a * b; }
}
