// Networking (out of scope: online play/friends/store/cloud saves): all hooked up as "network unavailable", the game takes the offline branch.
// Cuts: nifm reports no network; socket only hooks up Initialize (returns success, creates no session), the rest of socket/ssl/friends are hooked up as failures when they come into use.
#include "hle.h"
#include <stdatomic.h>

HLE_FORCE("_ZN2nn6socket10InitializeERKNS0_6ConfigE", "connects to the bsd socket service; offline, creates no session (cut)", Result, h_socket_Initialize, const void *cfg) { (void)cfg; return RES_OK; }
HLE_FORCE("_ZN2nn4nifm10InitializeEv", "connects to the nifm network management service; offline (cut)", Result, h_nifm_Initialize, void) { return RES_OK; }
// The request is asynchronous on Horizon: SubmitNetworkRequest only hands it to the nifm service, and IsNetworkRequestOnHold stays true until the service has evaluated it, so a query
// right after the submission finds it on hold. Pikmin 4 (Unreal Engine 4.26, main+0x203f3b0) relies on that: it starts its BCAT thread, whose Run (main+0x203f510) sets its status
// to 2, submits, waits while on hold and sets 6 (done, offline), and only then stores status 1 itself; answered at once, a thread that ran first finished before that store, the 1 stayed,
// and the boot state machine (main+0x1f2f764, state 7) waited for the status forever at the black loading screen (about 1 boot in 6, 3 in 4 with four at once). Pikmin 4 sleeps 10 ms
// between its queries, so a request that is on hold at the first query is all it needs.
// Each query is an IPC to the nifm service on Horizon, which evaluates the request as it serves them (offline, a rejection), so the request is on hold for the first query after
// the submission and settled at the next. The hold used to be 50 ms of host time: Hades' NetworkPollThread (main+0x27aaa0) polls with SleepThread(0) and cancels after 2,000
// queries, 2,000 IPCs on the Switch but 0.1 ms here, so it cancelled and submitted again forever (1.17 billion queries a minute, a host core). Its stamp, now_ns() | 1, also read
// as settled at once whenever the next query came within the same 42 ns clock tick (30% of Hades' submissions)
static _Atomic int g_nifm_hold;   // queries the pending request still answers "on hold" (0 = none, or settled)
HLE_FORCE("_ZN2nn4nifm20SubmitNetworkRequestEv", "offline: the request is on hold for the first query, then settles as unavailable", void, h_nifm_Submit, void) { atomic_store(&g_nifm_hold, 1); }
HLE_FORCE("_ZN2nn4nifm20CancelNetworkRequestEv", "offline: drops the pending request", void, h_nifm_Cancel, void) { atomic_store(&g_nifm_hold, 0); }
HLE_FORCE("_ZN2nn4nifm22IsNetworkRequestOnHoldEv", "offline: on hold for the first query after the submission", int, h_nifm_OnHold, void) {
    int h = atomic_load(&g_nifm_hold);
    while (h > 0 && !atomic_compare_exchange_weak(&g_nifm_hold, &h, h - 1)) {}
    return h > 0;
}
HLE_FORCE("_ZN2nn4nifm18IsNetworkAvailableEv", "offline: the network is always unavailable", int, h_nifm_Available, void) { return 0; }
// Returns bool (`and w0, w0, #1`, Metroid Dread's sdk+0x336210): whether the system's network screen resolved the error
HLE_FORCE("_ZN2nn4nifm31HandleNetworkRequestErrorResultEv", "offline: not resolved, shows no system network screen", unsigned, h_nifm_HandleErr, void) { return 0; }
HLE_FORCE("_ZN2nn4nifm26GetCurrentPrimaryIpAddressEPNS_6socket6InAddrE", "offline: no IP", Result, h_nifm_GetIp, uint32_t *a) { if (a) *a = 0; return RES_FAIL; }
HLE_FORCE("_ZN2nn4nifm22GetCurrentProxySettingEPNS0_12ProxySettingE", "offline: no proxy setting", Result, h_nifm_GetProxy, void *p) { (void)p; return RES_FAIL; }
HLE_FORCE("nnsslInitialize", "connects to the ssl service; the network library's global init needs it to succeed, offline creates no session (cut)", Result, h_ssl_Initialize, void) { return RES_OK; }
HLE_FORCE("nnsslFinalize", "pairs with nnsslInitialize (cut)", Result, h_ssl_Finalize, void) { return RES_OK; }

// ---------- socket: offline, all calls that contact the service return -1 in BSD convention, error code ENETDOWN ----------
// Not cut: Inet* / nnsocketInet* byte order and address text conversion (pure computation).
#define SK_ENETDOWN 50                  // Horizon sockets use FreeBSD error codes
static __thread int t_skerr = SK_ENETDOWN;
static int sk_fail(void) { t_skerr = SK_ENETDOWN; return -1; }
#define SKF(m, cname, ...) HLE_FORCE(m, "bsd socket service (cut): returns -1/ENETDOWN", int, cname, __VA_ARGS__)
// Poll / Select with nothing to wait on: the host creates no socket, so a game's descriptors are all -1, or it passes none. POSIX and Horizon's bsd service ignore negative
// descriptors and wait out the timeout, returning 0. Failing at once made Dark Souls Remastered's network thread ("NexusRevolution Socket", main+0x92c7c4) spin a core at 100%
// on Poll(fds, 0, 33 ms), 11,000 calls a second. An endless wait on nothing never returns on hardware; here it returns 0 after 100 ms: a thread in a host hook pauses for a garbage
// collector (IL2CPP's, hle_thread.c) only once the hook returns, and the caller can still see a shutdown
void dm_guest_sleep_ns(int64_t ns);
static int sk_idle(int64_t ms) { dm_guest_sleep_ns((ms < 0 ? 100 : ms) * 1000000); return 0; }
typedef struct { int32_t fd; int16_t events, revents; } sk_pollfd;

// ---------- eventfd: local, not network. Horizon's bsd service gives eventfds whether or not there is a network, and gRPC (NintendoSDK NPLN, the Switch 2's online library) needs one:
// grpc_wakeup_fd_global_init checks for it with EventFd(0, 0) + Close, and without it the only compiled event engine ("poll") returns null and grpc_event_engine_init aborts
// (Ocarina of Time main e0f3ccfa: check main+0x21f3bd0, abort main+0x21f0c7c, the game's online library starting at boot). So the host keeps real counters: EventFd / EventFdRead /
// EventFdWrite with FreeBSD semantics (a 64-bit counter; a read takes it all, or 1 in semaphore mode; an empty non-blocking read and an overflowing non-blocking write fail with EAGAIN),
// Close, Fcntl's flags and Poll's readiness on them, and a blocking read, write or Poll waits for another thread's write or close. Every network socket still fails with ENETDOWN.
// Descriptors are EFD_BASE.. (non-zero: gRPC never closes fd 0; under 1024, the FdSet size), the only descriptors the host ever hands out.
// Errors on these paths also go to the guest's libc errno (sdk __errno_location, Linux numbering: gRPC compares EINTR 4 / EAGAIN 11 there, main+0x21f3af8; the SDK's own
// detail::EventFd writes it, sdk+0x192cfc): a stale errno of 4 would make gRPC's consume and kick retry forever. Flag values are FreeBSD's (Semaphore 1, NonBlock 4; not checked
// against the bsd service; gRPC's create passes 6 for EFD_NONBLOCK | EFD_CLOEXEC, non-blocking under either reading of bits 1 and 2)
enum { EFD_BASE = 512, EFD_MAX = 256, EFD_SEMAPHORE = 1, EFD_NONBLOCK = 4 };
enum { SK_EINTR = 4, SK_EBADF = 9, SK_EAGAIN = 11, SK_EINVAL = 22, SK_EMFILE = 24 };   // the guest libc's (Linux) numbers
enum { SK_F_GETFD = 1, SK_F_SETFD = 2, SK_F_GETFL = 3, SK_F_SETFL = 4, SK_O_RDWR = 2, SK_O_NONBLOCK = 0x800 };
enum { SK_POLLIN = 0x1, SK_POLLOUT = 0x4, SK_POLLNVAL = 0x20, SK_POLLRDNORM = 0x40, SK_POLLWRNORM = 0x100 };
#define EFD_CNT_MAX 0xfffffffffffffffeull
static struct { uint8_t used, nonblock, sem; uint32_t gen; uint64_t cnt; } g_efd[EFD_MAX];
static pthread_mutex_t g_efd_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_efd_cv = PTHREAD_COND_INITIALIZER;   // broadcast on every count change and close
static int efd_slot(int fd) { unsigned i = (unsigned)(fd - EFD_BASE); return i < EFD_MAX && g_efd[i].used ? (int)i : -1; }   // g_efd_mu held
static int efd_is(int fd) { pthread_mutex_lock(&g_efd_mu); int r = efd_slot(fd) >= 0; pthread_mutex_unlock(&g_efd_mu); return r; }
static int efd_err(int e) {
    t_skerr = e;
    static int *(*loc)(void); static _Atomic int looked;
    if (!atomic_load(&looked)) { loc = (int *(*)(void))guest_sym("__errno_location"); atomic_store(&looked, 1); }
    if (loc) *loc() = e;
    return -1;
}
static uint64_t efd_clock(void) { return dm_detclock_enabled() ? guest_ns() : now_ns(); }
// One wait slice with g_efd_mu held: at most 100 ms (a thread in a host hook is seen by a garbage collector's pause or a shutdown only once the hook returns; 1 ms under DM_DETCLOCK,
// whose deadlines run on the guest clock), the guest core released while it waits (hle.h), and a suspended application parked outside the lock
static void efd_wait(uint64_t ns) {
    if (dm_guest_suspended()) { pthread_mutex_unlock(&g_efd_mu); dm_guest_suspend_wait(); pthread_mutex_lock(&g_efd_mu); return; }
    uint64_t cap = dm_detclock_enabled() ? 1000000 : 100000000;
    if (ns > cap) ns = cap;
    struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl);
    dl.tv_sec += (time_t)(ns / 1000000000); dl.tv_nsec += (long)(ns % 1000000000);
    if (dl.tv_nsec >= 1000000000) { dl.tv_sec++; dl.tv_nsec -= 1000000000; }
    dm_thread_native_wait(1);
    pthread_cond_timedwait(&g_efd_cv, &g_efd_mu, &dl);
    dm_thread_native_wait(0);
}
static int efd_create(uint64_t v, int flags) {
    pthread_mutex_lock(&g_efd_mu);
    for (int i = 0; i < EFD_MAX; i++) if (!g_efd[i].used) {
        g_efd[i].used = 1; g_efd[i].gen++; g_efd[i].cnt = v;
        g_efd[i].nonblock = !!(flags & EFD_NONBLOCK); g_efd[i].sem = !!(flags & EFD_SEMAPHORE);
        pthread_mutex_unlock(&g_efd_mu);
        static _Atomic int once;
        if (!atomic_exchange(&once, 1)) LOG("socket: eventfd %d made (flags %#x; local, the network stays down)", EFD_BASE + i, (unsigned)flags);
        return EFD_BASE + i;
    }
    pthread_mutex_unlock(&g_efd_mu);
    return efd_err(SK_EMFILE);
}
static int efd_read(int fd, uint64_t *out) {
    if (!out) return -1;   // as the SDK's EventFdRead: a null value pointer fails without a call
    pthread_mutex_lock(&g_efd_mu);
    for (;;) {
        int i = efd_slot(fd);
        if (i < 0) { pthread_mutex_unlock(&g_efd_mu); return efd_err(SK_EBADF); }
        if (g_efd[i].cnt) {
            uint64_t v = g_efd[i].sem ? 1 : g_efd[i].cnt;
            g_efd[i].cnt -= v;
            pthread_cond_broadcast(&g_efd_cv);
            pthread_mutex_unlock(&g_efd_mu);
            *out = v; return 0;
        }
        if (g_efd[i].nonblock) { pthread_mutex_unlock(&g_efd_mu); return efd_err(SK_EAGAIN); }
        efd_wait(100000000);
    }
}
static int efd_write(int fd, uint64_t v) {
    if (v == ~0ull) return efd_err(SK_EINVAL);
    pthread_mutex_lock(&g_efd_mu);
    for (;;) {
        int i = efd_slot(fd);
        if (i < 0) { pthread_mutex_unlock(&g_efd_mu); return efd_err(SK_EBADF); }
        if (g_efd[i].cnt <= EFD_CNT_MAX - v) {
            g_efd[i].cnt += v;
            pthread_cond_broadcast(&g_efd_cv);
            pthread_mutex_unlock(&g_efd_mu);
            return 0;
        }
        if (g_efd[i].nonblock) { pthread_mutex_unlock(&g_efd_mu); return efd_err(SK_EAGAIN); }
        efd_wait(100000000);
    }
}
static int sk_close(int fd) {   // an eventfd is freed; any other descriptor was never made by the host
    pthread_mutex_lock(&g_efd_mu);
    int i = efd_slot(fd);
    if (i >= 0) { g_efd[i].used = 0; g_efd[i].cnt = 0; pthread_cond_broadcast(&g_efd_cv); }
    pthread_mutex_unlock(&g_efd_mu);
    return i >= 0 ? 0 : sk_fail();
}
int dm_sk_fcntl(int fd, int cmd, int arg) {   // also hle_offline.c's int-argument form
    pthread_mutex_lock(&g_efd_mu);
    int i = efd_slot(fd), r = 0;
    if (i < 0) { pthread_mutex_unlock(&g_efd_mu); return sk_fail(); }
    switch (cmd) {
    case SK_F_GETFL: r = SK_O_RDWR | (g_efd[i].nonblock ? SK_O_NONBLOCK : 0); break;
    case SK_F_SETFL: g_efd[i].nonblock = !!(arg & SK_O_NONBLOCK); break;
    case SK_F_GETFD: case SK_F_SETFD: break;
    default: r = -2;
    }
    pthread_mutex_unlock(&g_efd_mu);
    return r == -2 ? efd_err(SK_EINVAL) : r;
}
// Poll with eventfds among the descriptors: each one's readiness from its counter (readable while it is not 0, writable while one more can be added), a descriptor closed meanwhile
// is POLLNVAL; nothing ready waits for a write or close until the timeout (0: at once; negative: up to 100 ms, then 0, as with nothing to wait on)
static int efd_poll(sk_pollfd *p, uint64_t n, int ms) {
    uint64_t start = efd_clock(), lim = ms < 0 ? 100000000ull : (uint64_t)ms * 1000000;
    pthread_mutex_lock(&g_efd_mu);
    for (;;) {
        int ready = 0;
        for (uint64_t k = 0; k < n; k++) {
            p[k].revents = 0;
            if (p[k].fd < 0) continue;
            int i = efd_slot(p[k].fd);
            if (i < 0) p[k].revents = SK_POLLNVAL;
            else {
                if (g_efd[i].cnt) p[k].revents |= p[k].events & (SK_POLLIN | SK_POLLRDNORM);
                if (g_efd[i].cnt < EFD_CNT_MAX) p[k].revents |= p[k].events & (SK_POLLOUT | SK_POLLWRNORM);
            }
            if (p[k].revents) ready++;
        }
        uint64_t spent = efd_clock() - start;
        if (ready || spent >= lim) { pthread_mutex_unlock(&g_efd_mu); return ready; }
        efd_wait(lim - spent);
    }
}
int dm_sk_poll(void *fds, uint64_t n, int ms) {   // also hle_offline.c's plain pollfd form
    sk_pollfd *p = fds; int efds = 0;
    for (uint64_t i = 0; i < n; i++) if (p[i].fd >= 0) { if (!efd_is(p[i].fd)) return sk_fail(); efds++; }
    if (efds) return efd_poll(p, n, ms);
    for (uint64_t i = 0; i < n; i++) p[i].revents = 0;
    return sk_idle(ms);
}
int dm_sk_select(int nfds, const uint64_t *r, const uint64_t *w, const uint64_t *e, const int64_t *tv) {   // fd_set: 1024 bits; timeval: {i64 seconds, i64 microseconds}
    const uint64_t *sets[3] = {r, w, e};
    for (int k = 0; k < 3; k++) for (int fd = 0; sets[k] && fd < nfds && fd < 1024; fd++) if (sets[k][fd >> 6] >> (fd & 63) & 1) return sk_fail();
    return sk_idle(tv ? tv[0] * 1000 + tv[1] / 1000 : -1);
}
SKF("_ZN2nn6socket6SocketENS0_6FamilyENS0_4TypeENS0_8ProtocolE", h_sk_Socket, int a, int b, int c) { (void)a; (void)b; (void)c; return sk_fail(); }
SKF("_ZN2nn6socket4BindEiPKNS0_8SockAddrEj", h_sk_Bind, int s, const void *a, unsigned n) { (void)s; (void)a; (void)n; return sk_fail(); }
SKF("_ZN2nn6socket7ConnectEiPKNS0_8SockAddrEj", h_sk_Connect, int s, const void *a, unsigned n) { (void)s; (void)a; (void)n; return sk_fail(); }
SKF("_ZN2nn6socket6ListenEii", h_sk_Listen, int s, int n) { (void)s; (void)n; return sk_fail(); }
SKF("_ZN2nn6socket6AcceptEiPNS0_8SockAddrEPj", h_sk_Accept, int s, void *a, unsigned *n) { (void)s; (void)a; (void)n; return sk_fail(); }
SKF("_ZN2nn6socket5CloseEi", h_sk_Close, int s) { return sk_close(s); }
SKF("_ZN2nn6socket11GetPeerNameEiPNS0_8SockAddrEPj", h_sk_GetPeerName, int s, void *a, unsigned *n) { (void)s; (void)a; (void)n; return sk_fail(); }
SKF("_ZN2nn6socket11GetSockNameEiPNS0_8SockAddrEPj", h_sk_GetSockName, int s, void *a, unsigned *n) { (void)s; (void)a; (void)n; return sk_fail(); }
SKF("_ZN2nn6socket10GetSockOptEiNS0_5LevelENS0_6OptionEPvPj", h_sk_GetSockOpt, int s, int l, int o, void *v, unsigned *n) { (void)s; (void)l; (void)o; (void)v; (void)n; return sk_fail(); }
SKF("_ZN2nn6socket10SetSockOptEiNS0_5LevelENS0_6OptionEPKvj", h_sk_SetSockOpt, int s, int l, int o, const void *v, unsigned n) { (void)s; (void)l; (void)o; (void)v; (void)n; return sk_fail(); }
SKF("_ZN2nn6socket5FcntlEiNS0_12FcntlCommandEz", h_sk_Fcntl, int s, int c, int a) { return dm_sk_fcntl(s, c, a); }   // the variadic argument is in w2 (the Switch ABI passes variadics in registers)
SKF("_ZN2nn6socket4PollEPNS0_6PollFdEmi", h_sk_Poll, void *f, unsigned long n, int t) { return dm_sk_poll(f, n, t); }
SKF("_ZN2nn6socket6SelectEiPNS0_5FdSetES2_S2_PNS0_7TimeValE", h_sk_Select, int n, void *r, void *w, void *e, void *t) { return dm_sk_select(n, r, w, e, t); }
SKF("_ZN2nn6socket7EventFdEmNS0_12EventFdFlagsE", h_sk_EventFd, unsigned long v, int f) { return efd_create(v, f); }
SKF("_ZN2nn6socket11EventFdReadEiPm", h_sk_EventFdRead, int s, unsigned long *v) { return efd_read(s, (uint64_t *)v); }
SKF("_ZN2nn6socket12EventFdWriteEim", h_sk_EventFdWrite, int s, unsigned long v) { return efd_write(s, v); }
SKF("_ZN2nn6socket17ResolverGetOptionEPNS0_14ResolverOptionENS0_17ResolverOptionKeyE", h_sk_ResGet, void *o, int k) { (void)o; (void)k; return sk_fail(); }
SKF("_ZN2nn6socket17ResolverSetOptionERKNS0_14ResolverOptionE", h_sk_ResSet, const void *o) { (void)o; return sk_fail(); }
HLE_FORCE("_ZN2nn6socket4RecvEiPvmNS0_7MsgFlagE", "bsd socket service (cut)", long, h_sk_Recv, int s, void *b, unsigned long n, int f) { (void)s; (void)b; (void)n; (void)f; return sk_fail(); }
HLE_FORCE("_ZN2nn6socket8RecvFromEiPvmNS0_7MsgFlagEPNS0_8SockAddrEPj", "bsd socket service (cut)", long, h_sk_RecvFrom, int s, void *b, unsigned long n, int f, void *a, unsigned *al) { (void)s; (void)b; (void)n; (void)f; (void)a; (void)al; return sk_fail(); }
HLE_FORCE("_ZN2nn6socket4SendEiPKvmNS0_7MsgFlagE", "bsd socket service (cut)", long, h_sk_Send, int s, const void *b, unsigned long n, int f) { (void)s; (void)b; (void)n; (void)f; return sk_fail(); }
HLE_FORCE("_ZN2nn6socket11GetAddrInfoEPKcS2_PKNS0_8AddrInfoEPPS3_", "name resolution (cut): EAI_FAIL", int, h_sk_GetAddrInfo, const char *h, const char *s, const void *hint, void **res) {
    (void)h; (void)s; (void)hint; if (res) *res = NULL; return 4;   // EAI_FAIL
}
HLE_FORCE("_ZN2nn6socket12FreeAddrInfoEPNS0_8AddrInfoE", "pairs with GetAddrInfo (cut)", void, h_sk_FreeAddrInfo, void *ai) { (void)ai; }
HLE_FORCE("_ZN2nn6socket16GetHostEntByNameEPKcPKNS0_14ResolverOptionEm", "name resolution (cut): NULL", void *, h_sk_GetHostEnt, const char *n, const void *o, unsigned long c) { (void)n; (void)o; (void)c; t_skerr = SK_ENETDOWN; return NULL; }
HLE_FORCE("_ZN2nn6socket12GetLastErrorEv", "socket error code kept by the host", int, h_sk_GetLastError, void) { return t_skerr; }
// C interface
SKF("nnsocketSocket", h_nsk_Socket, int a, int b, int c) { (void)a; (void)b; (void)c; return sk_fail(); }
SKF("nnsocketBind", h_nsk_Bind, int s, const void *a, unsigned n) { (void)s; (void)a; (void)n; return sk_fail(); }
SKF("nnsocketConnect", h_nsk_Connect, int s, const void *a, unsigned n) { (void)s; (void)a; (void)n; return sk_fail(); }
SKF("nnsocketClose", h_nsk_Close, int s) { return sk_close(s); }
SKF("nnsocketGetPeerName", h_nsk_GetPeerName, int s, void *a, unsigned *n) { (void)s; (void)a; (void)n; return sk_fail(); }
SKF("nnsocketGetSockName", h_nsk_GetSockName, int s, void *a, unsigned *n) { (void)s; (void)a; (void)n; return sk_fail(); }
SKF("nnsocketGetSockOpt", h_nsk_GetSockOpt, int s, int l, int o, void *v, unsigned *n) { (void)s; (void)l; (void)o; (void)v; (void)n; return sk_fail(); }
SKF("nnsocketSetSockOpt", h_nsk_SetSockOpt, int s, int l, int o, const void *v, unsigned n) { (void)s; (void)l; (void)o; (void)v; (void)n; return sk_fail(); }
SKF("nnsocketFcntl", h_nsk_Fcntl, int s, int c, int a) { return dm_sk_fcntl(s, c, a); }
SKF("nnsocketPoll", h_nsk_Poll, void *f, unsigned long n, int t) { return dm_sk_poll(f, n, t); }
SKF("nnsocketResolverGetOption", h_nsk_ResGet, void *o, int k) { (void)o; (void)k; return sk_fail(); }
SKF("nnsocketResolverSetOption", h_nsk_ResSet, const void *o) { (void)o; return sk_fail(); }
HLE_FORCE("nnsocketRecv", "bsd socket service (cut)", long, h_nsk_Recv, int s, void *b, unsigned long n, int f) { (void)s; (void)b; (void)n; (void)f; return sk_fail(); }
HLE_FORCE("nnsocketSend", "bsd socket service (cut)", long, h_nsk_Send, int s, const void *b, unsigned long n, int f) { (void)s; (void)b; (void)n; (void)f; return sk_fail(); }
HLE_FORCE("nnsocketGetHostByNameWithOptions", "name resolution (cut): NULL", void *, h_nsk_GetHost, const char *n, const void *o, unsigned long c) { (void)n; (void)o; (void)c; t_skerr = SK_ENETDOWN; return NULL; }
HLE_FORCE("nnsocketGetLastErrno", "socket error code kept by the host", int, h_nsk_GetErrno, void) { return t_skerr; }
HLE_FORCE("nnsocketSetLastErrno", "socket error code kept by the host", void, h_nsk_SetErrno, int e) { t_skerr = e; }

// ---------- ssl: offline, Initialize succeeds (the network library's global initialization depends on it), everything else that contacts the service fails ----------
// Not cut: Context / Connection construction and destruction, Context::GetContextId (only read and write game objects).
#define SSF(m, cname, ...) HLE_FORCE(m, "ssl service (cut): fails", Result, cname, __VA_ARGS__)
HLE_FORCE("_ZN2nn3ssl10InitializeEv", "same as nnsslInitialize: succeeds, creates no session (cut)", Result, h_ssl_InitCpp, void) { return RES_OK; }
HLE_FORCE("_ZN2nn3ssl8FinalizeEv", "pairs with Initialize (cut)", Result, h_ssl_FinCpp, void) { return RES_OK; }
SSF("_ZN2nn3ssl7Context6CreateENS1_10SslVersionE", h_ssl_CtxCreate, void *self, int v) { (void)self; (void)v; return RES_FAIL; }
SSF("_ZN2nn3ssl7Context7DestroyEv", h_ssl_CtxDestroy, void *self) { (void)self; return RES_FAIL; }
SSF("_ZN2nn3ssl7Context19RegisterInternalPkiEPmNS1_11InternalPkiE", h_ssl_CtxRegPki, void *self, uint64_t *id, int p) { (void)self; (void)id; (void)p; return RES_FAIL; }
SSF("_ZN2nn3ssl7Context9RemovePkiEm", h_ssl_CtxRemovePki, void *self, uint64_t id) { (void)self; (void)id; return RES_FAIL; }
SSF("_ZN2nn3ssl14BuiltInManager28GetBuiltInCertificateBufSizeEPjPNS0_15CaCertificateIdEj", h_ssl_BiBufSize, uint32_t *o, void *ids, unsigned n) { (void)ids; (void)n; if (o) *o = 0; return RES_FAIL; }
SSF("_ZN2nn3ssl14BuiltInManager22GetBuiltInCertificatesEPPNS1_22BuiltInCertificateInfoEPhjPNS0_15CaCertificateIdEj", h_ssl_BiCerts, void **o, uint8_t *b, unsigned bs, void *ids, unsigned n) { (void)o; (void)b; (void)bs; (void)ids; (void)n; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection6CreateEPNS0_7ContextE", h_ssl_CnCreate, void *self, void *ctx) { (void)self; (void)ctx; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection7DestroyEv", h_ssl_CnDestroy, void *self) { (void)self; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection11DoHandshakeEv", h_ssl_CnHandshake, void *self) { (void)self; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection11SetHostNameEPKcj", h_ssl_CnHost, void *self, const char *h, unsigned n) { (void)self; (void)h; (void)n; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection9SetIoModeENS1_6IoModeE", h_ssl_CnIoMode, void *self, int m) { (void)self; (void)m; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection19SetSessionCacheModeENS1_16SessionCacheModeE", h_ssl_CnCache, void *self, int m) { (void)self; (void)m; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection19SetSocketDescriptorEi", h_ssl_CnSock, void *self, int s) { (void)self; (void)s; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection12GetLastErrorEPNS_6ResultE", h_ssl_CnLastErr, void *self, uint32_t *r) { (void)self; if (r) *r = RES_FAIL; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection18GetVerifyCertErrorEPNS_6ResultE", h_ssl_CnVerifyErr, void *self, uint32_t *r) { (void)self; if (r) *r = RES_FAIL; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection4PeekEPcPij", h_ssl_CnPeek, void *self, char *b, int *o, unsigned n) { (void)self; (void)b; (void)n; if (o) *o = 0; return RES_FAIL; }
SSF("_ZN2nn3ssl10Connection4PollEPNS1_9PollEventES3_j", h_ssl_CnPoll, void *self, void *o, void *i, unsigned t) { (void)self; (void)o; (void)i; (void)t; return RES_FAIL; }
HLE_FORCE("_ZN2nn3ssl10Connection4ReadEPcj", "ssl service (cut): read returns -1", int, h_ssl_CnRead, void *self, char *b, unsigned n) { (void)self; (void)b; (void)n; return -1; }
HLE_FORCE("_ZN2nn3ssl10Connection5WriteEPKcj", "ssl service (cut): write returns -1", int, h_ssl_CnWrite, void *self, const char *b, unsigned n) { (void)self; (void)b; (void)n; return -1; }
// C interface: all return nnResult
#define SSC(m, cname) HLE_FORCE(m, "ssl service (cut): fails", Result, cname, void) { return RES_FAIL; }
SSC("nnsslConnectionCreate", h_nssl_Create) SSC("nnsslConnectionDestroy", h_nssl_Destroy)
SSC("nnsslConnectionDoHandshakeWithCertBuffer", h_nssl_Handshake) SSC("nnsslConnectionGetConnectionId", h_nssl_GetId)
SSC("nnsslConnectionGetNextAlpnProto", h_nssl_GetAlpn) SSC("nnsslConnectionGetVerifyCertError", h_nssl_VerifyErr)
SSC("nnsslConnectionPeek", h_nssl_Peek) SSC("nnsslConnectionPending", h_nssl_Pending) SSC("nnsslConnectionRead", h_nssl_Read)
SSC("nnsslConnectionSetHostName", h_nssl_SetHost) SSC("nnsslConnectionSetIoMode", h_nssl_SetIo)
SSC("nnsslConnectionSetNextAlpnProto", h_nssl_SetAlpn) SSC("nnsslConnectionSetOption", h_nssl_SetOpt)
SSC("nnsslConnectionSetSessionCacheMode", h_nssl_SetCache) SSC("nnsslConnectionSetSocketDescriptor", h_nssl_SetSock)
SSC("nnsslConnectionSetVerifyOption", h_nssl_SetVerify) SSC("nnsslConnectionWrite", h_nssl_Write)
SSC("nnsslContextDestroy", h_nssl_CtxDestroy) SSC("nnsslContextGetContextId", h_nssl_CtxGetId)
