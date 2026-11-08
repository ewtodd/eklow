// LD_PRELOAD shim: logs every Level Zero device/shared/host allocation and free of the process, with the size and the caller's
// return addresses (resolve with addr2line -f -e <binary> after subtracting the load base printed as "base"), to $EKLOW_ZETRACE
// (default stderr). The driver's own allocations are not seen; the DRM fdinfo poll (tools/alloctrace/poll.py) has those.
//   EKLOW_ZETRACE=/path LD_PRELOAD=$PWD/libzetrace.so ./build/eklow-server ...
//   kill -USR1 <pid> writes a "mark" line (the phase of the workload), so a log can be cut by stage.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int (*alloc_fn)(void*, const void*, size_t, size_t, void*, void**);
typedef int (*alloc_host_fn)(void*, const void*, size_t, size_t, void**);
typedef int (*free_fn)(void*, void*);

static FILE* out;
static int marks;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static void on_usr1(int s) { (void)s; ++marks; }
static void init(void) {
  if (out) return;
  const char* p = getenv("EKLOW_ZETRACE");
  out = p ? fopen(p, "a") : stderr;
  if (!out) out = stderr;
  signal(SIGUSR1, on_usr1);
}
static int last_mark;
static void log_call(const char* what, size_t size, void* ptr) {
  init();
  void* bt[48];
  const int n = backtrace(bt, 48);
  if (marks != last_mark) { fprintf(out, "mark %d %.3f\n", marks, now()); last_mark = marks; }
  fprintf(out, "%s %.3f %zu %p", what, now(), size, ptr);
  // Only the frames in the executable, as "offset-in-file"; addr2line -f -e <binary> resolves them.
  for (int i = 2; i < n; ++i) {
    Dl_info d;
    if (dladdr(bt[i], &d) && d.dli_fname && strstr(d.dli_fname, "eklow-")) fprintf(out, " %lx", (unsigned long)((char*)bt[i] - (char*)d.dli_fbase));
  }
  fputc('\n', out);
  fflush(out);
}

int zeMemAllocDevice(void* ctx, const void* desc, size_t size, size_t align, void* dev, void** pptr) {
  static alloc_fn real;
  if (!real) real = (alloc_fn)dlsym(RTLD_NEXT, "zeMemAllocDevice");
  const int r = real(ctx, desc, size, align, dev, pptr);
  log_call("dev", size, r == 0 ? *pptr : 0);
  return r;
}
int zeMemAllocShared(void* ctx, const void* d1, const void* d2, size_t size, size_t align, void* dev, void** pptr) {
  typedef int (*fn)(void*, const void*, const void*, size_t, size_t, void*, void**);
  static fn real;
  if (!real) real = (fn)dlsym(RTLD_NEXT, "zeMemAllocShared");
  const int r = real(ctx, d1, d2, size, align, dev, pptr);
  log_call("shared", size, r == 0 ? *pptr : 0);
  return r;
}
int zeMemAllocHost(void* ctx, const void* desc, size_t size, size_t align, void** pptr) {
  static alloc_host_fn real;
  if (!real) real = (alloc_host_fn)dlsym(RTLD_NEXT, "zeMemAllocHost");
  const int r = real(ctx, desc, size, align, pptr);
  log_call("host", size, r == 0 ? *pptr : 0);
  return r;
}
int zeMemFree(void* ctx, void* ptr) {
  static free_fn real;
  if (!real) real = (free_fn)dlsym(RTLD_NEXT, "zeMemFree");
  log_call("free", 0, ptr);
  return real(ctx, ptr);
}

// The kernel driver's buffer objects (xe: DRM_IOCTL_XE_GEM_CREATE, i386 nr 0x41), which include what the user-mode driver allocates for
// itself (kernel code, scratch, command buffers): "bo <time> <size> <placement mask> <flags>", then the executable's frames. VRAM0 is placement bit 1.
#include <stdarg.h>
int ioctl(int fd, unsigned long req, ...) {
  typedef int (*fn)(int, unsigned long, ...);
  static fn real;
  if (!real) real = (fn)dlsym(RTLD_NEXT, "ioctl");
  va_list ap;
  va_start(ap, req);
  void* arg = va_arg(ap, void*);
  va_end(ap);
  unsigned long long size = 0;
  unsigned placement = 0, flags = 0;
  const int is_bo = ((req >> 8) & 0xff) == 'd' && (req & 0xff) == 0x41 && arg;
  if (is_bo) {
    const unsigned long long* w = (const unsigned long long*)arg;
    size = w[1];
    placement = ((const unsigned*)arg)[4];
    flags = ((const unsigned*)arg)[5];
  }
  const int r = real(fd, req, arg);
  if (is_bo && r == 0) {
    char what[64];
    snprintf(what, sizeof what, "bo");
    log_call(what, size, (void*)(unsigned long)placement);
  }
  return r;
}
