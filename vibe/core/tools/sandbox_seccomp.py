"""Seccomp-BPF syscall filter for the Linux bubblewrap sandbox (defense-in-depth).

bubblewrap sets up namespaces + bind mounts but installs no syscall filter of
its own. This module builds a small classic-BPF program (a raw ``sock_filter``
array) that bwrap loads via ``--seccomp <fd>`` and applies with NO_NEW_PRIVS
right before ``execvp`` — so it also covers the shell and everything it spawns.

Policy: a denylist with default ALLOW. It returns EPERM for a handful of
escape/introspection syscalls that a namespace-only sandbox does not otherwise
contain, and KILLs on an architecture mismatch (the classic compat-ABI
syscall-number aliasing bypass). Network is left to bwrap ``--unshare-net``,
which removes the net namespace entirely — stronger than an EPERM filter and
avoids breaking AF_UNIX socketpair plumbing that many build tools rely on.

Denied (EPERM): ptrace, process_vm_readv/writev (cross-process memory), and
io_uring_setup/enter/register (io_uring submits I/O out-of-band and can bypass
syscall-level filtering). Mirrors codex's escape-hardening set; the arch-guard
KILL mirrors gemini-cli.
"""

from __future__ import annotations

import os
import platform
import struct
import tempfile

# classic-BPF instruction encodings (linux/bpf_common.h)
_BPF_LD_W_ABS = 0x20  # BPF_LD | BPF_W | BPF_ABS  — load a 32-bit seccomp_data field
_BPF_JEQ_K = 0x15  # BPF_JMP | BPF_JEQ | BPF_K  — if A == k jump jt else jf
_BPF_RET_K = 0x06  # BPF_RET | BPF_K            — return the constant action

# struct seccomp_data field byte offsets (linux/seccomp.h): {int nr; __u32 arch; ...}
_OFF_NR = 0
_OFF_ARCH = 4

# seccomp return actions (linux/seccomp.h)
_RET_KILL_PROCESS = 0x80000000
_RET_ERRNO = 0x00050000  # OR'd with the errno in the low 16 bits
_RET_ALLOW = 0x7FFF0000
_EPERM = 1

# AUDIT_ARCH_* tokens (linux/audit.h). The arch guard KILLs any syscall whose
# seccomp_data.arch != the running arch, blocking x86 compat-mode number aliasing.
_AUDIT_ARCH_X86_64 = 0xC000003E
_AUDIT_ARCH_AARCH64 = 0xC00000B7

# Per-arch (audit_arch, denied syscall numbers). Numbers are arch-specific: x86_64
# has its own table; aarch64 uses the asm-generic unistd numbers.
_ARCH_TABLE: dict[str, tuple[int, tuple[int, ...]]] = {
    # ptrace, process_vm_readv, process_vm_writev, io_uring_setup/enter/register
    "x86_64": (_AUDIT_ARCH_X86_64, (101, 310, 311, 425, 426, 427)),
    "aarch64": (_AUDIT_ARCH_AARCH64, (117, 270, 271, 425, 426, 427)),
}

_MACHINE_ALIASES = {
    "x86_64": "x86_64",
    "amd64": "x86_64",
    "aarch64": "aarch64",
    "arm64": "aarch64",
}


def _inst(code: int, jt: int, jf: int, k: int) -> bytes:
    # struct sock_filter { __u16 code; __u8 jt; __u8 jf; __u32 k; } — 8 bytes LE.
    return struct.pack("<HBBI", code, jt, jf, k)


def build_seccomp_bpf(machine: str | None = None) -> bytes | None:
    """Return the compiled ``sock_filter`` array, or None on an unsupported arch.

    Layout (instruction indices):
      0  LD  arch
      1  JEQ audit_arch -> jt=1 (skip kill), jf=0 (fall through to kill)
      2  RET KILL_PROCESS               # arch mismatch
      3  LD  nr
      4..4+n-1  JEQ syscall_i -> jt=(to EPERM ret), jf=0 (next check)
      4+n   RET ALLOW                   # default: unlisted syscalls pass
      4+n+1 RET ERRNO(EPERM)            # denylist landing pad
    """
    key = _MACHINE_ALIASES.get((machine or platform.machine()).lower())
    if key is None:
        return None
    audit_arch, denied = _ARCH_TABLE[key]
    n = len(denied)
    eperm_idx = 4 + n + 1  # index of the trailing RET ERRNO instruction

    prog = bytearray()
    prog += _inst(_BPF_LD_W_ABS, 0, 0, _OFF_ARCH)
    prog += _inst(_BPF_JEQ_K, 1, 0, audit_arch)
    prog += _inst(_BPF_RET_K, 0, 0, _RET_KILL_PROCESS)
    prog += _inst(_BPF_LD_W_ABS, 0, 0, _OFF_NR)
    for i, nr in enumerate(denied):
        idx = 4 + i
        # jt/jf are offsets from the NEXT instruction; jump to the EPERM ret on
        # match, fall through to the next JEQ on miss.
        prog += _inst(_BPF_JEQ_K, eperm_idx - (idx + 1), 0, nr)
    prog += _inst(_BPF_RET_K, 0, 0, _RET_ALLOW)
    prog += _inst(_BPF_RET_K, 0, 0, _RET_ERRNO | _EPERM)
    return bytes(prog)


def open_seccomp_fd(bpf: bytes) -> int:
    """Write *bpf* to a readable, seekable fd for bwrap ``--seccomp`` and return it.

    Prefers an anonymous memfd; falls back to an immediately-unlinked temp file
    when ``os.memfd_create`` is missing — some python-build-standalone runtimes
    (uv-managed 3.12) are compiled without it. Both give bwrap a real seekable
    file (matching gemini-cli's proven ``9< file`` fd), and neither leaves a path
    to clean up. Callers list the fd in the spawn's ``pass_fds`` (asyncio/Popen
    close every other inherited fd regardless of CLOEXEC) and close it after spawn.
    """
    if hasattr(os, "memfd_create"):
        fd = os.memfd_create("vibe-seccomp", 0)
        try:
            os.write(fd, bpf)
            os.lseek(fd, 0, os.SEEK_SET)
        except OSError:
            os.close(fd)
            raise
        return fd
    write_fd, path = tempfile.mkstemp(prefix="vibe-seccomp-")
    try:
        os.write(write_fd, bpf)
    finally:
        os.close(write_fd)
    # Reopen read-only, then unlink: the fd keeps the inode alive, so the on-disk
    # path disappears at once and there is nothing to clean up after the run.
    read_fd = os.open(path, os.O_RDONLY)
    os.unlink(path)
    return read_fd
