#!/bin/sh
#
# Runs the security regression tests.  Must be executed from the repository root
# (it uses ./enclave.signed.so and ./Training/*).
#
# Expected result on the fixed build: every group exits 0 (all hostile calls are
# refused).  On the vulnerable baseline the hostile groups return 1, and the
# "leak" group prints a TAG0 that is turned back into a plaintext enclave byte by
# tests/verify_leak.py.

set -u

BIN=${BIN:-./security_test}
FAILED=0
TMP=$(mktemp)

report() {
    name=$1
    shift
    echo "================ $name ================"
    "$BIN" "$name" "$@" > "$TMP" 2>&1
    rc=$?
    cat "$TMP"
    TAG=$(sed -n 's/^TAG0=//p' "$TMP")
    if [ -n "$TAG" ]; then
        echo "--- inverting the leaked tag with the all-zero HMAC key ---"
        python3 tests/verify_leak.py --tag "$TAG" --elf ./enclave.signed.so --vaddr 0x2008
    fi
    echo "---------------- exit=$rc"
    if [ "$rc" -ne 0 ]; then
        FAILED=$((FAILED + 1))
    fi
    echo
}

report selfcheck
report leak
report write 0x6c000
report short
report nulls

rm -f "$TMP"

echo "=============================================="
if [ "$FAILED" -eq 0 ]; then
    echo "ALL SECURITY TESTS PASSED (every hostile call was refused)"
else
    echo "$FAILED TEST GROUP(S) DID NOT BEHAVE AS EXPECTED"
fi
exit "$FAILED"
