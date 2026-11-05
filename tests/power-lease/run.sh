#!/usr/bin/env bash
# Decision-table test for transcribe_remote.sh's power lease, with ssh/wake/python stubbed.
# Asserts on two observable effects only: was a poweroff sent, and does a lease exist after.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$(cd "$HERE/../.." && pwd)/transcribe_remote.sh"
export PATH="$HERE/stub:$PATH"
BOOT_A=11111111-1111-1111-1111-111111111111
BOOT_B=22222222-2222-2222-2222-222222222222

pass=0; fail=0

run_case() {  # name, expect_poweroff(y/n), expect_lease(y/n/stale), then env assignments
    local name="$1" exp_off="$2" exp_lease="$3"; shift 3
    local dir; dir="$(mktemp -d "$HERE/case.XXXX")"
    export STUB_LOG="$dir"; mkdir -p "$dir"
    export TRANSCRIBE_LEASE_DIR="$dir/lease"; mkdir -p "$TRANSCRIBE_LEASE_DIR"
    export TRANSCRIBE_GPU_HOST=testhost
    export TRANSCRIBE_PY="$HERE/stub/fakepy"
    export TRANSCRIBE_WAKE="$HERE/stub/wake"
    export STUB_BOOT_ID="$BOOT_A"
    unset TRANSCRIBE_KEEP_ON TRANSCRIBE_RELEASE STUB_HOST_UP STUB_BUSY_RC STUB_WORK_RC STUB_WAKE_RC
    local preset_lease=""
    for kv in "$@"; do
        case "$kv" in
            lease=*) preset_lease="${kv#lease=}" ;;
            *) export "${kv?}" ;;
        esac
    done
    [ -n "$preset_lease" ] && printf '%s\n' "$preset_lease" > "$TRANSCRIBE_LEASE_DIR/transcribe-lease-testhost"

    local out rc
    out="$("$SCRIPT" -o /dev/null /dev/null m l "" true 2>&1)"; rc=$?

    local got_off=n got_lease=n
    [ -s "$dir/poweroff" ] && got_off=y
    [ -f "$TRANSCRIBE_LEASE_DIR/transcribe-lease-testhost" ] && got_lease=y

    if [ "$got_off" = "$exp_off" ] && [ "$got_lease" = "$exp_lease" ]; then
        printf 'PASS  %-46s rc=%s poweroff=%s lease=%s\n' "$name" "$rc" "$got_off" "$got_lease"
        pass=$((pass+1))
    else
        printf 'FAIL  %-46s rc=%s poweroff=%s(want %s) lease=%s(want %s)\n' "$name" "$rc" "$got_off" "$exp_off" "$got_lease" "$exp_lease"
        printf '      output: %s\n' "$out"
        fail=$((fail+1))
    fi
    rm -rf "$dir"
}

#         name                                         off lease   env
run_case "down, no flags -> wake+poweroff"              y  n
run_case "down, KEEP_ON -> stays up, lease taken"       n  y       TRANSCRIBE_KEEP_ON=1
run_case "up+lease, no flags -> stays up, lease kept"   n  y       STUB_HOST_UP=1 "lease=$BOOT_A"
run_case "up+lease, RELEASE -> poweroff, lease gone"    y  n       STUB_HOST_UP=1 TRANSCRIBE_RELEASE=1 "lease=$BOOT_A"
run_case "up+STALE lease, RELEASE -> no poweroff"       n  n       STUB_HOST_UP=1 TRANSCRIBE_RELEASE=1 "lease=$BOOT_B"
run_case "up, no lease, RELEASE -> no poweroff"         n  n       STUB_HOST_UP=1 TRANSCRIBE_RELEASE=1
run_case "up, KEEP_ON -> no lease (not ours to lease)"  n  n       STUB_HOST_UP=1 TRANSCRIBE_KEEP_ON=1
run_case "up+lease, RELEASE, broker busy -> keep both"  n  y       STUB_HOST_UP=1 TRANSCRIBE_RELEASE=1 STUB_BUSY_RC=0 "lease=$BOOT_A"
run_case "down, no flags, work FAILS -> still powers off" y n      STUB_WORK_RC=1
run_case "down, lease from an older boot -> dropped"    y  n       "lease=$BOOT_B"

# KEEP_ON + RELEASE together must refuse before touching anything
if TRANSCRIBE_GPU_HOST=testhost TRANSCRIBE_KEEP_ON=1 TRANSCRIBE_RELEASE=1 "$SCRIPT" 2>/dev/null; then
    echo "FAIL  KEEP_ON+RELEASE should exit non-zero"; fail=$((fail+1))
else
    echo "PASS  KEEP_ON+RELEASE refused"; pass=$((pass+1))
fi

echo
echo "passed $pass, failed $fail"
[ "$fail" = 0 ]
