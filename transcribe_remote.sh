#!/usr/bin/env bash
# Transcribe from a machine without a GPU, with nothing staged on the GPU host.
#
# Opens an SSH tunnel to the broker (8090) and the transcribe worker (8083) on
# the GPU host, then runs transcribe_via_broker.py --upload, which streams each
# file's audio track through it. Same arguments as whisper-batch-transcribe.sh.
#
# Why a tunnel and not the LAN: the broker and its workers bind 127.0.0.1 on
# purpose (gpu-broker docker_mgr.py, 2026-08-07). LAN exposure would be
# unauthenticated, and a caller that skips /acquire leaves the VRAM ledger
# wrong. Over the tunnel the broker hands back http://127.0.0.1:8083, which
# resolves here too, and acquire/release still run.
#
# Power: if the GPU host does not answer ssh, it is woken with `wake --wait`
# (WoL, ~40 s) and powered off on exit, success or failure, through its scoped
# `sudo -n /usr/bin/systemctl poweroff` rule. A host that was already up is left
# alone, and so is one whose broker still has other clients' handles open (a
# MIKASA gpu_job.sh run may have started using the host this run woke).
#
# Power lease, for a chain of several runs against one boot. Waking costs ~40 s
# and the poweroff takes the worker container and its resident weights with it,
# so back-to-back runs paid ~40 s of WoL plus ~13 s of cold start each. Measured
# 2026-11-05 on a 27-minute meeting transcribed twice: 51 s and 58 s wall for
# 19 s of actual transcription either time.
#
# TRANSCRIBE_KEEP_ON=1 on a run that woke the host leaves it on AND records a
# lease; TRANSCRIBE_RELEASE=1 on the last run of the chain powers it off and
# drops the lease. Runs in between need neither flag: they find the host up, do
# their work, and leave the lease alone. The lease stores the host's boot id, so
# it cannot be honoured across a reboot, and it is only ever taken by a run that
# did the waking: a host that was already up belongs to whoever turned it on.
#
# Env: TRANSCRIBE_GPU_HOST (ssh alias, default scully; must also be a `wake`
# node name), GPU_BROKER_SRC (a local gpu-broker checkout for its client,
# default /opt/brain/src/gpu-broker), TRANSCRIBE_PY (interpreter with
# `requests`, default python3), TRANSCRIBE_WAKE (default ~/bin/wake),
# TRANSCRIBE_KEEP_ON=1 (leave a host this run woke powered on, and lease it),
# TRANSCRIBE_RELEASE=1 (power off a host this chain leased), TRANSCRIBE_LEASE_DIR
# (where the lease file lives, default $XDG_RUNTIME_DIR).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${TRANSCRIBE_GPU_HOST:-scully}"
PY="${TRANSCRIBE_PY:-python3}"
WAKE="${TRANSCRIBE_WAKE:-$HOME/bin/wake}"
KEEP_ON="${TRANSCRIBE_KEEP_ON:-0}"
RELEASE="${TRANSCRIBE_RELEASE:-0}"
LEASE_DIR="${TRANSCRIBE_LEASE_DIR:-${XDG_RUNTIME_DIR:-/tmp}}"
LEASE="$LEASE_DIR/transcribe-lease-$HOST"
PORTS=(8090 8083)   # broker, then the transcribe role's port in roles.yaml

if [ "$KEEP_ON" = 1 ] && [ "$RELEASE" = 1 ]; then
    echo "ERROR: TRANSCRIBE_KEEP_ON=1 and TRANSCRIBE_RELEASE=1 contradict each other." >&2
    exit 2
fi

for port in "${PORTS[@]}"; do
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
        echo "ERROR: something already listens on 127.0.0.1:$port here; the tunnel needs it free." >&2
        exit 2
    fi
done

up() { ssh -o BatchMode=yes -o ConnectTimeout=4 "$HOST" true 2>/dev/null; }
broker_busy() {  # other clients' handles still open on the broker (needs the tunnel)
    "$PY" -c 'import json,sys,urllib.request
s=json.load(urllib.request.urlopen("http://127.0.0.1:8090/status",timeout=5))
sys.exit(0 if s.get("active_handles",0)>0 else 1)' 2>/dev/null
}

host_ssh() {  # reuse the tunnel's already-authenticated master socket once there is one
    if [ "${tunnel:-0}" = 1 ]; then ssh -S "$ctl" "$HOST" "$@"
    else ssh -o BatchMode=yes -o ConnectTimeout=6 "$HOST" "$@"; fi
}
host_boot_id() { host_ssh 'cat /proc/sys/kernel/random/boot_id' 2>/dev/null | tr -d '[:space:]'; }

lease_held() {  # ours only while the host is still on the boot the lease was taken against
    [ -f "$LEASE" ] || return 1
    local want now
    want="$(head -n1 "$LEASE" 2>/dev/null | tr -d '[:space:]')"
    [ -n "$want" ] || return 1
    now="$(host_boot_id)"
    [ -n "$now" ] && [ "$now" = "$want" ]
}
lease_take() {
    local id
    id="$(host_boot_id)"
    [ -n "$id" ] || { echo "WARN: could not read $HOST's boot id, so no power lease was taken." >&2; return 1; }
    mkdir -p "$LEASE_DIR" || return 1
    printf '%s\n# taken by pid %s at %s for %s\n' "$id" "$$" "$(date -u +%FT%TZ)" "$HOST" > "$LEASE"
}
lease_drop() { rm -f "$LEASE"; }

woke=0
if ! up; then
    [ -x "$WAKE" ] || { echo "ERROR: $HOST is down and $WAKE is missing, so it cannot be woken." >&2; exit 2; }
    echo "$HOST is down; waking it (WoL)..." >&2
    "$WAKE" --wait "$HOST" >&2 || { echo "ERROR: $HOST did not wake." >&2; exit 3; }
    woke=1
    # The host was off, so any lease on disk is for a boot that no longer exists.
    lease_drop
fi

ctl="$(mktemp -u "${XDG_RUNTIME_DIR:-/tmp}/transcribe-tunnel.XXXXXX")"
tunnel=0
cleanup() {
    rc=$?
    set +e
    power_down=0
    reason=""

    if [ "$woke" = 1 ] && [ "$KEEP_ON" != 1 ]; then
        power_down=1
        reason="this run woke it"
    elif [ "$woke" = 1 ]; then
        if lease_take; then
            echo "$HOST left on (TRANSCRIBE_KEEP_ON=1); power lease held at $LEASE." >&2
            echo "  the last run of this chain should set TRANSCRIBE_RELEASE=1 to power it off." >&2
        else
            echo "$HOST left on (TRANSCRIBE_KEEP_ON=1), but with NO lease, so no later run will power it off." >&2
        fi
    elif [ "$RELEASE" = 1 ]; then
        if lease_held; then
            power_down=1
            reason="releasing the lease an earlier run of this chain took"
        elif [ -f "$LEASE" ]; then
            echo "NOTE: the lease at $LEASE is against a different boot of $HOST, so this chain no longer owns it; leaving it on." >&2
            lease_drop
        else
            echo "NOTE: TRANSCRIBE_RELEASE=1 but there is no lease for $HOST, so it is not this chain's to power off." >&2
        fi
    elif [ "$KEEP_ON" = 1 ]; then
        echo "$HOST left on (TRANSCRIBE_KEEP_ON=1); it was already up, so no lease was taken." >&2
    elif lease_held; then
        echo "NOTE: $HOST stays on; this chain holds a power lease at $LEASE." >&2
        echo "  the last run of the chain should set TRANSCRIBE_RELEASE=1 to power it off." >&2
    fi

    if [ "$power_down" = 1 ]; then
        # broker_busy needs the tunnel, so a run that never got one cannot check
        # for other clients and powers off on the lease alone.
        if [ "$tunnel" = 1 ] && broker_busy; then
            # Keep the lease: a later TRANSCRIBE_RELEASE=1 can still finish the job.
            echo "NOTE: $HOST's broker still has other clients' handles open; leaving it on." >&2
        else
            ssh -o BatchMode=yes "$HOST" 'sudo -n /usr/bin/systemctl poweroff' >/dev/null 2>&1
            echo "$HOST: poweroff sent ($reason)." >&2
            lease_drop
        fi
    fi

    [ "$tunnel" = 1 ] && ssh -S "$ctl" -O exit "$HOST" 2>/dev/null
    exit "$rc"
}
trap cleanup EXIT

fwd=()
for port in "${PORTS[@]}"; do fwd+=(-L "$port:127.0.0.1:$port"); done
if ! ssh -f -N -M -S "$ctl" -o ExitOnForwardFailure=yes -o ConnectTimeout=8 "${fwd[@]}" "$HOST"; then
    echo "ERROR: could not open the tunnel to $HOST. Is gpu-broker running there?" >&2
    exit 2
fi
tunnel=1

# ssh answers before the broker (a lingering user unit) is listening after a cold boot.
for _ in $(seq 1 60); do
    "$PY" -c 'import urllib.request,sys; sys.exit(0 if urllib.request.urlopen("http://127.0.0.1:8090/health",timeout=3).status==200 else 1)' 2>/dev/null && break
    sleep 2
done

"$PY" "$SCRIPT_DIR/transcribe_via_broker.py" --upload "$@"
