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
# Env: TRANSCRIBE_GPU_HOST (ssh alias, default scully; must also be a `wake`
# node name), GPU_BROKER_SRC (a local gpu-broker checkout for its client,
# default /opt/brain/src/gpu-broker), TRANSCRIBE_PY (interpreter with
# `requests`, default python3), TRANSCRIBE_WAKE (default ~/bin/wake),
# TRANSCRIBE_KEEP_ON=1 (leave a host this run woke powered on).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${TRANSCRIBE_GPU_HOST:-scully}"
PY="${TRANSCRIBE_PY:-python3}"
WAKE="${TRANSCRIBE_WAKE:-$HOME/bin/wake}"
KEEP_ON="${TRANSCRIBE_KEEP_ON:-0}"
PORTS=(8090 8083)   # broker, then the transcribe role's port in roles.yaml

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

woke=0
if ! up; then
    [ -x "$WAKE" ] || { echo "ERROR: $HOST is down and $WAKE is missing, so it cannot be woken." >&2; exit 2; }
    echo "$HOST is down; waking it (WoL)..." >&2
    "$WAKE" --wait "$HOST" >&2 || { echo "ERROR: $HOST did not wake." >&2; exit 3; }
    woke=1
fi

ctl="$(mktemp -u "${XDG_RUNTIME_DIR:-/tmp}/transcribe-tunnel.XXXXXX")"
tunnel=0
cleanup() {
    rc=$?
    set +e
    if [ "$woke" = 1 ] && [ "$KEEP_ON" != 1 ]; then
        if [ "$tunnel" = 1 ] && broker_busy; then
            echo "NOTE: $HOST's broker still has other clients' handles open; leaving it on." >&2
        else
            ssh -o BatchMode=yes "$HOST" 'sudo -n /usr/bin/systemctl poweroff' >/dev/null 2>&1
            echo "$HOST: poweroff sent (this run woke it)." >&2
        fi
    elif [ "$woke" = 1 ]; then
        echo "$HOST left on (TRANSCRIBE_KEEP_ON=1)." >&2
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
