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
# Env: TRANSCRIBE_GPU_HOST (ssh alias, default scully), GPU_BROKER_SRC (a local
# gpu-broker checkout for its client, default /opt/brain/src/gpu-broker),
# TRANSCRIBE_PY (interpreter with `requests`, default python3).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${TRANSCRIBE_GPU_HOST:-scully}"
PY="${TRANSCRIBE_PY:-python3}"
PORTS=(8090 8083)   # broker, then the transcribe role's port in roles.yaml

for port in "${PORTS[@]}"; do
    if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
        echo "ERROR: something already listens on 127.0.0.1:$port here; the tunnel needs it free." >&2
        exit 2
    fi
done

ctl="$(mktemp -u "${XDG_RUNTIME_DIR:-/tmp}/transcribe-tunnel.XXXXXX")"
fwd=()
for port in "${PORTS[@]}"; do fwd+=(-L "$port:127.0.0.1:$port"); done
if ! ssh -f -N -M -S "$ctl" -o ExitOnForwardFailure=yes -o ConnectTimeout=8 "${fwd[@]}" "$HOST"; then
    echo "ERROR: could not open the tunnel to $HOST. Is it powered on and is gpu-broker running there?" >&2
    exit 2
fi
trap 'ssh -S "$ctl" -O exit "$HOST" 2>/dev/null || true' EXIT

"$PY" "$SCRIPT_DIR/transcribe_via_broker.py" --upload "$@"
