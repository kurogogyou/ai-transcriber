#!/bin/bash
# Whisper Batch Transcription — dispatches to gpu-broker /transcribe.
#
# Post-Phase-4 (2026-06-14): this script no longer loads whisperx locally.
# All transcription is dispatched to the broker-managed whisperx-server
# container via transcribe_via_broker.py (HTTP POST per file).
#
# Usage (unchanged for skill compat):
#   ./whisper-batch-transcribe.sh [options] <input_folder> [model_size] [language] [extension_filter] [diarize]
#
# Options:
#   -o, --output-dir <path>   Custom output directory (default: auto-generated in script dir)
#
# Examples:
#   ./whisper-batch-transcribe.sh "folder" small en               # English-only
#   ./whisper-batch-transcribe.sh "folder" medium es              # Spanish-only
#   ./whisper-batch-transcribe.sh "folder" large-v3 multi         # Mixed English/Spanish
#   ./whisper-batch-transcribe.sh "folder" medium multi m4v       # Only .m4v files
#   ./whisper-batch-transcribe.sh "folder" medium en "" true      # English with speaker diarization
#   ./whisper-batch-transcribe.sh -o ~/output "folder" medium en  # Custom output directory
#
# Model sizes: legacy positional (ignored — broker pins large-v3 server-side).
# Languages:   en | es | multi (auto-detect)
# Diarize:     "true" to enable speaker diarization (requires HF_TOKEN
#              configured in the broker's whisperx-server container)
#
# Env overrides:
#   GPU_BROKER_URL       broker base URL (default http://127.0.0.1:8090)
#   WHISPERX_BATCH_SIZE  override broker's pinned batch_size=4 (use sparingly)
#   TRANSCRIBE_TIMEOUT   per-file timeout seconds (default 1800)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="/opt/fast/venvs/ai-transcriber/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "ERROR: venv not found at $VENV_PY"
    echo "Provision with: python3 -m venv /opt/fast/venvs/ai-transcriber && \\"
    echo "                /opt/fast/venvs/ai-transcriber/bin/pip install -r $SCRIPT_DIR/requirements.txt"
    exit 1
fi

exec "$VENV_PY" "$SCRIPT_DIR/transcribe_via_broker.py" "$@"
