#!/bin/bash
# Whisper Batch Transcription Script
# Handles English, Spanish, and multilingual audio with optional speaker diarization
#
# Usage: ./whisper-batch-transcribe.sh [options] <input_folder> [model_size] [language] [extension_filter] [diarize]
#
# Options:
#   -o, --output-dir <path>   Custom output directory (default: auto-generated in script dir)
#
# Examples:
#   ./whisper-batch-transcribe.sh "folder" small en       # English-only (uses small.en)
#   ./whisper-batch-transcribe.sh "folder" medium es      # Spanish-only
#   ./whisper-batch-transcribe.sh "folder" large-v3 multi # Mixed English/Spanish
#   ./whisper-batch-transcribe.sh "folder" medium multi m4v       # Only .m4v files
#   ./whisper-batch-transcribe.sh "folder" medium en "" true      # English with speaker diarization
#   ./whisper-batch-transcribe.sh -o ~/output "folder" medium en  # Custom output directory
#
# Model sizes: tiny, base, small, medium, large-v3
# Languages: en (English), es (Spanish), multi (auto-detect, for mixed)
# Extension filter: optionally process only files with a specific extension (e.g., m4v), use "" to skip
# Diarize: set to "true" to enable speaker diarization (requires whisperx and HF_TOKEN)
#
# Speaker Diarization Setup:
#   1. pip install whisperx
#   2. Accept pyannote license at https://huggingface.co/pyannote/speaker-diarization-3.1
#   3. Set HF_TOKEN environment variable with your HuggingFace token
#   4. Note: Diarization uses more VRAM (~7-8GB) and is ~1.5-2x slower
#
# Defaults: small model, multilingual, no diarization

set -e

# Parse named flags
CUSTOM_OUTPUT=""
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output-dir)
            CUSTOM_OUTPUT="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL_ARGS[@]}"

INPUT_FOLDER="${1:?Usage: $0 [options] <input_folder> [model_size] [language] [extension_filter] [diarize]}"
MODEL_SIZE="${2:-small}"
LANGUAGE="${3:-multi}"
EXT_FILTER="${4:-}"
DIARIZE="${5:-false}"

# Determine model name and language flag based on language setting
case "$LANGUAGE" in
    en)
        # English-only: use .en model variant for better accuracy
        if [[ "$MODEL_SIZE" == "large-v3" ]]; then
            MODEL="$MODEL_SIZE"  # large-v3 has no .en variant
        else
            MODEL="${MODEL_SIZE}.en"
        fi
        LANG_FLAG="--language en"
        LANG_DESC="English-only"
        ;;
    es)
        # Spanish-only: use multilingual model with Spanish flag
        MODEL="$MODEL_SIZE"
        LANG_FLAG="--language es"
        LANG_DESC="Spanish-only"
        ;;
    multi|*)
        # Multilingual: use multilingual model, let Whisper auto-detect
        MODEL="$MODEL_SIZE"
        LANG_FLAG=""
        LANG_DESC="Multilingual (auto-detect)"
        ;;
esac

# Output folder: use custom path if provided, otherwise auto-generate in script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$DIARIZE" == "true" ]]; then
    DIARIZE_DESC="enabled"
else
    DIARIZE_DESC="disabled"
fi

if [[ -n "$CUSTOM_OUTPUT" ]]; then
    OUTPUT_FOLDER="$CUSTOM_OUTPUT"
else
    if [[ "$DIARIZE" == "true" ]]; then
        OUTPUT_FOLDER="${SCRIPT_DIR}/output/transcripts_${MODEL}_${LANGUAGE}_diarized"
    else
        OUTPUT_FOLDER="${SCRIPT_DIR}/output/transcripts_${MODEL}_${LANGUAGE}"
    fi
fi
LOG_TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
LOG_FILE="${OUTPUT_FOLDER}/transcription_${LOG_TIMESTAMP}.log"

# Check diarization requirements
if [[ "$DIARIZE" == "true" ]]; then
    if ! command -v whisperx &> /dev/null; then
        echo "ERROR: whisperx not found. Install with: pip install whisperx"
        exit 1
    fi
    if [[ -z "$HF_TOKEN" ]]; then
        echo "ERROR: HF_TOKEN environment variable required for diarization."
        echo "  1. Get token from https://huggingface.co/settings/tokens"
        echo "  2. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1"
        echo "  3. Export HF_TOKEN=your_token_here"
        exit 1
    fi
fi

# Detect device
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE="cuda"
    DEVICE_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
else
    DEVICE="cpu"
    DEVICE_NAME="CPU"
fi

echo "========================================"
echo "Whisper Batch Transcription"
echo "========================================"
echo "Input folder: $INPUT_FOLDER"
echo "Model: $MODEL"
echo "Language: $LANG_DESC"
echo "Device: $DEVICE ($DEVICE_NAME)"
echo "Diarization: $DIARIZE_DESC"
echo "Output: $OUTPUT_FOLDER"
echo "Log: $LOG_FILE"
echo "========================================"

mkdir -p "$OUTPUT_FOLDER"

# Start logging
{
    echo "Transcription started: $(date)"
    echo "Model: $MODEL"
    echo "Language: $LANG_DESC"
    echo "Device: $DEVICE ($DEVICE_NAME)"
    echo "Diarization: $DIARIZE_DESC"
    echo "----------------------------------------"
} > "$LOG_FILE"

# Build extension list
ALL_EXTS=(mkv mp4 m4v webm mp3 wav m4a ogg)
if [[ -n "$EXT_FILTER" ]]; then
    EXTS=("$EXT_FILTER")
    echo "Extension filter: *.$EXT_FILTER only"
else
    EXTS=("${ALL_EXTS[@]}")
fi

# Collect files recursively using find
FILES=()
for ext in "${EXTS[@]}"; do
    while IFS= read -r -d '' f; do
        FILES+=("$f")
    done < <(find "$INPUT_FOLDER" -type f -name "*.$ext" -print0 2>/dev/null)
done
FILE_COUNT=${#FILES[@]}
CURRENT=0
SKIPPED=0
TOTAL_START=$(date +%s)

echo "Found $FILE_COUNT media files to process (including subfolders)"
echo ""

# Resolve input folder to absolute path for reliable relative path computation
ABS_INPUT="$(cd "$INPUT_FOLDER" && pwd)"

# Process each file, mirroring subfolder structure in output
for file in "${FILES[@]}"; do
    CURRENT=$((CURRENT + 1))
    BASENAME=$(basename "$file")
    NAME="${BASENAME%.*}"

    # Compute path relative to input folder and mirror it in output
    ABS_FILE="$(cd "$(dirname "$file")" && pwd)/$(basename "$file")"
    REL_DIR="$(dirname "${ABS_FILE#"$ABS_INPUT"/}")"
    if [[ "$REL_DIR" == "$ABS_FILE" || "$REL_DIR" == "." ]]; then
        # File is directly in the input folder (no subfolder)
        FILE_OUTPUT_DIR="$OUTPUT_FOLDER"
        DISPLAY_NAME="$BASENAME"
    else
        FILE_OUTPUT_DIR="$OUTPUT_FOLDER/$REL_DIR"
        DISPLAY_NAME="$REL_DIR/$BASENAME"
    fi

    # Skip if already transcribed (output .txt exists in mirrored path)
    if [ -f "$FILE_OUTPUT_DIR/$NAME.txt" ]; then
        SKIPPED=$((SKIPPED + 1))
        echo "[$CURRENT/$FILE_COUNT] Skipping (already done): $DISPLAY_NAME"
        continue
    fi

    mkdir -p "$FILE_OUTPUT_DIR"
    echo "[$CURRENT/$FILE_COUNT] Processing: $DISPLAY_NAME"
    FILE_START=$(date +%s)

    if [[ "$DIARIZE" == "true" ]]; then
        # Run whisperx with diarization
        # Note: whisperx uses different argument format
        whisperx "$file" \
            --model "$MODEL_SIZE" \
            --device "$DEVICE" \
            --output_dir "$FILE_OUTPUT_DIR" \
            --output_format all \
            --diarize \
            --hf_token "$HF_TOKEN" \
            $LANG_FLAG \
            2>&1 | tee -a "$LOG_FILE"
    else
        # Run standard whisper (note: $LANG_FLAG is intentionally unquoted to allow empty value)
        whisper "$file" \
            --model "$MODEL" \
            --device "$DEVICE" \
            --output_dir "$FILE_OUTPUT_DIR" \
            --output_format all \
            $LANG_FLAG \
            --verbose False \
            2>&1 | tee -a "$LOG_FILE"
    fi

    FILE_END=$(date +%s)
    FILE_DURATION=$((FILE_END - FILE_START))

    echo "  Completed in ${FILE_DURATION}s"
    echo "[$CURRENT/$FILE_COUNT] $DISPLAY_NAME: ${FILE_DURATION}s" >> "$LOG_FILE"
done

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
MINUTES=$((TOTAL_DURATION / 60))
SECONDS=$((TOTAL_DURATION % 60))

PROCESSED=$((FILE_COUNT - SKIPPED))
echo ""
echo "========================================"
echo "Completed: $PROCESSED processed, $SKIPPED skipped (already done)"
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo "Output: $OUTPUT_FOLDER"
echo "========================================"

{
    echo "----------------------------------------"
    echo "Transcription completed: $(date)"
    echo "Total time: ${MINUTES}m ${SECONDS}s"
} >> "$LOG_FILE"
