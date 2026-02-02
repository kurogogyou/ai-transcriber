#!/bin/bash
# Whisper Batch Transcription Script
# Handles English, Spanish, and multilingual audio
#
# Usage: ./whisper-batch-transcribe.sh <input_folder> [model_size] [language] [extension_filter]
#
# Examples:
#   ./whisper-batch-transcribe.sh "folder" small en       # English-only (uses small.en)
#   ./whisper-batch-transcribe.sh "folder" medium es      # Spanish-only
#   ./whisper-batch-transcribe.sh "folder" large-v3 multi # Mixed English/Spanish
#   ./whisper-batch-transcribe.sh "folder" medium multi m4v  # Only .m4v files
#
# Model sizes: tiny, base, small, medium, large-v3
# Languages: en (English), es (Spanish), multi (auto-detect, for mixed)
# Extension filter: optionally process only files with a specific extension (e.g., m4v)
#
# Defaults: small model, multilingual (safest for unknown content)

set -e

INPUT_FOLDER="${1:?Usage: $0 <input_folder> [model_size] [language] [extension_filter]}"
MODEL_SIZE="${2:-small}"
LANGUAGE="${3:-multi}"
EXT_FILTER="${4:-}"

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

# Output to 'output/' folder in the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FOLDER="${SCRIPT_DIR}/output/transcripts_${MODEL}_${LANGUAGE}"
LOG_TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
LOG_FILE="${OUTPUT_FOLDER}/transcription_${LOG_TIMESTAMP}.log"

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

# Count files
FILE_COUNT=0
for ext in "${EXTS[@]}"; do
    FILE_COUNT=$((FILE_COUNT + $(find "$INPUT_FOLDER" -maxdepth 1 -name "*.$ext" 2>/dev/null | wc -l)))
done
CURRENT=0
SKIPPED=0
TOTAL_START=$(date +%s)

echo "Found $FILE_COUNT media files to process"
echo ""

# Process each file
shopt -s nullglob
FILE_GLOBS=()
for ext in "${EXTS[@]}"; do
    FILE_GLOBS+=("$INPUT_FOLDER"/*."$ext")
done
for file in "${FILE_GLOBS[@]}"; do
    [ -e "$file" ] || continue

    CURRENT=$((CURRENT + 1))
    BASENAME=$(basename "$file")
    NAME="${BASENAME%.*}"

    # Skip if already transcribed (output .txt exists)
    if [ -f "$OUTPUT_FOLDER/$NAME.txt" ]; then
        SKIPPED=$((SKIPPED + 1))
        echo "[$CURRENT/$FILE_COUNT] Skipping (already done): $BASENAME"
        continue
    fi

    echo "[$CURRENT/$FILE_COUNT] Processing: $BASENAME"
    FILE_START=$(date +%s)

    # Run whisper (note: $LANG_FLAG is intentionally unquoted to allow empty value)
    whisper "$file" \
        --model "$MODEL" \
        --device "$DEVICE" \
        --output_dir "$OUTPUT_FOLDER" \
        --output_format all \
        $LANG_FLAG \
        --verbose False \
        2>&1 | tee -a "$LOG_FILE"

    FILE_END=$(date +%s)
    FILE_DURATION=$((FILE_END - FILE_START))

    echo "  Completed in ${FILE_DURATION}s"
    echo "[$CURRENT/$FILE_COUNT] $BASENAME: ${FILE_DURATION}s" >> "$LOG_FILE"
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
