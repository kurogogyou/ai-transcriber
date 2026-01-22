#!/bin/bash
# Whisper Batch Transcription Script
# Handles English, Spanish, and multilingual audio
#
# Usage: ./whisper-batch-transcribe.sh <input_folder> [model_size] [language]
#
# Examples:
#   ./whisper-batch-transcribe.sh "folder" small en       # English-only (uses small.en)
#   ./whisper-batch-transcribe.sh "folder" medium es      # Spanish-only
#   ./whisper-batch-transcribe.sh "folder" large-v3 multi # Mixed English/Spanish
#
# Model sizes: tiny, base, small, medium, large-v3
# Languages: en (English), es (Spanish), multi (auto-detect, for mixed)
#
# Defaults: small model, multilingual (safest for unknown content)

set -e

INPUT_FOLDER="${1:?Usage: $0 <input_folder> [model_size] [language]}"
MODEL_SIZE="${2:-small}"
LANGUAGE="${3:-multi}"

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
LOG_FILE="${OUTPUT_FOLDER}/transcription.log"

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

# Count files
FILE_COUNT=$(find "$INPUT_FOLDER" -maxdepth 1 \( -name "*.mkv" -o -name "*.mp4" -o -name "*.webm" -o -name "*.mp3" -o -name "*.wav" -o -name "*.m4a" -o -name "*.ogg" \) 2>/dev/null | wc -l)
CURRENT=0
TOTAL_START=$(date +%s)

echo "Found $FILE_COUNT media files to process"
echo ""

# Process each file
shopt -s nullglob
for file in "$INPUT_FOLDER"/*.mkv "$INPUT_FOLDER"/*.mp4 "$INPUT_FOLDER"/*.webm "$INPUT_FOLDER"/*.mp3 "$INPUT_FOLDER"/*.wav "$INPUT_FOLDER"/*.m4a "$INPUT_FOLDER"/*.ogg; do
    [ -e "$file" ] || continue

    CURRENT=$((CURRENT + 1))
    BASENAME=$(basename "$file")
    NAME="${BASENAME%.*}"

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

echo ""
echo "========================================"
echo "Completed: $FILE_COUNT files"
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo "Output: $OUTPUT_FOLDER"
echo "========================================"

{
    echo "----------------------------------------"
    echo "Transcription completed: $(date)"
    echo "Total time: ${MINUTES}m ${SECONDS}s"
} >> "$LOG_FILE"
