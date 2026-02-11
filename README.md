# AI Transcriber

Batch audio/video transcription using OpenAI Whisper with GPU acceleration.

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (optional, falls back to CPU)
- FFmpeg

## Installation

### 1. System Dependencies

```bash
sudo apt install ffmpeg
```

### 2. Python Environment

```bash
cd ai-transcriber

# Activate the virtual environment
source venv/bin/activate

# Install PyTorch with CUDA support (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Whisper
pip install -r requirements.txt
```

### 3. Verify GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Usage

```bash
./whisper-batch-transcribe.sh [options] <input_folder> [model_size] [language] [extension_filter] [diarize]
```

### Options

| Option | Description |
|--------|-------------|
| `-o`, `--output-dir <path>` | Custom output directory (default: auto-generated in script dir) |

### Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| input_folder | path | required | Folder containing media files (subfolders are included) |
| model_size | tiny, base, small, medium, large-v3 | small | Whisper model size |
| language | en, es, multi | multi | Language mode |
| extension_filter | file extension (e.g., m4v, mp4) | all | Process only files with this extension (use "" to skip) |
| diarize | true, false | false | Enable speaker diarization (who said what) |

### Examples

```bash
# English content (uses optimized .en model)
./whisper-batch-transcribe.sh ~/Videos/lectures small en

# Spanish content
./whisper-batch-transcribe.sh ~/Videos/spanish medium es

# Mixed/unknown languages (auto-detect)
./whisper-batch-transcribe.sh ~/Videos/mixed large-v3 multi

# Process only .m4v files
./whisper-batch-transcribe.sh ~/Videos/meetings medium multi m4v

# English with speaker diarization (identifies who said what)
./whisper-batch-transcribe.sh ~/Videos/interviews medium en "" true

# Custom output directory
./whisper-batch-transcribe.sh -o ~/my-transcripts ~/Videos/lectures medium en
```

### Supported Formats

- Video: `.mkv`, `.mp4`, `.m4v`, `.webm`
- Audio: `.mp3`, `.wav`, `.m4a`, `.ogg`

## Output

Transcripts are saved to `output/transcripts_<model>_<language>/` in the script directory (or to a custom path via `-o`):

- `.txt` - Plain text
- `.vtt` - WebVTT subtitles
- `.srt` - SRT subtitles
- `.json` - Detailed JSON with timestamps
- `.tsv` - Tab-separated values
- `transcription_<timestamp>.log` - Processing log (one per session)

The subfolder structure of the input directory is mirrored in the output. For example, `input/lectures/lecture1.mp4` produces `output/lectures/lecture1.txt`.

Files that already have transcripts in the output folder are automatically skipped, allowing you to resume interrupted batches or add new files without re-processing.

## Model Selection Guide

| Model | VRAM | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| tiny | ~1GB | Fastest | Basic | Quick drafts |
| base | ~1GB | Fast | Good | Simple content |
| small | ~2GB | Medium | Better | General use |
| medium | ~5GB | Slow | Great | Quality transcripts |
| large-v3 | ~10GB | Slowest | Best | Professional/multilingual |

For a GTX 1070 (8GB VRAM), `small` or `medium` models work well.

## Speaker Diarization

Speaker diarization identifies **who said what** in multi-speaker recordings. It's optional and requires additional setup.

### When to Use Diarization

| Use Case | Diarization Recommended |
|----------|------------------------|
| 2-person interviews | Yes |
| Podcast with hosts | Yes |
| Solo recordings | No |
| Large group calls (5+) | Maybe (less accurate) |
| Quick bulk transcription | No (slower) |

### Setup

1. **Install whisperX:**
   ```bash
   pip install whisperx
   ```

2. **Get HuggingFace token:**
   - Create account at https://huggingface.co
   - Get token from https://huggingface.co/settings/tokens
   - Accept pyannote license at https://huggingface.co/pyannote/speaker-diarization-3.1

3. **Set environment variable:**
   ```bash
   export HF_TOKEN=your_token_here
   ```

### Resource Usage

| Mode | VRAM | Speed |
|------|------|-------|
| Whisper only | ~5GB (medium) | 1x |
| Whisper + Diarization | ~7-8GB | 1.5-2x slower |

For a GTX 1070 (8GB), diarization is tight but workable with the `medium` model.

### Output

With diarization enabled, transcripts include speaker labels:

```
[SPEAKER_00]: Hello, how are you doing today?
[SPEAKER_01]: I'm doing great, thanks for asking.
[SPEAKER_00]: Let's get started with the interview.
```

Output is saved to `output/transcripts_<model>_<language>_diarized/` to keep separate from non-diarized transcripts.

## Tips

- Use `.en` models (via `en` language flag) for English-only content - they're faster and more accurate
- For mixed English/Spanish, use `multi` to let Whisper auto-detect
- The `large-v3` model requires significant VRAM but handles accents and mixed languages best
- Use diarization selectively on important recordings where speaker ID matters
