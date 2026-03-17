# AI Transcriber

Batch audio/video transcription using [whisperx](https://github.com/m-bain/whisperX) (faster-whisper/CTranslate2 engine) with GPU acceleration and optional speaker diarization.

## Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (optional, falls back to CPU)
- FFmpeg
- libnvtoolsext1 (NVIDIA Tools Extension)

## Installation

### 1. System Dependencies

```bash
sudo apt install ffmpeg libnvtoolsext1
```

### 2. Python Environment

```bash
cd ai-transcriber

# Activate the virtual environment
source venv/bin/activate

# Install PyTorch with CUDA 11.8 support (required for GTX 1070 / compute capability 6.1)
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install whisperx and dependencies
pip install -r requirements.txt

# Install cuDNN 8 libs for CTranslate2 (coexists with cuDNN 9 needed by PyTorch)
bash install-cudnn8.sh
```

### 3. Verify GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### Version Constraints

All version pins exist because the GTX 1070 (Pascal, compute capability 6.1) requires CUDA 11.8:

| Package | Pin | Reason |
|---------|-----|--------|
| torch | 2.4.0+cu118 | Last version with CUDA 11.8 wheels |
| whisperx | <3.8 | 3.8+ requires torch>=2.8 |
| pyannote-audio | <4.0 | 4.0+ requires torch>=2.8 |
| nvidia-cudnn-cu12 | 9.1.0.70 | 9.20+ causes CUDNN_STATUS_EXECUTION_FAILED on Pascal |

**When to unpin:** After upgrading GPU to Ampere+ (RTX 30xx/40xx) where CUDA 12 is natively supported.

**Do NOT uninstall nvidia-cu12 runtime packages.** torch needs them for CUDA lib loading even on CUDA 11.8.

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
| model_size | tiny, base, small, medium, large-v3, large-v3-turbo, turbo | small | Whisper model size |
| language | en (English), es (Spanish), multi (auto-detect) | multi | Language mode |
| extension_filter | file extension (e.g., m4v, mp4) | all | Process only files with this extension (use "" to skip) |
| diarize | true, false | false | Enable speaker diarization (who said what) |

### Examples

```bash
# English content
./whisper-batch-transcribe.sh ~/Videos/lectures small en

# Spanish content
./whisper-batch-transcribe.sh ~/Videos/spanish medium es

# Mixed/unknown languages (auto-detect)
./whisper-batch-transcribe.sh ~/Videos/mixed large-v3 multi

# Process only .m4v files
./whisper-batch-transcribe.sh ~/Videos/meetings medium multi m4v

# English with speaker diarization
./whisper-batch-transcribe.sh ~/Videos/interviews medium en "" true

# Custom output directory
./whisper-batch-transcribe.sh -o ~/my-transcripts ~/Videos/lectures medium en
```

### Supported Formats

- Video: `.mkv`, `.mp4`, `.m4v`, `.webm`
- Audio: `.mp3`, `.wav`, `.m4a`, `.ogg`

## Output

Transcripts are saved to `output/transcripts_<model>_<language>/` (or custom path via `-o`):

- `.txt` - Plain text
- `.vtt` - WebVTT subtitles
- `.srt` - SRT subtitles
- `.json` - Detailed JSON with timestamps
- `.tsv` - Tab-separated values
- `transcription_<timestamp>.log` - Processing log

Subfolder structure is mirrored from input to output. Files already transcribed are automatically skipped.

## Model Selection Guide

| Model | VRAM | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| tiny | ~1GB | Fastest | Basic | Quick drafts |
| base | ~1GB | Fast | Good | Simple content |
| small | ~2GB | Medium | Better | General use |
| medium | ~5GB | Slow | Great | Quality transcripts |
| large-v3 | ~10GB | Slowest | Best | Professional/multilingual |

For a GTX 1070 (8GB VRAM), `small` or `medium` models work well. `large-v3` is tight but workable without diarization.

## Speaker Diarization

Speaker diarization identifies **who said what** in multi-speaker recordings.

### When to Use

| Use Case | Recommended |
|----------|-------------|
| 2-person interviews/screenings | Yes |
| Podcast with hosts | Yes |
| Solo recordings | No |
| Large group calls (5+) | Maybe (less accurate) |
| Quick bulk transcription | No (slower) |

### Setup

1. **Accept HuggingFace model licenses:**
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

2. **Get HuggingFace token:** https://huggingface.co/settings/tokens

3. **Set environment variable:**
   ```bash
   export HF_TOKEN=your_token_here
   ```
   Add to `~/.bashrc` to persist across sessions.

### Resource Usage

| Mode | VRAM | Speed |
|------|------|-------|
| Transcription only (int8) | ~3GB (medium) | 1x |
| Transcription + Diarization | ~7-8GB | 1.5-2x slower |

### Output

With diarization, transcripts include speaker labels:

```
[SPEAKER_00]: Hello, how are you doing today?
[SPEAKER_01]: I'm doing great, thanks for asking.
```

Output folder gets a `_diarized` suffix to keep separate from plain transcripts.

## Architecture

This tool uses **whisperx as the unified transcription engine** for both plain and diarized modes. Previously it used a dual-engine setup (whisper-ctranslate2 for plain, whisperx for diarized), which caused persistent dependency conflicts. Unified in March 2026.

whisperx uses faster-whisper (CTranslate2 backend) for transcription and pyannote-audio for diarization. Both run on GPU via the same CUDA/cuDNN stack.

## Troubleshooting

**"libnvToolsExt.so.1: cannot open shared object file"**
Install the system package: `sudo apt install libnvtoolsext1`

**"CUDNN_STATUS_EXECUTION_FAILED"**
cuDNN version mismatch. Pin nvidia-cudnn-cu12 to 9.1.0.70:
`pip install nvidia-cudnn-cu12==9.1.0.70`
Then re-run `bash install-cudnn8.sh`.

**"ffmpeg: No such file or directory"**
Install ffmpeg: `sudo apt install ffmpeg`

**Device shows "CPU" instead of CUDA**
Make sure venv is activated (`source venv/bin/activate`) before running the script. The LD_LIBRARY_PATH setup in the script needs the venv nvidia packages to be findable.

**pip dependency conflicts after upgrading whisperx**
whisperx upgrades often pull newer torch versions that break CUDA 11.8 compatibility. After any pip upgrade, always reinstall the pinned torch:
```bash
pip install torch==2.4.0+cu118 torchvision==0.19.0+cu118 torchaudio==2.4.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

## Tips

- Use `en` language flag for English-only content (faster, more accurate)
- `large-v3` handles accents and mixed languages best but needs significant VRAM
- Use diarization selectively on important recordings where speaker ID matters
- Resume interrupted batches safely (already-transcribed files are skipped)
