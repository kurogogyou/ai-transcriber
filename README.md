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
./whisper-batch-transcribe.sh <input_folder> [model_size] [language]
```

### Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| input_folder | path | required | Folder containing media files |
| model_size | tiny, base, small, medium, large-v3 | small | Whisper model size |
| language | en, es, multi | multi | Language mode |

### Examples

```bash
# English content (uses optimized .en model)
./whisper-batch-transcribe.sh ~/Videos/lectures small en

# Spanish content
./whisper-batch-transcribe.sh ~/Videos/spanish medium es

# Mixed/unknown languages (auto-detect)
./whisper-batch-transcribe.sh ~/Videos/mixed large-v3 multi
```

### Supported Formats

- Video: `.mkv`, `.mp4`, `.webm`
- Audio: `.mp3`, `.wav`, `.m4a`, `.ogg`

## Output

Transcripts are saved to `<input_folder>/transcripts_<model>_<language>/` with:

- `.txt` - Plain text
- `.vtt` - WebVTT subtitles
- `.srt` - SRT subtitles
- `.json` - Detailed JSON with timestamps
- `.tsv` - Tab-separated values
- `transcription.log` - Processing log

## Model Selection Guide

| Model | VRAM | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| tiny | ~1GB | Fastest | Basic | Quick drafts |
| base | ~1GB | Fast | Good | Simple content |
| small | ~2GB | Medium | Better | General use |
| medium | ~5GB | Slow | Great | Quality transcripts |
| large-v3 | ~10GB | Slowest | Best | Professional/multilingual |

For a GTX 1070 (8GB VRAM), `small` or `medium` models work well.

## Tips

- Use `.en` models (via `en` language flag) for English-only content - they're faster and more accurate
- For mixed English/Spanish, use `multi` to let Whisper auto-detect
- The `large-v3` model requires significant VRAM but handles accents and mixed languages best
