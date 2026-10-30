# AI Transcriber

Batch audio/video transcription via the [gpu-broker](/opt/brain/src/gpu-broker/)-managed whisperx-server container. No local whisperx, torch, or CUDA install required — the script POSTs each file to the broker's `/transcribe` endpoint and writes the standard 5 output formats from the response.

## Post-Phase-4 architecture (2026-06-14)

Before:

```
~/bigrepo/Mario/Projects/ai-transcriber/
├── venv/                  9.6 GB  (torch + whisperx + pyannote + cuDNN8 + weights)
├── install-cudnn8.sh      cuDNN 8/9 coexistence workaround for CTranslate2
├── whisper-batch-transcribe.sh   loops the local `whisperx` CLI per file
└── requirements.txt       torch 2.4.0+cu118 + whisperx <3.8 + cudnn 9.1.0.70
```

After:

```
/opt/brain/src/ai-transcriber/        (symlinked at ~/src/ai-transcriber)
├── transcribe_via_broker.py   POSTs to broker /transcribe, writes 5 output formats
├── whisper-batch-transcribe.sh  thin dispatcher → transcribe_via_broker.py
├── requirements.txt           `requests>=2.31` (one line)
└── /opt/fast/venvs/ai-transcriber/   19 MB venv (500× smaller)
```

Everything VRAM-related — whisperx large-v3, Pascal-compat torch pin, VAD-SHA, cuDNN8, batch_size=4 — lives inside the broker-managed `gpu-broker/whisperx-server:0.1.0` container. Frozen via Docker image build; no host-side drift possible.

## Requirements

- gpu-broker running on `http://127.0.0.1:8090` (`systemctl --user is-active gpu-broker.service`)
- `/opt/fast/venvs/ai-transcriber/` with `requests` installed (see provisioning below)
- ffmpeg for media inspection (not strictly required for the broker path, but `/transcribe` skill may call it for audio probing)

That's it. No CUDA, no torch, no Python 3.12+, no HF_TOKEN host-side (HF_TOKEN is configured **once** at the broker container's env for diarization).

## Provisioning the venv (one-time)

```bash
python3 -m venv /opt/fast/venvs/ai-transcriber
/opt/fast/venvs/ai-transcriber/bin/pip install -r /opt/brain/src/ai-transcriber/requirements.txt
```

## Usage

```bash
./whisper-batch-transcribe.sh [options] <input_folder> [model_size] [language] [extension_filter] [diarize]
```

Same CLI contract as the legacy bash script — the `/transcribe` skill keeps working without changes.

### Options

| Option | Description |
|--------|-------------|
| `-o`, `--output-dir <path>` | Custom output directory (default: auto-generated in script dir) |

### Parameters

| Parameter | Options | Default | Description |
|-----------|---------|---------|-------------|
| input_folder | path | required | Folder containing media files (subfolders included) |
| model_size | (any of the legacy names) | large-v3 | **Ignored** — broker pins large-v3 server-side via image; a different value logs a WARN |
| language | en, es, auto | auto | `auto` detects **one** language for the whole file; `multi` is a legacy alias for it. See § Language handling |
| extension_filter | file extension (e.g. m4v) | all | Process only files with this extension (use "" to skip) |
| diarize | true, false | false | Enable speaker diarization (requires HF_TOKEN on broker container) |

### Examples

```bash
# English content
./whisper-batch-transcribe.sh ~/Videos/lectures small en

# Spanish content
./whisper-batch-transcribe.sh ~/Videos/spanish medium es

# Unknown language (auto-detects ONE language per file; not a mixed-language mode)
./whisper-batch-transcribe.sh ~/Videos/mixed large-v3 auto

# Process only .m4v files
./whisper-batch-transcribe.sh ~/Videos/meetings medium auto m4v

# English with speaker diarization
./whisper-batch-transcribe.sh ~/Videos/interviews medium en "" true

# Custom output directory
./whisper-batch-transcribe.sh -o ~/my-transcripts ~/Videos/lectures medium en
```

### From another host (`transcribe_remote.sh`)

On a machine without the GPU (COLETTE), run `./transcribe_remote.sh` with the same arguments. It opens an SSH tunnel to the GPU host's broker (8090) and worker (8083), streams each file's audio track to `/transcribe_upload`, and writes outputs locally; nothing is written on the GPU host. Needs 127.0.0.1:8090 and :8083 free locally, a gpu-broker checkout for its client (`GPU_BROKER_SRC`, default `/opt/brain/src/gpu-broker`), and `python3` with `requests`. `TRANSCRIBE_GPU_HOST` picks the ssh alias (default `scully`). First real run 2026-10-30: 31m17s four-speaker meeting, diarized, ~54 s wall on the 3090 (~35x realtime, half of it diarization).

### Language handling

Whisper decodes **one language per file**. Nothing in the output marks when that assumption broke, so know these before trusting a transcript of mixed-language audio (found 2026-10-30, detail and measurements in `FEEDBACK-forced-language-translates-2026-10-30.md`):

| Setting | What happens to speech in another language |
|---|---|
| `en` (forced) | **Dropped** when it overlaps speech in the forced language. No gap in the `.srt`, no marker, fluent text around it: the transcript reads as complete. |
| `es` (forced) on English speech | **Translated** into fluent Spanish and emitted as if transcribed. Tell: English product/UI names left untranslated mid-sentence. |
| `auto` (`multi`) | Detects one language for the file, then behaves exactly like forcing it (byte-identical to `en` on the 2026-10-30 file). |

The log prints each file's resolved language and whether it was forced or detected. **For mixed-language audio, clip and re-run:** transcribe the whole file in its dominant language, cut each foreign-language span with `ffmpeg -ss <start> -to <end>`, transcribe each clip alone with its own language, and splice by timestamp (offset the clip's times by `<start>`). This only works when you know the span exists, so ask the person who was there. A diarizer-vs-transcript cross-check that would flag speaker turns with no text is open in the vault (`gpu-broker-upgrade` checklist § Transcription).

### Supported Formats

- Video: `.mkv`, `.mp4`, `.m4v`, `.webm`
- Audio: `.mp3`, `.wav`, `.m4a`, `.ogg`

### Path constraints

The broker's whisperx-server container bind-mounts `/home/mario`, `/opt/brain`, and `/mnt/bigrepo` (all read-only) — audio paths under any of these resolve directly inside the container. **Symlink gotcha:** the client canonicalizes via `Path.resolve()` before POSTing, so `~/bigrepo/...` (which symlinks to `/mnt/bigrepo/bigrepo/...` on FATEBURN) shows up as the canonical `/mnt/bigrepo/...` path. The `/mnt/bigrepo` mount covers this. Audio elsewhere (e.g. `/tmp/...` or other host trees) needs either:

- a `cp` / `mv` to bring it under one of the mounted trees, OR
- adding the source tree to `volumes` in `/opt/brain/src/gpu-broker/broker/docker_mgr.py` + restarting the broker.

The script warns on non-resolvable paths before POSTing.

### Env overrides

| Var | Default | Purpose |
|---|---|---|
| `GPU_BROKER_URL` | `http://127.0.0.1:8090` | broker base URL (override for remote brokers) |
| `WHISPERX_BATCH_SIZE` | broker default (`roles.yaml`, 16 on the 3090) | per-file batch_size override; rarely needed |
| `TRANSCRIBE_TIMEOUT` | 1800 | per-file timeout seconds |

## Output

Same 5 formats as before, mirrored from input folder structure:

- `.txt` — plain text, one segment per line (with `[SPEAKER_NN]` prefix when diarized)
- `.srt` — SubRip subtitles
- `.vtt` — WebVTT subtitles
- `.json` — full broker `TranscribeResponse` (segments, language, duration, transcribe_ms, align_ms, diarize_ms)
- `.tsv` — tab-separated (start, end, [speaker], text)
- `transcription_<timestamp>.log` — processing log

Already-transcribed files are skipped (existence check on the `.txt`).

## Speaker Diarization

`diarize=true` enables pyannote speaker labels. Requires `HF_TOKEN` to be set in the **broker container's** environment (`/opt/brain/src/gpu-broker/.env` or per docker-compose); the host doesn't need it.

| Use Case | Recommended |
|----------|-------------|
| 2-person interviews/screenings | Yes |
| Podcast with hosts | Yes |
| Solo recordings | No |
| Large group calls (5+) | Maybe (less accurate) |
| Quick bulk transcription | No (slower) |

Resource usage:

| Mode | VRAM (broker-side) | Speed |
|------|---------------------|-------|
| Transcription only | ~5.8 GiB (whisperx large-v3 float16) | 1x |
| Transcription + Diarization | adds pyannote pipeline | 1.5-2x slower |

## Troubleshooting

**"venv not found at /opt/fast/venvs/ai-transcriber/bin/python"**
Provision per the section above.

**"broker infeasible — role disabled in roles.yaml: transcribe"**
The broker has `transcribe.enabled: false` in `config/roles.yaml` — flip to `true` and `systemctl --user restart gpu-broker.service`.

**"broker — acquire transport error"**
The broker isn't running. `systemctl --user start gpu-broker.service`, then `curl http://127.0.0.1:8090/health`.

**"HTTP 400 — audio_path not found inside container"**
Audio path is outside the broker's bind mounts. Move under `/home/mario/` or `/opt/brain/`, or extend `docker_mgr.py` to add the source tree.

**Already-transcribed files re-processed**
Check `.txt` exists at the mirrored output path. The auto-skip is based solely on `.txt`'s existence; `.srt`/`.vtt`/`.json`/`.tsv` are regenerated if `.txt` is missing.

## Rollback path

The original local-whisperx repo is preserved at `~/bigrepo/Mario/Projects/ai-transcriber/` (including its 9.6 GB venv). To revert: update the `/transcribe` skill to point back at `~/bigrepo/Mario/Projects/ai-transcriber/whisper-batch-transcribe.sh`. Plan to remove the old copy 1 week after Phase 4 cutover lands green.

## History

- **Pre-2026-06-14:** local whisperx 3.3.2 + torch 2.4.0+cu118 + pyannote 3.3.2 + cuDNN8/9 coexistence at `~/bigrepo/Mario/Projects/ai-transcriber/`.
- **2026-06-14 (this revision):** repo moved to `/opt/brain/src/ai-transcriber/`, venv shrunk from 9.6 GB to 19 MB, script rewritten to POST to gpu-broker `/transcribe`. Phase 4 Task #4 of [gpu-model-broker](../../repo/Mind/projects/active/gpu-model-broker.md).
