"""Batch transcribe media files via the gpu-broker.

Drop-in replacement for the pre-Phase-4 whisperx CLI loop. No local whisperx
install, no torch, no CUDA/cuDNN host-side. Each file is POSTed to the
broker-managed `whisperx-server:0.1.0` container at the endpoint returned by
`broker.acquire("transcribe")`; the broker handles model loading, batch_size
pinning (4 server-side), eviction (rerank), and idle-shutdown.

The single hold around the batch loop keeps `active_handles > 0` so the
whisperx worker never idle-shuts mid-batch (idle_shutdown_seconds: 30 in
roles.yaml).

Output: writes the same 5 formats as the legacy CLI (.txt .srt .vtt .json
.tsv), reconstructed from the broker's TranscribeResponse.segments[]. The
auto-skip-already-done behavior is preserved.

CLI compatibility: positional args + `-o` flag match the legacy bash script
exactly, so the /transcribe skill keeps working without rewriting its
command-builder.

Run: python -m transcribe_via_broker [-o <output_dir>] [--upload] <input_folder>
                                     [model_size] [language] [extension_filter] [diarize]

--upload (added 2026-10-27) is for running on a host other than the GPU box,
through transcribe_remote.sh, which opens the SSH tunnel this needs. The audio
track is stream-copied out of each file and sent as the request body, so the
recording never lands on the GPU host and raw output is written locally.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

BROKER_URL = os.environ.get("GPU_BROKER_URL", "http://127.0.0.1:8090")
ALL_EXTS = ["mkv", "mp4", "m4v", "webm", "mp3", "wav", "m4a", "ogg"]
DEFAULT_BATCH_SIZE = int(os.environ.get("WHISPERX_BATCH_SIZE", "0")) or None  # None = server default
TRANSCRIBE_TIMEOUT = float(os.environ.get("TRANSCRIBE_TIMEOUT", "1800"))  # 30 min ceiling per file


# Make the broker client importable. GPU_BROKER_SRC lets a host other than the
# GPU box point at its own checkout (only client/ is used, which needs requests).
sys.path.insert(0, os.environ.get("GPU_BROKER_SRC", "/opt/brain/src/gpu-broker"))
from client.gpu_broker_client import GpuBrokerClient, BrokerError, InfeasibleError


def discover_files(input_folder: Path, ext_filter: str | None) -> list[Path]:
    """Return all media files under input_folder, recursively, sorted."""
    exts = [ext_filter] if ext_filter else ALL_EXTS
    files: list[Path] = []
    for ext in exts:
        files.extend(input_folder.rglob(f"*.{ext}"))
    return sorted(set(files))


def output_dir_for(file: Path, input_root: Path, output_root: Path) -> tuple[Path, str]:
    """Mirror input subfolder structure into output_root. Return (dir, display_name)."""
    rel = file.parent.relative_to(input_root) if input_root in file.parents else Path()
    if rel == Path():
        return output_root, file.name
    return output_root / rel, f"{rel}/{file.name}"


def fmt_srt_time(seconds: float) -> str:
    """SRT timestamp: HH:MM:SS,mmm"""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    ms = int((s - int(s)) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


def fmt_vtt_time(seconds: float) -> str:
    """WebVTT timestamp: HH:MM:SS.mmm"""
    return fmt_srt_time(seconds).replace(",", ".")


def write_outputs(response: dict, output_dir: Path, basename: str, diarize: bool) -> None:
    """Write .txt .srt .vtt .json .tsv from a TranscribeResponse payload."""
    output_dir.mkdir(parents=True, exist_ok=True)
    segments = response.get("segments", [])

    def speaker_prefix(seg: dict) -> str:
        spk = seg.get("speaker")
        return f"[{spk}] " if diarize and spk else ""

    # .txt — plain text, one segment per line (optionally with speaker tag for diarized)
    (output_dir / f"{basename}.txt").write_text(
        "\n".join(f"{speaker_prefix(s)}{s['text'].strip()}" for s in segments) + "\n",
        encoding="utf-8",
    )

    # .srt
    srt_lines: list[str] = []
    for i, s in enumerate(segments, 1):
        srt_lines.append(str(i))
        srt_lines.append(f"{fmt_srt_time(s['start'])} --> {fmt_srt_time(s['end'])}")
        srt_lines.append(f"{speaker_prefix(s)}{s['text'].strip()}")
        srt_lines.append("")
    (output_dir / f"{basename}.srt").write_text("\n".join(srt_lines), encoding="utf-8")

    # .vtt
    vtt_lines = ["WEBVTT", ""]
    for s in segments:
        vtt_lines.append(f"{fmt_vtt_time(s['start'])} --> {fmt_vtt_time(s['end'])}")
        vtt_lines.append(f"{speaker_prefix(s)}{s['text'].strip()}")
        vtt_lines.append("")
    (output_dir / f"{basename}.vtt").write_text("\n".join(vtt_lines), encoding="utf-8")

    # .json — full broker response (timestamps, words, diarization, language, duration)
    (output_dir / f"{basename}.json").write_text(
        json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # .tsv — start \t end \t text (+ speaker col when diarized)
    tsv_lines = ["start\tend\tspeaker\ttext" if diarize else "start\tend\ttext"]
    for s in segments:
        cols = [f"{s['start']:.3f}", f"{s['end']:.3f}"]
        if diarize:
            cols.append(s.get("speaker") or "")
        cols.append(s["text"].strip())
        tsv_lines.append("\t".join(cols))
    (output_dir / f"{basename}.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")


_BROKER_MOUNTS = ("/home/mario/", "/opt/brain/", "/mnt/bigrepo/")


def _check_path_accessible_to_container(audio: Path) -> None:
    """Whisperx-server container only sees the host trees listed in
    _BROKER_MOUNTS (RO). Audio paths elsewhere will fail with HTTP 400."""
    p = str(audio.resolve())
    if not any(p.startswith(m) for m in _BROKER_MOUNTS):
        sys.stderr.write(
            f"WARN: audio path {p} is outside the broker's bind mounts "
            f"({', '.join(m.rstrip('/') for m in _BROKER_MOUNTS)}). "
            f"Worker may return HTTP 400. Either move the audio under one of "
            f"those trees, or extend the mounts in "
            f"gpu-broker/broker/docker_mgr.py.\n"
        )


class AudioExtractError(RuntimeError):
    pass


def audio_stream(file: Path, chunk_bytes: int = 1 << 20):
    """Yield the file's audio track remuxed to Matroska, for /transcribe_upload.

    `-c:a copy`, not a re-encode: the worker decodes it with the same ffmpeg
    command whisperx.load_audio() runs, and that was verified bit-identical to
    the path API on 2026-10-27. Converting here (FLAC at 16 kHz) was not, since
    this host's ffmpeg resamples differently from the image's. Matroska because
    it streams; an MP4 whose index sits at the end cannot be read from a pipe.
    """
    proc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(file),
         "-vn", "-sn", "-dn", "-c:a", "copy", "-f", "matroska", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        for chunk in iter(lambda: proc.stdout.read(chunk_bytes), b""):
            yield chunk
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace").strip()
        if proc.wait() != 0:
            raise AudioExtractError(f"ffmpeg could not extract audio from {file}: {err[-500:]}")


def main() -> int:
    p = argparse.ArgumentParser(
        prog="transcribe_via_broker",
        description="Batch transcribe via gpu-broker /transcribe (replaces local whisperx).",
    )
    p.add_argument("-o", "--output-dir", default=None,
                   help="Custom output directory (default: auto-generated in script dir)")
    p.add_argument("--upload", action="store_true",
                   help="Send the audio bytes instead of a path (run via transcribe_remote.sh)")
    p.add_argument("input_folder", help="Folder containing media files (recursed)")
    p.add_argument("model_size", nargs="?", default="small",
                   help="(legacy positional; ignored — broker pins the model server-side)")
    p.add_argument("language", nargs="?", default="multi",
                   help="en | es | multi (auto-detect)")
    p.add_argument("extension_filter", nargs="?", default="",
                   help='Only process .EXT files (e.g. "m4v"); empty = all extensions')
    p.add_argument("diarize", nargs="?", default="false",
                   help='"true" to enable speaker diarization (requires HF_TOKEN in broker container)')
    args = p.parse_args()

    input_root = Path(args.input_folder).resolve()
    if not input_root.is_dir():
        sys.stderr.write(f"ERROR: input folder not found: {input_root}\n")
        return 1

    lang = None if args.language == "multi" else args.language
    diarize = args.diarize.lower() == "true"
    ext_filter = args.extension_filter or None

    script_dir = Path(__file__).resolve().parent
    if args.output_dir:
        output_root = Path(args.output_dir).resolve()
    else:
        suffix = f"transcripts_{args.model_size}_{args.language}"
        if diarize:
            suffix += "_diarized"
        output_root = script_dir / "output" / suffix
    output_root.mkdir(parents=True, exist_ok=True)

    log_path = output_root / f"transcription_{time.strftime('%Y-%m-%d_%H%M%S')}.log"
    log_fh = log_path.open("w", encoding="utf-8")

    def log(msg: str) -> None:
        print(msg)
        log_fh.write(msg + "\n")
        log_fh.flush()

    log("=" * 40)
    log("Whisper Batch Transcription (via gpu-broker)")
    log("=" * 40)
    log(f"Input:    {input_root}")
    log(f"Model:    broker-pinned (whisperx large-v3 in container)")
    log(f"Language: {args.language} ({lang or 'auto-detect'})")
    log(f"Diarize:  {'enabled' if diarize else 'disabled'}")
    log(f"Output:   {output_root}")
    log(f"Log:      {log_path}")
    log(f"Broker:   {BROKER_URL}")
    log(f"Mode:     {'upload (audio streamed, nothing written on the GPU host)' if args.upload else 'path (file must be on the GPU host)'}")
    log("=" * 40)

    files = discover_files(input_root, ext_filter)
    log(f"Found {len(files)} media files")
    if not files:
        return 0

    broker = GpuBrokerClient(base_url=BROKER_URL, client_id="ai-transcriber")
    processed = 0
    skipped = 0
    failed = 0
    t_total = time.time()

    try:
        with broker.acquire("transcribe", hold_seconds=int(TRANSCRIBE_TIMEOUT * len(files))) as h:
            log(f"acquired transcribe endpoint={h.endpoint} cold_start_ms={h.cold_start_ms} "
                f"evicted={h.evicted}")

            # Preflight the capability before spending the batch on it. Asking
            # the worker whether it CAN diarize costs one HTTP round trip; not
            # asking cost a 57s cold load per file and an opaque 500 on
            # 2026-09-08. Only gates when --diarize was actually requested.
            if diarize:
                try:
                    hz = requests.get(f"{h.endpoint}/health", timeout=30).json()
                except (requests.RequestException, ValueError) as exc:
                    log(f"WARN: could not preflight /health ({exc}); proceeding anyway")
                else:
                    if hz.get("diarize_available") is False:
                        log("\nERROR: diarization requested but the worker cannot "
                            "diarize — aborting before processing any files.")
                        log(f"  worker says: {hz.get('diarize_error')}")
                        log_fh.close()
                        return 3
                    if "diarize_available" not in hz:
                        log("WARN: worker predates the diarize capability probe "
                            "(image < 0.2.1); cannot preflight. Proceeding.")
            for i, file in enumerate(files, 1):
                file_output_dir, display = output_dir_for(file, input_root, output_root)
                basename = file.stem
                txt_target = file_output_dir / f"{basename}.txt"
                if txt_target.exists():
                    skipped += 1
                    log(f"[{i}/{len(files)}] Skipping (already done): {display}")
                    continue

                if not args.upload:
                    _check_path_accessible_to_container(file)
                log(f"[{i}/{len(files)}] Processing: {display}")
                t_file = time.time()
                try:
                    if args.upload:
                        params = {"align": "true", "diarize": str(diarize).lower()}
                        if lang:
                            params["language"] = lang
                        if DEFAULT_BATCH_SIZE is not None:
                            params["batch_size"] = DEFAULT_BATCH_SIZE
                        r = requests.post(
                            f"{h.endpoint}/transcribe_upload", params=params,
                            data=audio_stream(file), timeout=TRANSCRIBE_TIMEOUT,
                            headers={"Content-Type": "application/octet-stream"},
                        )
                    else:
                        payload = {
                            "audio_path": str(file.resolve()),
                            "language": lang,
                            "align": True,
                            "diarize": diarize,
                        }
                        if DEFAULT_BATCH_SIZE is not None:
                            payload["batch_size"] = DEFAULT_BATCH_SIZE
                        r = requests.post(
                            f"{h.endpoint}/transcribe", json=payload, timeout=TRANSCRIBE_TIMEOUT,
                        )
                    r.raise_for_status()
                    response = r.json()
                except AudioExtractError as exc:
                    log(f"  FAILED: {exc}")
                    failed += 1
                    continue
                except requests.RequestException as exc:
                    # Surface the server's own message, not just the status line.
                    # A bare "500 Server Error" cost a whole debugging session on
                    # 2026-09-08: the body carried LocalEntryNotFoundError (missing
                    # pyannote weights) and was thrown away here.
                    detail = ""
                    resp = getattr(exc, "response", None)
                    if resp is not None:
                        try:
                            body = resp.json()
                            detail = body.get("detail") or json.dumps(body)[:500]
                        except ValueError:
                            detail = resp.text[:500]
                    log(f"  FAILED: {exc}" + (f"\n    server said: {detail}" if detail else ""))
                    failed += 1
                    continue

                write_outputs(response, file_output_dir, basename, diarize)
                processed += 1
                dt = time.time() - t_file
                log(f"  Completed in {dt:.1f}s "
                    f"(audio={response.get('duration_s', 0):.1f}s, "
                    f"transcribe_ms={response.get('transcribe_ms', 0):.0f})")
    except InfeasibleError as e:
        log(f"\nERROR: broker infeasible — {e.reason} "
            f"(free={e.free_mb} MiB, would_evict={e.would_need_evict})")
        log("Resolution: release any held handles or /admin/unpin a pinned role.")
        return 2
    except BrokerError as e:
        log(f"\nERROR: broker — {e}")
        return 2

    dt_total = time.time() - t_total
    log("=" * 40)
    # The summary must not read the same on a failed run as on a good one. On
    # 2026-09-08 a run that transcribed nothing printed a bare "Done:" line and
    # was taken for success by two consecutive readers.
    if failed:
        log(f"FAILED: {failed} of {len(files)} file(s) did NOT transcribe "
            f"({processed} processed, {skipped} skipped) — exit code 1")
    elif processed == 0:
        log(f"NOTHING DONE: 0 processed, {skipped} skipped "
            f"(all inputs already had a .txt in the output dir) — exit code 0")
    else:
        log(f"Done: {processed} processed, {skipped} skipped, {failed} failed")
    log(f"Total: {dt_total / 60:.1f}m ({dt_total:.0f}s)")
    log(f"Output: {output_root}")
    log("=" * 40)
    log_fh.close()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
