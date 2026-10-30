# A forced `language` silently drops overlapping speech in other languages, and translates the rest

Found 2026-10-30 transcribing a 31m17s four-speaker meeting recording in which one participant spoke a few sentences of Spanish to someone off-call, **while two other participants carried on talking in English**. Two distinct defects came out of it. The first is the serious one.

## 1. Overlapping non-matching speech is discarded, leaving no trace

With `language=en`, the Spanish utterances are **entirely absent** from the output. Not mistranscribed, not translated: gone. The model locked onto the dominant English speakers and dropped the other voice.

What makes this bad is that it is undetectable from the artefacts. The other two speakers were talking continuously across that whole window, so:

- there is **no gap** in the `.srt` — the timeline is covered end to end;
- there is **no marker**, no low-confidence segment, nothing in the `.json`;
- the surrounding text is fluent and on-topic, so the transcript **reads as complete**.

I had already swept all 31 minutes and measured audio energy in every untranscribed gap ≥10s before the speaker told me roughly when it happened. That approach cannot find this, because the audio is never silent. Only re-transcribing the window as an isolated clip with `language=es` recovered the utterances — three of them, about 10 seconds of speech in total.

**The failure mode is that a forced language can silently delete a participant from the record**, and the output is indistinguishable from that person having said nothing. Here it was a household interruption and nothing was lost. Had it been a participant stating a commitment in their own language over someone else, it would have vanished with nothing to suggest anything was missing.

## 2. A forced `language` translates rather than transcribes

Three passes over the same file, same model, same diarization, differing only in the `language` positional:

| pass | `language` arg | `language` in JSON | result |
|---|---|---|---|
| 1 | `en` | `en` | English transcript; Spanish span missing (defect 1) |
| 2 | `es` | `es` | **the English speech rendered as fluent Spanish prose** |
| 3 | `multi` | `en` | byte-identical to pass 1 |

Pass 2 did not transcribe the Spanish and skip the English; it translated the English into Spanish and emitted it as though transcribed. The output is idiomatic throughout, including over stretches that are unambiguously English. The tell is that English product nouns and UI control names survive untranslated, embedded mid-sentence in otherwise clean Spanish, which is a shape no Spanish speaker produces and no honest transcript should contain.

Nothing marks a translated segment. The `.json` reports only `language: es`, which reads as a successful detection rather than an instruction obeyed too literally.

## `multi` is not a multilingual mode

`transcribe_via_broker.py:201` maps `multi` to `language=None`, i.e. auto-detect, which resolves **one** language for the whole file (`language` is a single top-level JSON field, never per-segment). Here it picked `en` and produced output byte-identical to the forced-`en` pass — including the same dropped Spanish.

So for mixed-language audio there is **no correct single-pass setting**. `en`/`multi` drop or translate the foreign span; `es` translates everything else. The name `multi` actively suggests otherwise, which is how I reached for it.

## Suggestions, roughly in order of value

1. **Treat the drop as a correctness bug, not a tuning issue.** Even a crude mitigation would help: a second detection pass over the audio, or a diarization-aware check for speaker turns present in the diarizer's output but absent from the transcript. The diarizer and the transcriber disagree here, and nothing compares them.
2. **Print the resolved language in the run log.** It is already in the JSON and never surfaced. One line (`Detected language: en`) would make pass 3's behaviour obvious immediately instead of after a diff.
3. **Warn when a forced `language` disagrees with detection.** Detect regardless, and if the caller forced something else, say so on stderr. That converts a silent transformation into a visible one.
4. **Rename `multi`, or make it real.** Either call it `auto` to match what it does, or do per-segment detection. The current name is the trap.
5. **Document clip-and-rerun** as the supported answer for mixed-language audio: cut the foreign span, transcribe it alone with its language, splice by timestamp. It is currently the only thing that works, and it only works if you already know the span is there.
6. **`model_size` is accepted and ignored.** `transcribe_via_broker.py:187` says so (`legacy positional; ignored — broker pins the model server-side`), but it is positional 2, so every caller passes something and reasonably believes it matters. Warn on a value that differs from the pinned model, or drop the positional.
7. **The `-o` output directory is not created.** Every run needed a prior `mkdir -p`. One `os.makedirs(..., exist_ok=True)` removes a stumble.

## Measurements, for reference

RTX 3090, upload mode, `large-v3`, diarization on, audio 1877.4s:

| pass | wall | transcribe_ms | align_ms | diarize_ms | cold start |
|---|---|---|---|---|---|
| `en` | 53.4s | 22853 | 5957 | 23002 | 8435 ms |
| `es` | 54.7s | 24342 | 5693 | 22793 | 8429 ms |
| `multi` | 53.7s | 23122 | 5938 | 23027 | 8430 ms |

Roughly 35x realtime including diarization, which is about half the total. Language choice costs nothing, so defaulting to detection is free. The 70s recovery clip took 3.0s wall.

Diarization produced **5** `SPEAKER_` labels for **4** actual participants; one label held 6 scattered cues that all belonged to a single other speaker, confirmed by the user listening back. Consistent with the known tendency to over-split, noted as another data point.

## Not a defect

The streaming/no-staging path worked exactly as intended across all four runs: tunnel opened and closed cleanly, nothing written on the GPU host, `exit=0`, `Done: 1 processed, 0 skipped, 0 failed`, five non-empty artefacts each time, last subtitle at 31:10 against a 31:17 source.

## Triage (2026-10-30, MIKASA brain session)

- **Done in this commit:** 2 (each file logs `Language: <x> (forced|detected)`), 4 (`auto` is the name and default; `multi` kept as an alias that WARNs), 5 (README § Language handling documents clip-and-rerun), 6 (a `model_size` other than the pinned `large-v3` WARNs; default changed to `large-v3`).
- **3 deferred:** a forced language skips detection inside the worker, so the client has nothing to compare against. Needs the worker to return detected language alongside the forced one (whisperx-server image change); tracked with 1.
- **1 open as a design item:** diarizer-vs-transcript cross-check, in the vault `gpu-broker-upgrade` checklist § Transcription.
- **7 not reproduced:** `transcribe_via_broker.py` already does `output_root.mkdir(parents=True, exist_ok=True)`; the 11:44 smoke test ran with a nonexistent `-o` and created it. The pre-emptive `mkdir -p` was harmless.
