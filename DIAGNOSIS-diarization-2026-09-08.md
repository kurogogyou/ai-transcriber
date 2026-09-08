# Diagnosis: `--diarize` fails, and the wrapper reports success anyway — RESOLVED 2026-09-08

All three reported defects are closed, plus a fourth this report did not name. **Two of the three root causes stated here were misattributed** — the wrapper's exit code was correct throughout (the skill never read it), and the torch 2.7 shim was already committed and running.

The corrected root-cause analysis, the fixes, and the end-to-end verification are canonical in the vault:

    Mind/projects/active/gpu-broker-upgrade/context/whisperx-diarization-defects-2026-09.md

Verified 2026-09-08: `gpu-broker/whisperx-server:0.2.1`, original failing file, exit 0, 303 segments with word timings and speaker labels, 84.3 s for 31 minutes of audio.

This stub is kept only so the filename does not read as an open issue. Do not extend it — write to the vault file.
