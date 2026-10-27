# Rerun audit

The full study completed with exit code 0. All 233 configured runs have matching
CSV and JSON files. The existing test suite passed all 28 tests before training.
The figures and tables were regenerated, and the eight-page paper compiled and
passed visual inspection. The submission archive contains 507 entries, including
all 466 raw result files, with no CRC errors.

The generated numerical macros match the original report except for DQN time,
which changed from 214 to 180 seconds, and REINFORCE time, which changed from 46
to 39 seconds. These are measurements from this machine and concurrent workload.
They do not establish an implementation speedup.

The README and paper now distinguish individual episode scores from evaluation
means. They also describe the selected tabular grid accurately and record the
machine used for timing. The original paper and settings remain in `paper-before/`
and `tuned-before.yaml` beside this audit.

## Evidence

- `tests.log` records the 28 passing tests.
- `study.log` and `study-exit.txt` record the completed experiment pipeline.
- `verification.json` records validation of all 233 runs and 23,300,000 training steps.
- `numbers-comparison.json` records the two changed timing macros.
- `report-claim-checks.json` checks the clarified tabular claims.
- `paper-build.log` records the successful final PDF compilation.
- `paper-contact-sheet.png` and `rendered/` contain the inspected PDF pages.
- `archive-verification.json` records archive counts and SHA256.
- `download-verification.log` records verification of all 32 downloaded wheel hashes.
- `uv-sync.log` and `package.log` record successful environment setup and packaging.
- `decisions.tsv` records decisions, checkpoints, and the corrected interpretation
  of individual seed results versus aggregate statistics.

## Principles applied

Prove It Works led to checking the actual CSV, JSON, PDF, and ZIP contents.
Build the Lever produced `verify_results.py` so the data checks can be rerun.
Sequence Work into Verifiable Units led to checking tests, tuning, and the
completed comparison and sweeps before final verification and packaging.

## Attention

Reviewed by GPT-6 Astra. No blocking findings.

The reviewer independently rechecked the configured runs, archive, saved logs,
and diff. Its two evidence-recording notes were addressed with completion rows,
a saved packaging log, and another archive hash check. The rebuilt archive has
the same SHA256 as the independently reviewed archive at that checkpoint.
The subsequent GitHub publication updates the README and rebuilds the archive.
Its final contents and SHA256 are checked in `publication-verification.json`.

No workspace transcript was available, so the audit cannot establish that every
conversational pivot was logged. This was a review by a different model within
the GPT family. A reviewer from another model family was unavailable.
