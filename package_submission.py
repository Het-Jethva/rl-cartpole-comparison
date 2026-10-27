"""Build the submission archive.

Produces `submission/rl-cartpole-comparison.zip` containing the two deliverables
(the code and the paper) plus the raw results, so a marker can check the figures
against the data without running anything.

The private study note is excluded deliberately, as are the virtual environment,
caches and LaTeX build artefacts.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT = ROOT / "submission" / "rl-cartpole-comparison.zip"

# (source, destination inside the archive)
INCLUDE_FILES = [
    (ROOT / "README.md", "README.md"),
    (ROOT / "README.md", "code/README.md"),
    (ROOT / "pyproject.toml", "code/pyproject.toml"),
    (ROOT / "uv.lock", "code/uv.lock"),
    (ROOT / "run.py", "code/run.py"),
    (ROOT / "package_submission.py", "code/package_submission.py"),
    (ROOT / "paper" / "paper.pdf", "paper.pdf"),
]

INCLUDE_TREES = [
    (ROOT / "rlcompare", "code/rlcompare"),
    (ROOT / "tests", "code/tests"),
    (ROOT / "configs", "code/configs"),
    (ROOT / "results", "results"),
    (ROOT / "paper", "paper"),
]

EXCLUDED_DIRECTORIES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv", ".git"}
EXCLUDED_SUFFIXES = {".aux", ".log", ".out", ".bbl", ".blg", ".synctex.gz", ".fls", ".fdb_latexmk", ".pyc"}
EXCLUDED_NAMES = {"NOTES.md"}


def _keep(path: Path) -> bool:
    if path.name in EXCLUDED_NAMES:
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False
    return not any(part in EXCLUDED_DIRECTORIES for part in path.parts)


def main() -> int:
    missing = [str(source) for source, _ in INCLUDE_FILES if not source.is_file()]
    if missing:
        print("Cannot package, these are missing:")
        for item in missing:
            print(f"  {item}")
        print("\nRun `python run.py all` and build the PDF first.")
        return 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, destination in INCLUDE_FILES:
            archive.write(source, destination)
            written += 1
        for source_root, destination_root in INCLUDE_TREES:
            if not source_root.is_dir():
                continue
            for path in sorted(source_root.rglob("*")):
                if not path.is_file() or not _keep(path):
                    continue
                archive.write(path, f"{destination_root}/{path.relative_to(source_root)}")
                written += 1

    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"Wrote {OUTPUT.relative_to(ROOT)}  ({written} files, {size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
