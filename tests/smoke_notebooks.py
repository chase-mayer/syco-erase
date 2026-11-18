"""Execute every notebook end to end on a tiny random model (CPU, no downloads besides data).

    python -m tests.smoke_notebooks [notebook-name-substring ...]

Catches code errors before spending GPU time; the numbers it produces are meaningless. A
throwaway MMLU cache is written for the run and removed afterwards, and results go to a
temporary folder so real results are never overwritten.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from src import experiment as E
from tests.tiny import build

ROOT = Path(__file__).resolve().parents[1]


def main(filters: list[str]) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="bes_smoke_"))
    model_dir = build(str(tmp / "tiny_model"))
    mmlu = E.RAW / "mmlu_sample.jsonl"
    wrote_mmlu = not mmlu.exists()
    if wrote_mmlu:
        mmlu.parent.mkdir(parents=True, exist_ok=True)
        with mmlu.open("w") as fh:
            for i in range(40):
                fh.write(json.dumps({"question": f"Question {i}?", "choices": ["w", "x", "y", "z"],
                                     "answer": i % 4, "subject": "smoke"}) + "\n")
    backup = None
    if E.RESULTS.exists():
        backup = tmp / "results_backup"
        shutil.copytree(E.RESULTS, backup)
    os.environ.update({"BES_SMOKE": "1", "BES_MODEL": model_dir, "BES_LLAMA_MODEL": model_dir})
    failed = []
    try:
        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            if filters and not any(f in path.name for f in filters):
                continue
            book = nbformat.read(path, as_version=4)
            try:
                NotebookClient(book, timeout=1800, kernel_name="python3",
                               resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
                print("ok  ", path.name)
                if os.environ.get("BES_SMOKE_OUT"):
                    nbformat.write(book, Path(os.environ["BES_SMOKE_OUT"]) / path.name)
            except Exception as exc:
                failed.append(path.name)
                print("FAIL", path.name, "\n", str(exc)[-3000:])
                if os.environ.get("BES_SMOKE_OUT"):  # keep the partial run so the failing cell can be inspected
                    nbformat.write(book, Path(os.environ["BES_SMOKE_OUT"]) / path.name)
    finally:
        if wrote_mmlu:
            mmlu.unlink()
        if E.RESULTS.exists():
            shutil.rmtree(E.RESULTS)
        if backup is not None:
            shutil.copytree(backup, E.RESULTS)
        shutil.rmtree(tmp, ignore_errors=True)
    if failed:
        sys.exit(f"{len(failed)} notebook(s) failed: {failed}")
    print("all notebooks executed")


if __name__ == "__main__":
    main(sys.argv[1:])
