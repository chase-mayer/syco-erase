"""Build opinion/neutral prompt pairs from Anthropic's model-written sycophancy evals.

Each source item is one prompt containing (a) a first-person biography that states an
opinion and (b) a multiple-choice question. BES needs the original ("opinion") prompt and
the same item with the biography removed ("neutral"), which is the oracle reference.

The three files hold 30,051 items but only 158 distinct questions (each question is
repeated with many biographies), so every split downstream is grouped by `qid`.

Usage:
    python -m src.data --dataset all
    python -m src.data --dataset political --review 25
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FILES = {
    "nlp": "sycophancy_on_nlp_survey.jsonl",
    "phil": "sycophancy_on_philpapers2020.jsonl",
    "political": "sycophancy_on_political_typology_quiz.jsonl",
}

STEMS = {
    "nlp": [r"Do you agree or disagree with the following claim"],
    "phil": [r"What is your view on the following topic:"],
    "political": [
        r"Which of the following statements comes? closest to your view",
        r"How much of a problem would you say",
        r"If you had to choose, would you rather",
        r"In general, would you say experts",
        r"How much more, if anything, needs",
        r"How much, if at all, would",
        r"How do you feel toward (?:Democrats|Republicans)",
        r"Which of these statements best describes",
        r"In general, how much do White",
        r"Do you think greater social acceptance",
        r"Overall, would you say people who",
        r"In the future, do you think",
        r"Thinking about increased trade of goods and services",
    ],
}

OPTION_RE = re.compile(r"(?:^|\n)\s*\(([A-Z])\)")


@dataclass
class Pair:
    uid: str
    qid: str
    dataset: str
    bio: str
    question: str
    opinion_prompt: str
    neutral_prompt: str
    options: list[str]
    matching: str
    not_matching: str
    not_matching_all: list[str]
    split_ok: bool


def option_letters(text: str) -> list[str]:
    """Answer letters that appear as '(X)' at the start of a line, in order, deduplicated."""
    seen: list[str] = []
    for letter in OPTION_RE.findall(text):
        if letter not in seen:
            seen.append(letter)
    return seen


def question_id(dataset: str, question: str) -> str:
    return f"{dataset}-" + hashlib.sha1(question.encode()).hexdigest()[:10]


def _split(text: str, dataset: str) -> tuple[str, str, bool]:
    """Return (bio, question_block, split_ok)."""
    for stem in STEMS[dataset]:
        m = re.search(stem, text)
        if m:
            return text[: m.start()].strip(), text[m.start():].strip(), True
    head = re.split(r"\n\s*\(A\)|\n\nChoices:", text)[0]
    tail = text[len(head):]
    sents = re.split(r"(?<=[.!?]) ", head.strip())
    return " ".join(sents[:-1]).strip(), (sents[-1] + tail).strip(), False


def _letters(x) -> list[str]:
    xs = x if isinstance(x, list) else [x]
    return [s.strip().strip("()") for s in xs]


def build(dataset: str, limit: int | None = None) -> list[Pair]:
    path = RAW / FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run scripts/download_data.sh")
    pairs: list[Pair] = []
    with path.open() as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            row = json.loads(line)
            bio, q, ok = _split(row["question"], dataset)
            match = _letters(row["answer_matching_behavior"])[0]
            others = _letters(row["answer_not_matching_behavior"])
            pairs.append(Pair(
                uid=f"{dataset}-{i}",
                qid=question_id(dataset, q),
                dataset=dataset,
                bio=bio,
                question=q,
                opinion_prompt=row["question"],
                neutral_prompt=q,
                options=option_letters(q),
                matching=match,
                not_matching=others[0],
                not_matching_all=others,
                split_ok=ok,
            ))
    return pairs


def write(pairs: list[Pair], dataset: str) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / f"pairs_{dataset}.jsonl"
    with out.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(asdict(p)) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(FILES) + ["all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--review", type=int, default=0, help="print N fallback splits")
    args = ap.parse_args()

    names = list(FILES) if args.dataset == "all" else [args.dataset]
    for name in names:
        pairs = build(name, args.limit)
        out = write(pairs, name)
        bad = [p for p in pairs if not p.split_ok]
        n_q = len({p.qid for p in pairs})
        bad_opts = sum(p.matching not in p.options for p in pairs)
        print(f"{name}: {len(pairs)} pairs, {n_q} unique questions -> {out.name}  "
              f"(stem match {1 - len(bad) / len(pairs):.1%}, label/option mismatches {bad_opts})")
        for p in bad[: args.review]:
            print("  FALLBACK bio:", p.bio[-120:])
            print("  FALLBACK  q :", p.question[:120], "\n")


if __name__ == "__main__":
    main()
