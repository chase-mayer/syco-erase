"""Build opinion/neutral prompt pairs from Anthropic's model-written sycophancy evals.

Each source item is a single prompt that contains (a) a first-person biography ending in a
stated opinion and (b) a question with A/B choices. Belief-Erasure Steering needs both the
original ("opinion") prompt and the same item with the biography removed ("neutral"), which
serves as the oracle target distribution.

Usage:
    python -m src.data --all
    python -m src.data --dataset political --limit 500 --review 25
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

FILES = {
    "nlp": "sycophancy_on_nlp_survey.jsonl",
    "phil": "sycophancy_on_philpapers2020.jsonl",
    "political": "sycophancy_on_political_typology_quiz.jsonl",
}

# Question stems. The biography is everything before the first match.
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


@dataclass
class Pair:
    uid: str
    dataset: str
    bio: str            # the user's self-description + stated opinion (erasure target span)
    question: str       # question + choices + "Answer:"
    opinion_prompt: str  # bio + question  (the original benchmark item)
    neutral_prompt: str  # question only   (oracle condition)
    matching: str       # answer letter that agrees with the user
    not_matching: str   # the other letter
    split_ok: bool      # False => stem heuristic fell back; inspect before using


def _split(question: str, dataset: str) -> tuple[str, str, bool]:
    """Return (bio, question_block, split_ok)."""
    for stem in STEMS[dataset]:
        m = re.search(stem, question)
        if m:
            return question[: m.start()].strip(), question[m.start():].strip(), True
    # Fallback: treat the last sentence before the choices as the question.
    head = re.split(r"\n\s*\(A\)|\n\nChoices:", question)[0]
    tail = question[len(head):]
    sents = re.split(r"(?<=[.!?]) ", head.strip())
    return " ".join(sents[:-1]).strip(), (sents[-1] + tail).strip(), False


def _letter(x) -> str:
    if isinstance(x, list):
        x = x[0]
    return x.strip()


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
            pairs.append(
                Pair(
                    uid=f"{dataset}-{i}",
                    dataset=dataset,
                    bio=bio,
                    question=q,
                    opinion_prompt=row["question"],
                    neutral_prompt=q,
                    matching=_letter(row["answer_matching_behavior"]),
                    not_matching=_letter(row["answer_not_matching_behavior"]),
                    split_ok=ok,
                )
            )
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
        print(f"{name}: {len(pairs)} pairs -> {out}  (stem match {1 - len(bad)/len(pairs):.1%})")
        for p in bad[: args.review]:
            print("  FALLBACK bio:", p.bio[-120:])
            print("  FALLBACK  q :", p.question[:120], "\n")


if __name__ == "__main__":
    main()
