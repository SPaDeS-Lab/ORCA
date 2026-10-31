import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


def extract_answer(raw_text: str) -> str:
    """Algorithm 1, Line 3: ExtractAnswer(pi) - drop reasoning and markup."""
    if not raw_text:
        return ""
    text = str(raw_text).strip()

    # 1. Look for explicit Final Answer markers:
    # "Final Answer: ...", "Answer: ...", "\boxed{...}", etc.
    patterns = [
        r"(?:final\s+answer|answer)\s*[:=]\s*([^\n\r]+)",
        r"(?:final\s+answer|answer)\s*[:=]?\s*\n\s*([^\n\r]+)",
        r"\\boxed\{([^}]+)\}",
        r"(?:the\s+(?:correct\s+)?answer\s+is|result\s+is)\s*[:=]?\s*([^\n\r]+)",
    ]
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            ans = matches[-1].strip()
            # Clean markup (markdown bold, italics, backticks, quotes)
            ans = re.sub(r"^\*+|\*+$", "", ans).strip()
            ans = re.sub(r"^`+|`+$", "", ans).strip()
            ans = re.sub(r"^\"+|\"+$", "", ans).strip()
            ans = re.sub(r"^\'+|\'+$", "", ans).strip()
            if ans:
                return ans

    # 2. Fallback to the last non-empty line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last = lines[-1]
        last = re.sub(r"^(?:final\s+answer|answer)\s*[:=]\s*", "", last, flags=re.IGNORECASE).strip()
        last = re.sub(r"^\*+|\*+$", "", last).strip()
        last = re.sub(r"^`+|`+$", "", last).strip()
        last = re.sub(r"^\"+|\"+$", "", last).strip()
        last = re.sub(r"^\'+|\'+$", "", last).strip()
        return last

    return text


def canonical(ans: str, answer_type: str) -> str:
    """Algorithm 1, Line 4: Canonicalize(si, tau) - map answer to canonical key."""
    s = str(ans).strip()
    if answer_type == "set":
        # Sort comma-separated elements so {A, B} == {B, A}
        return ",".join(sorted({x.strip().lower() for x in s.split(",") if x.strip()}))
    if answer_type == "number":
        try:
            # Handle standard numbers and fractions/percentages
            return format(float(s), ".6f").rstrip("0").rstrip(".") or "0"
        except ValueError:
            # Try extracting first number
            m = re.search(r"[-+]?\d*\.?\d+", s)
            if m:
                try:
                    return format(float(m.group(0)), ".6f").rstrip("0").rstrip(".") or "0"
                except ValueError:
                    pass
            return s.lower()
    return s.lower()


@dataclass(frozen=True)
class Task:
    id: int
    tier: str            # cheap | medium | expensive | unverifiable
    kind: str
    prompt: str
    answer: str          # canonical ground truth
    wrong: tuple         # canonical, de-duplicated plausible wrong answers
    answer_type: str
    verify_cost: float   # math.inf => cannot be audited
    difficulty: float

    @property
    def verifiable(self) -> bool:
        return math.isfinite(self.verify_cost)

    @property
    def candidates(self) -> tuple:
        return (self.answer,) + self.wrong


def load_tasks(path: str | Path) -> List[Task]:
    tasks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            at = r["answer_type"]
            truth = canonical(r["answer"], at)
            wrong, seen = [], {truth}
            for d in r["distractors"]:
                c = canonical(d, at)
                if c not in seen:
                    seen.add(c)
                    wrong.append(c)
            if not wrong:
                raise ValueError(f"task {r['id']} has no usable distractors")
            tasks.append(Task(
                id=int(r["id"]), tier=r["tier"], kind=r["kind"], prompt=r["prompt"],
                answer=truth, wrong=tuple(wrong), answer_type=at,
                verify_cost=float(r["verify_cost"]), difficulty=float(r["difficulty"]),
            ))
    return tasks


def full_audit_cost(tasks: List[Task]) -> float:
    """Cost of auditing exactly one candidate on every verifiable task."""
    return float(sum(t.verify_cost for t in tasks if t.verifiable))
