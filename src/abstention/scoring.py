"""Exact-match endpoints: these are not a complete factuality oracle."""
import math
import re
import string
import unicodedata
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Grade:
    outcome: str
    reason: str

    def asdict(self):
        return asdict(self)


def normalize_answer(text):
    # TriviaQA reference normalization: underscores are word separators.
    text = text.lower().replace("_", " ")
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def normalize_question(text):
    return " ".join(re.sub(r"[^\w\s]", " ", unicodedata.normalize("NFKC", text).casefold()).split())


def grade(text, aliases, truncated=False):
    if truncated:
        return Grade("error", "truncated")
    if not text.strip():
        return Grade("error", "empty")
    if text.strip() == "IDK":
        return Grade("abstain", "explicit_idk")
    normal = normalize_answer(text)
    if normal and normal in {normalize_answer(a) for a in aliases}:
        return Grade("correct", "exact_match")
    return Grade("error", "not_exact_match")


def reward(outcome, tau, arm="conditioned", fixed_tau=0.75):
    if outcome not in {"correct", "error", "abstain"}:
        raise ValueError(f"Unknown outcome: {outcome}")
    if not math.isfinite(tau) or not 0 < tau < 1:
        raise ValueError("tau must be finite and in (0,1)")
    if arm == "binary":
        return float(outcome == "correct")
    if arm == "fixed":
        tau = fixed_tau
    elif arm != "conditioned":
        raise ValueError(f"Unknown arm: {arm}")
    return {"correct": 1 - tau, "abstain": 0.0, "error": -tau}[outcome]


def centered_advantages(rewards):
    mean = sum(rewards) / len(rewards)
    return [r - mean for r in rewards]


def summarize(rows):
    n = len(rows)
    if not n:
        raise ValueError("Cannot summarize an empty evaluation.")
    counts = {key: sum(r["outcome"] == key for r in rows) for key in ("correct", "error", "abstain")}
    if sum(counts.values()) != n:
        raise ValueError("Unrecognized outcome in evaluation.")
    answered = counts["correct"] + counts["error"]
    return {"n": n, **counts, "accuracy": counts["correct"] / n,
            "error_rate": counts["error"] / n, "coverage": answered / n,
            "selective_risk": counts["error"] / answered if answered else None,
            "utility": sum(reward(r["outcome"], r["tau"]) for r in rows) / n,
            "invalid": sum(r.get("reason") in {"empty", "truncated"} for r in rows)}
