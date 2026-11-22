"""Score models on the golden set. The same rule checks are reused as GRPO rewards."""

import json
import re

from medrl.config import DATA_DIR


def load_golden(path=DATA_DIR / "golden.jsonl") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_prompt(item: dict) -> str:
    """MCQ items get their options and an instruction to answer with one letter."""
    if item["type"] != "mcq":
        return item["prompt"]
    opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
    return f"{item['prompt']}\n{opts}\nAnswer with the letter of the correct option."


def extract_letter(text: str) -> str | None:
    """First standalone A-D in the answer, e.g. 'B', 'B.', '(B)', 'Answer: B'."""
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else None


def score(item: dict, answer: str) -> float:
    """1.0 = pass, 0.0 = fail."""
    low = answer.lower()
    if item["type"] == "mcq":
        return float(extract_letter(answer) == item["answer"])
    if any(w in low for w in item["must_not_include"]):
        return 0.0
    return float(any(w in low for w in item["must_include"]))


def evaluate(model, tokenizer, items: list[dict], max_new_tokens: int = 150) -> dict:
    """Return {"scores": {type: accuracy, "overall": ...}, "rows": [...]}."""
    from medrl.generate import generate_answer

    rows = []
    for it in items:
        ans = generate_answer(model, tokenizer, build_prompt(it), max_new_tokens)
        rows.append({"id": it["id"], "type": it["type"], "answer": ans, "score": score(it, ans)})
    scores = {}
    for t in sorted({r["type"] for r in rows}):
        vals = [r["score"] for r in rows if r["type"] == t]
        scores[t] = sum(vals) / len(vals)
    scores["overall"] = sum(r["score"] for r in rows) / len(rows)
    return {"scores": scores, "rows": rows}
