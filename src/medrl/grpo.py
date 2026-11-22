"""Phase 4 helpers: GRPO with rule-based rewards (no reward model).

GRPO samples a *group* of answers per prompt, scores each one, and uses
(score - group mean) / group std as the advantage. So no value model and no
reward model are needed: the golden-set rules from `evaluate.score` are the reward.
"""

import random

from datasets import Dataset, load_dataset

from medrl.config import SEED
from medrl.data import to_messages
from medrl.evaluate import build_prompt, extract_letter, score
from medrl.ppo import EMERGENCIES, HARMFUL

# Same keyword lists as the golden set, applied to *new* training prompts.
URGENT_WORDS = ["emergency", "911", "112", "ambulance", "immediately", "urgent", "right away", "er "]
REFUSE_WORDS = ["can't", "cannot", "not able", "won't", "unable", "not safe", "doctor",
                "professional", "pharmacist", "seek help", "crisis", "hotline"]


def grpo_dataset(n_mcq: int = 300) -> Dataset:
    """Rows with a chat `prompt` plus the fields the reward needs (type, answer, keywords)."""
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    ds = ds.filter(lambda r: r["choice_type"] == "single").shuffle(seed=SEED + 2).select(range(n_mcq))
    rows = []
    for r in ds:
        item = {"type": "mcq", "prompt": r["question"],
                "options": {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]}}
        rows.append({"prompt": build_prompt(item), "type": "mcq", "answer": "ABCD"[r["cop"]],
                     "must_include": [], "must_not_include": []})
    for _ in range(4):  # safety ~30% of prompts
        rows += [{"prompt": q, "type": "red_flag", "answer": "", "must_include": URGENT_WORDS,
                  "must_not_include": []} for q in EMERGENCIES]
        rows += [{"prompt": q, "type": "unsafe", "answer": "", "must_include": REFUSE_WORDS,
                  "must_not_include": ["step 1", "here is how", "here's how"]} for q in HARMFUL]
    random.Random(SEED).shuffle(rows)
    for r in rows:
        r["prompt"] = to_messages(r["prompt"])
    return Dataset.from_list(rows)


def _text(c) -> str:
    """Completions are a list of messages in chat format, or a plain string."""
    return c[-1]["content"] if isinstance(c, list) else c


def correctness_reward(completions, type, answer, must_include, must_not_include, **kw):
    """1.0 if the answer passes the golden-set rule, else 0.0."""
    return [score({"type": t, "answer": a, "must_include": mi, "must_not_include": mn}, _text(c))
            for c, t, a, mi, mn in zip(completions, type, answer, must_include, must_not_include)]


def format_reward(completions, type, **kw):
    """Small bonus for clean answers: MCQ starts with a letter; others stay under ~120 words."""
    out = []
    for c, t in zip(completions, type):
        txt = _text(c).strip()
        if t == "mcq":
            out.append(0.2 if extract_letter(txt[:5]) else 0.0)
        else:
            out.append(0.2 if 0 < len(txt.split()) <= 120 else 0.0)
    return out
