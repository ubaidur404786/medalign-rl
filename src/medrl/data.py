"""Dataset helpers: load medical QA data and turn it into chat messages."""

from datasets import load_dataset

from medrl.config import SEED

SYSTEM_PROMPT = (
    "You are a careful medical assistant. Answer briefly and clearly. "
    "For emergencies, tell the user to call emergency services."
)


def to_messages(question: str, answer: str | None = None) -> list[dict]:
    """Build a chat conversation. If `answer` is given, it is the target reply."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def load_sft_dataset(n_train: int = 2000, n_test: int = 100):
    """Medical flashcards -> {"messages": [...]} rows, ready for TRL's SFTTrainer."""
    ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    ds = ds.filter(lambda r: r["input"] and r["output"] and len(r["output"]) < 600)
    ds = ds.shuffle(seed=SEED).select(range(n_train + n_test))
    ds = ds.map(
        lambda r: {"messages": to_messages(r["input"], r["output"])},
        remove_columns=ds.column_names,
    )
    split = ds.train_test_split(test_size=n_test, seed=SEED)
    return split["train"], split["test"]
