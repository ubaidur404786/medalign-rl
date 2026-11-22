"""Load a model (optionally with a LoRA adapter) and generate answers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from medrl.data import to_messages


def load_model(model_name: str, adapter: str | None = None):
    """Return (model, tokenizer). `adapter` is a local path or HF repo of LoRA weights."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # bf16 only on GPUs that support it (T4 does not); otherwise full float32
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="auto")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 150) -> str:
    """Greedy answer to one question (greedy = repeatable before/after comparisons)."""
    inputs = tokenizer.apply_chat_template(
        to_messages(question), add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()


def compare(models: dict, questions: list[str]) -> list[dict]:
    """Ask every model every question. `models` maps a label to (model, tokenizer)."""
    rows = []
    for q in questions:
        row = {"question": q}
        for label, (m, t) in models.items():
            row[label] = generate_answer(m, t, q)
        rows.append(row)
    return rows
