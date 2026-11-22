"""Phase 0 smoke test: can we load a model and generate one answer?

Run from the project root:
    uv run python scripts/smoke_test.py            # tiny model (fast, ~270 MB)
    uv run python scripts/smoke_test.py --base     # Qwen2.5-0.5B (~1 GB)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from medrl.config import BASE_MODEL, TINY_MODEL, SEED


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", action="store_true", help="use the main Qwen model")
    args = parser.parse_args()

    model_name = BASE_MODEL if args.base else TINY_MODEL
    torch.manual_seed(SEED)
    print(f"Loading {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    # A red-flag question: we want the model to advise urgent care.
    question = "I have sudden chest pain spreading to my left arm. What should I do?"
    messages = [{"role": "user", "content": question}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    prompt_len = inputs["input_ids"].shape[1]

    output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    answer = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    print(f"\nQ: {question}\nA: {answer}\n\nSmoke test OK.")


if __name__ == "__main__":
    main()
