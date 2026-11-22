"""Phase 6 helpers: model cards, pushing to the Hugging Face Hub, and the results GIF."""

import json

from medrl.config import BASE_MODEL, DISCLAIMER, HF_REPOS, RESULTS_DIR

METHODS = {
    "sft": "Supervised fine-tuning (LoRA) on medical Q&A.",
    "ppo": "PPO on top of SFT, with a learned reward model trained on preference pairs.",
    "grpo": "GRPO on top of SFT, with rule-based rewards (no reward model).",
}


def load_all_scores() -> dict:
    """{model_name: {type: acc, 'overall': acc}} from the saved notebook results."""
    table = {}
    for f, name in [("02_golden_scores.json", None), ("03_ppo_scores.json", "ppo"),
                    ("04_grpo_scores.json", "grpo")]:
        p = RESULTS_DIR / f
        if not p.exists():
            print("missing", f)
            continue
        d = json.load(open(p))
        table.update({k: v["scores"] for k, v in d.items()} if name is None else {name: d["scores"]})
    return table


def model_card(name: str, scores: dict) -> str:
    """Markdown README for one adapter repo, with the golden-set table."""
    rows = "\n".join(f"| {m} | " + " | ".join(f"{s.get(k, 0):.3f}" for k in ("mcq", "red_flag", "unsafe", "overall")) + " |"
                     for m, s in scores.items())
    return f"""---
base_model: {BASE_MODEL}
library_name: peft
license: apache-2.0
tags: [medical, rlhf, {name}, lora, trl]
---

# medalign-{name}

> **{DISCLAIMER}**

LoRA adapter for `{BASE_MODEL}`. {METHODS[name]}

## Golden-set scores (200 items: 150 MCQ, 30 red-flag, 20 unsafe)

| model | mcq | red_flag | unsafe | overall |
|---|---|---|---|---|
{rows}

## Use

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("{BASE_MODEL}")
model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained("{BASE_MODEL}"), "{HF_REPOS[name]}")
```

Code: https://github.com/ubaidur404786/medalign-rl
"""


def push_adapter(name: str, local_dir, scores: dict, private: bool = False) -> str:
    """Upload a saved adapter folder plus its model card. Needs `huggingface-cli login` first."""
    from huggingface_hub import HfApi
    api, repo = HfApi(), HF_REPOS[name]
    api.create_repo(repo, private=private, exist_ok=True)
    (local_dir / "README.md").write_text(model_card(name, scores), encoding="utf-8")
    api.upload_folder(repo_id=repo, folder_path=str(local_dir), commit_message=f"Upload medalign-{name}")
    return f"https://huggingface.co/{repo}"


def push_golden(path, private: bool = False) -> str:
    """Upload the golden evaluation set as a dataset repo."""
    from datasets import load_dataset
    repo = HF_REPOS["golden"]
    load_dataset("json", data_files=str(path), split="train").push_to_hub(repo, private=private)
    return f"https://huggingface.co/datasets/{repo}"


def make_gif(scores: dict, out_path, frames: int = 12) -> None:
    """Animated bar chart: bars grow from 0 to each model's score, per category."""
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cats = ["mcq", "red_flag", "unsafe", "overall"]
    models = list(scores)
    x, w = np.arange(len(cats)), 0.8 / max(len(models), 1)
    images = []
    for i in range(1, frames + 1):
        t = i / frames
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=90)
        for j, m in enumerate(models):
            vals = [scores[m].get(c, 0) * t for c in cats]
            ax.bar(x + (j - (len(models) - 1) / 2) * w, vals, w, label=m)
        ax.set_xticks(x, cats); ax.set_ylim(0, 1); ax.set_ylabel("accuracy")
        ax.set_title("medalign-rl: base vs SFT vs PPO vs GRPO (golden set)")
        ax.legend(loc="upper left", ncol=len(models), fontsize=8)
        fig.text(0.99, 0.01, DISCLAIMER, ha="right", fontsize=7, color="gray")
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    images += [images[-1]] * 8  # hold the final frame
    imageio.mimsave(out_path, images, duration=0.12, loop=0)
