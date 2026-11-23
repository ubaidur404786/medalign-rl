# MedAlign-RL — Plan

**Goal:** Take a small medical SFT model, improve it with **PPO** and with **GRPO**, compare the two on a fixed golden dataset, and publish the models, the code and a short LinkedIn post.

**Authorship:** You write and run the project. Claude Code only helps with small things: grammar, reusable helper functions, debugging and reviews. Commits use only your git identity and never mention AI.

---

## 1. The medical problem we solve

A small medical model usually has three problems:

1. **Wrong answers** on clinical questions (hallucination).
2. **Unsafe tone.** It answers emergencies such as "chest pain + left arm numb" without saying "seek urgent care".
3. **Rambling.** Answers are long and have no clear final answer.

RLHF fixes behaviour rather than knowledge, so it is a good fit for problems 2 and 3, and for problem 1 when the answer can be checked.

**Task:** Medical multiple-choice QA with a short explanation. The answer can be checked automatically, which works well for GRPO. We add a safety rule for red-flag symptoms.

| Method | Reward signal | Why it fits |
|---|---|---|
| **PPO** | A trained **reward model** (preference pairs: safe and correct = chosen, unsafe or wrong = rejected) | Classic RLHF. Learns "what a good medical answer looks like" from preferences. |
| **GRPO** | **Rule-based rewards**: correct option, answer format, safety-phrase check for red flags, length | Needs no value model or reward model, so it is cheaper and more stable on small GPUs. This is the DeepSeek-R1 approach. |

### Links to your earlier work
- `fine-tune-KBAs-Semi-Structure`: same pattern of tiny HF models + LoRA. Reuse its project layout. Lesson learned there: SFT does not inject facts. RL here is about **behaviour** (correctness when the answer can be checked, safety, format).
- `deep_eval_pipeline` / `rag-eval-deepeval` / `llm-sql-eval`: reuse the evaluation ideas (golden set, metrics table, LLM-as-judge).
- `Ai-Safety-And-LLM-Security`: the safety angle. Treat red-flag handling as a guardrail metric.

---

## 2. Choices (keep it simple)

- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct` (the reference repo uses it too). Fallback: `HuggingFaceTB/SmolLM2-135M-Instruct` for quick tests.
- **Libraries:** `transformers`, `trl`, `peft` (LoRA), `datasets`, `accelerate`. Pin the versions in `requirements.txt`.
- **Hardware:** Train on **Google Colab (T4/L4)** or Kaggle. The laptop (7.6 GB RAM) is only for code, the app and evaluation of small outputs.
- **Data (public, no patient data):**
  - SFT: `medalpaca/medical_meadow_medical_flashcards` or MedMCQA explanations (2–5k samples).
  - RL prompts + golden set: `openlifescienceai/medmcqa` (or `GBaker/MedQA-USMLE-4-options`).
  - PPO preference pairs: build them yourself (section 4).
- **Disclaimer everywhere:** "Educational demo, not medical advice."

---

## 3. Repository layout

```
RLHF/
├── README.md                  # story, results table, GIF, links
├── PLAN.md                    # this file
├── requirements.txt
├── configs/                   # one small YAML per stage (sft, ppo, grpo)
├── data/
│   ├── golden_eval.jsonl      # FIXED evaluation set (never trained on)
│   └── preference_pairs.jsonl # for the PPO reward model
├── notebooks/                 # one notebook per step, markdown explains each cell
│   ├── 00_explore_data.ipynb
│   ├── 01_sft.ipynb
│   ├── 02_build_golden_and_preferences.ipynb
│   ├── 03_ppo.ipynb
│   ├── 04_grpo.ipynb
│   ├── 05_compare.ipynb
│   └── 06_push_to_hub.ipynb
├── src/medrl/                 # reusable functions the notebooks import
│   ├── data.py                # load + format prompts
│   ├── rewards.py             # GRPO reward functions
│   ├── generate.py            # one generate() used by every stage
│   └── evaluate.py            # metrics on the golden set
├── results/                   # metrics.json, before/after examples, charts
└── assets/compare.gif         # LinkedIn animation
```

Rule: notebooks = explanation + calls; `src/` = logic. Every function has a docstring, and every notebook starts with a "What / Why / Output" markdown cell.

Your two existing notebooks (`RLHF_with_PPO_Final_Code.ipynb`, `Final_GRPO_Practical.ipynb`) are the reference copies. Move them to `reference/` and build `03_ppo` and `04_grpo` from them.

---

## 4. Step-by-step phases

### Phase 0 — Setup (½ day)
- Create the venv and `requirements.txt`, log in with `huggingface-cli login`, create the GitHub repo.
- Done when: a 5-line test loads Qwen2.5-0.5B and generates one answer.

### Phase 1 — Baseline + SFT (1 day)
- Run the **base model** on 10 sample questions and save them as "before" examples.
- SFT with LoRA (r=16) on 2–5k medical samples, 1 epoch, in this prompt format:
  `Question … Options A–D … → "Answer: X\nExplanation: …"`.
- Save the adapter and push `medalign-sft` to HF.
- Done when: SFT answers in the right format more often than the base model.

### Phase 2 — Golden dataset (½–1 day) ⭐ most important for a fair comparison
- `golden_eval.jsonl`, about **200 items**, held out from all training:
  - 150 MedMCQA questions (correct option known),
  - 30 **red-flag** prompts (emergencies; the expected behaviour is to advise urgent care),
  - 20 **out-of-scope / unsafe** prompts (dosing for self-harm, "diagnose me"; expected: refuse or defer).
- Fields: `id, type, prompt, options, answer, must_include, must_not_include`.
- Review every red-flag and unsafe item by hand. This is your own work and the core value of the project.

### Phase 3 — PPO (1–2 days), based on your PPO notebook
1. Build **preference pairs** (~300–500): for each prompt, *chosen* = correct + safe + concise, *rejected* = wrong option, or missing the safety advice, or rambling. Sources: SFT outputs (pick good and bad) + hand edits.
2. Train the **reward model** (`RewardTrainer`, SmolLM2 or Qwen-0.5B, `num_labels=1`).
3. Run **PPO** (`PPOTrainer`) starting from the SFT model, with frozen reference + reward model and a trainable value model. Keep it small: `response_length` ~128, a few hundred episodes, and watch the KL.
4. Save and push `medalign-ppo`.

### Phase 4 — GRPO (1 day), based on your GRPO notebook
- Start from the SFT model, LoRA, `GRPOTrainer`, `num_generations=4`.
- Rewards in `src/medrl/rewards.py`:
  - `correct_option_reward`: +2 if the chosen letter matches the answer,
  - `format_reward`: +0.5 for `Answer: X` + an explanation,
  - `safety_reward`: +1 if a red-flag prompt mentions urgent care / emergency, −1 if it is missing,
  - `length_reward`: a small penalty when the answer is over ~150 words.
- Save and push `medalign-grpo`.
- Keep the compute budget (steps, prompts) about equal to PPO so the comparison is fair.

### Phase 5 — Compare (1 day): `05_compare.ipynb`
Run **Base, SFT, PPO, GRPO** on the same golden set with the same decoding settings (greedy, fixed seed).

| Metric | How |
|---|---|
| Accuracy (MCQ) | exact option match |
| Format compliance | regex `Answer: [A-D]` |
| Red-flag safety rate | `must_include` keywords + LLM-judge spot check |
| Refusal correctness | unsafe prompts handled correctly |
| Avg length | tokens |
| Training cost | GPU minutes, peak VRAM, stability (reward/KL curves) |

Write the **analysis** in markdown, answering *where* each method improved and *why*:
- GRPO is expected to win on **accuracy and format**, because it optimizes the exact checkable rule with no reward-model noise and no value model.
- PPO is expected to win on **tone/helpfulness** where no rule exists, but it is limited by a small reward model (reward hacking risk: longer ≠ better).
- Report honestly where neither helped (for example, facts the base model doesn't know).
- Include 5 before/after examples per model in `results/examples.md`.

### Phase 6 — Publish (½ day)
- HF: 3 model repos (sft / ppo / grpo) + a dataset repo for `golden_eval`, each with a model card (metrics, intended use, medical disclaimer).
- GitHub: clean README with the results table, charts, GIF and how to reproduce.
- **GIF** (`assets/compare.gif`): a few frames that show one red-flag question → Base / SFT / PPO / GRPO answers appearing in turn, then the metric bar chart. Made with matplotlib + `imageio`.
- **LinkedIn post** (short, about 120 words): the problem, the method, one result line, the GIF. Put the **HF and GitHub links in the first comment**.

---

## 5. How we work together
- You run training (Colab) and paste back the metrics/errors. Claude helps debug.
- Claude helps with: grammar in markdown/README/post, small reusable functions, code review, fixing errors.
- After each phase: commit with your own message, update the README checklist.

## 6. Checklist
- [ ] P0 setup  - [ ] P1 SFT  - [ ] P2 golden set  - [ ] P3 PPO  - [ ] P4 GRPO
- [ ] P5 compare + analysis  - [ ] P6 HF + GitHub + GIF + post
