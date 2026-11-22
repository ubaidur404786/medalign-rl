# MedAlign-RL: PPO vs GRPO for a small medical LLM

Start from a small medical SFT model, improve it with **PPO** (reward model) and **GRPO** (rule rewards), and compare both on a hand-checked golden dataset.

> Educational demo only. Not medical advice.

## Status
- [x] Phase 0: setup
- [ ] Phase 1: baseline + SFT
- [x] Phase 2: golden dataset
- [ ] Phase 3: PPO
- [ ] Phase 4: GRPO
- [ ] Phase 5: comparison + analysis
- [ ] Phase 6: Hugging Face + GitHub + post
- [ ] Phase 7: FastAPI + React app

The full plan is in [PLAN.md](PLAN.md).

## Quick start
```bash
uv venv
uv pip install -r requirements.txt
uv run python scripts/smoke_test.py
```

## Layout
| Folder | Contents |
|---|---|
| `notebooks/` | one notebook per step (explanations in markdown) |
| `src/medrl/` | reusable functions the notebooks import |
| `data/` | golden eval set, preference pairs |
| `results/` | metrics, before/after examples, charts |
| `reference/` | original PPO / GRPO course notebooks |
| `backend/`, `frontend/` | FastAPI + React demo (Phase 7) |
