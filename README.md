# MedAlign-RL: PPO vs GRPO for a small medical LLM

Start from a small medical SFT model, improve it with **PPO** (reward model) and **GRPO** (rule rewards), and compare both on a hand-checked golden dataset.

> Educational demo only. Not medical advice.

## Status
- [x] Phase 0: setup
- [ ] Phase 1: baseline + SFT
- [x] Phase 2: golden dataset
- [x] Phase 3: PPO (code ready, training runs on Colab)
- [x] Phase 4: GRPO (code ready, training runs on Colab)
- [x] Phase 5: comparison + analysis (notebook ready, fills in after Colab runs)
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

## Contributors
- [ubaidur404786](https://github.com/ubaidur404786) — author
- [Claude Code](https://claude.com/claude-code) — coding assistant (co-author on commits from Phase 3 onward)

See the [Contributors](https://github.com/ubaidur404786/medalign-rl/graphs/contributors) page.
