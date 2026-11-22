"""medrl: reusable helpers for the MedAlign-RL project (SFT -> PPO vs GRPO on medical QA).

Notebooks stay short and explanatory; the logic lives here:
    config.py    - shared settings (model names, paths, seed)
    data.py      - load and format prompts            (Phase 1)
    generate.py  - one generate() used by every stage  (Phase 1)
    rewards.py   - GRPO reward functions               (Phase 4)
    evaluate.py  - golden-set metrics                  (Phase 5)
"""
