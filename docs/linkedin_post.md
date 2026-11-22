# LinkedIn post (draft)

Attach: `results/06_results.gif`. Replace every `[..]` with the real numbers from notebook 05.

---

I compared three ways to align a small medical LLM, end to end, on a free Colab GPU. 🩺

**Model:** Qwen2.5-0.5B-Instruct, LoRA adapters only.

**What I trained:**
1️⃣ **SFT**: fine-tuned on medical Q&A.
2️⃣ **PPO**: a reward model learned from preference pairs, then PPO on top of SFT.
3️⃣ **GRPO**: the DeepSeek-R1 method, with simple rule-based rewards and no reward model.

**How I measured it:** a 200-item golden set with 150 multiple-choice questions, 30 emergency "red-flag" cases (does it say *call emergency services*?) and 20 harmful requests (does it refuse?).

**Results (overall accuracy):**
• Base: [..]
• SFT: [..]
• PPO: [..]
• GRPO: [..]

**What I learned:**
• [biggest gain, e.g. "GRPO raised red-flag accuracy from [..] to [..]"]
• With only 30 and 20 safety items, a gap smaller than about ±[..] points can be chance.
• Rule rewards are cheap and hard to game on MCQ, but a model can learn to "say the keyword" on safety prompts. Read the answers, not just the scores.

Models: huggingface.co/ubaidii123
Code and notebooks: github.com/ubaidur404786/medalign-rl

⚠️ Educational demo only. Not medical advice.

#LLM #RLHF #GRPO #PPO #HuggingFace #MachineLearning #HealthAI
