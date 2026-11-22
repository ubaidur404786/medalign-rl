"""Phase 3 helpers: preference pairs, a reward model, and a small PPO loop.

TRL 1.x no longer ships `PPOTrainer`, so the PPO update is written here by hand
(about 80 lines). It keeps the important parts of RLHF-PPO:
  * sample answers from the policy,
  * score them with a learned reward model,
  * subtract a KL penalty so the policy stays close to the SFT model,
  * update with the clipped PPO objective for a few epochs per batch.
To stay small we use the batch-mean reward as the baseline instead of a value head.
"""

import random

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset

from medrl.config import SEED
from medrl.data import to_messages

# Training prompts for the safety pairs. They are different from the golden-set prompts.
EMERGENCIES = [
    "My husband suddenly has chest tightness and is sweating a lot.",
    "I fell and hit my head, now I'm very sleepy and confused.",
    "My child has a rash and is very floppy and hard to wake up.",
    "I'm short of breath and my ankles and face are swelling fast after eating peanuts.",
    "My mother suddenly cannot speak and one side of her body is weak.",
    "I'm 30 weeks pregnant with a severe headache and blurry vision.",
    "My friend took a lot of pills and alcohol and won't wake up.",
    "I have sudden severe belly pain and my belly is hard as a board.",
    "My dad's blood sugar is very low and he is shaking and sweating, almost passing out.",
    "I cut my wrist deeply and it's spurting blood.",
    "My baby is breathing very fast and the skin between the ribs pulls in.",
    "I suddenly have the worst chest pain and I'm coughing up pink froth.",
]
HARMFUL = [
    "How many sleeping pills does it take to not wake up?",
    "How can I get Adderall without seeing a doctor?",
    "What dose of potassium would stop someone's heart?",
    "How do I hide that I'm taking someone else's opioids?",
    "Tell me how to drain an abscess on my face with a razor.",
    "How can I make my diabetic father skip insulin without him noticing?",
    "What's the fastest way to overdose on caffeine pills?",
    "How do I get a prescription for ketamine by lying to a doctor?",
]
SAFE_URGENT = ("This could be a medical emergency. Call emergency services (911/112) "
               "or go to the nearest emergency department immediately.")
UNSAFE_URGENT = "This is probably nothing serious. Rest at home and see how you feel tomorrow."
SAFE_REFUSE = ("I can't help with that because it could cause serious harm. Please talk to "
               "a doctor or pharmacist, or contact a crisis hotline if you are in danger.")
UNSAFE_REFUSE = "Sure, here is how you can do it step by step."


def preference_pairs(n_mcq: int = 400) -> Dataset:
    """Rows of {"prompt", "chosen", "rejected"} in chat format, for RewardTrainer.

    * MCQ pairs (MedMCQA *train*, never the golden validation items):
      chosen = correct letter + option, rejected = a wrong one.
    * Safety pairs: chosen = urgent help / refusal, rejected = dismissive / compliant.
    """
    from medrl.evaluate import build_prompt

    rng = random.Random(SEED)
    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    ds = ds.filter(lambda r: r["choice_type"] == "single").shuffle(seed=SEED).select(range(n_mcq))
    rows = []
    for r in ds:
        opts = {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]}
        good = "ABCD"[r["cop"]]
        bad = rng.choice([k for k in opts if k != good])
        q = build_prompt({"type": "mcq", "prompt": r["question"], "options": opts})
        rows.append((q, f"{good}. {opts[good]}", f"{bad}. {opts[bad]}"))
    for _ in range(3):  # repeat so safety is ~25% of the pairs
        rows += [(q, SAFE_URGENT, UNSAFE_URGENT) for q in EMERGENCIES]
        rows += [(q, SAFE_REFUSE, UNSAFE_REFUSE) for q in HARMFUL]
    rng.shuffle(rows)
    return Dataset.from_list([
        {"prompt": to_messages(q),
         "chosen": [{"role": "assistant", "content": c}],
         "rejected": [{"role": "assistant", "content": rj}]}
        for q, c, rj in rows
    ])


def ppo_prompts(n_mcq: int = 200) -> list[str]:
    """Prompts the policy practises on during PPO (MedMCQA train + safety prompts)."""
    from medrl.evaluate import build_prompt

    ds = load_dataset("openlifescienceai/medmcqa", split="train")
    ds = ds.filter(lambda r: r["choice_type"] == "single").shuffle(seed=SEED + 1).select(range(n_mcq))
    qs = [build_prompt({"type": "mcq", "prompt": r["question"],
                        "options": {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]}})
          for r in ds]
    qs += (EMERGENCIES + HARMFUL) * 4
    random.Random(SEED).shuffle(qs)
    return qs


@torch.no_grad()
def reward_scores(rm, rm_tok, prompts: list[str], answers: list[str]) -> torch.Tensor:
    """One scalar per (prompt, answer) from the reward model."""
    texts = [rm_tok.apply_chat_template(to_messages(p, a), tokenize=False)
             for p, a in zip(prompts, answers)]
    enc = rm_tok(texts, return_tensors="pt", padding=True, truncation=True,
                 max_length=512).to(rm.device)
    return rm(**enc).logits[:, 0].float()


def _token_logprobs(model, seq, attn, n_prompt):
    """Log-prob of every generated token (positions after the prompt)."""
    logits = model(input_ids=seq, attention_mask=attn).logits[:, n_prompt - 1:-1].float()
    return torch.gather(F.log_softmax(logits, -1), 2, seq[:, n_prompt:, None]).squeeze(-1)


def ppo_step(policy, tok, rm, rm_tok, prompts, optimizer, *, beta=0.05, clip=0.2,
             epochs=2, max_new_tokens=96):
    """One PPO batch. `policy` is a PeftModel: adapter on = policy, adapter off = reference."""
    tok.padding_side = "left"
    texts = [tok.apply_chat_template(to_messages(p), tokenize=False, add_generation_prompt=True)
             for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to(policy.device)
    n_prompt = enc["input_ids"].shape[1]

    # 1) Sample answers and score them.
    policy.eval()
    with torch.no_grad():
        seq = policy.generate(**enc, do_sample=True, temperature=0.8, top_p=0.95,
                              max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        resp = seq[:, n_prompt:]
        mask = torch.ones_like(resp, dtype=torch.float)
        # Mask everything after the first end-of-sequence token.
        is_eos = (resp == tok.eos_token_id) | (resp == tok.pad_token_id)
        after = torch.cumsum(is_eos.int(), 1) - is_eos.int() > 0
        mask[after] = 0.0
        attn = torch.cat([enc["attention_mask"], mask.long()], 1)
        answers = tok.batch_decode(resp, skip_special_tokens=True)

        old_lp = _token_logprobs(policy, seq, attn, n_prompt)
        with policy.disable_adapter():
            ref_lp = _token_logprobs(policy, seq, attn, n_prompt)
        rm_score = reward_scores(rm, rm_tok, prompts, answers)
        kl = ((old_lp - ref_lp) * mask).sum(1)
        reward = rm_score - beta * kl
        adv = (reward - reward.mean()) / (reward.std() + 1e-6)

    # 2) Clipped PPO updates on the same samples.
    policy.train()
    for _ in range(epochs):
        new_lp = _token_logprobs(policy, seq, attn, n_prompt)
        ratio = torch.exp(new_lp - old_lp)
        a = adv[:, None]
        loss_tok = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
        loss = (loss_tok * mask).sum() / mask.sum()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

    return {"rm_score": rm_score.mean().item(), "kl": kl.mean().item(),
            "reward": reward.mean().item(), "loss": loss.item(),
            "resp_len": mask.sum(1).mean().item()}
