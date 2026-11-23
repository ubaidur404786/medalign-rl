"""Animated, light-theme explainer of the LLM alignment pipeline (for the LinkedIn post).

Pretraining -> SFT -> Human feedback -> {PPO, GRPO, DPO/ORPO} -> Aligned LLM.
Each stage lights up, data flows in along the arrows, and a detail panel
animates what that stage does. Output: assets/llm_alignment_flow.gif
"""
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "llm_alignment_flow.gif"
FD = "C:/Windows/Fonts/"
W, H = 1376, 925

BG, INK, MUTED, LINE, CARD = (247, 249, 252), (22, 30, 46), (100, 112, 135), (205, 212, 225), (255, 255, 255)
COL = {"pre": (100, 116, 139), "sft": (37, 99, 235), "hf": (147, 51, 234), "ppo": (234, 88, 12),
       "grpo": (22, 163, 74), "dpo": (13, 148, 136), "out": (2, 132, 199)}
BOX = {"pre": (40, 165, 245, 395), "sft": (295, 185, 480, 375), "hf": (530, 185, 715, 375),
       "ppo": (780, 92, 1000, 212), "grpo": (780, 222, 1000, 342), "dpo": (780, 352, 1000, 472),
       "out": (1075, 180, 1336, 380)}
ICON = {"pre": "🌐", "sft": "💬", "hf": "👍", "ppo": "🎯", "grpo": "👥", "dpo": "⚖️", "out": "🛡️"}
TITLE = {"pre": "PRETRAINING", "sft": "SFT", "hf": "HUMAN FEEDBACK", "ppo": "PPO", "grpo": "GRPO",
         "dpo": "DPO / ORPO", "out": "ALIGNED LLM"}
SUB = {"pre": "Raw internet text →\npredicts next word", "sft": "Question + ideal answer\n→ an assistant",
       "hf": "People pick the better\nanswer → reward model", "ppo": "Reward model · ChatGPT",
       "grpo": "Group compare · DeepSeek-R1", "dpo": "Pairs · no reward model",
       "out": "Helpful · Correct · Safe"}
TAG = {"pre": "STAGE 1", "sft": "STAGE 2", "hf": "STAGE 3", "ppo": "A", "grpo": "B", "dpo": "C", "out": "RESULT"}
# Arrows: (from, to, polyline)
ARROWS = {
    "sft": [(245, 280), (295, 280)], "hf": [(480, 280), (530, 280)],
    "ppo": [(715, 280), (745, 280), (745, 152), (780, 152)], "grpo": [(715, 280), (780, 282)],
    "dpo": [(715, 280), (745, 280), (745, 412), (780, 412)],
    "out_ppo": [(1000, 152), (1035, 152), (1035, 280), (1075, 280)],
    "out_grpo": [(1000, 282), (1075, 280)],
    "out_dpo": [(1000, 412), (1035, 412), (1035, 280), (1075, 280)],
}
CAPTION = {
    "pre": ("Stage 1 · Pretraining", "Trillions of words from the web, books and code.\nThe model only learns to predict the next word.\nIt knows a lot, but it is not an assistant yet."),
    "sft": ("Stage 2 · Supervised fine-tuning", "Thousands of (question, ideal answer) pairs.\nThe model learns to reply like a helpful assistant.\nBut it copies style; it does not know which answer is better."),
    "hf": ("Stage 3 · Human feedback", "People compare two answers and pick the better one.\nThese preferences train a reward model that scores\nany answer — a stand-in for human judgement."),
    "ppo": ("A · PPO (used for ChatGPT)", "The model writes an answer, the reward model scores it,\nand the model is nudged toward higher scores.\nA KL 'leash' keeps it close to the SFT model so it can't cheat."),
    "grpo": ("B · GRPO (DeepSeek-R1)", "Write about 8 answers to the same question.\nScore each one and compare it to the group average:\nabove average → reinforce, below → discourage. No value model."),
    "dpo": ("C · DPO / ORPO", "Skip the reward model and the RL loop.\nLearn straight from (chosen, rejected) pairs:\nraise the chosen answer, lower the rejected one."),
    "out": ("Result · Aligned LLM", "All three paths push the model toward answers\npeople prefer: helpful, correct and safe.\nSame question, very different answer."),
}
ORDER = ["pre", "sft", "hf", "ppo", "grpo", "dpo", "out"]


def F(name, size):
    return ImageFont.truetype(FD + name, size)


F_TITLE, F_BOX, F_SUB, F_TAG = F("segoeuib.ttf", 40), F("segoeuib.ttf", 22), F("segoeui.ttf", 16), F("seguisb.ttf", 13)
BADGE = {"pre": "1", "sft": "2", "hf": "3", "ppo": "A", "grpo": "B", "dpo": "C", "out": "R"}
F_BADGE = F("segoeuib.ttf", 20)
F_BOX_S, F_SUB_S = F("segoeuib.ttf", 18), F("segoeui.ttf", 14)
F_CAPH, F_CAP, F_SM, F_SMB = F("segoeuib.ttf", 26), F("segoeui.ttf", 19), F("segoeui.ttf", 16), F("seguisb.ttf", 16)
F_EMO, F_EMO_S = F("seguiemj.ttf", 34), F("seguiemj.ttf", 22)


def mix(c, t, base=BG):
    return tuple(int(b + (a - b) * t) for a, b in zip(c, base))


def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * max(0.0, min(1.0, t)))


def along(path, t):
    segs = list(zip(path, path[1:]))
    lens = [math.dist(a, b) for a, b in segs]
    d = max(0.0, min(1.0, t)) * sum(lens)
    for (a, b), L in zip(segs, lens):
        if d <= L:
            r = d / L if L else 0
            return a[0] + (b[0] - a[0]) * r, a[1] + (b[1] - a[1]) * r
        d -= L
    return path[-1]


def arrow(d, path, color, width=3):
    d.line(path, fill=color, width=width, joint="curve")
    (x0, y0), (x1, y1) = path[-2], path[-1]
    a = math.atan2(y1 - y0, x1 - x0)
    pts = [(x1, y1), (x1 - 11 * math.cos(a - 0.45), y1 - 11 * math.sin(a - 0.45)),
           (x1 - 11 * math.cos(a + 0.45), y1 - 11 * math.sin(a + 0.45))]
    d.polygon(pts, fill=color)


def draw_box(img, d, k, state, pulse=0.0):
    """state: 'future' (faded), 'done' (soft colour), 'active' (full colour + glow)."""
    x0, y0, x1, y1 = BOX[k]
    c = COL[k]
    if state == "active":
        for i in range(6, 0, -1):
            d.rounded_rectangle((x0 - i * 2 - pulse * 3, y0 - i * 2 - pulse * 3, x1 + i * 2 + pulse * 3, y1 + i * 2 + pulse * 3),
                                18, outline=mix(c, 0.06 * (7 - i)), width=2)
        fill, edge, tx, sub, w = mix(c, 0.10, CARD), c, INK, MUTED, 3
    elif state == "done":
        fill, edge, tx, sub, w = mix(c, 0.06, CARD), mix(c, 0.55), INK, MUTED, 2
    else:
        fill, edge, tx, sub, w = mix(CARD, 0.6), LINE, mix(INK, 0.35), mix(MUTED, 0.35), 2
    d.rounded_rectangle((x0, y0, x1, y1), 16, fill=fill, outline=edge, width=w)
    tagc = c if state != "future" else mix(c, 0.35)
    d.text((x0 + 14, y0 + 10), TAG[k], font=F_TAG, fill=tagc)
    cx = (x0 + x1) / 2
    small = y1 - y0 < 150
    if small:
        img_e = ICON[k]
        d.text((x0 + 16, y0 + 38), img_e, font=F_EMO, embedded_color=True)
        d.text((x0 + 66, y0 + 40), TITLE[k], font=F_BOX, fill=tx)
        d.text((x0 + 16, y0 + 88), SUB[k], font=F_SUB_S, fill=sub)
    else:
        d.text((cx, y0 + 40), ICON[k], font=F_EMO, embedded_color=True, anchor="mt")
        ft = F_BOX if d.textlength(TITLE[k], font=F_BOX) < (x1 - x0 - 20) else F_BOX_S
        d.text((cx, y0 + 98), TITLE[k], font=ft, fill=tx, anchor="mt")
        d.multiline_text((cx, y0 + 134), SUB[k], font=F_SUB, fill=sub, anchor="ma", align="center", spacing=4)
    if state == "future":  # fade emoji too
        ov = Image.new("RGBA", (x1 - x0, y1 - y0), mix(CARD, 0.6) + (150,))
        img.paste(ov, (x0, y0), ov)


def chip(d, xy, text, color, font=F_SM):
    x, y = xy
    w = d.textlength(text, font=font) + 20
    d.rounded_rectangle((x - w / 2, y - 14, x + w / 2, y + 14), 14, fill=CARD, outline=color, width=2)
    d.text((x, y), text, font=font, fill=color, anchor="mm")


# ---------- detail-panel animations (t goes 0..1) ----------
PX0, PY0, PX1, PY1 = 640, 585, 1336, 885
# Layout knobs (overridden by make_flow_gif_vertical.py)
ENTRY = [(5, 280), (40, 280)]
PANEL_TOP, CAP_XY, BODY_XY, DIVIDER, OUT_W = 505, (75, 530), (70, 595), True, 1100
IDLE_TEXT = "From raw text to a safe assistant — in three stages and three alignment methods."


def panel_pre(d, t):
    srcs = ["web pages", "books", "code", "forums"]
    for i, s in enumerate(srcs):
        p = ease(t * 1.6 - i * 0.15)
        x = PX0 + 70 + p * 220
        chip(d, (x, PY0 + 60 + i * 50), s, COL["pre"])
    d.text((PX0 + 360, PY0 + 115), "🧠", font=F_EMO, embedded_color=True)
    words = "The patient has chest pain spreading to the left".split()
    n = int(min(1, max(0, t * 1.4 - 0.25)) * len(words))
    d.text((PX0 + 30, PY0 + 255), " ".join(words[:max(n, 1)]), font=F_SM, fill=INK)
    if t > 0.85:
        x = PX0 + 30 + d.textlength(" ".join(words) + " ", font=F_SM)
        chip(d, (x + 25, PY0 + 266), "arm", COL["pre"], F_SMB)
        d.text((PX0 + 420, PY0 + 116), "next word?", font=F_SMB, fill=COL["pre"])


def panel_sft(d, t):
    c = COL["sft"]
    p = ease(t * 1.5)
    d.rounded_rectangle((PX0 + 30, PY0 + 30, PX0 + 330, PY0 + 80), 12, fill=mix(c, 0.08, CARD), outline=c, width=2)
    d.text((PX0 + 45, PY0 + 55), "User: I have chest pain, what do I do?", font=F_SM, fill=INK, anchor="lm")
    if p > 0.3:
        y = PY0 + 100 + (1 - ease((p - 0.3) / 0.7)) * 30
        d.rounded_rectangle((PX0 + 120, y, PX0 + 660, y + 50), 12, fill=CARD, outline=c, width=2)
        d.text((PX0 + 135, y + 25), "Assistant: It could be many things. Rest and see a doctor.", font=F_SM, fill=INK, anchor="lm")
    if t > 0.7:
        d.text((PX0 + 30, PY0 + 205), "+ sounds like an assistant", font=F_SMB, fill=c)
        d.text((PX0 + 30, PY0 + 235), "− misses that this could be a heart attack", font=F_SMB, fill=(200, 60, 60))


def panel_hf(d, t):
    c = COL["hf"]
    for i, (txt, good) in enumerate([("A: Call emergency services now.", True), ("B: Rest and wait a few days.", False)]):
        x = PX0 + 30 + i * 330
        pick = t > 0.45
        col = (22, 163, 74) if (pick and good) else ((200, 60, 60) if pick else LINE)
        d.rounded_rectangle((x, PY0 + 30, x + 310, PY0 + 90), 12, fill=CARD, outline=col, width=3 if pick else 2)
        d.text((x + 14, PY0 + 60), txt, font=F_SM, fill=INK, anchor="lm")
        if pick:
            d.text((x + 270, PY0 + 40), "👍" if good else "👎", font=F_EMO_S, embedded_color=True)
    if t > 0.6:
        p = ease((t - 0.6) / 0.3)
        arrow(d, [(PX0 + 185, PY0 + 95), (PX0 + 185, PY0 + 95 + 80 * p)], c)
        if p > 0.95:
            d.rounded_rectangle((PX0 + 60, PY0 + 185, PX0 + 470, PY0 + 235), 25, fill=mix(c, 0.1, CARD), outline=c, width=2)
            d.text((PX0 + 265, PY0 + 210), "Reward model: score(A) = 0.9 · score(B) = 0.2", font=F_SMB, fill=c, anchor="mm")


def gauge(d, cx, cy, r, val, c):
    d.arc((cx - r, cy - r, cx + r, cy + r), 180, 360, fill=LINE, width=14)
    d.arc((cx - r, cy - r, cx + r, cy + r), 180, 180 + 180 * val, fill=c, width=14)
    a = math.pi * (1 - val)
    d.line([(cx, cy), (cx + (r - 18) * math.cos(a), cy - (r - 18) * math.sin(a))], fill=INK, width=4)
    d.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=INK)


def panel_ppo(d, t):
    c = COL["ppo"]
    rounds = [("Rest and wait.", 0.2), ("See a doctor soon.", 0.5), ("Call emergency services now.", 0.9)]
    k = min(2, int(t * 3))
    lt = (t * 3) - k
    txt, sc = rounds[k]
    prev = rounds[k - 1][1] if k else 0.0
    val = prev + (sc - prev) * ease(lt * 2)
    d.text((PX0 + 30, PY0 + 30), f"round {k + 1}: model answers", font=F_SMB, fill=MUTED)
    d.rounded_rectangle((PX0 + 30, PY0 + 60, PX0 + 340, PY0 + 110), 12, fill=CARD, outline=c, width=2)
    d.text((PX0 + 45, PY0 + 85), txt, font=F_SM, fill=INK, anchor="lm")
    gauge(d, PX0 + 500, PY0 + 150, 95, val, c)
    d.text((PX0 + 500, PY0 + 170), f"reward {val:.1f}", font=F_SMB, fill=c, anchor="mt")
    d.text((PX0 + 30, PY0 + 140), "→ update the model\n   toward higher reward", font=F_SM, fill=MUTED, spacing=4)
    d.text((PX0 + 30, PY0 + 220), "KL leash: stay close to the SFT model", font=F_SMB, fill=INK)


def panel_grpo(d, t):
    c = COL["grpo"]
    scores = [0.2, 0.9, 0.4, 0.8, 0.1, 0.6, 0.3, 0.7]
    avg = sum(scores) / len(scores)
    base_y, top = PY0 + 250, 190
    d.text((PX0 + 30, PY0 + 20), "8 answers to the same question, each scored", font=F_SMB, fill=MUTED)
    for i, s in enumerate(scores):
        p = ease(t * 2 - i * 0.08)
        x = PX0 + 60 + i * 70
        h = top * s * p
        col = c if (t > 0.65 and s > avg) else ((200, 60, 60) if t > 0.65 else mix(c, 0.45, CARD))
        d.rounded_rectangle((x, base_y - h, x + 44, base_y), 6, fill=col)
        d.text((x + 22, base_y + 8), f"#{i + 1}", font=F_SM, fill=MUTED, anchor="mt")
    if t > 0.5:
        y = base_y - top * avg
        d.line([(PX0 + 45, y), (PX0 + 620, y)], fill=INK, width=2)
        d.text((PX0 + 625, y), "group\naverage", font=F_SM, fill=INK, anchor="lm")
    if t > 0.7:
        d.text((PX0 + 340, PY0 + 20), "", font=F_SM)
        d.text((PX0 + 30, PY0 + 290), "green: reinforce   red: discourage", font=F_SMB, fill=INK)


def panel_dpo(d, t):
    c = COL["dpo"]
    p = ease(t * 1.3)
    up, down = 30 * p, 30 * p
    d.rounded_rectangle((PX0 + 40, PY0 + 110 - up, PX0 + 330, PY0 + 180 - up), 12, fill=CARD, outline=(22, 163, 74), width=3)
    d.text((PX0 + 55, PY0 + 130 - up), "chosen", font=F_SMB, fill=(22, 163, 74))
    d.text((PX0 + 55, PY0 + 155 - up), "Call emergency services now.", font=F_SM, fill=INK)
    fade = mix(INK, 1 - 0.6 * p, CARD)
    d.rounded_rectangle((PX0 + 370, PY0 + 110 + down, PX0 + 660, PY0 + 180 + down), 12, fill=CARD, outline=mix((200, 60, 60), 1 - 0.5 * p, CARD), width=3)
    d.text((PX0 + 385, PY0 + 130 + down), "rejected", font=F_SMB, fill=mix((200, 60, 60), 1 - 0.5 * p, CARD))
    d.text((PX0 + 385, PY0 + 155 + down), "Rest and wait a few days.", font=F_SM, fill=fade)
    if p > 0.5:
        arrow(d, [(PX0 + 185, PY0 + 70), (PX0 + 185, PY0 + 40)], (22, 163, 74))
        arrow(d, [(PX0 + 515, PY0 + 225), (PX0 + 515, PY0 + 255)], (200, 60, 60))
    d.text((PX0 + 40, PY0 + 270), "one simple loss · no reward model · no RL loop", font=F_SMB, fill=c)


def panel_out(d, t):
    c = COL["out"]
    rows = [("Pretrained", "…and also my back, forums say it could be…", COL["pre"]),
            ("SFT", "It could be many things. Rest and see a doctor.", COL["sft"]),
            ("Aligned", "This could be a heart attack. Call emergency services now.", c)]
    d.text((PX0 + 30, PY0 + 20), "Q: Chest pain spreading to my left arm?", font=F_SMB, fill=INK)
    for i, (who, ans, col) in enumerate(rows):
        p = ease(t * 2 - i * 0.3)
        if p <= 0:
            continue
        y = PY0 + 60 + i * 75 + (1 - p) * 20
        bold = i == 2
        d.rounded_rectangle((PX0 + 30, y, PX0 + 670, y + 58), 12, fill=mix(col, 0.08 if bold else 0.03, CARD), outline=col, width=3 if bold else 1)
        d.text((PX0 + 45, y + 8), who, font=F_TAG, fill=col)
        d.text((PX0 + 45, y + 27), ans, font=F_SMB if bold else F_SM, fill=INK)


PANEL = {"pre": panel_pre, "sft": panel_sft, "hf": panel_hf, "ppo": panel_ppo, "grpo": panel_grpo,
         "dpo": panel_dpo, "out": panel_out}


def render(done, active, t=1.0, flow=None, pulse=0.0):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((W / 2, 28), "How an LLM learns to be helpful and safe", font=F_TITLE, fill=INK, anchor="mt")
    lit = set(done) | ({active} if active else set())
    for name, path in ARROWS.items():
        tgt = name.split("_")[-1] if name.startswith("out") else name
        on = (tgt in lit) if not name.startswith("out") else ("out" in lit and tgt in lit)
        arrow(d, path, mix(COL[tgt if not name.startswith("out") else "out"], 0.8) if on else LINE)
    arrow(d, ENTRY, mix(COL["pre"], 0.8) if "pre" in lit else LINE)
    for k in ORDER:
        draw_box(img, d, k, "active" if k == active else ("done" if k in done else "future"), pulse)
    for path, col, tt in flow or []:
        for j in range(4):
            x, y = along(path, tt - j * 0.12)
            r = 7 - j
            d.ellipse((x - r, y - r, x + r, y + r), fill=col)
    # detail panel
    d.rounded_rectangle((40, PANEL_TOP, W - 40, H - 20), 18, fill=CARD, outline=LINE, width=2)
    if active:
        c = COL[active]
        # Tinted header band with a step badge and a progress stepper.
        hy = PANEL_TOP + 34
        d.rounded_rectangle((41, PANEL_TOP + 1, W - 41, PANEL_TOP + 68), 17, fill=mix(c, 0.10, CARD),
                            corners=(True, True, False, False))
        d.line([(41, PANEL_TOP + 68), (W - 41, PANEL_TOP + 68)], fill=mix(c, 0.35, CARD), width=2)
        d.ellipse((60, hy - 19, 98, hy + 19), fill=c)
        d.text((79, hy), BADGE[active], font=F_BADGE, fill=CARD, anchor="mm")
        head, body = CAPTION[active]
        d.text((112, hy), head, font=F_CAPH, fill=mix(c, 0.85, INK), anchor="lm")
        x = W - 64
        for k in reversed(ORDER):
            if k == active:
                d.rounded_rectangle((x - 30, hy - 6, x, hy + 6), 6, fill=c); x -= 44
            else:
                d.ellipse((x - 12, hy - 6, x, hy + 6), fill=COL[k] if k in done else LINE); x -= 24
        d.multiline_text(BODY_XY, body, font=F_CAP, fill=INK, spacing=10)
        if DIVIDER:
            d.line([(PX0 - 20, PANEL_TOP + 90), (PX0 - 20, H - 45)], fill=LINE, width=2)
        PANEL[active](d, t)
    else:
        d.text((W / 2, (PANEL_TOP + H) / 2), IDLE_TEXT,
               font=F_CAPH, fill=MUTED, anchor="mm")
    return img


def flows_into(k):
    if k == "pre":
        return [ENTRY]
    if k == "out":
        return [ARROWS["out_ppo"], ARROWS["out_grpo"], ARROWS["out_dpo"]]
    return [ARROWS[k]]


def main():
    frames, durs, done = [], [], []
    frames.append(render([], None)); durs.append(2000)
    for k in ORDER:
        c = COL[k]
        for i in range(8):                       # data flowing in
            tt = i / 7 * 1.3
            frames.append(render(done, k, t=0.0, flow=[(p, c, tt) for p in flows_into(k)])); durs.append(60)
        n = 16                                   # stage animation
        for i in range(n + 1):
            frames.append(render(done, k, t=i / n, pulse=0.5 + 0.5 * math.sin(i / n * 2 * math.pi))); durs.append(110)
        durs[-1] = 2600                          # hold to read
        done.append(k)
    frames.append(render(done, None)); durs.append(2200)

    small = [f.resize((OUT_W, int(H * OUT_W / W)), Image.LANCZOS) for f in frames]
    sample = small[::3]
    sheet = Image.new("RGB", (small[0].width, small[0].height * len(sample)))
    for i, f in enumerate(sample):
        sheet.paste(f, (0, i * f.height))
    pal = sheet.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in small]
    OUT.parent.mkdir(exist_ok=True)
    q[0].save(OUT, save_all=True, append_images=q[1:], duration=durs, loop=0, optimize=True)
    print(OUT, len(frames), "frames", round(sum(durs) / 1000, 1), "s", round(OUT.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
