"""Animate the LLM alignment flow diagram into a GIF for the LinkedIn post.

Input : extra_ai_generated_explanation/<base image> (local only, gitignored)
Output: assets/llm_alignment_flow.gif
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "extra_ai_generated_explanation" / "Generated Image November 23, 2026 - 6_38PM.jpg"
CHART = ROOT / "results" / "05_comparison.png"
OUT = ROOT / "assets" / "llm_alignment_flow.gif"
FONT_DIR = Path("C:/Windows/Fonts")

# Box coordinates in the 1376x768 base image.
BOX = {
    "pre": (58, 197, 290, 542), "sft": (343, 277, 538, 518), "hf": (591, 277, 792, 508),
    "ppo": (853, 144, 1062, 294), "grpo": (853, 310, 1062, 450), "dpo": (853, 465, 1062, 610),
    "out": (1118, 296, 1320, 485),
}
COLOR = {"pre": (170, 180, 195), "sft": (66, 135, 245), "hf": (168, 85, 247), "ppo": (249, 140, 50),
         "grpo": (60, 200, 90), "dpo": (40, 190, 180), "out": (120, 230, 255)}
# Dot paths (polylines) between boxes.
PATH = {
    "pre": [(0, 370), (58, 370)], "sft": [(290, 380), (343, 380)], "hf": [(538, 380), (591, 380)],
    "ppo": [(792, 380), (825, 380), (825, 219), (853, 219)], "grpo": [(792, 380), (853, 380)],
    "dpo": [(792, 380), (825, 380), (825, 540), (853, 540)],
    "out": [(1062, 380), (1118, 390)],
}
STEPS = [  # (box, caption, example answer)
    ("pre", "1 · Pretraining: trillions of words of raw web text. The model learns to predict the next word.",
     "Chest pain spreading to left arm? ... and also my back, forums say..."),
    ("sft", "2 · SFT: thousands of (question, ideal answer) pairs turn it into an assistant.",
     "It could be many things. Rest and see a doctor if it continues."),
    ("hf", "3 · Human feedback: people pick the better of two answers. A reward model learns their taste.",
     "Answer A  vs  Answer B  ->  A is safer"),
    ("ppo", "A · PPO: reward model scores each answer; a KL penalty stops it gaming the score.",
     "Used for ChatGPT (InstructGPT)"),
    ("grpo", "B · GRPO: 8 answers to one question, each scored against the group average. No value model.",
     "Used for DeepSeek-R1 · great when answers can be checked"),
    ("dpo", "C · DPO / ORPO: learn straight from chosen vs rejected pairs. No reward model, no RL loop.",
     "Simpler and cheaper"),
    ("out", "Result: an aligned LLM that is helpful, correct and safe.",
     "Chest pain to left arm? Call emergency services now."),
]


def font(name, size):
    return ImageFont.truetype(str(FONT_DIR / name), size)


F_CAP, F_EX, F_BIG = font("seguisb.ttf", 22), font("segoeui.ttf", 20), font("segoeuib.ttf", 34)
BG = (16, 22, 36)


def point_on(path, t):
    """Point at fraction t (0..1) along a polyline."""
    segs = list(zip(path, path[1:]))
    lens = [((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in segs]
    d = t * sum(lens)
    for (a, b), L in zip(segs, lens):
        if d <= L:
            r = d / L if L else 0
            return a[0] + (b[0] - a[0]) * r, a[1] + (b[1] - a[1]) * r
        d -= L
    return path[-1]


def frame(base, done, active, caption, example, dots=None):
    img = base.copy()
    shade = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(shade)
    for k, (x0, y0, x1, y1) in BOX.items():
        if k == active:
            continue
        d.rectangle((x0 - 4, y0 - 4, x1 + 4, y1 + 4), fill=(0, 0, 0, 70 if k in done else 165))
    img = Image.alpha_composite(img, shade)
    if active:
        glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        x0, y0, x1, y1 = BOX[active]
        ImageDraw.Draw(glow).rounded_rectangle((x0 - 6, y0 - 6, x1 + 6, y1 + 6), 14,
                                               outline=COLOR[active] + (255,), width=8)
        img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(6)))
        ImageDraw.Draw(img).rounded_rectangle((x0 - 3, y0 - 3, x1 + 3, y1 + 3), 12,
                                              outline=COLOR[active] + (255,), width=3)
    dr = ImageDraw.Draw(img)
    for (x, y), c in dots or []:
        dr.ellipse((x - 7, y - 7, x + 7, y + 7), fill=c + (255,))
    # Replace the bottom example strip.
    dr.rectangle((40, 615, 1340, 740), fill=BG + (255,))
    dr.rounded_rectangle((56, 622, 1320, 730), 12, outline=(70, 85, 110, 255), width=2)
    dr.text((76, 634), caption, font=F_CAP, fill=(235, 240, 250, 255))
    if example:
        c = COLOR.get(active, (200, 200, 200))
        dr.rounded_rectangle((76, 675, 96 + dr.textlength(example, font=F_EX) + 20, 715), 10,
                             fill=(30, 40, 60, 255), outline=c + (255,), width=2)
        dr.text((96, 683), example, font=F_EX, fill=(220, 228, 240, 255))
    return img


def result_frame(size):
    img = Image.new("RGBA", size, BG + (255,))
    dr = ImageDraw.Draw(img)
    dr.text((size[0] // 2, 40), "My test: Qwen2.5-0.5B on a free Colab T4", font=F_BIG,
            fill=(255, 255, 255, 255), anchor="mt")
    chart = Image.open(CHART).convert("RGBA")
    chart.thumbnail((1200, 520))
    img.paste(chart, ((size[0] - chart.width) // 2, 100))
    dr.text((size[0] // 2, 640), "GRPO recovered 0.385 -> 0.44 overall · PPO reward-hacked until the KL penalty was raised",
            font=F_CAP, fill=(200, 230, 255, 255), anchor="mt")
    dr.text((size[0] // 2, 690), "github.com/ubaidur404786/medalign-rl  ·  huggingface.co/ubaidii123",
            font=F_EX, fill=(150, 170, 200, 255), anchor="mt")
    return img


def main():
    base = Image.open(BASE).convert("RGBA").resize((1376, 768))
    frames, durs, done = [], [], []
    frames.append(frame(base, [], None, "How does an LLM go from raw text to a safe assistant?", ""))
    durs.append(1800)
    for key, cap, ex in STEPS:
        c = COLOR[key]
        for i in range(8):
            t = i / 7
            dots = [(point_on(PATH[key], max(0, t - j * 0.15)), c) for j in range(3)]
            frames.append(frame(base, done, None, cap, "", dots))
            durs.append(70)
        frames.append(frame(base, done, key, cap, ex))
        durs.append(2600)
        done.append(key)
    frames.append(frame(base, done, "out", "PPO · GRPO · DPO all push answers toward what people prefer.", ""))
    durs.append(1500)
    frames.append(result_frame(base.size))
    durs.append(4500)

    small = [f.convert("RGB").resize((1100, 614), Image.LANCZOS) for f in frames]
    pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in small]
    OUT.parent.mkdir(exist_ok=True)
    pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True)
    print(OUT, len(frames), "frames", round(OUT.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
