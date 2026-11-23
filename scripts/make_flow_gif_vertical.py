"""Vertical 4:5 (1080x1350) version of the alignment-flow GIF, for the LinkedIn feed.

Reuses all drawing/animation code from make_flow_gif.py and only swaps the layout.
Output: assets/llm_alignment_flow_vertical.gif
"""
import make_flow_gif as g

g.W, g.H, g.OUT_W = 1080, 1350, 1080
g.OUT = g.ROOT / "assets" / "llm_alignment_flow_vertical.gif"

# Row 1: three stages, row 2: three methods, row 3: result.
g.BOX = {"pre": (40, 110, 340, 340), "sft": (390, 110, 690, 340), "hf": (740, 110, 1040, 340),
         "ppo": (40, 420, 340, 540), "grpo": (390, 420, 690, 540), "dpo": (740, 420, 1040, 540),
         "out": (290, 610, 790, 790)}
g.ENTRY = [(5, 225), (40, 225)]
g.ARROWS = {
    "sft": [(340, 225), (390, 225)], "hf": [(690, 225), (740, 225)],
    "ppo": [(890, 340), (890, 380), (190, 380), (190, 420)],
    "grpo": [(890, 340), (890, 380), (540, 380), (540, 420)],
    "dpo": [(890, 340), (890, 420)],
    "out_ppo": [(190, 540), (190, 575), (540, 575), (540, 610)],
    "out_grpo": [(540, 540), (540, 610)],
    "out_dpo": [(890, 540), (890, 575), (540, 575), (540, 610)],
}
# Explanation panel below: caption on top, animation underneath.
g.PANEL_TOP, g.CAP_XY, g.BODY_XY, g.DIVIDER = 825, (80, 850), (80, 900), False
g.PX0, g.PY0 = 190, 1010
g.F_CAPH, g.F_CAP = g.F("segoeuib.ttf", 30), g.F("segoeui.ttf", 22)
g.IDLE_TEXT = "From raw text to a safe assistant"

if __name__ == "__main__":
    g.main()
