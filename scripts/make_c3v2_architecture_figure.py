"""Render the adaptive-loop-v2 architecture figure (paper palette).

Usage: .venv/bin/python scripts/make_c3v2_architecture_figure.py
Writes reports/paper/figures/c3v2_architecture.{pdf,png}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
EVID, SYNTH, OK = "#1baf7a", "#eb6834", "#008300"   # matches c1_two_channels channel colors
BLUE = "#2a78d6"

fig, ax = plt.subplots(figsize=(9.6, 7.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

def box(x, y, w, h, text, edge, fill="#ffffff", fs=9.5, lw=1.6, style="round,pad=0.12"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                edgecolor=edge, facecolor=fill, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, linespacing=1.35)

def arrow(x1, y1, x2, y2, color=INK2, lw=1.6, style="-|>", con="arc3,rad=0.0"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 connectionstyle=con, color=color, linewidth=lw,
                                 mutation_scale=14, shrinkA=2, shrinkB=2))

def edge_label(x, y, text, color, fs=8.6, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color,
            fontstyle="italic", linespacing=1.25)

# --- main spine -------------------------------------------------------------
box(3.2, 9.05, 3.6, 0.72, "user query  +  persona\n(demographics + raw history)", MUTED, fill="#faf9f7", fs=9)
box(3.2, 7.85, 3.6, 0.72, "persona-conditioned fan-out\n(k = 4 queries)", BLUE)
box(3.2, 6.55, 3.6, 0.80, "web search  →  EVIDENCE POOL\n(deduped, accumulates across rounds)", EVID)
box(3.2, 5.25, 3.6, 0.80, "hardened synthesis (v2 prompt)\n→ DRAFT answer", SYNTH)
box(3.05, 3.80, 3.9, 0.95, "LEAK-FREE ANSWER CRITIC  (K = 3)\nscore · gaps · needs_more_evidence", INK, fill="#faf9f7", fs=9.5, lw=1.9)
box(3.2, 2.30, 3.6, 0.80, "FINAL ANSWER\n(approved round, or best round)", OK, fill="#f4faf4", fs=10, lw=2.0)

arrow(5.0, 9.05, 5.0, 8.62)
arrow(5.0, 7.85, 5.0, 7.40)
arrow(5.0, 6.55, 5.0, 6.10)
arrow(5.0, 5.25, 5.0, 4.80)
arrow(5.0, 3.80, 5.0, 3.15, color=OK, lw=2.0)
edge_label(5.75, 3.47, "score ≥ τ  (or max rounds)", OK, ha="left")

# --- right loop: synthesis-bound => revise (no search) ----------------------
box(7.35, 4.95, 2.45, 1.15, "REVISE draft\nagainst the critique\n(same evidence,\n0 searches)", SYNTH, fill="#fdf3ee", fs=9)
arrow(6.95, 4.30, 7.90, 4.95, color=SYNTH, lw=1.8, con="arc3,rad=-0.25")
edge_label(8.45, 4.35, "answer-bound\n(needs_more_evidence = false)", SYNTH)
# Revision returns straight to the critic (skips search AND synthesis).
arrow(7.35, 5.70, 6.70, 4.75, color=SYNTH, lw=1.4, con="arc3,rad=-0.30", style="-|>")
edge_label(7.32, 5.28, "re-critique", SYNTH, fs=8.2, ha="left")

# --- left loop: evidence-bound => targeted fetch ----------------------------
box(0.25, 4.95, 2.45, 1.15, "TARGETED fan-out\n(2 queries for the\nnamed gaps only)", EVID, fill="#eefaf5", fs=9)
arrow(3.05, 4.45, 2.30, 4.95, color=EVID, lw=1.8, con="arc3,rad=0.25")
edge_label(1.55, 4.42, "evidence-bound\n(needs_more_evidence = true)", EVID)
arrow(1.85, 6.10, 3.20, 6.80, color=EVID, lw=1.4, con="arc3,rad=0.3")
edge_label(1.85, 6.75, "merge into pool,\nre-draft", EVID, fs=8.2)

# --- footnotes --------------------------------------------------------------
ax.text(0.25, 1.45, "Retired from v1:  full k-query re-fan-out on every retry  ·  per-round evidence discard",
        fontsize=8.8, color=MUTED)
ax.text(0.25, 1.05, "max_rounds = 4  ·  approval τ = 4  ·  all models pinned gemini-3.5-flash  ·  critic never sees the rubric",
        fontsize=8.8, color=MUTED)
ax.text(0.25, 0.55, "Color code —  green/aqua: evidence channel  ·  orange: synthesis channel  (matches Fig. c1_two_channels)",
        fontsize=8.8, color=MUTED)

ax.set_title("Adaptive loop v2 — retries routed to the failing stage", fontsize=13, color=INK, pad=14)

for ext in ("pdf", "png"):
    fig.savefig(f"/Users/j/work/projects/personalized-search-agent-research/reports/paper/figures/c3v2_architecture.{ext}",
                dpi=300, bbox_inches="tight")
print("saved c3v2_architecture.pdf/.png")
