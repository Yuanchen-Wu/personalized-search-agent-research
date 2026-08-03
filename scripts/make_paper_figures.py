"""Render publication figures for the paper from saved experiment outputs.

Reads only files under outputs/ (no API calls). Writes vector PDFs (for the
LaTeX) plus PNG previews to reports/paper/figures/.

Usage:
    .venv/bin/python scripts/make_paper_figures.py

Placeholder protocol for pending replications (e.g. the Llama run): add an
entry to MODELS below pointing at that run's output directory; every figure
is re-rendered per model with a filename suffix, so the LaTeX can swap in
figures/<name>_<model-tag>.pdf without touching this script.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "reports", "paper", "figures")

# Pending replications land here: tag -> outputs dir with the same file layout.
# TODO(colleague): uncomment and point at the Llama outputs when the run exists.
MODELS = {
    "": os.path.join(ROOT, "outputs", "adaptive_refanout_v1"),  # default: gemini-3.5-flash
    # "llama": os.path.join(ROOT, "outputs", "adaptive_refanout_llama_v1"),
}

# --- palette & chart chrome (validated defaults; see dataviz palette notes) --
SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]  # blue green magenta yellow
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
MARKERS = ["o", "s", "^", "D"]  # secondary encoding alongside color
# darker text steps for low-contrast slots (relief rule): label text stays readable
LABEL_INK = {"#2a78d6": "#1c5cab", "#008300": "#006300",
             "#e87ba4": "#c04a74", "#eda100": "#9a6f00"}

plt.rcParams.update({
    "font.size": 8,
    "font.family": "sans-serif",
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "axes.titlesize": 8.5,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,  # embed TrueType so camera-ready checks pass
    "ps.fonttype": 42,
})


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5)


def dodge_labels(pairs, min_gap):
    """pairs: [(y, ...)] sorted ascending -> adjusted y positions >= min_gap apart."""
    ys = [p[0] for p in pairs]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    return ys


def fig_decoupling(frontier: pd.DataFrame, tag: str):
    """Two shared-scale panels: retrieval metrics climb with tau, answer metrics stay flat."""
    v2 = frontier[frontier["synth"] == "v2"].sort_values("tau")
    taus = v2["tau"].tolist()
    retrieval = [
        ("Evidence relevance", v2["retrieval_evidence_relevance_mean"].tolist()),
        ("Constraint coverage", v2["retrieval_constraint_coverage_mean"].tolist()),
        ("Persona fit", v2["retrieval_result_persona_fit_mean"].tolist()),
        ("Source quality", v2["retrieval_source_quality_mean"].tolist()),
    ]
    answer = [
        ("Intent satisfaction", v2["final_intent_satisfaction_mean"].tolist()),
        ("Personalization use", v2["final_personalization_target_use_mean"].tolist()),
        ("Specificity", v2["final_specificity_mean"].tolist()),
        ("Groundedness", v2["final_groundedness_mean"].tolist()),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.35), sharey=True)
    ylim = (3.5, 5.05)
    for ax, data, title in zip(axes, [retrieval, answer],
                               ["Retrieval quality", "Final-answer quality"]):
        series = [(lab, vals, SERIES[i], MARKERS[i]) for i, (lab, vals) in enumerate(data)]
        # draw lines
        for lab, vals, color, marker in series:
            ax.plot(taus, vals, color=color, linewidth=1.7, marker=marker,
                    markersize=4, markeredgecolor="white", markeredgewidth=0.6,
                    clip_on=False, zorder=3)
        # dodged direct end labels
        order = sorted(range(len(series)), key=lambda i: series[i][1][-1])
        raw = [series[i][1][-1] for i in order]
        adj = dodge_labels([(y,) for y in raw], min_gap=(ylim[1] - ylim[0]) * 0.085)
        for y_adj, i in zip(adj, order):
            lab, vals, color, _ = series[i]
            ax.annotate(lab, xy=(taus[-1] + 0.06, y_adj), va="center", ha="left",
                        fontsize=6.8, color=LABEL_INK[color], annotation_clip=False)
        ax.set_xticks(taus)
        ax.set_xticklabels([f"$\\tau$={t:g}" for t in taus])
        ax.set_xlim(taus[0] - 0.15, taus[-1] + 1.25)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", fontweight="bold")
        style_axis(ax)
    axes[0].set_ylabel("Judge score (1–5)", color=INK_2)
    fig.tight_layout(w_pad=1.4)
    save(fig, "refanout_decoupling", tag)


def fig_synthesis_vs_loop(frontier: pd.DataFrame, tag: str):
    """Intent vs tau for both synthesizers: the vertical gap dwarfs the slope."""
    taus = sorted(frontier["tau"].unique().tolist())
    base = frontier[frontier["synth"] == "baseline"].sort_values("tau")[
        "final_intent_satisfaction_mean"].tolist()
    v2 = frontier[frontier["synth"] == "v2"].sort_values("tau")[
        "final_intent_satisfaction_mean"].tolist()
    fig, ax = plt.subplots(figsize=(3.25, 2.3))
    ax.plot(taus, v2, color=SERIES[0], linewidth=1.8, marker="o", markersize=4,
            markeredgecolor="white", markeredgewidth=0.6, zorder=3, clip_on=False)
    ax.plot(taus, base, color=MUTED, linewidth=1.6, marker="s", markersize=3.6,
            markeredgecolor="white", markeredgewidth=0.6, zorder=3, clip_on=False)
    ax.annotate("hardened synthesizer", xy=(taus[0], v2[0]), xytext=(0, 7),
                textcoords="offset points", ha="left", fontsize=7, color=SERIES[0],
                fontweight="bold")
    ax.annotate("baseline synthesizer", xy=(taus[0], base[0]), xytext=(0, -11),
                textcoords="offset points", ha="left", fontsize=7, color=INK_2)
    # gap annotation at tau=5
    ax.annotate("", xy=(taus[-1], v2[-1] - 0.03), xytext=(taus[-1], base[-1] + 0.03),
                arrowprops=dict(arrowstyle="<->", color=INK_2, linewidth=0.8))
    ax.annotate(f"synthesis\n+{v2[-1] - base[-1]:.2f}", xy=(taus[-1], (v2[-1] + base[-1]) / 2),
                xytext=(-4, 0), textcoords="offset points", ha="right", va="center",
                fontsize=7, color=INK)
    ax.annotate(f"loop +{v2[-1] - v2[0]:.2f} (n.s.)",
                xy=(taus[1], v2[1]), xytext=(0, -14), textcoords="offset points",
                ha="center", fontsize=7, color=LABEL_INK[SERIES[0]])
    ax.set_xticks(taus)
    ax.set_xticklabels([f"$\\tau$={t:g}" for t in taus])
    ax.set_ylim(2.9, 4.35)
    ax.set_xlim(taus[0] - 0.15, taus[-1] + 0.15)
    ax.set_ylabel("Intent satisfaction (1–5)", color=INK_2)
    style_axis(ax)
    fig.tight_layout()
    save(fig, "synthesis_vs_loop", tag)


def round1_hist(path):
    counts = Counter()
    with open(path) as f:
        for line in f:
            run = json.loads(line)
            for ev in run.get("events", []):
                if ev.get("round") == 1 and ev.get("coverage_score") is not None:
                    counts[int(round(ev["coverage_score"]))] += 1
                    break
    return counts


def fig_recalibration(runs_dir: str, tag: str):
    """Grouped bars: round-1 coverage histogram before vs after judge recalibration."""
    before = round1_hist(os.path.join(runs_dir, "runs.jsonl"))
    after = round1_hist(os.path.join(runs_dir, "runs_recalibrated_smoke.jsonl"))
    scores = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(3.25, 2.1))
    w = 0.36
    for off, counts, color, label in [(-w / 2 - 0.01, before, MUTED, "before"),
                                      (w / 2 + 0.01, after, SERIES[0], "after")]:
        xs = [s + off for s in scores]
        ys = [counts.get(s, 0) for s in scores]
        bars = ax.bar(xs, ys, width=w, color=color, zorder=3)
        for x, y in zip(xs, ys):
            if y:
                ax.annotate(str(y), xy=(x, y), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=6.8, color=INK_2)
    ax.annotate("before recalibration", xy=(0.55, 5.75), fontsize=7, color=INK_2)
    ax.annotate("after recalibration", xy=(0.55, 5.1), fontsize=7,
                color=LABEL_INK[SERIES[0]], fontweight="bold")
    ax.set_xticks(scores)
    ax.set_xlabel("Round-1 coverage score", color=INK_2)
    ax.set_ylabel("Runs (of 12)", color=INK_2)
    ax.set_ylim(0, 6.6)
    style_axis(ax)
    fig.tight_layout()
    save(fig, "judge_recalibration", tag)


FOREST_METRICS = [
    ("retrieval_evidence_relevance", "Evidence relevance", "retrieval"),
    ("retrieval_constraint_coverage", "Constraint coverage", "retrieval"),
    ("retrieval_result_persona_fit", "Persona fit", "retrieval"),
    ("retrieval_source_quality", "Source quality", "retrieval"),
    ("final_intent_satisfaction", "Intent satisfaction", "answer"),
    ("final_personalization_target_use", "Personalization use", "answer"),
    ("final_specificity", "Specificity", "answer"),
    ("final_groundedness", "Groundedness", "answer"),
]


def load_gains(path):
    df = pd.read_csv(path)
    df = df[df["comparison"] == "t3_to_t5"].set_index("metric")
    return df


def fig_tau_gains_forest(run_dir: str, tag: str):
    """Paired per-pair gains tau=3 -> tau=5 with bootstrap CIs, full set vs
    the engaged subset (pairs whose tau=5 loop actually re-fanned). The zero
    line does the arguing: retrieval CIs clear it, answer CIs sit on it."""
    full = load_gains(os.path.join(run_dir, "v2synth_tau_gains.csv"))
    engaged = load_gains(os.path.join(run_dir, "engaged_v2_tau_gains.csv"))
    fig, ax = plt.subplots(figsize=(3.25, 3.05))
    # retrieval group at y=9..6, gap row, answer group at y=3..0
    positions = [9, 8, 7, 6, 3, 2, 1, 0]
    yticks, ylabels = [], []
    for y, (key, label, group) in zip(positions, FOREST_METRICS):
        for df, off, color, ms in [(full, -0.17, MUTED, "o"),
                                   (engaged, 0.17, SERIES[0], "D")]:
            if key not in df.index:
                continue
            r = df.loc[key]
            mid, lo, hi = (float(r["mean_paired_diff"]),
                           float(r["ci95_low"]), float(r["ci95_high"]))
            ax.plot([lo, hi], [y + off, y + off], color=color, linewidth=1.1,
                    solid_capstyle="round", zorder=3)
            ax.plot([mid], [y + off], marker=ms, markersize=3.8, color=color,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=4)
        yticks.append(y)
        ylabels.append(label)
    ax.axvline(0, color=BASELINE, linewidth=0.9, zorder=2)
    ax.axhline(4.5, color=GRID, linewidth=0.6)
    # bold group headers in the label margin, above each group
    for y, text in [(10.05, "Retrieval quality"), (4.05, "Final-answer quality")]:
        ax.annotate(text, xy=(-0.02, y), xycoords=("axes fraction", "data"),
                    ha="right", va="center", fontsize=7, fontweight="bold",
                    color=INK, annotation_clip=False)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7, color=INK_2)
    ax.set_xlim(-0.45, 0.52)
    ax.set_ylim(-0.7, 10.9)
    ax.set_xlabel("Paired gain, $\\tau$=3 $\\rightarrow$ $\\tau$=5  (95% CI)",
                  color=INK_2)
    # two-series text legend in the empty top-right (color + marker carry identity)
    ax.text(0.51, 10.35, "loop engaged (n=55)", fontsize=6.8, ha="right",
            va="center", color=LABEL_INK[SERIES[0]], fontweight="bold")
    ax.text(0.51, 9.72, "all pairs (n=72)", fontsize=6.8, ha="right",
            va="center", color=INK_2)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, axis="x")
    ax.tick_params(length=0, axis="y")
    fig.tight_layout()
    save(fig, "tau_gains_forest", tag)


def save(fig, name, tag):
    suffix = f"_{tag}" if tag else ""
    for ext in ("pdf", "png"):
        path = os.path.join(FIG_DIR, f"{name}{suffix}.{ext}")
        fig.savefig(path, dpi=220, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"wrote {name}{suffix}.pdf/.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    for tag, run_dir in MODELS.items():
        frontier = pd.read_csv(os.path.join(run_dir, "synth_tau_frontier.csv"))
        fig_decoupling(frontier, tag)
        fig_synthesis_vs_loop(frontier, tag)
        fig_tau_gains_forest(run_dir, tag)
        fig_recalibration(run_dir, tag)


if __name__ == "__main__":
    main()
