"""Chart of benchmark results (PNG, matplotlib)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
MUTED = "#b9b8b2"
# Fixed colour per method (categorical slots 1-4), identical in every panel.
METHOD_COLORS = {
    "dual-memory": "#2a78d6",
    "summarize-all": "#eb6834",
    "vector-rag": "#1baf7a",
    "full-history": "#eda100",
}
MAIN = ["dual-memory", "summarize-all", "vector-rag", "full-history"]
TYPES = [
    ("current", "current\n(changed)"),
    ("previous", "previous\nvalue"),
    ("static", "static"),
    ("friend", "friend\n(distractor)"),
]


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def render_chart(main: dict[str, Any], sweep: list[dict[str, Any]] | None, out: str | Path) -> Path:
    """Three panels: overall accuracy, accuracy by question type, accuracy vs budget."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = {r["method"]: r for r in main["results"]}
    has_sweep = bool(sweep and len(sweep) > 1)
    fig, axes = plt.subplots(
        1,
        3 if has_sweep else 2,
        figsize=(15 if has_sweep else 10.5, 5.2),
        gridspec_kw={"width_ratios": [1.1, 1.3, 1.1][: 3 if has_sweep else 2]},
    )
    fig.patch.set_facecolor(SURFACE)

    # A: overall accuracy, every method, sorted.
    ax = axes[0]
    ordered = sorted(res.values(), key=lambda r: (r["accuracy"], r["method"] in METHOD_COLORS))
    names = [r["method"] for r in ordered]
    vals = [100 * r["accuracy"] for r in ordered]
    colors = [METHOD_COLORS.get(n, MUTED) for n in names]
    ax.barh(names, vals, color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    for y, v in enumerate(vals):
        ax.text(v + 1, y, f"{v:.1f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 108)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_title(
        f"Answer accuracy at {main['config']['budget_tokens']} tokens (%)",
        loc="left",
        fontsize=11,
        color=INK,
        pad=10,
    )

    # B: accuracy by question type, main methods.
    ax = axes[1]
    present = [m for m in MAIN if m in res]
    width = 0.8 / max(len(present), 1)
    for i, m in enumerate(present):
        xs = [j + (i - (len(present) - 1) / 2) * width for j in range(len(TYPES))]
        ys = [100 * (res[m]["by_type"][t] or 0.0) for t, _ in TYPES]
        ax.bar(xs, ys, width=width, color=METHOD_COLORS[m], label=m, edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(range(len(TYPES)), [label for _, label in TYPES])
    ax.set_ylim(0, 105)
    _style(ax)
    ax.legend(
        frameon=False,
        fontsize=9,
        ncol=len(present),
        loc="upper center",
        labelcolor=INK_2,
        bbox_to_anchor=(0.5, -0.13),
        handlelength=1.2,
        columnspacing=1.2,
    )
    ax.set_title("Accuracy by question type (%)", loc="left", fontsize=11, color=INK, pad=10)

    # C: accuracy against context budget.
    if has_sweep and sweep is not None:
        ax = axes[2]
        budgets = [s["config"]["budget_tokens"] for s in sweep]
        ends: list[tuple[float, str]] = []
        for m in present:
            ys = [100 * next(r["accuracy"] for r in s["results"] if r["method"] == m) for s in sweep]
            ax.plot(
                budgets,
                ys,
                color=METHOD_COLORS[m],
                linewidth=2,
                marker="o",
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
            )
            ends.append((ys[-1], m))
        placed: list[float] = []
        for y_end, m in sorted(ends, reverse=True):  # keep end labels at least 6 points apart
            y_label = min([y_end, *[p - 6 for p in placed]]) if placed else y_end
            placed.append(y_label)
            ax.annotate(
                m,
                (budgets[-1], y_end),
                xytext=(budgets[-1] * 1.12, y_label),
                va="center",
                fontsize=8.5,
                color=INK_2,
                textcoords="data",
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.set_ylim(0, 105)
        ax.set_xlim(budgets[0] / 1.15, budgets[-1] * 2.4)
        _style(ax)
        ax.set_xlabel("context budget (tokens)", fontsize=9, color=INK_2)
        ax.set_title("Accuracy vs context budget (%)", loc="left", fontsize=11, color=INK, pad=10)

    c, d = main["config"], main["dataset"]
    fig.text(
        0.01,
        0.015,
        f"Synthetic benchmark, seed {d['seed']}: {d['personas']} personas, {d['turns']} turns, "
        f"{d['questions']} questions. Embedder {c['embedder_label']}; mode {c['mode']} "
        f"(deterministic reader, exact-match judge); budget {c['budget_tokens']} tokens, k={c['k']}.",
        fontsize=8.5,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out
