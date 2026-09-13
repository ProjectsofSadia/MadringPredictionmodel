"""Three pre-race figures. Clean white engineering aesthetic, no F1 branding,
no team logos, no invented numbers - everything comes from the run that produced
the frozen prediction JSON.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import FIGURES  # noqa: E402

INK = "#111111"
ACCENT = "#1a1a1a"
GREY = "#9a9a9a"


def _style(ax, title: str, subtitle: str = "") -> None:
    ax.set_facecolor("white")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=INK, labelsize=9)
    ax.set_title(title, color=INK, fontsize=13, fontweight="bold", loc="left", pad=14)
    if subtitle:
        ax.text(
            0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color="#555555",
            ha="left", va="bottom",
        )


def plot_model_validation(summary: dict, selected: str, path=None):
    labels = ["B0_null", "B1_qualifying", "B2_recent_form", "B3_ridge", "XGBoost"]
    labels = [l for l in labels if l in summary]
    maes = [summary[l]["mae_pct"] for l in labels]
    pretty = {
        "B0_null": "Null (field median)",
        "B1_qualifying": "Qualifying only",
        "B2_recent_form": "Recent form",
        "B3_ridge": "Ridge",
        "XGBoost": "XGBoost",
    }

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    fig.patch.set_facecolor("white")
    colors = [ACCENT if l == selected else "#cfcfcf" for l in labels]
    bars = ax.bar([pretty[l] for l in labels], maes, color=colors, width=0.6)
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.set_ylabel("Out-of-fold MAE (% of lap time)", color=INK, fontsize=10)
    _style(ax, "MODEL CHECK", "Walk-forward validation on 2026 races. Lower is better. "
                              f"Selected model: {pretty.get(selected, selected)}")
    fig.tight_layout()
    path = path or FIGURES / "01_model_validation.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def plot_win_probability(rows: list[dict], n_sims: int, case: str, path=None):
    rows = [r for r in rows if r["win_pct"] > 0][:12]
    if not rows:
        rows = sorted(rows, key=lambda r: -r["win_pct"])[:12]
    names = [r["driver"] for r in rows][::-1]
    vals = [r["win_pct"] for r in rows][::-1]
    errs = [r["win_mc_stderr_pct"] for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.barh(names, vals, xerr=errs, color=ACCENT, height=0.62,
            error_kw=dict(ecolor=GREY, lw=1, capsize=2))
    for y, v in enumerate(vals):
        ax.text(v + max(vals) * 0.015, y, f"{v:.1f}%", va="center", fontsize=9, color=INK)
    ax.set_xlabel("Win probability (%)", color=INK, fontsize=10)
    _style(
        ax,
        "WIN PROBABILITY",
        f"{n_sims:,} simulated races - {case} overtaking case - error bars are Monte Carlo standard error",
    )
    fig.tight_layout()
    path = path or FIGURES / "02_win_probability.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def plot_grid_vs_expected(rows: list[dict], n_sims: int, path=None):
    rows = sorted(rows, key=lambda r: r["grid_position"])
    names = [r["driver"] for r in rows]
    gained = [r["expected_positions_gained"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 6.2), dpi=200)
    fig.patch.set_facecolor("white")
    colors = [ACCENT if g >= 0 else "#b9b9b9" for g in gained]
    ax.barh(names[::-1], gained[::-1], color=colors[::-1], height=0.62)
    ax.axvline(0, color=GREY, lw=1)
    for y, r in enumerate(rows[::-1]):
        ax.text(
            r["expected_positions_gained"]
            + (0.12 if r["expected_positions_gained"] >= 0 else -0.12),
            y,
            f"P{r['grid_position']} to {r['expected_finish']:.1f}",
            va="center",
            ha="left" if r["expected_positions_gained"] >= 0 else "right",
            fontsize=8,
            color="#555555",
        )
    ax.set_xlabel("Expected positions gained (+) or lost (-)", color=INK, fontsize=10)
    _style(ax, "GRID vs EXPECTED FINISH",
           f"Mean simulated finishing position across {n_sims:,} races, ordered by starting grid")
    fig.tight_layout()
    path = path or FIGURES / "03_grid_vs_expected_finish.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path
