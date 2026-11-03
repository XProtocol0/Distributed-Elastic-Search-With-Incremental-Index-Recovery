"""Plots and a summary table from the raw evaluation CSVs (D-033).

    python eval/plot.py

Reads eval/results/raw/*.csv (whatever exists), writes PNGs to eval/plots/ and
medians/percentiles to eval/results/summary.md. Generated output: never edit
it by hand, re-run this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "eval" / "results" / "raw"
PLOTS = ROOT / "eval" / "plots"
SUMMARY = ROOT / "eval" / "results" / "summary.md"

# Reference categorical palette, light mode, fixed order (validated: CVD and
# normal-vision separation pass; slots 3-4 are below 3:1 contrast, so every
# series is also direct-labelled, has its own marker, and is in summary.md).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BACKENDS = {"process": "Real processes", "docker": "Docker Compose"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 11,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "font.size": 9.5, "lines.linewidth": 2,
    "legend.frameon": False,
})


def line(ax: plt.Axes, x: pd.Series, y: pd.Series, slot: int, label: str) -> None:
    """One series: 2px line, >=8px markers (thinned on long series); label it
    with `label_ends` once every series on the axes is drawn."""
    ax.plot(x, y, color=SERIES[slot], marker=MARKERS[slot], markersize=6.5,
            markevery=max(1, len(x) // 10), markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)


def label_ends(ax: plt.Axes) -> None:
    """Direct labels at each line's last point, nudged apart so none overlap."""
    ends = []
    for artist in ax.get_lines():
        xs, ys = artist.get_data()
        ends.append([float(ys[-1]), float(xs[-1]), artist.get_label()])
    ends.sort()
    low, high = ax.get_ylim()
    gap = (high - low) * 0.055
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, x, label in ends:
        ax.annotate(label, (x, y), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK_2, annotation_clip=False)


def finish(fig: plt.Figure, name: str, title: str) -> None:
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / name, dpi=160)
    plt.close(fig)
    print(f"wrote {PLOTS / name}")


# ---- recovery time ----------------------------------------------------------


def load_recovery() -> pd.DataFrame | None:
    frames = [pd.read_csv(p) for p in sorted(RAW.glob("recovery_*.csv"))]
    return pd.concat(frames) if frames else None


def recovery_medians(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["backend", "index_docs", "missed", "mode"], as_index=False)
              [["wall_s", "open_s", "catchup_s", "entries_replayed", "snapshot_docs"]].median())


def plot_recovery(med: pd.DataFrame, x: str, fixed: str, name: str, title: str, xlabel: str) -> None:
    backends = [b for b in BACKENDS if b in set(med.backend)]
    fig, axes = plt.subplots(2, len(backends), figsize=(5.2 * len(backends), 6.4), squeeze=False)
    for col, backend in enumerate(backends):
        sub = med[med.backend == backend]
        value = sub.groupby(fixed)[x].nunique().idxmax()  # the sweep's fixed value
        sub = sub[sub[fixed] == value]
        for row, (metric, ylabel) in enumerate([("wall_s", "seconds until caught up (wall clock)"),
                                                 ("catchup_s", "seconds in network catch-up")]):
            ax = axes[row][col]
            for slot, mode in enumerate(["incremental", "full"]):
                s = sub[sub["mode"] == mode].sort_values(x)
                if len(s):
                    line(ax, s[x], s[metric], slot, "incremental replay" if mode == "incremental" else "full rebuild")
            ax.set_xscale("log")
            ax.set_ylim(bottom=0)
            label_ends(ax)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            ax.margins(x=0.25)
            fixed_label = f"{int(value):,} docs in the index" if x == "missed" else f"{int(value):,} missed updates"
            ax.set_title(f"{BACKENDS[backend]} · {fixed_label}", loc="left")
    finish(fig, name, title)


# ---- other experiments ----------------------------------------------------------


def plot_log_growth(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for slot, scenario in enumerate(["insert-only", "churn", "delete-heavy", "churn+truncate"]):
        s = df[df.scenario == scenario]
        if len(s):
            line(ax, s.updates, s.log_bytes / 1e6, slot, scenario)
    ax.set_xlabel("updates after the initial load")
    ax.set_ylabel("update log size (MB)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, df.updates.max() * 1.16)
    label_ends(ax)
    ax.set_title("One shard; churn scenarios start from 5,000 loaded docs", loc="left")
    finish(fig, "log_growth.png", "Update-log growth")


def latency_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["backend", "shards"]).latency_ms
              .quantile([0.5, 0.95, 0.99]).unstack().reset_index()
              .rename(columns={0.5: "p50_ms", 0.95: "p95_ms", 0.99: "p99_ms"}))


def plot_latency(summary: pd.DataFrame, docs: int) -> None:
    backends = [b for b in BACKENDS if b in set(summary.backend)]
    fig, axes = plt.subplots(1, len(backends), figsize=(5.2 * len(backends), 3.9), squeeze=False)
    for col, backend in enumerate(backends):
        ax = axes[0][col]
        s = summary[summary.backend == backend].sort_values("shards")
        for slot, q in enumerate(["p50_ms", "p95_ms", "p99_ms"]):
            line(ax, s.shards, s[q], slot, q.replace("_ms", ""))
        ax.set_xticks(s.shards)
        ax.set_xlabel("shards (1 replica each, primary reads)")
        ax.set_ylabel("search latency (ms)")
        ax.set_ylim(bottom=0)
        label_ends(ax)
        ax.margins(x=0.18)
        ax.set_title(f"{BACKENDS[backend]} · {docs:,} docs, top-10", loc="left")
    finish(fig, "query_latency.png", "Query latency as shards are added")


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v) for v in r) + " |")
    return "\n".join(rows)


def main() -> int:
    sections = ["# Evaluation summary", "", "Generated by `eval/plot.py` from `eval/results/raw/*.csv`. Do not edit by hand.", ""]

    recovery = load_recovery()
    if recovery is not None:
        med = recovery_medians(recovery)
        plot_recovery(med, "missed", "index_docs", "recovery_vs_missed.png",
                      "Recovery time vs missed updates (median of trials)", "missed updates (log scale)")
        plot_recovery(med, "index_docs", "missed", "recovery_vs_index_size.png",
                      "Recovery time vs index size (median of trials)", "docs in the index (log scale)")
        trials = recovery.groupby(["backend", "index_docs", "missed", "mode"]).size().max()
        sections += ["## Recovery time (medians, seconds)", "",
                     f"Up to {trials} trials per point. `wall_s`: node start until live and caught up; "
                     "`open_s`: local snapshot load + log replay; `catchup_s`: network session "
                     "(snapshot fetch + install for a full rebuild).", "",
                     md_table(med.astype({"index_docs": int, "missed": int, "entries_replayed": int, "snapshot_docs": int})), ""]

    growth_path = RAW / "log_growth.csv"
    if growth_path.exists():
        growth = pd.read_csv(growth_path)
        plot_log_growth(growth)
        last = growth.groupby("scenario").tail(1)[["scenario", "updates", "log_bytes", "log_entries", "db_bytes", "live_docs"]]
        last = last.assign(bytes_per_entry=(last.log_bytes / last.log_entries).round(1))
        sections += ["## Update-log growth (end of each scenario)", "", md_table(last, "{:.1f}"), ""]

    latency_frames = [pd.read_csv(p) for p in sorted(RAW.glob("latency_*.csv"))]
    if latency_frames:
        latency = pd.concat(latency_frames)
        summary = latency_summary(latency)
        plot_latency(summary, int(latency.docs.iloc[0]))
        sections += ["## Query latency (ms)", "", md_table(summary.astype({"shards": int}), "{:.2f}"), ""]

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(sections))
    print(f"wrote {SUMMARY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
