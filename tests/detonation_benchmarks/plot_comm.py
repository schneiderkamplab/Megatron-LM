"""Figures for the communication profile and the break-even analysis.

    python plot_comm.py break_even.json out_dir/

Writes comm_profile_8n.png (per-step time and inter-node traffic per method)
and break_even.png (time to target loss vs. inter-node bandwidth).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#1a1a19", "#5f5e58", "#9a998f", "#e6e5df"
# Categorical slots in fixed order; a method keeps its colour in every panel.
COLORS = {
    "none": "#2a78d6", "full_sync": "#eb6834", "striding": "#1baf7a",
    "demo-sync": "#eda100", "random-sync": "#e87ba4", "random-async": "#4a3aa7",
}
COMPUTE, INTRA, INTER = "#c9c8bf", "#2a78d6", "#eb6834"
GPUS_PER_NODE = 8


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def comm_profile(results: dict, out: Path):
    order = ["none", "full_sync", "full", "slicing", "striding",
             "random-sync", "random", "demo-sync", "demo"]
    rows = [(n, results[f"8n/{n}"]) for n in order if f"8n/{n}" in results]
    names = [n if not results[f"8n/{n}"]["async"] else f"{n} (async)" for n, _ in rows]
    y = np.arange(len(rows))[::-1]

    fig, (ax_t, ax_b) = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True,
                                     gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax_t, ax_b):
        style(ax)
        ax.grid(axis="y", visible=False)

    # Step time = compute (incl. hidden communication) + exposed intra + exposed inter.
    for yi, (_, r) in zip(y, rows):
        c = r["comm"]
        parts = [c["step_sec"] - c["intra_exposed_sec"] - c["inter_exposed_sec"],
                 c["intra_exposed_sec"], c["inter_exposed_sec"]]
        left = 0.0
        for value, color in zip(parts, (COMPUTE, INTRA, INTER)):
            ax_t.barh(yi, value, left=left, height=0.62, color=color,
                      edgecolor=SURFACE, linewidth=2)
            left += value
        ax_t.text(left + 0.03, yi, f"{c['step_sec']:.2f} s", va="center", fontsize=9, color=INK)
    ax_t.set_yticks(y, names, fontsize=10, color=INK)
    ax_t.set_xlabel("seconds per step", color=INK_2, fontsize=10)
    ax_t.set_xlim(0, 3.0)
    ax_t.set_title("Step time: where communication is not hidden", loc="left",
                   fontsize=11, color=INK, pad=26)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (COMPUTE, INTRA, INTER)]
    fig.legend(handles, ["compute + overlapped communication",
                         "exposed intra-node (FSDP)", "exposed inter-node (replication)"],
               loc="upper left", bbox_to_anchor=(0.12, 0.93), ncol=3, fontsize=8.5,
               frameon=False, labelcolor=INK_2)

    for yi, (_, r) in zip(y, rows):
        gb = r["comm"]["inter_bytes"] / 1e9
        ax_b.barh(yi, gb, height=0.62, color=INTER, edgecolor=SURFACE, linewidth=2)
        ax_b.text(gb + 0.1, yi, f"{gb:.2f} GB", va="center", fontsize=9, color=INK)
    ax_b.set_xlabel("GB sent across nodes per rank per step", color=INK_2, fontsize=10)
    ax_b.set_xlim(0, 7.8)
    ax_b.set_title("Inter-node traffic", loc="left", fontsize=11, color=INK, pad=26)

    intra = rows[0][1]["comm"]["intra_bytes"] / 1e9
    fig.text(0.01, 0.01, f"OLMo-2-7B, 8 nodes x 8 GCDs on LUMI, rank 0, profiled steps 100-110. "
             f"Intra-node FSDP traffic is {intra:.0f} GB per rank per step for every method.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "comm_profile_8n.png", dpi=160, facecolor=SURFACE)


def break_even(data: dict, out: Path):
    bw = np.array(data["bandwidths_per_rank"]) * GPUS_PER_NODE * 8 / 1e9  # Gbit/s per node
    results = data["results"]
    panels = [(8, "5.0"), (8, "4.6"), (32, "5.6")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    drawn = []
    for ax, (nodes, target) in zip(axes, panels):
        style(ax)
        b_eff = results[f"{nodes}n/none"]["b_eff"] * GPUS_PER_NODE * 8 / 1e9
        ax.axvline(b_eff, color=MUTED, linewidth=1, linestyle="--")
        ax.text(b_eff, 380, "LUMI ", fontsize=8, color=INK_2, va="top", ha="right")
        labels = []
        for name, color in COLORS.items():
            r = results.get(f"{nodes}n/{name}")
            if r is None:
                continue
            hours = r["targets"][target]["hours_measured"]
            if hours is None:
                continue
            ax.plot(bw, hours, color=color, linewidth=2)
            labels.append([np.log10(hours[-1]), name])
            if name not in drawn:
                drawn.append(name)
        # Direct labels at the right end, pushed apart to at least 0.16 decades.
        labels.sort()
        for prev, cur in zip(labels, labels[1:]):
            cur[0] = max(cur[0], prev[0] + 0.16)
        for log_y, name in labels:
            ax.text(bw[-1] * 1.25, 10**log_y, name, color=INK, fontsize=8.5, va="center")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.1, 400)
        ax.set_xlim(bw[0], bw[-1] * 12)
        ax.set_xticks([t for t in (1, 10, 100, 1000) if t >= bw[0]])
        ax.spines["bottom"].set_bounds(bw[0], bw[-1])
        ax.set_xlabel("inter-node bandwidth per node (Gbit/s)", color=INK_2, fontsize=10)
        ax.set_title(f"{nodes} nodes: hours to val loss {target}", loc="left",
                     fontsize=11, color=INK)
    axes[0].set_ylabel("hours", color=INK_2, fontsize=10)
    handles = [plt.Line2D([], [], color=COLORS[n], linewidth=2) for n in drawn]
    fig.legend(handles, drawn, loc="upper right", ncol=len(drawn), fontsize=8.5,
               frameon=False, labelcolor=INK_2)
    fig.text(0.01, 0.01, "Model: measured steps-to-target x step time at bandwidth B "
             "(measured overlap). Methods that never reach the target are omitted.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out / "break_even.png", dpi=160, facecolor=SURFACE)


def main():
    data = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    comm_profile(data["results"], out)
    break_even(data, out)


if __name__ == "__main__":
    main()
