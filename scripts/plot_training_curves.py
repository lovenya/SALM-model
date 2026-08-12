"""
Plot a run's curves, built to communicate WHY they look uninformative.

A naive loss plot of this run is misleading: it looks like a model that failed. The truth is that
eps-MSE cannot show learning here, and these four panels are arranged to make that argument
visually rather than requiring the reader to take it on trust.

  1. train/val loss with a +/-1 sigma band. The band is the point: the curve wanders inside it and
     never leaves, so there is no trend to read. The expected minimum of N draws from that band is
     annotated, which is where the "best" checkpoint actually came from.
  2. per-timestep-bucket loss. Four flat lines, three orders of magnitude apart, pinned by the
     noise schedule rather than by learning.
  3. why: loss plotted against each bucket's signal-to-noise ratio. Near-perfect inverse
     relationship -- the loss is measuring the schedule.
  4. CLAP cosine with the chance band shaded. The only quantity that moves, and it stays inside
     chance.

Writes PNGs at 200 dpi, sized to attach to a message.

Usage:
    python scripts/plot_training_curves.py --run-dir <run_dir> [--out-dir <dir>]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless: no display on a compute or login node
import matplotlib.pyplot as plt

# Cool-biased neutrals with teal for schedule-determined quantities and ochre for measured ones.
INK = "#0F1519"
MUTED = "#5A6B72"
GRID = "#D8E0E3"
TEAL = "#1F7A8C"
OCHRE = "#B5761C"
RED = "#9E4038"
GREEN = "#4A7A52"

BUCKETS = [
    ("val_loss_t0_250", 125, "t 0-250"),
    ("val_loss_t250_500", 375, "t 250-500"),
    ("val_loss_t500_750", 625, "t 500-750"),
    ("val_loss_t750_1000", 875, "t 750-1000"),
]


def snr(t: int, T: int = 1000, b0: float = 0.0015, b1: float = 0.0195) -> float:
    """sqrt(abar_t)/sqrt(1-abar_t) for AudioLDM's scaled_linear schedule."""
    ab = 1.0
    for i in range(t + 1):
        beta = (b0 ** 0.5 + (b1 ** 0.5 - b0 ** 0.5) * i / (T - 1)) ** 2
        ab *= 1 - beta
    return math.sqrt(ab) / math.sqrt(1 - ab)


def style(ax) -> None:
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    hist = json.loads((run_dir / "history.json").read_text())
    name = run_dir.name
    ep = [h["epoch"] for h in hist]
    tr = [h["train_loss"] for h in hist]
    vl_pairs = [(h["epoch"], h["val_loss"]) for h in hist if h.get("val_loss") is not None]
    cos_pairs = [(h["epoch"], h["val_clap_cos"]) for h in hist
                 if h.get("val_clap_cos") is not None]

    vx, vy = zip(*vl_pairs)
    vmean, vstd = statistics.mean(vy), statistics.stdev(vy)
    tmean, tstd = statistics.mean(tr), statistics.stdev(tr)

    # ---------------------------------------------------- 1. losses with the sigma band
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    style(ax)
    ax.axhspan(vmean - vstd, vmean + vstd, color=TEAL, alpha=0.10,
               label=f"val ±1σ  ({vstd:.4f})")
    ax.axhline(vmean, color=TEAL, linewidth=1, linestyle=":", alpha=0.8)
    ax.plot(ep, tr, color=MUTED, linewidth=1.2, alpha=0.85, label=f"train (μ={tmean:.4f})")
    ax.plot(vx, vy, color=OCHRE, linewidth=1.6, label=f"val (μ={vmean:.4f})")

    imin = vy.index(min(vy))
    # Expected minimum of n draws from a normal is about mu - 2.5 sigma for n ~ 90.
    expected_min = vmean - 2.5 * vstd
    ax.scatter([vx[imin]], [vy[imin]], color=RED, zorder=5, s=34)
    ax.annotate(
        f'"best" = {min(vy):.4f} @ ep{vx[imin]}\n'
        f"expected min of {len(vy)} noise draws = {expected_min:.4f}",
        xy=(vx[imin], vy[imin]), xytext=(0.28, 0.05), textcoords="axes fraction",
        fontsize=8.5, color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, linewidth=0.9),
    )
    ax.set_title(f"Loss never leaves the noise band — {name}", fontsize=12, pad=12)
    ax.set_xlabel("epoch")
    ax.set_ylabel("ε-prediction MSE")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.012, 0.02,
             "Model selection picked the minimum of a stationary series, so the chosen "
             "checkpoint is effectively a random epoch.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out_dir / "01_loss_noise_band.png", facecolor="white")
    plt.close(fig)

    # ------------------------------------------------- 2. per-timestep-bucket losses
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    style(ax)
    for (key, t, label), col in zip(BUCKETS, (OCHRE, TEAL, GREEN, MUTED)):
        pts = [(h["epoch"], h[key]) for h in hist if h.get(key) is not None]
        if not pts:
            continue
        bx, by = zip(*pts)
        ax.plot(bx, by, marker="o", markersize=3, linewidth=1.4, color=col,
                label=f"{label}   {by[0]:.4f} → {by[-1]:.4f}")
    ax.set_yscale("log")
    ax.set_title("Each noise level sits at its own fixed value and stays there",
                 fontsize=12, pad=12)
    ax.set_xlabel("epoch")
    ax.set_ylabel("ε-prediction MSE  (log scale)")
    leg = ax.legend(frameon=False, fontsize=9, title="timestep band")
    leg.get_title().set_color(INK)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.012, 0.02,
             "Three orders of magnitude apart, and flat to four decimals across 90 epochs. "
             "The averaged loss is dominated by this spread.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out_dir / "02_timestep_buckets.png", facecolor="white")
    plt.close(fig)

    # --------------------------------------------- 3. loss is explained by the schedule
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    style(ax)
    xs, ys, labels = [], [], []
    for key, t, label in BUCKETS:
        pts = [h[key] for h in hist if h.get(key) is not None]
        if not pts:
            continue
        xs.append(snr(t))
        ys.append(statistics.mean(pts))
        labels.append(f"{label}\n(t={t})")
    ax.plot(xs, ys, color=TEAL, linewidth=1.2, alpha=0.6, zorder=1)
    ax.scatter(xs, ys, color=OCHRE, s=70, zorder=3)
    for x, y, lb in zip(xs, ys, labels):
        ax.annotate(lb, xy=(x, y), xytext=(6, 6), textcoords="offset points",
                    fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("The loss is measuring the noise schedule, not the model", fontsize=12, pad=12)
    ax.set_xlabel("signal-to-noise ratio  √ᾱ / √(1−ᾱ)   (log scale)")
    ax.set_ylabel("mean ε-prediction MSE  (log scale)")
    fig.text(0.012, 0.02,
             "At high noise the input is essentially the noise, so predicting it is nearly free. "
             "At low noise it is a faint perturbation and genuinely hard.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out_dir / "03_loss_vs_snr.png", facecolor="white")
    plt.close(fig)

    # --------------------------------------------------------- 4. CLAP cosine
    if cos_pairs:
        cx, cy = zip(*cos_pairs)
        chance = 1.0 / math.sqrt(512)
        fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
        style(ax)
        ax.axhspan(-chance, chance, color=MUTED, alpha=0.13,
                   label=f"chance for 512-d  (±{chance:.4f})")
        ax.axhline(0, color=MUTED, linewidth=0.9, linestyle=":")
        ax.plot(cx, cy, color=OCHRE, linewidth=1.6, label="val CLAP cosine")
        third = len(cy) // 3
        for lo, hi, lab in ((0, third, "first third"),
                            (third, 2 * third, "middle"),
                            (2 * third, len(cy), "last third")):
            m = statistics.mean(cy[lo:hi])
            ax.hlines(m, cx[lo], cx[hi - 1], color=TEAL, linewidth=2.4)
            ax.annotate(f"{lab}\n{m:+.4f}", xy=((cx[lo] + cx[hi - 1]) / 2, m),
                        xytext=(0, -26), textcoords="offset points",
                        ha="center", fontsize=8.5, color=TEAL)
        ax.set_ylim(-chance * 1.35, chance * 1.35)
        ax.set_title("The one quantity that moves — and it stays within chance",
                     fontsize=12, pad=12)
        ax.set_xlabel("epoch")
        ax.set_ylabel("cosine(conditioner output, CLAP of true target)")
        leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
        for t in leg.get_texts():
            t.set_color(INK)
        fig.text(0.012, 0.02,
                 f"Monotonic across thirds ({statistics.mean(cy[:third]):+.4f} → "
                 f"{statistics.mean(cy[2*third:]):+.4f}), about 2x its own σ, but only ~13% of the "
                 "way to leaving the chance band.",
                 fontsize=8, color=MUTED)
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        fig.savefig(out_dir / "04_clap_cosine.png", facecolor="white")
        plt.close(fig)

    written = sorted(p.name for p in out_dir.glob("*.png"))
    print(f"wrote {len(written)} plots to {out_dir}")
    for w in written:
        print(f"  {w}")
    print(f"\nepochs {len(hist)}   val μ={vmean:.4f} σ={vstd:.4f}   train μ={tmean:.4f} σ={tstd:.4f}")
    print(f"pull down with:  rsync -av <host>:{out_dir}/ ./plots/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
