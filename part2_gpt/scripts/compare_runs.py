"""Plot the validation loss of several training runs on one chart.

Usage: python part2_gpt/scripts/compare_runs.py gpt_cpu gpt_cpu_noscale
       python part2_gpt/scripts/compare_runs.py gpt_cpu gpt_cpu_noscale \
           --labels "with sqrt(d) scaling" "without" --out ablation_attention_scale.png

Each run name refers to a loss history saved by train.py in part2_gpt/logs/<run>.json.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw straight to image files; no window needed

import matplotlib.pyplot as plt  # noqa: E402

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order: blue, orange, aqua, yellow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="run names, e.g. gpt_cpu gpt_cpu_noscale")
    parser.add_argument("--labels", nargs="+", help="legend labels, one per run (default: run names)")
    parser.add_argument("--title", default="Validation loss")
    parser.add_argument("--from-step", type=int, default=0,
                        help="hide earlier points (the step-0 loss of ~4.2 squashes the rest of the chart)")
    parser.add_argument("--out", default="comparison.png", help="file name inside assets/")
    args = parser.parse_args()

    if len(args.runs) > len(COLORS):
        raise SystemExit(f"Compare at most {len(COLORS)} runs at once")
    labels = args.labels or args.runs
    if len(labels) != len(args.runs):
        raise SystemExit("Give exactly one --labels entry per run")

    fig, ax = plt.subplots(figsize=(7, 4))
    for run, label, color in zip(args.runs, labels, COLORS):
        path = LOG_DIR / f"{run}.json"
        if not path.exists():
            raise SystemExit(f"No log at {path}\nTrain it first with train.py")
        log = json.loads(path.read_text(encoding="utf-8"))
        points = [(s, v) for s, v in zip(log["history"]["step"], log["history"]["val"]) if s >= args.from_step]
        steps, val = [s for s, _ in points], [v for _, v in points]
        ax.plot(steps, val, color=color, linewidth=2, label=f"{label} (final {val[-1]:.3f})")
        print(f"{run:24s} final val loss {val[-1]:.4f}  best {log['best_val']:.4f}")

    ax.set_title(args.title, loc="left")
    ax.set_xlabel("Step", color="#52514e")
    ax.set_ylabel("Cross-entropy loss", color="#52514e")
    ax.legend(frameon=False)
    ax.grid(color="#e8e8e6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors="#52514e")
    fig.tight_layout()

    out_path = ASSETS_DIR / args.out
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
