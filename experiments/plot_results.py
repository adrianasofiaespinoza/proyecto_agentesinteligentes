"""
Curvas de aprendizaje: media entre semillas con banda de ± 1 desviacion estandar.

Cada experimento es una carpeta results/<exp_name>/ con subcarpetas seed_*/eval.csv
(columnas minimas: step, eval_mean). Sirve tambien para los resultados de Q-learning
si se guardan con el mismo formato.

Ejemplos:
  # Tier 1: Q-learning vs DQN en CartPole
  python -m experiments.plot_results --exps results/cartpole_qlearning results/cartpole_dqn \
      --labels "Q-learning" "DQN" --threshold 475 --out results/figures/tier1_cartpole.png

  # Tier 3: DQN vs Double DQN en LunarLander (retorno de evaluacion)
  python -m experiments.plot_results --exps results/lunarlander_dqn results/lunarlander_ddqn \
      --labels "DQN" "Double DQN" --threshold 200 --out results/figures/tier3_eval.png

  # Tier 3: sobreestimacion de Q (estimado vs real en el estado inicial)
  python -m experiments.plot_results --exps results/lunarlander_dqn results/lunarlander_ddqn \
      --labels "DQN" "Double DQN" --mode overestimation --out results/figures/tier3_q.png

  # Retorno de entrenamiento (con exploracion), separado del de evaluacion
  python -m experiments.plot_results --exps ... --mode train --out ...
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Paleta categorica fija (orden estable: el experimento i siempre tiene el mismo color)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e1e0d9"


def load_eval(exp_dir, column):
    """Matriz (n_semillas, n_puntos) de `column`, alineada por 'step'."""
    frames = []
    for f in sorted(Path(exp_dir).glob("seed_*/eval.csv"), key=lambda f: int(f.parent.name.split("_")[1])):
        df = pd.read_csv(f)[["step", column]].set_index("step")[column]
        frames.append(df.rename(f.parent.name))
    if not frames:
        raise SystemExit(f"No se encontraron seed_*/eval.csv en {exp_dir}")
    table = pd.concat(frames, axis=1).dropna()       # solo pasos presentes en todas las semillas
    return table.index.values, table.values.T


def load_train(exp_dir, bin_size):
    """Retorno medio de los episodios de entrenamiento que terminan en cada ventana de pasos."""
    rows = []
    for f in sorted(Path(exp_dir).glob("seed_*/train_episodes.csv"), key=lambda f: int(f.parent.name.split("_")[1])):
        df = pd.read_csv(f)
        df["bin"] = ((df["step"] - 1) // bin_size + 1) * bin_size
        rows.append(df.groupby("bin")["episode_return"].mean().rename(f.parent.name))
    table = pd.concat(rows, axis=1).dropna()
    return table.index.values, table.values.T


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))


def band(ax, x, Y, color, label, ls="-", clip_max=None):
    mean, std = Y.mean(axis=0), Y.std(axis=0)
    upper = mean + std if clip_max is None else np.minimum(mean + std, clip_max)
    ax.plot(x, mean, color=color, lw=1.8, ls=ls, label=label)
    ax.fill_between(x, mean - std, upper, color=color, alpha=0.18, lw=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exps", nargs="+", required=True)
    p.add_argument("--labels", nargs="+")
    p.add_argument("--mode", choices=["eval", "train", "overestimation"], default="eval")
    p.add_argument("--threshold", type=float)
    p.add_argument("--bin", type=int, default=10000, help="ventana de pasos para --mode train")
    p.add_argument("--title", default="")
    p.add_argument("--clip-max", type=float,
                   help="recorta la banda superior al retorno maximo posible (p. ej. 500 en CartPole)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    labels = args.labels or [Path(e).name for e in args.exps]

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    style(ax)

    for i, (exp, label) in enumerate(zip(args.exps, labels)):
        c = COLORS[i % len(COLORS)]
        if args.mode == "eval":
            x, Y = load_eval(exp, "eval_mean")
            band(ax, x, Y, c, f"{label} (n={len(Y)})", clip_max=args.clip_max)
            ylabel = "Evaluation return (10 greedy episodes)"
        elif args.mode == "train":
            x, Y = load_train(exp, args.bin)
            band(ax, x, Y, c, f"{label} (n={len(Y)})", clip_max=args.clip_max)
            ylabel = "Training episode return (with ε-greedy)"
        else:
            x, est = load_eval(exp, "q0_estimate")
            _, real = load_eval(exp, "q0_actual")
            band(ax, x, est, c, f"{label}: estimated max Q(s₀,a)", ls="--")
            band(ax, x, real, c, f"{label}: actual discounted return")
            ylabel = "Value of initial state"

    if args.threshold is not None and args.mode != "overestimation":
        ax.axhline(args.threshold, color=MUTED, lw=1, ls=":")
        ax.annotate(f"threshold {args.threshold:g}", xy=(0, args.threshold),
                    xycoords=("axes fraction", "data"), xytext=(4, -10),
                    textcoords="offset points", ha="left", fontsize=8, color=MUTED,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    ax.set_xlabel("Environment steps", color=INK, fontsize=10)
    ax.set_ylabel(ylabel, color=INK, fontsize=10)
    if args.title:
        ax.set_title(args.title, color=INK, fontsize=11, loc="left")
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.85, fontsize=8, labelcolor=INK)
    fig.text(0.99, 0.01, "Line: mean across seeds · band: ±1 std", ha="right",
             fontsize=7, color=MUTED)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"Guardado {args.out}")


if __name__ == "__main__":
    main()
