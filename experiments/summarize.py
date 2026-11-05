"""
Tabla resumen de la evaluacion final (100 episodios greedy por semilla).

Uso:
  python -m experiments.summarize results/cartpole_dqn
  python -m experiments.summarize results/lunarlander_dqn results/lunarlander_ddqn --out results/tier3_summary.csv

Columnas:
  final_mean_across_seeds  media de las medias de cada semilla
  std_across_seeds         desviacion estandar entre semillas (variabilidad del algoritmo)
  std_episodes             desviacion estandar de los 100 x n_semillas episodios juntos
  solved_seeds             cuantas semillas superan el umbral en la evaluacion final
  steps_to_threshold       primer paso en que la evaluacion de 10 episodios supera el umbral
  train_time_min           tiempo de entrenamiento medio (sin contar evaluaciones)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(exp_dir):
    runs = [json.load(open(f)) for f in sorted(Path(exp_dir).glob("seed_*/final.json"), key=lambda f: int(f.parent.name.split("_")[1]))]
    if not runs:
        raise SystemExit(f"No hay final.json en {exp_dir}")
    means = np.array([r["final_mean"] for r in runs])
    all_eps = np.concatenate([r["final_returns"] for r in runs])
    steps = [r["steps_to_threshold"] for r in runs]
    return {
        "experiment": Path(exp_dir).name,
        "seeds": ",".join(str(r["seed"]) for r in runs),
        "final_mean_across_seeds": round(means.mean(), 2),
        "std_across_seeds": round(means.std(), 2),
        "std_episodes": round(all_eps.std(), 2),
        "per_seed_mean": " / ".join(f"{m:.1f}" for m in means),
        "per_seed_std": " / ".join(f"{r['final_std']:.1f}" for r in runs),
        "solved_seeds": f"{sum(r['solved'] for r in runs)}/{len(runs)}",
        "best_ckpt_mean": round(np.mean([r["best_ckpt_mean"] for r in runs]), 2),
        "steps_to_threshold": " / ".join("not reached" if s is None else f"{s//1000}k" for s in steps),
        "train_time_min": round(np.mean([r["train_time_s"] for r in runs]) / 60, 1),
        "q0_estimate": round(np.mean([r["final_q0_estimate"] for r in runs]), 1),
        "q0_actual": round(np.mean([r["final_q0_actual"] for r in runs]), 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exps", nargs="+")
    p.add_argument("--out")
    args = p.parse_args()
    df = pd.DataFrame([summarize(e) for e in args.exps])
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(df.T.to_string(header=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Guardado {args.out}")


if __name__ == "__main__":
    main()
