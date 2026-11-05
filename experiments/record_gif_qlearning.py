"""
Graba un GIF del agente de Q-learning tabular entrenado (tabla Q guardada por train_qlearning.py).

Uso:
  python -m experiments.record_gif_qlearning --run results/cartpole_qlearning/seed_42
  python -m experiments.record_gif_qlearning --run results/cartpole_qlearning/seed_123 --seed 20005
"""
import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.tabular import discretizar_estado, epsilon_greedy  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="carpeta results/cartpole_qlearning/seed_<s>")
    p.add_argument("--seed", type=int, default=20000, help="semilla del episodio grabado")
    p.add_argument("--every", type=int, default=2, help="guarda 1 de cada N frames")
    p.add_argument("--out")
    args = p.parse_args()

    run = Path(args.run)
    cfg = json.load(open(run / "config.json"))
    Q = np.load(run / "q_table.npy")
    n_bins = cfg["n_bins"]

    env = gym.make(cfg["env_id"], render_mode="rgb_array")
    rng = np.random.default_rng(args.seed)          # solo para desempatar, igual que en la evaluación
    obs, _ = env.reset(seed=args.seed)
    frames, total, done, t = [env.render()], 0.0, False, 0
    while not done:
        accion = epsilon_greedy(Q, discretizar_estado(obs, n_bins), 0.0, rng)   # epsilon = 0
        obs, r, term, trunc, _ = env.step(accion)
        total += r
        t += 1
        done = term or trunc
        if t % args.every == 0:
            frames.append(env.render())
    env.close()

    out = Path(args.out or f"results/gifs/{cfg['exp_name']}_seed{cfg['seed']}.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, duration=1000 * args.every / env.metadata.get("render_fps", 50), loop=0)
    print(f"Guardado {out} | retorno del episodio = {total:.1f} | pasos = {t}")


if __name__ == "__main__":
    main()
