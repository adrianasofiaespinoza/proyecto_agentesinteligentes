"""
Graba un GIF del agente entrenado (evidencia cualitativa pedida en el enunciado).

Uso:
  python -m experiments.record_gif --run results/cartpole_dqn/seed_42 --out results/gifs/cartpole_dqn.gif
  python -m experiments.record_gif --run results/lunarlander_ddqn/seed_42 --model model_final.pt
"""
import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.dqn import DQNAgent  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="carpeta results/<exp>/seed_<s>")
    p.add_argument("--model", default="model_final.pt")
    p.add_argument("--seed", type=int, default=20000)
    p.add_argument("--every", type=int, default=2, help="guarda 1 de cada N frames")
    p.add_argument("--out")
    args = p.parse_args()

    run = Path(args.run)
    cfg = json.load(open(run / "config.json"))
    env = gym.make(cfg["env_id"], render_mode="rgb_array")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, cfg, np.random.default_rng(0))
    agent.load(run / args.model)

    obs, _ = env.reset(seed=args.seed)
    frames, total, done, t = [env.render()], 0.0, False, 0
    while not done:
        obs, r, term, trunc, _ = env.step(agent.greedy_action(obs))
        total += r
        t += 1
        done = term or trunc
        if t % args.every == 0:
            frames.append(env.render())
    out = Path(args.out or f"results/gifs/{cfg['exp_name']}_seed{cfg['seed']}.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, frames, duration=1000 * args.every / env.metadata.get("render_fps", 50), loop=0)
    print(f"Guardado {out} | retorno del episodio = {total:.1f}")


if __name__ == "__main__":
    main()
