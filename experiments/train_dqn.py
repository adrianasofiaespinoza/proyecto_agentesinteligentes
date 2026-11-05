"""
Entrena un agente DQN (o una variante) con UNA semilla y guarda todos los resultados.

Uso:
  python -m experiments.train_dqn --config experiments/configs/cartpole_dqn.json --seed 42
  python -m experiments.train_dqn --config experiments/configs/lunarlander_ddqn.json --seed 123
  # Cambiar cualquier hiperparametro sin tocar el JSON:
  python -m experiments.train_dqn --config ... --seed 42 --set lr=1e-3 --set exp_name=ll_lr1e-3

Salida en results/<exp_name>/seed_<seed>/:
  config.json          configuracion exacta usada
  eval.csv             una fila cada `eval_every` pasos (10 episodios greedy)
  train_episodes.csv   retorno de cada episodio de entrenamiento (con exploracion)
  final.json           evaluacion final (100 episodios, semillas 20000-20099) + tiempos
  model_final.pt       pesos al terminar el entrenamiento
  model_best.pt        pesos con mejor evaluacion intermedia
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.dqn import DQNAgent  # noqa: E402
from common.utils import evaluate, linear_epsilon, load_config, save_json, set_global_seed  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="sobrescribe una clave del config (valor en formato JSON)")
    return p.parse_args()


def parse_overrides(pairs):
    out = {}
    for item in pairs:
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v          # texto plano, p. ej. exp_name=mi_experimento
    return out


def main():
    args = parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    cfg["seed"] = args.seed
    seed = args.seed

    out_dir = Path(args.results_dir) / cfg["exp_name"] / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, out_dir / "config.json")

    # ---------------- Reproducibilidad ----------------
    torch.set_num_threads(1)        # 1 hilo por proceso: resultados estables y semillas en paralelo
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    env = gym.make(cfg["env_id"])
    eval_env = gym.make(cfg["env_id"])          # entorno separado: no altera el de entrenamiento
    env.action_space.seed(seed)
    obs, _ = env.reset(seed=seed)               # semilla solo en el primer reset

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent = DQNAgent(obs_dim, n_actions, cfg, rng)

    total_steps = cfg["total_steps"]
    decay_steps = int(cfg["exploration_fraction"] * total_steps)
    eval_seeds = list(range(cfg["eval_seed_start"], cfg["eval_seed_start"] + cfg["eval_episodes"]))
    final_seeds = list(range(cfg["final_eval_seed_start"],
                             cfg["final_eval_seed_start"] + cfg["final_eval_episodes"]))
    threshold = cfg["success_threshold"]

    # ---------------- Logs ----------------
    eval_f = open(out_dir / "eval.csv", "w", newline="")
    eval_w = csv.writer(eval_f)
    eval_w.writerow(["step", "eval_mean", "eval_std", "epsilon", "train_time_s",
                     "loss", "q_mean", "q0_estimate", "q0_actual"])
    train_f = open(out_dir / "train_episodes.csv", "w", newline="")
    train_w = csv.writer(train_f)
    train_w.writerow(["step", "episode_return", "episode_length"])

    ep_return, ep_len = 0.0, 0
    losses, qs = [], []
    best_eval, steps_to_threshold = -np.inf, None
    train_time = 0.0                      # solo tiempo de entrenamiento (sin evaluaciones)
    wall_start = time.perf_counter()
    t0 = time.perf_counter()

    for step in range(1, total_steps + 1):
        # 1) Actuar con epsilon-greedy
        epsilon = linear_epsilon(step, cfg["eps_start"], cfg["eps_end"], decay_steps)
        action = agent.act(obs, epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)

        # 2) Guardar transicion. Se guarda `terminated` (no `truncated`): si el episodio
        #    se corta por limite de tiempo, el estado siguiente SI tiene valor futuro.
        agent.buffer.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        ep_return += reward
        ep_len += 1

        if terminated or truncated:
            train_w.writerow([step, ep_return, ep_len])
            obs, _ = env.reset()
            ep_return, ep_len = 0.0, 0

        # 3) Aprender (despues de llenar un poco el buffer)
        if step > cfg["learning_starts"] and step % cfg["train_freq"] == 0:
            loss, q = agent.update()
            losses.append(loss)
            qs.append(q)

        # 4) Sincronizar red objetivo
        if step % cfg["target_update_freq"] == 0:
            agent.sync_target()

        # 5) Evaluacion periodica (greedy, epsilon = 0) -- el cronometro se pausa
        if step % cfg["eval_every"] == 0:
            train_time += time.perf_counter() - t0
            ev = evaluate(agent, eval_env, eval_seeds, cfg["gamma"])
            m, s = ev["returns"].mean(), ev["returns"].std()
            eval_w.writerow([step, m, s, epsilon, round(train_time, 2),
                             np.mean(losses) if losses else "", np.mean(qs) if qs else "",
                             ev["q0_estimates"].mean(), ev["q0_actual"].mean()])
            eval_f.flush()
            train_f.flush()
            losses, qs = [], []
            if m > best_eval:
                best_eval = m
                agent.save(out_dir / "model_best.pt")
            if steps_to_threshold is None and m >= threshold:
                steps_to_threshold = step
            print(f"[{cfg['exp_name']} s{seed}] step {step:>7d} | eval {m:8.2f} ± {s:6.2f} "
                  f"| eps {epsilon:.3f} | train {train_time:7.1f}s", flush=True)
            t0 = time.perf_counter()

    train_time += time.perf_counter() - t0
    eval_f.close()
    train_f.close()
    agent.save(out_dir / "model_final.pt")

    # ---------------- Evaluacion final: 100 episodios, semillas 20000-20099, epsilon = 0 -------
    final = evaluate(agent, eval_env, final_seeds, cfg["gamma"])
    best_agent = DQNAgent(obs_dim, n_actions, cfg, np.random.default_rng(0))
    best_agent.load(out_dir / "model_best.pt")
    best = evaluate(best_agent, eval_env, final_seeds, cfg["gamma"])

    summary = {
        "exp_name": cfg["exp_name"],
        "env_id": cfg["env_id"],
        "seed": seed,
        "final_mean": float(final["returns"].mean()),
        "final_std": float(final["returns"].std()),
        "final_min": float(final["returns"].min()),
        "final_max": float(final["returns"].max()),
        "solved": bool(final["returns"].mean() >= threshold),
        "best_ckpt_mean": float(best["returns"].mean()),
        "best_ckpt_std": float(best["returns"].std()),
        "steps_to_threshold": steps_to_threshold,     # None = "not reached" (en eval de 10 ep.)
        "train_time_s": round(train_time, 2),
        "wall_time_s": round(time.perf_counter() - wall_start, 2),
        "final_q0_estimate": float(final["q0_estimates"].mean()),
        "final_q0_actual": float(final["q0_actual"].mean()),
        "final_returns": final["returns"].tolist(),
    }
    save_json(summary, out_dir / "final.json")
    print(f"[{cfg['exp_name']} s{seed}] FINAL ({len(final_seeds)} ep) {summary['final_mean']:.2f} ± "
          f"{summary['final_std']:.2f} | solved={summary['solved']} | "
          f"steps_to_threshold={steps_to_threshold} | train {train_time:.1f}s")


if __name__ == "__main__":
    main()
