"""
Entrena Q-learning tabular en CartPole-v1 con el MISMO protocolo que DQN y guarda los
resultados en el mismo formato (eval.csv + final.json), para comparar en las mismas graficas.

El bucle de entrenamiento reproduce `entrenar_presupuesto` del notebook de Q-learning
(Ajuste A, 10 bins, alpha 0.1, epsilon 1.0 -> 0.05 en 150k pasos). Lo unico agregado es
la evaluacion cada 10.000 pasos (10 episodios greedy, semillas 10000-10009), que usa su
propio generador y su propio entorno, asi que NO altera la trayectoria de entrenamiento.

Uso:
  python -m experiments.train_qlearning --seed 42
  python -m experiments.train_qlearning --seed 42 --seed 123 --seed 456
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.tabular import NUM_ACTIONS, discretizar_estado, epsilon_greedy, update_q  # noqa: E402
from common.utils import save_json  # noqa: E402

CFG = {
    "exp_name": "cartpole_qlearning",
    "env_id": "CartPole-v1",
    "total_steps": 300_000,
    "n_bins": 10,
    "alpha": 0.1,
    "gamma": 0.99,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_steps": 150_000,
    "eval_every": 10_000,
    "eval_episodes": 10,
    "eval_seed_start": 10_000,
    "final_eval_episodes": 100,
    "final_eval_seed_start": 20_000,
    "success_threshold": 475,
}


def evaluate_q_learning(Q, n_bins, seed, episodes):
    """Igual que en el notebook: epsilon = 0, sin modificar Q, semillas seed..seed+episodes-1."""
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    recompensas = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            estado = discretizar_estado(obs, n_bins)
            total, terminado = 0.0, False
            while not terminado:
                accion = epsilon_greedy(Q, estado, 0.0, rng)
                obs, r, terminated, truncated, _ = env.step(accion)
                estado = discretizar_estado(obs, n_bins)
                total += r
                terminado = terminated or truncated
            recompensas.append(total)
    finally:
        env.close()
    return np.array(recompensas)


def train(seed, results_dir):
    c = CFG
    out = Path(results_dir) / c["exp_name"] / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    save_json({**c, "seed": seed}, out / "config.json")

    rng = np.random.default_rng(seed)
    env = gym.make(c["env_id"])
    n = c["n_bins"]
    Q = np.zeros((n, n, n, n, NUM_ACTIONS))

    ef = open(out / "eval.csv", "w", newline="")
    ew = csv.writer(ef)
    ew.writerow(["step", "eval_mean", "eval_std", "epsilon", "train_time_s"])
    tf = open(out / "train_episodes.csv", "w", newline="")
    tw = csv.writer(tf)
    tw.writerow(["step", "episode_return", "episode_length"])

    pasos, episodio, train_time, steps_to_threshold = 0, 0, 0.0, None
    t0 = time.perf_counter()
    wall0 = t0
    while pasos < c["total_steps"]:
        obs, _ = env.reset(seed=seed + episodio)          # igual que el notebook
        estado = discretizar_estado(obs, n)
        terminado, ep_ret, ep_len = False, 0.0, 0
        while not terminado and pasos < c["total_steps"]:
            progreso = min(pasos / c["eps_decay_steps"], 1.0)
            epsilon = c["eps_start"] - (c["eps_start"] - c["eps_end"]) * progreso
            accion = epsilon_greedy(Q, estado, epsilon, rng)
            obs, r, terminated, truncated, _ = env.step(accion)
            sig = discretizar_estado(obs, n)
            update_q(Q, estado, accion, r, sig, terminated, c["alpha"], c["gamma"])
            estado = sig
            terminado = terminated or truncated
            pasos += 1
            ep_ret += r
            ep_len += 1

            if pasos % c["eval_every"] == 0:
                train_time += time.perf_counter() - t0
                ev = evaluate_q_learning(Q, n, c["eval_seed_start"], c["eval_episodes"])
                ew.writerow([pasos, ev.mean(), ev.std(), epsilon, round(train_time, 2)])
                ef.flush()
                if steps_to_threshold is None and ev.mean() >= c["success_threshold"]:
                    steps_to_threshold = pasos
                print(f"[qlearning s{seed}] step {pasos:>7d} | eval {ev.mean():7.2f} ± {ev.std():6.2f} "
                      f"| eps {epsilon:.3f}", flush=True)
                t0 = time.perf_counter()
        if terminado:
            tw.writerow([pasos, ep_ret, ep_len])
        episodio += 1
    train_time += time.perf_counter() - t0
    env.close()
    ef.close()
    tf.close()
    np.save(out / "q_table.npy", Q)

    final = evaluate_q_learning(Q, n, c["final_eval_seed_start"], c["final_eval_episodes"])
    summary = {
        "exp_name": c["exp_name"], "env_id": c["env_id"], "seed": seed,
        "final_mean": float(final.mean()), "final_std": float(final.std()),
        "final_min": float(final.min()), "final_max": float(final.max()),
        "solved": bool(final.mean() >= c["success_threshold"]),
        "best_ckpt_mean": float("nan"), "best_ckpt_std": float("nan"),   # no se guardan checkpoints
        "steps_to_threshold": steps_to_threshold,
        "train_time_s": round(train_time, 2),
        "wall_time_s": round(time.perf_counter() - wall0, 2),
        "final_q0_estimate": float("nan"), "final_q0_actual": float("nan"),
        "final_returns": final.tolist(),
    }
    save_json(summary, out / "final.json")
    print(f"[qlearning s{seed}] FINAL ({len(final)} ep) {final.mean():.2f} ± {final.std():.2f} "
          f"| train {train_time:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, action="append", required=True)
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()
    for s in args.seed:
        train(s, args.results_dir)


if __name__ == "__main__":
    main()
