"""Utilidades compartidas: semillas, schedule de epsilon y evaluacion greedy."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int):
    """Fija las semillas de Python, NumPy y PyTorch (reproducibilidad)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """Decaimiento lineal de eps_start a eps_end en `decay_steps` pasos; luego constante."""
    frac = min(1.0, step / max(1, decay_steps))
    return eps_start + frac * (eps_end - eps_start)


def evaluate(agent, env, seeds, gamma: float):
    """
    Ejecuta un episodio greedy (epsilon = 0) por cada semilla de `seeds`.

    Devuelve un dict con:
      returns      : retorno (no descontado) de cada episodio  -> metrica principal
      lengths      : longitud de cada episodio
      q0_estimates : max_a Q(s0, a) que el agente predice en el estado inicial
      q0_actual    : retorno DESCONTADO real obtenido desde s0
    Las dos ultimas permiten medir la sobreestimacion de Q (clave para Double DQN).
    """
    returns, lengths, q0_est, q0_real = [], [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        q0_est.append(agent.max_q(obs))
        done, ep_ret, disc_ret, discount, t = False, 0.0, 0.0, 1.0, 0
        while not done:
            action = agent.greedy_action(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += r
            disc_ret += discount * r
            discount *= gamma
            t += 1
            done = terminated or truncated
        returns.append(ep_ret)
        lengths.append(t)
        q0_real.append(disc_ret)
    return {
        "returns": np.array(returns),
        "lengths": np.array(lengths),
        "q0_estimates": np.array(q0_est),
        "q0_actual": np.array(q0_real),
    }


def load_config(path, overrides: dict | None = None) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
