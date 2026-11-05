"""
Agente DQN implementado desde cero (PyTorch).

Componentes:
  - QNetwork:      MLP que estima Q(s, a) para todas las acciones a la vez.
                   Opcional: arquitectura Dueling (V(s) + A(s,a)).
  - ReplayBuffer:  memoria circular de transiciones (s, a, r, s', terminated).
  - DQNAgent:      politica epsilon-greedy, actualizacion por TD con red objetivo.
                   Opcional: Double DQN, sin red objetivo, sin replay (ablaciones).

Variantes que se activan con flags del config:
  double=True             -> Double DQN (van Hasselt et al., 2016)
  dueling=True            -> Dueling DQN (Wang et al., 2016)
  use_target_network=False-> ablacion: el objetivo se calcula con la misma red online
  use_replay=False        -> ablacion: se entrena con las ultimas `batch_size`
                             transiciones consecutivas (sin muestreo aleatorio)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Red Q
# --------------------------------------------------------------------------- #
class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden=(128, 128), dueling: bool = False):
        super().__init__()
        self.dueling = dueling

        # Cuerpo compartido: capas lineales + ReLU
        layers, in_dim = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.body = nn.Sequential(*layers)

        if dueling:
            # Dos cabezas: valor del estado V(s) y ventaja A(s, a)
            self.value_head = nn.Linear(in_dim, 1)
            self.adv_head = nn.Linear(in_dim, n_actions)
        else:
            self.q_head = nn.Linear(in_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.body(x)
        if self.dueling:
            v = self.value_head(z)                       # (B, 1)
            a = self.adv_head(z)                         # (B, nA)
            # Restamos la media de A para que V y A sean identificables
            return v + a - a.mean(dim=1, keepdim=True)
        return self.q_head(z)                            # (B, nA)


# --------------------------------------------------------------------------- #
# Replay buffer
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Memoria circular en arrays de NumPy (rapida y con poca RAM)."""

    def __init__(self, capacity: int, obs_dim: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)   # 1.0 solo si terminated
        self.pos = 0       # siguiente posicion donde escribir
        self.size = 0      # cuantas transiciones validas hay

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.pos
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(terminated)
        self.pos = (self.pos + 1) % self.capacity          # sobrescribe lo mas antiguo
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)   # muestreo uniforme
        return self._gather(idx)

    def latest(self, batch_size: int):
        """Ultimas `batch_size` transiciones en orden (para la ablacion sin replay)."""
        n = min(batch_size, self.size)
        idx = (self.pos - n + np.arange(n)) % self.capacity
        return self._gather(idx)

    def _gather(self, idx):
        return (
            torch.as_tensor(self.obs[idx]),
            torch.as_tensor(self.actions[idx]),
            torch.as_tensor(self.rewards[idx]),
            torch.as_tensor(self.next_obs[idx]),
            torch.as_tensor(self.dones[idx]),
        )


# --------------------------------------------------------------------------- #
# Agente
# --------------------------------------------------------------------------- #
class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.n_actions = n_actions
        self.rng = rng
        self.gamma = cfg["gamma"]
        self.batch_size = cfg["batch_size"]
        self.double = cfg.get("double", False)
        self.use_target = cfg.get("use_target_network", True)
        self.use_replay = cfg.get("use_replay", True)

        self.q_net = QNetwork(obs_dim, n_actions, tuple(cfg["hidden"]), cfg.get("dueling", False))
        if self.use_target:
            # Copia congelada de la red; se sincroniza cada `target_update_freq` pasos
            self.target_net = copy.deepcopy(self.q_net)
            self.target_net.requires_grad_(False)
        else:
            # Ablacion: la "red objetivo" ES la red online (objetivo movil)
            self.target_net = self.q_net

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg["lr"])

        # Sin replay, la memoria solo guarda las ultimas `batch_size` transiciones
        capacity = cfg["buffer_size"] if self.use_replay else self.batch_size
        self.buffer = ReplayBuffer(capacity, obs_dim, rng)

    # ----------------------------- politica -------------------------------- #
    @torch.no_grad()
    def greedy_action(self, obs) -> int:
        q = self.q_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def act(self, obs, epsilon: float) -> int:
        """epsilon-greedy: con prob. epsilon accion aleatoria, si no la de mayor Q."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(obs)

    @torch.no_grad()
    def max_q(self, obs) -> float:
        """max_a Q(s, a): estimacion del agente del retorno desde s (para medir sobreestimacion)."""
        q = self.q_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return float(q.max().item())

    # ---------------------------- aprendizaje ------------------------------ #
    def update(self) -> tuple[float, float]:
        """Un paso de descenso de gradiente sobre un minibatch. Devuelve (loss, Q medio)."""
        if self.use_replay:
            obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)
        else:
            obs, actions, rewards, next_obs, dones = self.buffer.latest(self.batch_size)

        # Q(s, a) de las acciones realmente tomadas
        q_values = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double:
                # Double DQN: la red ONLINE elige a* = argmax_a Q_online(s', a)
                #             la red OBJETIVO la evalua: Q_target(s', a*)
                next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs).gather(1, next_actions).squeeze(1)
            else:
                # DQN vanilla: max_a Q_target(s', a) (elige y evalua la misma red)
                next_q = self.target_net(next_obs).max(dim=1).values
            # Objetivo TD. (1 - done) corta el bootstrap SOLO en estados terminales reales
            targets = rewards + self.gamma * (1.0 - dones) * next_q

        # Huber loss: menos sensible a errores TD grandes que MSE (como en Mnih et al., 2015)
        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.cfg.get("max_grad_norm", 10.0))
        self.optimizer.step()
        return float(loss.item()), float(q_values.mean().item())

    def sync_target(self):
        """Copia los pesos de la red online a la red objetivo (hard update)."""
        if self.use_target:
            self.target_net.load_state_dict(self.q_net.state_dict())

    # ---------------------------- guardado --------------------------------- #
    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location="cpu"))
        self.sync_target()
