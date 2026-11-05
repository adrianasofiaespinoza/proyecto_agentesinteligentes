"""
Q-learning tabular para CartPole-v1.

Transcripcion fiel de las funciones del notebook de Q-learning del equipo
(Q_Learning_CartPole.ipynb, configuracion final "Ajuste A"). No se cambio la logica:
mismas funciones de discretizacion, epsilon-greedy con desempate aleatorio,
actualizacion Q y schedule de epsilon.
"""
import numpy as np

NUM_ACTIONS = 2

# Limites de la configuracion seleccionada (Ajuste A):
# posicion, velocidad, angulo, velocidad angular
MINIMOS = np.array([-2.4, -2.0, -0.2095, -2.5])
MAXIMOS = np.array([2.4, 2.0, 0.2095, 2.5])


def discretizar_estado(estado, n_bins):
    """Convierte las 4 observaciones continuas en 4 indices discretos."""
    indices = []
    for i in range(4):
        limites = np.linspace(MINIMOS[i], MAXIMOS[i], n_bins + 1)
        indice = np.digitize(estado[i], limites[1:-1])
        indices.append(int(indice))
    return tuple(indices)


def epsilon_greedy(Q, state, epsilon, rng):
    # Explorar: accion aleatoria
    if rng.random() < epsilon:
        return int(rng.integers(NUM_ACTIONS))
    # Explotar, resolviendo empates al azar
    q_values = Q[state]
    max_q = np.max(q_values)
    mejores_acciones = np.flatnonzero(np.isclose(q_values, max_q))
    return int(rng.choice(mejores_acciones))


def update_q(Q, state, action, reward, next_state, terminated, alpha, gamma):
    q_actual = Q[state + (action,)]
    if terminated:                       # fallo real: sin valor futuro
        objetivo = reward
    else:
        objetivo = reward + gamma * np.max(Q[next_state])
    error_td = objetivo - q_actual
    Q[state + (action,)] += alpha * error_td
    return error_td
