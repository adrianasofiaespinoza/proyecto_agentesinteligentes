# Q-learning tabular y DQN en Gymnasium

Proyecto de aprendizaje por refuerzo del curso *Agentes Inteligentes*: compara Q-learning tabular
y DQN en CartPole-v1, estudia DQN en LunarLander-v3 y evalúa si Double DQN corrige la
sobreestimación de los valores Q. DQN está implementado desde cero en PyTorch.

**Autoras:** Tamya Chilcañan y Adriana Espinoza.

| Tier | Contenido | Entorno |
|---|---|---|
| 1 | Q-learning tabular vs. DQN | CartPole-v1 |
| 2 | DQN: línea base, ablaciones (sin red objetivo, sin replay buffer), learning rate (1e-4 / 5e-4 / 2e-3) y frecuencia de la red objetivo (100 / 1000 / 5000 pasos) | LunarLander-v3 |
| 3 | Double DQN vs. DQN | LunarLander-v3 y CartPole-v1 |

El artículo está en [report/paper.pdf](report/paper.pdf).

## Estructura

```text
project/
├── agents/          Agentes: dqn.py (DQN y variantes) y tabular.py (Q-learning)
├── common/          Utilidades compartidas: semillas, epsilon lineal, evaluación, configuración
├── experiments/     Scripts de entrenamiento, resumen, gráficas y GIFs
│   └── configs/     Hiperparámetros de cada experimento (.json)
├── notebooks/       Notebooks explicativos de Tier 1 y Tier 3
├── results/         Resultados por experimento, figures/, gifs/ y *_summary.csv
├── report/          Artículo: main.tex, figures/ y paper.pdf
├── requirements.txt Dependencias con versiones exactas
└── README.md
```

Cada corrida se guarda en `results/<experimento>/seed_<semilla>/`:

| Archivo | Contenido |
|---|---|
| `config.json` | Configuración exacta usada |
| `eval.csv` | Una fila por evaluación periódica: `step, eval_mean, eval_std, epsilon, train_time_s, loss, q_mean, q0_estimate, q0_actual` |
| `train_episodes.csv` | Retorno de cada episodio de entrenamiento (con exploración) |
| `final.json` | Evaluación final de 100 episodios, pasos hasta el umbral, tiempos y mejor checkpoint |
| `model_final.pt`, `model_best.pt` | Pesos de DQN (última iteración y mejor evaluación intermedia) |
| `q_table.npy` | Tabla Q (solo Q-learning) |

## Instalación

Versiones probadas con Python 3.14.8. Con Python 3.14, Box2D se compila al instalar, así que antes
hay que instalar `swig` y `base-devel` (en Arch Linux):

```bash
sudo pacman -S --needed swig base-devel
```

Después:

```bash
git clone https://github.com/adrianasofiaespinoza/proyecto_agentesinteligentes.git
cd proyecto_agentesinteligentes
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Todos los comandos se ejecutan **desde la raíz del proyecto**.

## Protocolo común

| Elemento | Valor |
|---|---|
| Semillas de entrenamiento | 42, 123, 456 |
| Evaluación periódica | Cada 10.000 pasos, 10 episodios greedy (semillas 10000–10009) |
| Evaluación final | 100 episodios con ε = 0 (semillas 20000–20099), sobre el modelo final |
| Presupuesto | CartPole-v1: 300.000 pasos · LunarLander-v3: 500.000 pasos |
| Umbral de éxito | CartPole-v1: 475 · LunarLander-v3: 200 |

La métrica principal es la del **modelo final**. `best_ckpt_mean` se reporta aparte porque elegir
un checkpoint ya es una decisión de diseño.

## Reproducir los experimentos

`run_seeds` entrena las tres semillas en paralelo. **Estos comandos sobrescriben la carpeta del
experimento correspondiente en `results/`.**

### Tier 1: CartPole-v1

```bash
# Q-learning tabular
python -m experiments.train_qlearning --seed 42 --seed 123 --seed 456

# DQN
python -m experiments.run_seeds --config experiments/configs/cartpole_dqn.json
```

### Tier 2: DQN en LunarLander-v3

Todo el Tier 2 y el Double DQN del Tier 3 se lanzan, uno tras otro, con:

```bash
bash experiments/run_lunarlander_all.sh
```

O por separado:

```bash
# Línea base (también es el DQN de referencia del Tier 3)
python -m experiments.run_seeds --config experiments/configs/lunarlander_dqn.json

# Ablaciones
python -m experiments.run_seeds --config experiments/configs/lunarlander_dqn.json --set use_target_network=false --set exp_name=ll_ablation_no_target
python -m experiments.run_seeds --config experiments/configs/lunarlander_dqn.json --set use_replay=false --set exp_name=ll_ablation_no_replay

# Learning rate (la línea base usa 5e-4)
python -m experiments.run_seeds --config experiments/configs/lunarlander_dqn.json --set lr=0.0001 --set exp_name=ll_lr_1e-4
python -m experiments.run_seeds --config experiments/configs/lunarlander_dqn.json --set lr=0.002 --set exp_name=ll_lr_2e-3

# Frecuencia de la red objetivo (la línea base usa 1000)
python -m experiments.run_seeds --config experiments/configs/lunarlander_dqn.json --set target_update_freq=100 --set exp_name=ll_target_100
python -m experiments.run_seeds --config experiments/configs/lunarlander_dqn.json --set target_update_freq=5000 --set exp_name=ll_target_5000
```

### Tier 3: Double DQN

```bash
# LunarLander-v3
python -m experiments.run_seeds --config experiments/configs/lunarlander_ddqn.json

# CartPole-v1
python -m experiments.run_seeds --config experiments/configs/cartpole_dqn.json --set double=true --set exp_name=cartpole_ddqn
```

### Tablas de resumen

```bash
python -m experiments.summarize results/cartpole_qlearning results/cartpole_dqn results/cartpole_ddqn --out results/tier1_summary.csv
python -m experiments.summarize results/lunarlander_dqn results/ll_ablation_no_target results/ll_ablation_no_replay results/ll_lr_1e-4 results/ll_lr_2e-3 results/ll_target_100 results/ll_target_5000 --out results/tier2_summary.csv
python -m experiments.summarize results/lunarlander_dqn results/lunarlander_ddqn --out results/tier3_summary.csv
```

## Figuras del artículo

Cada comando escribe en `results/figures/`. Las imágenes que usa `report/main.tex` están en
`report/figures/`. Allí, `tier1_cartpole.png` y `tier3_cartpole_eval.png` son versiones anteriores
de la gráfica: muestran las mismas curvas, pero con la banda sin recortar a 500 y con la etiqueta
del umbral y la leyenda en otra posición.

**Fig. 2. Tier 1: Q-learning vs. DQN**

```bash
python -m experiments.plot_results --exps results/cartpole_qlearning results/cartpole_dqn --labels "Q-learning" "DQN" --threshold 475 --clip-max 500 --out results/figures/tier1_cartpole.png
```

**Fig. 3a. Ablaciones**

```bash
python -m experiments.plot_results --exps results/lunarlander_dqn results/ll_ablation_no_target results/ll_ablation_no_replay --labels "DQN" "Sin red objetivo" "Sin replay" --threshold 200 --out results/figures/tier2_ablation.png
```

**Fig. 3b. Frecuencia de la red objetivo**

```bash
python -m experiments.plot_results --exps results/ll_target_100 results/lunarlander_dqn results/ll_target_5000 --labels "target 100" "target 1000" "target 5000" --threshold 200 --out results/figures/tier2_target.png
```

**Fig. 3c. Learning rate**

```bash
python -m experiments.plot_results --exps results/ll_lr_1e-4 results/lunarlander_dqn results/ll_lr_2e-3 --labels "lr 1e-4" "lr 5e-4" "lr 2e-3" --threshold 200 --out results/figures/tier2_lr.png
```

**Fig. 4a. CartPole: retorno de evaluación**

```bash
python -m experiments.plot_results --exps results/cartpole_dqn results/cartpole_ddqn --labels "DQN" "Double DQN" --threshold 475 --clip-max 500 --out results/figures/tier3_cartpole_eval.png
```

**Fig. 4b. CartPole: Q estimado vs. real**

```bash
python -m experiments.plot_results --exps results/cartpole_dqn results/cartpole_ddqn --labels "DQN" "Double DQN" --mode overestimation --out results/figures/tier3_cartpole_overestimation.png
```

**Fig. 4c. LunarLander: retorno de evaluación**

```bash
python -m experiments.plot_results --exps results/lunarlander_dqn results/lunarlander_ddqn --labels "DQN" "Double DQN" --threshold 200 --out results/figures/tier3_eval.png
```

**Fig. 6a. Ablaciones: Q estimado vs. real**

```bash
python -m experiments.plot_results --exps results/lunarlander_dqn results/ll_ablation_no_target results/ll_ablation_no_replay --labels "DQN" "Sin red objetivo" "Sin replay" --mode overestimation --out results/figures/tier2_ablation_overestimation.png
```

**Fig. 6b. Double DQN: Q estimado vs. real**

```bash
python -m experiments.plot_results --exps results/lunarlander_dqn results/lunarlander_ddqn --labels "DQN" "Double DQN" --mode overestimation --out results/figures/tier3_overestimation.png
```

**Fig. 6c. Retorno de entrenamiento**

```bash
python -m experiments.plot_results --exps results/lunarlander_dqn results/lunarlander_ddqn --labels "DQN" "Double DQN" --mode train --threshold 200 --out results/figures/tier3_train.png
```

**Fig. 5. GIFs de los que salen los fotogramas**

```bash
python -m experiments.record_gif --run results/cartpole_dqn/seed_123
python -m experiments.record_gif --run results/lunarlander_ddqn/seed_42
python -m experiments.record_gif --run results/ll_lr_1e-4/seed_42
```

**Figura adicional (no usada en el artículo):** las tres curvas de CartPole juntas.

```bash
python -m experiments.plot_results --exps results/cartpole_qlearning results/cartpole_dqn results/cartpole_ddqn --labels "Q-learning" "DQN" "Double DQN" --threshold 475 --clip-max 500 --out results/figures/tier1_cartpole_all.png
```

La Fig. 1 es un diagrama del artículo y no se genera con estos scripts. Tampoco hay un script para
montar los fotogramas de la Fig. 5 a partir de los GIFs: la imagen final es
`report/figures/qualitative_frames.png`.

## Notebooks

Abrir desde `notebooks/`. Los de Tier 1 (DQN) y Tier 3 leen los resultados de `results/` y no
reentrenan salvo que se active `DEMO = True`.

| Notebook | Qué hace |
|---|---|
| `FINAL_Q_Learning_CartPole.ipynb` | Tier 1, versión final de Q-learning: discretización, entrenamiento, experimentos, presupuesto de 300.000 pasos, validación y configuración elegida. Entrena dentro del notebook. |
| `Q_Learning_CartPole.ipynb` | Tier 1, desarrollo completo de Q-learning con todos los experimentos (epsilon, alpha, límites de discretización y otros). Entrena dentro del notebook. |
| `TIER1_DQN_CartPole.ipynb` | Tier 1: explica DQN paso a paso, selección de hiperparámetros, resultados por semilla y comparación con Q-learning. |
| `TIER3_DoubleDQN.ipynb` | Tier 3: sesgo de maximización, objetivo de Double DQN y resultados en CartPole-v1 y LunarLander-v3. |

## Resultados

Evaluación final: 100 episodios greedy por semilla, sobre el modelo final. La media ± std se
calcula entre semillas. Los datos vienen de `results/*_summary.csv` y de los `final.json`.

### Tier 1: CartPole-v1

| Método | Media final ± std | Semillas 42 / 123 / 456 | Resueltas | Pasos al umbral (42 / 123 / 456) | Tiempo (min) |
|---|---|---|---|---|---|
| Q-learning | 493,88 ± 5,14 | 499,0 / 486,8 / 495,8 | 3/3 | 190k / 140k / 150k | 0,5 |
| DQN | 369,51 ± 184,54 | 108,5 / 500,0 / 500,0 | 2/3 | 190k / 200k / 170k | 7,4 |

### Tier 2: DQN en LunarLander-v3

| Variante | Media final ± std | Semillas 42 / 123 / 456 | Resueltas | Pasos al umbral (42 / 123 / 456) | Q(s₀) estimado / real |
|---|---|---|---|---|---|
| Línea base (lr 5e-4, red objetivo cada 1000) | 250,20 ± 9,27 | 248,3 / 262,4 / 239,9 | 3/3 | 190k / 210k / 270k | 71,3 / 71,3 |
| Sin red objetivo | 214,46 ± 31,09 | 235,2 / 237,6 / 170,5 | 2/3 | 110k / 70k / 90k | 78,5 / 66,1 |
| Sin replay buffer | 98,12 ± 95,13 | −34,8 / 146,6 / 182,6 | 0/3 | 460k / no alcanzado / 350k | 82,0 / 33,9 |
| lr 1e-4 | 153,91 ± 108,26 | 1,4 / 241,5 / 218,9 | 2/3 | no alcanzado / 350k / 330k | 60,6 / 57,7 |
| lr 2e-3 | 204,68 ± 35,81 | 236,3 / 154,6 / 223,2 | 2/3 | 70k / 160k / 180k | 80,5 / 70,2 |
| Red objetivo cada 100 | 193,86 ± 85,65 | 72,7 / 254,5 / 254,3 | 2/3 | 360k / 250k / 90k | 70,2 / 68,7 |
| Red objetivo cada 5000 | 223,66 ± 25,17 | 190,4 / 229,2 / 251,4 | 2/3 | 200k / 230k / 200k | 61,0 / 70,8 |

### Tier 3: Double DQN vs. DQN

| Entorno | Método | Media final ± std | Semillas 42 / 123 / 456 | Resueltas | Pasos al umbral (42 / 123 / 456) | Q(s₀) estimado / real |
|---|---|---|---|---|---|---|
| LunarLander-v3 | DQN | 250,20 ± 9,27 | 248,3 / 262,4 / 239,9 | 3/3 | 190k / 210k / 270k | 71,3 / 71,3 |
| LunarLander-v3 | Double DQN | 240,84 ± 28,86 | 239,4 / 206,2 / 276,9 | 3/3 | 90k / 260k / 110k | 71,3 / 69,4 |
| CartPole-v1 | DQN | 369,51 ± 184,54 | 108,5 / 500,0 / 500,0 | 2/3 | 190k / 200k / 170k | 109,5 / 87,8 |
| CartPole-v1 | Double DQN | 473,75 ± 37,12 | 421,2 / 500,0 / 500,0 | 2/3 | 130k / 120k / 130k | 106,0 / 99,0 |

## Mapa del código

| Concepto | Dónde |
|---|---|
| Red Q | `agents/dqn.py` → `QNetwork` |
| Replay buffer circular | `agents/dqn.py` → `ReplayBuffer` |
| ε-greedy | `DQNAgent.act` y `common/utils.py` → `linear_epsilon` |
| Objetivo TD de DQN y Double DQN | `DQNAgent.update` (bloque `if self.double`) |
| Red objetivo y ablación sin ella | `DQNAgent.__init__` y `sync_target` |
| Ablación sin replay | `ReplayBuffer.latest` |
| Evaluación greedy y sobreestimación | `common/utils.py` → `evaluate` |
| Q-learning | `agents/tabular.py` y `experiments/train_qlearning.py` |

Cualquier hiperparámetro se puede cambiar sin editar el JSON, por ejemplo:
`--set lr=0.001 --set exp_name=nuevo_nombre`.

## Hardware

Todas las corridas usaron solo CPU, en un portátil con 16 hilos y 16 GB de RAM. Cada proceso usó un
hilo.

## Uso de IA generativa

T. Chilcañan usó ChatGPT (OpenAI) para apoyar la implementación, depuración y análisis de
Q-learning; A. Espinoza usó Claude (Anthropic) para implementar DQN y sus variantes, los scripts de
experimentos y gráficas, reproducir Q-learning con el protocolo común, diagnosticar resultados y
redactar el paper.
