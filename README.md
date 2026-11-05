# Tabular Q-learning and DQN on Gymnasium

Reinforcement learning project for the *Intelligent Agents* course. It compares tabular Q-learning
and DQN on CartPole-v1, studies DQN on LunarLander-v3, and evaluates whether Double DQN corrects the
overestimation of Q-values. DQN is implemented from scratch in PyTorch.

**Authors:** Tamya Chilcañan and Adriana Espinoza.

| Tier | Content | Environment |
|---|---|---|
| 1 | Tabular Q-learning vs. DQN | CartPole-v1 |
| 2 | DQN: baseline, ablations (no target network, no replay buffer), learning rate (1e-4 / 5e-4 / 2e-3) and target network update period (100 / 1000 / 5000 steps) | LunarLander-v3 |
| 3 | Double DQN vs. DQN | LunarLander-v3 and CartPole-v1 |

The paper is at [report/paper.pdf](report/paper.pdf).

## Structure

```text
project/
├── agents/          Agents: dqn.py (DQN and variants) and tabular.py (Q-learning)
├── common/          Shared utilities: seeding, linear epsilon, evaluation, configuration
├── experiments/     Training, summary, plotting and GIF scripts
│   └── configs/     Hyperparameters for each experiment (.json)
├── notebooks/       Explanatory notebooks for Tier 1 and Tier 3
├── results/         Results per experiment, figures/, gifs/ and *_summary.csv
├── report/          Paper: main.tex, figures/ and paper.pdf
├── requirements.txt Dependencies with exact versions
└── README.md
```

Each run is saved in `results/<experiment>/seed_<seed>/`:

| File | Content |
|---|---|
| `config.json` | Exact configuration used |
| `eval.csv` | One row per periodic evaluation: `step, eval_mean, eval_std, epsilon, train_time_s, loss, q_mean, q0_estimate, q0_actual` |
| `train_episodes.csv` | Return of each training episode (with exploration) |
| `final.json` | Final evaluation over 100 episodes, steps to threshold, times and best checkpoint |
| `model_final.pt`, `model_best.pt` | DQN weights (last iteration and best intermediate evaluation) |
| `q_table.npy` | Q-table (Q-learning only) |

## Installation

Versions tested with Python 3.14.8. On Python 3.14, Box2D is compiled during installation, so you
must first install `swig` and `base-devel` (on Arch Linux):

```bash
sudo pacman -S --needed swig base-devel
```

Then:

```bash
git clone https://github.com/adrianasofiaespinoza/proyecto_agentesinteligentes.git
cd proyecto_agentesinteligentes
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run all commands **from the project root**.

## Common protocol

| Item | Value |
|---|---|
| Training seeds | 42, 123, 456 |
| Periodic evaluation | Every 10,000 steps, 10 greedy episodes (seeds 10000–10009) |
| Final evaluation | 100 episodes with ε = 0 (seeds 20000–20099), on the final model |
| Budget | CartPole-v1: 300,000 steps · LunarLander-v3: 500,000 steps |
| Success threshold | CartPole-v1: 475 · LunarLander-v3: 200 |

The main metric is that of the **final model**. `best_ckpt_mean` is reported separately because
choosing a checkpoint is already a design decision.

## Reproducing the experiments

`run_seeds` trains the three seeds in parallel. **These commands overwrite the corresponding
experiment folder in `results/`.**

### Tier 1: CartPole-v1

```bash
# Tabular Q-learning
python -m experiments.train_qlearning --seed 42 --seed 123 --seed 456

# DQN
python -m experiments.run_seeds --config experiments/configs/cartpole_dqn.json
```

### Tier 2: DQN on LunarLander-v3

All of Tier 2 and the Double DQN run of Tier 3 are launched, one after another, with:

```bash
bash experiments/run_lunarlander_all.sh
```

Or separately:

```bash
# Baseline (also the reference DQN for Tier 3)
python -m experiments.run_seeds --config experiments/configs/lunarlander_dqn.json

# Ablations
python -m experiments.run_seeds --config experiments/configs/lunarlander_dqn.json --set use_target_network=false --set exp_name=ll_ablation_no_target
python -m experiments.run_seeds --config experiments/configs/lunarlander_dqn.json --set use_replay=false --set exp_name=ll_ablation_no_replay

# Learning rate (the baseline uses 5e-4)
python -m experiments.run_seeds --config experiments/configs/lunarlander_dqn.json --set lr=0.0001 --set exp_name=ll_lr_1e-4
python -m experiments.run_seeds --config experiments/configs/lunarlander_dqn.json --set lr=0.002 --set exp_name=ll_lr_2e-3

# Target network update period (the baseline uses 1000)
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

### Summary tables

```bash
python -m experiments.summarize results/cartpole_qlearning results/cartpole_dqn results/cartpole_ddqn --out results/tier1_summary.csv
python -m experiments.summarize results/lunarlander_dqn results/ll_ablation_no_target results/ll_ablation_no_replay results/ll_lr_1e-4 results/ll_lr_2e-3 results/ll_target_100 results/ll_target_5000 --out results/tier2_summary.csv
python -m experiments.summarize results/lunarlander_dqn results/lunarlander_ddqn --out results/tier3_summary.csv
```

## Paper figures

Each command writes to `results/figures/`. The images used by `report/main.tex` are in
`report/figures/`. There, `tier1_cartpole.png` and `tier3_cartpole_eval.png` are earlier versions of
the plot: they show the same curves, but with the band not clipped at 500 and with the threshold
label and legend in a different position.

**Fig. 2. Tier 1: Q-learning vs. DQN**

```bash
python -m experiments.plot_results --exps results/cartpole_qlearning results/cartpole_dqn --labels "Q-learning" "DQN" --threshold 475 --clip-max 500 --out results/figures/tier1_cartpole.png
```

**Fig. 3a. Ablations**

```bash
python -m experiments.plot_results --exps results/lunarlander_dqn results/ll_ablation_no_target results/ll_ablation_no_replay --labels "DQN" "Sin red objetivo" "Sin replay" --threshold 200 --out results/figures/tier2_ablation.png
```

**Fig. 3b. Target network update period**

```bash
python -m experiments.plot_results --exps results/ll_target_100 results/lunarlander_dqn results/ll_target_5000 --labels "target 100" "target 1000" "target 5000" --threshold 200 --out results/figures/tier2_target.png
```

**Fig. 3c. Learning rate**

```bash
python -m experiments.plot_results --exps results/ll_lr_1e-4 results/lunarlander_dqn results/ll_lr_2e-3 --labels "lr 1e-4" "lr 5e-4" "lr 2e-3" --threshold 200 --out results/figures/tier2_lr.png
```

**Fig. 4a. CartPole: evaluation return**

```bash
python -m experiments.plot_results --exps results/cartpole_dqn results/cartpole_ddqn --labels "DQN" "Double DQN" --threshold 475 --clip-max 500 --out results/figures/tier3_cartpole_eval.png
```

**Fig. 4b. CartPole: estimated vs. actual Q**

```bash
python -m experiments.plot_results --exps results/cartpole_dqn results/cartpole_ddqn --labels "DQN" "Double DQN" --mode overestimation --out results/figures/tier3_cartpole_overestimation.png
```

**Fig. 4c. LunarLander: evaluation return**

```bash
python -m experiments.plot_results --exps results/lunarlander_dqn results/lunarlander_ddqn --labels "DQN" "Double DQN" --threshold 200 --out results/figures/tier3_eval.png
```

**Fig. 6a. Ablations: estimated vs. actual Q**

```bash
python -m experiments.plot_results --exps results/lunarlander_dqn results/ll_ablation_no_target results/ll_ablation_no_replay --labels "DQN" "Sin red objetivo" "Sin replay" --mode overestimation --out results/figures/tier2_ablation_overestimation.png
```

**Fig. 6b. Double DQN: estimated vs. actual Q**

```bash
python -m experiments.plot_results --exps results/lunarlander_dqn results/lunarlander_ddqn --labels "DQN" "Double DQN" --mode overestimation --out results/figures/tier3_overestimation.png
```

**Fig. 6c. Training return**

```bash
python -m experiments.plot_results --exps results/lunarlander_dqn results/lunarlander_ddqn --labels "DQN" "Double DQN" --mode train --threshold 200 --out results/figures/tier3_train.png
```

**Fig. 5. GIFs the frames come from**

```bash
python -m experiments.record_gif --run results/cartpole_dqn/seed_123
python -m experiments.record_gif --run results/lunarlander_ddqn/seed_42
python -m experiments.record_gif --run results/ll_lr_1e-4/seed_42
```

**Additional figure (not used in the paper):** the three CartPole curves together.

```bash
python -m experiments.plot_results --exps results/cartpole_qlearning results/cartpole_dqn results/cartpole_ddqn --labels "Q-learning" "DQN" "Double DQN" --threshold 475 --clip-max 500 --out results/figures/tier1_cartpole_all.png
```

Fig. 1 is a diagram in the paper and is not generated by these scripts. There is also no script to
assemble the frames of Fig. 5 from the GIFs: the final image is
`report/figures/qualitative_frames.png`.

## Notebooks

Open them from `notebooks/`. The Tier 1 (DQN) and Tier 3 notebooks read the results in `results/`
and do not retrain unless `DEMO = True` is set.

| Notebook | What it does |
|---|---|
| `FINAL_Q_Learning_CartPole.ipynb` | Tier 1, final version of Q-learning: discretization, training, experiments, 300,000-step budget, validation and selected configuration. Trains inside the notebook. |
| `Q_Learning_CartPole.ipynb` | Tier 1, full Q-learning development with all the experiments (epsilon, alpha, discretization limits and others). Trains inside the notebook. |
| `TIER1_DQN_CartPole.ipynb` | Tier 1: explains DQN step by step, hyperparameter selection, per-seed results and comparison with Q-learning. |
| `TIER3_DoubleDQN.ipynb` | Tier 3: maximization bias, the Double DQN target and results on CartPole-v1 and LunarLander-v3. |

## Results

Final evaluation: 100 greedy episodes per seed, on the final model. Mean ± std is computed across
seeds. The data comes from `results/*_summary.csv` and the `final.json` files.

### Tier 1: CartPole-v1

| Method | Final mean ± std | Seeds 42 / 123 / 456 | Solved | Steps to threshold (42 / 123 / 456) | Time (min) |
|---|---|---|---|---|---|
| Q-learning | 493.88 ± 5.14 | 499.0 / 486.8 / 495.8 | 3/3 | 190k / 140k / 150k | 0.5 |
| DQN | 369.51 ± 184.54 | 108.5 / 500.0 / 500.0 | 2/3 | 190k / 200k / 170k | 7.4 |

### Tier 2: DQN on LunarLander-v3

| Variant | Final mean ± std | Seeds 42 / 123 / 456 | Solved | Steps to threshold (42 / 123 / 456) | Q(s₀) estimated / actual |
|---|---|---|---|---|---|
| Baseline (lr 5e-4, target network every 1000) | 250.20 ± 9.27 | 248.3 / 262.4 / 239.9 | 3/3 | 190k / 210k / 270k | 71.3 / 71.3 |
| No target network | 214.46 ± 31.09 | 235.2 / 237.6 / 170.5 | 2/3 | 110k / 70k / 90k | 78.5 / 66.1 |
| No replay buffer | 98.12 ± 95.13 | −34.8 / 146.6 / 182.6 | 0/3 | 460k / not reached / 350k | 82.0 / 33.9 |
| lr 1e-4 | 153.91 ± 108.26 | 1.4 / 241.5 / 218.9 | 2/3 | not reached / 350k / 330k | 60.6 / 57.7 |
| lr 2e-3 | 204.68 ± 35.81 | 236.3 / 154.6 / 223.2 | 2/3 | 70k / 160k / 180k | 80.5 / 70.2 |
| Target network every 100 | 193.86 ± 85.65 | 72.7 / 254.5 / 254.3 | 2/3 | 360k / 250k / 90k | 70.2 / 68.7 |
| Target network every 5000 | 223.66 ± 25.17 | 190.4 / 229.2 / 251.4 | 2/3 | 200k / 230k / 200k | 61.0 / 70.8 |

### Tier 3: Double DQN vs. DQN

| Environment | Method | Final mean ± std | Seeds 42 / 123 / 456 | Solved | Steps to threshold (42 / 123 / 456) | Q(s₀) estimated / actual |
|---|---|---|---|---|---|---|
| LunarLander-v3 | DQN | 250.20 ± 9.27 | 248.3 / 262.4 / 239.9 | 3/3 | 190k / 210k / 270k | 71.3 / 71.3 |
| LunarLander-v3 | Double DQN | 240.84 ± 28.86 | 239.4 / 206.2 / 276.9 | 3/3 | 90k / 260k / 110k | 71.3 / 69.4 |
| CartPole-v1 | DQN | 369.51 ± 184.54 | 108.5 / 500.0 / 500.0 | 2/3 | 190k / 200k / 170k | 109.5 / 87.8 |
| CartPole-v1 | Double DQN | 473.75 ± 37.12 | 421.2 / 500.0 / 500.0 | 2/3 | 130k / 120k / 130k | 106.0 / 99.0 |

## Code map

| Concept | Where |
|---|---|
| Q-network | `agents/dqn.py` → `QNetwork` |
| Circular replay buffer | `agents/dqn.py` → `ReplayBuffer` |
| ε-greedy | `DQNAgent.act` and `common/utils.py` → `linear_epsilon` |
| TD target for DQN and Double DQN | `DQNAgent.update` (`if self.double` block) |
| Target network and its ablation | `DQNAgent.__init__` and `sync_target` |
| Replay ablation | `ReplayBuffer.latest` |
| Greedy evaluation and overestimation | `common/utils.py` → `evaluate` |
| Q-learning | `agents/tabular.py` and `experiments/train_qlearning.py` |

Any hyperparameter can be changed without editing the JSON, for example:
`--set lr=0.001 --set exp_name=new_name`.

## Hardware

All runs used CPU only, on a laptop with 16 threads and 16 GB of RAM. Each process used one thread.

## Use of generative AI

T. Chilcañan used ChatGPT (OpenAI) to support the implementation, debugging and analysis of
Q-learning; A. Espinoza used Claude (Anthropic) to implement DQN and its variants, the experiment
and plotting scripts, to reproduce Q-learning with the shared protocol, to diagnose results and to
draft the paper.
