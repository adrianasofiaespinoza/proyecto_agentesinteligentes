#!/usr/bin/env bash
# Todos los entrenamientos de LunarLander (Tier 2 + Tier 3), 3 semillas cada uno.
# Ejecutar desde la carpeta raiz del proyecto:  bash experiments/run_lunarlander_all.sh
# Cada linea se puede correr por separado (p. ej. de noche). PAR = procesos en paralelo.
set -e
PAR=${PAR:-3}
CFG=experiments/configs/lunarlander_dqn.json
RUN="python -m experiments.run_seeds --parallel $PAR"

# --- Linea base (sirve para Tier 2 Y como "vanilla DQN" del Tier 3) ---
$RUN --config $CFG

# --- Tier 3: Double DQN (opcional: Dueling + Double) ---
$RUN --config experiments/configs/lunarlander_ddqn.json
# $RUN --config experiments/configs/lunarlander_dueling_ddqn.json

# --- Tier 2: ablaciones ---
$RUN --config $CFG --set use_target_network=false --set exp_name=ll_ablation_no_target
$RUN --config $CFG --set use_replay=false        --set exp_name=ll_ablation_no_replay

# --- Tier 2: estudio de hiperparametros (un parametro a la vez; el valor por defecto es la base) ---
# learning rate (default 5e-4)
$RUN --config $CFG --set lr=0.0001 --set exp_name=ll_lr_1e-4
$RUN --config $CFG --set lr=0.002  --set exp_name=ll_lr_2e-3
# frecuencia de actualizacion de la red objetivo (default 1000)
$RUN --config $CFG --set target_update_freq=100  --set exp_name=ll_target_100
$RUN --config $CFG --set target_update_freq=5000 --set exp_name=ll_target_5000
