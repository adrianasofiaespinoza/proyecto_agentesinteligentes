"""
Lanza train_dqn.py para varias semillas, en paralelo (un proceso por semilla).

Uso:
  python -m experiments.run_seeds --config experiments/configs/cartpole_dqn.json
  python -m experiments.run_seeds --config experiments/configs/lunarlander_ddqn.json --parallel 3
  python -m experiments.run_seeds --config ... --seeds 42 123 456 --set lr=1e-3 --set exp_name=ll_lr1e-3

Con 16 GB de RAM y >= 4 nucleos, --parallel 3 es seguro (cada proceso usa 1 hilo, < 1 GB).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()

    def cmd(seed):
        c = [sys.executable, "-m", "experiments.train_dqn", "--config", args.config,
             "--seed", str(seed), "--results-dir", args.results_dir]
        for s in args.set:
            c += ["--set", s]
        return c

    pending = list(args.seeds)
    running = []
    failed = []
    while pending or running:
        while pending and len(running) < args.parallel:
            seed = pending.pop(0)
            running.append((seed, subprocess.Popen(cmd(seed), cwd=ROOT)))
        for seed, proc in list(running):
            if proc.poll() is not None:
                running.remove((seed, proc))
                if proc.returncode != 0:
                    failed.append(seed)
        time.sleep(1)
    if failed:
        sys.exit(f"Fallaron las semillas: {failed}")
    print("Todas las semillas terminaron.")


if __name__ == "__main__":
    main()
