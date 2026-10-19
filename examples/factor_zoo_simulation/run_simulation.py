"""
Run the factor-zoo Monte Carlo.

Each replication is written to results/raw/<scenario>/rep_<rep>.pkl, so an
interrupted run continues where it stopped. Aggregate with summarize.py.

Usage
-----
  python run_simulation.py --list
  python run_simulation.py --scenario D2_proxy --reps 100 --n-jobs 4
  python run_simulation.py --scenario D1_sparse D0_market_only --reps 50
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import pickle
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HERE), os.environ.get("PYTHONPATH")]))

from joblib import Parallel, delayed  # noqa: E402

import montecarlo as mc  # noqa: E402

RAW_DIR = HERE / "results" / "raw"


def _run_one(name, rep, out_dir):
    warnings.simplefilter("ignore")
    target = out_dir / f"rep_{rep:04d}.pkl"
    if target.exists():
        return rep, 0.0, "skipped"
    t0 = time.time()
    result = mc.run_replication(mc.SCENARIOS[name], rep)
    tmp = target.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(target)
    return rep, time.time() - t0, "done"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", nargs="+", default=["D2_proxy"])
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--list", action="store_true", help="List the scenarios and exit.")
    args = parser.parse_args(argv)

    if args.list:
        for name, scen in mc.SCENARIOS.items():
            print(f"{name:22s} {scen.description}")
        return

    for name in args.scenario:
        if name not in mc.SCENARIOS:
            sys.exit(f"Unknown scenario {name!r}; use --list.")
        out_dir = RAW_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        reps = range(args.start, args.start + args.reps)
        print(f"[{name}] {len(reps)} replications on {args.n_jobs} worker(s)", flush=True)
        t0 = time.time()
        jobs = (delayed(_run_one)(name, rep, out_dir) for rep in reps)
        for rep, secs, status in Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(jobs):
            print(f"[{name}] rep {rep:4d} {status} {secs:7.1f}s  (elapsed {time.time() - t0:7.0f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
