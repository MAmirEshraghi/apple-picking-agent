"""Summarise the PPO modality ablation (scripts/run_ppo_ablation.sh). Run from the repo root:
    python scripts/summarize_ppo_ablation.py
Reads logs/eval/{ppo,ctl}_<mod>_s<seed>_<steps>.json, logs/eval/bc20k_<mod>.json and the training logs.
"""
import glob
import json
import os
import re
from collections import defaultdict

MODS = ["none", "m", "d", "dm", "f", "fm", "dmf"]
EVAL_RE = re.compile(r"(ppo|ctl)_(\w+?)_s(\d+)_(\d+)\.json$")


def load_evals():
    evals = defaultdict(dict)  # (arm, mod, seed) -> {steps: summary}
    for f in glob.glob("logs/eval/*.json"):
        m = EVAL_RE.search(os.path.basename(f))
        if m:
            arm, mod, seed, steps = m.groups()
            evals[(arm, mod, int(seed))][int(steps)] = json.load(open(f))["summary"]
    return evals


def online_stats(run_name):
    """Online episodes and update health from the run's training log (PPO arm only)."""
    for f in sorted(glob.glob("logs/train_apple_picking_expert/*.log"), reverse=True):
        with open(f, errors="ignore") as fh:
            text = fh.read()
        if f"run_name={run_name} " not in text[:5000]:
            continue
        eps = re.findall(r"\[train:episode\] t=\d+ env=0 reason=(\w+)", text)
        last = eps[-max(1, len(eps) // 3):]

        def vals(key):
            return [float(v) for v in re.findall(r"\|\s+" + key + r"\s+\|\s+([-\d.e]+)", text)]

        ev, kl, clip, bc = vals("explained_variance"), vals("approx_kl_div"), vals("clip_fraction"), vals("bc_loss")
        return {
            "episodes": len(eps),
            "goal_last3": sum(r == "GOAL" for r in last) / max(len(last), 1),
            "coll_last3": sum(r == "COLLISION" for r in last) / max(len(last), 1),
            "expl_var": ev[-1] if ev else float("nan"),
            "kl": sum(kl) / len(kl) if kl else float("nan"),
            "clip": sum(clip) / len(clip) if clip else float("nan"),
            "bc_loss": bc[-1] if bc else float("nan"),
        }
    return None


def pct(x):
    return "  -  " if x is None else f"{x:4.0%}"


def main():
    evals = load_evals()
    if not evals:
        print("no PPO ablation evals found in logs/eval/")
        return
    print("\n=== Eval (10 training starts, deterministic, 120 steps): success / collision / final dist, last checkpoint")
    print(f"{'mod':5s} {'BC base':>9s}   {'arm':4s} {'seed':>4s} {'steps':>6s}  {'succ':>5s} {'coll':>5s} {'dist':>5s}   (earlier ckpt succ)")
    gains = defaultdict(list)
    for mod in MODS:
        base_f = f"logs/eval/bc20k_{mod}.json"
        base = json.load(open(base_f))["summary"]["success_rate"] if os.path.exists(base_f) else None
        seeds = sorted({s for (a, m, s) in evals if m == mod})
        for seed in seeds:
            last = {}
            for arm in ("ppo", "ctl"):
                runs = evals.get((arm, mod, seed))
                if not runs:
                    continue
                steps = sorted(runs)
                s = runs[steps[-1]]
                last[arm] = s
                earlier = " ".join(f"{runs[k]['success_rate']:.0%}@{k}" for k in steps[:-1])
                print(f"{mod:5s} {pct(base):>9s}   {arm:4s} {seed:>4d} {steps[-1]:>6d}  {pct(s['success_rate']):>5s} "
                      f"{pct(s['collision_rate']):>5s} {s['mean_final_ee_dist']:5.2f}   {earlier}")
            if "ppo" in last and "ctl" in last:
                gains[mod].append((last["ppo"]["success_rate"] - last["ctl"]["success_rate"],
                                   last["ppo"]["collision_rate"] - last["ctl"]["collision_rate"],
                                   last["ppo"]["mean_final_ee_dist"] - last["ctl"]["mean_final_ee_dist"]))

    print("\n=== What online training adds: PPO - control at the last checkpoint (per seed, then mean)")
    print(f"{'mod':5s} {'d_success':>22s} {'d_collision':>22s} {'d_final_dist':>24s}")
    for mod in MODS:
        g = gains.get(mod)
        if not g:
            continue
        cols = []
        for i in range(3):
            per = " ".join(f"{x[i]:+.0%}" if i < 2 else f"{x[i]:+.2f}" for x in g)
            mean = sum(x[i] for x in g) / len(g)
            cols.append(f"{per} (mean {mean:+.0%})" if i < 2 else f"{per} (mean {mean:+.2f})")
        print(f"{mod:5s} {cols[0]:>22s} {cols[1]:>22s} {cols[2]:>24s}")

    print("\n=== Online training (PPO arm): episodes, success/collision in the last third, update health")
    print(f"{'run':14s} {'eps':>4s} {'succ':>5s} {'coll':>5s} {'expl_var':>8s} {'kl':>6s} {'clip':>5s} {'bc_loss':>8s}")
    for mod in MODS:
        for seed in sorted({s for (a, m, s) in evals if m == mod and a == "ppo"}):
            st = online_stats(f"ppo_{mod}_s{seed}")
            if st:
                print(f"{'ppo_' + mod + '_s' + str(seed):14s} {st['episodes']:4d} {st['goal_last3']:5.0%} {st['coll_last3']:5.0%} "
                      f"{st['expl_var']:8.2f} {st['kl']:6.3f} {st['clip']:5.2f} {st['bc_loss']:8.4f}")
    print("\nWith 10 eval episodes one episode = 10 points; treat gaps under ~20 points as noise unless both seeds agree.")


if __name__ == "__main__":
    main()
