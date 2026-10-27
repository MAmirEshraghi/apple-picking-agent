#!/usr/bin/env bash
# BC 20k image-channel ablation: cache flow once, then train + eval each config in turn.
# Usage: bash scripts/run_bc_ablation.sh [run_id]      (run from anywhere; finished evals are skipped)
source /home/rob/Codes/apple_env.sh

# one instance only: a second copy would share the same RAM, log folders and checkpoints
exec 9>/tmp/run_bc_ablation.lock
flock -n 9 || { echo "already running, exiting"; exit 1; }
RUN=${1:-tree08_rank1_v2}
STEPS=20000

# name:image_channels
EXPERIMENTS=(
  "bc20k_dm:depth,mask"         # baseline
  "bc20k_dmf:depth,mask,flow"   # main test: does flow add anything?
  "bc20k_fm:flow,mask"          # the old 0% case, with the fixed flow at 128
  "bc20k_d:depth"               # is the mask needed?
  "bc20k_m:mask"                # does depth carry the result?
  "bc20k_f:flow"                # flow alone
  "bc20k_none:none"             # proprio-only reference
)

mkdir -p logs/eval
python -m apple_agent.data.cache_flow --run-id "$RUN" || { echo "flow cache failed"; exit 1; }

failed=()
for exp in "${EXPERIMENTS[@]}"; do
  name=${exp%%:*}; channels=${exp#*:}
  if [[ -f logs/eval/$name.json ]]; then echo "== $name: already evaluated, skipping"; continue; fi
  echo "== $name ($channels) $(date +%T)"
  python -m apple_agent.rl.train_apple_picking_expert --run_id "$RUN" --run_name "$name" \
      --image_channels "$channels" --bc_only --bc_warmup_steps 0 --total_timesteps $STEPS \
      --checkpoint_freq 5000 --no-memory_debug \
  && python -m apple_agent.rl.eval_apple_picking_expert --run_id "$RUN" \
      --checkpoint "logs/$name/model_${STEPS}_steps.zip" --eval_all --max_steps 120 \
      --output "logs/eval/$name.json" \
  || { echo "!! $name failed"; failed+=("$name"); }
done

echo "== done $(date +%T); failed: ${failed[*]:-none}"
python - <<'PY'
import glob, json, os
for f in sorted(glob.glob("logs/eval/bc20k_*.json")):
    s = json.load(open(f))["summary"]
    print(f"{os.path.basename(f)[:-5]:12s} success {s['success_rate']:.0%}  collision {s['collision_rate']:.0%}  "
          f"timeout {s['timeout_rate']:.0%}  final dist {s['mean_final_ee_dist']:.2f} m")
PY
