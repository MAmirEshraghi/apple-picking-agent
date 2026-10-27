#!/usr/bin/env bash
# PPO modality ablation. For every image-channel set, starting from that set's own 20k BC checkpoint:
#   ppo_<mod>_s<seed>   PPO+BC (online)                 \  identical settings; the only difference
#   ctl_<mod>_s<seed>   BC only, same extra steps        /  is --bc_only (no sim rollouts, no PPO loss)
# Each run is evaluated at its last EVAL_LAST checkpoints (10 training starts, deterministic, 120 steps).
# Compare modalities by PPO - control (what online training adds), plus the online-training curves.
#
# Usage:  bash scripts/run_ppo_ablation.sh            (finished runs / evals are skipped, so re-running resumes)
#         SEEDS="1" MODS="none mask" bash scripts/run_ppo_ablation.sh      (subset)
source /home/rob/Codes/apple_env.sh
exec 9>/tmp/run_bc_ablation.lock          # shared with the BC script: one experiment at a time
flock -n 9 || { echo "another experiment script is running, exiting"; exit 1; }

RUN=${RUN:-tree08_rank1_v2}
# Cost: depth and flow need a camera render per env step (~0.6 s on this CPU, TinyRenderer); none and mask
# do not (~0.05 s). So one seed at 10k steps is ~15 h on the laptop; 30k steps or 2 seeds multiply that.
STEPS=${STEPS:-10000}                     # extra steps on top of the 20k BC checkpoint (same for every modality)
CKPT_FREQ=${CKPT_FREQ:-5000}
EVAL_LAST=${EVAL_LAST:-2}                 # evaluate the last N checkpoints of each run (here +5k and +10k)
SEEDS=${SEEDS:-"1"}
# cheapest first; flow sets last (slowest)
MODS=${MODS:-"none mask depth depth,mask flow flow,mask depth,mask,flow"}

# Same for every modality and for both arms (fairness). Chosen from the proprio-only PPO trial:
# strong BC anchor, lower exploration noise, gentler updates, episodes close to the eval length.
COMMON=(--bc_warmup_steps 0 --bc_coeff 5.0 --lr_scale 0.3 --batch_size 64 --ent_coef 0
        --max_episode_steps 150 --total_timesteps "$STEPS" --checkpoint_freq "$CKPT_FREQ" --no-memory_debug)
PPO_ONLY=(--init_std 0.1)

short() { case $1 in none) echo none;; mask) echo m;; depth) echo d;; depth,mask) echo dm;; flow) echo f;;
                      flow,mask) echo fm;; depth,mask,flow) echo dmf;; *) echo "${1//,/_}";; esac; }

mkdir -p logs/eval
[[ "$MODS" == *flow* ]] && python -m apple_agent.data.cache_flow --run-id "$RUN"

failed=()
train_eval() {   # name channels base extra-args...
  local name=$1 ch=$2 base=$3; shift 3
  if [[ ! -f logs/$name/final_model.zip ]]; then        # final_model.zip = training finished
    echo "== train $name ($ch) $(date +%T)"
    python -m apple_agent.rl.train_apple_picking_expert --run_id "$RUN" --run_name "$name" --image_channels "$ch" \
        --load_checkpoint "$base" "${COMMON[@]}" "$@" || { echo "!! $name training failed"; failed+=("$name"); return; }
  fi
  for f in $(ls logs/$name | grep -E '^model_[0-9]+_steps\.zip$' | sort -t_ -k2 -n | tail -n "$EVAL_LAST"); do
    local ck=logs/$name/$f steps=${f//[!0-9]/}
    local out=logs/eval/${name}_${steps}.json
    [[ -f $out ]] && continue
    echo "== eval $name @ $steps $(date +%T)"
    python -m apple_agent.rl.eval_apple_picking_expert --run_id "$RUN" --checkpoint "$ck" \
        --eval_all --max_steps 120 --output "$out" > "logs/eval/${name}_${steps}.log" 2>&1 \
      || { echo "!! eval $name @ $steps failed (see logs/eval/${name}_${steps}.log)"; failed+=("$name@$steps"); }
  done
}

for seed in $SEEDS; do                    # seed-major: each pass is a complete, comparable set
  for ch in $MODS; do
    m=$(short "$ch"); base=logs/bc20k_$m/model_20000_steps.zip
    [[ -f $base ]] || { echo "!! missing BC base $base, skipping $ch"; failed+=("base_$m"); continue; }
    train_eval "ppo_${m}_s${seed}" "$ch" "$base" --seed "$seed" "${PPO_ONLY[@]}"
    train_eval "ctl_${m}_s${seed}" "$ch" "$base" --seed "$seed" --bc_only
  done
done

echo "== done $(date +%T); failed: ${failed[*]:-none}"
python scripts/summarize_ppo_ablation.py
