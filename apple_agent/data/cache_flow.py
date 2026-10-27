"""
Precompute the optical flow of an agent-data file so BC training never runs RAFT on it.

    python -m apple_agent.data.cache_flow --run-id tree08_rank1_v2 [--flow_size 128] [--flow_updates 12] [--all] [--verify 8]

Writes output/flow_cache/flow_<H>x<W>_u<N>__<run_id>.npz (one entry per distinct (rgb, prev_rgb) pair of every
successful trajectory; --all also takes the failed ones). The policy loads every file in that folder made with
its own flow size and iteration count (config.TRAIN_CONFIG['flow']), so there is nothing to pass to training.
Entries are keyed by image content, so a stale or foreign cache can only miss, never return a wrong flow.
"""
import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch as th

from apple_agent.apple_picking_sb3.apple_gym.optical_flow import FlowCache, OpticalFlow, flow_cache_name
from apple_agent.config import TRAIN_CONFIG
from apple_agent.data import agent_data


def main() -> int:
    cfg = TRAIN_CONFIG["flow"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True, help="output/agent_data/<run-id>.hdf5")
    parser.add_argument("--flow_size", type=int, default=cfg["size"][0], help="square RAFT input size (multiple of 8)")
    parser.add_argument("--flow_updates", type=int, default=cfg["num_flow_updates"], help="RAFT iterations")
    parser.add_argument("--all", action="store_true", help="also cache failed trajectories (BC only uses successful ones)")
    parser.add_argument("--verify", type=int, default=8, metavar="N",
                        help="after writing, check N pairs: cached flow must equal live RAFT (0 = skip)")
    args = parser.parse_args()

    size = (args.flow_size, args.flow_size)
    path = Path(cfg["cache_dir"] or "output/flow_cache")
    path.mkdir(parents=True, exist_ok=True)
    out_file = path / flow_cache_name(size, args.flow_updates, args.run_id)
    data_file = Path("output") / "agent_data" / f"{args.run_id}.hdf5"
    if not data_file.is_file():
        print(f"not found: {data_file}")
        return 1

    flow = OpticalFlow(size=size, num_flow_updates=args.flow_updates, compile=False)
    keys, flows, first_pairs = [], [], None
    seen = set()
    t0 = time.time()
    with h5py.File(data_file, "r") as f:
        for name, grp in f.items():
            agent_data.check_schema(grp)
            if not (args.all or bool(grp.attrs["success"])):
                continue
            frames = agent_data.load_frames(grp, ["rgb", "prev_rgb"])  # T+1 frames; prev_rgb rebuilt from rgb
            cur, prev = frames["rgb"], frames["prev_rgb"]
            first_pairs = first_pairs or (cur, prev)
            todo = []
            for i in range(len(cur)):
                k = FlowCache.key(cur[i], prev[i])
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
                    todo.append(i)
            for b in range(0, len(todo), cfg["precompute_batch_size"]):
                idx = todo[b:b + cfg["precompute_batch_size"]]
                out = flow.compute_uncached(th.from_numpy(cur[idx]), th.from_numpy(prev[idx]))
                flows.extend(out.cpu().numpy().astype(cfg["cache_dtype"]))
            print(f"{name}: {len(cur)} frames, {len(todo)} new pairs  [{len(flows)} total, {time.time() - t0:.0f}s]", flush=True)

    if not flows:
        print("no trajectories to cache (use --all to include failed ones)")
        return 1
    FlowCache.save_file(out_file, keys, np.stack(flows), size, args.flow_updates)
    print(f"wrote {out_file}  ({out_file.stat().st_size / 1e6:.0f} MB, {len(flows)} entries, {time.time() - t0:.0f}s)")

    if args.verify > 0:
        cur, prev = first_pairs
        idx = np.linspace(0, len(cur) - 1, num=min(args.verify, len(cur)), dtype=int)
        cached = OpticalFlow(size=size, num_flow_updates=args.flow_updates, compile=False,
                             cache_dir=str(path), cache_dtype=cfg["cache_dtype"])
        a, b = th.from_numpy(cur[idx]), th.from_numpy(prev[idx])
        from_cache = cached.calculate_optical_flow(a, b)
        live = cached.compute_uncached(a, b)
        err = (from_cache - live).abs().max().item()
        scale = live.abs().max().item()
        ok = cached.cache.misses == 0 and err <= 1e-3 * max(scale, 1e-6)
        print(f"verify: {len(idx)} pairs, hits={cached.cache.hits} misses={cached.cache.misses}, "
              f"max |cached - live| = {err:.2e} (max |flow| {scale:.2e}) -> {'OK' if ok else 'MISMATCH'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
