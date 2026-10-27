import glob
import hashlib
import os
from collections import OrderedDict

import numpy as np
import torch as th
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from torchvision.transforms import functional as F


def flow_cache_name(size, num_flow_updates, tag) -> str:
    """File name of a precomputed flow cache; the size and iteration count are part of the name."""
    return f"flow_{size[0]}x{size[1]}_u{num_flow_updates}__{tag}.npz"


class FlowCache:
    """Flow of (rgb, prev_rgb) pairs, keyed by a hash of the two images' pixels.

    A hit is exact by construction (same pixels in -> same flow out), wherever the pair comes from:
    the expert HDF5, a rollout buffer or the live env. Two tiers:
      * fixed:   loaded from files written by apple_agent.data.cache_flow, never evicted;
      * runtime: flow computed during training / eval, bounded LRU (0 entries = off).
    Values are stored as float16 (or float32) arrays of shape (2, h, w).
    """

    def __init__(self, runtime_entries=0, dtype="float16"):
        self.dtype = np.dtype(dtype)
        self.runtime_entries = int(runtime_entries)
        self.fixed = {}
        self.runtime = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(cur: np.ndarray, prev: np.ndarray) -> bytes:
        h = hashlib.blake2b(digest_size=16)
        h.update(np.ascontiguousarray(cur))
        h.update(np.ascontiguousarray(prev))
        return h.digest()

    def get(self, key):
        flow = self.fixed.get(key)
        if flow is None:
            flow = self.runtime.get(key)
            if flow is not None:
                self.runtime.move_to_end(key)
        self.hits += flow is not None
        self.misses += flow is None
        return flow

    def put(self, key, flow: np.ndarray):
        if self.runtime_entries <= 0:
            return
        self.runtime[key] = flow
        while len(self.runtime) > self.runtime_entries:
            self.runtime.popitem(last=False)

    def load_dir(self, cache_dir, size, num_flow_updates) -> int:
        """Load every cache file in `cache_dir` made with this size / iteration count; returns #entries."""
        before = len(self.fixed)
        for path in sorted(glob.glob(os.path.join(cache_dir, flow_cache_name(size, num_flow_updates, "*")))):
            with np.load(path) as f:
                if tuple(f["size"]) != tuple(size) or int(f["num_flow_updates"]) != int(num_flow_updates):
                    raise ValueError(f"{path} was made with a different flow setup than its file name says")
                for k, flow in zip(f["keys"], f["flow"]):
                    self.fixed[k.tobytes()] = flow
        return len(self.fixed) - before

    @staticmethod
    def save_file(path, keys, flows, size, num_flow_updates):
        """keys: list of 16-byte digests, flows: (N, 2, h, w) array already in the cache dtype."""
        tmp = f"{path}.tmp.npz"
        np.savez(tmp, keys=np.frombuffer(b"".join(keys), dtype=np.uint8).reshape(len(keys), -1), flow=flows,
                 size=np.asarray(size), num_flow_updates=np.asarray(num_flow_updates))
        os.replace(tmp, path)


class OpticalFlow:
    """RAFT-small flow between two RGB frames, in units of image size (pixels / size).

    :param size: RAFT input resolution (H, W), multiples of 8, at least 128 (RAFT's minimum). Lower = faster; the flow is normalised
        by `size`, so its values do not depend on the resolution.
    :param num_flow_updates: RAFT refinement iterations (torchvision default 12; fewer = faster).
    :param compile: torch.compile the model. None = legacy default (on, except on Windows).
    :param cache_dir / runtime_cache_entries: see FlowCache. With neither, nothing is cached.
    """

    def __init__(self, size=(240, 240), subprocess=False, shared_var=(None, None), num_envs=1,
                 num_flow_updates=12, compile=None, cache_dir=None, runtime_cache_entries=0, cache_dtype="float16"):
        assert min(size) >= 128 and size[0] % 8 == 0 and size[1] % 8 == 0, \
            f"RAFT-small needs a size that is a multiple of 8 and at least 128, got {size}"
        self.device = "cuda" if th.cuda.is_available() else "cpu"
        weights = Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms()
        model = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).to(self.device)
        self.model = model.eval()
        if compile if compile is not None else os.name != 'nt':
            self.model = th.compile(model)
        self.size = tuple(size)
        self.num_flow_updates = int(num_flow_updates)
        self.subprocess = subprocess
        self.shared_queue, self.shared_dict = shared_var
        self.cache = None
        if cache_dir or runtime_cache_entries > 0:
            self.cache = FlowCache(runtime_cache_entries, cache_dtype)
            if cache_dir:
                n = self.cache.load_dir(cache_dir, self.size, self.num_flow_updates)
                print(f"flow cache: {n} precomputed entries from {cache_dir} "
                      f"({self.size[0]}x{self.size[1]}, {self.num_flow_updates} updates)")
        print("raft model loaded")
        self.num_envs = num_envs
        if self.subprocess:
            self._run_subprocess()

    def _run_subprocess(self):
        while True:
            # rgb, previous_rgb, pid = self.shared_queue.get()
            # while queue is not empty

            # make batch of all the elements in the queue
            rgb_array = []
            previous_rgb_array = []
            pid_list = []

            # while len(pid_list) < self.num_envs:
                # make an array of all the elements in the queue
                # rgb, previous_rgb, pid, name = self.shared_queue.get()
                # rgb_array.append(rgb)
                # previous_rgb_array.append(previous_rgb)
                # pid_list.append(pid)
            rgb, previous_rgb, pid, name = self.shared_queue.get()
            rgb = th.tensor(rgb)
            previous_rgb = th.tensor(previous_rgb)
            # optical_flow = self.calculate_optical_flow(rgb_array, previous_rgb_array)
            optical_flow = self.calculate_optical_flow(rgb, previous_rgb)
            # print(optical_flow.shape)
            self.shared_dict[pid] = optical_flow.cpu().numpy()[0]

    def _preprocess(self, img1, img2):
        # RAFT's transforms rescale only uint8 to [0, 1]; a float image in [0, 255] passes through
        # unscaled and yields garbage flow, so require uint8 (what the env emits).
        if img1.dtype != th.uint8 or img2.dtype != th.uint8:
            raise TypeError(f"OpticalFlow expects uint8 images, got {img1.dtype} / {img2.dtype}")
        img1 = F.resize(img1, size=self.size, antialias=False)
        img2 = F.resize(img2, size=self.size, antialias=False)
        return self.transforms(img1, img2)  # uint8 -> [-1, 1] float

    def compute_uncached(self, current_rgb, previous_rgb):
        """RAFT on a batch (B, 3, H, W) uint8 -> (B, 2, size[0], size[1]) float32 on self.device."""
        current_rgb, previous_rgb = self._preprocess(current_rgb, previous_rgb)
        with th.no_grad():
            list_of_flows = self.model(current_rgb.to(self.device), previous_rgb.to(self.device),
                                       num_flow_updates=self.num_flow_updates)
        predicted_flows = list_of_flows[-1]
        predicted_flows[:, 0, :, :] /= self.size[0]
        predicted_flows[:, 1, :, :] /= self.size[1]
        return predicted_flows

    def calculate_optical_flow(self, current_rgb, previous_rgb):
        if len(current_rgb.shape) == 3:
            current_rgb = th.unsqueeze(current_rgb, 0)
            previous_rgb = th.unsqueeze(previous_rgb, 0)
        if self.cache is None:
            return self.compute_uncached(current_rgb, previous_rgb)

        # Only pairs not seen before go through RAFT; the policy asks for the same frames again
        # whenever it re-evaluates a buffer (BC epochs, PPO update epochs).
        if current_rgb.dtype != th.uint8 or previous_rgb.dtype != th.uint8:
            raise TypeError(f"OpticalFlow expects uint8 images, got {current_rgb.dtype} / {previous_rgb.dtype}")
        cur, prev = current_rgb.cpu().numpy(), previous_rgb.cpu().numpy()
        keys = [self.cache.key(c, p) for c, p in zip(cur, prev)]
        flows = [self.cache.get(k) for k in keys]
        miss = [i for i, f in enumerate(flows) if f is None]
        if miss:
            computed = self.compute_uncached(current_rgb[miss], previous_rgb[miss]).cpu().numpy()
            for i, flow in zip(miss, computed):
                flows[i] = flow.astype(self.cache.dtype)
                self.cache.put(keys[i], flows[i])
        return th.from_numpy(np.stack(flows)).to(self.device, th.float32)
