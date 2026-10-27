import numpy as np
import pytest
import torch as th

from apple_agent.apple_picking_sb3.apple_gym.optical_flow import FlowCache, OpticalFlow, flow_cache_name


def test_key_depends_on_pixels_and_order():
    a, b = np.zeros((3, 8, 8), np.uint8), np.ones((3, 8, 8), np.uint8)
    assert FlowCache.key(a, b) == FlowCache.key(a.copy(), b.copy())
    assert FlowCache.key(a, b) != FlowCache.key(b, a)


def test_runtime_tier_is_bounded_lru():
    cache = FlowCache(runtime_entries=2)
    for i in range(3):
        cache.put(bytes([i]), np.zeros((2, 8, 8), np.float16))
    assert cache.get(bytes([0])) is None and cache.get(bytes([2])) is not None


def test_file_roundtrip_and_memoised_flow(tmp_path):
    size, updates = (128, 128), 2
    rng = np.random.default_rng(0)
    cur = rng.integers(0, 255, (3, 3, 48, 48), dtype=np.uint8)
    prev = rng.integers(0, 255, (3, 3, 48, 48), dtype=np.uint8)
    try:
        live = OpticalFlow(size=size, num_flow_updates=updates, compile=False)
    except Exception as e:  # RAFT weights not downloadable offline
        pytest.skip(str(e))
    a, b = th.from_numpy(cur), th.from_numpy(prev)
    expected = live.compute_uncached(a, b).cpu().numpy().astype(np.float16)
    keys = [FlowCache.key(c, p) for c, p in zip(cur, prev)]
    FlowCache.save_file(tmp_path / flow_cache_name(size, updates, "t"), keys, expected, size, updates)

    cached = OpticalFlow(size=size, num_flow_updates=updates, compile=False, cache_dir=str(tmp_path))
    out = cached.calculate_optical_flow(a, b)
    assert cached.cache.misses == 0 and cached.cache.hits == 3
    np.testing.assert_allclose(out.numpy(), expected.astype(np.float32))

    other = OpticalFlow(size=(136, 136), num_flow_updates=updates, compile=False, cache_dir=str(tmp_path),
                        runtime_cache_entries=8)  # different size -> the file is ignored
    other.calculate_optical_flow(a, b)
    other.calculate_optical_flow(a, b)
    assert (other.cache.misses, other.cache.hits) == (3, 3)
