"""Planner / expert-generator wiring.

Fast tests run by default. The end-to-end smoke test (plan one apple on tree 8 headless, then generate one
expert trajectory from it) needs the simulator data and takes a few minutes:

    APPLE_AGENT_SMOKE=1 pytest tests/test_planning_pipeline.py -s
"""
import copy
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

from apple_agent.config import CONFIG
from apple_agent.planning.common import (
    ResultMode,
    apply_tree_setup_from_waypoints_hdf5,
    resolve_run_id,
    tree_urdf_relpath,
)


def test_entry_points_import():
    # catches broken imports after renames / moves (these modules import the planning helpers)
    import apple_agent.data.generate_expert_data  # noqa: F401
    import apple_agent.planning.plan_waypoints  # noqa: F401
    import apple_agent.rl.train_apple_picking_expert  # noqa: F401


def test_tree_urdf_relpath():
    from pybullet_tree_sim import DATA_PATH

    rel = "urdf/trees/envy/generated/LPy_envy_00008.urdf"
    assert tree_urdf_relpath(os.path.join(DATA_PATH, rel)) == rel
    assert tree_urdf_relpath("/old/checkout/pybullet_tree_sim/" + rel) == rel  # path from an older file
    assert tree_urdf_relpath(rel) == rel


def test_resolve_run_id(tmp_path):
    (tmp_path / "tree01.hdf5").touch()
    (tmp_path / "tree02.hdf5").touch()
    assert resolve_run_id(None, tmp_path) == ("tree02", tmp_path / "tree02.hdf5")
    assert resolve_run_id("tree01", tmp_path) == ("tree01", tmp_path / "tree01.hdf5")
    with pytest.raises(FileNotFoundError):
        resolve_run_id("missing", tmp_path)


def test_apply_tree_setup_from_waypoints_hdf5(tmp_path):
    path = tmp_path / "w.hdf5"
    with h5py.File(path, "w") as f:
        f.create_group("0").attrs["fail_mode"] = ResultMode.NO_PATH.value
        g = f.create_group("1")
        g.attrs.update(fail_mode=ResultMode.SUCCESS.value, tree_id=8, tree_scale=0.65,
                       tree_pos=np.array([0.5, 0.0, 0.0]), tree_orientation=np.array([0, 0, 0, 1.0]))
    cfg = copy.deepcopy(CONFIG)
    assert apply_tree_setup_from_waypoints_hdf5(path, config=cfg) == "1"
    ts = cfg["tree_setup"]
    assert (ts["tree_id"], ts["scale"]) == (8, 0.65)
    np.testing.assert_allclose(ts["position"], [0.5, 0.0, 0.0])


@pytest.mark.skipif(not os.environ.get("APPLE_AGENT_SMOKE"), reason="set APPLE_AGENT_SMOKE=1 (slow, needs simulator data)")
def test_plan_and_generate_one_apple(tmp_path):
    def run(*args):
        subprocess.run([sys.executable, "-m", *args], cwd=tmp_path, check=True, timeout=1800)

    run("apple_agent.planning.plan_waypoints", "--stage", "planner", "--tree-id", "8",
        "--num_apples", "1", "--run_id", "smoke", "--no-render")
    with h5py.File(tmp_path / "output/waypoints/smoke.hdf5", "r") as f:
        success = [g for g in f.values() if int(g.attrs["fail_mode"]) == ResultMode.SUCCESS.value]
        assert success, "planner found no path for apple 1 of tree 8"
        assert not os.path.isabs(str(success[0].attrs["tree_urdf"]))
        assert len(success[0]["waypoints"]) == int(success[0].attrs["n_waypoints_saved"])

    run("apple_agent.data.generate_expert_data", "--run-id", "smoke", "--path-rank", "1", "--max-groups", "1")
    with h5py.File(tmp_path / "output/agent_data/smoke.hdf5", "r") as f:
        (traj,) = f.values()
        assert traj.attrs["schema_version"] == 2
        assert traj.attrs["gen_step_scale"] == 1.4
        assert not os.path.isabs(str(np.asarray(traj.attrs["tree_urdf"]).reshape(-1)[0]))
        assert len(traj["actions"]) > 0
    assert any((tmp_path / "logs/plan_waypoints").iterdir())
    assert any((tmp_path / "logs/generate_expert_data").iterdir())
