"""
Planning helpers shared by the planner (plan_waypoints), the expert generator (data/generate_expert_data)
and training (rl/train_apple_picking_expert): collision checking, path refinement, waypoints-HDF5 I/O.
"""
import os
from enum import Enum
from pathlib import Path

import cv2
import h5py
import numpy as np
from pybullet_planning.interfaces.planner_interface.joint_motion_planning import (
    get_difference_fn,
    get_distance_fn,
    get_extend_fn,
    get_sample_fn,
)
from pybullet_planning.motion_planners.smoothing import smooth_path
from pybullet_tree_sim import DATA_PATH
from pybullet_tree_sim.robot import Robot
from pybullet_tree_sim.tree import Tree
from pybullet_tree_sim.utils.viz_debug import draw_debug_sphere
from zenlog import log

from apple_agent.config import CONFIG


class ResultMode(Enum):
    NO_SOLUTION = 0
    SUCCESS = 1
    NO_PATH = 2


def tree_id_str_from_config(config: dict | None = None) -> str:
    """PyBullet mesh id, e.g. LPy_envy_00017."""
    cfg = config or CONFIG
    ts = cfg["tree_setup"]
    return f"{ts['tree_namespace']}_{ts['tree_type']}_{int(ts['tree_id']):05d}"


def tree_urdf_relpath(urdf_path) -> str:
    """Tree URDF path relative to the simulator data folder, e.g. urdf/trees/envy/generated/LPy_envy_00008.urdf.

    Saved to HDF5 instead of the absolute path so data stays valid when PYBULLET_TREE_SIM_DATA moves.
    Older files hold absolute paths; those are cut at their 'urdf/' folder.
    """
    path = str(urdf_path)
    rel = os.path.relpath(path, DATA_PATH)
    if os.path.isabs(path) and not rel.startswith(".."):
        return rel
    idx = path.rfind("/urdf/")
    return path[idx + 1:] if idx >= 0 else path


def _waypoint_tree_signature(attrs) -> tuple[int, float] | None:
    """(tree_id, tree_scale) from HDF5 group attrs, or None if incomplete."""
    if attrs.get("tree_id") is None or attrs.get("tree_scale") is None:
        return None
    return (int(attrs["tree_id"]), float(attrs["tree_scale"]))


def apply_tree_setup_from_waypoints_hdf5(
    waypoints_path,
    *,
    config: dict | None = None,
    ref_group_key: str | None = None,
) -> str:
    """
    Patch CONFIG tree_setup from a SUCCESS group in the waypoints HDF5.

    Uses *ref_group_key* when set; otherwise the first SUCCESS group.
    Warns if SUCCESS groups disagree on tree_id / tree_scale.
    """
    cfg = config or CONFIG
    ts = cfg["tree_setup"]
    path = Path(waypoints_path)
    with h5py.File(path, "r") as f:
        success_keys = [
            k for k in f.keys()
            if int(f[k].attrs.get("fail_mode", -1)) == ResultMode.SUCCESS.value
        ]
        if not success_keys:
            raise ValueError(f"No SUCCESS groups in {path}")

        if ref_group_key is not None:
            if ref_group_key not in f:
                raise KeyError(f"group {ref_group_key!r} not in {path}")
            ref_key = ref_group_key
        else:
            ref_key = success_keys[0]

        ref_attrs = dict(f[ref_key].attrs)
        ref_sig = _waypoint_tree_signature(ref_attrs)
        if ref_sig is None:
            raise ValueError(
                f"Group {ref_key!r} missing tree_id/tree_scale — cannot auto-apply CONFIG"
            )

        for k in success_keys:
            sig = _waypoint_tree_signature(dict(f[k].attrs))
            if sig is not None and sig != ref_sig:
                log.warning(
                    "Waypoints HDF5 mixed tree setup: %s has tree_id=%s scale=%s "
                    "but ref %s has tree_id=%s scale=%s — using ref group only",
                    k, sig[0], sig[1], ref_key, ref_sig[0], ref_sig[1],
                )

    before = {
        "tree_id": int(ts["tree_id"]),
        "scale": float(ts["scale"]),
        "position": np.asarray(ts["position"], dtype=np.float64).copy(),
    }
    ts["tree_id"] = ref_sig[0]
    ts["scale"] = ref_sig[1]
    if ref_attrs.get("tree_pos") is not None:
        ts["position"] = np.asarray(ref_attrs["tree_pos"], dtype=np.float64).reshape(3)
    if ref_attrs.get("tree_orientation") is not None:
        ts["orientation"] = np.asarray(
            ref_attrs["tree_orientation"], dtype=np.float64
        ).reshape(4)

    log.info(
        "CONFIG tree_setup from HDF5 group %r: tree_id %s→%s  scale %.4f→%.4f  "
        "pos %s→%s  id_str=%s",
        ref_key,
        before["tree_id"],
        ts["tree_id"],
        before["scale"],
        ts["scale"],
        np.round(before["position"], 4).tolist(),
        np.round(np.asarray(ts["position"]), 4).tolist(),
        ref_attrs.get("tree_id_str", tree_id_str_from_config(cfg)),
    )
    return ref_key


def ensure_output_dirs(config: dict | None = None):
    """Create output/waypoints, output/agent_data and logs (relative to the workspace)."""
    cfg = config or CONFIG
    base_dir = Path(cfg['output']['base_dir'])
    waypoints_dir = base_dir / cfg['output']['waypoints_subdir']
    agent_data_dir = base_dir / cfg['output']['agent_data_subdir']
    logs_dir = Path("logs")
    for directory in (waypoints_dir, agent_data_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return waypoints_dir, agent_data_dir, logs_dir


def resolve_run_id(run_id, directory, should_exist=True):
    directory.mkdir(parents=True, exist_ok=True)
    if run_id:
        target_file = directory / f"{run_id}.hdf5"
        if should_exist and not target_file.exists():
            raise FileNotFoundError(f"Run ID '{run_id}' not found in {directory}.")
        return run_id, target_file
    existing_files = sorted(directory.glob("*.hdf5"))
    if not existing_files:
        raise FileNotFoundError(f"No run outputs found in {directory}.")
    latest_file = existing_files[-1]
    return latest_file.stem, latest_file


def setup_planning_functions(robot: Robot, 
                             tree_pb_id: int, 
                             planning_config: dict, 
                             visualization_config: dict,
                             pb_client_ref, 
                             tree_object_ref: Tree = None, 
                             camera_sensor=None,
                             metrics=None):
    """Joint-space planning functions for pybullet_planning.

    ``is_state_valid_fn(q)`` returns True when q is INVALID (unacceptable collision or failed
    visibility check). ``metrics`` (plan_waypoints.PlannerStats) counts rejections when given.
    """

    controllable_joints = robot.control_joint_idxs
    robot_id = robot.robot
    distance_fn = get_distance_fn(robot_id, controllable_joints)
    sample_fn = get_sample_fn(robot_id, controllable_joints)
    #extend_fn = get_extend_fn(robot_id, controllable_joints)
    resolutions = 0.05
    extend_fn = get_extend_fn(robot_id, controllable_joints, resolutions=resolutions)

    obstacle_dict_for_robot_check = {label: tree_pb_id for label in planning_config['collision_labels']}
    acceptable_labels = planning_config.get('acceptable_collision_labels', [])
    unacceptable_labels = planning_config.get('unacceptable_collision_labels', [])
    vis_check_config = planning_config.get('visibility_check', {})

    def is_state_valid_fn(q):
        current_q_robot = robot.get_joint_angles()
        robot.set_joint_angles_no_collision(q)
        
        # Force PyBullet to update collision detection (from old version)
        pb_client_ref.performCollisionDetection()
        
        # 1. --- Enhanced Collision Check with Fallback ---
        is_unacceptable_collision = False
        detailed_col_info = {}
        
        try:
            # Try advanced collision checking first
            if not hasattr(robot, 'check_collisions'):
                raise AttributeError("Robot object does not have 'check_collisions' method.")
                
            is_unacceptable_collision, detailed_col_info = robot.check_collisions(
                obstacle_dict_for_robot_check,
                tree_object_for_labeling=tree_object_ref,
                acceptable_labels=acceptable_labels,
                unacceptable_labels=unacceptable_labels)
            
            # Check if ANY collision occurred (acceptable or unacceptable)
            has_acceptable_collision = detailed_col_info.get("collisions_acceptable", False)
            has_unacceptable_collision = detailed_col_info.get("collisions_unacceptable", False)
            
            # CORRECTED: Only invalidate state for UNACCEPTABLE collisions
            if has_unacceptable_collision:
                col_body = (
                    "self"
                    if detailed_col_info.get("is_self_collision_unacceptable")
                    else "tree"
                )
                log.info(
                    "PLAN_COLLISION reject body=%s label=%s",
                    col_body,
                    detailed_col_info.get("collided_obstacle_label"),
                )
                log.warning("State INVALID: Unacceptable collision detected.")
                if metrics is not None:
                    if detailed_col_info.get('is_self_collision_unacceptable', False):
                        metrics.collision_self_rejections += 1
                    else:
                        metrics.collision_env_rejections += 1
                if visualization_config.get('visualize_actual_collision_points', False):
                    env_contact_point = detailed_col_info.get('contact_point_on_obstacle')
                    collided_label = detailed_col_info.get('collided_obstacle_label')
                    if env_contact_point is not None and collided_label is not None:
                        log.debug(f"Visualizing unacceptable ENV collision with determined label '{collided_label}' at {np.round(env_contact_point, 3)}")
                        draw_debug_sphere(pb_client_ref, env_contact_point, 
                                        visualization_config['collision_point_radius'], 
                                        visualization_config['collision_point_color_env'])
                    self_contact_point = detailed_col_info.get('self_collision_contact_pos')
                    if self_contact_point is not None and detailed_col_info.get('is_self_collision_unacceptable', False):
                        log.debug(f"Visualizing unacceptable SELF collision at {np.round(self_contact_point, 3)}")
                        draw_debug_sphere(pb_client_ref, self_contact_point, 
                                        visualization_config['collision_point_radius'], 
                                        visualization_config['collision_point_color_self'])
                robot.set_joint_angles_no_collision(current_q_robot)
                return True  # State is invalid
            
            # ACCEPTABLE collisions (like APPLE contact) are OK - continue to visibility check
            if has_acceptable_collision:
                log.debug(f"State has acceptable collision with: {detailed_col_info.get('collided_obstacle_label', 'UNKNOWN')}")
                
        except AttributeError as ae:
            # FALLBACK: Use basic PyBullet collision detection (from old version)
            log.warning(f"{ae}. Using generic collision check fallback for environment.")
            
            # Check environment collisions
            contacts_env = pb_client_ref.getContactPoints(bodyA=robot.robot, bodyB=tree_pb_id)
            if contacts_env:
                for contact in contacts_env:
                    if contact[8] < -0.001:  # Penetration threshold from old version
                        is_unacceptable_collision = True
                        log.info("PLAN_COLLISION reject body=tree label=generic_fallback")
                        if visualization_config.get('visualize_actual_collision_points', False):
                            cp = contact[6]  # Contact point on bodyB (tree)
                            log.debug(f"Visualizing generic ENV collision at {np.round(cp, 3)} (due to AttributeError in robot.check_collisions)")
                            draw_debug_sphere(pb_client_ref, cp,
                                            visualization_config['collision_point_radius'],
                                            visualization_config['collision_point_color_env'])
                        break
            
            # Check self-collisions if no environment collision found
            if not is_unacceptable_collision:
                self_contacts = pb_client_ref.getContactPoints(bodyA=robot.robot, bodyB=robot.robot)
                if self_contacts:
                    for sc in self_contacts:
                        if sc[3] != sc[4] and sc[8] < -0.015:  # Different links, deeper penetration threshold
                            is_unacceptable_collision = True
                            if metrics is not None:
                                metrics.collision_self_rejections += 1
                            if visualization_config.get('visualize_actual_collision_points', False):
                                cp_self = sc[5]  # Contact point on bodyA (robot)
                                log.debug(f"Visualizing generic SELF collision at {np.round(cp_self, 3)} (due to AttributeError in robot.check_collisions)")
                                draw_debug_sphere(pb_client_ref, cp_self,
                                                visualization_config['collision_point_radius'],
                                                visualization_config['collision_point_color_self'])
                            break
            
            # If fallback found collision, return invalid
            if is_unacceptable_collision:
                robot.set_joint_angles_no_collision(current_q_robot)
                return True
                
        except Exception as e:
            # Handle any other errors gracefully
            log.error(f"Error during collision check for q={np.round(q, 3)}: {e}", exc_info=True)
            robot.set_joint_angles_no_collision(current_q_robot)
            return True  # Assume invalid state on error
        
        # 2. --- Camera-Based Visibility Check ---
        if vis_check_config.get('enable', False) and camera_sensor is not None:
            try:
                view_matrix = robot.get_view_mat_at_curr_pose(camera_sensor)
                rgb_image_float, _ = robot.get_rgbd_at_cur_pose(camera=camera_sensor, type="sensor", view_matrix=view_matrix)

                rgb_image_uint8 = (np.array(rgb_image_float) * 255).astype(np.uint8)

                # Support multiple target colors (e.g., red for goals, blue for active apples)
                target_colors_rgb = vis_check_config.get('target_colors_rgb', [vis_check_config.get('target_color_rgb', (255, 0, 0))])
                # Backward compatibility: if only single target_color_rgb is provided, convert to list
                if 'target_color_rgb' in vis_check_config and 'target_colors_rgb' not in vis_check_config:
                    target_colors_rgb = [vis_check_config['target_color_rgb']]
                
                tolerance = np.array(vis_check_config['color_tolerance'])
                
                # Create combined mask for all target colors
                combined_mask = np.zeros(rgb_image_uint8.shape[:2], dtype=np.uint8)
                for target_rgb in target_colors_rgb:
                    target_rgb_array = np.array(target_rgb)
                    lower_bound = np.clip(target_rgb_array - tolerance, 0, 255)
                    upper_bound = np.clip(target_rgb_array + tolerance, 0, 255)
                    color_mask = cv2.inRange(rgb_image_uint8, lower_bound, upper_bound)
                    combined_mask = cv2.bitwise_or(combined_mask, color_mask)
                
                visible_pixels = cv2.countNonZero(combined_mask)

         #       log.debug(f"Visibility check: Found {visible_pixels} target pixels.")

                if visible_pixels < vis_check_config.get('min_visible_pixels', 2000):
                    log.warning(f"State INVALID: Visibility check failed. Found {visible_pixels} pixels, need {vis_check_config.get('min_visible_pixels', 2000)}.")
                    if metrics is not None:
                        metrics.visibility_rejections += 1
                    robot.set_joint_angles_no_collision(current_q_robot)
                    return True  # State is invalid due to poor visibility
                    
            except Exception as e:
                log.error(f"Error during visibility check for q={np.round(q, 3)}: {e}", exc_info=True)
                # Continue without visibility check rather than failing completely
                log.warning("Continuing without visibility check due to error.")
        
        # 3. --- If all checks pass, the state is valid ---
        robot.set_joint_angles_no_collision(current_q_robot)
        return False  # State is VALID

    return distance_fn, sample_fn, extend_fn, is_state_valid_fn


def task_space_distance(robot, q1, q2):
    original_q = robot.get_joint_angles()
    robot.set_joint_angles_no_collision(q1)
    pos1, _ = robot.get_current_pose(robot.tool0_link_idx)
    robot.set_joint_angles_no_collision(q2)
    pos2, _ = robot.get_current_pose(robot.tool0_link_idx)
    robot.set_joint_angles_no_collision(original_q)
    return np.linalg.norm(np.array(pos1) - np.array(pos2))
def get_refine_fn_task(robot, joints, task_threshold):
    difference_fn = get_difference_fn(robot.robot, joints)
    def fn(q1, q2):
        num_steps = int(np.ceil(task_space_distance(robot, q1, q2) / task_threshold))
        if num_steps == 0:
            yield q2
            return
        for i in range(num_steps + 1):
            alpha = i / num_steps
            yield tuple(np.array(q1) + alpha * np.array(difference_fn(q2, q1)))
    return fn
def refine_waypoints_in_task_space(
    robot,
    joints,
    waypoints,
    task_threshold,
    collision_fn=None,
    extend_fn=None,
):
    """Insert intermediate joint poses along each edge (task-space spacing heuristic).

    If ``collision_fn(q)`` is provided, it must match ``is_state_valid_fn`` semantics:
    return **True** if the state is **invalid** (collision / visibility fail).

    Linearly interpolated refinement samples are not guaranteed collision-free even
    when endpoints are valid. If any intermediate sample is invalid, this function
    falls back to ``extend_fn(q1, q2)`` (collision-checked dense joint extension
    from pybullet_planning) for that edge. If the fallback still contains invalid
    states, raises ``RuntimeError``.
    """
    if not waypoints:
        return []
    refine_fn = get_refine_fn_task(robot, joints, task_threshold)
    refined_path = [waypoints[0]]
    for edge_idx, (v1, v2) in enumerate(zip(waypoints, waypoints[1:])):
        candidates = list(refine_fn(v1, v2))[1:]
        use_candidates = candidates
        if collision_fn is not None and candidates:
            invalid_flags = [bool(collision_fn(q)) for q in candidates]
            if any(invalid_flags):
                log.warning(
                    "refine_waypoints_in_task_space: edge %s — %d/%d interpolated states invalid "
                    "(linear task-space refine) — using extend_fn fallback",
                    edge_idx,
                    sum(invalid_flags),
                    len(candidates),
                )
                if extend_fn is None:
                    raise RuntimeError(
                        "Task-space refinement hit invalid interpolated states but extend_fn is None"
                    )
                use_candidates = list(extend_fn(tuple(v1), tuple(v2)))[1:]
                if not use_candidates:
                    use_candidates = [tuple(v2)]
                if any(collision_fn(q) for q in use_candidates):
                    raise RuntimeError(
                        "extend_fn fallback still produced invalid states on edge %s" % edge_idx
                    )
        elif collision_fn is not None and not candidates:
            pass
        refined_path.extend(use_candidates)
    return refined_path

def shortcut_and_refine_path(robot, path, extend_fn, collision_fn, task_threshold, enable_smoothing=True):
    if len(path) < 2: 
        log.warning("Path is too short to refine, returning as is.")
        return path
        
    log.info(f"Refining path. Initial waypoints: {len(path)}")
    
    if enable_smoothing:
        path = smooth_path(path, extend_fn, collision_fn, iterations=20)
        log.info(f"Waypoints after shortcutting: {len(path)}")
    else:
        log.info("Skipping path smoothing per configuration.")
    
    # Task-space refinement (collision-check intermediates; extend_fn fallback on failure)
    path = refine_waypoints_in_task_space(
        robot,
        robot.control_joint_idxs,
        path,
        task_threshold,
        collision_fn=collision_fn,
        extend_fn=extend_fn,
    )
    log.info(f"Waypoints after task-space refinement: {len(path)}")
    
    return path
