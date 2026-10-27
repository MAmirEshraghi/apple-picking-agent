# apple-picking-agent

Expert data generation and policy learning (behaviour cloning, PPO+BC) for a UR5 arm reaching apples on
simulated trees. The robot, tree and sensor simulation comes from the separate simulator package
`pybullet_tree_sim`, developed in the fork
[MAmirEshraghi/UR5-ApplePathPlanning-Sim-PyBullet](https://github.com/MAmirEshraghi/UR5-ApplePathPlanning-Sim-PyBullet).

**Pipeline:** RRT planner → waypoints → closed-loop expert rollouts → validated HDF5 dataset →
BC / PPO+BC training → evaluation in simulation.

---

## 1. Requirements
- Linux, Python 3.10, a CUDA GPU for training (CPU works for data generation, validation and small BC runs).
- The simulator package `pybullet_tree_sim` and its data: tree meshes (~13 GB, from its `mesh_downloader`)
  and textures.
- ROS 2 is **not** needed. (Only regenerating the simulator's robot URDF from xacro needs it.)

## 2. Installation

```bash
git clone https://github.com/MAmirEshraghi/apple-picking-agent.git
cd apple-picking-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # installs this package and the simulator pinned in pyproject.toml
```

**Developing the simulator at the same time.** Clone the fork next to this repo and install both
editable. The simulator is then used from your local clone:

```bash
git clone -b refactor/prepare-split https://github.com/MAmirEshraghi/UR5-ApplePathPlanning-Sim-PyBullet.git
pip install -e UR5-ApplePathPlanning-Sim-PyBullet
pip install -e apple-picking-agent --no-deps
```

Without installing anything, `export PYTHONPATH=<path>/apple-picking-agent:<path>/UR5-ApplePathPlanning-Sim-PyBullet`
works too.

### Simulator data (meshes, textures)
Meshes and some textures are not in git. Either:
- download them into the simulator package with its `mesh_downloader` command (editable installs), or
- point `PYBULLET_TREE_SIM_DATA` at a folder with the simulator's data layout, e.g. an existing
  `UR5-ApplePathPlanning-Sim-PyBullet/pybullet_tree_sim` that already has them:

```bash
export PYBULLET_TREE_SIM_DATA=/path/to/UR5-ApplePathPlanning-Sim-PyBullet/pybullet_tree_sim
# must contain meshes/, textures/, pkl/ and urdf/ (generated tree URDFs and caches are written there)
```

## 3. Workspace
All commands run from a **workspace folder**. Paths are relative to it:

```
<workspace>/
  output/waypoints/<run_id>.hdf5    planner output (stage 1)
  output/agent_data/<run_id>.hdf5   expert datasets (schema v2)
  logs/                             text logs, checkpoints (logs/<run_name>/model_<N>_steps.zip), eval JSON
  runs/                             TensorBoard
```

The repo root can serve as the workspace (`output/`, `logs/` and `runs/` are git-ignored). Existing
data can be reused with a symlink, e.g. `ln -s /path/to/old/output output`.

## 4. Usage

### 4.1 Plan waypoints (stage 1)
```bash
python -m apple_agent.planning.plan_waypoints --stage planner --tree-id 8 --run_id tree08 --no-render
```
This writes collision-free RRT paths (`planning.planner_type`, RRT* by default) to the pre-grasp pose of each
apple into `output/waypoints/tree08.hdf5`. Without `--tree-id` it plans every exported tree (`tree00`, `tree01`, …).
`--stage visualize --run_id tree08` replays saved paths in the GUI. Logs: `logs/plan_waypoints/` and
`logs/path_planner_result/<run_id>.log`.

### 4.2 Generate expert data (stage 2)
```bash
python -m apple_agent.data.generate_expert_data --run-id tree08 --out-id tree08_v2
#   --path-rank 1        only the best path per apple
#   --save-only-success  drop failed rollouts (otherwise kept for RL; BC uses successes only)
#   --max-groups N       quick test on N groups
```
- Each path is executed with a closed-loop Jacobian controller in `ApplePickingEnv` (headless, same renderer
  as training).
- Each trajectory is saved once per frame in **agent-data schema v2**: RGB, metric depth (mm), apple mask,
  proprioception, `pregrasp_in_tool` target, actions, rewards and terminal/truncated flags.
- The controller settings (`--step-scale` 1.4, `--action-smoothing-alpha`, `--record-joint-dq-cap`, …) are saved
  with each trajectory as `gen_*` attributes. Log: `logs/generate_expert_data/`.
- Merge several trees: `python -m apple_agent.data.merge_agent_data --range 0-9 --prefix tree --output trees00to09_v2`
  (inputs need the same naming, e.g. `tree00_v2`: use `--inputs` for explicit ids).

### 4.3 Validate before training
```bash
python -m apple_agent.data.validate_agent_data --run-id tree08_v2 --policy-smoke --replay 2
```
Checks schema, depth, frame motion, targets and success, builds the policy for every input setting
(`--policy-smoke`) and replays trajectories in the simulator (`--replay N`). Exit code 1 on failure.
Log: `logs/validate_agent_data/`.

### 4.4 Train
Behaviour cloning only (no simulation; fast):
```bash
python -m apple_agent.rl.train_apple_picking_expert --run_id tree08_v2 --run_name dm_bc \
    --image_channels depth,mask --bc_only --bc_warmup_steps 0 \
    --total_timesteps 20000 --checkpoint_freq 5000 --no-memory_debug
```
PPO+BC fine-tuning from a BC checkpoint (GPU recommended):
```bash
python -m apple_agent.rl.train_apple_picking_expert --run_id tree08_v2 --run_name dm_ppo \
    --image_channels depth,mask --load_checkpoint logs/dm_bc/model_20000_steps.zip \
    --bc_warmup_steps 0 --bc_coeff 1.0 --total_timesteps 50000 --checkpoint_freq 5000 --no-memory_debug
```
- `--image_channels`: any of `flow`, `depth`, `mask`, or `none` (proprioception only). It must match the
  checkpoint when resuming. `depth,mask` is the recommended setting; `flow` tends to copy the previous
  motion under BC.
- `--total_timesteps` counts environment steps (when resuming: steps added). One PPO update every 512 steps.
- Checkpoints go to `logs/<run_name>/` and TensorBoard to `runs/<run_name>`.
- Training needs about 2–3 GB of RAM for depth+mask buffers (more with flow).
- **Settings** (flow size, RAFT iterations, cache, PPO and network hyper-parameters, image size) are in
  `TRAIN_CONFIG` in `apple_agent/config.py`. `--flow_size` / `--flow_updates` override the flow ones for one run;
  they are saved in the checkpoint, so eval and resumed runs use the same flow.
- **Flow speed:** RAFT runs at `flow.size` (128 by default, its minimum) and the flow is resized to the CNN size.
  Precompute it once per dataset with `python -m apple_agent.data.cache_flow --run-id tree08_rank1_v2`
  (writes `output/flow_cache/`, checks the result against live RAFT); training then only runs RAFT on frames
  it has not seen. Entries are keyed by image content, so a stale cache can only miss. Run it again with the
  same `--flow_size` / `--flow_updates` you train with. Training prints `[train:flow] cache hits=… misses=…`.

### 4.5 Evaluate and inspect
```bash
python -m apple_agent.rl.eval_apple_picking_expert --run_id tree08_v2 \
    --checkpoint logs/dm_bc/model_20000_steps.zip --eval_all --max_steps 150 --output logs/eval/dm_bc_20k.json
python -m apple_agent.rl.replay_expert_actions --run_id tree08_v2 --n_episodes 5    # expert actions open-loop
python -m apple_agent.rl.debug.run_all_step0 --run_id tree08_v2 --checkpoint <zip> # step-0 diagnostics
```
Eval reads the input channels from the checkpoint. It reports success, collision and timeout rates, and the
final distance to the target. `--max_steps 150` is about 1.3× the longest expert trajectory.

## 5. Package layout
| Path | Contents |
|---|---|
| `apple_agent/config.py` | Shared `CONFIG`: simulation, robot, tree, planning, generator and output settings |
| `apple_agent/envs/` | `ApplePickingEnv` (Gymnasium) and tool/world frame conversions |
| `apple_agent/planning/` | `plan_waypoints` (RRT planner, stage 1), `common` (collision checks, path refinement, waypoints-HDF5 helpers shared with the generator and training), goal camera filter |
| `apple_agent/data/` | `agent_data` (schema v2 I/O), `obs_spec`, `generate_expert_data`, `validate_agent_data`, `merge_agent_data` |
| `apple_agent/rl/` | Train / eval / replay entry points; `debug/` diagnostics |
| `apple_agent/apple_picking_sb3/` | SB3 PPO+BC with LSTM policy (`algo/PPOLSTMAE`), encoder, optical flow, callbacks (`apple_gym/`) |
| `apple_agent/legacy/` | Old scripts and unused vendored modules, kept for reference only (not maintained) |

## 6. Notes
- **Observation:** proprioception, apple centre and pre-grasp target in the tool frame, plus the chosen image
  channels from the wrist RGB-D camera. **Action:** 6-D tool-frame velocity in [-1, 1].
- **Checkpoints from earlier versions** of this package load unchanged: an import alias maps old module names.
- **Simulator settings this package relies on:** `PYBULLET_TREE_SIM_DATA` (data root), and the simulator's
  portable `robot.urdf` with bundled meshes.
- **Results so far** (tree 08, 10 demos, BC 20k steps, 150-step eval): depth+mask 50% success / 0% collisions;
  proprio-only 60% / 20%; flow+mask 0% / 60%.

---

# OLD

# Perception-Guided Motion Planning and Deep Reinforcement Learning for Robotic Fruit Harvesting

### 🎥 Demo Video
Click below to watch the demo:

[![Watch the video](https://img.youtube.com/vi/AE1qpv3drOY/0.jpg)](https://youtu.be/AE1qpv3drOY)

## (1) Simulation Environment Development: 
Extended the PyBullet-based tree simulation to support a 6-DoF UR5 manipulator with an end-effector gripper, RGB-D camera sensor, and a detailed tree model with apples and leaves, enabling realistic orchard manipulation scenarios. 

## (2) Perception-Integrated Path Planning: 
Developed a collision-free RRT-Connect planner with a novel camera-based visibility validation system to ensure the end-effector maintains visual contact with targets throughout the trajectory and to ensure natural arm movement for the imitation learning task. The planner incorporates collision checking, multiple inverse kinematics candidate generation with orientation perturbation, and task-space path refinement for smooth trajectories. 
## Quantitative results (minimal) 
Path planning with RRT-connect and having a visibility check. These results for a few samples show that using the visibility check leads to smooth and shorter paths.

<img width="1600" height="1067" alt="Code_Generated_Image" src="https://github.com/user-attachments/assets/e6b586bb-6699-4a35-af90-d80e48d677c7" />
<img width="1600" height="533" alt="Code_Generated_Image (3)" src="https://github.com/user-attachments/assets/0a9441a6-18b1-4d79-bae9-27bc6700f19e" />
<img width="1600" height="533" alt="Code_Generated_Image (2)" src="https://github.com/user-attachments/assets/55efbf3d-d26a-4d99-a6df-313dfa7a7e4b" />
<img width="1400" height="800" alt="Code_Generated_Image (1)" src="https://github.com/user-attachments/assets/c52dea6d-ebed-46d0-9e2b-fc212a6277ca" />

## (3) Expert Data Generation Pipeline: 
Implemented a two-stage pipeline (planner and generator) that converts joint-space waypoints to end-effector velocity commands using Jacobian-based transformations, executes closed-loop trajectories with feedback, and stores multi-modal observations (RGB images, point masks, joint states, goal positions) with optical flow in HDF5 format for efficient dataset management. 


## (4) Deep Reinforcement Learning Training: 
Designed and implemented a RecurrentPPOAEWithExpert architecture combining Proximal Policy Optimization (PPO) with Long Short-Term Memory (LSTM) networks for temporal reasoning, integrated with online behavioral cloning (BC) using offline expert demonstrations. The model employs a convolutional encoder for RGB feature extraction, supports optical flow for motion understanding, uses a dual-LSTM architecture (actor and critic), and includes a custom imitation learning callback that dynamically reconfigures scene conditions from expert trajectories during training. 
### RL Agent training results (minimal)
Trained only on 4 trajectories.
<img width="1773" height="925" alt="image" src="https://github.com/user-attachments/assets/a349a3c4-370a-4f58-99cc-1edd840a6f18" />
<img width="1765" height="973" alt="image" src="https://github.com/user-attachments/assets/9fbf6f2b-6ba7-4dc6-8105-bfbab07a4b8e" />



### Demo (YouTube) [7/10/2025]:
[![Watch the video](https://img.youtube.com/vi/L9cdALAOvzs/maxresdefault.jpg)](https://www.youtube.com/watch?v=L9cdALAOvzs)





## Script Description: 

### 1: ```apple_agent/planning/plan_waypoints.py``` (formerly run_apple_data_generator.py)
This is the main executable script for the data generation pipeline. It uses command-line arguments to run different stages of the process, including:

- Planner (Stage 1): Loads the simulation, finds apples, and uses RRT-Connect to plan collision-free paths to them, saving the paths to ```rrt_paths.hdf5```.

- Generator (Stage 2): ```apple_agent/data/generate_expert_data.py``` (see Stage 2 below); the old `--stage generator` was removed.

- Visualization: Loads and replays the saved RRT paths in the PyBullet GUI for visual inspection.

#### 2: ```apple_agent/envs/apple_picking_env.py```: 
This script defines the core simulation environment, ApplePickingEnv, using the gymnasium (formerly Gym) API. It manages the PyBullet physics, loads the robot and tree models, provides sensor data (like camera images and joint states), and calculates rewards. It's the "world" in which the robot operates and is essential for both data generation and (later) RL agent training.


#### 3: ```apple_agent/envs/utils_conversions.py```: 

This is a utility script containing mathematical helper functions for coordinate frame transformations. Its primary functions, convert_local_action_to_global and convert_global_action_to_local, are used to translate velocity commands between the robot's local end-effector frame (e.g., "move 0.1 m/s along the gripper's 'forward' direction") and the simulation's global world frame.

#### 4: ```pybullet_tree_sim/tree.py``` (simulator dependency): 
This script defines the ```Tree``` class used in the simulation. It is responsible for loading the tree's URDF (for physics) and its high-resolution ```.obj``` mesh (for geometry). Its most important function is to process a labeled mesh file where vertex colors are used to identify tree parts. It contains the logic to parse these colors and generate a list of 3D coordinates for each individual apple's "centroid" (```get_apple_centroids```), which are then used as goals for the path planner.

#### 5: ```pybullet_tree_sim/robot.py``` (simulator dependency): 
This script defines the core ```Robot``` class. It is responsible for loading the robot arm model (from ```.xacro``` and config files) into the simulation. It provides the essential API for robot control, such as setting joint velocities (```set_joint_velocities```), calculating inverse kinematics (```calculate_ik```), and getting sensor data (like camera images from ``get_rgbd_at_cur_pose``). It also contains critical modified functions for the project, including a sophisticated ```check_collisions``` method that uses a K-D tree to identify the label of a collided point (e.g., "TRUNK" vs. "APPLE").

## How to Use ```plan_waypoints.py```:

### Stage 1: Run the Path Planner 
This stage loads the environment and uses the RRT-Connect algorithm to find collision-free joint-space paths from the robot's start position to a pre-grasp pose near each apple. These paths are saved to ```rrt_paths.hdf5```

```
python -m apple_agent.planning.plan_waypoints --stage planner
```
### Stage 2: Run the Data Generator 
Loads the planner waypoints (`output/waypoints/<run_id>.hdf5`), refines them and executes them with a closed-loop Jacobian controller in `ApplePickingEnv` (DIRECT mode, same renderer as training). Each trajectory is saved in agent-data schema v2 (`apple_agent/data/agent_data.py`): every frame once (rgb, depth in mm, apple mask, proprio, `pregrasp_in_tool`), plus actions, rewards and env terminal/truncated flags. Then validate the file before training:

```
python -m apple_agent.data.generate_expert_data --run-id tree08 --path-rank 1 --out-id tree08_rank1_v2 --save-only-success
python -m apple_agent.data.validate_agent_data --run-id tree08_rank1_v2 --policy-smoke --replay 2
```

Training picks the policy's image inputs from the same data: `python -m apple_agent.rl.train_apple_picking_expert --run_id tree08_rank1_v2 --image_channels depth,mask` (any of `flow`, `depth`, `mask`, or `none` for proprio only).

## Utility & Debugging Commands

### 1: Visualize Saved Paths 
If you want to visually inspect the paths generated by the planner stage without generating new agent data, use this command. It loads rrt_paths.hdf5 and plays back the robot's movement in the GUI.
```
python -m apple_agent.planning.plan_waypoints --visualize_planning
```
### 2: Run Coordinate Frame Axis Test 
This is a debugging tool to check that the robot's coordinate frames are set up correctly. It loads the robot and commands a simple movement along its local X-axis (the red line). This helps verify that "move forward" in the tool's frame translates to the correct motion in the simulation.
```
python -m apple_agent.planning.plan_waypoints --stage axis_test
```

## Setup and running

This package (`apple_agent`) runs on the simulator package `pybullet_tree_sim`, developed in the fork
[MAmirEshraghi/UR5-ApplePathPlanning-Sim-PyBullet](https://github.com/MAmirEshraghi/UR5-ApplePathPlanning-Sim-PyBullet)
and pinned in `pyproject.toml`.

```bash
# install (venv activated)
pip install -e .                                         # installs the pinned simulator too
# or, when changing the simulator as well:
pip install -e ../UR5-ApplePathPlanning-Sim-PyBullet && pip install -e . --no-deps
```

The simulator's meshes and textures are not in git (meshes come from its `mesh_downloader`). If the simulator is
installed read-only (not editable), point `PYBULLET_TREE_SIM_DATA` at a folder with its data layout
(`meshes/`, `textures/`, `pkl/`, `urdf/`), e.g. an existing `pybullet_tree_sim/` directory.

```bash
# run from the workspace root (where output/, logs/ and runs/ are created)
python -m apple_agent.planning.plan_waypoints --stage planner      # RRT waypoints -> output/waypoints/
python -m apple_agent.data.generate_expert_data --run-id tree08 --out-id tree08_v2
python -m apple_agent.data.validate_agent_data --run-id tree08_v2 --policy-smoke --replay 2
python -m apple_agent.rl.train_apple_picking_expert --run_id tree08_v2 --run_name <name> --image_channels depth,mask
python -m apple_agent.rl.eval_apple_picking_expert --run_id tree08_v2 --checkpoint logs/<name>/model_20000_steps.zip --eval_all --max_steps 150
```

### Package layout
- `apple_agent/config.py`: shared `CONFIG` (simulation, robot, tree, planning, generator, output)
- `apple_agent/envs/`: `ApplePickingEnv` and frame conversions
- `apple_agent/planning/`: RRT-Connect planner (stage 1) and goal camera filter
- `apple_agent/data/`: agent-data schema v2 I/O, observation spec, expert generator, validator, merge tool
- `apple_agent/rl/`: train / eval / replay entry points and debug diagnostics
- `apple_agent/apple_picking_sb3/`: vendored SB3 algorithms (PPO+BC with LSTM) and helpers
- `apple_agent/legacy/`: old playground and archive scripts, and unused vendored modules (`unused_apple_picking_sb3/`, see its `MOVED.md`); kept for reference only
