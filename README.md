# ACT_adapted

A fork of [ACT (Action Chunking with Transformers)](https://tonyzhaozh.github.io/aloha/) extended for master's thesis research. The two core additions relative to the original ACT repo are:

1. **Joint velocity (`qvel`) as an optional observation input** — the policy encoder receives both joint positions and joint velocities, giving the model direct access to the robot's dynamic state.
2. **Temporal context window** — instead of a single-timestep observation, the policy receives a window of `context_length` consecutive timesteps of state, velocity, and camera images, letting the model reason over recent history before predicting an action chunk.

Both features are controlled entirely through command-line flags and flow end-to-end from data loading through the model forward pass.

---

## What changed vs. original ACT

### `--velocity_control` flag — joint velocity in the observation

When `--velocity_control` is passed, every stage of the pipeline loads and processes `joint_velocity` alongside `joint_position`:

**Data loading (`utils.py` — `EpisodicDataset`):**
- Reads `states/articulation/robot/joint_velocity` from the HDF5 demo file for every sampled timestep window.
- Computes per-dimension mean and std for `qvel` in `get_norm_stats` and includes `qvel_mean` / `qvel_std` in the stats dict.
- Returns a 6-element tuple `(image_data, qpos_data, action_data, is_pad, qvel_data, subtask_label)` when velocity is enabled; a 5-element tuple otherwise.

**Training loop (`imitate_episodes.py` — `forward_pass`):**
- Detects whether the batch has 5 or 6 elements and routes accordingly, passing `qvel_data` to the policy when present.

**Policy forward pass (`policy.py` — `ACTPolicy.__call__`):**
- `qvel` is an optional keyword argument. When provided it is forwarded to the underlying DETR/ACT model so the encoder can consume it alongside `qpos`.

**Inference (`imitate_episodes.py` — `eval_bc`):**
- At every timestep, reads `obs['qvel']`, normalises it with `pre_process_qvel`, and passes it to `policy(qpos, curr_image, qvel=qvel)`.

### `--context_length` flag — multi-timestep observation context

`context_length` (default: `1`, i.e. original ACT behaviour) controls how many consecutive past timesteps are packed into a single observation.

**Data loading (`utils.py` — `EpisodicDataset`):**
- `qpos`, `qvel`, images, and `subtask_label` are all sliced as `[start_ts : start_ts + context_length]` instead of a single index, producing tensors of shape `(context_length, ...)`.
- The action chunk target starts at `start_ts + context_length - 1` so predictions are always relative to the *last* timestep in the context window.
- Padding of the action chunk at episode boundaries is handled correctly regardless of context length.

**Image tensor layout:**
- With context, images are shaped `(num_cameras, context_length, C, H, W)` — an extra temporal dimension compared to the original `(num_cameras, C, H, W)`. The einsum permutation in `__getitem__` is updated to `b k h w c -> b k c h w` to handle this.

**Training (`imitate_episodes.py`):**
- `context_length` is passed through to `load_data` → `EpisodicDataset` and also stored in `policy_config` so the model architecture knows what to expect.

**Command-line:**
- `--context_length <int>` (default `1`). Setting it to `1` reproduces original ACT exactly.

### `ChunkedEpisodicDataset.py`

A separate dataset class (`ChunkedEpisodicDataset`) that pre-segments episodes into fixed-length chunks at load time. This is an alternative to the random-window sampling in `EpisodicDataset` and is imported in `utils.py`.

### Subtask label encoding

Both dataset classes compute a 4-dimensional one-hot subtask label per timestep based on detected gripper state transitions (open/close events detected from `obs/gripper_pos`). These labels are passed to the policy as `subtask_label` and forwarded to the ACT model to provide a coarse task-progress signal.

### Beta-scheduled KL loss (`policy.py`)

The KL divergence term is annealed via a sigmoid schedule rather than a fixed `kl_weight`. `beta` starts at 0 and smoothly ramps to 1 over training, centred at epoch 2500:

```python
beta = min(1.0, 1.0 / (1.0 + np.exp(-(epoch - 2500) / 500)))
loss = l1 + beta * total_kld[0]
```

### Image augmentation (`imitate_episodes.py`)

When `--image_aug` is passed, colour jitter and Gaussian blur are applied to camera 0, and colour jitter + Gaussian blur + random crop to camera 1. Augmentation is applied stochastically (25% of batches) to avoid over-regularising. The augmentation function handles the full `(batch, num_cameras, context_length, C, H, W)` tensor layout introduced by the context window.

---

## Installation

```bash
conda create -n act_adapted python=3.8.10
conda activate act_adapted
pip install torchvision torch
pip install pyquaternion pyyaml rospkg pexpect
pip install mujoco==2.3.7 dm_control==1.0.14
pip install opencv-python matplotlib einops packaging h5py ipython tqdm
cd detr && pip install -e .
```

---

## Training

### Baseline (original ACT behaviour, no velocity, no context)

```bash
python imitate_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir <ckpt_dir> \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 8 \
  --dim_feedforward 3200 \
  --num_epochs 2000 \
  --lr 1e-5 \
  --seed 0
```

### With velocity observation

Add `--velocity_control`. The model receives `[qpos; qvel]` as proprioceptive input at each timestep.

```bash
python imitate_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir <ckpt_dir> \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 8 \
  --dim_feedforward 3200 \
  --num_epochs 2000 \
  --lr 1e-5 \
  --seed 0 \
  --velocity_control
```

### With context history

Set `--context_length` to the desired number of past timesteps. The observation fed to the policy at each step will be a window of that many consecutive frames.

```bash
python imitate_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir <ckpt_dir> \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 8 \
  --dim_feedforward 3200 \
  --num_epochs 2000 \
  --lr 1e-5 \
  --seed 0 \
  --context_length 5
```

### With both velocity and context

```bash
python imitate_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir <ckpt_dir> \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 8 \
  --dim_feedforward 3200 \
  --num_epochs 2000 \
  --lr 1e-5 \
  --seed 0 \
  --velocity_control \
  --context_length 5
```

### Additional flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--velocity_control` | bool | off | Include joint velocities in the observation |
| `--context_length` | int | 1 | Number of consecutive past timesteps in the observation |
| `--image_aug` | bool | off | Apply stochastic colour jitter, blur, and crop augmentation |
| `--temporal_agg` | bool | off | Use exponential temporal ensembling at inference |
| `--dropout` | float | 0.1 | Dropout rate in the transformer |
| `--pretrained` | str | None | Path to a checkpoint to initialise from |

---

## Evaluation
Best option is to use the play_cvae.py scrip from:

    https://github.com/mschmotzer/Isaaaclab_MT_Michael
using the command:

    ./isaaclab.sh -p scripts/imitation_learning/robomimic/^Cay_cvae.py --device cuda --task Isaac-Stack-Cube-Franka-IK-Abs-Transformer-RGB-v0 --num_rollouts
     100 --checkpoint /home/pdz/MasterThesis_MSC/Results_EUler/small_ws_simplified_400/policy_best.ckpt --enable_cameras --data_path /home/pdz/MasterThesis_MSC/Results_EUler/small_ws_400/
     --velocity_control --context_length 4 --seed 1 --horizon 400 --action_length 12 --success_output_file_dir data_storage/successtest_adaption1
---
"action_length" can have a mixmal length of action chunk size and if the numbers of actions ensembled in rollout.
All this is for cube stacking, if other task used use the standart act replay modaliity.

## HDF5 data format

The dataset loader expects a single HDF5 file with the following structure (as produced by the Isaac Lab data collection pipeline):

```
data/
  demo_0/
    actions                                      # (T, action_dim)
    states/articulation/robot/joint_position     # (T, n_joints+1)  — last col unused
    states/articulation/robot/joint_velocity     # (T, n_joints+1)  — last col unused
    obs/
      <camera_name>                              # (T, H, W, 3)  uint8 RGB
      gripper_pos                                # (T, 1)  used for subtask label
  demo_1/
    ...
```

`joint_position` and `joint_velocity` have their last column stripped (`[:, :-1]`) before use.

---

## Repo structure

```
ACT_adapted/
├── imitate_episodes.py      # Train / eval entry point
├── policy.py                # ACTPolicy wrapper — qvel + context forwarding, beta KL schedule
├── utils.py                 # EpisodicDataset with velocity + context, get_norm_stats, load_data
├── ChunkedEpisodicDataset.py # Alternative dataset: fixed chunk pre-segmentation
├── detr/                    # ACT model definition (DETR-based encoder-decoder)
├── constants.py             # Task configs, shared constants
├── sim_env.py               # MuJoCo joint-space simulation environments
├── ee_sim_env.py            # MuJoCo EE-space simulation environments
├── scripted_policy.py       # Scripted demo policies
├── record_sim_episodes.py   # Collect and save demonstration episodes
├── visualize_episodes.py    # Render saved episodes to video
└── policy_runner.py         # Standalone inference runner
```

---

## References

- [Original ACT repository](https://github.com/tonyzhaozh/act)
- [ACT project website](https://tonyzhaozh.github.io/aloha/)
- [ACT tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing)
