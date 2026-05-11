# Fine-Tuning pi0-FAST-DROID on the Pen-in-Cup Dataset

This note describes the pipeline for the custom DROID pen-in-cup dataset at:

```bash
/scr/jasonyan/droid_success
```

The dataset contains 32 successful DROID trajectories. We use the small custom dataset path: convert the raw DROID recordings to LeRobot, then fine-tune `pi0_fast_droid`.

## 1. Convert to LeRobot

Run conversion with controller-disabled frames removed:

```bash
cd /iris/u/tiangao/projects/openpi

uv run examples/droid/convert_pen_in_cup_droid_data_to_lerobot.py \
  --data-dir /scr/jasonyan/droid_success \
  --repo-id skybhh19/droid_pen_in_cup_success \
  --drop-movement-disabled
```

The converted dataset is written under:

```bash
/iris/u/tiangao/lerobot_datasets/skybhh19/droid_pen_in_cup_success
```

Conversion details:

- Task prompt: `Pick the pen and put it in the cup`
- Exterior camera 1: `25916956`
- Wrist camera: `18650758`
- `exterior_image_2_left`: black placeholder image
- Actions: `joint_velocity + gripper_position`
- Filtering: removes only frames where `observation/controller_info/movement_enabled` is false

Do not run `compute_droid_nonidle_ranges.py` for this pipeline. That script creates an RLDS sampling filter for full-DROID TFDS/RLDS training; this custom dataset uses LeRobot.

## 2. Add Training Config

Add this config near the DROID fine-tuning configs in `src/openpi/training/config.py`:

```python
TrainConfig(
    name="pi0_fast_droid_pen_in_cup_finetune_validation_10k",
    model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, max_token_len=180),
    data=LeRobotDROIDDataConfig(
        repo_id="skybhh19/droid_pen_in_cup_success",
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_fast_droid/assets",
            asset_id="droid",
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_fast_droid/params"
    ),
    num_train_steps=10_000,
    batch_size=32,
    validation_split=0.05,
    validation_num_batches=10,
    save_interval=1_000,
    keep_period=2_000,
)
```

Use the existing `pi0_fast_droid` DROID normalization statistics. Do not compute fresh normalization statistics for the first run; this dataset is small and follows the DROID action space.

## 3. Train

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_fast_droid_pen_in_cup_finetune_validation_10k \
  --exp-name=pi0_fast_droid_pen_in_cup_finetune_validation_10k \
  --overwrite
```

Checkpoints will be saved under:

```bash
checkpoints/pi0_fast_droid_pen_in_cup_finetune_validation_10k/pi0_fast_droid_pen_in_cup_finetune_validation_10k/
```

## 4. Run Inference

Start the policy server on a machine with a GPU. Replace the final path component with the checkpoint step you want:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_fast_droid_pen_in_cup_finetune_validation_10k \
  --policy.dir=checkpoints/pi0_fast_droid_pen_in_cup_finetune_validation_10k/pi0_fast_droid_pen_in_cup_finetune_validation_10k/10000 \
  --port=8000
```

On the DROID control laptop, use the DROID client flow from `examples/droid/README.md`: install `openpi-client`, make sure `examples/droid/main.py` is available under the DROID scripts directory, and point it at the policy server.

For this dataset, use exterior camera 1 as the policy external camera:

```bash
python3 scripts/main.py \
  --remote_host=<server_ip> \
  --remote_port=8000 \
  --right_camera_id=25916956 \
  --wrist_camera_id=18650758 \
  --external_camera=right
```

The script will prompt for a language instruction. Use:

```text
Pick the pen and put it in the cup
```

The policy request sent by `main.py` has this mapping:

```python
{
    "observation/exterior_image_1_left": right_camera_25916956_image,
    "observation/wrist_image_left": wrist_camera_18650758_image,
    "observation/joint_position": joint_positions_7d,
    "observation/gripper_position": gripper_position_1d,
    "prompt": "Pick the pen and put it in the cup",
}
```

`exterior_image_2_left` is not needed by the OpenPI DROID inference transform.

Note: `examples/droid/main.py` only requires the selected external camera and the wrist camera. The unused external camera ID can be omitted.
