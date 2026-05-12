from __future__ import annotations

# ruff: noqa

"""Evaluate zero-shot and finetuned pi0-FAST-DROID policies on pen-in-cup.

This script is intended to run on the DROID control laptop. Start a policy
server on a GPU machine first, then run this client against that server.
"""

import argparse
import contextlib
import csv
import dataclasses
import datetime as _datetime
import faulthandler
import json
import signal
import time
from pathlib import Path

faulthandler.enable()

DROID_CONTROL_FREQUENCY = 15
DEFAULT_PROMPT = "Pick the pen and put it in the cup"
RESULT_FIELDS = [
    "policy_label",
    "trial_index",
    "prompt",
    "success",
    "duration_steps",
    "external_video_path",
    "wrist_video_path",
    "preview_path",
    "timestamp",
    "remote_host",
    "remote_port",
    "external_camera",
    "left_camera_id",
    "right_camera_id",
    "wrist_camera_id",
    "notes",
]


@dataclasses.dataclass
class Args:
    # Remote server parameters.
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000

    # Evaluation labels and output.
    policy_label: str = "finetuned"
    output_dir: Path = Path("results/pen_in_cup")
    num_trials: int = 20
    prompt: str = DEFAULT_PROMPT

    # Hardware parameters for the pen-in-cup collection setup.
    left_camera_id: str | None = None
    right_camera_id: str | None = "25916956"
    wrist_camera_id: str = "18650758"
    external_camera: str = "right"

    # Rollout parameters.
    max_timesteps: int = 600
    open_loop_horizon: int = 8


np = None
Image = None
tqdm = None
image_tools = None
websocket_client_policy = None


def _load_runtime_deps() -> None:
    global Image, image_tools, np, tqdm, websocket_client_policy

    if np is not None:
        return

    import numpy as _np
    from PIL import Image as _Image
    import tqdm as _tqdm

    from openpi_client import image_tools as _image_tools
    from openpi_client import websocket_client_policy as _websocket_client_policy

    np = _np
    Image = _Image
    tqdm = _tqdm
    image_tools = _image_tools
    websocket_client_policy = _websocket_client_policy


def _parse_args() -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-host", default=Args.remote_host)
    parser.add_argument("--remote-port", type=int, default=Args.remote_port)
    parser.add_argument("--policy-label", default=Args.policy_label)
    parser.add_argument("--output-dir", type=Path, default=Args.output_dir)
    parser.add_argument("--num-trials", type=int, default=Args.num_trials)
    parser.add_argument("--prompt", default=Args.prompt)
    parser.add_argument("--left-camera-id", default=Args.left_camera_id)
    parser.add_argument("--right-camera-id", default=Args.right_camera_id)
    parser.add_argument("--wrist-camera-id", default=Args.wrist_camera_id)
    parser.add_argument("--external-camera", choices=["left", "right"], default=Args.external_camera)
    parser.add_argument("--max-timesteps", type=int, default=Args.max_timesteps)
    parser.add_argument("--open-loop-horizon", type=int, default=Args.open_loop_horizon)
    return Args(**vars(parser.parse_args()))


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        del signum, frame
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def main(args: Args) -> None:
    _load_runtime_deps()

    if args.external_camera not in {"left", "right"}:
        raise ValueError(f"--external-camera must be 'left' or 'right', got {args.external_camera!r}")
    if args.open_loop_horizon <= 0:
        raise ValueError("--open-loop-horizon must be positive")
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive")

    from droid.robot_env import RobotEnv

    run_timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / args.policy_label / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "trials.csv"
    jsonl_path = run_dir / "trials.jsonl"

    print(f"Writing results to: {run_dir}")
    print(f"Policy label: {args.policy_label}")
    print(f"Prompt: {args.prompt!r}")

    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    print("Created the DROID env.")

    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    print(f"Connected to policy server at {args.remote_host}:{args.remote_port}.")

    _ensure_csv_header(csv_path)
    for trial_index in range(args.num_trials):
        if trial_index > 0:
            env.reset()
            input(f"Reset scene for trial {trial_index + 1}/{args.num_trials}, then press Enter to continue.")
        else:
            input(f"Set scene for trial 1/{args.num_trials}, then press Enter to start.")

        result = _run_trial(args, env, policy_client, run_dir, trial_index)
        _append_result(csv_path, jsonl_path, result)
        print(f"Recorded trial {trial_index + 1}/{args.num_trials}: success={result['success']}")

    print(f"Finished {args.num_trials} trials.")
    print(f"CSV: {csv_path}")
    print(f"JSONL: {jsonl_path}")


def _run_trial(args: Args, env, policy_client, run_dir: Path, trial_index: int) -> dict:
    trial_timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    external_video_path = run_dir / f"trial_{trial_index:03d}_{trial_timestamp}_external.mp4"
    wrist_video_path = run_dir / f"trial_{trial_index:03d}_{trial_timestamp}_wrist.mp4"
    preview_path = run_dir / f"trial_{trial_index:03d}_{trial_timestamp}_preview.jpg"

    actions_from_chunk_completed = 0
    pred_action_chunk = None
    external_video_frames = []
    wrist_video_frames = []
    duration_steps = 0

    bar = tqdm.tqdm(range(args.max_timesteps), desc=f"trial {trial_index + 1}")
    print("Running rollout. Press Ctrl+C to stop this rollout early.")
    for t_step in bar:
        start_time = time.time()
        try:
            curr_obs = _extract_observation(
                args,
                env.get_observation(),
                preview_path=preview_path if t_step == 0 else None,
            )
            external_video_frames.append(curr_obs[f"{args.external_camera}_image"])
            wrist_video_frames.append(curr_obs["wrist_image"])

            if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
                actions_from_chunk_completed = 0
                request_data = {
                    "observation/exterior_image_1_left": image_tools.resize_with_pad(
                        curr_obs[f"{args.external_camera}_image"], 224, 224
                    ),
                    "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                    "observation/joint_position": curr_obs["joint_position"],
                    "observation/gripper_position": curr_obs["gripper_position"],
                    "prompt": args.prompt,
                }

                with prevent_keyboard_interrupt():
                    pred_action_chunk = policy_client.infer(request_data)["actions"]
                if pred_action_chunk.shape != (10, 8):
                    raise ValueError(f"Expected action chunk shape (10, 8), got {pred_action_chunk.shape}")

            action = pred_action_chunk[actions_from_chunk_completed]
            actions_from_chunk_completed += 1

            gripper_action = np.ones((1,)) if action[-1].item() > 0.5 else np.zeros((1,))
            action = np.concatenate([action[:-1], gripper_action])
            action = np.clip(action, -1, 1)
            env.step(action)

            duration_steps = t_step + 1
            elapsed_time = time.time() - start_time
            if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
        except KeyboardInterrupt:
            duration_steps = t_step + 1
            break

    _write_video(external_video_frames, external_video_path)
    _write_video(wrist_video_frames, wrist_video_path)
    success = _prompt_success()
    notes = input("Optional notes for this trial: ")

    return {
        "policy_label": args.policy_label,
        "trial_index": trial_index,
        "prompt": args.prompt,
        "success": success,
        "duration_steps": duration_steps,
        "external_video_path": str(external_video_path),
        "wrist_video_path": str(wrist_video_path),
        "preview_path": str(preview_path),
        "timestamp": trial_timestamp,
        "remote_host": args.remote_host,
        "remote_port": args.remote_port,
        "external_camera": args.external_camera,
        "left_camera_id": args.left_camera_id or "",
        "right_camera_id": args.right_camera_id or "",
        "wrist_camera_id": args.wrist_camera_id,
        "notes": notes,
    }


def _extract_observation(args: Args, obs_dict, *, preview_path: Path | None = None):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None
    for key in image_observations:
        if args.left_camera_id is not None and args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id is not None and args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    external_image = left_image if args.external_camera == "left" else right_image
    external_camera_id = args.left_camera_id if args.external_camera == "left" else args.right_camera_id
    if external_camera_id is None:
        raise ValueError(f"Must provide {args.external_camera}_camera_id when --external-camera={args.external_camera}")
    if external_image is None:
        raise ValueError(f"Could not find selected {args.external_camera} external camera image for ID {external_camera_id}")
    if wrist_image is None:
        raise ValueError(f"Could not find wrist camera image for ID {args.wrist_camera_id}")

    if left_image is not None:
        left_image = left_image[..., :3][..., ::-1]
    if right_image is not None:
        right_image = right_image[..., :3][..., ::-1]
    external_image = external_image[..., :3][..., ::-1]
    wrist_image = wrist_image[..., :3][..., ::-1]

    robot_state = obs_dict["robot_state"]
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if preview_path is not None:
        preview_images = [img for img in (left_image, wrist_image, right_image) if img is not None]
        Image.fromarray(np.concatenate(preview_images, axis=1)).save(preview_path)

    obs = {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }
    obs[f"{args.external_camera}_image"] = external_image
    return obs


def _write_video(video_frames: list[np.ndarray], video_path: Path) -> None:
    if not video_frames:
        print(f"No video frames captured; skipping video write for {video_path}")
        return

    from moviepy.editor import ImageSequenceClip

    ImageSequenceClip(list(video_frames), fps=10).write_videofile(str(video_path), codec="libx264")


def _prompt_success() -> float:
    while True:
        raw = input("Did the rollout succeed? Enter y, n, or numeric score 0-100: ").strip().lower()
        if raw == "y":
            return 1.0
        if raw == "n":
            return 0.0
        try:
            value = float(raw)
        except ValueError:
            print(f"Could not parse success value: {raw!r}")
            continue
        if 0 <= value <= 1:
            return value
        if 1 < value <= 100:
            return value / 100
        print(f"Success must be y, n, or a number in [0, 100], got {value}")


def _ensure_csv_header(csv_path: Path) -> None:
    if csv_path.exists():
        return
    with csv_path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=RESULT_FIELDS).writeheader()


def _append_result(csv_path: Path, jsonl_path: Path, result: dict) -> None:
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writerow({field: result.get(field, "") for field in RESULT_FIELDS})
    with jsonl_path.open("a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main(_parse_args())
