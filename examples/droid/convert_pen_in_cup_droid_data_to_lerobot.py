"""Convert the locally collected pen-in-cup DROID dataset to LeRobot format.

This script is tailored to the data under /scr/jasonyan/droid_success.  Those
episodes already contain MP4 recordings and store the language task in the
trajectory HDF5 attrs, so they do not need the official DROID postprocess.py
pipeline or the public-DROID annotation JSON used by the generic example.

Usage:
    uv run examples/droid/convert_pen_in_cup_droid_data_to_lerobot.py

The resulting dataset is written under $LEROBOT_HOME/<repo_id>.
"""

from collections.abc import Iterator
from pathlib import Path
import shutil
from typing import Literal

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import tyro


IMAGE_SIZE = (320, 180)  # width, height; DROID LeRobot convention stores (180, 320, 3)
DEFAULT_TASK = "Pick the pen and put it in the cup"


def _resize_rgb(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = image_bgr[..., ::-1]
    image = Image.fromarray(image_rgb)
    return np.asarray(image.resize(IMAGE_SIZE, resample=Image.BICUBIC))


def _decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        return _decode_attr(value.reshape(-1)[0])
    return str(value)


def _episode_paths(data_dir: Path, max_episodes: int | None) -> list[Path]:
    paths = sorted(data_dir.glob("**/trajectory.h5"))
    if max_episodes is not None:
        paths = paths[:max_episodes]
    return paths


def _video_capture(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def _iter_episode_frames(
    episode_path: Path,
    *,
    exterior_camera_serial: str,
    wrist_camera_serial: str,
    action_space: Literal["joint_velocity", "joint_position"],
    drop_movement_disabled: bool,
) -> Iterator[dict]:
    mp4_dir = episode_path.parent / "recordings" / "MP4"
    exterior_video = mp4_dir / f"{exterior_camera_serial}.mp4"
    wrist_video = mp4_dir / f"{wrist_camera_serial}.mp4"
    if not exterior_video.exists():
        raise FileNotFoundError(f"Missing exterior camera MP4: {exterior_video}")
    if not wrist_video.exists():
        raise FileNotFoundError(f"Missing wrist camera MP4: {wrist_video}")

    with h5py.File(episode_path, "r") as trajectory:
        horizon = len(trajectory["observation/robot_state/joint_positions"])
        exterior_cap = _video_capture(exterior_video)
        wrist_cap = _video_capture(wrist_video)
        try:
            for step_idx in range(horizon):
                exterior_ok, exterior_frame = exterior_cap.read()
                wrist_ok, wrist_frame = wrist_cap.read()
                if not exterior_ok or exterior_frame is None:
                    raise RuntimeError(f"Failed to read {exterior_video} at frame {step_idx}")
                if not wrist_ok or wrist_frame is None:
                    raise RuntimeError(f"Failed to read {wrist_video} at frame {step_idx}")

                if drop_movement_disabled:
                    movement_enabled = trajectory["observation/controller_info/movement_enabled"][step_idx]
                    if not bool(movement_enabled):
                        continue

                joint_position = trajectory["observation/robot_state/joint_positions"][step_idx].astype(np.float32)
                gripper_position = np.asarray(
                    [trajectory["observation/robot_state/gripper_position"][step_idx]], dtype=np.float32
                )
                joint_action = trajectory[f"action/{action_space}"][step_idx].astype(np.float32)
                gripper_action = np.asarray([trajectory["action/gripper_position"][step_idx]], dtype=np.float32)

                exterior_image = _resize_rgb(exterior_frame)
                wrist_image = _resize_rgb(wrist_frame)
                missing_exterior_image = np.zeros_like(exterior_image)

                yield {
                    # Keep the DROID/LeRobot feature names even though this collected exterior view is the
                    # physical right-side camera. OpenPI's DROID transform consumes exterior_image_1_left.
                    "exterior_image_1_left": exterior_image,
                    # The collected dataset has one exterior camera. Keep the second expected feature present
                    # with a black placeholder; current OpenPI DROID transforms ignore image_2.
                    "exterior_image_2_left": missing_exterior_image,
                    "wrist_image_left": wrist_image,
                    "joint_position": joint_position,
                    "gripper_position": gripper_position,
                    "actions": np.concatenate([joint_action, gripper_action], dtype=np.float32),
                }
        finally:
            exterior_cap.release()
            wrist_cap.release()


def _task_for_episode(episode_path: Path, task_override: str | None) -> str:
    if task_override is not None:
        return task_override
    return DEFAULT_TASK


def main(
    data_dir: Path = Path("/scr/jasonyan/droid_success"),
    *,
    repo_id: str = "jasonyan/droid_pen_in_cup_success",
    exterior_camera_serial: str = "25916956",
    wrist_camera_serial: str = "18650758",
    action_space: Literal["joint_velocity", "joint_position"] = "joint_velocity",
    task: str | None = None,
    drop_movement_disabled: bool = False,
    max_episodes: int | None = None,
    overwrite: bool = True,
    push_to_hub: bool = False,
):
    """Convert collected DROID trajectories to the LeRobot schema expected by OpenPI."""

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to recreate it.")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=15,
        features={
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    episodes = _episode_paths(data_dir, max_episodes)
    if not episodes:
        raise FileNotFoundError(f"No trajectory.h5 files found under {data_dir}")
    print(f"Converting {len(episodes)} episodes to {output_path}")

    for episode_path in tqdm(episodes, desc="Converting episodes"):
        language_instruction = _task_for_episode(episode_path, task)
        num_frames = 0
        for frame in _iter_episode_frames(
            episode_path,
            exterior_camera_serial=exterior_camera_serial,
            wrist_camera_serial=wrist_camera_serial,
            action_space=action_space,
            drop_movement_disabled=drop_movement_disabled,
        ):
            dataset.add_frame({**frame, "task": language_instruction})
            num_frames += 1
        if num_frames == 0:
            raise RuntimeError(f"No frames were written for {episode_path}")
        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["droid", "panda", "pen-in-cup"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
