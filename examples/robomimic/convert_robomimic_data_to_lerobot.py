"""
Minimal example script for converting a dataset to LeRobot format.

We use the Robomimic dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/robomimic/convert_robomimic_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/robomimic/convert_robomimic_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`
"""

import shutil
import h5py
import numpy as np

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from utils import get_language_instruction
import tyro

DEFAULT_ENV_NAME = "square"
DEFAULT_CAMERA_VIEW = "agentview"
DEFAULT_DATASET = "ph"
DEFAULT_DATASET_ROOT = "/iris/u/tiangao/projects/robomimic/robomimic/datasets"


def main(
    env_name: str = DEFAULT_ENV_NAME,
    camera_view: str = DEFAULT_CAMERA_VIEW,
    dataset: str = DEFAULT_DATASET,
    raw_dataset_path: str | None = None,
    push_to_hub: bool = False,
):
    repo_name = f"skybhh19/lerobot_robomimic_{env_name}_{dataset}_{camera_view}"
    if raw_dataset_path is None:
        raw_dataset_path = f"{DEFAULT_DATASET_ROOT}/{env_name}/{dataset}/image.hdf5"

    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    hdf5_file = h5py.File(raw_dataset_path, "r")
    demo_keys = sorted([k for k in hdf5_file['data'].keys() if k.startswith('demo_')])
    first_demo = hdf5_file['data'][demo_keys[0]]
    action_dim = first_demo['actions'].shape[1]
    state_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
    state_dim = sum(first_demo['obs'][k].shape[1] for k in state_keys)
    image_key = f"{camera_view}_image"
    wrist_image_key = "robot0_eye_in_hand_image"
    image_shape = first_demo["obs"][image_key].shape[1:]
    wrist_image_shape = first_demo["obs"][wrist_image_key].shape[1:]

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=20, # TODO: check the fps for robomimic, (20Hz for robomimic? 10Hz for libero?)
        features={
            "image": { # agentview_image
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {# robot0_eye_in_hand_image
                "dtype": "image",
                "shape": wrist_image_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    # Loop over raw Robomimic datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    saved_demo_count = 0
    total_episode_length = 0
    for demo_key in demo_keys:
        demo = hdf5_file['data'][demo_key]
        traj_len = len(demo['actions'])
        for t in range(traj_len):
            state = np.concatenate([demo['obs'][k][t] for k in state_keys])
            lerobot_dataset.add_frame(
                {
                    "image": demo["obs"][image_key][t].astype(np.uint8),
                    "wrist_image": demo["obs"][wrist_image_key][t].astype(np.uint8),
                    "state": state.astype(np.float32),
                    "actions": demo["actions"][t].astype(np.float32),
                    "task": get_language_instruction(env_name),
                }
            )
        lerobot_dataset.save_episode()
        saved_demo_count += 1
        total_episode_length += traj_len
        print(f"Saved {demo_key}")

    hdf5_file.close()
    average_episode_length = total_episode_length / saved_demo_count if saved_demo_count else 0
    print(f"Saved {saved_demo_count} demos to {output_path}")
    print(f"Average episode length: {average_episode_length:.2f} frames")
    
    # Optionally push to Hub
    if push_to_hub:
        lerobot_dataset.push_to_hub(
            tags=["robomimic", "panda", "rlds"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
