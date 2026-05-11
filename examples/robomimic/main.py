import collections
import dataclasses
import logging
import pathlib

import imageio
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from robomimic.config import config_factory
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from utils import get_language_instruction
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

ROBOMIMIC_DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclasses.dataclass
class Args:
    host: str = "10.79.12.252"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    dataset_path: str = ""
    task_name: str = "square"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "debug_images/robomimic/videos"
    seed: int = 7


def eval_robomimic(args: Args) -> None:
    np.random.seed(args.seed)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    max_steps = 400 if args.task_name == "square" else 400
    
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    env, task_description = _get_robomimic_env(args.dataset_path, args.seed)

    total_episodes, total_successes = 0, 0
    
    for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
        obs = env.reset()
        action_plan = collections.deque()
        t = 0
        frames = []
        success = False

        while t < max_steps + args.num_steps_wait:
            try:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(ROBOMIMIC_DUMMY_ACTION)
                    t += 1
                    continue

                # Process images
                img = obs["agentview_image"].transpose(1, 2, 0)
                img_uint8 = (img * 255).astype(np.uint8)
                img_resized = image_tools.resize_with_pad(img_uint8, args.resize_size, args.resize_size)

                wrist_img = obs["robot0_eye_in_hand_image"].transpose(1, 2, 0)
                wrist_img_resized = image_tools.resize_with_pad((wrist_img * 255).astype(np.uint8), args.resize_size, args.resize_size)

                # Save frame for video
                frames.append(img_uint8)

                # Get action
                if not action_plan:
                    element = {
                        "observation/image": img_resized,
                        "observation/wrist_image": wrist_img_resized,
                        "observation/state": np.concatenate((
                            obs["robot0_eef_pos"],
                            obs["robot0_eef_quat"],
                            obs["robot0_gripper_qpos"],
                        )),
                        "prompt": str(task_description),
                    }
                    action_chunk = client.infer(element)["actions"]
                    action_plan.extend(action_chunk[:args.replan_steps])

                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                
                # Check success using env.is_success() method (like in reference code)
                success = env.is_success()["task"]
                
                # Break if done or success
                if done or success:
                    if success:
                        total_successes += 1
                    break
                t += 1

            except Exception as e:
                logging.error(f"Exception: {e}")
                break

        total_episodes += 1

        # Save video
        suffix = "success" if success else "failure"
        filename = f"ep{episode_idx:03d}_{suffix}.mp4"
        imageio.mimwrite(pathlib.Path(args.video_out_path) / filename, frames, fps=10)
        
        logging.info(f"Episode {episode_idx}: {suffix} - Success rate: {total_successes/total_episodes*100:.1f}%")

    logging.info(f"Final success rate: {total_successes/total_episodes*100:.1f}% ({total_successes}/{total_episodes})")


def _get_robomimic_env(dataset_path, seed):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_name = env_meta['env_name']
    config = config_factory(algo_name='iql')
    config.observation.modalities.obs.rgb = [f"{camera_name}_image" for camera_name in env_meta['env_kwargs']['camera_names']]
    ObsUtils.initialize_obs_utils_with_config(config)
    env_kwargs = dict(
        env_meta=env_meta,
        env_name=env_name,
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )
    env = EnvUtils.create_env_from_metadata(**env_kwargs)
    return env, get_language_instruction(env_name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_robomimic)