import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cosine

def verify_robomimic_processing(dataset_path, env):
    """Verify if 180° rotation is needed for RoboMimic images."""
    
    debug_dir = Path("debug_images")
    debug_dir.mkdir(exist_ok=True)
    
    # Load image from dataset
    with h5py.File(dataset_path, 'r') as f:
        dataset_img = f['data/demo_0/obs/agentview_image'][0]
        if dataset_img.shape[0] == 3:  # (C, H, W) -> (H, W, C)
            dataset_img = dataset_img.transpose(1, 2, 0)
    
    # Get image from environment
    obs = env.reset()
    env_img = obs["agentview_image"].transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    
    # Process: no rotation
    env_no_rot = (env_img * 255).astype(np.uint8)
    
    # Process: with 180° rotation
    env_rotated = (env_img[::-1, ::-1] * 255).astype(np.uint8)
    
    # Save comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(dataset_img)
    axes[0].set_title('Dataset (Training)')
    axes[0].axis('off')
    
    axes[1].imshow(env_no_rot)
    axes[1].set_title('Env (No Rotation)')
    axes[1].axis('off')
    
    axes[2].imshow(env_rotated)
    axes[2].set_title('Env (180° Rotation)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(debug_dir / "comparison.png", dpi=150)
    
    # Compute similarity
    sim_no_rot = cosine(dataset_img.flatten(), env_no_rot.flatten())
    sim_rot = cosine(dataset_img.flatten(), env_rotated.flatten())
    
    print(f"\nSimilarity (no rotation):   {sim_no_rot:.6f}")
    print(f"Similarity (180° rotation): {sim_rot:.6f}")
    
    needs_rotation = sim_rot < sim_no_rot
    print(f"\n{'✅ USE' if needs_rotation else '❌ DO NOT USE'} 180° rotation")
    print(f"See comparison at: {debug_dir}/comparison.png\n")
    
    return needs_rotation


# Usage in your script
if __name__ == "__main__":
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    
    from main import _get_robomimic_env
    
    dataset_path = "../aiorl/robomimic/datasets/square_ph/image_v141_n30.hdf5"
    env, _ = _get_robomimic_env(dataset_path, seed=7)
    
    needs_rotation = verify_robomimic_processing(dataset_path, env)