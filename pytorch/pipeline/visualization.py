import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from pytorch.env import Point2D
from pytorch.models import DistanceConverter, TIMERNetwork
from pytorch.utils import get_trajectory_visualization


def load_policy_from_checkpoint(checkpoint_path: str, device: str = "cpu"):
    device_t = torch.device(device)
    ckpt = torch.load(checkpoint_path, map_location=device_t, weights_only=False)
    args = ckpt.get("args", {})

    model = TIMERNetwork(
        obs_dim=6,
        act_dim=2,
        num_dist_bins=int(args.get("num_distance_bins", 50)),
        hidden_dims=[int(args.get("hidden_dim", 256))] * int(args.get("num_layers", 3)),
        min_act_scale=float(args.get("min_act_scale", 1e-2)),
    ).to(device_t)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dist_converter = DistanceConverter(
        min_distance=float(args.get("min_distance", 0.0)),
        max_distance=float(args.get("max_distance", 140.0)),
        num_bins=int(args.get("num_distance_bins", 50)),
        device=device_t,
    )
    norm_stats = ckpt["norm_stats"]

    return model, dist_converter, norm_stats, args


def render_policy_videos(
    checkpoint_path: str,
    out_dir: str,
    title: str,
    num_trajs: int,
    max_distance: int = 140,
    device: str = "cpu",
):
    os.makedirs(out_dir, exist_ok=True)

    model, dist_converter, norm_stats, args = load_policy_from_checkpoint(checkpoint_path, device=device)
    env = Point2D(
        physics_substeps=int(args.get("env_physics_substeps", 10)),
        success_radius=float(args.get("env_success_radius", 0.15)),
    )

    all_frames: List[np.ndarray] = []
    for i in range(num_trajs):
        print(f"Generating traj {i + 1}/{num_trajs}")
        frames = get_trajectory_visualization(
            env,
            model,
            dist_converter,
            norm_stats,
            title=title,
            min_distance=float(args.get("min_distance", 0.0)),
            max_distance=max_distance,
            device=device,
        )
        all_frames.extend(frames)

    from moviepy import ImageSequenceClip

    clip = ImageSequenceClip(all_frames, fps=10)
    out_path = os.path.join(out_dir, f"{title.lower().replace(' ', '_')}.mp4")
    clip.write_videofile(out_path, fps=10, codec="libx264", logger=None)
    print(f"Saved {out_path}")
    return out_path


def collect_env_plot_sequences(
    checkpoint_path: str,
    num_figure_episodes: int,
    max_distance: int = 140,
    device: str = "cpu",
):
    model, dist_converter, norm_stats, args = load_policy_from_checkpoint(checkpoint_path, device=device)
    env = Point2D(
        physics_substeps=int(args.get("env_physics_substeps", 10)),
        success_radius=float(args.get("env_success_radius", 0.15)),
    )

    full_imgs = []
    env_imgs = []
    plot_imgs = []
    for _ in range(num_figure_episodes):
        full, (env_sub, plot_sub) = get_trajectory_visualization(
            env,
            model,
            dist_converter,
            norm_stats,
            title="",
            min_distance=float(args.get("min_distance", 0.0)),
            max_distance=max_distance,
            device=device,
            return_sub_images=True,
        )
        full_imgs.append(full)
        env_imgs.append(env_sub)
        plot_imgs.append(plot_sub)
    return full_imgs, env_imgs, plot_imgs
