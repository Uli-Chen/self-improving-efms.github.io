import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

from pytorch.env import Point2D
from pytorch.pipeline.visualization import collect_env_plot_sequences


def _tight_concat_last_frames(env_episodes: List[List[np.ndarray]]) -> np.ndarray:
    out = []
    for i, episode_imgs in enumerate(env_episodes):
        img = episode_imgs[-1]
        if i == 0:
            img = img[:, :-30]
        elif i == len(env_episodes) - 1:
            img = img[:, 30:]
        else:
            img = img[:, 30:-30]
        out.append(img)
    return np.concatenate(out, axis=1)


def _collect_data_env_sequences(trajs_path: str, num_figure_episodes: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    episodes = torch.load(trajs_path, map_location="cpu", weights_only=False)
    env = Point2D()

    inds = rng.choice(len(episodes), size=num_figure_episodes, replace=False)
    data_env_imgs = []
    for i in inds:
        points = episodes[i]["obs_cur_pos"].numpy()
        goal_pos = episodes[i]["obs_goal_pos"][0].numpy()
        imgs = []
        for t in range(points.shape[0]):
            imgs.append(env.render(title="", points=points[: t + 1], goal_pos=goal_pos))
        data_env_imgs.append(imgs)
    return data_env_imgs


def export_notebook_style_figures(
    stage1_checkpoint: str,
    stage2_checkpoint: str,
    trajs_path: str,
    out_dir: str,
    num_figure_episodes: int = 10,
    max_distance: int = 140,
    device: str = "cpu",
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)

    _, sft_env_imgs, _ = collect_env_plot_sequences(
        checkpoint_path=stage1_checkpoint,
        num_figure_episodes=num_figure_episodes,
        max_distance=max_distance,
        device=device,
    )
    _, self_env_imgs, _ = collect_env_plot_sequences(
        checkpoint_path=stage2_checkpoint,
        num_figure_episodes=num_figure_episodes,
        max_distance=max_distance,
        device=device,
    )
    data_env_imgs = _collect_data_env_sequences(trajs_path, num_figure_episodes, seed=seed)

    figure_sft_env_imgs = _tight_concat_last_frames(sft_env_imgs)
    figure_self_env_imgs = _tight_concat_last_frames(self_env_imgs)
    figure_data_env_imgs = _tight_concat_last_frames(data_env_imgs)

    buffer = 255 * np.ones((300, figure_data_env_imgs.shape[1], 3), dtype=figure_data_env_imgs.dtype)
    figure_env_imgs = np.concatenate(
        [buffer, figure_data_env_imgs, buffer, figure_sft_env_imgs, buffer, figure_self_env_imgs],
        axis=0,
    )

    p1 = os.path.join(out_dir, "ten_tight_pointmass_sft_env_imgs.png")
    p2 = os.path.join(out_dir, "ten_tight_pointmass_self_improvement_env_imgs.png")
    p3 = os.path.join(out_dir, "ten_tight_pointmass_data_env_imgs.png")
    p4 = os.path.join(out_dir, "ten_tight_pointmass_figure_env_imgs.png")

    plt.imsave(p1, figure_sft_env_imgs)
    plt.imsave(p2, figure_self_env_imgs)
    plt.imsave(p3, figure_data_env_imgs)
    plt.imsave(p4, figure_env_imgs)

    print(f"Saved {p1}")
    print(f"Saved {p2}")
    print(f"Saved {p3}")
    print(f"Saved {p4}")

    return [p1, p2, p3, p4]
