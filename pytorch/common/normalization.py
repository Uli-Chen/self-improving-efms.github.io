from typing import Dict

import torch


def normalize_observation(obs: Dict[str, torch.Tensor], norm_stats: Dict[str, torch.Tensor], eps: float = 1e-8) -> Dict[str, torch.Tensor]:
    return {
        "cur_pos": (obs["cur_pos"] - norm_stats["cur_pos_mean"]) / (norm_stats["cur_pos_std"] + eps),
        "cur_vel": (obs["cur_vel"] - norm_stats["cur_vel_mean"]) / (norm_stats["cur_vel_std"] + eps),
        "goal_pos": (obs["goal_pos"] - norm_stats["cur_pos_mean"]) / (norm_stats["cur_pos_std"] + eps),
    }


def normalize_action(action: torch.Tensor, norm_stats: Dict[str, torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    return (action - norm_stats["act_mean"]) / (norm_stats["act_std"] + eps)


def unnormalize_action(action: torch.Tensor, norm_stats: Dict[str, torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    return action * (norm_stats["act_std"] + eps) + norm_stats["act_mean"]
