from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class DatasetBundle:
    train_tensors: Dict[str, torch.Tensor]
    val_tensors: Dict[str, torch.Tensor]
    norm_stats: Dict[str, torch.Tensor]


def _ensure_float32(x: torch.Tensor) -> torch.Tensor:
    return x.to(dtype=torch.float32)


def load_tuple_tensors(data_path: str) -> Dict[str, torch.Tensor]:
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    required = [
        "obs_cur_pos",
        "obs_cur_vel",
        "obs_goal_pos",
        "actions",
        "times_to_success",
    ]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in dataset: {data_path}")

    return {
        "obs_cur_pos": _ensure_float32(data["obs_cur_pos"]),
        "obs_cur_vel": _ensure_float32(data["obs_cur_vel"]),
        "obs_goal_pos": _ensure_float32(data["obs_goal_pos"]),
        "action": _ensure_float32(data["actions"]),
        "time_to_success": _ensure_float32(data["times_to_success"]),
    }


def compute_norm_stats(all_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cur_pos_mean = all_tensors["obs_cur_pos"].mean(dim=0, keepdim=True)
    cur_pos_std = all_tensors["obs_cur_pos"].std(dim=0, keepdim=True)
    cur_vel_mean = all_tensors["obs_cur_vel"].mean(dim=0, keepdim=True)
    cur_vel_std = all_tensors["obs_cur_vel"].std(dim=0, keepdim=True)
    act_mean = all_tensors["action"].mean(dim=0, keepdim=True)
    act_std = all_tensors["action"].std(dim=0, keepdim=True)

    return {
        "cur_pos_mean": cur_pos_mean,
        "cur_pos_std": cur_pos_std,
        "cur_vel_mean": cur_vel_mean,
        "cur_vel_std": cur_vel_std,
        "act_mean": act_mean,
        "act_std": act_std,
    }


def normalize_tensors(
    tensors: Dict[str, torch.Tensor],
    norm_stats: Dict[str, torch.Tensor],
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    return {
        "obs_cur_pos": (tensors["obs_cur_pos"] - norm_stats["cur_pos_mean"]) / (norm_stats["cur_pos_std"] + eps),
        "obs_cur_vel": (tensors["obs_cur_vel"] - norm_stats["cur_vel_mean"]) / (norm_stats["cur_vel_std"] + eps),
        "obs_goal_pos": (tensors["obs_goal_pos"] - norm_stats["cur_pos_mean"]) / (norm_stats["cur_pos_std"] + eps),
        "action": (tensors["action"] - norm_stats["act_mean"]) / (norm_stats["act_std"] + eps),
        "time_to_success": tensors["time_to_success"],
    }


def split_tensors(tensors: Dict[str, torch.Tensor], train_ratio: float = 0.9) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    total_size = tensors["action"].shape[0]
    train_size = int(total_size * train_ratio)
    train = {k: v[:train_size] for k, v in tensors.items()}
    val = {k: v[train_size:] for k, v in tensors.items()}
    return train, val


def build_dataset_bundle(data_path: str, train_ratio: float = 0.9) -> DatasetBundle:
    all_tensors = load_tuple_tensors(data_path)
    norm_stats = compute_norm_stats(all_tensors)
    normalized_all = normalize_tensors(all_tensors, norm_stats)
    train_tensors, val_tensors = split_tensors(normalized_all, train_ratio=train_ratio)
    return DatasetBundle(train_tensors=train_tensors, val_tensors=val_tensors, norm_stats=norm_stats)


class PointMassDataset(Dataset):
    def __init__(self, tensors: Dict[str, torch.Tensor]):
        self.tensors = tensors

    def __len__(self) -> int:
        return self.tensors["action"].shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "obs_cur_pos": self.tensors["obs_cur_pos"][idx],
            "obs_cur_vel": self.tensors["obs_cur_vel"][idx],
            "obs_goal_pos": self.tensors["obs_goal_pos"][idx],
            "action": self.tensors["action"][idx],
            "time_to_success": self.tensors["time_to_success"][idx],
        }
