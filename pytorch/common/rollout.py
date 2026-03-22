from typing import Dict, Tuple

import torch

from pytorch.common.normalization import normalize_observation, unnormalize_action


def rollout_policy_step(
    model,
    dist_converter,
    norm_stats: Dict[str, torch.Tensor],
    observation: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    obs_pos = torch.as_tensor(observation["cur_pos"], dtype=torch.float32, device=device).unsqueeze(0)
    obs_vel = torch.as_tensor(observation["cur_vel"], dtype=torch.float32, device=device).unsqueeze(0)
    obs_goal = torch.as_tensor(observation["goal_pos"], dtype=torch.float32, device=device).unsqueeze(0)

    obs = normalize_observation(
        {
            "cur_pos": obs_pos,
            "cur_vel": obs_vel,
            "goal_pos": obs_goal,
        },
        {k: v.to(device) for k, v in norm_stats.items()},
    )

    net_obs = torch.cat([obs["cur_pos"], obs["cur_vel"], obs["goal_pos"]], dim=-1)
    preds = model(net_obs)

    norm_act = model.sample_act(preds["act_loc"], preds["act_scale"])
    unnorm_act = unnormalize_action(norm_act, {k: v.to(device) for k, v in norm_stats.items()})

    dist_pred = dist_converter.network_format_to_distance(preds["dist_logits"])

    extras = {
        "pred_dist_to_succ": dist_pred,
        "dist_logits": preds["dist_logits"],
        "norm_obs": obs,
        "norm_act": norm_act,
    }
    return unnorm_act[0], extras
