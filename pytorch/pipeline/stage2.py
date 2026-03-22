import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim

from pytorch.common.normalization import normalize_observation
from pytorch.common.rollout import rollout_policy_step
from pytorch.env import Point2D
from pytorch.models import DistanceConverter, TIMERNetwork
from pytorch.utils import (
    evaluate_timer_rollout_policy,
    eval_mode,
    finish_wandb_run,
    get_trajectory_visualization,
    init_wandb_run,
    log_wandb,
)


class ReinforceBatch(dict):
    pass


def _norm_obs_from_env_obs(obs: Dict[str, np.ndarray], norm_stats: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    obs_t = {
        "cur_pos": torch.as_tensor(obs["cur_pos"], dtype=torch.float32, device=device).unsqueeze(0),
        "cur_vel": torch.as_tensor(obs["cur_vel"], dtype=torch.float32, device=device).unsqueeze(0),
        "goal_pos": torch.as_tensor(obs["goal_pos"], dtype=torch.float32, device=device).unsqueeze(0),
    }
    return normalize_observation(obs_t, {k: v.to(device) for k, v in norm_stats.items()})


def generate_timer_reinforce_dataset(
    env,
    model,
    dist_converter,
    norm_stats,
    num_steps: int,
    gamma: float,
    max_distance: int = 140,
    device: str = "cpu",
) -> Tuple[ReinforceBatch, Dict[str, np.ndarray]]:
    device_t = torch.device(device)
    total_steps = 0
    data_tuples: List[ReinforceBatch] = []
    data_stats: List[Dict[str, float]] = []

    with torch.no_grad():
        with eval_mode(model):
            while total_steps < num_steps:
                traj_obs = []
                traj_acts = []
                traj_dist_preds = []

                episode_steps = 0
                episode_return = 0.0
                ts = env.reset()
                cur_obs_norm = _norm_obs_from_env_obs(ts.observation, norm_stats, device_t)
                traj_obs.append(cur_obs_norm)

                while (not env.success()) and episode_steps < max_distance:
                    action, extras = rollout_policy_step(
                        model=model,
                        dist_converter=dist_converter,
                        norm_stats=norm_stats,
                        observation=ts.observation,
                        device=device_t,
                    )
                    traj_acts.append(extras["norm_act"])
                    traj_dist_preds.append(extras["pred_dist_to_succ"])

                    ts = env.step(action.detach().cpu().numpy())
                    episode_return += float(ts.reward)
                    cur_obs_norm = _norm_obs_from_env_obs(ts.observation, norm_stats, device_t)
                    traj_obs.append(cur_obs_norm)
                    episode_steps += 1

                if episode_steps < 1:
                    continue

                total_steps += len(traj_acts)

                # Add one final prediction to form distance differences.
                _, extras = rollout_policy_step(
                    model=model,
                    dist_converter=dist_converter,
                    norm_stats=norm_stats,
                    observation=ts.observation,
                    device=device_t,
                )
                traj_dist_preds.append(extras["pred_dist_to_succ"])

                traj_obs_cat = {
                    "cur_pos": torch.cat([x["cur_pos"] for x in traj_obs], dim=0),
                    "cur_vel": torch.cat([x["cur_vel"] for x in traj_obs], dim=0),
                    "goal_pos": torch.cat([x["goal_pos"] for x in traj_obs], dim=0),
                }
                traj_acts_cat = torch.cat(traj_acts, dim=0)
                traj_dist_preds_cat = torch.cat(traj_dist_preds, dim=0)

                rews = -1.0 * (traj_dist_preds_cat[1:] - traj_dist_preds_cat[:-1])
                weights = []
                temp = 0.0
                for i in range(rews.shape[0] - 1, -1, -1):
                    temp = float(rews[i].item()) + gamma * temp
                    weights.append(temp)
                weights = torch.tensor(weights[::-1], dtype=torch.float32, device=device_t)

                data_tuples.append(
                    ReinforceBatch(
                        {
                            "obs_cur_pos": traj_obs_cat["cur_pos"][:-1],
                            "obs_cur_vel": traj_obs_cat["cur_vel"][:-1],
                            "obs_goal_pos": traj_obs_cat["goal_pos"][:-1],
                            "action": traj_acts_cat,
                            "weight": weights,
                        }
                    )
                )

                data_stats.append(
                    {
                        "success": float(env.success()),
                        "return": episode_return,
                        "len": float(episode_steps),
                    }
                )

    batch = ReinforceBatch(
        {
            "obs_cur_pos": torch.cat([x["obs_cur_pos"] for x in data_tuples], dim=0),
            "obs_cur_vel": torch.cat([x["obs_cur_vel"] for x in data_tuples], dim=0),
            "obs_goal_pos": torch.cat([x["obs_goal_pos"] for x in data_tuples], dim=0),
            "action": torch.cat([x["action"] for x in data_tuples], dim=0),
            "weight": torch.cat([x["weight"] for x in data_tuples], dim=0),
        }
    )

    stats = {
        "success": np.asarray([x["success"] for x in data_stats], dtype=np.float32),
        "return": np.asarray([x["return"] for x in data_stats], dtype=np.float32),
        "len": np.asarray([x["len"] for x in data_stats], dtype=np.float32),
    }
    return batch, stats


def _reinforce_loss(model, batch: ReinforceBatch, max_distance: float) -> torch.Tensor:
    obs = torch.cat([batch["obs_cur_pos"], batch["obs_cur_vel"], batch["obs_goal_pos"]], dim=-1)
    preds = model(obs)
    act_log_probs = model.act_log_prob(preds["act_loc"], preds["act_scale"], batch["action"])
    weights = batch["weight"] / float(max_distance)
    return -1.0 * (weights * act_log_probs).mean()


def train_stage2(args):
    if args.num_reinforce_sgd_steps % args.reinforce_num_minibatches != 0:
        raise ValueError("num_reinforce_sgd_steps must be divisible by reinforce_num_minibatches")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    norm_stats = checkpoint["norm_stats"]

    model = TIMERNetwork(
        obs_dim=6,
        act_dim=2,
        num_dist_bins=args.num_distance_bins,
        hidden_dims=[args.hidden_dim] * args.num_layers,
        min_act_scale=args.min_act_scale,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dist_converter = DistanceConverter(
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        num_bins=args.num_distance_bins,
        device=device,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-7)
    env = Point2D(physics_substeps=args.env_physics_substeps, success_radius=args.env_success_radius)
    wandb_run = init_wandb_run(args, job_type="train-stage2")

    metrics_list = defaultdict(list)

    outer_steps = args.num_reinforce_sgd_steps // args.reinforce_num_minibatches
    try:
        for outer in range(outer_steps):
            model.train()

            reinforce_data, data_stats = generate_timer_reinforce_dataset(
                env=env,
                model=model,
                dist_converter=dist_converter,
                norm_stats=norm_stats,
                num_steps=args.reinforce_global_batch_size,
                gamma=args.gamma,
                max_distance=args.max_distance,
                device=str(device),
            )

            # Truncate like notebook before update.
            reinforce_data = ReinforceBatch({k: v[: args.reinforce_global_batch_size] for k, v in reinforce_data.items()})

            cur_losses = []
            for m in range(args.reinforce_num_minibatches):
                s = m * args.reinforce_global_minibatch_size
                e = (m + 1) * args.reinforce_global_minibatch_size
                minibatch = ReinforceBatch({k: v[s:e] for k, v in reinforce_data.items()})

                loss = _reinforce_loss(model, minibatch, max_distance=args.max_distance)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cur_losses.append(loss.item())

            metrics_list["reinforce_loss"].append(float(np.mean(cur_losses)))
            metrics_list["success_rate"].append(float(np.mean(data_stats["success"])))
            metrics_list["return_mean"].append(float(np.mean(data_stats["return"])))
            metrics_list["return_std"].append(float(np.std(data_stats["return"])))
            metrics_list["len_mean"].append(float(np.mean(data_stats["len"])))
            metrics_list["len_std"].append(float(np.std(data_stats["len"])))
            metrics_list["max_len"].append(float(np.max(data_stats["len"])))
            metrics_list["min_len"].append(float(np.min(data_stats["len"])))
            step = (outer + 1) * args.reinforce_num_minibatches
            log_wandb(
                wandb_run,
                {
                    "stage2/reinforce_loss": metrics_list["reinforce_loss"][-1],
                    "stage2/success_rate": metrics_list["success_rate"][-1],
                    "stage2/return_mean": metrics_list["return_mean"][-1],
                    "stage2/return_std": metrics_list["return_std"][-1],
                    "stage2/len_mean": metrics_list["len_mean"][-1],
                    "stage2/len_std": metrics_list["len_std"][-1],
                    "stage2/max_len": metrics_list["max_len"][-1],
                    "stage2/min_len": metrics_list["min_len"][-1],
                },
                step=step,
            )

            print(
                f"[{outer + 1}/{outer_steps}] reinforce_steps={step} "
                f"loss={metrics_list['reinforce_loss'][-1]:.4f} success={metrics_list['success_rate'][-1]:.3f}"
            )

            if (outer + 1) % args.eval_freq == 0:
                eval_metrics = evaluate_timer_rollout_policy(
                    env,
                    model,
                    dist_converter,
                    norm_stats,
                    num_episodes=25,
                    max_distance=args.max_distance,
                    device=str(device),
                )
                print(
                    "Eval | "
                    f"success={eval_metrics['success_rate']:.3f}, "
                    f"return={eval_metrics['return_mean']:.2f} +/- {eval_metrics['return_std']:.2f}, "
                    f"len={eval_metrics['len_mean']:.2f} +/- {eval_metrics['len_std']:.2f}"
                )
                log_wandb(
                    wandb_run,
                    {
                        "stage2/eval_success_rate": eval_metrics["success_rate"],
                        "stage2/eval_return_mean": eval_metrics["return_mean"],
                        "stage2/eval_return_std": eval_metrics["return_std"],
                        "stage2/eval_len_mean": eval_metrics["len_mean"],
                        "stage2/eval_len_std": eval_metrics["len_std"],
                        "stage2/eval_max_len": eval_metrics["max_len"],
                        "stage2/eval_min_len": eval_metrics["min_len"],
                    },
                    step=step,
                )

                os.makedirs(args.vis_out_dir, exist_ok=True)
                vis_imgs = get_trajectory_visualization(
                    env,
                    model,
                    dist_converter,
                    norm_stats,
                    title=f"RL Iter {(outer + 1)}",
                    min_distance=args.min_distance,
                    max_distance=args.max_distance,
                    device=str(device),
                )
                if len(vis_imgs) > 0:
                    from moviepy import ImageSequenceClip

                    clip = ImageSequenceClip(vis_imgs, fps=10)
                    clip.write_videofile(
                        os.path.join(args.vis_out_dir, f"stage2_iter_{outer + 1:04d}.mp4"),
                        fps=10,
                        codec="libx264",
                        logger=None,
                    )

            if (outer + 1) % args.save_freq == 0 or (outer + 1) == outer_steps:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                save_path = os.path.join(args.checkpoint_dir, "stage2_model.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "norm_stats": norm_stats,
                        "args": vars(args),
                        "metrics": dict(metrics_list),
                    },
                    save_path,
                )
    finally:
        finish_wandb_run(wandb_run)

    return os.path.join(args.checkpoint_dir, "stage2_model.pt")
