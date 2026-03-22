import argparse
import os
from typing import Dict, List, NamedTuple, Optional

import numpy as np
import torch

from pytorch.env import Point2D


class DataTuple(NamedTuple):
    observation: Dict[str, np.ndarray]
    action: np.ndarray
    time_to_success: float
    reward: float
    discount: float
    next_observation: Dict[str, np.ndarray]


def pd_controller(cur_pos: np.ndarray, cur_vel: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
    kp = 0.0002
    kd = 0.0125
    return kp * (goal_pos - cur_pos) + kd * (-1.0 * cur_vel)


def _stack_traj(traj: List[DataTuple]) -> Dict[str, torch.Tensor]:
    obs_cur_pos = np.stack([x.observation["cur_pos"] for x in traj]).astype(np.float32)
    obs_cur_vel = np.stack([x.observation["cur_vel"] for x in traj]).astype(np.float32)
    obs_goal_pos = np.stack([x.observation["goal_pos"] for x in traj]).astype(np.float32)

    next_obs_cur_pos = np.stack([x.next_observation["cur_pos"] for x in traj]).astype(np.float32)
    next_obs_cur_vel = np.stack([x.next_observation["cur_vel"] for x in traj]).astype(np.float32)
    next_obs_goal_pos = np.stack([x.next_observation["goal_pos"] for x in traj]).astype(np.float32)

    actions = np.stack([x.action for x in traj]).astype(np.float32)
    rewards = np.asarray([x.reward for x in traj], dtype=np.float32)
    discounts = np.asarray([x.discount for x in traj], dtype=np.float32)

    traj_len = len(traj)
    times_to_success = np.arange(traj_len - 1, -1, -1, dtype=np.float32)

    return {
        "obs_cur_pos": torch.from_numpy(obs_cur_pos),
        "obs_cur_vel": torch.from_numpy(obs_cur_vel),
        "obs_goal_pos": torch.from_numpy(obs_goal_pos),
        "actions": torch.from_numpy(actions),
        "times_to_success": torch.from_numpy(times_to_success),
        "rewards": torch.from_numpy(rewards),
        "discounts": torch.from_numpy(discounts),
        "next_obs_cur_pos": torch.from_numpy(next_obs_cur_pos),
        "next_obs_cur_vel": torch.from_numpy(next_obs_cur_vel),
        "next_obs_goal_pos": torch.from_numpy(next_obs_goal_pos),
    }


def _concat_episodes(episodes: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = episodes[0].keys()
    return {k: torch.cat([ep[k] for ep in episodes], dim=0) for k in keys}


def generate_dataset(
    num_episodes: int,
    num_waypoints_per_episode: int,
    episode_len_discard_thresh: int,
    seed: Optional[int] = None,
) -> tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor], np.ndarray]:
    if seed is not None:
        np.random.seed(seed)

    env = Point2D()
    episodes: List[Dict[str, torch.Tensor]] = []
    episode_lens: List[int] = []

    while len(episodes) < num_episodes:
        traj: List[DataTuple] = []

        ts = env.reset()
        cur_obs = ts.observation
        succ = env.success()

        waypoint_idx = 0
        if num_waypoints_per_episode == 0:
            cur_waypoint = cur_obs["goal_pos"]
        else:
            cur_waypoint = env.sample_goal()
        waypoint_succ = env.success(waypoint=cur_waypoint)

        while not succ:
            if waypoint_succ:
                waypoint_idx += 1
                waypoint_idx = min(waypoint_idx, num_waypoints_per_episode)
                if waypoint_idx == num_waypoints_per_episode:
                    cur_waypoint = cur_obs["goal_pos"]
                else:
                    cur_waypoint = env.sample_goal()

            act = pd_controller(cur_obs["cur_pos"], cur_obs["cur_vel"], cur_waypoint)
            ts = env.step(act)
            next_obs = ts.observation

            traj.append(
                DataTuple(
                    observation={
                        "cur_pos": cur_obs["cur_pos"].copy(),
                        "cur_vel": cur_obs["cur_vel"].copy(),
                        "goal_pos": cur_obs["goal_pos"].copy(),
                    },
                    action=act.astype(np.float32),
                    time_to_success=0.0,
                    reward=float(ts.reward) if ts.reward is not None else 0.0,
                    discount=1.0,
                    next_observation={
                        "cur_pos": next_obs["cur_pos"].copy(),
                        "cur_vel": next_obs["cur_vel"].copy(),
                        "goal_pos": next_obs["goal_pos"].copy(),
                    },
                )
            )

            cur_obs = next_obs
            succ = env.success()
            waypoint_succ = env.success(waypoint=cur_waypoint)

        # Add terminal tuple exactly like notebook.
        act = pd_controller(cur_obs["cur_pos"], cur_obs["cur_vel"], cur_waypoint)
        traj.append(
            DataTuple(
                observation={
                    "cur_pos": cur_obs["cur_pos"].copy(),
                    "cur_vel": cur_obs["cur_vel"].copy(),
                    "goal_pos": cur_obs["goal_pos"].copy(),
                },
                action=act.astype(np.float32),
                time_to_success=0.0,
                reward=float(ts.reward) if ts.reward is not None else 0.0,
                discount=0.0,
                next_observation={
                    "cur_pos": cur_obs["cur_pos"].copy(),
                    "cur_vel": cur_obs["cur_vel"].copy(),
                    "goal_pos": cur_obs["goal_pos"].copy(),
                },
            )
        )

        if len(traj) < episode_len_discard_thresh:
            continue

        stacked = _stack_traj(traj)
        episodes.append(stacked)
        episode_lens.append(stacked["obs_cur_pos"].shape[0])

        if len(episodes) % 100 == 0:
            print(f"Generated {len(episodes)}/{num_episodes} episodes")

    all_tuples = _concat_episodes(episodes)
    return episodes, all_tuples, np.asarray(episode_lens, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pointmass dataset aligned with notebook")
    parser.add_argument("--num_episodes", type=int, default=10000)
    parser.add_argument("--num_waypoints_per_episode", type=int, default=5)
    parser.add_argument("--episode_len_discard_thresh", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="data/pointmass_dataset.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    episodes, all_tuples, episode_lens = generate_dataset(
        num_episodes=args.num_episodes,
        num_waypoints_per_episode=args.num_waypoints_per_episode,
        episode_len_discard_thresh=args.episode_len_discard_thresh,
        seed=args.seed,
    )

    head, tail = os.path.splitext(args.save_path)
    trajs_path = f"{head}_trajs{tail}"
    tuples_path = f"{head}_tuples{tail}"

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    torch.save(episodes, trajs_path)
    torch.save(all_tuples, tuples_path)

    print(f"Saved trajectories: {trajs_path}")
    print(f"Saved tuples: {tuples_path}")
    print(f"Num Episodes: {len(episode_lens)}")
    print(f"Episode Lens: {episode_lens.mean():.2f} +/- {episode_lens.std():.2f}")
    print(f"Max Episode Len: {episode_lens.max()}")
    print(f"Min Episode Len: {episode_lens.min()}")


if __name__ == "__main__":
    main()
