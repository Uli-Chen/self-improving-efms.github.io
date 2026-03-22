import argparse
import os

import torch


def cmd_generate_data(args):
    from pytorch.generate_data import generate_dataset

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


def cmd_train_stage1(args):
    from pytorch.pipeline.stage1 import train_stage1

    train_stage1(args)


def cmd_train_stage2(args):
    from pytorch.pipeline.stage2 import train_stage2

    train_stage2(args)


def cmd_vis_stage1(args):
    from pytorch.pipeline.visualization import render_policy_videos

    render_policy_videos(
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        title="Stage 1 SFT Policy",
        num_trajs=args.num_trajs,
        max_distance=args.max_distance,
        device=args.device,
    )


def cmd_vis_stage2(args):
    from pytorch.pipeline.visualization import render_policy_videos

    render_policy_videos(
        checkpoint_path=args.checkpoint,
        out_dir=args.out_dir,
        title="Stage 2 Self-Improvement Policy",
        num_trajs=args.num_trajs,
        max_distance=args.max_distance,
        device=args.device,
    )


def cmd_eval_stage1(args):
    from pytorch.env import Point2D
    from pytorch.pipeline.visualization import load_policy_from_checkpoint
    from pytorch.utils import evaluate_timer_rollout_policy

    model, dist_converter, norm_stats, ckpt_args = load_policy_from_checkpoint(args.checkpoint, device=args.device)
    env = Point2D(
        physics_substeps=int(ckpt_args.get("env_physics_substeps", 10)),
        success_radius=float(ckpt_args.get("env_success_radius", 0.15)),
    )
    metrics = evaluate_timer_rollout_policy(
        env,
        model,
        dist_converter,
        norm_stats,
        num_episodes=args.num_episodes,
        max_distance=args.max_distance,
        device=args.device,
    )
    print(metrics)


def cmd_make_figures(args):
    from pytorch.pipeline.figure_export import export_notebook_style_figures

    export_notebook_style_figures(
        stage1_checkpoint=args.stage1_ckpt,
        stage2_checkpoint=args.stage2_ckpt,
        trajs_path=args.trajs_path,
        out_dir=args.out_dir,
        num_figure_episodes=args.num_figure_episodes,
        max_distance=args.max_distance,
        device=args.device,
        seed=args.seed,
    )


def _add_distance_model_args(parser: argparse.ArgumentParser):
    parser.add_argument("--min-distance", type=float, default=0.0)
    parser.add_argument("--max-distance", type=float, default=140.0)
    parser.add_argument("--num-distance-bins", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--min-act-scale", type=float, default=1e-2)


def _add_wandb_args(parser: argparse.ArgumentParser):
    parser.add_argument("--wandb-project", type=str, default="timer-pointmass")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")


def build_parser():
    parser = argparse.ArgumentParser(description="Notebook-aligned PointMass CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate-data")
    p.add_argument("--num-episodes", type=int, default=10000)
    p.add_argument("--num-waypoints-per-episode", type=int, default=5)
    p.add_argument("--episode-len-discard-thresh", type=int, default=10)
    p.add_argument("--save-path", type=str, default="data/pointmass_dataset.pt")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_generate_data)

    p = sub.add_parser("train-stage1")
    p.add_argument("--data-path", type=str, default="data/pointmass_dataset_tuples.pt")
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--global-minibatch-size", type=int, default=256)
    p.add_argument("--num-minibatches", type=int, default=128)
    p.add_argument("--num-steps", type=int, default=32768)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    _add_distance_model_args(p)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    _add_wandb_args(p)
    p.set_defaults(func=cmd_train_stage1)

    p = sub.add_parser("train-stage2")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-reinforce-sgd-steps", type=int, default=2048)
    p.add_argument("--reinforce-global-minibatch-size", type=int, default=64)
    p.add_argument("--reinforce-global-batch-size", type=int, default=2048)
    p.add_argument("--reinforce-num-minibatches", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    _add_distance_model_args(p)
    p.add_argument("--env-physics-substeps", type=int, default=10)
    p.add_argument("--env-success-radius", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-freq", type=int, default=5)
    p.add_argument("--save-freq", type=int, default=5)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--vis-out-dir", type=str, default="outputs/vis_stage2")
    _add_wandb_args(p)
    p.set_defaults(func=cmd_train_stage2)

    p = sub.add_parser("vis-stage1")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-trajs", type=int, default=10)
    p.add_argument("--max-distance", type=int, default=140)
    p.add_argument("--out-dir", type=str, default="outputs/vis_stage1")
    p.add_argument("--device", type=str, default="cpu")
    p.set_defaults(func=cmd_vis_stage1)

    p = sub.add_parser("vis-stage2")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-trajs", type=int, default=5)
    p.add_argument("--max-distance", type=int, default=140)
    p.add_argument("--out-dir", type=str, default="outputs/vis_stage2")
    p.add_argument("--device", type=str, default="cpu")
    p.set_defaults(func=cmd_vis_stage2)

    p = sub.add_parser("eval-stage1")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--num-episodes", type=int, default=25)
    p.add_argument("--max-distance", type=int, default=140)
    p.add_argument("--device", type=str, default="cpu")
    p.set_defaults(func=cmd_eval_stage1)

    p = sub.add_parser("make-figures")
    p.add_argument("--stage1-ckpt", type=str, required=True)
    p.add_argument("--stage2-ckpt", type=str, required=True)
    p.add_argument("--trajs-path", type=str, default="data/pointmass_dataset_trajs.pt")
    p.add_argument("--num-figure-episodes", type=int, default=10)
    p.add_argument("--max-distance", type=int, default=140)
    p.add_argument("--out-dir", type=str, default="outputs/figures")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_make_figures)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
