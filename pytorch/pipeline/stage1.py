import os
from typing import Dict, Iterator, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from pytorch.data import PointMassDataset, build_dataset_bundle
from pytorch.models import DistanceConverter, TIMERNetwork
from pytorch.utils import finish_wandb_run, init_wandb_run, log_wandb


def _cycle(loader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _split_into_minibatches(batch: Dict[str, torch.Tensor], num_minibatches: int) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in batch.items():
        bsz = v.shape[0]
        assert bsz % num_minibatches == 0, f"Batch size {bsz} not divisible by num_minibatches {num_minibatches}"
        mb = bsz // num_minibatches
        out[k] = v.view(num_minibatches, mb, *v.shape[1:])
    return out


def _compute_pretrain_terms(model, dist_converter, obs, acts, times_to_success):
    preds = model(obs)
    target_bins = dist_converter.distance_to_network_format(times_to_success)

    act_log_prob = model.act_log_prob(preds["act_loc"], preds["act_scale"], acts).mean()
    dist_log_prob = model.dist_log_prob(preds["dist_logits"], target_bins).mean()
    loss = -act_log_prob - dist_log_prob
    return loss, act_log_prob, dist_log_prob


def train_stage1(args):
    if args.num_steps % args.num_minibatches != 0:
        raise ValueError("num_steps must be divisible by num_minibatches")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    bundle = build_dataset_bundle(args.data_path, train_ratio=args.train_ratio)
    train_dataset = PointMassDataset(bundle.train_tensors)
    val_dataset = PointMassDataset(bundle.val_tensors)

    batch_size = args.global_minibatch_size * args.num_minibatches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    train_iter = _cycle(train_loader)
    val_iter = _cycle(val_loader)

    model = TIMERNetwork(
        obs_dim=6,
        act_dim=2,
        num_dist_bins=args.num_distance_bins,
        hidden_dims=[args.hidden_dim] * args.num_layers,
        min_act_scale=args.min_act_scale,
    ).to(device)

    dist_converter = DistanceConverter(
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        num_bins=args.num_distance_bins,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-7)
    wandb_run = init_wandb_run(args, job_type="train-stage1")

    train_metrics = {
        "losses": [],
        "act_log_probs": [],
        "dist_log_probs": [],
        "val_losses": [],
        "val_act_log_probs": [],
        "val_dist_log_probs": [],
    }

    outer_steps = args.num_steps // args.num_minibatches
    try:
        for outer in range(outer_steps):
            model.train()
            batch = next(train_iter)
            mini = _split_into_minibatches(batch, args.num_minibatches)

            step_loss = 0.0
            step_act_log_prob = 0.0
            step_dist_log_prob = 0.0

            for m in range(args.num_minibatches):
                obs = torch.cat(
                    [
                        mini["obs_cur_pos"][m],
                        mini["obs_cur_vel"][m],
                        mini["obs_goal_pos"][m],
                    ],
                    dim=-1,
                ).to(device)
                acts = mini["action"][m].to(device)
                times = mini["time_to_success"][m].to(device)

                loss, act_log_prob, dist_log_prob = _compute_pretrain_terms(model, dist_converter, obs, acts, times)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                step_loss += loss.item()
                step_act_log_prob += act_log_prob.item()
                step_dist_log_prob += dist_log_prob.item()

            step_loss /= args.num_minibatches
            step_act_log_prob /= args.num_minibatches
            step_dist_log_prob /= args.num_minibatches

            train_metrics["losses"].append(step_loss)
            train_metrics["act_log_probs"].append(step_act_log_prob)
            train_metrics["dist_log_probs"].append(step_dist_log_prob)

            model.eval()
            with torch.no_grad():
                val_batch = next(val_iter)
                val_mini = _split_into_minibatches(val_batch, args.num_minibatches)

                val_loss = 0.0
                val_act_log_prob = 0.0
                val_dist_log_prob = 0.0
                for m in range(args.num_minibatches):
                    obs = torch.cat(
                        [
                            val_mini["obs_cur_pos"][m],
                            val_mini["obs_cur_vel"][m],
                            val_mini["obs_goal_pos"][m],
                        ],
                        dim=-1,
                    ).to(device)
                    acts = val_mini["action"][m].to(device)
                    times = val_mini["time_to_success"][m].to(device)
                    loss, act_log_prob, dist_log_prob = _compute_pretrain_terms(model, dist_converter, obs, acts, times)
                    val_loss += loss.item()
                    val_act_log_prob += act_log_prob.item()
                    val_dist_log_prob += dist_log_prob.item()

                val_loss /= args.num_minibatches
                val_act_log_prob /= args.num_minibatches
                val_dist_log_prob /= args.num_minibatches

            train_metrics["val_losses"].append(val_loss)
            train_metrics["val_act_log_probs"].append(val_act_log_prob)
            train_metrics["val_dist_log_probs"].append(val_dist_log_prob)
            step = (outer + 1) * args.num_minibatches
            log_wandb(
                wandb_run,
                {
                    "stage1/loss": step_loss,
                    "stage1/act_log_prob": step_act_log_prob,
                    "stage1/dist_log_prob": step_dist_log_prob,
                    "stage1/val_loss": val_loss,
                    "stage1/val_act_log_prob": val_act_log_prob,
                    "stage1/val_dist_log_prob": val_dist_log_prob,
                },
                step=step,
            )

            print(
                f"[{outer + 1}/{outer_steps}] step={step} "
                f"loss={step_loss:.4f} val_loss={val_loss:.4f}"
            )
    finally:
        finish_wandb_run(wandb_run)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_path = os.path.join(args.checkpoint_dir, "stage1_model.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "norm_stats": bundle.norm_stats,
            "args": vars(args),
            "metrics": train_metrics,
        },
        save_path,
    )
    print(f"Saved Stage1 checkpoint: {save_path}")
    return save_path
