from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from pytorch.common.rollout import rollout_policy_step


@contextmanager
def eval_mode(model):
    prev_mode = model.training
    try:
        model.eval()
        yield model
    finally:
        model.train(prev_mode)


def init_wandb_run(args, job_type: str):
    if getattr(args, "disable_wandb", False):
        return None

    try:
        import wandb
    except ImportError:
        print("wandb is not installed. Continuing without wandb logging.")
        return None

    mode = "offline" if getattr(args, "wandb_offline", False) else "online"
    run_name = getattr(args, "wandb_run_name", None)
    run_group = getattr(args, "wandb_group", None)
    config = dict(vars(args))
    config["job_type"] = job_type

    try:
        return wandb.init(
            project=getattr(args, "wandb_project", "timer-pointmass"),
            entity=getattr(args, "wandb_entity", None),
            name=run_name,
            group=run_group,
            mode=mode,
            config=config,
            job_type=job_type,
        )
    except Exception as exc:
        print(f"wandb init failed ({exc}). Continuing without wandb logging.")
        return None


def log_wandb(run, metrics: Dict[str, Any], step: Optional[int] = None):
    if run is None:
        return
    run.log(metrics, step=step)


def finish_wandb_run(run):
    if run is None:
        return
    run.finish()


def evaluate_timer_rollout_policy(
    env,
    model,
    dist_converter,
    norm_stats,
    num_episodes: int = 25,
    max_distance: int = 140,
    device: str = "cpu",
):
    device_t = torch.device(device)
    stats = []

    with torch.no_grad():
        with eval_mode(model):
            for _ in range(num_episodes):
                timesteps = []
                ts = env.reset()
                ts = ts._replace(reward=0.0)
                timesteps.append(ts)

                while (not env.success()) and len(timesteps) < max_distance:
                    action, _ = rollout_policy_step(
                        model=model,
                        dist_converter=dist_converter,
                        norm_stats=norm_stats,
                        observation=ts.observation,
                        device=device_t,
                    )
                    ts = env.step(action.detach().cpu().numpy())
                    timesteps.append(ts)

                stats.append(
                    {
                        "success": env.success(),
                        "return": float(sum(x.reward for x in timesteps)),
                        "len": len(timesteps),
                    }
                )

    success_rate = np.mean([s["success"] for s in stats])
    returns = np.asarray([s["return"] for s in stats], dtype=np.float32)
    lengths = np.asarray([s["len"] for s in stats], dtype=np.int32)

    return {
        "success_rate": float(success_rate),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "len_mean": float(lengths.mean()),
        "len_std": float(lengths.std()),
        "max_len": int(lengths.max()),
        "min_len": int(lengths.min()),
    }


def get_distance_plot(distances, logits, min_distance: float, max_distance: float):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    dpi = 200
    render_height_inches = 5

    fig, _ = plt.subplots(figsize=(2 * render_height_inches, render_height_inches), dpi=dpi)
    plt.clf()

    plt.subplot(1, 2, 1)
    plt.plot(distances, color="blue", linewidth=4)
    plt.ylim(min_distance, max_distance)
    plt.title("E[steps-to-go]", fontsize=18, fontweight="bold")

    ax = plt.gca()
    ax.tick_params(axis="both", which="major", width=2, length=6, labelsize=14)
    ax.tick_params(axis="both", which="minor", width=1.5, length=4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_xlabel("Episode Step", fontsize=16, fontweight="bold")
    ax.set_ylabel("E[steps-to-go]", fontsize=16, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(4)

    plt.subplot(1, 2, 2)
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    xs = np.linspace(min_distance, max_distance, probs.shape[0] + 1, endpoint=True)[:-1]
    plt.bar(xs, probs, width=(max_distance - min_distance) / probs.shape[0] * 0.8, color="blue")
    plt.ylim(0.0, 1.0)
    plt.title(r"p(steps-to-go) at Curr Frame", fontsize=18, fontweight="bold")

    ax = plt.gca()
    ax.tick_params(axis="both", which="major", width=2, length=6, labelsize=14)
    ax.tick_params(axis="both", which="minor", width=1.5, length=4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_xlabel("Step-to-go", fontsize=16, fontweight="bold")
    ax.set_ylabel("Probability", fontsize=16, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(4)

    plt.tight_layout()

    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    image = buf.reshape(int(height), int(width), 4)[:, :, :3]
    plt.close(fig)
    return image


def generate_policy_traj(
    env,
    model,
    dist_converter,
    norm_stats,
    max_distance: int = 140,
    device: str = "cpu",
    title: str = "Policy",
):
    device_t = torch.device(device)
    imgs: List[np.ndarray] = []
    extras: Dict[str, List[np.ndarray]] = {
        "pred_dist_to_succ": [],
        "pred_dist_to_succ_dist_params_logits": [],
    }

    ts = env.reset()
    t = 0
    succ = env.success()
    imgs.append(env.render(title=title))

    with torch.no_grad():
        with eval_mode(model):
            while (not succ) and t < max_distance:
                action, step_extras = rollout_policy_step(
                    model=model,
                    dist_converter=dist_converter,
                    norm_stats=norm_stats,
                    observation=ts.observation,
                    device=device_t,
                )
                extras["pred_dist_to_succ"].append(step_extras["pred_dist_to_succ"][0].detach().cpu().item())
                extras["pred_dist_to_succ_dist_params_logits"].append(step_extras["dist_logits"][0].detach().cpu().numpy())

                ts = env.step(action.detach().cpu().numpy())
                imgs.append(env.render(title=title))
                succ = env.success()
                t += 1

    if len(extras["pred_dist_to_succ"]) == 0:
        zero_logits = np.zeros((dist_converter.num_bins,), dtype=np.float32)
        extras["pred_dist_to_succ"] = [float(max_distance)]
        extras["pred_dist_to_succ_dist_params_logits"] = [zero_logits]

    extras["pred_dist_to_succ"].append(extras["pred_dist_to_succ"][-1])
    extras["pred_dist_to_succ_dist_params_logits"].append(extras["pred_dist_to_succ_dist_params_logits"][-1])

    packed = {
        "pred_dist_to_succ": np.asarray(extras["pred_dist_to_succ"], dtype=np.float32),
        "pred_dist_to_succ_dist_params_logits": np.stack(extras["pred_dist_to_succ_dist_params_logits"]).astype(np.float32),
    }
    return imgs, packed


def generate_distance_plots(all_extras, min_distance: float, max_distance: float):
    imgs = []
    for i in range(all_extras["pred_dist_to_succ"].shape[0]):
        dist_preds = all_extras["pred_dist_to_succ"][: i + 1]
        logits = all_extras["pred_dist_to_succ_dist_params_logits"][i]
        imgs.append(get_distance_plot(dist_preds, logits, min_distance, max_distance))
    return imgs


def get_trajectory_visualization(
    env,
    model,
    dist_converter,
    norm_stats,
    title: str = "Policy",
    min_distance: float = 0,
    max_distance: float = 140,
    device: str = "cpu",
    return_sub_images: bool = False,
):
    env_imgs, all_extras = generate_policy_traj(
        env=env,
        model=model,
        dist_converter=dist_converter,
        norm_stats=norm_stats,
        max_distance=max_distance,
        device=device,
        title=title,
    )
    plot_imgs = generate_distance_plots(all_extras, min_distance=min_distance, max_distance=max_distance)
    full_imgs = [np.concatenate([x, y], axis=1) for x, y in zip(env_imgs, plot_imgs)]

    if return_sub_images:
        return full_imgs, (env_imgs, plot_imgs)
    return full_imgs
