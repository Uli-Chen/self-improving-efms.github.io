import torch
import numpy as np
import os
from contextlib import contextmanager

@contextmanager
def eval_mode(model):
    """Context manager for model evaluation mode."""
    prev_mode = model.training
    try:
        model.eval()
        yield model
    finally:
        model.train(prev_mode)

def evaluate_timer_rollout_policy(env, model, dist_converter, norm_stats, num_episodes=25, max_distance=140, device="cpu", seed=42):
    stats = []
    
    # Initialize random generator for reproducible evaluations if needed
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)
        
    with torch.no_grad():
        with eval_mode(model):
            for eps_num in range(num_episodes):
                timesteps = []
                ts = env.reset()
                ts = ts._replace(reward=0.0)
                timesteps.append(ts)
                
                while (not env.success()) and len(timesteps) < max_distance:
                    cur_obs = ts.observation
                    
                    # Convert to tensor and add batch dim
                    obs_pos = torch.tensor(cur_obs['cur_pos'], dtype=torch.float32).unsqueeze(0).to(device)
                    obs_vel = torch.tensor(cur_obs['cur_vel'], dtype=torch.float32).unsqueeze(0).to(device)
                    obs_goal = torch.tensor(cur_obs['goal_pos'], dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # Normalize
                    obs_pos = (obs_pos - norm_stats['cur_pos_mean'].to(device)) / (norm_stats['cur_pos_std'].to(device) + 1e-8)
                    obs_vel = (obs_vel - norm_stats['cur_vel_mean'].to(device)) / (norm_stats['cur_vel_std'].to(device) + 1e-8)
                    obs_goal = (obs_goal - norm_stats['cur_pos_mean'].to(device)) / (norm_stats['cur_pos_std'].to(device) + 1e-8)
                    
                    obs = torch.cat([obs_pos, obs_vel, obs_goal], dim=-1)
                    
                    preds = model(obs)
                    
                    # Dist converter returns actual float value
                    dist_pred = dist_converter.network_format_to_distance(preds['dist_logits'])
                    
                    # Act sample
                    norm_act = model.sample_act(preds['act_loc'], preds['act_scale'])
                    
                    # Unnormalize action for env
                    unnorm_act = norm_act * (norm_stats['act_std'].to(device) + 1e-8) + norm_stats['act_mean'].to(device)
                    
                    ts = env.step(unnorm_act[0].cpu().numpy())
                    timesteps.append(ts)
                    
                episode_stats = {}
                episode_stats['success'] = env.success()
                episode_stats['return'] = sum(x.reward for x in timesteps)
                episode_stats['len'] = len(timesteps)
                stats.append(episode_stats)
                
    success_rate = np.mean([s['success'] for s in stats])
    returns = [s['return'] for s in stats]
    lengths = [s['len'] for s in stats]
    
    metrics = {
        'success_rate': success_rate,
        'return_mean': np.mean(returns),
        'return_std': np.std(returns),
        'len_mean': np.mean(lengths),
        'len_std': np.std(lengths),
        'max_len': np.max(lengths),
        'min_len': np.min(lengths)
    }
    
    return metrics

def generate_policy_traj(env, model, dist_converter, norm_stats, max_distance=140, device="cpu", title="Policy"):
    imgs = []
    all_extras = {
        'pred_dist_to_succ': [],
        'pred_dist_to_succ_dist_params_logits': []
    }
    
    ts = env.reset()
    t = 0
    succ = env.success()
    
    imgs.append(env.render(title=title))
    
    with torch.no_grad():
        with eval_mode(model):
            while (not succ) and t < max_distance:
                cur_obs = ts.observation
                
                obs_pos = torch.tensor(cur_obs['cur_pos'], dtype=torch.float32).unsqueeze(0).to(device)
                obs_vel = torch.tensor(cur_obs['cur_vel'], dtype=torch.float32).unsqueeze(0).to(device)
                obs_goal = torch.tensor(cur_obs['goal_pos'], dtype=torch.float32).unsqueeze(0).to(device)
                
                obs_pos = (obs_pos - norm_stats['cur_pos_mean'].to(device)) / (norm_stats['cur_pos_std'].to(device) + 1e-8)
                obs_vel = (obs_vel - norm_stats['cur_vel_mean'].to(device)) / (norm_stats['cur_vel_std'].to(device) + 1e-8)
                obs_goal = (obs_goal - norm_stats['cur_pos_mean'].to(device)) / (norm_stats['cur_pos_std'].to(device) + 1e-8)
                
                obs = torch.cat([obs_pos, obs_vel, obs_goal], dim=-1)
                
                preds = model(obs)
                dist_pred = dist_converter.network_format_to_distance(preds['dist_logits'])
                norm_act = model.sample_act(preds['act_loc'], preds['act_scale'])
                unnorm_act = norm_act * (norm_stats['act_std'].to(device) + 1e-8) + norm_stats['act_mean'].to(device)
                
                all_extras['pred_dist_to_succ'].append(dist_pred[0].item())
                all_extras['pred_dist_to_succ_dist_params_logits'].append(preds['dist_logits'][0].cpu().numpy())
                
                ts = env.step(unnorm_act[0].cpu().numpy())
                imgs.append(env.render(title=title))
                
                succ = env.success()
                t += 1
                
    # Repeat the last extra to match imgs len
    if len(all_extras['pred_dist_to_succ']) > 0:
        all_extras['pred_dist_to_succ'].append(all_extras['pred_dist_to_succ'][-1])
        all_extras['pred_dist_to_succ_dist_params_logits'].append(all_extras['pred_dist_to_succ_dist_params_logits'][-1])
        
    all_extras['pred_dist_to_succ'] = np.array(all_extras['pred_dist_to_succ'])
    all_extras['pred_dist_to_succ_dist_params_logits'] = np.stack(all_extras['pred_dist_to_succ_dist_params_logits'])
    
    
    return imgs, all_extras

def get_distance_plot(distances, logits, min_distance: float, max_distance: float):
    # This renders the distance plot next to the environment rendering
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import torch.nn.functional as F

    DPI = 200
    RENDER_HEIGHT_INCHES = 5

    fig, ax = plt.subplots(figsize=(2 * RENDER_HEIGHT_INCHES, RENDER_HEIGHT_INCHES), dpi=DPI)
    plt.clf()

    plt.subplot(1, 2, 1)
    plt.plot(distances, color='blue', linewidth=4)
    plt.ylim(min_distance, max_distance)
    plt.title('E[steps-to-go]', fontsize=18, fontweight='bold')

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', width=2, length=6, labelsize=14)
    ax.tick_params(axis='both', which='minor', width=1.5, length=4)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.set_xlabel('Episode Step', fontsize=16, fontweight='bold')
    ax.set_ylabel('E[steps-to-go]', fontsize=16, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_linewidth(4)

    plt.subplot(1, 2, 2)
    # Ensure logits are PyTorch tensors to use softmax
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    
    plt.bar(
        np.linspace(min_distance, max_distance, probs.shape[0] + 1, endpoint=True)[:-1],
        probs,
        width=(max_distance - min_distance) / probs.shape[0] * 0.8,
        color='blue'
    )
    plt.ylim(0., 1.)
    plt.title(r'p(steps-to-go) at Curr Frame', fontsize=18, fontweight='bold')

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', width=2, length=6, labelsize=14)
    ax.tick_params(axis='both', which='minor', width=1.5, length=4)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.set_xlabel('Step-to-go', fontsize=16, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=16, fontweight='bold')
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

def generate_distance_plots(all_extras, min_distance, max_distance):
    imgs = []
    for i in range(all_extras['pred_dist_to_succ'].shape[0]):
        dist_preds = all_extras['pred_dist_to_succ'][:i+1]
        logits = all_extras['pred_dist_to_succ_dist_params_logits'][i]
        img = get_distance_plot(dist_preds, logits, min_distance, max_distance)
        imgs.append(img)
    return imgs

def get_trajectory_visualization(env, model, dist_converter, norm_stats, title="Policy", min_distance=0, max_distance=140, device="cpu"):
    imgs, all_extras = generate_policy_traj(env, model, dist_converter, norm_stats, max_distance, device, title)
    if len(imgs) == 0:
        return []
    
    # We might only have bounds for rendering distance if it succeeded? Wait, the model produces it at each step
    plot_imgs = generate_distance_plots(all_extras, min_distance, max_distance)
    video_imgs = []
    for (x, y) in zip(imgs, plot_imgs):
        video_imgs.append(np.concatenate([x, y], axis=1))

    return video_imgs
