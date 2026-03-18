import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
import os
import argparse

from pytorch.config import get_base_parser
from pytorch.models import TIMERNetwork, DistanceConverter
from pytorch.env import Point2D
from pytorch.utils import evaluate_timer_rollout_policy, get_trajectory_visualization, eval_mode
from moviepy import ImageSequenceClip

def generate_timer_reinforce_dataset(env, model, dist_converter, norm_stats, num_steps, gamma, max_distance=140, device="cpu"):
    total_steps = 0
    data_tuples = []
    data_stats = []
    
    with torch.no_grad():
        with eval_mode(model):
            while total_steps < num_steps:
                traj_obs_pos = []
                traj_obs_vel = []
                traj_obs_goal = []
                traj_acts = []
                traj_dist_preds = []
                traj_rewards = []
                
                ts = env.reset()
                ts = ts._replace(reward=0.0)
                succ = env.success()
                
                while (not succ) and len(traj_obs_pos) < max_distance:
                    cur_obs = ts.observation
                    
                    obs_pos = torch.tensor(cur_obs['cur_pos'], dtype=torch.float32).unsqueeze(0).to(device)
                    obs_vel = torch.tensor(cur_obs['cur_vel'], dtype=torch.float32).unsqueeze(0).to(device)
                    obs_goal = torch.tensor(cur_obs['goal_pos'], dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # Normalize for network
                    obs_pos_n = (obs_pos - norm_stats['cur_pos_mean'].to(device)) / (norm_stats['cur_pos_std'].to(device) + 1e-8)
                    obs_vel_n = (obs_vel - norm_stats['cur_vel_mean'].to(device)) / (norm_stats['cur_vel_std'].to(device) + 1e-8)
                    obs_goal_n = (obs_goal - norm_stats['cur_pos_mean'].to(device)) / (norm_stats['cur_pos_std'].to(device) + 1e-8)
                    
                    obs = torch.cat([obs_pos_n, obs_vel_n, obs_goal_n], dim=-1)
                    
                    preds = model(obs)
                    dist_pred = dist_converter.network_format_to_distance(preds['dist_logits'])
                    norm_act = model.sample_act(preds['act_loc'], preds['act_scale'])
                    unnorm_act = norm_act * (norm_stats['act_std'].to(device) + 1e-8) + norm_stats['act_mean'].to(device)
                    
                    traj_obs_pos.append(obs_pos_n[0])
                    traj_obs_vel.append(obs_vel_n[0])
                    traj_obs_goal.append(obs_goal_n[0])
                    traj_acts.append(norm_act[0])
                    traj_dist_preds.append(dist_pred[0])
                    
                    ts = env.step(unnorm_act[0].cpu().numpy())
                    traj_rewards.append(ts.reward)
                    
                    succ = env.success()
                
                episode_len = len(traj_obs_pos)
                total_steps += episode_len
                
                episode_stats = {
                    'success': succ,
                    'return': sum(traj_rewards),
                    'len': episode_len
                }
                data_stats.append(episode_stats)
                
                # Compute returns to go
                returns_to_go = []
                ret = 0
                for r in reversed(traj_rewards):
                    ret = r + gamma * ret
                    returns_to_go.append(ret)
                returns_to_go.reverse()
                
                # Store tuple
                for i in range(episode_len):
                    data_tuples.append({
                        'obs_pos': traj_obs_pos[i],
                        'obs_vel': traj_obs_vel[i],
                        'obs_goal': traj_obs_goal[i],
                        'action': traj_acts[i],
                        'weight': returns_to_go[i]
                    })
                    
    # Log stats
    if len(data_stats) > 0:
        print(f"Collected {len(data_stats)} episodes.")
        print(f"Success Rate: {np.mean([s['success'] for s in data_stats]):.2f}")
        print(f"Returns: {np.mean([s['return'] for s in data_stats]):.2f}")
        print(f"Lengths: {np.mean([s['len'] for s in data_stats]):.2f}")
        
    return data_tuples, data_stats

def train_stage2(args):
    # Initialize wandb
    if not args.disable_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Stage 1 model and stats
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    norm_stats = checkpoint['norm_stats']
    
    # Check if network params are in kwargs or just use the ones from config
    model = TIMERNetwork(
        obs_dim=6, 
        act_dim=2, 
        num_dist_bins=args.num_distance_bins,
        hidden_dims=[args.hidden_dim] * args.num_layers,
        min_act_scale=args.min_act_scale
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    dist_converter = DistanceConverter(
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        num_bins=args.num_distance_bins,
        device=device
    )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-7)
    env = Point2D(physics_substeps=args.env_physics_substeps, success_radius=args.env_success_radius)

    # Training loop
    print("Starting Stage 2 RL training...")
    
    for iteration in range(args.iterations):
        model.train()
        
        # 1. Collect dataset
        data_tuples, data_stats = generate_timer_reinforce_dataset(
            env=env,
            model=model,
            dist_converter=dist_converter,
            norm_stats=norm_stats,
            num_steps=args.steps_per_iter,
            gamma=args.gamma,
            max_distance=args.max_distance,
            device=device
        )
        
        # We process the dataset into tensors
        obs_batch = torch.stack([
            torch.cat([t['obs_pos'], t['obs_vel'], t['obs_goal']], dim=-1) for t in data_tuples
        ]).to(device)
        act_batch = torch.stack([t['action'] for t in data_tuples]).to(device)
        weight_batch = torch.tensor([t['weight'] for t in data_tuples], dtype=torch.float32).to(device)
        
        # Baseline per step
        baseline = weight_batch.mean()
        adv_batch = weight_batch - baseline
        
        # Standardize advantages
        if args.standardize_adv:
            adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)
            
        # We do a single full-batch update per REINFORCE iteration, identical to the JAX implementation
        
        preds = model(obs_batch)
        
        # REINFORCE loss: -1 * adv * log_prob
        log_prob = model.act_log_prob(preds['act_loc'], preds['act_scale'], act_batch)
        policy_loss = -(adv_batch * log_prob).mean()
        
        # Entropy bonus
        entropy = model.act_entropy(preds['act_loc'], preds['act_scale']).mean()
        entropy_loss = -args.entropy_coef * entropy
        
        total_loss = policy_loss + entropy_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        
        train_policy_loss = policy_loss.item()
        train_entropy_loss = entropy_loss.item()
        
        print(f"Iter {iteration+1}/{args.iterations} | Policy Loss: {train_policy_loss:.4f} | Entropy: {train_entropy_loss:.4f}")
        
        if not args.disable_wandb:
            wandb.log({
                "rl/success_rate": np.mean([s['success'] for s in data_stats]),
                "rl/return_mean": np.mean([s['return'] for s in data_stats]),
                "rl/len_mean": np.mean([s['len'] for s in data_stats]),
                "rl/policy_loss": train_policy_loss,
                "rl/entropy_loss": train_entropy_loss,
                "rl/iteration": iteration
            })
            
        # Evaluation and save
        if (iteration + 1) % args.eval_freq == 0:
            print("Evaluating...")
            eval_metrics = evaluate_timer_rollout_policy(env, model, dist_converter, norm_stats, num_episodes=25, max_distance=args.max_distance, device=device)
            
            # Generate video visualization
            video_imgs = get_trajectory_visualization(env, model, dist_converter, norm_stats, title=f"RL Iter {iteration+1}", min_distance=args.min_distance, max_distance=args.max_distance, device=device)
            
            if not args.disable_wandb:
                wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})
                
                if len(video_imgs) > 0:
                    # Wandb requires video shape as (time, channel, height, width) [0-255]
                    # Also needs fps
                    video_tensor = np.array(video_imgs).transpose(0, 3, 1, 2)
                    wandb.log({"eval/video": wandb.Video(video_tensor, fps=10, format="mp4")})
                    
            # Save checkpoint
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            save_path = os.path.join(args.checkpoint_dir, "stage2_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'norm_stats': norm_stats,
                'args': args
            }, save_path)

    if not args.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to Stage 1 checkpoint")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--steps_per_iter", type=int, default=4000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--ppo_epochs", type=int, default=1) # number of updates over collected data
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--standardize_adv", action="store_true", default=True, help="Standardize advantages")
    parser.add_argument("--eval_freq", type=int, default=5, help="Iterations between evaluations")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    train_stage2(args)
