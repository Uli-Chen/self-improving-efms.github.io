import numpy as np
import torch
import os
import argparse
from pytorch.env import Point2D

def pd_controller(env: Point2D, t: int, kp: float = 0.01, kd: float = 0.1):
    # Spring towards goal, damp velocity
    vector_to_goal = env._goal_pos - env._cur_pos
    action = kp * vector_to_goal - kd * env._cur_vel
    return action

def generate_expert_data(num_trajs, max_distance, save_path):
    env = Point2D()
    
    all_obs_cur_pos = []
    all_obs_cur_vel = []
    all_obs_goal_pos = []
    all_actions = []
    all_times_to_success = []
    
    for i in range(num_trajs):
        ts = env.reset()
        traj_obs = []
        traj_acts = []
        
        succ = env.success()
        t = 0
        
        while (not succ) and (t < max_distance):
            traj_obs.append(ts.observation)
            act = pd_controller(env, t)
            traj_acts.append(act)
            ts = env.step(act)
            succ = env.success()
            t += 1
            
        if succ:
            # Reverse engineer times to success
            times_to_success = np.arange(len(traj_acts), 0, -1)
            
            for obs, act, time_left in zip(traj_obs, traj_acts, times_to_success):
                all_obs_cur_pos.append(obs['cur_pos'])
                all_obs_cur_vel.append(obs['cur_vel'])
                all_obs_goal_pos.append(obs['goal_pos'])
                all_actions.append(act)
                all_times_to_success.append(time_left)
                
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_trajs} trajectories.")
            
    print(f"Total transitions generated: {len(all_actions)}")
    
    # Save directly to PyTorch tensor format
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'obs_cur_pos': torch.tensor(np.array(all_obs_cur_pos), dtype=torch.float32),
        'obs_cur_vel': torch.tensor(np.array(all_obs_cur_vel), dtype=torch.float32),
        'obs_goal_pos': torch.tensor(np.array(all_obs_goal_pos), dtype=torch.float32),
        'actions': torch.tensor(np.array(all_actions), dtype=torch.float32),
        'times_to_success': torch.tensor(np.array(all_times_to_success), dtype=torch.float32)
    }, save_path)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trajs", type=int, default=1000, help="Number of trajectories to generate")
    parser.add_argument("--max_distance", type=int, default=140, help="Max distance per trajectory")
    parser.add_argument("--save_path", type=str, default="data/expert_data.pt", help="Path to save data (relative to project root)")
    args = parser.parse_args()
    
    # Ensure current working directory is correct when calling
    generate_expert_data(args.num_trajs, args.max_distance, args.save_path)
