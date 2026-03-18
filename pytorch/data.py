import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import os
from .env import Point2D

class PointMassDataset(Dataset):
    def __init__(self, data_path, split="train", train_ratio=0.9):
        super().__init__()
        self.data_path = data_path
        
        # Load from npz or pt
        if data_path.endswith('.npz'):
            data = np.load(data_path)
            self.obs_cur_pos = torch.tensor(data['obs_cur_pos'], dtype=torch.float32)
            self.obs_cur_vel = torch.tensor(data['obs_cur_vel'], dtype=torch.float32)
            self.obs_goal_pos = torch.tensor(data['obs_goal_pos'], dtype=torch.float32)
            self.actions = torch.tensor(data['actions'], dtype=torch.float32)
            self.times_to_success = torch.tensor(data['times_to_success'], dtype=torch.float32)
        elif data_path.endswith('.pt'):
            data = torch.load(data_path, map_location='cpu')
            self.obs_cur_pos = data['obs_cur_pos']
            self.obs_cur_vel = data['obs_cur_vel']
            self.obs_goal_pos = data['obs_goal_pos']
            self.actions = data['actions']
            self.times_to_success = data['times_to_success']
        else:
            raise ValueError("Unsupported data format. Use .npz or .pt")
            
        total_size = len(self.actions)
        train_size = int(total_size * train_ratio)
        
        if split == "train":
            self.obs_cur_pos = self.obs_cur_pos[:train_size]
            self.obs_cur_vel = self.obs_cur_vel[:train_size]
            self.obs_goal_pos = self.obs_goal_pos[:train_size]
            self.actions = self.actions[:train_size]
            self.times_to_success = self.times_to_success[:train_size]
            
            # Compute stats for normalization
            self.cur_pos_mean = self.obs_cur_pos.mean(dim=0, keepdim=True)
            self.cur_pos_std = self.obs_cur_pos.std(dim=0, keepdim=True)
            self.cur_vel_mean = self.obs_cur_vel.mean(dim=0, keepdim=True)
            self.cur_vel_std = self.obs_cur_vel.std(dim=0, keepdim=True)
            self.act_mean = self.actions.mean(dim=0, keepdim=True)
            self.act_std = self.actions.std(dim=0, keepdim=True)
            
        elif split == "val":
            # For validation, we still need to normalize using train stats. 
            # In a clean setup we compute train stats first, but here we can just compute on whole dataset or pass them.
            # To be simple and robust, we compute on the train split here.
            train_obs_cur_pos = self.obs_cur_pos[:train_size]
            train_obs_cur_vel = self.obs_cur_vel[:train_size]
            train_actions = self.actions[:train_size]
            
            self.cur_pos_mean = train_obs_cur_pos.mean(dim=0, keepdim=True)
            self.cur_pos_std = train_obs_cur_pos.std(dim=0, keepdim=True)
            self.cur_vel_mean = train_obs_cur_vel.mean(dim=0, keepdim=True)
            self.cur_vel_std = train_obs_cur_vel.std(dim=0, keepdim=True)
            self.act_mean = train_actions.mean(dim=0, keepdim=True)
            self.act_std = train_actions.std(dim=0, keepdim=True)
            
            self.obs_cur_pos = self.obs_cur_pos[train_size:]
            self.obs_cur_vel = self.obs_cur_vel[train_size:]
            self.obs_goal_pos = self.obs_goal_pos[train_size:]
            self.actions = self.actions[train_size:]
            self.times_to_success = self.times_to_success[train_size:]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Apply normalization
        self.obs_cur_pos = (self.obs_cur_pos - self.cur_pos_mean) / (self.cur_pos_std + 1e-8)
        self.obs_cur_vel = (self.obs_cur_vel - self.cur_vel_mean) / (self.cur_vel_std + 1e-8)
        self.obs_goal_pos = (self.obs_goal_pos - self.cur_pos_mean) / (self.cur_pos_std + 1e-8)
        self.actions = (self.actions - self.act_mean) / (self.act_std + 1e-8)
            
    def __len__(self):
        return len(self.actions)
        
    def __getitem__(self, idx):
        return {
            'obs_cur_pos': self.obs_cur_pos[idx],
            'obs_cur_vel': self.obs_cur_vel[idx],
            'obs_goal_pos': self.obs_goal_pos[idx],
            'action': self.actions[idx],
            'time_to_success': self.times_to_success[idx]
        }
        
    def get_norm_stats(self):
        return {
            'cur_pos_mean': self.cur_pos_mean,
            'cur_pos_std': self.cur_pos_std,
            'cur_vel_mean': self.cur_vel_mean,
            'cur_vel_std': self.cur_vel_std,
            'act_mean': self.act_mean,
            'act_std': self.act_std
        }
