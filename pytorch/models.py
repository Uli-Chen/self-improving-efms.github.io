import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

class DistanceConverter:
    def __init__(self, min_distance: float, max_distance: float, num_bins: int = 50, device="cpu"):
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.num_bins = num_bins
        self.bin_size = (max_distance - min_distance) / num_bins
        self.device = device
        
        # dist_vals represents the start value of each bin
        self.dist_vals = torch.linspace(
            min_distance, max_distance, num_bins + 1, device=device
        )[:-1]

    def distance_to_network_format(self, d: torch.Tensor) -> torch.Tensor:
        """Converts continuous distance to discrete bin indices."""
        d = torch.clamp(d, self.min_distance, self.max_distance - self.bin_size / 2.0)
        bin_index = torch.floor_divide(d - self.min_distance, self.bin_size).long()
        return bin_index

    def network_format_to_distance(self, logits: torch.Tensor) -> torch.Tensor:
        """Converts network logits back to expected continuous distance."""
        probs = F.softmax(logits, dim=-1)
        # Expected value
        dist = torch.sum(self.dist_vals * probs, dim=-1)
        return dist


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            curr_dim = h_dim
        self.net = nn.Sequential(*layers)
        self.output_dim = curr_dim
        
    def forward(self, x):
        return self.net(x)


class TIMERNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, num_dist_bins: int, 
                 hidden_dims: list = [256, 256, 256], min_act_scale: float = 1e-4):
        super().__init__()
        self.act_dim = act_dim
        self.num_dist_bins = num_dist_bins
        self.min_act_scale = min_act_scale
        
        # Action Head
        self.act_mlp = MLP(obs_dim, hidden_dims)
        self.act_loc = nn.Linear(self.act_mlp.output_dim, act_dim)
        self.act_scale = nn.Linear(self.act_mlp.output_dim, act_dim)
        
        # Initialize final layers for action near zero carefully
        nn.init.orthogonal_(self.act_loc.weight, gain=1e-4) # VarianceScaling eq
        nn.init.constant_(self.act_loc.bias, 0.0)
        nn.init.orthogonal_(self.act_scale.weight, gain=1e-4)
        nn.init.constant_(self.act_scale.bias, 0.0)

        # Distance Head
        self.dist_mlp = MLP(obs_dim, hidden_dims)
        self.dist_logits = nn.Linear(self.dist_mlp.output_dim, num_dist_bins, bias=False)

    def forward(self, obs):
        # Action distribution
        h_act = self.act_mlp(obs)
        loc = self.act_loc(h_act)
        scale = F.softplus(self.act_scale(h_act)) + self.min_act_scale
        
        # Distance logits
        h_dist = self.dist_mlp(obs)
        logits = self.dist_logits(h_dist)
        
        return {
            'act_loc': loc,
            'act_scale': scale,
            'dist_logits': logits
        }

    def get_action_dist(self, loc, scale):
        return Normal(loc, scale)
        
    def get_distance_dist(self, logits):
        return Categorical(logits=logits)
        
    def act_log_prob(self, loc, scale, action):
        dist = self.get_action_dist(loc, scale)
        # Sum over action dims for independent normals
        return dist.log_prob(action).sum(dim=-1)
        
    def act_entropy(self, loc, scale):
        dist = self.get_action_dist(loc, scale)
        return dist.entropy().sum(dim=-1)
        
    def sample_act(self, loc, scale):
        dist = self.get_action_dist(loc, scale)
        return dist.sample()
        
    def sample_act_mode(self, loc, scale):
        return loc # Mode of Normal is its mean
        
    def dist_log_prob(self, logits, target_bins):
        dist = self.get_distance_dist(logits)
        return dist.log_prob(target_bins)
        
    def dist_entropy(self, logits):
        dist = self.get_distance_dist(logits)
        return dist.entropy()
        
    def sample_dist(self, logits):
        dist = self.get_distance_dist(logits)
        return dist.sample()

    def sample_dist_mode(self, logits):
        return torch.argmax(logits, dim=-1)
