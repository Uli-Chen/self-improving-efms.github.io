import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os

from pytorch.config import get_base_parser
from pytorch.models import TIMERNetwork, DistanceConverter
from pytorch.data import PointMassDataset

def train_stage1(args):
    # Initialize wandb
    if not args.disable_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading datasets...")
    train_dataset = PointMassDataset(args.data_path, split="train")
    val_dataset = PointMassDataset(args.data_path, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    norm_stats = train_dataset.get_norm_stats()

    # Initialize model
    model = TIMERNetwork(
        obs_dim=6, 
        act_dim=2, 
        num_dist_bins=args.num_distance_bins,
        hidden_dims=[args.hidden_dim] * args.num_layers,
        min_act_scale=args.min_act_scale
    ).to(device)

    dist_converter = DistanceConverter(
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        num_bins=args.num_distance_bins,
        device=device
    )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-7)

    # Training loop
    print("Starting training...")
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses, epoch_act_losses, epoch_dist_losses = [], [], []
        
        for batch in train_loader:
            obs = torch.cat([
                batch['obs_cur_pos'], 
                batch['obs_cur_vel'], 
                batch['obs_goal_pos']
            ], dim=-1).to(device)
            
            acts = batch['action'].to(device)
            times_to_success = batch['time_to_success'].to(device)
            
            # Map continuous time to success to discrete bins
            target_bins = dist_converter.distance_to_network_format(times_to_success)
            
            # Forward pass
            preds = model(obs)
            
            # Action loss (Negative Log Likelihood)
            act_log_prob = model.act_log_prob(preds['act_loc'], preds['act_scale'], acts).mean()
            act_loss = -act_log_prob
            
            # Distance loss (Negative Log Likelihood)
            dist_log_prob = model.dist_log_prob(preds['dist_logits'], target_bins).mean()
            dist_loss = -dist_log_prob
            
            # Total loss
            loss = act_loss + dist_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_act_losses.append(act_loss.item())
            epoch_dist_losses.append(dist_loss.item())
            global_step += 1

        # Validation loop
        model.eval()
        val_losses, val_act_losses, val_dist_losses = [], [], []
        
        with torch.no_grad():
            for batch in val_loader:
                obs = torch.cat([
                    batch['obs_cur_pos'], 
                    batch['obs_cur_vel'], 
                    batch['obs_goal_pos']
                ], dim=-1).to(device)
                
                acts = batch['action'].to(device)
                times_to_success = batch['time_to_success'].to(device)
                target_bins = dist_converter.distance_to_network_format(times_to_success)
                
                preds = model(obs)
                
                act_log_prob = model.act_log_prob(preds['act_loc'], preds['act_scale'], acts).mean()
                dist_log_prob = model.dist_log_prob(preds['dist_logits'], target_bins).mean()
                
                val_losses.append((-act_log_prob - dist_log_prob).item())
                val_act_losses.append(-act_log_prob.item())
                val_dist_losses.append(-dist_log_prob.item())

        train_loss = sum(epoch_losses) / len(epoch_losses)
        train_act_loss = sum(epoch_act_losses) / len(epoch_act_losses)
        train_dist_loss = sum(epoch_dist_losses) / len(epoch_dist_losses)
        
        val_loss = sum(val_losses) / len(val_losses)
        val_act_loss = sum(val_act_losses) / len(val_act_losses)
        val_dist_loss = sum(val_dist_losses) / len(val_dist_losses)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if not args.disable_wandb:
            wandb.log({
                "train/loss": train_loss,
                "train/act_loss": train_act_loss,
                "train/dist_loss": train_dist_loss,
                "val/loss": val_loss,
                "val/act_loss": val_act_loss,
                "val/dist_loss": val_dist_loss,
                "epoch": epoch,
                "global_step": global_step
            })

    # Save model
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_path = os.path.join(args.checkpoint_dir, "stage1_model.pt")
    
    # Save the norm stats alongside model weights so we can use it during rollouts
    torch.save({
        'model_state_dict': model.state_dict(),
        'norm_stats': norm_stats,
        'args': args
    }, save_path)
    print(f"Saved model to {save_path}")
    
    if not args.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = get_base_parser()
    parser.add_argument("--data_path", type=str, default="data/expert_data.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    train_stage1(args)
