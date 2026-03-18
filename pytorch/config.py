import argparse

def get_base_parser():
    parser = argparse.ArgumentParser(description="TIMER PointMass Agent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="timer-pointmass", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable WandB logging")
    
    # Environment params
    parser.add_argument("--env_physics_substeps", type=int, default=10, help="Physics substeps")
    parser.add_argument("--env_success_radius", type=float, default=0.15, help="Success radius")
    
    # Distance Converter params
    parser.add_argument("--min_distance", type=float, default=0.0)
    parser.add_argument("--max_distance", type=float, default=140.0)
    parser.add_argument("--num_distance_bins", type=int, default=50)

    # Network params
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--min_act_scale", type=float, default=1e-4)

    return parser
