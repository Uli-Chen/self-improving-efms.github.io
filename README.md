# TIMER PointMass PyTorch Implementation

This repository contains the PyTorch migration of the TIMER (Self-Improving Episode-Focused Models) agent for a 2D PointMass environment, originally implemented in JAX and TensorFlow. The project is organized as a modular Python package supporting command-line execution and modern experiment tracking.

## Project Structure

The codebase is organized into a modular structure for clarity and maintainability:

- `pytorch/config.py`: Centralized configuration management using argparse for hyperparameter tuning and experiment setup.
- `pytorch/env.py`: The Point2D environment implementation using the dm_env interface, including custom rendering logic.
- `pytorch/data.py`: PyTorch Dataset implementation and data normalization utilities.
- `pytorch/models.py`: Neural network architectures for the TIMER agent, including the Action Head (Normal distribution) and Distance Head (Categorical distribution).
- `pytorch/utils.py`: Support functions for policy evaluation, trajectory generation, and video visualization.
- `pytorch/generate_data.py`: CLI script for generating expert demonstration data using a PD controller.
- `pytorch/train_stage1.py`: CLI script for Stage 1 Supervised Fine-Tuning (Behavioral Cloning).
- `pytorch/train_stage2.py`: CLI script for Stage 2 Reinforcement Learning Fine-Tuning (REINFORCE).

## Requirements and Installation

This project uses `uv` for fast and reliable dependency management.

1. Install `uv` if it is not already available on your system.
2. Clone the repository and navigate to the project root.
3. Synchronize the environment and install dependencies:
   ```bash
   uv sync
   ```
   This will create a `.venv` directory with all necessary packages including PyTorch, WandB, and MoviePy.

## Experiment Execution

The pipeline consists of three sequential stages.

### 1. Data Generation

Generate expert trajectories to be used for initial supervised training:
```bash
uv run python -m pytorch.generate_data --num_trajs 1000 --save_path data/expert_data.pt
```

### 2. Stage 1: Supervised Fine-Tuning (SFT)

Train the model using Behavioral Cloning on the generated expert data:
```bash
uv run python -m pytorch.train_stage1 --data_path data/expert_data.pt --epochs 100 --batch_size 256
```
Metrics and checkpoints will be saved to the `checkpoints/` directory.

### 3. Stage 2: Reinforcement Learning (RL)

Fine-tune the Stage 1 model using the REINFORCE algorithm:
```bash
uv run python -m pytorch.train_stage2 --checkpoint checkpoints/stage1_model.pt --iterations 100 --steps_per_iter 4000
```

## Features and Integrations

- **Configuration Management**: All hyperparameters for the environment, network, and training loops are exposed via CLI arguments in `config.py`.
- **Experiment Tracking**: Weights & Biases (WandB) is integrated for real-time logging of losses, success rates, and evaluation videos. You can disable it using the `--disable_wandb` flag.
- **Parallel Training**: The CLI design allows for launching multiple independent training runs. Parallel hyperparameter optimization can be easily achieved using WandB Sweeps.
- **Visual Evaluation**: Training scripts automatically generate and log trajectory videos that visualize the agent's movement alongside its internal "steps-to-go" probability distributions.

## Migration Details

The original JAX-based implementation was refactored into this PyTorch version with the following improvements:
- Replaced JAX/Haiku neural networks with `torch.nn.Module`.
- Converted TensorFlow Data pipelines to PyTorch `DataLoader` and `Dataset`.
- Decoupled the environment logic from the training loops for better modularity.
- Implemented a unified configuration system to support robust CLI experimentation.
