# TIMER PointMass PyTorch (Notebook-Aligned)

This repo now provides a notebook-aligned PyTorch pipeline with a unified CLI.

## Structure

- `pytorch/cli.py`: unified experiment CLI entrypoint.
- `pytorch/generate_data.py`: waypoint-based dataset generation (aligned with notebook).
- `pytorch/pipeline/stage1.py`: Stage 1 SFT training loop.
- `pytorch/pipeline/stage2.py`: Stage 2 self-improvement REINFORCE loop.
- `pytorch/pipeline/visualization.py`: policy rollout video utilities.
- `pytorch/pipeline/figure_export.py`: notebook-style final figure exports.
- `pytorch/common/`: shared normalization and rollout helpers.

## Run Experiments

1. Generate data

```bash
uv run python -m pytorch.cli generate-data \
  --save-path data/pointmass_dataset.pt \
  --num-episodes 10000 \
  --num-waypoints-per-episode 5 \
  --episode-len-discard-thresh 10
```

2. Train Stage 1

```bash
uv run python -m pytorch.cli train-stage1 \
  --data-path data/pointmass_dataset_tuples.pt \
  --num-steps 32768 \
  --global-minibatch-size 256 \
  --num-minibatches 128 \
  --wandb-project timer-pointmass \
  --wandb-run-name stage1-baseline
```

3. Visualize Stage 1

```bash
uv run python -m pytorch.cli vis-stage1 \
  --checkpoint checkpoints/stage1_model.pt \
  --num-trajs 10 \
  --out-dir outputs/vis_stage1
```

4. Train Stage 2

```bash
uv run python -m pytorch.cli train-stage2 \
  --checkpoint checkpoints/stage1_model.pt \
  --num-reinforce-sgd-steps 2048 \
  --reinforce-global-batch-size 2048 \
  --reinforce-global-minibatch-size 64 \
  --reinforce-num-minibatches 32 \
  --gamma 0.9 \
  --wandb-project timer-pointmass \
  --wandb-run-name stage2-rl
```

5. Visualize Stage 2

```bash
uv run python -m pytorch.cli vis-stage2 \
  --checkpoint checkpoints/stage2_model.pt \
  --num-trajs 5 \
  --out-dir outputs/vis_stage2
```

6. Export figures

```bash
uv run python -m pytorch.cli make-figures \
  --stage1-ckpt checkpoints/stage1_model.pt \
  --stage2-ckpt checkpoints/stage2_model.pt \
  --trajs-path data/pointmass_dataset_trajs.pt \
  --num-figure-episodes 10 \
  --out-dir outputs/figures
```

## wandb notes

- Stage1/Stage2 training now log metrics to `wandb` by default.
- Use `--wandb-offline` for offline runs.
- Use `--disable-wandb` to disable logging entirely.
