# Self-Improving EFMs: PointMass (JAX)

This repository contains experimental code and notebooks for PointMass environment studies using JAX and uv-managed environments.

---

## Results Overview

This repository is part of a technical assessment project. The included result files demonstrate a reproduction of the expected behaviors and experimental pipeline.

- `pointmass_dataset_trajs.pkl` and `pointmass_dataset_tuples.pkl` contain generated trajectory data and processed training tuples derived from the PointMass environment.
- `__temp__.mp4` provides a visualization of the agent behavior and trajectory evolution during the experiment.
- The `.png` files present qualitative results, including environment states, trajectory distributions, and self-improvement outcomes across different stages.

These artifacts are produced by running the provided code and notebook, and serve as a reference for verifying correctness and reproducibility.

---

## Environment Setup

This project uses **uv** for dependency and environment management.

### 1. Install dependencies

```bash
uv sync
```

This will:

- Create a virtual environment (`.venv`)
- Install all dependencies specified in `pyproject.toml`
- Ensure reproducibility via `uv.lock`

---

### 2. Register Jupyter kernel

```bash
uv run python -m ipykernel install --user --name=pointmass --display-name "PointMass (uv)"
```

This registers the project environment as a selectable kernel in Jupyter.

---

### 3. Launch Jupyter Notebook

```bash
uv run jupyter notebook
```

Then open your notebook and select:

```
PointMass (uv)
```
