# Self-Improving EFMs: PointMass (JAX)

This repository is part of a technical assessment project.

- `pointmass_dataset_trajs.pkl` and `pointmass_dataset_tuples.pkl` contain generated trajectory data and processed training tuples derived from the PointMass environment.
- `__temp__.mp4` provides a visualization of the agent behavior and trajectory evolution during the experiment.
- The `.png` files present qualitative results, including environment states, trajectory distributions, and self-improvement outcomes across different stages.

These artifacts are produced by running the provided code and notebook, and serve as a reference for verifying correctness and reproducibility.
---


### Modifications

We made a single modification to the original implementation: replacing the deprecated tostring_rgb function (removed in Matplotlib ≥ 3.8) with buffer_rgba for image extraction from the rendering canvas.

---

### Option 1: Docker (Recommended)

This project provides a Docker configuration for a one-click environment setup that is isolated from your local machine's configuration. It strictly follows the versions specified in `uv.lock`.

1. **Run with Docker Compose:**
   The following scripts will build the image (if needed) and start the environment:
   ```bash
   ./run_docker.sh  # On Linux/macOS
   run_docker.bat   # On Windows
   ```
   Or use Docker Compose directly:
   ```bash
   docker compose up --build
   ```

2. **Access Jupyter Notebook:**
   Once the container is running, open your browser and go to:
   [http://localhost:8888](http://localhost:8888)
   (Password/Token: `pointmass`)

---

### Option 2: Local Setup (uv)

This project uses **uv** for dependency and environment management.

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Register Jupyter kernel:**
   ```bash
   uv run python -m ipykernel install --user --name=pointmass --display-name "PointMass (uv)"
   ```

3. **Launch Jupyter Notebook:**
   ```bash
   uv run jupyter notebook
   ```
   Then open your notebook and select `PointMass (uv)` as the Kernel.
