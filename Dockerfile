# Use Python 3.12-slim as the base image
FROM python:3.12-slim

# Use Aliyun mirrors for faster apt access in China
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv via pip (more robust for some network environments)
RUN pip install --no-cache-dir uv -i https://pypi.tuna.tsinghua.edu.cn/simple

# Set the working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
# --frozen ensures we use the exact versions from uv.lock
RUN uv sync --frozen

# Copy the rest of the application code
COPY . .

# Register Jupyter kernel
RUN uv run python -m ipykernel install --user --name=pointmass --display-name "PointMass (Docker)"

# Expose Jupyter port
EXPOSE 8888

# Command to run Jupyter Notebook with a fixed token
CMD ["uv", "run", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=pointmass"]
