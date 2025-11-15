# Getting Started with Federation-X

Welcome! This guide will help you set up and run Federation-X on your local machine or cluster.

## Prerequisites

- **Python**: 3.8 or higher
- **Git**: For cloning the repository
- **GPU** (optional): NVIDIA GPU with CUDA support for faster training
- **RAM**: 8GB+ recommended
- **Disk Space**: 5GB+ for datasets

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/niranjanxprt/federation-x.git
cd federation-x
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate the environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install the project with all dependencies
pip install -e .

# Verify installation
python -c "from cold_start_hackathon import *; print('✓ Installation successful!')"
```

### Step 4: Verify Setup

```bash
# Check Flower installation
python -c "import flwr; print(f'Flower version: {flwr.__version__}')"

# Check PyTorch
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# Check GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

---

## Quick Start: Local Training

### Run Local Simulation (No GPU needed)

```bash
# Single hospital training
python -m cold_start_hackathon.task --hospital A

# Full federated simulation (3 hospitals)
flwr run . local
```

### Monitor Training

```bash
# Watch logs in real-time
tail -f training.log
```

---

## Cluster Deployment

### Prerequisites for Cluster

1. **Access**: SSH access to cluster
2. **Credentials**: Cluster account and authentication
3. **Environment**: SLURM or similar job scheduler
4. **Dataset**: Preprocessed datasets at `/shared/hackathon/datasets/`

### Deploy to Cluster

```bash
# Submit training job
./submit-job.sh "flwr run . cluster --stream" --gpu --name "fedx_run_001"

# Check job status
squeue -u $USER

# View job logs
tail -f ~/logs/fedx_run_001_*.out

# Cancel job if needed
scancel <job_id>
```

---

## Configuration

### pyproject.toml Settings

The main configuration file is `pyproject.toml`:

```toml
[tool.flwr.app.config]
image-size = 128        # Image size (128 or 224)
num-server-rounds = 9   # Number of federated rounds
local-epochs = 3        # Local training epochs per client
lr = 0.01              # Learning rate
```

### Environment Variables

```bash
# W&B logging
export WANDB_API_KEY="your_key_here"
export WANDB_PROJECT="your_project"
export WANDB_ENTITY="your_entity"

# Cluster paths
export DATASET_DIR="/shared/hackathon/datasets/"
export MODEL_DIR="./models"
export CHECKPOINT_DIR="./checkpoints"
```

---

## Dataset Access

### Local Testing

For local testing with small datasets:

```bash
# Create sample data
python scripts/create_sample_data.py

# This creates a small dataset for testing
```

### Cluster Access

Full datasets are available on cluster:

```bash
# Raw dataset
/shared/hackathon/datasets/xray_fl_datasets/

# Preprocessed (128x128)
/shared/hackathon/datasets/xray_fl_datasets_preprocessed_128/

# Preprocessed (224x224)
/shared/hackathon/datasets/xray_fl_datasets_preprocessed_224/
```

---

## Basic Training Loop

### 1. Initialize Model

```python
from cold_start_hackathon.task import Net

# Create model
model = Net()
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
```

### 2. Load Data

```python
from cold_start_hackathon.task import load_data

# Load hospital data
train_loader = load_data("Hospital_A", "train", batch_size=64)
eval_loader = load_data("Hospital_A", "test", batch_size=64)

print(f"Training batches: {len(train_loader)}")
print(f"Evaluation batches: {len(eval_loader)}")
```

### 3. Train Locally

```python
from cold_start_hackathon.task import train, evaluate

# Train for one epoch
loss = train(model, train_loader, epochs=1, lr=0.01, device='cpu')
print(f"Training loss: {loss:.4f}")

# Evaluate
metrics = evaluate(model, eval_loader, device='cpu')
print(f"Evaluation metrics: {metrics}")
```

---

## Federated Training

### Server Setup

The server aggregates model updates from hospitals:

```bash
# Start server (usually automatic with flwr run)
python -m cold_start_hackathon.server_app
```

### Client Setup

Each hospital runs a client:

```bash
# Client for Hospital A
python -m cold_start_hackathon.client_app --hospital A
```

### Run Full Simulation

```bash
# Automatic setup of 3 clients + server
flwr run . local
```

---

## Monitoring & Evaluation

### Weights & Biases Dashboard

Track experiments in real-time:

```bash
# View dashboard
# https://wandb.ai/your_entity/your_project
```

### Local Evaluation

```bash
# Evaluate best model
python evaluate.py --model models/best_model.pt --split test

# Get detailed metrics
python evaluate.py --model models/best_model.pt --split test --verbose
```

---

## Common Issues & Solutions

### Issue: "No module named 'flwr'"

**Solution:**
```bash
pip install --upgrade flwr
```

### Issue: GPU Memory Error

**Solution:**
```bash
# Reduce batch size in pyproject.toml
# image-size = 128  (was 224)
# Or in code:
load_data(..., batch_size=32)  # was 64
```

### Issue: Dataset Not Found

**Solution:**
```bash
# Check dataset directory
ls -la /shared/hackathon/datasets/

# Set correct path
export DATASET_DIR="/shared/hackathon/datasets/"
```

### Issue: Slow Training

**Solution:**
```bash
# Enable GPU
torch.cuda.is_available()  # Should return True

# Use mixed precision (if supported)
from torch.cuda.amp import autocast

# Increase num_workers
load_data(..., num_workers=4)
```

---

## Next Steps

1. **Read Architecture Guide**: Understand the system design
2. **Explore Code**: Review `cold_start_hackathon/` directory
3. **Run Examples**: Try different configurations
4. **Review Metrics**: Check evaluation results
5. **Experiment**: Modify hyperparameters and algorithms
6. **Deploy**: Run on cluster with optimized settings

---

## Performance Tips

### For Faster Local Training
- Use smaller image size: `image-size = 128`
- Reduce rounds: `num-server-rounds = 3`
- Use CPU if no GPU available

### For Better Accuracy
- Increase local epochs: `local-epochs = 5`
- Use larger batch size: Increase RAM availability
- Train longer: More rounds = better convergence

### For Cluster Optimization
- See [FL_GUIDE_20MIN_UPDATED.md](../FL_GUIDE_20MIN_UPDATED.md)
- Use 20-minute job strategy
- Queue multiple jobs for progression

---

## Getting Help

- **Documentation**: Check [Architecture](./architecture.md) and [FAQ](./faq.md)
- **GitHub Issues**: [Report bugs](https://github.com/niranjanxprt/federation-x/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/niranjanxprt/federation-x/discussions)
- **Code Examples**: See `examples/` directory

---

## What's Next?

Now that you're set up:
- ✅ [Understand the Architecture](./architecture.md)
- ✅ [Learn Training Strategies](./training-guide.md)
- ✅ [Explore Research](./research.md)
- ✅ [Check FAQ](./faq.md)

Good luck! 🚀

---

[← Back Home](index.md) | [Architecture →](./architecture.md)
