# System Architecture

This document describes the technical architecture of Federation-X.

## Overview

Federation-X is built on the **Flower Federated Learning Framework** with a client-server architecture where hospitals act as federated clients and a central server aggregates their model updates.

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEDERATED LEARNING SYSTEM                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Hospital A   │    │ Hospital B   │    │ Hospital C   │       │
│  │ (Client)     │    │ (Client)     │    │ (Client)     │       │
│  │              │    │              │    │              │       │
│  │ • Local Data │    │ • Local Data │    │ • Local Data │       │
│  │ • Local GPU  │    │ • Local GPU  │    │ • Local GPU  │       │
│  │ • Local Train│    │ • Local Train│    │ • Local Train│       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                    │                    │               │
│         └────────────────────┼────────────────────┘               │
│                              │                                    │
│                        Model Updates                             │
│                        (Encrypted)                               │
│                              │                                    │
│                         ┌────▼────┐                              │
│                         │ Server   │                              │
│                         │(Central) │                              │
│                         │          │                              │
│                         │ • Fetch  │                              │
│                         │ • Agg.   │                              │
│                         │ • Eval.  │                              │
│                         └────┬────┘                               │
│                              │                                    │
│                         Updated Model                            │
│                              │                                    │
│         ┌────────────────────┼────────────────────┐              │
│         │                    │                    │              │
│  ┌──────▼───────┐    ┌──────▼───────┐    ┌──────▼───────┐      │
│  │Hospital A    │    │ Hospital B   │    │ Hospital C   │      │
│  │ (Next Round) │    │ (Next Round) │    │ (Next Round) │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Server Application (`server_app.py`)

**Responsibility**: Orchestrates federated training and model aggregation

**Key Functions**:
- **Initialization**: Creates global model and Flower server
- **Client Selection**: Selects available hospitals for training
- **Aggregation**: Combines hospital updates using FedProx
- **Evaluation**: Evaluates aggregated model on hospital test sets
- **Checkpointing**: Saves best models and intermediate states
- **Logging**: Tracks metrics to Weights & Biases

**Strategy**: FedProx with configurable proximal term (μ)
```python
# FedProx penalty term
L_prox = (μ / 2) * ||w - w_t||²
```

**Aggregation Logic**:
```
For each round t:
  1. Select K hospitals from M total
  2. Send current model w_t to selected hospitals
  3. Wait for hospitals to train locally
  4. Receive updated weights w_t^i from each hospital
  5. Aggregate: w_{t+1} = (1/K) * Σ w_t^i
  6. Evaluate on test sets
  7. Log metrics
  8. Save checkpoint
```

### 2. Client Application (`client_app.py`)

**Responsibility**: Represents a hospital in federated training

**Key Functions**:
- **Model Download**: Receives aggregated model from server
- **Local Training**: Trains on local hospital data
- **Gradient Computation**: Calculates weight updates
- **Model Upload**: Sends updated weights to server
- **Evaluation**: Evaluates on local test set

**Data Pipeline**:
```
Hospital Data
    ↓
Data Loading (128x128 images)
    ↓
Preprocessing (normalization)
    ↓
Mini-batch Creation
    ↓
Training Loop
    ↓
Weight Update
```

### 3. Task Module (`task.py`)

**Responsibility**: Core ML components (model, training, evaluation)

**Key Classes**:

#### Net (Model Architecture)
```python
class Net(nn.Module):
    """
    ResNet18-based model for binary classification

    Architecture:
    - Input: 128x128 grayscale X-ray
    - Backbone: ResNet18 (pretrained on ImageNet)
    - Head: Linear(512 → 1) for binary classification

    Forward pass:
    X-ray → Conv layers → Feature extraction → Classification
    """
```

**Key Design Decisions**:
- **Pre-trained Weights**: ImageNet weights for faster convergence
- **Grayscale Input**: 1 channel for X-rays (adapted from 3-channel RGB)
- **Binary Output**: Single neuron with sigmoid for pathology presence

#### Training Loop
```python
def train(net, trainloader, epochs, lr, device):
    """
    Local training at hospital

    Algorithm:
    For each epoch:
      For each batch (X, y):
        1. Forward pass: y_pred = model(X)
        2. Compute loss: L = FocalLoss(y_pred, y)
        3. Backward: ∂L/∂w
        4. Optimizer step: w := w - lr * ∂L/∂w
        5. Update learning rate scheduler

    Returns: Average loss over all batches
    """
```

**Loss Function**: Focal Loss
```
FL(pt) = -α(1-pt)^γ * log(pt)

Where:
- pt: probability of correct class
- α: weighting factor (0.25 for positives)
- γ: focusing parameter (2.0)

Benefit: Down-weights easy examples, focuses on hard negatives
```

**Optimizer**: AdamW
```python
AdamW(lr=0.01, weight_decay=0.01)
```

#### Evaluation Function
```python
def evaluate(net, testloader, device):
    """
    Compute AUROC and other metrics

    Metrics:
    - AUROC: Primary metric
    - Sensitivity: True positive rate
    - Specificity: True negative rate
    - Accuracy: Overall correctness
    """
```

### 4. Utilities (`util.py`)

**Utility Functions**:
- Data loading and caching
- Preprocessing pipelines
- Metric computation
- Logging helpers
- Device management

---

## Data Flow

### Training Round Flow

```
Round t:
┌──────────────────────────────────┐
│ 1. Server broadcasts model w_t   │
└────────────┬─────────────────────┘
             │
      ┌──────┴──────┐
      │             │
┌─────▼──┐   ┌─────▼──┐   ┌─────▼──┐
│Hospital │   │Hospital │   │Hospital │
│    A    │   │    B    │   │    C    │
├─────────┤   ├─────────┤   ├─────────┤
│ Train   │   │ Train   │   │ Train   │
│w_t → w'│   │w_t → w''│   │w_t → w'''
│ A      │   │ B       │   │ C       │
└─────┬───┘   └─────┬───┘   └─────┬───┘
      │             │             │
      └──────┬──────┴─────────────┘
             │
      ┌──────▼──────────────┐
      │ 2. Send updates     │
      │ (weights only)      │
      └──────┬──────────────┘
             │
      ┌──────▼──────────────┐
      │ 3. Aggregate        │
      │ w_{t+1} = Avg(w'_A, │
      │           w'_B, w'_C)
      └──────┬──────────────┘
             │
      ┌──────▼──────────────┐
      │ 4. Evaluate global  │
      │ model on all tests  │
      └──────┬──────────────┘
             │
      ┌──────▼──────────────┐
      │ 5. Log metrics & CP │
      │ (W&B, disk)         │
      └─────────────────────┘
```

---

## Communication Protocol

### Message Format

**Client → Server** (Model Updates):
```
Frame {
  hospital_id: str
  round: int
  weights: dict[str, tensor]
  num_examples: int
  metrics: dict
}
```

**Server → Client** (Model Weights):
```
Frame {
  round: int
  weights: dict[str, tensor]
  config: dict
}
```

### Privacy Considerations

- ✅ **Patient data stays local**: Never transmitted
- ✅ **Only model updates shared**: Weights and gradients only
- ✅ **Aggregation on server**: No hospital sees others' gradients
- ⚠️ **Current**: No additional encryption (future improvement)
- 🔒 **Recommended**: Add secure aggregation or differential privacy

---

## Federated Averaging (FedAvg)

### Algorithm

```
Input: number of rounds T, clients K
Output: global model w

Initialize: w_0 randomly

For t = 0 to T-1:

  // Server selects clients
  S_t ← random sample of K hospitals

  // Clients train locally
  For each hospital i in S_t (in parallel):
    w_t^i ← ClientUpdate(i, w_t)

  // Server aggregates
  w_{t+1} ← (1/K) * Σ_{i ∈ S_t} w_t^i

  // Server evaluates
  metrics_t ← Evaluate(w_{t+1})
```

### FedProx Extension

We use **FedProx** which adds a regularization term to prevent hospitals from drifting too far from the global model:

```
Hospital local loss:
L_i(w) = original_loss(w) + (μ/2) * ||w - w_t||²

Effect: Keeps hospitals' updates closer to global model
Benefit: Better convergence with heterogeneous data
```

---

## File Structure

```
cold_start_hackathon/
├── __init__.py                # Package initialization
├── server_app.py              # Flower ServerApp
│   ├── HackathonFedProx       # Custom FedProx strategy
│   ├── main()                 # Server initialization
│   └── Checkpointing          # Model persistence
├── client_app.py              # Flower ClientApp
│   ├── Client class           # Implements ClientApp
│   ├── fit()                  # Local training
│   └── evaluate()             # Local evaluation
├── task.py                    # ML components
│   ├── Net                    # Model architecture
│   ├── train()                # Training loop
│   ├── evaluate()             # Evaluation metrics
│   └── load_data()            # Data loading
├── util.py                    # Utilities
│   ├── Preprocessing          # Image normalization
│   ├── Metrics                # Evaluation metrics
│   └── Logging                # W&B integration
└── losses.py                  # Custom loss functions
    ├── FocalLoss              # Handles class imbalance
    └── AdaptiveFocalLoss      # Dynamic weighting
```

---

## Distributed Execution

### Local Simulation
```bash
flwr run . local
# Spawns 3 virtual clients locally
# Ideal for testing on single machine
```

### Cluster Deployment
```bash
flwr run . cluster
# Hospitals run as separate SLURM jobs
# Each hospital: 1 GPU, 2 vCPUs, 32GB RAM
# Up to 4 parallel hospitals
```

### Compute Resources

| Component | Resource | Details |
|-----------|----------|---------|
| **Server** | 1 vCPU, 4GB RAM | Central aggregation |
| **Client** | 2 vCPUs, 32GB RAM | Training + data loading |
| **GPU** | 1× NVIDIA GPU | Shared among clients |
| **Storage** | 5GB | Model + data |

---

## Performance Characteristics

### Timing per Round

```
Round breakdown (20-min constraint):
├─ Data loading:        5-10 seconds
├─ 3 hospitals train:    2-3 minutes each (parallel)
├─ Model aggregation:    10-20 seconds
├─ Evaluation:           20-30 seconds
├─ Checkpointing:        10-15 seconds
└─ Total:                ~3-4 minutes per round

⟹ Fits 9 rounds in 20 minutes
```

### Scalability

| Aspect | Current | Maximum |
|--------|---------|---------|
| **Hospitals** | 3 | Limited by cluster |
| **Parallel Clients** | 4 | 4 (cluster limit) |
| **Images/Hospital** | ~20-40K | Limited by disk |
| **Model Size** | ~40-50MB | Depends on architecture |

---

## Error Handling & Resilience

### Client Failures
```python
# Server tolerates missing hospitals
min_available_clients = 2  # Can proceed with 2 of 3

# Recovers from:
- Network timeouts
- Training failures
- Evaluation errors
```

### Checkpointing
```python
# Automatic saves every round
checkpoint = {
    'model_weights': state_dict,
    'round': round_number,
    'metrics': evaluation_results,
    'timestamp': current_time
}
```

### Disaster Recovery
```bash
# Resume from checkpoint
./submit-job.sh "flwr run . cluster --checkpoint latest"
```

---

## Monitoring & Logging

### Metrics Tracked

**Training Metrics**:
- Loss per round
- Learning rate
- Gradient norms

**Evaluation Metrics**:
- AUROC (all hospitals)
- Sensitivity / Specificity
- Per-hospital breakdown

**System Metrics**:
- Round duration
- Communication overhead
- GPU/CPU utilization

### Logging Backends

```
Logs → Weights & Biases (cloud)
     → Local files (./logs/)
     → Console output (terminal)
```

---

## Future Architecture Improvements

- [ ] Asynchronous aggregation (FedAsync)
- [ ] Differential privacy integration
- [ ] Secure multi-party computation
- [ ] Knowledge distillation for compression
- [ ] Personalized federated learning
- [ ] Horizontal + vertical partitioning

---

## References

- [Flower Framework Docs](https://flower.ai/)
- [FedAvg Paper](https://arxiv.org/pdf/1602.05629.pdf)
- [FedProx Paper](https://arxiv.org/pdf/1812.06127.pdf)
- [Focal Loss Paper](https://arxiv.org/pdf/1708.02002.pdf)

---

[← Back Home](index.md) | [Getting Started](./getting-started.md) | [Training Guide →](./training-guide.md)
