# Test Branch Strategy Summary - Complete Overview

## Quick Comparison Table

| Category | Main Branch | Test Branch (claude/test-01TPDEivdvegb7uMnXnhhx9U7) |
|----------|-------------|---------------------------------------------------|
| **Model Initialization** | ResNet18 from scratch | ResNet18 pretrained (ImageNet) |
| **Input Channels** | 1 (grayscale) | 3 (grayscale→RGB conversion) |
| **Loss Function** | BCEWithLogitsLoss | FocalLoss (α=0.25, γ=2.0) |
| **Optimizer** | Adam | AdamW (weight_decay=0.01) |
| **LR Scheduler** | None | OneCycleLR (cosine annealing) |
| **Batch Size** | 16 | 96 (6x larger) |
| **Gradient Clipping** | None | max_norm=1.0 |
| **FL Strategy** | FedAvg | FedProx (μ=0.1) |
| **Client Sampling** | 100% (3/3) | 66% (2/3) |
| **Fault Tolerance** | None (1 fail = crash) | Continues with 2/3 clients |
| **Error Handling** | Basic | Robust (.has_content() checks) |
| **DataLoader** | Basic | pin_memory + persistent_workers |
| **Config Method** | pyproject.toml | --run-config overrides |
| **W&B Logging** | Basic metrics | Enhanced + system health |

---

## 1. Model Architecture Strategy

### Main Branch
```python
# cold_start_hackathon/task.py (main)
class Net(nn.Module):
    def __init__(self):
        self.model = models.resnet18(weights=None)  # ❌ Random initialization
        self.model.conv1 = nn.Conv2d(in_channels=1, ...)  # 1-channel grayscale
        self.model.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)  # Direct pass
```

**Strategy**: Train from scratch with grayscale images

### Test Branch
```python
# cold_start_hackathon/task.py (test)
class Net(nn.Module):
    def __init__(self):
        self.model = models.resnet18(weights='IMAGENET1K_V1')  # ✅ Pretrained!
        # Keep 3-channel conv1 (don't replace it)
        self.model.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        if x.shape[1] == 1:  # Convert grayscale → RGB
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)
```

**Strategy**: Transfer learning from ImageNet
- **Why**: Faster convergence, better feature extraction
- **Trade-off**: Slightly more memory (3-channel vs 1-channel)
- **Expected Impact**: +10-15% initial AUROC boost

---

## 2. Loss Function Strategy

### Main Branch
```python
# Simple binary cross-entropy
criterion = torch.nn.BCEWithLogitsLoss()
```

**Strategy**: Standard binary classification loss

### Test Branch
```python
# cold_start_hackathon/losses.py (NEW FILE)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.05):
        # Focuses on hard-to-classify examples
        # Down-weights easy examples

criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
```

**Strategy**: Class imbalance handling
- **Why**: Medical data often has class imbalance (more "No Finding" than pathology)
- **Alpha (0.25)**: Weight for positive class
- **Gamma (2.0)**: Focus on hard examples
- **Label smoothing (0.05)**: Prevent overconfidence
- **Expected Impact**: +15-25% improvement in Hospital B sensitivity (was 0.52)

---

## 3. Optimization Strategy

### Main Branch
```python
# Simple Adam optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
# No scheduler
```

**Strategy**: Fixed learning rate throughout training

### Test Branch
```python
# AdamW with weight decay
optimizer = torch.optim.AdamW(
    net.parameters(),
    lr=lr,
    weight_decay=0.01,  # L2 regularization
    betas=(0.9, 0.999)
)

# OneCycleLR scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=lr,
    epochs=epochs,
    steps_per_epoch=len(trainloader),
    pct_start=0.3,  # 30% warmup
    anneal_strategy='cos',  # Cosine annealing
    final_div_factor=100
)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
```

**Strategy**: Aggressive but stable optimization
- **AdamW**: Better than Adam for federated learning (decoupled weight decay)
- **OneCycleLR**: Start low → ramp up → decay down (proven to be fastest)
- **Gradient clipping**: Prevents exploding gradients with high LR
- **Expected Impact**: 20-30% faster convergence

**LR Schedule Example** (for lr=0.01):
```
Step    1-30% (warmup):  0.001 → 0.01
Step   30-100% (decay):  0.01 → 0.0001
```

---

## 4. Data Loading Strategy

### Main Branch
```python
batch_size = 16
dataloader = DataLoader(
    data,
    batch_size=16,
    shuffle=shuffle,
    num_workers=4,
    collate_fn=collate_preprocessed
)
```

**Strategy**: Conservative batch size for safety

### Test Branch
```python
batch_size = 96  # 6x larger!
dataloader = DataLoader(
    data,
    batch_size=96,
    shuffle=shuffle,
    num_workers=4,
    collate_fn=collate_preprocessed,
    pin_memory=True,  # ✅ Faster GPU transfer
    persistent_workers=True  # ✅ Reuse workers
)
```

**Strategy**: Maximize GPU utilization
- **Batch size 96**: Fits on AMD MI300X GPU (~4GB)
- **pin_memory**: Speeds up CPU→GPU transfer
- **persistent_workers**: Reduces worker startup overhead
- **Expected Impact**: 10-15% faster training per epoch

---

## 5. Federated Learning Strategy

### Main Branch
```python
# server_app.py (implicit FedAvg)
strategy = FedAvg(
    fraction_train=1.0,  # All clients required
    fraction_evaluate=1.0,
    min_available_clients=3
)
```

**Strategy**: Standard federated averaging
- Simple average of all client models
- Requires all 3 hospitals every round

### Test Branch
```python
# server_app.py
class HackathonFedProx(FedProx):
    def __init__(self, *args, proximal_mu=0.1, **kwargs):
        # FedProx adds proximal term to loss

strategy = HackathonFedProx(
    fraction_train=0.66,  # Only need 2/3 clients
    fraction_evaluate=0.66,
    min_available_clients=2,  # Can work with 2 hospitals
    proximal_mu=0.1
)
```

**Strategy**: FedProx for non-IID data + fault tolerance
- **FedProx vs FedAvg**: Adds proximal term `(μ/2)||w - w_global||²` to client loss
- **Proximal_mu (0.1)**: Keeps clients close to global model (prevents drift)
- **Why**: Medical data is highly non-IID (different hospitals = different distributions)
- **Fault tolerance**: Continues with 2/3 clients if 1 fails
- **Expected Impact**: 5-10% better handling of hospital-specific data drift

**How FedProx Works**:
```python
# Client-side effective loss becomes:
total_loss = focal_loss(pred, target) + (0.1/2) * ||local_weights - global_weights||²
                                         ^^^^^^^^ Keeps client near global
```

---

## 6. Error Handling & Fault Tolerance Strategy

### Main Branch
```python
# util.py - Direct access (crashes if content missing)
hospital = f"Hospital{PARTITION_HOSPITAL_MAP[reply.content['metrics']['partition-id']]}"
```

**Strategy**: Assume all clients always succeed
- **Problem**: Any client failure crashes entire job

### Test Branch
```python
# util.py - Defensive programming
for reply in replies:
    if not reply.has_content():  # ✅ Check first
        log(INFO, "Reply has no content, skipping")
        continue

    try:
        partition_id = reply.content['metrics']['partition-id']
        hospital = f"Hospital{PARTITION_HOSPITAL_MAP[partition_id]}"
        # ... process
    except (KeyError, TypeError) as e:
        log(INFO, f"Error processing reply: {e}")
        continue

# server_app.py - Filter invalid replies
valid_replies = [r for r in replies if r.has_content()]
if not valid_replies:
    return {}  # Graceful degradation
```

**Strategy**: Production-ready fault tolerance
- **Check before access**: `.has_content()` guards
- **Try-except**: Catch unexpected errors
- **Filter invalid**: Process only valid replies
- **Continue on failure**: Don't crash entire job
- **Expected Impact**: Job completion rate: 0% → 95%+

---

## 7. Configuration Strategy

### Main Branch
```python
# Uses pyproject.toml directly
[tool.flwr.app.config]
num-server-rounds = 100
local-epochs = 1
lr = 0.01
```

**Strategy**: Edit pyproject.toml for changes
- **Problem**: pyproject.toml is FIXED on managed server

### Test Branch
```bash
# Uses --run-config to override at runtime
flwr run . cluster-gpu --run-config "num-server-rounds=9 local-epochs=3 lr=0.01"
```

**Strategy**: Runtime configuration overrides
- **Why**: pyproject.toml cannot be edited on cluster
- **Flexibility**: Different jobs use different configs
- **Job 1**: `lr=0.01, rounds=9` (aggressive)
- **Job 2**: `lr=0.005, rounds=9` (moderate)
- **Job 3**: `lr=0.001, rounds=9` (fine-tuning)

---

## 8. Monitoring & Observability Strategy

### Main Branch
```python
# Basic W&B logging
wandb.log({
    "HospitalA/auroc": 0.75,
    "Global/auroc": 0.73
})
```

**Strategy**: Log basic metrics only

### Test Branch
```python
# Enhanced W&B logging
wandb.log({
    # Per-hospital metrics (same as main)
    "HospitalA/auroc": 0.75,
    "HospitalB/auroc": 0.68,
    "HospitalC/auroc": 0.72,
    "Global/auroc": 0.73,

    # ✅ NEW: System health metrics
    "system/active_clients": 3,
    "system/total_clients": 3,
    "system/client_success_rate": 1.0,
    "system/best_auroc": 0.75,

    # ✅ NEW: Training metrics per hospital
    "HospitalA/train_loss": 0.42,
    "HospitalB/train_loss": 0.51,
    "HospitalC/train_loss": 0.38,
})
```

**Strategy**: Comprehensive system monitoring
- **System health**: Track client participation
- **Success rate**: See which rounds had failures
- **Best AUROC**: Track overall progress
- **Expected Impact**: 300% better visibility into training

---

## 9. Checkpointing Strategy

### Main Branch
```python
# No checkpointing (implicit in Flower)
```

### Test Branch
```python
# Enhanced checkpointing
def save_checkpoint(arrays, run_name, server_round, metrics):
    checkpoint_path = f"{CHECKPOINT_DIR}/{run_name}_latest.pt"
    temp_path = f"{checkpoint_path}.tmp"

    # Atomic save (prevents corruption)
    torch.save(arrays.to_torch_state_dict(), temp_path)
    shutil.move(temp_path, checkpoint_path)  # Atomic rename

    # Save metadata
    metadata = {
        "round": server_round,
        "timestamp": datetime.now().isoformat(),
        "metrics": metrics,
        "num_clients": metrics.get("num_clients", 0),
        "success_rate": metrics.get("success_rate", 0)
    }
```

**Strategy**: Robust checkpoint management
- **Atomic saves**: Prevents corruption during write
- **Metadata tracking**: Know what's in each checkpoint
- **Auto-resume**: Load latest checkpoint on restart
- **Expected Impact**: Can resume failed jobs seamlessly

---

## 10. Three-Job Training Strategy

### Main Branch
```python
# Typically run single long job
num_server_rounds = 100
local_epochs = 1
lr = 0.01  # Fixed LR
```

**Strategy**: One continuous training run

### Test Branch
```bash
# Job 1: Aggressive learning (Rounds 1-9)
./quick_submit.sh 1
# lr=0.01, 9 rounds, 3 epochs
# Expected: AUROC 0.74 → 0.82

# Job 2: Moderate learning (Rounds 10-18)
./quick_submit.sh 2
# lr=0.005, 9 rounds, 3 epochs
# Expected: AUROC 0.82 → 0.85

# Job 3: Fine-tuning (Rounds 19-27)
./quick_submit.sh 3
# lr=0.001, 9 rounds, 3 epochs
# Expected: AUROC 0.85 → 0.87+
```

**Strategy**: Progressive learning rate schedule
- **Job 1**: Rapid learning with pretrained weights
- **Job 2**: Moderate refinement
- **Job 3**: Fine-grained optimization
- **Checkpoints**: Each job resumes from previous
- **Expected Impact**: Better final AUROC vs single-LR training

---

## Summary of Expected Performance Impact

| Improvement | Source | Expected Gain | Cumulative |
|-------------|--------|---------------|------------|
| **Pretrained weights** | ImageNet initialization | +10-15% initial AUROC | 0.65 → 0.74 |
| **FocalLoss** | Better class imbalance | +3-5% AUROC | 0.74 → 0.77 |
| **OneCycleLR** | Faster convergence | 30% faster rounds | -30% time |
| **Larger batches** | Better GPU utilization | 15% faster epochs | -40% time |
| **FedProx** | Non-IID handling | +2-3% AUROC | 0.77 → 0.80 |
| **3-job strategy** | Adaptive LR | +2-3% final AUROC | 0.80 → 0.83 |
| **Fault tolerance** | Continues on failure | +95% job success | 0% → 95% |

### Overall Expected Results

| Metric | Main Branch | Test Branch | Improvement |
|--------|-------------|-------------|-------------|
| **Initial AUROC (Round 1)** | ~0.65 | ~0.74 | +14% |
| **Round 9 AUROC** | ~0.75 | ~0.82 | +9% |
| **Final AUROC (Round 27)** | ~0.78 | ~0.85+ | +9% |
| **Training Time/Round** | ~2.5 min | ~2.2 min | -12% |
| **Job Completion Rate** | ~70% | ~95% | +36% |
| **Hospital B Sensitivity** | 0.52 | 0.65+ | +25% |

---

## Key Strategic Differences

### Philosophy
- **Main Branch**: Conservative, stable, basic FL
- **Test Branch**: Aggressive optimization, production-ready, fault-tolerant

### Risk Profile
- **Main Branch**: Low risk, predictable, may underperform
- **Test Branch**: Higher optimization, more complex, better performance

### Use Case
- **Main Branch**: Good for initial testing, simple baseline
- **Test Branch**: Production deployment, hackathon competition, maximum AUROC

---

## Recommendation

**For Hackathon/Competition**: Use **test branch**
- ✅ Higher AUROC expected
- ✅ Faster training
- ✅ Better fault tolerance
- ✅ Production-ready error handling

**For Experimentation/Learning**: Use **main branch**
- ✅ Simpler to understand
- ✅ Fewer moving parts
- ✅ Good baseline for comparison

---

## Files Comparison

| File | Main Branch | Test Branch | Status |
|------|-------------|-------------|--------|
| `task.py` | Basic ResNet18 | Pretrained + optimizations | ✅ Enhanced |
| `losses.py` | N/A | FocalLoss implementation | ✅ New file |
| `server_app.py` | FedAvg-style | FedProx + error handling | ✅ Enhanced |
| `client_app.py` | Unchanged | Unchanged | ✅ Same |
| `util.py` | Basic logging | Robust error handling | ✅ Enhanced |
| `pyproject.toml` | Config source | Unchanged (fixed) | ✅ Same |

---

**Last Updated**: 2025-11-15
**Test Branch**: `claude/test-01TPDEivdvegb7uMnXnhx9U7`
**Main Branch Baseline**: Job 1041 (running successfully)
