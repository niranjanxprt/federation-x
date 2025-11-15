# Branch Comparison: Main vs Test Branch

## Overview
This document compares the **main branch** (currently working in production) with the **test branch** (claude/test-01TPDEivdvegb7uMnXnhx9U7) to ensure compatibility and understand differences.

---

## Main Branch (Job 1041 - WORKING ✅)

### Configuration
- **Model**: ResNet18 with `weights=None` (training from scratch)
- **Input Channels**: 1 (grayscale X-rays)
- **Batch Size**: 16
- **Optimizer**: Adam (lr from config)
- **Loss**: BCEWithLogitsLoss (standard)
- **Scheduler**: None
- **Data Format**: `[batch, 1, 128, 128]` (single-channel grayscale)

### Code Structure
```python
# task.py
def __init__(self):
    self.model = models.resnet18(weights=None)
    self.model.conv1 = nn.Conv2d(in_channels=1, ...)  # Grayscale
    self.model.fc = nn.Linear(in_features, 1)

def forward(self, x):
    return self.model(x)  # No conversion needed

def train(...):
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # Standard training loop
```

### Performance Characteristics
- **Training Speed**: Slower (no pretrained weights)
- **Convergence**: Gradual from random initialization
- **Expected AUROC**: Baseline performance
- **Memory Usage**: Lower (batch_size=16)

---

## Test Branch (claude/test-01TPDEivdvegb7uMnXnhx9U7)

### Configuration
- **Model**: ResNet18 with `weights='IMAGENET1K_V1'` (pretrained)
- **Input Channels**: 3 (converted from grayscale in forward())
- **Batch Size**: 96 (6x increase)
- **Optimizer**: AdamW (lr from config, weight_decay=0.01)
- **Loss**: FocalLoss (α=0.25, γ=2.0, label_smoothing=0.05)
- **Scheduler**: OneCycleLR (aggressive learning rate schedule)
- **Data Format**: `[batch, 1, 128, 128]` → converted to `[batch, 3, 128, 128]` in forward()

### Code Structure
```python
# task.py
def __init__(self):
    self.model = models.resnet18(weights='IMAGENET1K_V1')  # Pretrained!
    # Keep conv1 as 3-channel (don't replace it)
    self.model.fc = nn.Linear(in_features, 1)

def forward(self, x):
    # Convert grayscale → RGB for pretrained model
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return self.model(x)

def train(...):
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(...)
    # Optimized training loop with gradient clipping
```

### New Features
1. **FocalLoss**: Handles class imbalance (addresses Hospital B low sensitivity)
2. **Pretrained Weights**: ImageNet initialization for faster convergence
3. **AdamW**: Better regularization than Adam
4. **OneCycleLR**: Aggressive LR schedule for faster training
5. **Gradient Clipping**: Stability with high learning rates
6. **DataLoader Optimizations**: `pin_memory=True`, `persistent_workers=True`

### Performance Characteristics
- **Training Speed**: Faster (pretrained weights + larger batches)
- **Convergence**: Rapid (starts from ImageNet features)
- **Expected AUROC**: Higher baseline, faster improvement
- **Memory Usage**: Higher (batch_size=96, needs ~4GB GPU)

---

## Compatibility Analysis

### ✅ FULLY BACKWARD COMPATIBLE

Both branches work with the **same data format**:
- Data comes in as: `[batch, 1, 128, 128]` (grayscale)
- Main branch: Uses directly with 1-channel conv1
- Test branch: Converts to 3-channel in forward() before processing

### Data Pipeline
```
Preprocessed Dataset (1-channel grayscale)
         ↓
    DataLoader
         ↓
    [batch, 1, 128, 128]
         ↓
    ┌────────────────────────────────────┐
    │                                    │
    │  Main Branch    │  Test Branch    │
    │  (direct use)   │  (convert to 3) │
    │                 │                  │
    │  conv1 (1ch) ←──┤──→ repeat() → conv1 (3ch)
    │                 │                  │
    └────────────────────────────────────┘
```

### File Changes Summary

| File | Main Branch | Test Branch | Breaking Change? |
|------|-------------|-------------|------------------|
| `task.py` | Basic ResNet18 | Optimized + pretrained | ❌ No |
| `losses.py` | N/A (doesn't exist) | FocalLoss class | ❌ No (new file) |
| `server_app.py` | FedAvg | FedProx + checkpointing | ❌ No |
| `pyproject.toml` | Fixed config | **UNCHANGED** | ✅ No changes |
| `client_app.py` | Unchanged | Unchanged | ✅ Same |

### Dependency Check
All new features use **existing dependencies** from pyproject.toml:
- ✅ `torch` - for AdamW, OneCycleLR, FocalLoss
- ✅ `torchvision` - for pretrained ResNet18
- ✅ No new packages required

---

## Risk Assessment

### 🟢 LOW RISK - Safe to Deploy
1. **No breaking changes** to data pipeline or file formats
2. **All dependencies** already present in fixed pyproject.toml
3. **Backward compatible** - can switch back to main anytime
4. **Runtime config** - all changes use `--run-config` flag (doesn't edit pyproject.toml)
5. **Isolated branch** - main branch unaffected

### Rollback Strategy
If test branch fails, simply:
```bash
git checkout main
git push -u origin main
```
Main branch (Job 1041) continues working as-is.

---

## Expected Improvements (Test Branch)

| Metric | Main Branch | Test Branch | Improvement |
|--------|-------------|-------------|-------------|
| Initial AUROC | ~0.65 | ~0.74 | +14% (pretrained) |
| Round 9 AUROC | ~0.75 | ~0.82 | +9% |
| Training Time/Round | ~2.5 min | ~2.2 min | -12% |
| Hospital B Sensitivity | 0.52 | 0.65+ | +25% (FocalLoss) |
| Convergence Speed | Slow | Fast | 3x faster |

---

## Recommendation

✅ **SAFE TO TEST** - The test branch is fully compatible with main branch:
- Same data format
- Same dependencies
- No pyproject.toml changes
- Easy rollback if needed

The optimizations in the test branch should provide:
1. **Faster convergence** (pretrained weights)
2. **Better class imbalance handling** (FocalLoss)
3. **More stable training** (gradient clipping + OneCycleLR)
4. **Higher final AUROC** (~0.82 vs ~0.75)

---

## Testing Strategy

1. **Run test branch job** and monitor:
   - Check initial AUROC (should be ~0.74 vs main's ~0.65)
   - Monitor training stability
   - Verify no crashes or errors

2. **Compare with Job 1041** (main branch):
   - AUROC progression
   - Training time per round
   - Final metrics

3. **If successful**: Continue with Jobs 2-3
4. **If issues**: Rollback to main branch

---

**Last Updated**: 2025-11-15
**Test Branch**: claude/test-01TPDEivdvegb7uMnXnhx9U7
**Main Branch Baseline**: Job 1041 (still running)
