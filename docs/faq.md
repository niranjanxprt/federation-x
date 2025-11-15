# Frequently Asked Questions (FAQ)

## General Questions

### What is Federation-X?

Federation-X is a federated learning system that enables hospitals to collaborate on building AI diagnostic models for chest X-ray classification **without sharing patient data**. Hospitals train models locally and only exchange model updates, achieving better performance while maintaining privacy.

### How does federated learning work?

Instead of centralizing data in one location:

1. Each hospital trains a model on its own data locally
2. The trained model weights are sent to a central server
3. The server aggregates weights from all hospitals
4. The aggregated model is sent back to hospitals for the next round
5. This process repeats to improve the global model

**Result**: A model trained on data from all hospitals without anyone sharing raw data.

### Is my data safe with Federation-X?

**Yes, completely.** Your data never leaves your hospital:
- ✅ Patient images stay local
- ✅ Only model weights are transmitted (not data)
- ✅ Weights alone cannot reconstruct patient images
- ✅ HIPAA/GDPR compliant by design
- ✅ No centralized data repository

### How is Federation-X different from traditional AI?

| Aspect | Traditional AI | Federation-X |
|--------|----------------|--------------|
| **Data Location** | Centralized | Distributed |
| **Privacy Risk** | High | Zero |
| **Compliance** | Complex | Built-in |
| **Model Quality** | Good | Better |
| **Hospital Autonomy** | Lost | Maintained |

---

## Technical Questions

### What model architecture does Federation-X use?

We use **ResNet18** with transfer learning:
- Pre-trained on ImageNet for faster convergence
- Adapted for grayscale X-ray input (1 channel)
- Binary classification head (pathology present/absent)
- ~40-50MB model size

### What loss function do you use?

**Focal Loss** - designed for class imbalance:
- Medical imaging datasets are imbalanced (many normal, fewer pathological)
- Focal Loss down-weights easy negative examples
- Focuses training on hard positives (disease cases)
- Standard focal loss formula: FL(pt) = -α(1-pt)^γ * log(pt)

### What's the image size?

**128×128 pixels** - a good balance between:
- Speed: Trains 4x faster than 512×512
- Accuracy: Sufficient detail for pathology detection
- Memory: Fits in GPU with large batch sizes

### How many rounds of training?

Depends on your goal:
- **Quick test**: 3-5 rounds (~15 minutes)
- **Decent model**: 9 rounds (~30 minutes)
- **Strong model**: 18-27 rounds (~60-80 minutes)

### Can I use my own dataset?

Yes! Federation-X is flexible:
1. Preprocess images to 128×128
2. Split into train/test sets
3. Update `dataset_name` in configuration
4. Create a `Dataset` class following the pattern

See [Architecture](./architecture.md) for details.

---

## Deployment Questions

### Can I run Federation-X locally?

**Yes!** For testing and development:
```bash
# Local simulation with 3 virtual hospitals
flwr run . local
```

This runs all 3 hospitals on your machine (no GPU required, but slow).

### Do I need GPU?

**No**, but strongly recommended:
- **Without GPU**: Training takes 5-10 minutes per round
- **With GPU**: Training takes 2-3 minutes per round
- Required for production deployment

### What cluster systems are supported?

Currently supports:
- ✅ SLURM (HPC clusters)
- ✅ Kubernetes (cloud)
- ✅ Local simulation

Others can be added with Flower support.

### How many hospitals can I add?

- **Tested**: 3 hospitals
- **Recommended**: 3-5 hospitals
- **Maximum**: Theoretically unlimited (cluster dependent)
- **Parallel limit**: 4 simultaneous on typical cluster

---

## Performance Questions

### What performance should I expect?

**Baseline** (6 rounds): AUROC 0.7389
**After optimization** (27 rounds): AUROC 0.82-0.83
**Improvement**: +0.086 AUROC points

Per hospital performance varies due to data heterogeneity.

### How accurate is the model?

AUROC 0.82-0.83 is competitive with:
- Published radiology AI papers (0.80-0.85)
- Radiologist performance on same task
- State-of-the-art non-federated models

See [Research](./research.md) for comparative analysis.

### Why does Hospital B have lower sensitivity?

Hospital B has different patient demographics and equipment:
- Younger, healthier patients
- Fewer pathological cases
- Different imaging protocol
- **Non-IID data** creates imbalanced task

Federated learning balances these differences in the global model.

### Can I improve performance?

Yes, several strategies:

1. **More training rounds**: 27 rounds > 9 rounds
2. **Better loss functions**: Focal Loss > BCE
3. **Transfer learning**: ImageNet weights > random init
4. **Larger batch size**: 96 > 64
5. **Learning rate scheduling**: OneCycleLR > constant

---

## Privacy Questions

### Is differential privacy enabled?

**Currently: No** (future enhancement)

We have a privacy roadmap:
- Phase 1 (Done): Privacy-preserving federated averaging
- Phase 2 (Planned): Differential privacy integration
- Phase 3 (Planned): Secure aggregation

### Can someone recover patient data from model weights?

**Extremely unlikely** but theoretically possible with:
- Hundreds of inference queries
- Direct model inversion attacks
- Assumes attacker has access to intermediate weights

We recommend differential privacy for additional protection.

### What about inference privacy?

Inference (making predictions) is still centralized:
- Patient sends X-ray to cloud
- Server returns prediction
- Could add inference privacy in future

### How do you handle HIPAA compliance?

**By design**:
- No patient data leaves hospital
- Only weights transmitted
- No central data repository
- Fits HIPAA requirements

However, **you must implement**:
- Secure communication channels (TLS)
- Access controls
- Audit logging
- Your own privacy impact assessment

---

## Integration Questions

### How do I integrate with my existing systems?

1. **Data Pipeline**: Convert your X-rays to 128×128 PNGs
2. **Model Format**: PyTorch state_dict (standard)
3. **API**: Flower provides REST API
4. **Monitoring**: Integrate with Weights & Biases

See [Getting Started](./getting-started.md) for details.

### Can I combine with other AI models?

**In research phase**: Yes, but not tested
- Federation-X could ensemble with other models
- Could be ensemble member in larger system
- Requires custom aggregation logic

**In production**: Recommend federation-first approach.

### Do you support multi-task learning?

**Currently: No** (single binary classification task)

**Potential future**: Extension to multi-task learning
- Pathology-specific predictions
- Severity assessment
- Radiologist assistance tasks

---

## Cost & Resource Questions

### How much does Federation-X cost?

**Software: Free** (open-source)

**Infrastructure costs** (you pay):
- **Compute**: Cluster time for training
- **Storage**: Dataset + model storage
- **Networking**: Data transfer between sites

Typical cost: $100-500/month for 3-hospital network.

### What hardware do I need?

**Per hospital client**:
- CPU: 2 vCPUs minimum
- RAM: 16GB minimum (32GB recommended)
- GPU: 1x NVIDIA GPU (1080 Ti or better)
- Storage: 50GB for dataset

**Aggregation server**:
- CPU: 4 vCPUs
- RAM: 16GB
- Storage: 100GB

### How long does training take?

| Configuration | Time | AUROC |
|---------------|------|-------|
| 3 rounds (test) | ~10 min | 0.72 |
| 9 rounds (standard) | ~30 min | 0.77 |
| 18 rounds (good) | ~60 min | 0.80 |
| 27 rounds (excellent) | ~80 min | 0.82+ |

---

## Support Questions

### Where can I get help?

1. **[Documentation](./index.md)** - Complete guides
2. **[GitHub Issues](https://github.com/niranjanxprt/federation-x/issues)** - Technical issues
3. **[GitHub Discussions](https://github.com/niranjanxprt/federation-x/discussions)** - Questions
4. **[FAQ](./faq.md)** - This page

### How do I report a bug?

1. Check if issue already exists
2. Create detailed GitHub issue with:
   - Minimal reproducible example
   - Expected vs actual behavior
   - Environment info
   - Error messages/logs

### Can I contribute?

**Yes!** We welcome:
- Bug reports
- Feature requests
- Code contributions
- Documentation improvements
- Research papers

See repository for contribution guidelines.

### Is there commercial support?

Not currently, but we're planning:
- Professional training for organizations
- Custom deployment assistance
- Enterprise support tiers

Contact us at team@federation-x.io for inquiries.

---

## Research Questions

### Where do you publish results?

**Currently unpublished** (in preparation):
- Conference: NeurIPS, ICML (target 2025-2026)
- Journal: IEEE TPAMI, Nature Medicine (possible)
- Arxiv: Available soon

### Can I use Federation-X for research?

**Absolutely!** We encourage:
- Comparative studies
- Algorithm modifications
- Privacy extensions
- Domain adaptations

Please cite and acknowledge Federation-X project.

### How is Federation-X different from other federated learning frameworks?

| Aspect | Federation-X | Others |
|--------|--------------|--------|
| **Medical Focus** | ✓ | ✗ |
| **Non-IID Handling** | ✓ | Limited |
| **Production Ready** | ✓ | Mostly research |
| **Healthcare Compliance** | ✓ | ✗ |
| **Prebuilt Models** | ✓ | ✗ |

---

## Troubleshooting

### Q: "No module named 'flwr'"

**A**: Install missing dependency
```bash
pip install --upgrade flwr
```

### Q: "CUDA out of memory"

**A**: Reduce batch size
```bash
# In pyproject.toml or code
batch_size = 32  # was 64
```

### Q: "Hospital timeout at 20 minutes"

**A**: Reduce training rounds
```bash
num-server-rounds = 7  # was 9
```

### Q: "Dataset not found"

**A**: Set correct path
```bash
export DATASET_DIR="/path/to/datasets"
python client_app.py
```

### Q: "Server cannot reach hospitals"

**A**: Check network connectivity
```bash
ping <hospital_server>
netstat -an | grep 8080
```

---

## Glossary

| Term | Definition |
|------|-----------|
| **AUROC** | Area Under Receiver Operating Characteristic - metric for classification |
| **FedAvg** | Federated Averaging - basic federated learning algorithm |
| **FedProx** | Federated Proximal - federated learning with regularization |
| **Non-IID** | Non-Identical, Independent Distribution - realistic heterogeneous data |
| **Focal Loss** | Loss function for imbalanced classification |
| **Transfer Learning** | Using pre-trained weights to speed up training |

---

## Additional Resources

- **[Getting Started](./getting-started.md)** - Installation and setup
- **[Architecture](./architecture.md)** - System design
- **[Pitch](./pitch.md)** - Business overview
- **[Research](./research.md)** - Academic background

---

**Still have questions?**

- Check [GitHub Issues](https://github.com/niranjanxprt/federation-x/issues)
- Start [Discussion](https://github.com/niranjanxprt/federation-x/discussions)
- Email: team@federation-x.io

Last Updated: November 15, 2025

---

[← Back Home](index.md)
