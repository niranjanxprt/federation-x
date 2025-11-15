# Federation-X: Privacy-Preserving Medical AI

[![Flower Framework](https://img.shields.io/badge/Framework-Flower-brightgreen.svg)](https://flower.ai/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/niranjanxprt/federation-x?style=social)](https://github.com/niranjanxprt/federation-x)

## Welcome to Federation-X

**Federation-X** is a cutting-edge federated learning project that demonstrates how hospitals can collaborate to build better diagnostic AI models while keeping patient data private and secure.

### 🎯 Mission

Enable robust, privacy-preserving artificial intelligence for medical imaging across distributed healthcare institutions. We prove that distributed AI can outperform centralized models while maintaining strict data privacy standards.

### 🏥 The Problem

In real healthcare systems:
- **Data Privacy** is critical - patient data cannot leave hospital premises
- **Data Heterogeneity** is severe - different hospitals have different patient populations, imaging equipment, and clinical practices
- **Data Scarcity** limits individual hospitals from training robust models
- **Model Generalization** is poor - models trained in one hospital often fail in another

### 💡 The Solution

**Federated Learning** allows hospitals to:
1. Train models locally on their own patient data
2. Share only encrypted model updates (never raw data)
3. Aggregate updates to build a better global model
4. Achieve better performance while maintaining privacy

### 📊 Our Dataset

We use the **NIH Chest X-Ray Dataset**:
- **112,000+** medical images
- **30,000+** unique patients
- **3 distinct hospital silos** with different characteristics
- **Binary classification**: Detect any pathological finding
- **15 pathology types** including pneumonia, edema, nodules, and more

### 🎓 What You'll Learn

By exploring Federation-X, you'll understand:
- ✅ Federated learning fundamentals and algorithms
- ✅ How to handle non-IID (non-identical) data across clients
- ✅ Privacy-preserving machine learning techniques
- ✅ Medical image analysis and classification
- ✅ Distributed AI optimization and scaling
- ✅ Real-world challenges in deploying federated systems

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Privacy-First** | Patient data never leaves hospital premises |
| **Federated** | Train on 3+ hospitals simultaneously |
| **Non-IID Data** | Realistic hospital heterogeneity |
| **Medical Domain** | Real chest X-ray classification |
| **Production Ready** | Optimized for real cluster deployment |
| **Fully Documented** | Complete guides and code examples |

---

## Quick Links

- **[GitHub Repository](https://github.com/niranjanxprt/federation-x)** - Source code and issues
- **[Getting Started](./getting-started.md)** - Installation and setup
- **[Architecture Guide](./architecture.md)** - System design and components
- **[Training Guide](./training-guide.md)** - How to train models
- **[Research](./research.md)** - Background and academic references
- **[FAQ](./faq.md)** - Common questions and troubleshooting

---

## Project Structure

```
federation-x/
├── docs/                          # GitHub Pages documentation
├── cold_start_hackathon/          # Main implementation
│   ├── server_app.py              # Federated server (aggregator)
│   ├── client_app.py              # Hospital clients (local trainers)
│   ├── task.py                    # Model and training logic
│   └── util.py                    # Utilities
├── README.md                      # Quick start guide
├── pyproject.toml                 # Project configuration
└── FL_GUIDE_20MIN_UPDATED.md      # Optimization strategies
```

---

## Technology Stack

```
Frontend/Docs:        GitHub Pages + Markdown
Federated Learning:   Flower Framework
Deep Learning:        PyTorch
Medical Imaging:      OpenCV, PIL
Data Processing:      NumPy, Pandas
Monitoring:           Weights & Biases
Cluster:              HPC + GPU (NVIDIA)
```

---

## Hospital Simulation

We simulate 3 realistic hospital scenarios:

### 🏥 Hospital A: Portable Inpatient Care
- **Size**: 47,583 images (42,093 test + 5,490 eval)
- **Patient Profile**: Elderly males (60+)
- **Equipment**: AP (anterior-posterior) view dominant
- **Specialty**: Fluid-related conditions (Effusion, Edema)

### 🏥 Hospital B: Outpatient Clinic
- **Size**: 24,613 images (21,753 train + 2,860 eval)
- **Patient Profile**: Younger females (20-65)
- **Equipment**: PA (posterior-anterior) view dominant
- **Specialty**: Lung pathologies (Nodules, Pneumothorax)

### 🏥 Hospital C: Research & Rare Conditions
- **Size**: 23,324 images (20,594 train + 2,730 eval)
- **Patient Profile**: Mixed demographics
- **Equipment**: PA view preferred
- **Specialty**: Rare conditions (Fibrosis, Emphysema, Hernia)

---

## Performance Metrics

Our models are evaluated using:

| Metric | Purpose |
|--------|---------|
| **AUROC** | Primary metric - robustness to class imbalance |
| **Sensitivity** | Recall for pathological findings (minimize false negatives) |
| **Specificity** | Precision for normal cases (minimize false positives) |
| **Per-Hospital Performance** | Ensure generalization across hospitals |

---

## Getting Started

### For Users
1. Read the [Getting Started Guide](./getting-started.md)
2. Review [Architecture](./architecture.md)
3. Run local examples
4. Deploy to cluster

### For Researchers
1. Check [Research Background](./research.md)
2. Review papers and references
3. Analyze datasets
4. Experiment with algorithms

### For Contributors
1. Fork the repository
2. Read [Contributing Guidelines](./contributing.md)
3. Submit pull requests
4. Join discussions

---

## Current Performance

**Model**: Federated Learning with FedProx aggregation
**Status**: Production optimized for 20-minute jobs
**Expected AUROC**: 0.82-0.85 (target range)

---

## Roadmap

- [x] Basic federated learning implementation
- [x] Multi-hospital simulation setup
- [x] Medical imaging preprocessing
- [x] Evaluation pipeline
- [ ] Advanced aggregation strategies
- [ ] Differential privacy implementation
- [ ] Secure aggregation protocols
- [ ] Hardware heterogeneity handling
- [ ] Model compression techniques

---

## Resources

- **[Flower AI Documentation](https://flower.ai/)** - Official Flower framework docs
- **[Federated Learning Paper](https://arxiv.org/pdf/1602.05629.pdf)** - Original FedAvg paper
- **[AUROC Explanation](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)** - Google ML Crash Course
- **[NIH CheXpert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/)** - Related dataset with papers

---

## Citation

If you use Federation-X in your research, please cite:

```bibtex
@misc{federationx2025,
  title = {Federation-X: Privacy-Preserving Federated Learning for Medical Imaging},
  author = {Cold Start Hackathon Team},
  year = {2025},
  howpublished = {\url{https://github.com/niranjanxprt/federation-x}}
}
```

---

## License

Federation-X is licensed under the [Apache License 2.0](https://github.com/niranjanxprt/federation-x/blob/main/LICENSE)

---

## Questions?

- **Issues**: [GitHub Issues](https://github.com/niranjanxprt/federation-x/issues)
- **Discussions**: [GitHub Discussions](https://github.com/niranjanxprt/federation-x/discussions)
- **Documentation**: See [docs/](./docs/) directory

---

**Last Updated**: November 15, 2025
**Status**: Active Development 🚀

---

[← Back to GitHub](https://github.com/niranjanxprt/federation-x)
