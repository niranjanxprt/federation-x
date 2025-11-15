# Cold Start Hackathon: Federated Learning for X-ray Classification

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flower Framework](https://img.shields.io/badge/Framework-Flower-brightgreen.svg)](https://flower.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/niranjanxprt/federation-x?style=social)](https://github.com/niranjanxprt/federation-x)

This challenge builds on the NIH Chest X-Ray dataset, which contains over **112,000 medical images** from **30,000 patients**. Participants will explore how federated learning can enable robust diagnostic models that generalize across hospitals, without sharing sensitive patient data.

## 🎯 Key Features

- **Privacy-Preserving**: Hospital data stays local; only model updates are shared
- **Non-IID Data**: Realistic simulation of diverse hospital environments
- **Multi-Hospital Setup**: Three distinct hospital silos with unique characteristics
- **Binary Classification**: Detect presence of any pathological finding
- **Large-Scale Dataset**: 112,000+ medical images across distributed nodes
- **GPU-Optimized Training**: Cluster-based distributed learning with resource management

## Background

In real healthcare systems, hospitals differ in their imaging devices, patient populations, and clinical practices. A model trained in one hospital often struggles in another, but because the data distributions differ.

Your task is to design a model that performs reliably across diverse hospital environments. By simulating a federated setup, where each hospital trains on local data and only model updates are shared, you’ll investigate how distributed AI can improve performance and robustness under privacy constraints.

## 🏥 Hospital Data Distribution

Chest X-rays are among the most common and cost-effective imaging exams, yet diagnosing them remains challenging.
For this challenge, the dataset has been artificially partitioned into hospital silos to simulate a federated learning scenario with **strong non-IID characteristics**. Each patient appears in only one silo. However, age, sex, view position, and pathology distributions vary across silos.

Each patient appears in only one hospital. All splits (train/eval/test) are patient-disjoint to prevent data leakage.

### Hospital A: Portable Inpatient (42,093 test, 5,490 eval)
- **Demographics**: Elderly males (age 60+)
- **Equipment**: AP (anterior-posterior) view dominant
- **Common findings**: Fluid-related conditions (Effusion, Edema, Atelectasis)

### Hospital B: Outpatient Clinic (21,753 train, 2,860 eval)
- **Demographics**: Younger females (age 20-65)
- **Equipment**: PA (posterior-anterior) view dominant
- **Common findings**: Nodules, masses, pneumothorax

### Hospital C: Mixed with Rare Conditions (20,594 train, 2,730 eval)
- **Demographics**: Mixed age and gender
- **Equipment**: PA view preferred
- **Common findings**: Rare conditions (Hernia, Fibrosis, Emphysema)


## 📊 Task Details

**Binary classification**: Detect presence of any pathological finding
- **Class 0**: No Finding
- **Class 1**: Any Finding present

**Pathologies (15 types)**: Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia

**Evaluation Metric**: [AUROC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)


## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone your team's repository
git clone https://github.com/YOUR_ORG/hackathon-2025-team-YOUR_TEAM.git
cd hackathon-2025-team-YOUR_TEAM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e .
```

### 2. Test Locally (Optional)

```bash
python local_train.py --hospital A
```

Note: Full datasets are only available on the cluster.

### 3. Submit Jobs to Cluster

```bash
# Submit training job
./submit-job.sh "flwr run . cluster --stream" --gpu

# Submit with custom name for easier tracking
./submit-job.sh "flwr run . cluster --stream" --gpu --name exp_lr001

# Test evaluation pipeline
./submit-job.sh "python evaluate.py" --gpu --name eval_v5
```

### 4. Monitor Results

```bash
# Check job status
squeue -u $USER

# View logs
tail -f ~/logs/exp_lr001_*.out

# View W&B dashboard
# https://wandb.ai/coldstart2025-teamXX/coldstart2025
```


## 📚 Dataset Details

Datasets on cluster:
- **Raw**: `/shared/hackathon/datasets/xray_fl_datasets/`
- **Preprocessed (128x128)**: `/shared/hackathon/datasets/xray_fl_datasets_preprocessed_128/`

These are automatically linked in your job workspace.

## ⚙️ Resource Limits

Per job:
- **1 GPU**
- **32GB RAM**
- **20 minutes** runtime
- **Max 4 concurrent jobs** per team

## 📊 Weights & Biases

All metrics automatically logged to W&B: `https://wandb.ai/coldstart2025-teamXX/coldstart2025`

Login with your team's service account credentials (provided by organizers).

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **Federated Learning** | Flower Framework |
| **Deep Learning** | PyTorch |
| **Experiment Tracking** | Weights & Biases |
| **Data Processing** | NumPy, Pandas, OpenCV |
| **Infrastructure** | HPC Cluster with GPU (NVIDIA) |

## 📁 Repository Structure

```
federation-x/
├── cold_start_hackathon/
│   ├── server_app.py          # Federated server implementation
│   ├── client_app.py          # Client-side training logic
│   ├── models/                # Neural network architectures
│   └── utils/                 # Helper utilities
├── local_train.py             # Local testing script
├── evaluate.py                # Evaluation pipeline
├── submit-job.sh              # Cluster job submission script
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
└── README.md                  # This file
```

## 📚 Additional Resources

- **[Flower Framework Documentation](https://flower.ai/)** - Federated learning framework reference
- **[AUROC Explanation](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)** - Understanding the evaluation metric
- **[Federated Learning Overview](https://arxiv.org/pdf/1602.05629.pdf)** - Academic foundation paper
- **[NIH Chest X-Ray Dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)** - Original dataset information

## 🎓 Learning Objectives

By completing this challenge, you'll master:
- ✅ Federated Learning fundamentals and architectures
- ✅ Non-IID data challenges and mitigation strategies
- ✅ Distributed training at scale
- ✅ Privacy-preserving machine learning
- ✅ Medical image analysis and classification
- ✅ Experiment tracking and reproducibility

## 📝 Dataset Reference

```
@article{wang2017chestxray,
  title={ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks},
  author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and
          Bagheri, Mohammadhadi and Summers, Ronald M},
  journal={CVPR},
  year={2017}
}
```

---

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Report bugs via GitHub Issues
- Submit improvements via Pull Requests
- Share your results and insights

## 📧 Contact & Support

- **Repository**: https://github.com/niranjanxprt/federation-x
- **Issues**: https://github.com/niranjanxprt/federation-x/issues
- **Organizers**: Contact the hackathon team for cluster access and credentials

---

**Good luck, and happy hacking!** 🚀

*Last Updated: November 15, 2025*
