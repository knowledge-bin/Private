# PROFILE: Privacy-Preserving Federated Learning with Robust Detection

**Private Repository - Do Not Share**

## 🎯 Overview

PROFILE is a federated learning framework that combines:
- **Bucketing**: Groups similar clients to limit adversarial influence
- **xMK-CKKS Homomorphic Encryption**: Secure aggregation without decryption
- **Differential Privacy**: Gradient perturbation for privacy protection
- **Byzantine Validators**: Reputation-based ensemble detection of malicious updates

**Architecture**: All PROFILE features are **integrated into the custom Flower server** (3857 lines).  
Reviewers will use this same integrated system by installing our modified `flower-xmkckks` package.

This repository contains the complete implementation for running ablation experiments on GPU servers.

## 📦 What's Included

- **Core FL System**: Server, client, model, data loading
- **Ablation Framework**: Automated experiment runner for 30 configurations
- **Analysis Tools**: Metrics collection, visualization, table generation
- **Documentation**: Setup guides, integration instructions, troubleshooting

## 🚀 Quick Start

⚠️ **IMPORTANT**: PROFILE requires **custom Flower** with integrated PROFILE features (bucketing, validators, reputation, DP, HE).  
📖 See `DEPENDENCIES.md` for why standard `pip install flwr` won't work - you need our `flower-xmkckks` repo.

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/profile-ablation.git
cd profile-ablation
```

### 2. Install Dependencies (Automatic)

```bash
# This script installs custom Flower + RLWE-xMKCKKS automatically
./setup_gpu_environment.sh
```

**What it does:**
- Creates conda environment with Python 3.10
- Installs PyTorch + TensorFlow with GPU support
- Clones and installs **custom Flower** from https://github.com/knowledge-bin/fl-core-bin
- Clones and installs **encryption library** from https://github.com/knowledge-bin/crypto-utils
- Installs all other dependencies

### 3. Verify Setup

```bash
python test_ablation_setup.py
```

Expected: `✅ All 6 tests pass`

**If tests fail**, see `DEPENDENCIES.md` for troubleshooting.

### 4. Run Experiments

```bash
# Single test experiment (~1 hour)
python run_single_ablation_experiment.py \
    --config A_Bucketing_Only \
    --attack label_flip \
    --seed 42

# All 30 experiments (30-50 hours)
./run_all_30_experiments.sh
```

### 5. Generate Analysis

```bash
python plot_ablation_results.py ablation_results_YYYYMMDD_HHMMSS/
```

## 📚 Documentation

- **DEPENDENCIES.md** - ⭐ **START HERE**: Custom library requirements and installation
- **README_ABLATION.md** - Detailed setup and usage guide
- **START_HERE.md** - Quick reference for running experiments
- **ABLATION_STUDY_README.md** - Complete experiment specification
- **INTEGRATION_GUIDE.py** - Advanced integration instructions

## 🔬 Ablation Study

**Configuration**: 5 configs × 2 attacks × 3 seeds = 30 experiments

| Config | Components | Purpose |
|--------|-----------|---------|
| A | Bucketing + HE | Baseline |
| B | Bucketing + HE + DP | Privacy impact |
| C | Bucketing + HE + Validators | Detection effectiveness |
| D | All components | Full system |
| E | FedAvg (no defense) | Comparison baseline |

**Attacks**: Label-flip, Min-Max  
**Dataset**: MNIST with LeNet-5  
**Parameters**: 50 clients, 10 per round, 50 rounds, 20% malicious

## 📊 Expected Outputs

After running experiments and analysis:

- `ablation_table.csv` - Summary table with all metrics
- `ablation_table.tex` - LaTeX table for manuscript
- `accuracy_*.png` - Accuracy plots over rounds
- `detection_f1.png` - Detection performance bar chart
- `rebuttal_paragraph.txt` - Pre-written rebuttal text

## 🐛 Troubleshooting

### GPU Out of Memory
```bash
# Reduce batch size or use smaller model
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

### Port Already in Use
```bash
# Check and kill existing processes
ps aux | grep PROFILE_server
kill <PID>
```

### Import Errors
```bash
# Verify all packages installed
python -c "import flwr, tensorflow, torch, numpy, sklearn"
python -c "from rlwe_xmkckks import RLWE"
```

## 🔐 Security

This is a **private repository** containing proprietary research code. Do not:
- Share code publicly
- Upload to public repositories
- Distribute without permission

## 📧 Contact

For questions or issues, contact the PROFILE research team.

## 📄 License

Proprietary - All Rights Reserved

---

**Version**: 1.0  
**Last Updated**: November 23, 2025  
**Framework**: Flower 1.5+ with xMK-CKKS
