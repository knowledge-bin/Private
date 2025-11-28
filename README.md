# PROFILE: Privacy-Preserving Federated Learning with Robust Detection

## Overview

PROFILE combines four defense mechanisms:
- **Bucketing**: Client grouping for privacy amplification
- **Multi-key CKKS Homomorphic Encryption**: Secure aggregation without plaintext exposure
- **Differential Privacy**: Bucket-level noise injection with Moments Accountant
- **Reputation-based Validators**: Byzantine detection through ensemble voting

This repository contains the ablation study with 5 configurations (A-E) tested against label-flipping and min-max attacks.

## Repository Contents

- **Core Implementation**: PROFILE_server.py (852 lines), Clean-client2.py (1046 lines)
- **Ablation Framework**: 5 configurations × 2 attacks × 3 seeds = 30 experiments
- **Models and Data**: LeNet-5 for MNIST, data partitioning utilities
- **Analysis**: Automated metrics extraction and LaTeX table generation

## Quick Start

**Note**: PROFILE requires custom Flower framework with integrated features. See `DEPENDENCIES.md` for details.

### 1. Clone Repository

```bash
git clone https://github.com/knowledge-bin/Private.git
cd Private
```

### 2. Install Dependencies

```bash
bash setup_gpu_environment.sh
```

Installs: Python 3.10 environment, PyTorch, TensorFlow, custom Flower (fl-core-bin), RLWE library (crypto-utils), and standard packages.

### 3. Verify Installation

```bash
python test_ablation_setup.py
```

Expected output: 6/6 tests pass. See `DEPENDENCIES.md` if tests fail.

### 4. Run Experiments

```bash
# Single experiment (2 rounds for testing)
python run_single_ablation_experiment.py \
    --config A_Bucketing_Only \
    --attack label_flip \
    --seed 42 \
    --num-rounds 2

# Full ablation study (30 experiments, 50 rounds each)
bash run_all_30_experiments.sh
```

### 5. Generate Analysis

```bash
python plot_ablation_results.py ablation_results/batch_YYYYMMDD_HHMMSS/
```

## Documentation

- **DEPENDENCIES.md** - Custom library installation guide  
- **README_ABLATION.md** - Detailed ablation study methodology
- **HE_COST_ANALYSIS.md** - Homomorphic encryption overhead analysis
- **QUICK_START.md** - Minimal setup instructions

## Ablation Study Design

30 experiments: 5 configurations × 2 attacks × 3 random seeds

| Config | Bucketing | HE | DP | Validators | Purpose |
|--------|-----------|----|----|------------|----------|
| A | ✓ | ✓ | × | × | Bucketing baseline |
| B | ✓ | ✓ | ✓ | × | Privacy impact |
| C | ✓ | ✓ | × | ✓ | Detection effectiveness |
| D | ✓ | ✓ | ✓ | ✓ | Full PROFILE |
| E | × | × | × | × | FedAvg baseline |

**Attacks**: Label-flip, Min-Max  
**Dataset**: MNIST with LeNet-5  
**Parameters**: 50 clients, 10 per round, 50 rounds, 20% malicious

## Expected Outputs

After running experiments and analysis:

- `ablation_table.csv` - Summary table with all metrics
- `ablation_table.tex` - LaTeX table for manuscript
- `accuracy_*.png` - Accuracy plots over rounds
- `detection_f1.png` - Detection performance bar chart

## Troubleshooting

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

## Performance Analysis

### Homomorphic Encryption Cost Analysis

PROFILE automatically logs HE performance metrics. To analyze:

```bash
# Automated analysis of all experiments
python analyze_he_costs.py

# Or manually from logs
grep "Public Key Aggregation Time" ablation_results/*/server.log
grep "Bucket.*processing time" ablation_results/*/server.log
```

**Expected HE Overhead:**
- Public key aggregation (one-time): ~41.7s for 50 clients
- Bucket processing: ~4.3s per bucket per round
- Total per round (16 buckets): ~71s (~1.2 minutes)

See `HE_COST_ANALYSIS.md` for detailed explanation and interpretation.

## Contact

For questions or issues, please open an issue in the repository.

## License

Provided for review purposes.

---

**Version**: 1.0  
**Last Updated**: January 2025  
**Framework**: Flower 1.5+ with xMK-CKKS
