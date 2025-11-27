# PROFILE Dependencies Guide

## Overview

PROFILE requires **custom** versions of both Flower and RLWE-xMKCKKS libraries that include homomorphic encryption (HE) and differential privacy (DP) support for federated learning.

**Note**: You must use these custom versions. Standard `pip install flwr` will not work.

---

## Required Custom Libraries

### 1. Custom Flower Framework (fl-core-bin) - PROFILE-Enhanced

**Repository**: https://github.com/knowledge-bin/fl-core-bin  
**Purpose**: Complete PROFILE system with integrated bucketing, validators, reputation, DP, and HE  
**Key Features** (server.py - 3857 lines):
- **Bucketing System**: Adaptive semantic bucketing for privacy amplification
- **Validator Ensemble**: Reputation-based Byzantine detection
- **Differential Privacy**: Bucket-level DP with Moments Accountant composition
- **Homomorphic Encryption**: xMK-CKKS threshold aggregation
- **Reputation System**: Self-correcting validator selection (no pre-knowledge needed)
- **Privacy Metrics**: Comprehensive logging and analysis tools

**Note**: This is NOT the base Flower framework. It contains the complete PROFILE implementation integrated into the Flower server module.

**Installation**:
```bash
# Clone the customized Flower repository
git clone https://github.com/knowledge-bin/fl-core-bin.git
cd fl-core-bin

# Install in development mode
pip install -e .

# Verify installation
python -c "import flwr; print(flwr.__version__)"
```

**Expected Output**: `1.4.0` or similar (custom build)

---

### 2. Homomorphic Encryption Library

**Repository**: https://github.com/knowledge-bin/crypto-utils.git  
**Purpose**: Multi-key CKKS homomorphic encryption scheme  
**Key Features**:
- Ring-LWE based encryption (n=8192, q≈218 bits)
- Multi-key homomorphic operations
- Threshold decryption for secure aggregation
- Efficient ciphertext aggregation

**Installation**:
```bash
# Clone the encryption library repository
git clone https://github.com/knowledge-bin/crypto-utils.git
cd crypto-utils

# Install in development mode
pip install -e .

# Verify installation
python -c "from rlwe_xmkckks import RLWE, Rq; print('xMK-CKKS OK')"
```

**Expected Output**: `xMK-CKKS OK`

---

## Installation Order

**Follow this sequence**:

### Step 1: Create Python Environment
```bash
conda create -n profile_gpu python=3.10 -y
conda activate profile_gpu
```

### Step 2: Install PyTorch with CUDA (GPU support)
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Step 3: Install TensorFlow with GPU
```bash
pip install tensorflow[and-cuda]
```

### Step 4: Install Custom Flower
```bash
git clone https://github.com/knowledge-bin/fl-core-bin.git
cd fl-core-bin
pip install -e .
cd ..
```

### Step 5: Install Encryption Library
```bash
git clone https://github.com/knowledge-bin/crypto-utils.git
cd crypto-utils
pip install -e .
cd ..
```

### Step 6: Install Other Dependencies
```bash
pip install numpy==1.24.3 pandas scikit-learn scipy matplotlib seaborn tqdm psutil grpcio protobuf
```

### Step 7: Verify Complete Setup
```bash
python test_ablation_setup.py
```

**Expected**: All 6 tests pass

---

## Why Custom Libraries Are Required

### Flower Modifications

**Standard Flower Limitation**: The official Flower framework (from `pip install flwr`) does not support:
- Homomorphic encryption during aggregation
- Custom encryption timing metrics
- Multi-key threshold decryption coordination

**Our Modifications** (`fl-core-bin`):
- **Integrated PROFILE system** into Flower server (server.py: 532 → 3857 lines)
- Bucketing system with adaptive client assignment
- Reputation-based validator ensemble selection
- Differential privacy with multiple composition methods (Moments Accountant, zCDP)
- Privacy metrics collection and IEEE-standard visualization
- Homomorphic encryption phases: public key setup, encrypted aggregation, threshold decryption
- Attack simulation support (label_flip, min_max, targeted)
- Ablation study controls (`--disable_bucketing`, `--disable_he`, `--disable_validation`, `--disable_dp`)

**Location of Changes**: `src/py/flwr/server/server.py` (entire file is PROFILE-enhanced)

**Note**: The `Clean-client2.py` and `PROFILE_server.py` files in the ablation package use this integrated server via `import flwr as fl`.

### RLWE-xMKCKKS Necessity

**Purpose**: Implements the xMK-CKKS scheme described in PROFILE manuscript
- **Ring-LWE Security**: Post-quantum secure encryption
- **Multi-Key Support**: Each client has independent key pair
- **Homomorphic Aggregation**: Server aggregates without decryption
- **Threshold Decryption**: Validators jointly decrypt aggregates

**Why Not Standard Libraries**:
- Standard CKKS (e.g., Microsoft SEAL) doesn't support multi-key
- TenSEAL lacks threshold decryption
- Pyfhel incompatible with federated key distribution

---

## Verification Checklist

After installation, verify each component:

### ✅ Custom Flower Installed
```bash
python -c "import flwr; import os; print(os.path.dirname(flwr.__file__))"
# Should show path containing "fl-core-bin"
```

### ✅ RLWE-xMKCKKS Available
```bash
python -c "from rlwe_xmkckks import RLWE, Rq; r = RLWE(); print('Success')"
# Should print: Success
```

### ✅ PyTorch CUDA Works
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# Should print: CUDA: True
```

### ✅ TensorFlow GPU Works
```bash
python -c "import tensorflow as tf; print('GPUs:', len(tf.config.list_physical_devices('GPU')))"
# Should print: GPUs: 1 (or higher)
```

### ✅ Complete System Test
```bash
python test_ablation_setup.py
# Should pass all 6 tests
```

---

## Common Installation Issues

### Issue 1: "ModuleNotFoundError: No module named 'flwr'"
**Cause**: Standard `pip install flwr` was used  
**Solution**: Uninstall and use custom version
```bash
pip uninstall flwr -y
cd fl-core-bin
pip install -e .
```

### Issue 2: "ImportError: cannot import name 'RLWE'"
**Cause**: RLWE-xMKCKKS not installed  
**Solution**: Clone and install
```bash
git clone https://github.com/knowledge-bin/crypto-utils.git
cd crypto-utils
pip install -e .
```

### Issue 3: "CUDA not available" despite having GPU
**Cause**: PyTorch installed without CUDA  
**Solution**: Reinstall with CUDA
```bash
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Issue 4: Flower version conflict
**Cause**: Multiple Flower installations  
**Solution**: Clean install
```bash
pip uninstall flwr -y
conda remove flwr -y
# Then reinstall custom version
cd fl-core-bin && pip install -e .
```

---

## For Reviewers: Quick Setup

If you're a reviewer wanting to run PROFILE experiments:

```bash
# 1. Clone PROFILE repository
git clone https://github.com/knowledge-bin/Private.git
cd Private

# 2. Run automated setup (installs everything)
bash setup_gpu_environment.sh

# 3. Verify
python test_ablation_setup.py

# 4. Run single test experiment
python run_single_ablation_experiment.py --config A_Bucketing_Only --attack label_flip --seed 42 --num-rounds 2
```

The `setup_gpu_environment.sh` script handles all custom library installations automatically.

---

## Version Information

| Component | Version | Source |
|-----------|---------|--------|
| Python | 3.10 | Conda |
| PyTorch | 2.0+ | Conda (with CUDA) |
| TensorFlow | 2.16+ | pip |
| **Flower (custom)** | **1.4.0** | **GitHub: knowledge-bin/fl-core-bin** |
| **RLWE-xMKCKKS** | **custom** | **GitHub: knowledge-bin/crypto-utils** |
| NumPy | 1.24.3 | pip |
| scikit-learn | 1.3+ | pip |

---

## Additional Notes

- **Development Mode (`-e`)**: Both custom libraries installed with `-e` flag for easier debugging
- **No PyPI**: These custom libraries are NOT on PyPI; must install from GitHub
- **Private Repositories**: If repositories are private, you'll need GitHub access tokens
- **GPU Recommended**: While CPU works, experiments take 10-20× longer
- **Disk Space**: ~5GB for environments + ~2GB for datasets

---

## Contact

If you encounter installation issues:
1. Check `TROUBLESHOOTING.md`
2. Verify all steps in this file were followed exactly
3. Run `python test_ablation_setup.py` and share output
4. Contact PROFILE research team with error messages

---

**Last Updated**: November 27, 2025  
**Tested On**: Ubuntu 20.04, CUDA 12.1, A100 GPUs
