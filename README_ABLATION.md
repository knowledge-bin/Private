# PROFILE Ablation Study - GPU Server Setup

## 🎯 Overview

This repository contains the complete code for running PROFILE ablation experiments on a GPU server. The ablation study evaluates 5 configurations × 2 attacks × 3 seeds = **30 experiments** on MNIST with LeNet-5.

## 📦 Repository Structure

```
.
├── README_ABLATION.md              # This file
├── requirements_gpu.txt            # Python dependencies
├── setup_gpu_environment.sh        # Environment setup script
│
├── Core PROFILE System
│   ├── PROFILE_server.py          # Main FL server with bucketing + HE + validators
│   ├── Clean-client2.py           # FL client with attack support
│   ├── cnn.py                     # LeNet-5 model definition
│   ├── utils.py                   # Utility functions
│   ├── federated_data_loader.py   # Data partitioning
│   ├── load_covid.py              # MNIST data loading
│   ├── strong_attacks.py          # MinMax and Fang attacks
│   └── detect.py                  # Detection utilities
│
├── Ablation Framework
│   ├── run_single_ablation_experiment.py   # Run one experiment
│   ├── run_all_30_experiments.sh           # Run all 30 experiments
│   ├── ablation_metrics.py                 # Metrics collection
│   ├── plot_ablation_results.py            # Analysis and visualization
│   └── test_ablation_setup.py              # Verification script
│
└── Documentation
    ├── ABLATION_STUDY_README.md     # Detailed user guide
    ├── INTEGRATION_GUIDE.py         # Integration instructions
    └── START_HERE.md                # Quick start guide
```

## 🚀 Quick Setup on GPU Server

### Step 1: Clone Repository

```bash
# Clone from your private GitHub
git clone https://github.com/YOUR_USERNAME/profile-ablation.git
cd profile-ablation
```

### Step 2: Install xMK-CKKS Homomorphic Encryption

```bash
# Clone the xMK-CKKS library
git clone https://github.com/MetisPrometheus/rlwe-xmkckks.git
cd rlwe-xmkckks
pip install -e .
cd ..
```

### Step 3: Setup Python Environment

```bash
# Create conda environment with GPU support
conda create -n profile_gpu python=3.10 -y
conda activate profile_gpu

# Install PyTorch with CUDA (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install TensorFlow with GPU
pip install tensorflow[and-cuda]

# Install other dependencies
pip install -r requirements_gpu.txt
```

### Step 4: Verify GPU Access

```bash
# Check NVIDIA GPU
nvidia-smi

# Test PyTorch GPU
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"

# Test TensorFlow GPU
python -c "import tensorflow as tf; print(f'TensorFlow GPUs: {tf.config.list_physical_devices(\"GPU\")}')"
```

### Step 5: Verify Setup

```bash
# Run verification tests
python test_ablation_setup.py
```

Expected output: `✅ All 6 tests pass`

## 🎮 Running Experiments

### Option 1: Test Single Experiment (~1 hour)

```bash
conda activate profile_gpu

python run_single_ablation_experiment.py \
    --config A_Bucketing_Only \
    --attack label_flip \
    --seed 42
```

### Option 2: Run All 30 Experiments (30-50 hours)

```bash
conda activate profile_gpu

# Run in background with logging
nohup ./run_all_30_experiments.sh > ablation_run.log 2>&1 &

# Monitor progress
tail -f ablation_run.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Option 3: Run Experiments in Parallel (if multiple GPUs)

```bash
# Edit run_all_30_experiments.sh and add:
# export CUDA_VISIBLE_DEVICES=0  # For GPU 0
# Or run multiple instances on different GPUs

# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python run_single_ablation_experiment.py --config A_Bucketing_Only --attack label_flip --seed 42

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python run_single_ablation_experiment.py --config B_Bucketing_DP --attack label_flip --seed 42
```

## 📊 After Experiments Complete

### Generate Analysis and Figures

```bash
# Find your results directory
ls -lh ablation_results_*/

# Run analysis
python plot_ablation_results.py ablation_results_YYYYMMDD_HHMMSS/
```

This generates:
- `ablation_table.csv` - Summary table
- `ablation_table.tex` - LaTeX table for manuscript
- `accuracy_label_flip.png` - Accuracy over rounds (label-flip attack)
- `accuracy_min_max.png` - Accuracy over rounds (min-max attack)
- `detection_f1.png` - Detection F1 scores bar chart
- `rebuttal_paragraph.txt` - Pre-written rebuttal text with numbers

## 🔍 Experiment Configurations

| Config | Bucketing | HE | DP (σ) | Validators | Purpose |
|--------|-----------|----|---------|-----------|---------| 
| **A** Bucketing_Only | ✅ | ✅ | ❌ | ❌ | Baseline bucketing benefit |
| **B** Bucketing+DP | ✅ | ✅ | 0.01 | ❌ | Privacy-utility tradeoff |
| **C** Bucketing+Validators | ✅ | ✅ | ❌ | 5 per bucket | Detection effectiveness |
| **D** PROFILE_Full | ✅ | ✅ | 0.01 | 5 per bucket | Complete system |
| **E** FedAvg_Baseline | ❌ | ✅ | ❌ | ❌ | No-defense baseline |

### Attacks

- **Label-Flip**: Simple poisoning (t → (t+1) % 10)
- **Min-Max**: Sophisticated scaled gradients (γ=50)

### Parameters

- **Total Clients (K)**: 50
- **Clients per Round**: 10 (20% participation)
- **Malicious Clients**: 10 (20%, IDs 0-9)
- **Global Rounds**: 50
- **Dataset**: MNIST (LeNet-5)
- **Seeds**: 42, 123, 456

## 📈 Expected Results

Based on federated learning literature:

| Configuration | Test Accuracy | Attack Success | Detection F1 |
|---------------|---------------|----------------|--------------|
| E (FedAvg) | 20-40% | 60-80% | N/A |
| A (Bucketing) | 60-75% | 30-50% | N/A |
| B (+ DP) | 58-72% | 32-52% | N/A |
| C (+ Validators) | 70-80% | 15-25% | 0.70-0.85 |
| D (Full) | 68-78% | 17-27% | 0.68-0.82 |

## 🐛 Troubleshooting

### GPU Out of Memory

```bash
# Reduce batch size in cnn.py (default: 32)
# Edit: batch_size = 16

# Or limit GPU memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

### Connection Issues

```bash
# Check if ports are available
netstat -tuln | grep 8080

# Change server port if needed
python PROFILE_server.py --port 8081
```

### Environment Issues

```bash
# Verify all imports work
python -c "import flwr, tensorflow, torch, numpy, sklearn, matplotlib"

# Check rlwe_xmkckks
python -c "from rlwe_xmkckks import RLWE; print('xMK-CKKS OK')"
```

## 📝 Important Notes

1. **GPU Memory**: Each experiment needs ~4-6 GB GPU memory. Monitor with `nvidia-smi`.

2. **Long Running**: Full ablation study takes 30-50 hours. Use `nohup` or `screen`:
   ```bash
   screen -S ablation
   ./run_all_30_experiments.sh
   # Detach: Ctrl+A, D
   # Reattach: screen -r ablation
   ```

3. **Checkpoint Saving**: Results are auto-saved after each experiment. Safe to interrupt and resume.

4. **Disk Space**: Each experiment generates ~500MB (metrics + checkpoints). Total: ~15GB.

## 🔐 Private Repository

This repository is **private** and contains proprietary PROFILE implementation. Do not share without permission.

## 📧 Contact

For issues or questions about the ablation study, contact the PROFILE team.

## 🎯 Next Steps After Results

1. ✅ Generate analysis figures
2. ✅ Review ablation_table.csv for numbers
3. ✅ Include LaTeX table in manuscript
4. ✅ Use rebuttal_paragraph.txt for reviewer response
5. ✅ Package reproducibility artifact (code + results + README)

---

**Last Updated**: November 23, 2025  
**PROFILE Version**: 1.0  
**Ablation Framework Version**: 1.0
