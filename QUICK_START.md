# 🚀 PROFILE GitHub Setup - Quick Reference

## ✅ What Was Created

A **clean repository** in `profile-ablation-clean/` with only essential files (20 files, 512KB).

### Essential Files Included:
- ✅ Core PROFILE system (server, client, models, data loading)
- ✅ Ablation experiment framework (runner, metrics, analysis)
- ✅ Attack implementations (label-flip, min-max)
- ✅ Complete documentation (4 README files)
- ✅ GPU setup scripts
- ✅ Dependencies list
- ✅ .gitignore (excludes results, logs, data)

### Excluded (automatically by .gitignore):
- ❌ Experiment results (ablation_results_*)
- ❌ Temporary test files
- ❌ Data directories
- ❌ Logs and metrics
- ❌ Old experimental code

---

## 📋 Push to GitHub (Step-by-Step)

### 1️⃣ Create Private GitHub Repository

```bash
# Go to: https://github.com/new
# Repository name: profile-ablation
# Visibility: ✅ Private (IMPORTANT!)
# Do NOT initialize with README
# Click "Create repository"
```

### 2️⃣ Push Code from Your Machine

```bash
cd /home/bderessa/NEW_FL/profile-ablation-clean

# Initialize git
git init
git add .
git commit -m "Initial commit: PROFILE ablation study framework"

# Connect to GitHub (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/profile-ablation.git

# Push
git branch -M main
git push -u origin main
```

### 3️⃣ Setup on GPU Server

```bash
# SSH to GPU server
ssh user@your-gpu-server.com

# Clone repository (replace YOUR_USERNAME)
git clone https://github.com/YOUR_USERNAME/profile-ablation.git
cd profile-ablation

# Setup environment
./setup_gpu_environment.sh

# Verify
python test_ablation_setup.py
```

### 4️⃣ Run Experiments on GPU

```bash
# Activate environment
conda activate profile_gpu

# Test single experiment
python run_single_ablation_experiment.py \
    --config A_Bucketing_Only \
    --attack label_flip \
    --seed 42

# Run all 30 experiments (background)
nohup ./run_all_30_experiments.sh > ablation.log 2>&1 &

# Monitor progress
tail -f ablation.log
watch -n 1 nvidia-smi
```

### 5️⃣ Generate Results

```bash
# After experiments complete
python plot_ablation_results.py ablation_results_YYYYMMDD_HHMMSS/

# Download results to local machine
scp -r user@gpu-server:~/profile-ablation/ablation_results_*/ ./results/
```

---

## 🔑 Key Commands

### On Local Machine (Development)
```bash
# Update code
cd /home/bderessa/NEW_FL/profile-ablation-clean
git add .
git commit -m "Update: description"
git push
```

### On GPU Server (Execution)
```bash
# Get latest code
cd ~/profile-ablation
git pull

# Run experiments
conda activate profile_gpu
./run_all_30_experiments.sh
```

### Check Status
```bash
# Local: Check what will be pushed
git status
git diff

# GPU: Check running experiments
ps aux | grep PROFILE_server
tail -f ablation_results_*/experiments.log
```

---

## 📁 What's in the Clean Repository

```
profile-ablation-clean/
├── README.md                          # Main documentation
├── PUSH_TO_GITHUB.md                  # Push instructions
├── README_ABLATION.md                 # Detailed setup guide
├── START_HERE.md                      # Quick start
├── ABLATION_STUDY_README.md          # Experiment specs
├── INTEGRATION_GUIDE.py               # Advanced integration
│
├── Core System (8 files)
│   ├── PROFILE_server.py              # FL server
│   ├── Clean-client2.py               # FL client
│   ├── cnn.py                         # LeNet-5 model
│   ├── utils.py                       # Utilities
│   ├── federated_data_loader.py       # Data partitioning
│   ├── load_covid.py                  # MNIST loading
│   ├── strong_attacks.py              # Attacks
│   └── detect.py                      # Detection utilities
│
├── Ablation Framework (5 files)
│   ├── run_single_ablation_experiment.py
│   ├── run_all_30_experiments.sh
│   ├── ablation_metrics.py
│   ├── plot_ablation_results.py
│   └── test_ablation_setup.py
│
└── Setup (3 files)
    ├── requirements_gpu.txt           # Dependencies
    ├── setup_gpu_environment.sh       # Auto-setup
    └── .gitignore                     # Git exclusions
```

**Total**: 21 files, 512 KB (no bloat!)

---

## ⚠️ Important Notes

1. **Keep Repository Private**: Contains proprietary research code
2. **Don't commit results**: .gitignore excludes them automatically
3. **xMK-CKKS separate**: Must clone and install separately (included in setup script)
4. **GPU required**: Experiments need CUDA-capable GPU
5. **Long running**: 30-50 hours for full ablation study

---

## 🎯 Expected Timeline

| Task | Time | Where |
|------|------|-------|
| Push to GitHub | 5 min | Local machine |
| Clone on GPU server | 2 min | GPU server |
| Environment setup | 15 min | GPU server |
| Verify setup | 5 min | GPU server |
| Test single experiment | 1 hour | GPU server |
| Run all 30 experiments | 30-50 hours | GPU server (background) |
| Generate analysis | 1-2 hours | GPU server or local |

**Total active time**: ~2 hours  
**Total background time**: 30-50 hours

---

## 🆘 Troubleshooting

### Git Push Fails
```bash
# If authentication fails, use personal access token
# GitHub Settings → Developer settings → Personal access tokens
# Use token as password when prompted
```

### xMK-CKKS Not Found
```bash
# Install manually after cloning
git clone https://github.com/MetisPrometheus/rlwe-xmkckks.git
cd rlwe-xmkckks
pip install -e .
```

### GPU Out of Memory
```bash
# Limit TensorFlow memory
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Or reduce batch size in cnn.py
# Change: batch_size = 32 → batch_size = 16
```

### Port 8080 In Use
```bash
# Find and kill process
lsof -ti:8080 | xargs kill -9

# Or change port in PROFILE_server.py
# Add: --port 8081
```

---

## ✅ Checklist

### Before Pushing
- [ ] Reviewed files in `profile-ablation-clean/`
- [ ] Created private GitHub repository
- [ ] Have GitHub credentials ready

### After Pushing
- [ ] Repository shows as "Private" on GitHub
- [ ] All 21 files visible on GitHub
- [ ] README.md displays correctly

### On GPU Server
- [ ] Repository cloned successfully
- [ ] Environment setup completed
- [ ] `python test_ablation_setup.py` passes (6/6)
- [ ] GPU detected: `nvidia-smi` works
- [ ] xMK-CKKS installed

### Running Experiments
- [ ] Single test experiment runs (~1 hour)
- [ ] All 30 experiments launched
- [ ] Monitoring with `tail -f` or `screen`
- [ ] Results directory created

---

## 📧 Quick Help

**Repository ready?** ✅ Yes! Directory: `profile-ablation-clean/`

**Next action**: 
1. Create private GitHub repo
2. `cd profile-ablation-clean`
3. Run commands from section 2️⃣ above

**Questions?** Check `PUSH_TO_GITHUB.md` in the clean directory.

---

*Last Updated: November 23, 2025*
