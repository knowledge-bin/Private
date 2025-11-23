#!/bin/bash
# Setup GPU Environment for PROFILE Ablation Study

set -e

echo "========================================"
echo "PROFILE GPU Environment Setup"
echo "========================================"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check if NVIDIA GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  Warning: nvidia-smi not found. GPU may not be available."
else
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Create conda environment
ENV_NAME="profile_gpu"
echo "Creating conda environment: $ENV_NAME"

if conda env list | grep -q "^$ENV_NAME "; then
    echo "⚠️  Environment '$ENV_NAME' already exists."
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME -y
    else
        echo "Using existing environment."
        conda activate $ENV_NAME
        exit 0
    fi
fi

conda create -n $ENV_NAME python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

echo ""
echo "✅ Conda environment created: $ENV_NAME"
echo ""

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install TensorFlow with GPU
echo "Installing TensorFlow with GPU support..."
pip install tensorflow[and-cuda]

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements_gpu.txt

# Install xMK-CKKS if available
echo ""
if [ -d "rlwe-xmkckks" ]; then
    echo "Installing xMK-CKKS from local directory..."
    cd rlwe-xmkckks
    pip install -e .
    cd ..
    echo "✅ xMK-CKKS installed"
else
    echo "⚠️  xMK-CKKS directory not found."
    echo "Please clone it manually:"
    echo "  git clone https://github.com/MetisPrometheus/rlwe-xmkckks.git"
    echo "  cd rlwe-xmkckks"
    echo "  pip install -e ."
    echo "  cd .."
fi

echo ""
echo "========================================"
echo "Verifying Installation"
echo "========================================"

# Verify GPU access
echo ""
echo "1. Checking PyTorch GPU access..."
python -c "import torch; print(f'   PyTorch version: {torch.__version__}'); print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'   GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

echo ""
echo "2. Checking TensorFlow GPU access..."
python -c "import tensorflow as tf; print(f'   TensorFlow version: {tf.__version__}'); gpus = tf.config.list_physical_devices('GPU'); print(f'   GPUs available: {len(gpus)}'); [print(f'     - {gpu.name}') for gpu in gpus]"

echo ""
echo "3. Checking core dependencies..."
python -c "import flwr; import numpy; import sklearn; import matplotlib; print('   ✅ All core packages installed')"

echo ""
echo "4. Checking xMK-CKKS..."
python -c "from rlwe_xmkckks import RLWE; print('   ✅ xMK-CKKS available')" 2>/dev/null || echo "   ⚠️  xMK-CKKS not installed (install manually)"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Activate environment with:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Next steps:"
echo "  1. Test setup: python test_ablation_setup.py"
echo "  2. Run single experiment: python run_single_ablation_experiment.py --config A_Bucketing_Only --attack label_flip --seed 42"
echo "  3. Run all experiments: ./run_all_30_experiments.sh"
echo ""
