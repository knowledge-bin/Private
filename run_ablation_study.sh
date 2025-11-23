#!/bin/bash
# Quick setup and run script for PROFILE Ablation Study

set -e  # Exit on error

echo "=========================================="
echo "PROFILE Ablation Study - MNIST LeNet-5"
echo "=========================================="
echo ""

# Check if in correct environment
if [[ "$CONDA_DEFAULT_ENV" != "homomorphic" ]]; then
    echo "⚠️  Warning: Not in 'homomorphic' environment"
    echo "Please activate: conda activate homomorphic"
    exit 1
fi

echo "✅ Environment: $CONDA_DEFAULT_ENV"
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import numpy, pandas, matplotlib, seaborn, sklearn, psutil" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ All dependencies installed"
else
    echo "❌ Missing dependencies. Installing..."
    pip install numpy pandas matplotlib seaborn scikit-learn psutil
fi

echo ""

# Check if this is a dry run
DRY_RUN=""
if [ "$1" == "--dry-run" ]; then
    DRY_RUN="--dry-run"
    echo "🧪 Running in DRY RUN mode (setup only)"
    echo ""
fi

# Run ablation study
echo "=========================================="
echo "Starting Ablation Study"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  - 5 configurations"
echo "  - 2 attacks (label-flip, min-max)"
echo "  - 3 seeds (42, 123, 456)"
echo "  - Total: 30 experiments"
echo ""

if [ -z "$DRY_RUN" ]; then
    echo "⏱️  Estimated time: 30-50 hours"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo ""
echo "🚀 Running ablation_mnist_lenet.py..."
echo ""

python3 ablation_mnist_lenet.py $DRY_RUN | tee ablation_study.log

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Ablation study failed. Check ablation_study.log"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Ablation Study Complete!"
echo "=========================================="
echo ""

# Find results directory
RESULTS_DIR=$(ls -td ablation_results_* 2>/dev/null | head -1)

if [ -z "$RESULTS_DIR" ]; then
    echo "⚠️  No results directory found"
    exit 0
fi

echo "Results saved to: $RESULTS_DIR"
echo ""

# Ask if user wants to analyze results
if [ -z "$DRY_RUN" ]; then
    read -p "Analyze results now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "📊 Analyzing results..."
        python3 plot_ablation_results.py "$RESULTS_DIR"
        
        echo ""
        echo "✅ Analysis complete!"
        echo "📁 Check: $RESULTS_DIR/figures/"
    fi
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Review results:"
echo "   cd $RESULTS_DIR"
echo ""
echo "2. Generate analysis (if not done):"
echo "   python3 plot_ablation_results.py $RESULTS_DIR"
echo ""
echo "3. Check figures:"
echo "   ls $RESULTS_DIR/figures/"
echo ""
echo "4. Use rebuttal text:"
echo "   cat $RESULTS_DIR/figures/rebuttal_paragraph.txt"
echo ""
echo "=========================================="
