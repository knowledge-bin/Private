#!/bin/bash
# Run All 30 Ablation Experiments
# 5 configs × 2 attacks × 3 seeds = 30 experiments

set -e

echo "========================================"
echo "Running All 30 Ablation Experiments"
echo "========================================"
echo ""

# Activate environment
if [[ "$CONDA_DEFAULT_ENV" != "profile_gpu" ]]; then
    echo "⚠️  Please activate profile_gpu environment:"
    echo "   conda activate profile_gpu"
    exit 1
fi

# Check if server and client exist
if [[ ! -f "PROFILE_server.py" ]] || [[ ! -f "Clean-client2.py" ]]; then
    echo "❌ PROFILE_server.py or Clean-client2.py not found"
    exit 1
fi

# Configurations
CONFIGS=("A_Bucketing_Only" "B_Bucketing_DP" "C_Bucketing_Validators" "D_PROFILE_Full" "E_FedAvg_Baseline")
ATTACKS=("label_flip" "min_max")
SEEDS=(42 123 456)

# Results directory
RESULTS_DIR="ablation_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Counter
TOTAL=$((5 * 2 * 3))
CURRENT=0
FAILED=0

# Log file
LOG_FILE="$RESULTS_DIR/experiments.log"
echo "Experiment log: $LOG_FILE"
echo ""

# Start time
START_TIME=$(date +%s)

# Run all experiments
for config in "${CONFIGS[@]}"; do
    for attack in "${ATTACKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            CURRENT=$((CURRENT + 1))
            
            echo "========================================"
            echo "Experiment $CURRENT/$TOTAL"
            echo "Config: $config"
            echo "Attack: $attack"  
            echo "Seed: $seed"
            echo "========================================"
            
            # Run experiment
            python3 run_single_ablation_experiment.py \
                --config "$config" \
                --attack "$attack" \
                --seed "$seed" \
                --results-dir "$RESULTS_DIR" \
                2>&1 | tee -a "$LOG_FILE"
            
            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                echo "✅ Experiment $CURRENT completed successfully"
            else
                echo "❌ Experiment $CURRENT FAILED"
                FAILED=$((FAILED + 1))
            fi
            
            echo ""
            
            # Brief pause between experiments
            sleep 5
        done
    done
done

# End time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))

echo "========================================"
echo "All Experiments Complete!"
echo "========================================"
echo "Total: $TOTAL"
echo "Completed: $((TOTAL - FAILED))"
echo "Failed: $FAILED"
echo "Time: ${HOURS}h ${MINUTES}m"
echo "Results: $RESULTS_DIR"
echo "========================================"

# Generate summary
python3 << EOF
import json
from pathlib import Path

results_dir = Path("$RESULTS_DIR")
experiments = list(results_dir.glob("*/experiment_config.json"))

summary = {
    "total_experiments": $TOTAL,
    "completed": $((TOTAL - FAILED)),
    "failed": $FAILED,
    "elapsed_seconds": $ELAPSED,
    "experiments": []
}

for exp_config in experiments:
    with open(exp_config) as f:
        summary["experiments"].append(json.load(f))

summary_file = results_dir / "all_experiments_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved: {summary_file}")
EOF

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "🎉 All experiments completed successfully!"
    echo ""
    echo "Next step: Generate analysis"
    echo "  python plot_ablation_results.py $RESULTS_DIR"
else
    echo ""
    echo "⚠️  Some experiments failed. Check logs."
fi
