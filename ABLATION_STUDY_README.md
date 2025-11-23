# PROFILE Ablation Study: MNIST with LeNet-5

## Overview

This ablation study implements the comprehensive evaluation requested by reviewers for the PROFILE federated learning system. It runs **30 experiments** (5 configurations × 2 attacks × 3 seeds) on MNIST with LeNet-5.

## Experiment Configuration

### Setup Parameters
- **Dataset**: MNIST
- **Model**: LeNet-5
- **Total Clients (K)**: 50 (simulated)
- **Clients per Round**: 10 (20% participation)
- **Global Rounds**: 50
- **Local Epochs**: 1
- **Batch Size**: 32
- **Learning Rate**: 0.01 (SGD with momentum=0.9)
- **Malicious Clients**: 10 (20% of 50)
- **Bucket Size**: 3 (maximizes adversary effect)
- **Number of Buckets**: ~16

### 5 Configurations Tested

#### A. Bucketing_Only
- **Description**: Bucketing with xMK-CKKS secure aggregation
- **Components**: Semantic bucketing + HE
- **No**: DP noise, validators

#### B. Bucketing+DP
- **Description**: Bucketing with DP noise
- **Components**: Semantic bucketing + HE + DP (σ=0.01)
- **No**: Validators

#### C. Bucketing+Validators
- **Description**: Bucketing with validation
- **Components**: Semantic bucketing + HE + Validators (E=5, S=0.3)
- **No**: DP noise

#### D. PROFILE_Full
- **Description**: Complete PROFILE system
- **Components**: ALL (Bucketing + HE + DP + Validators)
- **Parameters**: σ=0.01, E=5, S=0.3, reputation decay enabled

#### E. FedAvg_Baseline
- **Description**: Standard Federated Averaging
- **Components**: Secure aggregation only
- **No**: Bucketing, DP, validators

### 2 Attacks Tested

#### 1. Label-Flip
- **Type**: Simple poisoning
- **Method**: Flip true label `t` → `(t + 1) % 10`
- **Poison Ratio**: 100% of malicious client data
- **Purpose**: Basic robustness evaluation

#### 2. Min-Max (Scaled Gradient Proxy)
- **Type**: Sophisticated poisoning
- **Method**: Scale gradients by γ = min(50, K/m), optionally negate
- **Attack Strength**: 2.0
- **Target Class**: 1
- **Purpose**: Evaluate against stronger attacks

### Seeds
- **Seed 1**: 42
- **Seed 2**: 123
- **Seed 3**: 456

## File Structure

```
NEW_FL/
├── ablation_mnist_lenet.py          # Main experiment runner
├── ablation_metrics.py               # Comprehensive metrics collection
├── plot_ablation_results.py         # Analysis and visualization
├── ABLATION_STUDY_README.md         # This file
├── PROFILE_server.py                # Server implementation
├── Clean-client2.py                 # Client implementation
├── federated_data_loader.py         # Data partitioning
├── strong_attacks.py                # Min-Max and Fang attacks
└── ablation_results_YYYYMMDD_HHMMSS/
    ├── study_config.json
    ├── experiments_summary.json
    ├── mnist_lenet5_A_Bucketing_Only_label_flip_seed42/
    │   ├── experiment_config.json
    │   ├── mnist_lenet5_A_Bucketing_Only_label_flip_seed42.jsonl
    │   └── checkpoints/
    ├── mnist_lenet5_A_Bucketing_Only_label_flip_seed123/
    ├── ... (30 experiment directories total)
    └── figures/
        ├── ablation_table.csv
        ├── ablation_table.tex
        ├── accuracy_label_flip.png
        ├── accuracy_min_max.png
        ├── detection_f1.png
        └── rebuttal_paragraph.txt
```

## Metrics Collected Per Round

Each experiment saves per-round metrics in JSONL format:

### Accuracy Metrics
- `test_accuracy`: Overall test accuracy
- `test_loss`: Overall test loss
- `class_accuracies`: Per-class accuracy (dict)
- `mean_class_accuracy`: Average across classes

### Attack Metrics
- `attack_success_rate`: Proportion of misclassifications (ASR)

### Detection Metrics (Validators only)
- `detection_metrics.precision`: Detection precision
- `detection_metrics.recall`: Detection recall
- `detection_metrics.f1`: Detection F1 score
- `detection_metrics.true_positives`: TP count
- `detection_metrics.false_positives`: FP count

### Validation Metrics (Validators only)
- `validation_votes`: Validator votes per bucket

### Model Metrics
- `global_model_norm`: L2 norm of global model

### Communication Metrics
- `bytes_sent_this_round`: Bytes sent in round
- `bytes_received_this_round`: Bytes received in round
- `total_bytes_sent`: Cumulative bytes sent
- `total_bytes_received`: Cumulative bytes received

### Resource Metrics
- `elapsed_round_seconds`: Round duration
- `elapsed_total_seconds`: Total elapsed time
- `memory_mb`: Memory usage in MB

## How to Run

### Prerequisites

1. **Activate homomorphic environment**:
```bash
conda activate homomorphic
# OR
source ~/anaconda3/envs/homomorphic/bin/activate
```

2. **Install dependencies**:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn psutil
```

### Quick Start (Dry Run)

Test the setup without running experiments:

```bash
python ablation_mnist_lenet.py --dry-run
```

### Full Ablation Study

Run all 30 experiments:

```bash
python ablation_mnist_lenet.py
```

**Expected Duration**: ~30-50 hours (60-100 min per experiment)

### Run Specific Configuration

To run experiments manually, you can use your existing server/client scripts with appropriate flags.

Example for **Bucketing_Only** configuration:

```bash
# Terminal 1: Start server
python PROFILE_server.py \
    --dataset mnist \
    --num_clients 50 \
    --num_buckets 16 \
    --num_rounds 50 \
    --use_bucketing \
    --use_he \
    --no_dp \
    --no_validators

# Terminal 2-51: Start 50 clients (script this)
for i in {0..49}; do
    python Clean-client2.py \
        --client_id $i \
        --dataset mnist \
        --num_clients 50 \
        --seed $((42 + i)) \
        $(if [ $i -lt 10 ]; then echo "--malicious --attack_type label_flip"; fi) &
done
```

## Analyze Results

After experiments complete:

```bash
python plot_ablation_results.py ablation_results_YYYYMMDD_HHMMSS/
```

This generates:
1. **ablation_table.csv**: Summary table (CSV)
2. **ablation_table.tex**: LaTeX table for manuscript
3. **accuracy_label_flip.png**: Accuracy plot for label-flip
4. **accuracy_min_max.png**: Accuracy plot for min-max
5. **detection_f1.png**: Bar chart of detection F1
6. **rebuttal_paragraph.txt**: Suggested rebuttal text with numbers

## Expected Results

Based on prior FL literature and similar systems:

### FedAvg Baseline
- Under 20% malicious (label-flip): **20-40% accuracy**
- Severe degradation due to poisoning

### Bucketing_Only (A)
- Recovery to: **60-75% accuracy**
- Bucketing confines attacker influence

### Bucketing+Validators (C)
- Accuracy: **70-80%**
- Detection F1: **0.70-0.85**
- ASR reduction: **40% → 15-25%**

### Bucketing+DP (B)
- Slight utility drop vs A: **-2 to -5% absolute**
- DP adds noise but preserves bucketing benefits

### PROFILE_Full (D)
- Near robustness of C with privacy of B
- Expected: **68-78% accuracy**
- Small drop vs C but far better than FedAvg

### Variance Across Seeds
- Standard deviation: **±2-5%** with 3 seeds

## Reproducibility Package

For reviewer artifact submission:

### Include
1. **Code snapshot**: Git commit SHA
2. **requirements.txt**: Python dependencies
3. **Exact commands**: Scripts used to run experiments
4. **All JSONL results**: Raw metrics files
5. **Figures**: Generated plots and tables
6. **README**: Instructions to replay experiments

### Example commands.txt
```bash
# Git commit
git rev-parse HEAD > commit_sha.txt

# Run ablation
python ablation_mnist_lenet.py > ablation_run.log 2>&1

# Generate figures
python plot_ablation_results.py ablation_results_YYYYMMDD_HHMMSS/
```

## Troubleshooting

### Common Issues

#### 1. Memory Error
- **Symptom**: OOM during training
- **Solution**: Reduce `batch_size` or `clients_per_round`

#### 2. Slow Execution
- **Symptom**: >2 hours per experiment
- **Solution**: Check HE parameters, reduce `num_rounds`

#### 3. Import Errors
- **Symptom**: Cannot import modules
- **Solution**: Ensure `homomorphic` environment is activated

#### 4. No Detection Metrics
- **Symptom**: Detection F1 = N/A
- **Solution**: Only configs C and D have validators

## Integration with Existing Code

### Server Integration Points

Your existing `PROFILE_server.py` needs:

1. **Metrics collection**:
```python
from ablation_metrics import AblationMetricsCollector, CommunicationTracker

# Initialize
metrics = AblationMetricsCollector(
    experiment_name="...",
    results_dir="...",
    num_clients=50,
    num_malicious=10,
    malicious_client_ids=[0,1,2,3,4,5,6,7,8,9]
)

comm_tracker = CommunicationTracker()
```

2. **Per-round logging**:
```python
# In training loop
metrics.start_round(round_num)

# ... training ...

# After evaluation
metrics.log_round_metrics(
    round_num=round_num,
    test_accuracy=accuracy,
    test_loss=loss,
    predictions=preds,
    true_labels=labels,
    validation_votes=votes,  # If validators used
    detected_malicious_buckets=detected,  # If validators used
    bucket_assignments=assignments,
    bytes_sent_this_round=comm_tracker.round_bytes_sent,
    bytes_received_this_round=comm_tracker.round_bytes_received
)
```

3. **Finalize**:
```python
# After all rounds
metrics.finalize()
metrics.save_final_checkpoint(model, "checkpoints/")
```

### Client Integration Points

Your existing `Clean-client2.py` needs:

1. **Communication tracking**:
```python
from ablation_metrics import CommunicationTracker

comm_tracker = CommunicationTracker()

# Before sending update
bytes_sent = comm_tracker.track_model_send(parameters)

# After receiving model
bytes_received = comm_tracker.track_model_receive(parameters)
```

## Timeline for Completion

- **Day 1**: Setup and dry-run testing (2 hours)
- **Day 2-3**: Run all 30 experiments (30-50 hours, can run overnight)
- **Day 4**: Analyze results and generate figures (2 hours)
- **Day 5**: Write rebuttal with numbers (2 hours)

**Total**: ~5 days (mostly background execution)

## Contact

For questions about the ablation study setup, refer to this README or check the inline documentation in:
- `ablation_mnist_lenet.py`
- `ablation_metrics.py`
- `plot_ablation_results.py`

## References

- McMahan et al. (2017): Communication-Efficient Learning of Deep Networks from Decentralized Data
- Bagdasaryan et al. (2020): How To Backdoor Federated Learning
- Li et al. (2020): Federated Learning: Challenges, Methods, and Future Directions
- Wang et al. (2020): Attack of the Tails: Yes, You Really Can Backdoor Federated Learning
