#!/usr/bin/env python3
"""
Integration Guide: Connect Ablation Study with Existing PROFILE Code

This file shows EXACTLY how to integrate:
1. AblationMetricsCollector into PROFILE_server.py
2. CommunicationTracker into both server and client
3. Configuration flags for 5 ablation configs
4. Attack configuration for clients

CRITICAL: This file provides COPY-PASTE code snippets for integration.
"""

# ============================================================================
# PART 1: Server-Side Integration (PROFILE_server.py)
# ============================================================================

"""
Step 1: Add imports at the top of PROFILE_server.py
"""

# ADD THESE IMPORTS:
from ablation_metrics import AblationMetricsCollector, CommunicationTracker

"""
Step 2: Parse ablation configuration flags
"""

# ADD THESE ARGUMENTS to your argument parser:
parser.add_argument('--experiment_name', type=str, required=True,
                   help='Unique experiment name (e.g., mnist_lenet5_A_Bucketing_Only_label_flip_seed42)')
parser.add_argument('--results_dir', type=str, default='ablation_results',
                   help='Directory to save ablation results')
parser.add_argument('--use_bucketing', action='store_true',
                   help='Enable bucketing')
parser.add_argument('--use_dp', action='store_true',
                   help='Enable DP noise')
parser.add_argument('--dp_sigma', type=float, default=0.01,
                   help='DP noise standard deviation')
parser.add_argument('--use_validators', action='store_true',
                   help='Enable validators')
parser.add_argument('--validators_per_bucket', type=int, default=5,
                   help='Number of validators per bucket')
parser.add_argument('--validation_threshold', type=float, default=0.3,
                   help='Validation threshold for flagging buckets')
parser.add_argument('--malicious_client_ids', type=str, default='0,1,2,3,4,5,6,7,8,9',
                   help='Comma-separated list of malicious client IDs')

"""
Step 3: Initialize metrics collector in main()
"""

# AFTER parsing args, ADD:
malicious_ids = [int(x) for x in args.malicious_client_ids.split(',')]

metrics_collector = AblationMetricsCollector(
    experiment_name=args.experiment_name,
    results_dir=args.results_dir,
    num_clients=args.num_clients,  # Your existing num_clients arg
    num_malicious=len(malicious_ids),
    malicious_client_ids=malicious_ids
)

comm_tracker = CommunicationTracker()

"""
Step 4: Apply configuration flags
"""

# ADD THIS LOGIC to control PROFILE components:

# Bucketing
if args.use_bucketing:
    print("✅ Bucketing ENABLED")
    # Use your existing bucketing logic
else:
    print("❌ Bucketing DISABLED (FedAvg mode)")
    # Skip bucketing, use standard aggregation

# DP Noise
if args.use_dp:
    print(f"✅ DP ENABLED (σ={args.dp_sigma})")
    dp_sigma = args.dp_sigma
else:
    print("❌ DP DISABLED")
    dp_sigma = 0.0  # No DP noise

# Validators
if args.use_validators:
    print(f"✅ Validators ENABLED (E={args.validators_per_bucket}, S={args.validation_threshold})")
    num_validators = args.validators_per_bucket
    validation_threshold = args.validation_threshold
else:
    print("❌ Validators DISABLED")
    num_validators = 0

"""
Step 5: Track communication in training loop
"""

# FIND YOUR TRAINING LOOP and ADD:

for round_num in range(1, args.num_rounds + 1):
    
    # Start round timing
    metrics_collector.start_round(round_num)
    comm_tracker.reset_round()
    
    # === Your existing training code ===
    
    # WHEN SENDING MODEL TO CLIENTS:
    # (Find where you serialize/send model parameters)
    bytes_sent = comm_tracker.track_model_send(model_parameters)  # model_parameters = list of np arrays
    
    # WHEN RECEIVING UPDATES FROM CLIENTS:
    # (Find where you receive/deserialize client updates)
    for client_update in client_updates:
        bytes_received = comm_tracker.track_model_receive(client_update)
    
    # IF USING HE (encrypted communication):
    # encrypted_size = len(encrypted_data_bytes)
    # comm_tracker.track_encrypted_send(encrypted_size)
    
    # === End of training code ===
    
    # AFTER AGGREGATION, BEFORE EVALUATION:
    # Calculate global model norm
    global_model_norm = np.sqrt(sum(np.sum(p**2) for p in model_parameters))
    
    # AFTER EVALUATION (with test set):
    # Get predictions and labels
    predictions = model.predict(X_test)  # Your prediction logic
    predictions = np.argmax(predictions, axis=1)  # Convert to class labels
    
    # IF USING VALIDATORS:
    # Extract validation votes and detected buckets
    validation_votes = {
        'bucket_id': bucket_id,
        'validator_votes': votes_list,
        'flagged': is_flagged
    }  # Your validator logic
    
    detected_malicious_buckets = [b for b in buckets if buckets[b]['flagged']]
    
    # Get bucket assignments
    bucket_assignments = {client_id: bucket_id for ...}  # Your bucketing logic
    
    # Get communication stats
    comm_stats = comm_tracker.get_round_communication()
    
    # LOG METRICS
    metrics_collector.log_round_metrics(
        round_num=round_num,
        test_accuracy=test_accuracy,  # Your accuracy calculation
        test_loss=test_loss,          # Your loss calculation
        predictions=predictions,
        true_labels=y_test,
        num_classes=10,  # For MNIST
        validation_votes=validation_votes if args.use_validators else None,
        detected_malicious_buckets=detected_malicious_buckets if args.use_validators else None,
        bucket_assignments=bucket_assignments if args.use_bucketing else None,
        global_model_norm=global_model_norm,
        bytes_sent_this_round=comm_stats['bytes_sent'],
        bytes_received_this_round=comm_stats['bytes_received']
    )

"""
Step 6: Finalize after all rounds
"""

# AFTER TRAINING LOOP COMPLETES:
metrics_collector.finalize()

# Save final model checkpoint
checkpoint_dir = os.path.join(args.results_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
metrics_collector.save_final_checkpoint(model, checkpoint_dir)


# ============================================================================
# PART 2: Client-Side Integration (Clean-client2.py)
# ============================================================================

"""
Step 1: Add communication tracking (if you want client-side metrics)
"""

# ADD IMPORT:
from ablation_metrics import CommunicationTracker

# IN CLIENT INITIALIZATION:
comm_tracker = CommunicationTracker()

# WHEN RECEIVING MODEL FROM SERVER:
comm_tracker.track_model_receive(model_parameters)

# WHEN SENDING UPDATE TO SERVER:
comm_tracker.track_model_send(model_parameters)

# Note: Client-side tracking is optional. Server-side tracking is sufficient.


# ============================================================================
# PART 3: Running Experiments with Correct Flags
# ============================================================================

"""
Configuration A: Bucketing_Only
"""
COMMAND_A = """
python PROFILE_server.py \\
    --experiment_name mnist_lenet5_A_Bucketing_Only_label_flip_seed42 \\
    --results_dir ablation_results \\
    --dataset mnist \\
    --num_clients 50 \\
    --num_rounds 50 \\
    --use_bucketing \\
    --malicious_client_ids 0,1,2,3,4,5,6,7,8,9
    # NO --use_dp, NO --use_validators
"""

"""
Configuration B: Bucketing+DP
"""
COMMAND_B = """
python PROFILE_server.py \\
    --experiment_name mnist_lenet5_B_Bucketing_DP_label_flip_seed42 \\
    --results_dir ablation_results \\
    --dataset mnist \\
    --num_clients 50 \\
    --num_rounds 50 \\
    --use_bucketing \\
    --use_dp \\
    --dp_sigma 0.01 \\
    --malicious_client_ids 0,1,2,3,4,5,6,7,8,9
    # NO --use_validators
"""

"""
Configuration C: Bucketing+Validators
"""
COMMAND_C = """
python PROFILE_server.py \\
    --experiment_name mnist_lenet5_C_Bucketing_Validators_label_flip_seed42 \\
    --results_dir ablation_results \\
    --dataset mnist \\
    --num_clients 50 \\
    --num_rounds 50 \\
    --use_bucketing \\
    --use_validators \\
    --validators_per_bucket 5 \\
    --validation_threshold 0.3 \\
    --malicious_client_ids 0,1,2,3,4,5,6,7,8,9
    # NO --use_dp
"""

"""
Configuration D: PROFILE_Full
"""
COMMAND_D = """
python PROFILE_server.py \\
    --experiment_name mnist_lenet5_D_PROFILE_Full_label_flip_seed42 \\
    --results_dir ablation_results \\
    --dataset mnist \\
    --num_clients 50 \\
    --num_rounds 50 \\
    --use_bucketing \\
    --use_dp \\
    --dp_sigma 0.01 \\
    --use_validators \\
    --validators_per_bucket 5 \\
    --validation_threshold 0.3 \\
    --malicious_client_ids 0,1,2,3,4,5,6,7,8,9
"""

"""
Configuration E: FedAvg_Baseline
"""
COMMAND_E = """
python PROFILE_server.py \\
    --experiment_name mnist_lenet5_E_FedAvg_Baseline_label_flip_seed42 \\
    --results_dir ablation_results \\
    --dataset mnist \\
    --num_clients 50 \\
    --num_rounds 50 \\
    --malicious_client_ids 0,1,2,3,4,5,6,7,8,9
    # NO --use_bucketing, NO --use_dp, NO --use_validators
"""

"""
Client Launch (for all configs)
"""
CLIENT_COMMAND = """
# Launch 50 clients (0-49)
# Clients 0-9 are malicious with label_flip attack
# Clients 10-49 are honest

for i in {0..9}; do
    python Clean-client2.py \\
        --client_id $i \\
        --dataset mnist \\
        --num_clients 50 \\
        --seed $((42 + $i)) \\
        --malicious \\
        --attack_type label_flip \\
        --poison_ratio 1.0 &
done

for i in {10..49}; do
    python Clean-client2.py \\
        --client_id $i \\
        --dataset mnist \\
        --num_clients 50 \\
        --seed $((42 + $i)) &
done

wait  # Wait for all clients to finish
"""


# ============================================================================
# PART 4: Complete Example Integration
# ============================================================================

"""
Here's a complete example showing the integration into your existing server loop:
"""

def example_integrated_server_main():
    """
    This is a SKELETON showing where to add ablation metrics.
    Adapt this to your actual PROFILE_server.py structure.
    """
    import argparse
    import numpy as np
    from ablation_metrics import AblationMetricsCollector, CommunicationTracker
    
    # === 1. Parse arguments (ADD ablation flags to your existing parser) ===
    parser = argparse.ArgumentParser()
    # ... your existing args ...
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--results_dir', type=str, default='ablation_results')
    parser.add_argument('--use_bucketing', action='store_true')
    parser.add_argument('--use_dp', action='store_true')
    parser.add_argument('--dp_sigma', type=float, default=0.01)
    parser.add_argument('--use_validators', action='store_true')
    parser.add_argument('--validators_per_bucket', type=int, default=5)
    parser.add_argument('--validation_threshold', type=float, default=0.3)
    parser.add_argument('--malicious_client_ids', type=str, default='0,1,2,3,4,5,6,7,8,9')
    args = parser.parse_args()
    
    # === 2. Initialize metrics collector ===
    malicious_ids = [int(x) for x in args.malicious_client_ids.split(',')]
    
    metrics_collector = AblationMetricsCollector(
        experiment_name=args.experiment_name,
        results_dir=args.results_dir,
        num_clients=args.num_clients,
        num_malicious=len(malicious_ids),
        malicious_client_ids=malicious_ids
    )
    
    comm_tracker = CommunicationTracker()
    
    # === 3. Load data (your existing code) ===
    # X_train, y_train, X_test, y_test = load_data()
    # model = create_model()
    
    # === 4. Training loop ===
    for round_num in range(1, args.num_rounds + 1):
        print(f"\n=== Round {round_num}/{args.num_rounds} ===")
        
        # Start round
        metrics_collector.start_round(round_num)
        comm_tracker.reset_round()
        
        # --- Your existing FL training logic ---
        
        # Sample clients
        # selected_clients = sample_clients(args.num_clients, args.clients_per_round)
        
        # Send model to clients
        # bytes_sent = comm_tracker.track_model_send(get_model_parameters(model))
        
        # Receive updates from clients
        # client_updates = []
        # for client in selected_clients:
        #     update = client.fit(model_parameters)
        #     bytes_received = comm_tracker.track_model_receive(update)
        #     client_updates.append(update)
        
        # Aggregate (with optional bucketing, DP, validators based on flags)
        # if args.use_bucketing:
        #     aggregated = aggregate_with_bucketing(client_updates, ...)
        # else:
        #     aggregated = fedavg(client_updates)
        
        # if args.use_dp:
        #     aggregated = add_dp_noise(aggregated, sigma=args.dp_sigma)
        
        # if args.use_validators:
        #     validation_results = run_validators(...)
        #     detected_buckets = filter_malicious_buckets(validation_results)
        
        # Update model
        # set_model_parameters(model, aggregated)
        
        # --- End of FL training logic ---
        
        # === 5. Evaluate model ===
        # predictions = model.predict(X_test)
        # predictions = np.argmax(predictions, axis=1)
        # test_accuracy = accuracy(predictions, y_test)
        # test_loss = loss(predictions, y_test)
        
        # Calculate model norm
        # global_model_norm = np.sqrt(sum(np.sum(p**2) for p in get_model_parameters(model)))
        
        # Get communication stats
        comm_stats = comm_tracker.get_round_communication()
        
        # === 6. Log metrics ===
        metrics_collector.log_round_metrics(
            round_num=round_num,
            test_accuracy=0.85,  # Replace with actual test_accuracy
            test_loss=0.5,        # Replace with actual test_loss
            predictions=np.random.randint(0, 10, 1000),  # Replace with actual predictions
            true_labels=np.random.randint(0, 10, 1000),  # Replace with actual y_test
            num_classes=10,
            validation_votes=None,  # Add if args.use_validators
            detected_malicious_buckets=None,  # Add if args.use_validators
            bucket_assignments=None,  # Add if args.use_bucketing
            global_model_norm=1.0,  # Replace with actual global_model_norm
            bytes_sent_this_round=comm_stats['bytes_sent'],
            bytes_received_this_round=comm_stats['bytes_received']
        )
    
    # === 7. Finalize ===
    metrics_collector.finalize()
    
    # Save checkpoint
    # checkpoint_dir = os.path.join(args.results_dir, 'checkpoints')
    # os.makedirs(checkpoint_dir, exist_ok=True)
    # metrics_collector.save_final_checkpoint(model, checkpoint_dir)
    
    print("\n✅ Experiment complete!")


# ============================================================================
# PART 5: Testing the Integration
# ============================================================================

"""
Test the integration with a dry run:
"""

DRY_RUN_TEST = """
# 1. Add the integration code to PROFILE_server.py (as shown above)

# 2. Test with a short experiment (5 rounds instead of 50):
python PROFILE_server.py \\
    --experiment_name test_integration \\
    --results_dir test_results \\
    --dataset mnist \\
    --num_clients 10 \\
    --num_rounds 5 \\
    --use_bucketing \\
    --malicious_client_ids 0,1

# 3. Check that metrics file was created:
ls test_results/test_integration.jsonl

# 4. Load and verify metrics:
python3 << 'PYEOF'
from ablation_metrics import load_experiment_metrics
metrics = load_experiment_metrics('test_results/test_integration.jsonl')
print(f"Collected {len(metrics)} rounds of metrics")
print("First round:", metrics[0])
PYEOF

# 5. If successful, proceed with full 30-experiment ablation study
"""

print(__doc__)
print("\n" + "="*80)
print("INTEGRATION CHECKLIST")
print("="*80)
print("""
☐ 1. Add imports to PROFILE_server.py (AblationMetricsCollector, CommunicationTracker)
☐ 2. Add argument parser flags (--experiment_name, --use_bucketing, etc.)
☐ 3. Initialize metrics_collector and comm_tracker in main()
☐ 4. Add configuration logic (bucketing, DP, validators based on flags)
☐ 5. Add comm_tracker.track_model_send() when sending to clients
☐ 6. Add comm_tracker.track_model_receive() when receiving from clients
☐ 7. Add metrics_collector.log_round_metrics() after each round evaluation
☐ 8. Add metrics_collector.finalize() after training completes
☐ 9. Test with short dry run (5 rounds, 10 clients)
☐ 10. Verify metrics JSONL file is created and contains expected fields
☐ 11. Run full ablation study (30 experiments × 50 rounds)
☐ 12. Analyze results with plot_ablation_results.py

Ready to integrate? Start with step 1!
""")
print("="*80 + "\n")
