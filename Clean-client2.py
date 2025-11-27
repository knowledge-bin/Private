import importlib
import math
import os
import sys
import time
import warnings
from typing import List, Tuple, Dict, Any, Optional
import argparse
import json
import datetime
import csv  # Now correctly imported
from detect import *
# Third Party Imports
import flwr as fl
import tensorflow as tf
from memory_profiler import memory_usage
from rlwe_xmkckks import RLWE, Rq
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

# Import the enhanced metrics system
from improved_metrics_system import ExperimentMetricsManager, create_client_metrics_manager
# Import FederatedDataLoader
from federated_data_loader import FederatedDataLoader, FederatedPoisoningExperiment

# Get absolute paths to let a user run the script from anywhere
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.basename(current_directory)
working_directory = os.getcwd()
# Add parent directory to Python's module search path
sys.path.append(os.path.join(current_directory, '..'))
# Compare paths
if current_directory == working_directory:
    from cnn import CNN
    import utils
else:
    # Add current directory to Python's module search path
    CNN = importlib.import_module(f"{parent_directory}.cnn").CNN
    import utils


if __name__ == "__main__":
    # Parse command line arguments for client configuration
    parser = argparse.ArgumentParser(description='PROFILE Client')
    
    parser.add_argument('--client_id', type=int, default=0, help='Client ID')
    parser.add_argument('--server_address', type=str, default='0.0.0.0:8081', help='Server address')
    parser.add_argument('--malicious', action='store_true', help='Run as malicious client')
    parser.add_argument('--attack_type', type=str, default='label_flip', 
                    choices=['label_flip', 'targeted', 'random', 'backdoor', 'fang', 'min_max'], 
                    help='Type of attack if malicious')
    parser.add_argument('--poison_ratio', type=float, default=0.5, help='Ratio of data to poison')
    parser.add_argument('--target_class', type=int, default=7, help='Target class for attacks')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset', type=str, default='mnist', 
                       choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100'], 
                       help='Dataset to use')
    parser.add_argument('--num_clients', type=int, default=15,
                       help='Total number of clients in federated learning')
    # ADD THESE ARGUMENTS TO MATCH SERVER CONFIGURATION
    parser.add_argument('--num_buckets', type=int, default=2,
                       help='Number of buckets for client grouping')
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Privacy budget per round')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Custom experiment name')
    args = parser.parse_args()

    memory_usage_start = memory_usage()[0]

    # CREATE METRICS MANAGER BEFORE EVERYTHING ELSE
    metrics_manager = create_client_metrics_manager(args, args.client_id)
    print(f"🔬 Client {args.client_id} using experiment: {metrics_manager.experiment_name}")
    print(f"📁 Metrics directory: {metrics_manager.experiment_dir}")

    # Revised RLWE Parameter Calculation
    WEIGHT_DECIMALS = 4
    model = CNN.create_for_dataset(args.dataset, WEIGHT_DECIMALS)
    utils.set_initial_params(model)
    params, _ = utils.get_flat_weights(model)
    print("Initial parameters", params[0:20])

    # Step 1: Calculate ring dimension (keep your current approach)
    num_weights = len(params)
    n = 2 ** math.ceil(math.log2(num_weights))
    print(f"n: {n}")

    # CHANGE THIS LINE: Use args.num_clients instead of hardcoded 6
    num_clients = args.num_clients  # CHANGED FROM: num_clients = 6
    
    # Step 2: Choose plaintext modulus t based on our testing
    max_weight_value = 10 ** WEIGHT_DECIMALS
    t = utils.next_prime(num_clients * max_weight_value * 4)
    print(f"t: {t}")

    # Step 3: Choose ciphertext modulus q with optimal ratio to t
    q = utils.next_prime(t * num_clients * 100)
    print(f"q: {q}")

    # Step 4: Set noise standard deviation
    std = 4
    print(f"std: {std}")

    # Initialize RLWE with these parameters
    rlwe = RLWE(n, q, t, std)     

    # Initialize data loader
    data_loader = FederatedDataLoader(
        dataset_name=args.dataset,
        num_clients=num_clients,  # CHANGED FROM hardcoded 6
        iid=True,
        alpha=0.5,
        partition_method='dirichlet',
        seed=args.seed
    )
    
    # Partition data
    data_loader.partition_data()
    
    # Configure malicious client IDs
    if args.malicious:
        malicious_client_ids = [args.client_id]
    else:
        malicious_client_ids = []
    
    # Initialize poisoning experiment
    experiment = FederatedPoisoningExperiment(
        num_clients=num_clients,  # CHANGED FROM hardcoded 6
        malicious_client_ids=malicious_client_ids,
        attack_config={
            'type': args.attack_type,
            'poison_ratio': args.poison_ratio,
            'target_class': args.target_class
        },
        start_round=10,
        end_round=20
    )
    
    class CnnClient(fl.client.NumPyClient):
        def __init__(
            self, 
            rlwe_instance: RLWE, 
            data_loader: FederatedDataLoader,
            experiment: FederatedPoisoningExperiment,
            client_id: int,
            WEIGHT_DECIMALS: int,
            metrics_manager: ExperimentMetricsManager,  # ADD THIS
            *args, 
            **kwargs
        ):
            super().__init__(*args, **kwargs)
            self.rlwe = rlwe_instance
            self.data_loader = data_loader
            self.experiment = experiment
            self.client_id = client_id
            self.allpub = None
            self.model_shape = None
            self.WEIGHT_DECIMALS = WEIGHT_DECIMALS
            self.current_round = 0
            self.metrics_manager = metrics_manager  # STORE THE METRICS MANAGER

            # Initialize the model based on the dataset from data_loader
            self.model = CNN.create_for_dataset(
                data_loader.dataset_name, 
                weight_decimals=WEIGHT_DECIMALS
            )
            # Use the model's set_initial_params() method
            self.model.set_initial_params()
            
            # GET THE MODEL LENGTH FIRST by simulating what happens in encrypt_parameters
            # Update to use the model's get_weights() method
            params = self.model.get_weights()
            # Flatten the weights
            flat_params = []
            for weight in params:
                # Use TensorFlow's reshape to flatten (same as server)
                weight_flat = tf.reshape(weight, [-1])
                flat_params.extend(weight_flat.numpy().tolist())
            padded_params, self.model_length = utils.pad_to_power_of_2(flat_params, self.rlwe.n, self.WEIGHT_DECIMALS)
                
            # Now initialize other attributes
            self.flat_params = None
            
            # KEEP OLD METRICS FOR BACKWARD COMPATIBILITY
            self.training_metrics = []
            self.attack_metrics = []
            self.bucket_evaluation_metrics = []
                      
            # REMOVE OLD SESSION/METRICS DIRECTORY CODE - REPLACED BY METRICS MANAGER
            # Create unique session identifier
            self.session_id = f"session_{int(time.time())}_{client_id}"
            self.start_time = time.time()
            
            print(f"📁 Enhanced metrics will be saved to: {metrics_manager.experiment_dir}")            
            
            # Load client data
            (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = \
                self.data_loader.get_client_data(client_id)
                
            # Store original data for restoring
            self.original_X_train = self.X_train.copy()
            self.original_y_train = self.y_train.copy()
            
            print(f"Client {client_id} initialized with {len(self.X_train)} training samples")
            print(f"Client {client_id} {'is' if client_id in self.experiment.malicious_client_ids else 'is not'} malicious")

            # Initialize the bucket detector
            self.bucket_detector = AdvancedBucketDetector(
                n_features=self.model_length,
                sample_size=1000,
                memory_size=50,
                sensitivity=1.0,         # Reduced from 3.0 to make detection more sensitive
                vote_threshold=0.25,     # Reduced from 0.5 to require fewer flags
                pca_components=5,
                warmup_rounds=1          # Start detection much earlier
            )
            self.accepted_bucket_samples = []

            # ENHANCED: Save client initialization metrics using new system
            init_metrics = {
                'client_id': client_id,
                'is_malicious': client_id in self.experiment.malicious_client_ids,
                'training_samples': len(self.X_train),
                'validation_samples': len(self.X_val),
                'test_samples': len(self.X_test),
                'dataset': data_loader.dataset_name,
                'model_length': self.model_length,
                'attack_config': self.experiment.get_attack_config() if client_id in self.experiment.malicious_client_ids else None,
                'initialization_time': time.time(),
                'bucket_detector_config': {
                    'sensitivity': self.bucket_detector.sensitivity,
                    'vote_threshold': self.bucket_detector.vote_threshold,
                    'warmup_rounds': self.bucket_detector.warmup_rounds
                }
            }
            
            # USE NEW METRICS SYSTEM
            self.metrics_manager.save_client_metrics(client_id, init_metrics, "initialization")
            
            print(f"✅ Client {client_id} initialized with enhanced metrics system")

        def get_parameters(self, config):
            return utils.get_model_parameters(self.model)

        def fit(self, parameters, config):
            start_fit_time = time.time()
            
            # Update round counter
            self.current_round = config.get("server_round", 0)
            print(f"\n{'='*60}")
            print(f"🔄 Client {self.client_id} - Starting Round {self.current_round}")
            print(f"{'='*60}")
            
            # Check if poisoning is active
            is_poisoning = self.experiment.should_poison(self.client_id, self.current_round)
            
            # ENHANCED: Record attack attempt metrics using new system
            if is_poisoning:
                attack_config = self.experiment.get_attack_config()
                attack_metrics = {
                    'round': self.current_round,
                    'client_id': self.client_id,
                    'attack_type': attack_config.get('type', 'unknown'),
                    'poison_ratio': attack_config.get('poison_ratio', 0),
                    'target_class': attack_config.get('target_class', None),
                    'attack_start_time': time.time()
                }
                
                print(f"Client {self.client_id} - Applying {attack_config.get('type')} attack")
                
                # Apply poisoning attack
                poisoning_config = attack_config.copy()
                if 'type' in poisoning_config:
                    poisoning_config['attack_type'] = poisoning_config.pop('type')
                
                X_train_poisoned, y_train_poisoned = self.data_loader.apply_poisoning(
                    self.original_X_train, 
                    self.original_y_train, 
                    **poisoning_config
                )
                
                # Calculate attack statistics
                original_labels = np.unique(self.original_y_train, return_counts=True)
                poisoned_labels = np.unique(y_train_poisoned, return_counts=True)
                
                attack_metrics.update({
                    'original_label_distribution': dict(zip(original_labels[0].tolist(), original_labels[1].tolist())),
                    'poisoned_label_distribution': dict(zip(poisoned_labels[0].tolist(), poisoned_labels[1].tolist())),
                    'samples_modified': int(np.sum(self.original_y_train != y_train_poisoned)),
                    'attack_preparation_time': time.time() - attack_metrics['attack_start_time']
                })
                
                # USE NEW METRICS SYSTEM
                self.metrics_manager.save_client_metrics(self.client_id, attack_metrics, "attack")
                # KEEP OLD SYSTEM FOR BACKWARD COMPATIBILITY
                self.attack_metrics.append(attack_metrics)
                
                # Train with poisoned data
                print(f"🔴 Client {self.client_id}: Training with POISONED data ({len(X_train_poisoned)} samples)...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    training_history = self.model.fit(X_train_poisoned, y_train_poisoned, 
                                                    self.X_val, self.y_val, epochs=1)
                print(f"🔴 Client {self.client_id}: Poisoned training complete")
            else:
                # Train with original data
                print(f"🟢 Client {self.client_id}: Training with CLEAN data ({len(self.original_X_train)} samples)...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    training_history = self.model.fit(self.original_X_train, self.original_y_train, 
                                                    self.X_val, self.y_val, epochs=1)
                print(f"🟢 Client {self.client_id}: Clean training complete")

            # ENHANCED: Process and save comprehensive training metrics
            if hasattr(training_history, 'history'):
                history_dict = training_history.history
            else:
                history_dict = training_history
            
            def get_metric_value(metric_dict, key):
                if key in metric_dict:
                    value = metric_dict[key]
                    if isinstance(value, list):
                        return value[0] if len(value) > 0 else None
                    else:
                        return value
                return None
            
            training_metrics = {
                'round': self.current_round,
                'client_id': self.client_id,
                'is_malicious': is_poisoning,
                'training_time': time.time() - start_fit_time,
                'training_loss': get_metric_value(history_dict, 'loss'),
                'training_accuracy': get_metric_value(history_dict, 'accuracy'),
                'val_loss': get_metric_value(history_dict, 'val_loss'),
                'val_accuracy': get_metric_value(history_dict, 'val_accuracy'),
                'data_samples_used': len(X_train_poisoned) if is_poisoning else len(self.original_X_train),
                'model_parameters_count': self.model_length,
                'training_epochs': 1,
                'timestamp': time.time()
            }
            
            # USE NEW METRICS SYSTEM
            self.metrics_manager.save_client_metrics(self.client_id, training_metrics, "training")
            # KEEP OLD SYSTEM FOR BACKWARD COMPATIBILITY
            self.training_metrics.append(training_metrics)
            
            # Format metrics safely
            loss_str = f"{training_metrics.get('training_loss', 0):.4f}" if training_metrics.get('training_loss') is not None else 'N/A'
            acc_str = f"{training_metrics.get('training_accuracy', 0):.4f}" if training_metrics.get('training_accuracy') is not None else 'N/A'
            print(f"✅ Client {self.client_id} Round {self.current_round} complete - "
                  f"Loss: {loss_str}, Acc: {acc_str}, Time: {training_metrics['training_time']:.2f}s")
            print("=" * 60 + "\n")
            
            return [], len(self.X_train), {}

        def evaluate(self, parameters, config):
            """Evaluate parameters using local test data or do bucket-level poisoning checks."""
            start_eval_time = time.time()
            print(f"📊 Client {self.client_id}: Starting evaluation...")

            # --- Bucket evaluation path ---
            if config and "bucket_evaluation" in config:
                bucket_id = config.get("bucket_id", -1)
                print(f"Received bucket evaluation request for bucket {bucket_id}")

                try:
                    import json, numpy as np

                    # Deserialize the sample
                    sample = np.array(json.loads(config["bucket_model_sample"]), dtype=float)
                    
                    # Ensure sample has the expected size
                    if len(sample) != 1000:
                        print(f"Warning: Expected sample size 1000, got {len(sample)}")

                    # Analyze with our detector
                    S, flags, metrics = self.bucket_detector.analyze(
                        x=sample,
                        other_clean=self.accepted_bucket_samples
                    )

                    # If deemed clean, add to accepted list for spatial checks
                    if S <= self.bucket_detector.vote_threshold:
                        self.accepted_bucket_samples.append(sample)

                    # Record detailed bucket metrics
                    bucket_metrics = {
                        'round': self.current_round,
                        'client_id': self.client_id,
                        'bucket_id': bucket_id,
                        'suspicion_score': float(S),  # Ensure JSON serializable
                        'flags': {k: bool(v) for k, v in flags.items()},  # Convert to bool
                        'metrics': {k: float(v) for k, v in metrics.items()},  # Convert to float
                        'thresholds': {
                            k: float(self.bucket_detector._dynamic_threshold(k))
                            for k in ['mad95','phi_t','phi_s','eps']
                        },
                        'evaluation_time': time.time() - start_eval_time,
                        'timestamp': time.time(),
                        'is_malicious': self.client_id in self.experiment.malicious_client_ids,
                        'verdict': 'Clean' if S <= self.bucket_detector.vote_threshold else 'Poisoned'
                    }

                    # USE NEW METRICS SYSTEM
                    self.metrics_manager.save_client_metrics(self.client_id, bucket_metrics, "bucket_evaluation")
                    # KEEP OLD SYSTEM FOR BACKWARD COMPATIBILITY
                    self.bucket_evaluation_metrics.append(bucket_metrics)

                    print(f"Bucket {bucket_id} verdict: {'Clean' if S<=self.bucket_detector.vote_threshold else 'Poisoned'} (score={S:.3f})")
                    print(f"Detection flags: {flags}")
                    print(f"Metrics: {metrics}")
                    
                    return 0.0, 1, {"bucket_verdict": float(1.0 - S)}
                except Exception as e:
                    import traceback
                    print(f"Error during bucket evaluation: {e}")
                    traceback.print_exc()
                    return 0.0, 1, {"bucket_verdict": 1.0}

            # --- Standard evaluation path ---
            utils.set_model_params(self.model, parameters)
            loss, accuracy = self.model.evaluate(self.X_test, self.y_test)

            eval_metrics = {
                'round': self.current_round,
                'client_id': self.client_id,
                'test_loss': float(loss),  # Ensure JSON serializable
                'test_accuracy': float(accuracy),
                'evaluation_time': time.time() - start_eval_time,
                'timestamp': time.time()
            }
            
            # USE NEW METRICS SYSTEM
            self.metrics_manager.save_client_metrics(self.client_id, eval_metrics, "evaluation")
            # KEEP OLD SYSTEM FOR BACKWARD COMPATIBILITY
            self.training_metrics.append(eval_metrics)

            return loss, len(self.X_test), {"accuracy": accuracy}

        # KEEP ALL YOUR EXISTING METHODS (analyze_bucket_sample, measure_gradient_variance, etc.)
        def analyze_bucket_sample(self, bucket_model_sample, bucket_id=-1):
            """
            Analyze a sample of bucket model parameters to detect potential poisoning.
            This method doesn't require reshaping the parameters into model weights.
            """
            print(f"Analyzing parameter sample for bucket {bucket_id}...")
            
            try:
                # Get sample statistics
                import numpy as np
                sample = np.array(bucket_model_sample)
                
                # Calculate basic statistics
                mean = np.mean(sample)
                std = np.std(sample)
                min_val = np.min(sample)
                max_val = np.max(sample)
                median = np.median(sample)
                
                # Print statistics
                print(f"Sample statistics for bucket {bucket_id}:")
                print(f"  Mean: {mean:.4f}")
                print(f"  Std Dev: {std:.4f}")
                print(f"  Min: {min_val:.4f}")
                print(f"  Max: {max_val:.4f}")
                print(f"  Median: {median:.4f}")
                
                # Simple anomaly detection based on statistics
                suspicion_score = 0.0
                
                # 1. Check for extreme values
                max_abs_val = max(abs(min_val), abs(max_val))
                if max_abs_val > 1000:
                    print(f"Warning: Extreme values detected in bucket {bucket_id}")
                    suspicion_score = 0.7
                    
                # 2. Check for unusual distribution
                if std > 0:
                    skewness = np.mean(((sample - mean) / std) ** 3)
                    kurtosis = np.mean(((sample - mean) / std) ** 4) - 3
                    
                    print(f"  Skewness: {skewness:.4f}")
                    print(f"  Kurtosis: {kurtosis:.4f}")
                    
                    # High absolute skewness or kurtosis can indicate unusual distribution
                    if abs(skewness) > 2 or abs(kurtosis) > 5:
                        print(f"Warning: Unusual distribution detected in bucket {bucket_id}")
                        suspicion_score += 0.3
                
                # 3. Check for other anomalies
                zero_ratio = np.sum(sample == 0) / len(sample)
                if zero_ratio > 0.5:
                    print(f"Warning: High number of zeros ({zero_ratio:.2%}) in bucket {bucket_id}")
                    suspicion_score += 0.3
                
                # Calculate final verdict (higher is cleaner)
                verdict = max(0, min(1, 1.0 - suspicion_score))
                
                print(f"Verdict for bucket {bucket_id}: {verdict:.2f} (higher is cleaner)")
                return verdict
                
            except Exception as e:
                print(f"Error analyzing bucket sample: {str(e)}")
                return 1.0  # Default to clean in case of errors

        def measure_gradient_variance(self, client_updates, bucket_assignments):
            """Measure variance metrics for reviewer response"""
            import numpy as np
            
            # Global pairwise variance (random partitioning baseline)
            global_variance = 0.0
            count = 0
            for i in range(len(client_updates)):
                for j in range(i+1, len(client_updates)):
                    diff = np.linalg.norm(client_updates[i] - client_updates[j])**2
                    global_variance += diff
                    count += 1
            sigma_global_sq = global_variance / count if count > 0 else 0.0
            
            # Within-bucket maximum pairwise variance
            rho_sq = 0.0
            for bucket in bucket_assignments:
                if len(bucket) > 1:
                    bucket_updates = [client_updates[i] for i in bucket]
                    for i in range(len(bucket_updates)):
                        for j in range(i+1, len(bucket_updates)):
                            diff = np.linalg.norm(bucket_updates[i] - bucket_updates[j])**2
                            rho_sq = max(rho_sq, diff)
            
            # Clustering effectiveness
            alpha = rho_sq / sigma_global_sq if sigma_global_sq > 0 else 0.0
            
            variance_metrics = {
                'round': self.current_round,
                'client_id': self.client_id,
                'sigma_global_sq': float(sigma_global_sq),
                'rho_sq': float(rho_sq),
                'alpha': float(alpha),
                'timestamp': time.time()
            }
            
            # USE NEW METRICS SYSTEM
            self.metrics_manager.save_client_metrics(self.client_id, variance_metrics, "variance_analysis")
            
            return variance_metrics

        ##############################################################################################################
        #  Below steps are involved in the implementation of multi-key homomorphic encryption for federated learning #
        ##############################################################################################################

        def example_response(self, question: str, l: List[int]) -> Tuple[str, int]:
            response = "Here you go Alice!"
            answer = sum(l)
            return response, answer

        # Step 1) Server sends shared vector_a to clients and they all send back vector_b
        def generate_pubkey(self, vector_a: List[int]) -> List[int]:
            vector_a = self.rlwe.list_to_poly(vector_a, "q")
            self.rlwe.set_vector_a(vector_a)
            (_, pub) = rlwe.generate_keys()
            print(f"client pub: {pub}")
            return pub[0].poly_to_list()
        
        # Step 2) Server sends aggregated publickey allpub to clients and receive boolean confirmation
        def store_aggregated_pubkey(self, allpub: List[int]) -> bool:
            aggregated_pubkey = self.rlwe.list_to_poly(allpub, "q")
            self.allpub = (aggregated_pubkey, self.rlwe.get_vector_a())
            print(f"client allpub: {self.allpub}")
            return True

        def encrypt_parameters(self, request) -> Tuple[List[int], List[int]]:
            print(f"request msg is: {request}")
            
            # Get nested model parameters and turn into long list
            flattened_weights, self.model_shape = utils.get_flat_weights(self.model)

            # Pad list until length 2**20 with random numbers that mimic the weights
            flattened_weights, self.model_length = utils.pad_to_power_of_2(flattened_weights, self.rlwe.n, self.WEIGHT_DECIMALS)
            print(f"Client old plaintext: {self.flat_params[925:935]}") if self.flat_params is not None else None
            print(f"Client new plaintext: {flattened_weights[925:935]}")
            # Turn list into polynomial
            poly_weights = Rq(np.array(flattened_weights), self.rlwe.t)
            # print(f"Client plainpoly: {poly_weights}")

            # get gradient instead of full weights
            if request == "gradient":
                gradient = list(np.array(flattened_weights) - np.array(self.flat_params))
                print(f"Client gradient: {gradient[925:935]}")
                poly_weights = Rq(np.array(gradient), self.rlwe.t)

            # Encrypt the polynomial
            c0, c1 = self.rlwe.encrypt(poly_weights, self.allpub)
            c0 = list(c0.poly.coeffs)
            c1 = list(c1.poly.coeffs)
            print(f"c0: {c0[:10]}")
            print(f"c1: {c1[:10]}")
            return c0, c1
        # def encrypt_parameters(self, request) -> Tuple[List[int], List[int]]:
        #     print(f"request msg is: {request}")
            
        #     # Get nested model parameters and turn into long list
        #     flattened_weights, self.model_shape = utils.get_flat_weights(self.model)

        #     # Pad list until length 2**20 with random numbers that mimic the weights
        #     flattened_weights, self.model_length = utils.pad_to_power_of_2(flattened_weights, self.rlwe.n, self.WEIGHT_DECIMALS)
        #     print(f"Client old plaintext: {self.flat_params[925:935]}") if self.flat_params is not None else None
        #     print(f"Client new plaintext: {flattened_weights[925:935]}")
            
        #     # Turn list into polynomial
        #     poly_weights = Rq(np.array(flattened_weights), self.rlwe.t)

        #     # get gradient instead of full weights
        #     if request == "gradient":
        #         # FIX: Check if flat_params is None before calculating gradient
        #         if self.flat_params is None:
        #             print("Warning: flat_params is None, using zero gradient")
        #             gradient = [0.0] * len(flattened_weights)  # Use zero gradient as fallback
        #         else:
        #             gradient = list(np.array(flattened_weights) - np.array(self.flat_params))
                
        #         print(f"Client gradient: {gradient[925:935]}")
        #         poly_weights = Rq(np.array(gradient), self.rlwe.t)

        #     # Encrypt the polynomial
        #     c0, c1 = self.rlwe.encrypt(poly_weights, self.allpub)
        #     c0 = list(c0.poly.coeffs)
        #     c1 = list(c1.poly.coeffs)
        #     print(f"c0: {c0[:10]}")
        #     print(f"c1: {c1[:10]}")
        #     return c0, c1

        # Step 4) Use csum1 to calculate partial decryption share di
        def compute_decryption_share(self, csum1) -> List[int]:
            std = 5
            csum1_poly = self.rlwe.list_to_poly(csum1, "q")
            error = Rq(np.round(std * np.random.randn(n)), q)
            d1 = self.rlwe.decrypt(csum1_poly, self.rlwe.s, error)
            d1 = list(d1.poly.coeffs) #d1 is poly_t not poly_q
            return d1

        def receive_updated_weights(self, server_flat_weights) -> bool:
            # update temporal reference for bucket detector
            import numpy as np
            # Create a sample from the global model for temporal comparison
            if server_flat_weights is not None and len(server_flat_weights) >= 1000:
                global_sample = np.array(server_flat_weights[:1000], dtype=float)
                self.bucket_detector.last_global = global_sample           
            start_update_time = time.time()
            # Convert list of python integers into list of np.float64
            server_flat_weights = list(np.array(server_flat_weights, dtype=np.float64))
            
            if self.flat_params is None:
                # first round (server gives full weights)
                self.flat_params = server_flat_weights
            else:
                # next rounds (server gives only gradient)
                self.flat_params = list(np.array(self.flat_params) + np.array(server_flat_weights))
            
            # Remove padding and return weights to original tensor structure and set model weights
            server_flat_weights = utils.remove_padding(self.flat_params, self.model_length)
            
            # Restore the long list of weights into the neural network's original structure
            weights = []
            index = 0
            for shape in self.model_shape:
                size = np.prod(shape)
                weights.append(np.array(server_flat_weights[index:index + size]).reshape(shape))
                index += size
            
            print(f"Fedavg plaintext: {server_flat_weights[925:935]}")

            utils.set_model_params(self.model, weights)
            y_pred = self.model.model.predict(self.X_test)
            
            # Check for NaN values in predictions
            has_nan = np.isnan(y_pred).any()
            if has_nan:
                print("WARNING: NaN values detected in predictions. Using fallback metrics calculation.")
                
            # Handle predictions with NaNs for classification metrics
            if has_nan:
                # Replace NaN values with 0 for argmax operation
                y_pred_clean = np.nan_to_num(y_pred, nan=0.0)
                predicted = np.argmax(y_pred_clean, axis=-1)
            else:
                predicted = np.argmax(y_pred, axis=-1)
                




            # Calculate

            # Calculate accuracy - this should work even with some NaNs
            accuracy = np.equal(self.y_test, predicted).mean()
            
            # Try to calculate log_loss, but catch errors
            try:
                loss = log_loss(self.y_test, y_pred)
            except (ValueError, AssertionError):
                print("WARNING: Couldn't calculate log_loss due to NaN values. Using 999 as placeholder.")
                loss = 999.0  # Use a placeholder value
            
            # Try to calculate other metrics with error handling
            try:
                precision = precision_score(self.y_test, predicted, average='weighted')
                recall = recall_score(self.y_test, predicted, average='weighted')
                f1_score_ = f1_score(self.y_test, predicted, average='weighted')
                # Use the correct number of classes based on dataset
                num_classes = self.model.num_classes
                confusion_matrix_ = confusion_matrix(self.y_test, predicted)
            except (ValueError, AssertionError) as e:
                print(f"WARNING: Metrics calculation error: {e}")
                precision = recall = f1_score_ = 0.0
                confusion_matrix_ = np.zeros((num_classes, num_classes))
            
            # Also handle NaNs in validation predictions
            y_pred_val = self.model.model.predict(self.X_val)
            
            if np.isnan(y_pred_val).any():
                y_pred_val_clean = np.nan_to_num(y_pred_val, nan=0.0)
                predicted_val = np.argmax(y_pred_val_clean, axis=-1)
            else:
                predicted_val = np.argmax(y_pred_val, axis=-1)
                
            val_accuracy = np.equal(self.y_val, predicted_val).mean()

            print()
            print(f"\nLen(X_test): {len(self.X_test)}")
            print(f"Accuracy: {accuracy}")
            print(f"Val Accuracy: {val_accuracy}")
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1-Score:", f1_score_)
            print(f"Loss: {loss}")
            print("\nConfusion matrix")
            print(confusion_matrix_)
            print()
            
            # If we encountered NaN values in predictions, print model summary for debugging
            if has_nan:
                print("MODEL WEIGHTS SUMMARY (due to NaN detection):")
                for i, weight in enumerate(weights):
                    print(f"Layer {i}: shape={weight.shape}, contains_nan={np.isnan(weight).any()}, min={np.nanmin(weight)}, max={np.nanmax(weight)}")
                        
            # ENHANCED: Save comprehensive global model metrics using new system
            global_model_metrics = {
                'round': self.current_round,
                'client_id': self.client_id,
                'test_accuracy': float(accuracy),
                'test_loss': float(loss) if loss != 999.0 else None,
                'test_precision': float(precision),
                'test_recall': float(recall),
                'test_f1': float(f1_score_),
                'val_accuracy': float(val_accuracy),
                'update_time': time.time() - start_update_time,
                'timestamp': time.time(),
                'has_nan': bool(has_nan),
                'confusion_matrix': confusion_matrix_.tolist(),  # Convert to list for JSON serialization
                'model_weights_summary': {
                    'total_layers': len(weights),
                    'total_parameters': sum(np.prod(w.shape) for w in weights),
                    'layers_with_nan': sum(1 for w in weights if np.isnan(w).any()),
                    'weight_statistics': [
                        {
                            'layer': i,
                            'shape': w.shape,
                            'mean': float(np.nanmean(w)),
                            'std': float(np.nanstd(w)),
                            'min': float(np.nanmin(w)),
                            'max': float(np.nanmax(w)),
                            'has_nan': bool(np.isnan(w).any())
                        } for i, w in enumerate(weights)
                    ]
                }
            }
            
            # USE NEW METRICS SYSTEM
            self.metrics_manager.save_client_metrics(self.client_id, global_model_metrics, "global_model")
            # KEEP OLD SYSTEM FOR BACKWARD COMPATIBILITY
            self.training_metrics.append(global_model_metrics)
            
            # ENHANCED: Save detailed per-class metrics if possible
            if not has_nan:
                try:
                    class_report = classification_report(self.y_test, predicted, output_dict=True)
                    per_class_metrics = {
                        'round': self.current_round,
                        'client_id': self.client_id,
                        'timestamp': time.time(),
                        'classification_report': class_report,
                        'per_class_accuracy': {
                            str(class_id): float(metrics['precision'])
                            for class_id, metrics in class_report.items()
                            if class_id.isdigit()
                        },
                        'macro_avg': {
                            'precision': float(class_report['macro avg']['precision']),
                            'recall': float(class_report['macro avg']['recall']),
                            'f1_score': float(class_report['macro avg']['f1-score'])
                        },
                        'weighted_avg': {
                            'precision': float(class_report['weighted avg']['precision']),
                            'recall': float(class_report['weighted avg']['recall']),
                            'f1_score': float(class_report['weighted avg']['f1-score'])
                        }
                    }
                    
                    # USE NEW METRICS SYSTEM
                    self.metrics_manager.save_client_metrics(self.client_id, per_class_metrics, "per_class")
                    
                except Exception as e:
                    print(f"Warning: Could not generate detailed per-class metrics: {e}")
            
            return True

        # ENHANCED: Backward compatible save methods with new system integration
        def save_metrics_to_file(self, metrics, filename_prefix="metrics"):
            """Save evaluation metrics to file for later analysis and plotting"""
            try:
                # Add client and session information
                metrics["client_id"] = self.client_id
                metrics["session_id"] = self.session_id
                
                # Convert all numpy types to native Python types
                metrics = self._convert_numpy_types(metrics)
                
                # ENHANCED: Use new metrics system based on filename prefix
                if filename_prefix == "training_metrics":
                    self.metrics_manager.save_client_metrics(self.client_id, metrics, "training")
                elif filename_prefix == "bucket_evaluation_metrics":
                    self.metrics_manager.save_client_metrics(self.client_id, metrics, "bucket_evaluation")
                elif filename_prefix == "evaluation_metrics":
                    self.metrics_manager.save_client_metrics(self.client_id, metrics, "evaluation")
                elif filename_prefix == "global_model_metrics":
                    self.metrics_manager.save_client_metrics(self.client_id, metrics, "global_model")
                elif filename_prefix == "per_class_metrics":
                    self.metrics_manager.save_client_metrics(self.client_id, metrics, "per_class")
                elif filename_prefix == "variance_analysis":
                    self.metrics_manager.save_client_metrics(self.client_id, metrics, "variance_analysis")
                else:
                    self.metrics_manager.save_client_metrics(self.client_id, metrics, "general")
                
                # KEEP OLD SYSTEM FOR BACKWARD COMPATIBILITY
                # Save to a file specific to this metric type
                json_file = f"{self.metrics_manager.experiment_dir}/raw_data/{filename_prefix}_{self.client_id}.jsonl"
                
                # Append to JSONL file (JSON Lines format for better parsing)
                with open(json_file, 'a') as f:
                    f.write(json.dumps(metrics) + '\n')
                
                # Also save to a consolidated CSV file for easy analysis
                csv_file = f"{self.metrics_manager.experiment_dir}/raw_data/{filename_prefix}_all.csv"
                file_exists = os.path.isfile(csv_file)
                
                with open(csv_file, 'a', newline='') as f:
                    # Flatten nested structures for CSV
                    flat_metrics = self.flatten_dict(metrics)
                    writer = csv.DictWriter(f, fieldnames=sorted(flat_metrics.keys()))
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(flat_metrics)
                
                print(f"✅ Enhanced metrics saved: {filename_prefix}")
                
            except Exception as e:
                print(f"❌ Error saving metrics to file: {str(e)}")

        def _convert_numpy_types(self, obj):
            """Convert numpy types to native Python types"""
            import numpy as np
            if isinstance(obj, dict):
                return {k: self._convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj

        def flatten_dict(self, d, parent_key='', sep='_'):
            """Flatten nested dictionary for CSV storage"""
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(self.flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
                    # Handle nested lists (like confusion matrix)
                    items.append((new_key, str(v)))
                else:
                    items.append((new_key, v))
            return dict(items)

        def save_final_summary(self):
            """ENHANCED: Save a comprehensive summary using new metrics system"""
            try:
                # Calculate comprehensive statistics
                total_training_rounds = len([m for m in self.training_metrics if 'training_loss' in m])
                total_attacks = len(self.attack_metrics)
                total_bucket_evaluations = len(self.bucket_evaluation_metrics)
                
                # Calculate average metrics
                accuracies = [m.get('test_accuracy', 0) for m in self.training_metrics if 'test_accuracy' in m]
                avg_accuracy = np.mean(accuracies) if accuracies else 0
                
                losses = [m.get('test_loss', 0) for m in self.training_metrics if 'test_loss' in m and m['test_loss'] is not None]
                avg_loss = np.mean(losses) if losses else 0
                
                # Bucket evaluation statistics
                bucket_verdicts = [m.get('verdict', 'Unknown') for m in self.bucket_evaluation_metrics]
                clean_verdicts = bucket_verdicts.count('Clean')
                poisoned_verdicts = bucket_verdicts.count('Poisoned')
                
                summary = {
                    'session_metadata': {
                        'session_id': self.session_id,
                        'client_id': self.client_id,
                        'experiment_name': self.metrics_manager.experiment_name,
                        'start_time': self.start_time,
                        'end_time': time.time(),
                        'total_duration': time.time() - self.start_time
                    },
                    'training_summary': {
                        'total_rounds': total_training_rounds,
                        'average_accuracy': float(avg_accuracy),
                        'average_loss': float(avg_loss),
                        'final_accuracy': float(accuracies[-1]) if accuracies else 0,
                        'accuracy_trend': 'improving' if len(accuracies) > 1 and accuracies[-1] > accuracies[0] else 'declining'
                    },
                    'attack_summary': {
                        'total_attacks_performed': total_attacks,
                        'is_malicious_client': self.client_id in self.experiment.malicious_client_ids,
                        'attack_config': self.experiment.get_attack_config() if self.client_id in self.experiment.malicious_client_ids else None,
                        'attack_rounds': [m['round'] for m in self.attack_metrics] if self.attack_metrics else []
                    },
                    'detection_summary': {
                        'total_bucket_evaluations': total_bucket_evaluations,
                        'clean_verdicts': clean_verdicts,
                        'poisoned_verdicts': poisoned_verdicts,
                        'detection_accuracy': clean_verdicts / total_bucket_evaluations if total_bucket_evaluations > 0 else 0,
                        'average_suspicion_score': float(np.mean([m.get('suspicion_score', 0) for m in self.bucket_evaluation_metrics])) if self.bucket_evaluation_metrics else 0
                    },
                    'model_summary': {
                        'model_length': self.model_length,
                        'dataset': self.data_loader.dataset_name,
                        'training_samples': len(self.original_X_train),
                        'validation_samples': len(self.X_val),
                        'test_samples': len(self.X_test)
                    },
                    'experiment_config': self.metrics_manager.config
                }
                
                # USE NEW METRICS SYSTEM
                self.metrics_manager.save_client_metrics(self.client_id, summary, "session_summary")
                
                # ALSO SAVE TO OLD LOCATION FOR BACKWARD COMPATIBILITY
                summary_file = f"{self.metrics_manager.experiment_dir}/raw_data/session_summary_{self.client_id}.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=4)
                
                print(f"📊 Comprehensive session summary saved for client {self.client_id}")
                print(f"📈 Final stats: Accuracy={avg_accuracy:.4f}, Attacks={total_attacks}, Evaluations={total_bucket_evaluations}")
                
            except Exception as e:
                print(f"❌ Error saving session summary: {str(e)}")

    # Start Flower client with our enhanced CnnClient implementation
    client = None
    try:
        client = CnnClient(
            rlwe,
            data_loader,
            experiment,
            client_id=args.client_id,
            WEIGHT_DECIMALS=WEIGHT_DECIMALS,
            metrics_manager=metrics_manager  # Pass the metrics manager
        )
        
        print(f"🚀 Starting enhanced PROFILE client {args.client_id}")
        print(f"📊 Metrics will be saved to: {metrics_manager.experiment_dir}")
        
        fl.client.start_numpy_client(
            server_address=args.server_address,
            client=client
        )
        
    except Exception as e:
        print(f"❌ Error during client execution: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Stop recording memory usage
        memory_usage_end = memory_usage()[0]
        memory_usage_total = memory_usage_end - memory_usage_start
        print(f"💾 Memory usage: {memory_usage_total:.2f} MiB")        
        
        # Only save final summary if client was created successfully
        if client is not None:
            client.save_final_summary()
            
            # ENHANCED: Save comprehensive memory and performance metrics
            performance_metrics = {
                'client_id': client.client_id,
                'session_id': client.session_id,
                'experiment_name': client.metrics_manager.experiment_name,
                'memory_usage_mb': float(memory_usage_total),
                'total_execution_time': time.time() - client.start_time if hasattr(client, 'start_time') else 0,
                'training_rounds_completed': len(client.training_metrics),
                'attacks_performed': len(client.attack_metrics),
                'bucket_evaluations_completed': len(client.bucket_evaluation_metrics),
                'final_model_accuracy': client.training_metrics[-1].get('test_accuracy', 0) if client.training_metrics else 0,
                'timestamp': time.time(),
                'system_info': {
                    'dataset': args.dataset,
                    'num_clients': args.num_clients,
                    'num_buckets': getattr(args, 'num_buckets', 2),
                    'is_malicious': args.malicious,
                    'attack_type': args.attack_type if args.malicious else 'none'
                }
            }
            
            # USE NEW METRICS SYSTEM
            client.metrics_manager.save_client_metrics(client.client_id, performance_metrics, "performance")
            
            print(f"✅ Enhanced client {args.client_id} completed successfully")
            print(f"📁 All metrics saved to: {client.metrics_manager.experiment_dir}")
            
            # Print experiment summary
            exp_summary = client.metrics_manager.get_experiment_summary()
            print(f"🔬 Experiment: {exp_summary['experiment_name']}")
            if 'files_created' in exp_summary:
                print(f"📊 Files created: {len(sum(exp_summary['files_created'].values(), []))}")
        
        else:
            print("❌ Client was not initialized properly")

