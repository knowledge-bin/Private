#!/usr/bin/env python3
"""
Comprehensive Metrics Collection for PROFILE Ablation Study

Tracks all metrics required by reviewers:
- Per-round test accuracy
- Clean accuracy per class
- Attack success rate (ASR)
- Validation votes (if validators used)
- Detection precision/recall/F1
- Global loss
- Server aggregate norm
- Time elapsed per round
- Communication bytes sent/received
- Memory usage

Saves to JSONL format: results/mnist_lenet5_<config>_<attack>_seed<seed>.jsonl
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import psutil
import os


class AblationMetricsCollector:
    """Collects and saves comprehensive metrics for ablation study"""
    
    def __init__(
        self, 
        experiment_name: str,
        results_dir: str,
        num_clients: int,
        num_malicious: int,
        malicious_client_ids: List[int]
    ):
        """
        Initialize metrics collector
        
        Args:
            experiment_name: Unique experiment identifier
            results_dir: Directory to save results
            num_clients: Total number of clients
            num_malicious: Number of malicious clients
            malicious_client_ids: List of malicious client IDs
        """
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_clients = num_clients
        self.num_malicious = num_malicious
        self.malicious_client_ids = set(malicious_client_ids)
        
        # Metrics file path
        self.metrics_file = self.results_dir / f"{experiment_name}.jsonl"
        
        # Round tracking
        self.current_round = 0
        self.round_start_time = None
        self.experiment_start_time = time.time()
        
        # Communication tracking
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        
        # Process for memory monitoring
        self.process = psutil.Process(os.getpid())
        
        print(f"📊 Metrics Collector initialized for: {experiment_name}")
        print(f"   Saving to: {self.metrics_file}")
        print(f"   Tracking {num_clients} clients ({num_malicious} malicious)")
        
    def start_round(self, round_num: int):
        """Mark the start of a new round"""
        self.current_round = round_num
        self.round_start_time = time.time()
        
    def log_round_metrics(
        self,
        round_num: int,
        test_accuracy: float,
        test_loss: float,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        num_classes: int = 10,
        validation_votes: Optional[Dict[str, Any]] = None,
        detected_malicious_buckets: Optional[List[int]] = None,
        bucket_assignments: Optional[Dict[int, int]] = None,
        global_model_norm: Optional[float] = None,
        bytes_sent_this_round: int = 0,
        bytes_received_this_round: int = 0
    ):
        """
        Log comprehensive metrics for a single round
        
        Args:
            round_num: Current round number
            test_accuracy: Overall test accuracy
            test_loss: Overall test loss
            predictions: Model predictions on test set
            true_labels: Ground truth labels
            num_classes: Number of classes
            validation_votes: Validator votes (if applicable)
            detected_malicious_buckets: Buckets flagged as malicious
            bucket_assignments: Mapping of client_id -> bucket_id
            global_model_norm: L2 norm of global model parameters
            bytes_sent_this_round: Bytes sent in this round
            bytes_received_this_round: Bytes received in this round
        """
        
        # Calculate round duration
        round_duration = time.time() - self.round_start_time if self.round_start_time else 0
        
        # Per-class accuracy
        class_accuracies = self._calculate_class_accuracies(
            predictions, true_labels, num_classes
        )
        
        # Attack success rate
        attack_success_rate = self._calculate_attack_success_rate(
            predictions, true_labels
        )
        
        # Detection metrics (if validators used)
        detection_metrics = None
        if detected_malicious_buckets is not None and bucket_assignments is not None:
            detection_metrics = self._calculate_detection_metrics(
                detected_malicious_buckets,
                bucket_assignments
            )
        
        # Update communication tracking
        self.total_bytes_sent += bytes_sent_this_round
        self.total_bytes_received += bytes_received_this_round
        
        # Memory usage
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        # Construct metrics dictionary
        metrics = {
            'round': round_num,
            'timestamp': time.time(),
            'elapsed_total_seconds': time.time() - self.experiment_start_time,
            'elapsed_round_seconds': round_duration,
            
            # Accuracy metrics
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'class_accuracies': {str(i): float(acc) for i, acc in enumerate(class_accuracies)},
            'mean_class_accuracy': float(np.mean(class_accuracies)),
            
            # Attack metrics
            'attack_success_rate': float(attack_success_rate),
            
            # Model metrics
            'global_model_norm': float(global_model_norm) if global_model_norm else None,
            
            # Validation metrics
            'validation_votes': validation_votes,
            
            # Detection metrics
            'detection_metrics': detection_metrics,
            
            # Communication metrics
            'bytes_sent_this_round': bytes_sent_this_round,
            'bytes_received_this_round': bytes_received_this_round,
            'total_bytes_sent': self.total_bytes_sent,
            'total_bytes_received': self.total_bytes_received,
            
            # Resource metrics
            'memory_mb': float(memory_mb)
        }
        
        # Write to JSONL file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        # Print summary
        print(f"\n📊 Round {round_num} Metrics:")
        print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Attack Success Rate: {attack_success_rate*100:.2f}%")
        if detection_metrics:
            print(f"   Detection F1: {detection_metrics['f1']:.3f}")
        print(f"   Round Time: {round_duration:.2f}s")
        print(f"   Memory: {memory_mb:.1f} MB")
        
        return metrics
    
    def _calculate_class_accuracies(
        self, 
        predictions: np.ndarray, 
        true_labels: np.ndarray,
        num_classes: int
    ) -> List[float]:
        """Calculate per-class accuracy"""
        
        class_accuracies = []
        
        for class_id in range(num_classes):
            # Find samples of this class
            class_mask = (true_labels == class_id)
            
            if np.sum(class_mask) == 0:
                class_accuracies.append(0.0)
                continue
            
            # Calculate accuracy for this class
            class_predictions = predictions[class_mask]
            class_true = true_labels[class_mask]
            
            accuracy = np.mean(class_predictions == class_true)
            class_accuracies.append(accuracy)
        
        return class_accuracies
    
    def _calculate_attack_success_rate(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        target_class: Optional[int] = None
    ) -> float:
        """
        Calculate attack success rate
        
        For label-flip: percentage of misclassifications
        For targeted: percentage classified as target
        """
        
        # For now, we calculate general misclassification rate
        # This represents how much the attack has degraded model performance
        
        misclassifications = np.sum(predictions != true_labels)
        total = len(true_labels)
        
        asr = misclassifications / total if total > 0 else 0.0
        
        return asr
    
    def _calculate_detection_metrics(
        self,
        detected_malicious_buckets: List[int],
        bucket_assignments: Dict[int, int]
    ) -> Dict[str, float]:
        """
        Calculate detection precision, recall, and F1
        
        Args:
            detected_malicious_buckets: List of bucket IDs flagged as malicious
            bucket_assignments: Mapping of client_id -> bucket_id
        """
        
        # Determine ground truth: which buckets contain malicious clients
        malicious_buckets_ground_truth = set()
        for client_id in self.malicious_client_ids:
            if client_id in bucket_assignments:
                bucket_id = bucket_assignments[client_id]
                malicious_buckets_ground_truth.add(bucket_id)
        
        # Get all unique buckets
        all_buckets = set(bucket_assignments.values())
        
        # Calculate TP, FP, TN, FN
        detected_set = set(detected_malicious_buckets)
        
        true_positives = len(detected_set & malicious_buckets_ground_truth)
        false_positives = len(detected_set - malicious_buckets_ground_truth)
        false_negatives = len(malicious_buckets_ground_truth - detected_set)
        true_negatives = len(all_buckets - detected_set - malicious_buckets_ground_truth)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            'total_malicious_buckets': len(malicious_buckets_ground_truth),
            'total_detected_buckets': len(detected_set)
        }
    
    def save_final_checkpoint(self, model, checkpoint_dir: str):
        """Save final model checkpoint"""
        checkpoint_path = Path(checkpoint_dir) / f"{self.experiment_name}.pt"
        
        # Save using your existing save mechanism
        # This is a placeholder - adapt to your actual model saving code
        print(f"💾 Saving checkpoint to: {checkpoint_path}")
        
        # Example (adapt to your framework):
        # import torch
        # torch.save(model.state_dict(), checkpoint_path)
        
        return str(checkpoint_path)
    
    def finalize(self):
        """Finalize metrics collection"""
        total_time = time.time() - self.experiment_start_time
        
        print(f"\n{'='*60}")
        print(f"📊 Metrics Collection Complete: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Total rounds: {self.current_round}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Total bytes sent: {self.total_bytes_sent / 1024 / 1024:.2f} MB")
        print(f"Total bytes received: {self.total_bytes_received / 1024 / 1024:.2f} MB")
        print(f"Metrics saved to: {self.metrics_file}")
        print(f"{'='*60}\n")
        
        # Save final summary
        summary = {
            'experiment_name': self.experiment_name,
            'total_rounds': self.current_round,
            'total_time_seconds': total_time,
            'total_bytes_sent': self.total_bytes_sent,
            'total_bytes_received': self.total_bytes_received,
            'metrics_file': str(self.metrics_file)
        }
        
        summary_path = self.results_dir / f"{self.experiment_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary


class CommunicationTracker:
    """Track communication bytes for FL"""
    
    def __init__(self):
        self.round_bytes_sent = 0
        self.round_bytes_received = 0
        
    def reset_round(self):
        """Reset counters for new round"""
        self.round_bytes_sent = 0
        self.round_bytes_received = 0
    
    def track_model_send(self, model_parameters: List[np.ndarray]):
        """Track bytes sent when sending model"""
        total_bytes = sum(param.nbytes for param in model_parameters)
        self.round_bytes_sent += total_bytes
        return total_bytes
    
    def track_model_receive(self, model_parameters: List[np.ndarray]):
        """Track bytes received when receiving model"""
        total_bytes = sum(param.nbytes for param in model_parameters)
        self.round_bytes_received += total_bytes
        return total_bytes
    
    def track_encrypted_send(self, encrypted_data_size: int):
        """Track encrypted communication"""
        self.round_bytes_sent += encrypted_data_size
        return encrypted_data_size
    
    def track_encrypted_receive(self, encrypted_data_size: int):
        """Track encrypted communication"""
        self.round_bytes_received += encrypted_data_size
        return encrypted_data_size
    
    def get_round_communication(self) -> Dict[str, int]:
        """Get communication for current round"""
        return {
            'bytes_sent': self.round_bytes_sent,
            'bytes_received': self.round_bytes_received,
            'total': self.round_bytes_sent + self.round_bytes_received
        }


def load_experiment_metrics(metrics_file: str) -> List[Dict[str, Any]]:
    """Load metrics from JSONL file"""
    
    metrics = []
    
    with open(metrics_file, 'r') as f:
        for line in f:
            metrics.append(json.loads(line))
    
    return metrics


def calculate_experiment_summary(metrics_file: str) -> Dict[str, Any]:
    """Calculate summary statistics from metrics"""
    
    metrics = load_experiment_metrics(metrics_file)
    
    if not metrics:
        return {}
    
    final_round = metrics[-1]
    
    # Extract accuracy over time
    accuracies = [m['test_accuracy'] for m in metrics]
    losses = [m['test_loss'] for m in metrics]
    asrs = [m['attack_success_rate'] for m in metrics]
    
    # Detection metrics (if available)
    detection_f1s = [
        m['detection_metrics']['f1'] 
        for m in metrics 
        if m.get('detection_metrics')
    ]
    
    summary = {
        'total_rounds': len(metrics),
        'final_accuracy': final_round['test_accuracy'],
        'final_loss': final_round['test_loss'],
        'final_asr': final_round['attack_success_rate'],
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'mean_asr': np.mean(asrs),
        'std_asr': np.std(asrs),
        'total_time_minutes': final_round['elapsed_total_seconds'] / 60,
        'total_communication_mb': (
            final_round['total_bytes_sent'] + 
            final_round['total_bytes_received']
        ) / 1024 / 1024
    }
    
    if detection_f1s:
        summary['mean_detection_f1'] = np.mean(detection_f1s)
        summary['std_detection_f1'] = np.std(detection_f1s)
        summary['final_detection_f1'] = detection_f1s[-1]
    
    return summary


# Example usage
if __name__ == "__main__":
    # Test metrics collector
    
    print("Testing AblationMetricsCollector...")
    
    # Create dummy data
    collector = AblationMetricsCollector(
        experiment_name="test_experiment",
        results_dir="test_results",
        num_clients=50,
        num_malicious=10,
        malicious_client_ids=list(range(10))
    )
    
    # Simulate 3 rounds
    for round_num in range(1, 4):
        collector.start_round(round_num)
        
        # Dummy metrics
        predictions = np.random.randint(0, 10, 1000)
        true_labels = np.random.randint(0, 10, 1000)
        
        collector.log_round_metrics(
            round_num=round_num,
            test_accuracy=0.85 + np.random.uniform(-0.05, 0.05),
            test_loss=0.5 + np.random.uniform(-0.1, 0.1),
            predictions=predictions,
            true_labels=true_labels,
            num_classes=10,
            bytes_sent_this_round=1024 * 1024,  # 1 MB
            bytes_received_this_round=1024 * 1024
        )
        
        time.sleep(1)
    
    # Finalize
    collector.finalize()
    
    # Load and summarize
    summary = calculate_experiment_summary(collector.metrics_file)
    print("\nExperiment Summary:")
    print(json.dumps(summary, indent=2))
    
    print("\n✅ Metrics collector test complete!")
