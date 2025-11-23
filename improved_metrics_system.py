# simplified_metrics_system.py
# Clean and simple metrics organization for PROFILE experiments

import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # For 0-d numpy arrays
        return obj.item()
    else:
        return obj

class ExperimentMetricsManager:
    """
    Simple metrics manager that saves results in an organized, descriptive way.
    No fancy features - just clean organization for easy identification.
    """
    
    def __init__(self, 
                 dataset: str,
                 num_clients: int, 
                 num_buckets: int,
                 attack_type: str = "none",
                 poison_ratio: float = 0.0,
                 epsilon: float = 1.0,
                 experiment_name: Optional[str] = None):
        
        # Generate descriptive experiment name if not provided
        if experiment_name is None:
            experiment_name = self._generate_experiment_name(
                dataset, num_clients, num_buckets, attack_type, poison_ratio, epsilon
            )
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join("experiments", experiment_name)
        
        # Create organized directory structure
        self._create_directories()
        
        # Save experiment configuration for reference
        self._save_config(dataset, num_clients, num_buckets, attack_type, poison_ratio, epsilon)
        
        print(f"📁 Experiment Directory: {self.experiment_dir}")
    
    def _generate_experiment_name(self, dataset, num_clients, num_buckets, attack_type, poison_ratio, epsilon):
        """Generate descriptive experiment name from parameters"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        parts = [
            f"dataset_{dataset}",
            f"clients_{num_clients}",
            f"buckets_{num_buckets}",
            f"eps_{epsilon:.1f}"
        ]
        
        # Add attack info if there's an attack
        if attack_type != "none":
            parts.append(f"attack_{attack_type}")
            if poison_ratio > 0:
                parts.append(f"poison_{poison_ratio:.1f}")
        
        parts.append(timestamp)
        return "_".join(parts)
    
    def _create_directories(self):
        """Create clean directory structure"""
        subdirs = ["server", "clients", "privacy", "detection", "plots"]
        
        for subdir in subdirs:
            os.makedirs(os.path.join(self.experiment_dir, subdir), exist_ok=True)
    
    def _save_config(self, dataset, num_clients, num_buckets, attack_type, poison_ratio, epsilon):
        """Save experiment configuration"""
        config = {
            'experiment_name': self.experiment_name,
            'dataset': dataset,
            'num_clients': num_clients,
            'num_buckets': num_buckets,
            'attack_type': attack_type,
            'poison_ratio': poison_ratio,
            'epsilon': epsilon,
            'created_at': datetime.now().isoformat()
        }
        
        config_file = os.path.join(self.experiment_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
    
    def save_server_metrics(self, metrics: Dict[str, Any], filename: str = "server_metrics"):
        """Save server metrics to organized location"""
        try:
            # Convert NumPy types
            metrics = convert_numpy_types(metrics)
            metrics['timestamp'] = time.time()
            
            # Save as JSON Lines for easy reading
            jsonl_file = os.path.join(self.experiment_dir, "server", f"{filename}.jsonl")
            with open(jsonl_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
            
            # Also save as CSV for easy analysis
            csv_file = os.path.join(self.experiment_dir, "server", f"{filename}.csv")
            self._save_to_csv(metrics, csv_file)
            
        except Exception as e:
            print(f"❌ Error saving server metrics: {e}")
    
    def save_client_metrics(self, client_id: int, metrics: Dict[str, Any], filename: str = "client_metrics"):
        """Save client metrics to organized location"""
        try:
            # Convert NumPy types
            metrics = convert_numpy_types(metrics)
            metrics['client_id'] = client_id
            metrics['timestamp'] = time.time()
            
            # Save individual client file
            client_file = os.path.join(self.experiment_dir, "clients", f"client_{client_id}_{filename}.jsonl")
            with open(client_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
            
            # Save to consolidated file
            all_clients_file = os.path.join(self.experiment_dir, "clients", f"all_{filename}.csv")
            self._save_to_csv(metrics, all_clients_file)
            
        except Exception as e:
            print(f"❌ Error saving client {client_id} metrics: {e}")
    
    def save_privacy_metrics(self, metrics: Dict[str, Any], filename: str = "privacy_metrics"):
        """Save privacy analysis metrics"""
        try:
            # Convert NumPy types
            metrics = convert_numpy_types(metrics)
            metrics['timestamp'] = time.time()
            
            # Save privacy metrics
            privacy_file = os.path.join(self.experiment_dir, "privacy", f"{filename}.jsonl")
            with open(privacy_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
            
            csv_file = os.path.join(self.experiment_dir, "privacy", f"{filename}.csv")
            self._save_to_csv(metrics, csv_file)
            
        except Exception as e:
            print(f"❌ Error saving privacy metrics: {e}")
    
    def save_detection_metrics(self, metrics: Dict[str, Any], filename: str = "detection_metrics"):
        """Save detection analysis metrics"""
        try:
            # Convert NumPy types
            metrics = convert_numpy_types(metrics)
            metrics['timestamp'] = time.time()
            
            # Save detection metrics
            detection_file = os.path.join(self.experiment_dir, "detection", f"{filename}.jsonl")
            with open(detection_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
            
            csv_file = os.path.join(self.experiment_dir, "detection", f"{filename}.csv")
            self._save_to_csv(metrics, csv_file)
            
        except Exception as e:
            print(f"❌ Error saving detection metrics: {e}")
    
    def _save_to_csv(self, metrics: Dict[str, Any], csv_file: str):
        """Save metrics to CSV file"""
        try:
            # Flatten nested dictionaries for CSV
            flat_metrics = self._flatten_dict(metrics)
            
            # Check if file exists to write headers
            file_exists = os.path.isfile(csv_file)
            
            with open(csv_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(flat_metrics.keys()))
                if not file_exists:
                    writer.writeheader()
                writer.writerow(flat_metrics)
                
        except Exception as e:
            print(f"⚠️  CSV save warning: {e}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_'):
        """Flatten nested dictionary for CSV"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list) and len(v) <= 10:  # Small lists only
                items.append((new_key, ','.join(map(str, v))))
            else:
                items.append((new_key, str(v)))
        
        return dict(items)
    
    # Add this property to the ExperimentMetricsManager class
    @property
    def config(self):
        """Get experiment config for compatibility"""
        try:
            config_file = os.path.join(self.experiment_dir, "experiment_config.json")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    return json.load(f)
            return {}
        except:
            return {}    
    def get_experiment_summary(self):
        """Get experiment summary for compatibility"""
        try:
            config_file = os.path.join(self.experiment_dir, "experiment_config.json")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Return basic summary
            return {
                'experiment_name': self.experiment_name,
                'experiment_dir': self.experiment_dir,
                'config': config,
                'created_at': config.get('created_at', datetime.now().isoformat())
            }
        except Exception as e:
            print(f"⚠️ Error getting experiment summary: {e}")
            return {
                'experiment_name': self.experiment_name,
                'experiment_dir': self.experiment_dir,
                'error': str(e)
            }    
    def get_experiment_path(self, subdir: str = "") -> str:
        """Get path to experiment directory or subdirectory"""
        if subdir:
            return os.path.join(self.experiment_dir, subdir)
        return self.experiment_dir


# Simple factory functions for easy integration

def create_server_metrics_manager(args):
    """Create metrics manager for server"""
    return ExperimentMetricsManager(
        dataset=args.dataset,
        num_clients=args.num_clients,
        num_buckets=args.num_buckets,
        epsilon=getattr(args, 'epsilon', 1.0),
        attack_type=getattr(args, 'attack_type', 'none'),
        poison_ratio=getattr(args, 'poison_ratio', 0.0)
    )

def create_client_metrics_manager(args, client_id: int):
    """Create metrics manager for client"""
    attack_type = args.attack_type if args.malicious else "none"
    poison_ratio = args.poison_ratio if args.malicious else 0.0
    
    return ExperimentMetricsManager(
        dataset=args.dataset,
        num_clients=args.num_clients,
        num_buckets=getattr(args, 'num_buckets', 2),
        epsilon=getattr(args, 'epsilon', 1.0),
        attack_type=attack_type,
        poison_ratio=poison_ratio
    )


# Test the system
if __name__ == "__main__":
    # Test with sample data
    class MockArgs:
        dataset = "mnist"
        num_clients = 6
        num_buckets = 2
        epsilon = 1.0
        attack_type = "label_flip"
        poison_ratio = 0.3
        malicious = True
    
    args = MockArgs()
    
    # Test server metrics
    server_metrics = create_server_metrics_manager(args)
    
    # Test saving some metrics
    test_metrics = {
        'round': 1,
        'accuracy': np.float64(0.95),
        'loss': np.float32(0.05),
        'confusion_matrix': np.array([[100, 5], [2, 98]])
    }
    
    server_metrics.save_server_metrics(test_metrics, "test_round")
    print(f"✅ Test metrics saved to: {server_metrics.experiment_dir}")