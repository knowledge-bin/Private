#!/usr/bin/env python3
"""
Single Ablation Experiment Runner

This script runs ONE ablation experiment by:
1. Starting your existing PROFILE_server.py with appropriate flags
2. Starting 50 clients (10 malicious, 40 honest)
3. Monitoring progress
4. Collecting results after completion

Usage:
    python run_single_ablation_experiment.py \
        --config A_Bucketing_Only \
        --attack label_flip \
        --seed 42
"""

import os
import sys
import time
import subprocess
import argparse
import signal
import json
from pathlib import Path
from datetime import datetime


class SingleExperimentRunner:
    """Runs a single ablation experiment with existing PROFILE code"""
    
    def __init__(self, config_name, attack_name, seed, results_base_dir="ablation_results", num_rounds=50):
        self.config_name = config_name
        self.attack_name = attack_name
        self.seed = seed
        
        # Create experiment name
        self.experiment_name = f"mnist_lenet5_{config_name}_{attack_name}_seed{seed}"
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(results_base_dir) / f"batch_{timestamp}"
        self.exp_dir = self.results_dir / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration mapping (using disable flags as PROFILE_server.py expects)
        self.configs = {
            'A_Bucketing_Only': {
                'num_buckets': 16,
                'disable_he': False,       # HE enabled
                'disable_dp': True,        # DP disabled
                'disable_validation': True,   # Validators disabled
            },
            'B_Bucketing_DP': {
                'num_buckets': 16,
                'disable_he': False,       # HE enabled
                'disable_dp': False,       # DP enabled (epsilon=1.0 hardcoded in server)
                'disable_validation': True,   # Validators disabled
            },
            'C_Bucketing_Validators': {
                'num_buckets': 16,
                'disable_he': False,       # HE enabled
                'disable_dp': True,        # DP disabled
                'disable_validation': False,  # Validators enabled
            },
            'D_PROFILE_Full': {
                'num_buckets': 16,
                'disable_he': False,       # HE enabled
                'disable_dp': False,       # DP enabled
                'disable_validation': False,  # Validators enabled (all components)
            },
            'E_FedAvg_Baseline': {
                'num_buckets': 1,          # Effectively no bucketing (single bucket = FedAvg)
                'disable_he': False,       # Still use HE for fairness
                'disable_dp': True,        # DP disabled
                'disable_validation': True,   # Validators disabled
            }
        }
        
        # Experiment parameters
        self.num_clients = 50
        self.num_rounds = num_rounds  # Now customizable
        self.malicious_fraction = 0.3  # 30% malicious (matches paper contribution)
        self.malicious_count = int(self.num_clients * self.malicious_fraction)  # 15 clients
        self.malicious_ids = list(range(self.malicious_count))
        
        # Get config
        if config_name not in self.configs:
            raise ValueError(f"Unknown config: {config_name}")
        
        self.config = self.configs[config_name]
        
        # Process list
        self.processes = []
        
        print(f"\n{'='*80}")
        print(f"Experiment: {self.experiment_name}")
        print(f"{'='*80}")
        print(f"Config: {config_name}")
        print(f"Attack: {attack_name}")
        print(f"Seed: {seed}")
        print(f"Clients: {self.num_clients} (malicious: {self.malicious_count})")
        print(f"Rounds: {self.num_rounds}")
        print(f"Output: {self.exp_dir}")
        print(f"{'='*80}\n")
    
    def save_config(self):
        """Save experiment configuration"""
        config_data = {
            'experiment_name': self.experiment_name,
            'config_name': self.config_name,
            'attack_name': self.attack_name,
            'seed': self.seed,
            'num_clients': self.num_clients,
            'num_rounds': self.num_rounds,
            'malicious_count': self.malicious_count,
            'malicious_ids': self.malicious_ids,
            'config_params': self.config,
            'start_time': datetime.now().isoformat()
        }
        
        config_file = self.exp_dir / 'experiment_config.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Config saved: {config_file}")
    
    def start_server(self):
        """Start PROFILE server"""
        print("\n🖥️  Starting server...")
        
        server_log = self.exp_dir / 'server.log'
        
        # Use system Python or conda environment
        conda_python = 'python'
        
        # Build server command
        cmd = [
            conda_python,
            'PROFILE_server.py',
            '--dataset', 'mnist',
            '--num_clients', str(self.num_clients),
            '--num_buckets', str(self.config['num_buckets']),
            '--num_rounds', str(self.num_rounds),
            '--seed', str(self.seed)
        ]
        
        # Add disable flags
        if self.config.get('disable_he', False):
            cmd.append('--disable_he')
        
        if self.config.get('disable_dp', False):
            cmd.append('--disable_dp')
        
        if self.config.get('disable_validation', False):
            cmd.append('--disable_validation')
        
        print(f"Server command: {' '.join(cmd)}")
        
        # Start server
        with open(server_log, 'w') as log:
            server_process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd()
            )
        
        self.processes.append(server_process)
        print(f"✅ Server started (PID: {server_process.pid})")
        print(f"   Log: {server_log}")
        
        # Wait for server to initialize
        print("⏳ Waiting 10 seconds for server initialization...")
        time.sleep(10)
        
        return server_process
    
    def start_clients(self):
        """Start all clients"""
        print(f"\n👥 Starting {self.num_clients} clients...")
        
        client_processes = []
        
        for client_id in range(self.num_clients):
            # Determine if malicious
            is_malicious = client_id in self.malicious_ids
            
            client_log = self.exp_dir / f'client_{client_id}.log'
            
            # Use system Python or conda environment
            conda_python = 'python'
            
            # Build client command
            cmd = [
                conda_python,
                'Clean-client2.py',
                '--client_id', str(client_id),
                '--dataset', 'mnist',
                '--num_clients', str(self.num_clients),
                '--seed', str(self.seed + client_id)
            ]
            
            # Add attack if malicious
            if is_malicious:
                cmd.extend([
                    '--malicious',
                    '--attack_type', self.attack_name,
                    '--poison_ratio', '1.0',
                    '--target_class', '1'
                ])
            
            # Start client
            with open(client_log, 'w') as log:
                client_process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd()
                )
            
            client_processes.append(client_process)
            self.processes.append(client_process)
            
            status = "🔴 MALICIOUS" if is_malicious else "🟢 HONEST"
            print(f"  Client {client_id:2d}: {status} (PID: {client_process.pid})")
            
            # Small delay between clients
            time.sleep(0.2)
        
        print(f"✅ All {self.num_clients} clients started")
        
        return client_processes
    
    def monitor_experiment(self, server_process, client_processes):
        """Monitor experiment progress"""
        print(f"\n📊 Monitoring experiment...")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            while True:
                # Check if server is still running
                if server_process.poll() is not None:
                    print("\n✅ Server completed!")
                    break
                
                # Status update every 60 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 60 == 0 and elapsed > 0:
                    active_clients = sum(1 for p in client_processes if p.poll() is None)
                    print(f"⏱️  Elapsed: {elapsed/60:.1f} min | Active clients: {active_clients}/{self.num_clients}")
                
                time.sleep(5)
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            self.cleanup()
            return False
        
        # Final timing
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"{'='*80}")
        
        return True
    
    def cleanup(self):
        """Clean up all processes"""
        print("\n🧹 Cleaning up processes...")
        
        for process in self.processes:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        
        print("✅ Cleanup complete")
    
    def collect_results(self):
        """Collect and summarize results"""
        print("\n📊 Collecting results...")
        
        # Look for metrics in the experiment directory
        metrics_dir = Path('metrics')
        
        if metrics_dir.exists():
            # Find relevant metrics
            pattern = f"dataset_mnist_*"
            metrics_files = list(metrics_dir.glob(pattern))
            
            if metrics_files:
                # Get most recent
                latest_metrics = max(metrics_files, key=lambda p: p.stat().st_mtime)
                print(f"✅ Found metrics: {latest_metrics}")
                
                # Copy to experiment directory
                import shutil
                dest = self.exp_dir / 'metrics'
                if latest_metrics.is_dir():
                    shutil.copytree(latest_metrics, dest, dirs_exist_ok=True)
                    print(f"✅ Metrics copied to: {dest}")
            else:
                print("⚠️  No metrics files found")
        
        print(f"\n{'='*80}")
        print(f"✅ Experiment Complete: {self.experiment_name}")
        print(f"{'='*80}")
        print(f"Results: {self.exp_dir}")
        print(f"Logs: {self.exp_dir}/*.log")
        print(f"{'='*80}\n")
    
    def run(self):
        """Run the complete experiment"""
        try:
            # Save config
            self.save_config()
            
            # Start server
            server_process = self.start_server()
            
            # Start clients
            client_processes = self.start_clients()
            
            # Monitor
            success = self.monitor_experiment(server_process, client_processes)
            
            if success:
                # Collect results
                self.collect_results()
                return True
            else:
                return False
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            # Always cleanup
            self.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description='Run a single PROFILE ablation experiment'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        choices=['A_Bucketing_Only', 'B_Bucketing_DP', 'C_Bucketing_Validators', 
                 'D_PROFILE_Full', 'E_FedAvg_Baseline'],
        help='Ablation configuration'
    )
    parser.add_argument(
        '--attack',
        type=str,
        required=True,
        choices=['label_flip', 'min_max'],
        help='Attack type'
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=True,
        help='Random seed (e.g., 42, 123, 456)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='ablation_results',
        help='Base directory for results'
    )
    parser.add_argument(
        '--num-rounds',
        type=int,
        default=50,
        help='Number of training rounds (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Create and run experiment
    runner = SingleExperimentRunner(
        config_name=args.config,
        attack_name=args.attack,
        seed=args.seed,
        results_base_dir=args.results_dir,
        num_rounds=args.num_rounds
    )
    
    success = runner.run()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
