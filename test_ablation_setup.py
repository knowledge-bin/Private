#!/usr/bin/env python3
"""
Test Script: Verify Ablation Study Setup

This script tests all components without running actual experiments:
1. Imports work correctly
2. Metrics collector functions
3. Communication tracker works
4. Result analyzer can load dummy data
5. All required files exist
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("\n" + "="*60)
    print("TEST 1: Importing Modules")
    print("="*60)
    
    try:
        print("  Testing ablation_mnist_lenet...")
        import ablation_mnist_lenet
        print("  ✅ ablation_mnist_lenet imported")
    except Exception as e:
        print(f"  ❌ ablation_mnist_lenet failed: {e}")
        return False
    
    try:
        print("  Testing ablation_metrics...")
        import ablation_metrics
        print("  ✅ ablation_metrics imported")
    except Exception as e:
        print(f"  ❌ ablation_metrics failed: {e}")
        return False
    
    try:
        print("  Testing plot_ablation_results...")
        import plot_ablation_results
        print("  ✅ plot_ablation_results imported")
    except Exception as e:
        print(f"  ❌ plot_ablation_results failed: {e}")
        return False
    
    print("\n✅ All imports successful!")
    return True


def test_metrics_collector():
    """Test metrics collector with dummy data"""
    print("\n" + "="*60)
    print("TEST 2: Metrics Collector")
    print("="*60)
    
    try:
        from ablation_metrics import AblationMetricsCollector
        import numpy as np
        import tempfile
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"  Creating metrics collector in {tmpdir}")
            
            collector = AblationMetricsCollector(
                experiment_name="test_experiment",
                results_dir=tmpdir,
                num_clients=10,
                num_malicious=2,
                malicious_client_ids=[0, 1]
            )
            
            print("  ✅ Metrics collector created")
            
            # Test logging one round
            collector.start_round(1)
            
            predictions = np.random.randint(0, 10, 100)
            true_labels = np.random.randint(0, 10, 100)
            
            metrics = collector.log_round_metrics(
                round_num=1,
                test_accuracy=0.85,
                test_loss=0.5,
                predictions=predictions,
                true_labels=true_labels,
                num_classes=10,
                bytes_sent_this_round=1024,
                bytes_received_this_round=1024
            )
            
            print("  ✅ Round metrics logged")
            
            # Check metrics file exists
            metrics_file = Path(tmpdir) / "test_experiment.jsonl"
            if metrics_file.exists():
                print(f"  ✅ Metrics file created: {metrics_file}")
                
                # Read file
                with open(metrics_file, 'r') as f:
                    line = f.readline()
                    import json
                    data = json.loads(line)
                    
                    required_fields = [
                        'round', 'test_accuracy', 'test_loss',
                        'attack_success_rate', 'bytes_sent_this_round'
                    ]
                    
                    for field in required_fields:
                        if field in data:
                            print(f"    ✓ Field '{field}' present")
                        else:
                            print(f"    ✗ Field '{field}' MISSING")
                            return False
                
                print("  ✅ All required fields present")
            else:
                print(f"  ❌ Metrics file not created")
                return False
        
        print("\n✅ Metrics collector test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Metrics collector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_communication_tracker():
    """Test communication tracker"""
    print("\n" + "="*60)
    print("TEST 3: Communication Tracker")
    print("="*60)
    
    try:
        from ablation_metrics import CommunicationTracker
        import numpy as np
        
        tracker = CommunicationTracker()
        print("  ✅ Communication tracker created")
        
        # Test tracking
        dummy_params = [np.random.rand(100, 100) for _ in range(5)]
        
        bytes_sent = tracker.track_model_send(dummy_params)
        print(f"  ✅ Tracked sending: {bytes_sent} bytes")
        
        bytes_received = tracker.track_model_receive(dummy_params)
        print(f"  ✅ Tracked receiving: {bytes_received} bytes")
        
        stats = tracker.get_round_communication()
        print(f"  ✅ Round stats: {stats}")
        
        if stats['bytes_sent'] == bytes_sent and stats['bytes_received'] == bytes_received:
            print("  ✅ Stats match tracked values")
        else:
            print("  ❌ Stats don't match")
            return False
        
        print("\n✅ Communication tracker test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Communication tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_experiment_runner():
    """Test experiment runner configuration"""
    print("\n" + "="*60)
    print("TEST 4: Experiment Runner Configuration")
    print("="*60)
    
    try:
        from ablation_mnist_lenet import AblationStudyRunner
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = AblationStudyRunner(base_results_dir=tmpdir)
            print("  ✅ Experiment runner created")
            
            # Check configurations
            expected_configs = [
                'A_Bucketing_Only',
                'B_Bucketing_DP',
                'C_Bucketing_Validators',
                'D_PROFILE_Full',
                'E_FedAvg_Baseline'
            ]
            
            for config in expected_configs:
                if config in runner.configurations:
                    print(f"    ✓ Config '{config}' present")
                else:
                    print(f"    ✗ Config '{config}' MISSING")
                    return False
            
            print("  ✅ All 5 configurations present")
            
            # Check attacks
            expected_attacks = ['label_flip', 'min_max']
            for attack in expected_attacks:
                if attack in runner.attacks:
                    print(f"    ✓ Attack '{attack}' present")
                else:
                    print(f"    ✗ Attack '{attack}' MISSING")
                    return False
            
            print("  ✅ All 2 attacks present")
            
            # Check seeds
            if len(runner.seeds) == 3:
                print(f"  ✅ 3 seeds configured: {runner.seeds}")
            else:
                print(f"  ❌ Expected 3 seeds, got {len(runner.seeds)}")
                return False
            
            # Check total experiments
            total = len(runner.configurations) * len(runner.attacks) * len(runner.seeds)
            if total == 30:
                print(f"  ✅ Total experiments: {total}")
            else:
                print(f"  ❌ Expected 30 experiments, got {total}")
                return False
        
        print("\n✅ Experiment runner test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Experiment runner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """Test that all required files exist"""
    print("\n" + "="*60)
    print("TEST 5: File Structure")
    print("="*60)
    
    required_files = [
        'ablation_mnist_lenet.py',
        'ablation_metrics.py',
        'plot_ablation_results.py',
        'ABLATION_STUDY_README.md',
        'INTEGRATION_GUIDE.py',
        'ABLATION_PACKAGE_SUMMARY.md',
        'run_ablation_study.sh'
    ]
    
    all_exist = True
    
    for filename in required_files:
        filepath = Path(filename)
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✅ {filename} ({size} bytes)")
        else:
            print(f"  ❌ {filename} MISSING")
            all_exist = False
    
    if all_exist:
        print("\n✅ All required files present!")
        return True
    else:
        print("\n❌ Some files are missing")
        return False


def test_dependencies():
    """Test that required dependencies are installed"""
    print("\n" + "="*60)
    print("TEST 6: Dependencies")
    print("="*60)
    
    required_packages = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'sklearn',
        'psutil'
    ]
    
    all_installed = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} NOT INSTALLED")
            all_installed = False
    
    if all_installed:
        print("\n✅ All dependencies installed!")
        return True
    else:
        print("\n❌ Some dependencies missing. Install with:")
        print("  pip install numpy pandas matplotlib seaborn scikit-learn psutil")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ABLATION STUDY SETUP VERIFICATION")
    print("="*60)
    print("\nThis script verifies that your ablation study setup is correct.")
    print("Running 6 tests...\n")
    
    tests = [
        ("Imports", test_imports),
        ("Metrics Collector", test_metrics_collector),
        ("Communication Tracker", test_communication_tracker),
        ("Experiment Runner", test_experiment_runner),
        ("File Structure", test_file_structure),
        ("Dependencies", test_dependencies)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}  {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour ablation study setup is ready!")
        print("\nNext steps:")
        print("1. Review INTEGRATION_GUIDE.py for server/client integration")
        print("2. Test with dry run: python ablation_mnist_lenet.py --dry-run")
        print("3. Run full study: ./run_ablation_study.sh")
        print("\nGood luck! 🚀")
        return 0
    else:
        print("\n" + "="*60)
        print("⚠️  SOME TESTS FAILED")
        print("="*60)
        print("\nPlease fix the failed tests before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
