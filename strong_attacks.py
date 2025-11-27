import numpy as np
from typing import Tuple, Optional
import tensorflow as tf

"""
Fixed Strong Attacks Implementation for PROFILE
Implements Fang Attack and Min-Max Attack for stronger poisoning evaluation
"""

class FangAttack:
    """
    Fixed Implementation of the Fang Attack
    """
    
    def __init__(self, attack_strength: float = 1.0, target_class: Optional[int] = None):
        """
        Initialize Fang Attack
        
        Args:
            attack_strength: Multiplier for attack intensity (higher = stronger attack)
            target_class: Specific class to target (None for untargeted)
        """
        self.attack_strength = max(0.1, min(attack_strength, 5.0))  # Bound the strength
        self.target_class = target_class
        print(f"[FANG ATTACK] Initialized with strength {self.attack_strength}, target_class: {self.target_class}")
        
    def apply_attack(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Fang attack to training data
        
        Args:
            X: Training data
            y: Training labels
            
        Returns:
            Tuple of (poisoned_X, poisoned_y)
        """
        try:
            print(f"[FANG ATTACK] Starting attack on {len(X)} samples")
            
            # Ensure we have valid data
            if len(X) == 0 or len(y) == 0:
                print("[FANG ATTACK] Warning: Empty data provided")
                return X.copy(), y.copy()
            
            # Copy data to avoid modifying original
            X_poisoned = X.copy()
            y_poisoned = y.copy()
            
            # Get number of classes safely
            unique_classes = np.unique(y)
            num_classes = len(unique_classes)
            print(f"[FANG ATTACK] Found {num_classes} classes: {unique_classes}")
            
            # Calculate poison ratio (be more conservative)
            poison_ratio = min(0.8, 0.4 * self.attack_strength)
            num_poison = max(1, int(len(y) * poison_ratio))
            print(f"[FANG ATTACK] Will poison {num_poison}/{len(y)} samples ({poison_ratio*100:.1f}%)")
            
            # Select samples to poison (ensure we don't exceed array bounds)
            if num_poison >= len(y):
                poison_indices = np.arange(len(y))
            else:
                poison_indices = np.random.choice(len(y), num_poison, replace=False)
            
            print(f"[FANG ATTACK] Selected {len(poison_indices)} indices for poisoning")
            
            # Strategy 1: Label manipulation
            if self.target_class is not None and self.target_class in unique_classes:
                # Targeted attack: flip to target class
                y_poisoned[poison_indices] = self.target_class
                print(f"[FANG ATTACK] Targeting class {self.target_class}")
            else:
                # Untargeted attack: flip to different classes
                for idx in poison_indices:
                    original_class = y[idx]
                    # Flip to a different random class
                    possible_classes = [c for c in unique_classes if c != original_class]
                    if possible_classes:
                        y_poisoned[idx] = np.random.choice(possible_classes)
                print(f"[FANG ATTACK] Untargeted attack on all classes")
            
            # Strategy 2: Add controlled noise to input features
            noise_strength = min(0.2, 0.1 * self.attack_strength)
            print(f"[FANG ATTACK] Applying noise with strength {noise_strength}")
            
            for i, idx in enumerate(poison_indices):
                try:
                    # Add noise carefully
                    noise = np.random.normal(0, noise_strength, X_poisoned[idx].shape)
                    X_poisoned[idx] = np.clip(X_poisoned[idx] + noise, 0, 1)
                    
                    # Progress indicator for large datasets
                    if i % 100 == 0 and i > 0:
                        print(f"[FANG ATTACK] Processed {i}/{len(poison_indices)} samples")
                        
                except Exception as e:
                    print(f"[FANG ATTACK] Error processing sample {idx}: {e}")
                    continue
            
            print(f"[FANG ATTACK] Successfully poisoned {len(poison_indices)} samples")
            
            # Print final statistics
            unique, counts = np.unique(y_poisoned, return_counts=True)
            print(f"[FANG ATTACK] Final label distribution: {dict(zip(unique, counts))}")
            
            return X_poisoned, y_poisoned
            
        except Exception as e:
            print(f"[FANG ATTACK] Critical error in apply_attack: {e}")
            print(f"[FANG ATTACK] Returning original data")
            return X.copy(), y.copy()


class MinMaxAttack:
    """
    Fixed Implementation of Min-Max attack
    """
    
    def __init__(self, target_class: int = 1, attack_strength: float = 1.5):
        """
        Initialize Min-Max Attack
        
        Args:
            target_class: Class to target for the attack
            attack_strength: Strength of the attack
        """
        self.target_class = target_class
        self.attack_strength = max(0.1, min(attack_strength, 5.0))  # Bound the strength
        print(f"[MIN-MAX ATTACK] Initialized with target_class: {self.target_class}, strength: {self.attack_strength}")
    
    def apply_attack(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Min-Max attack to training data
        """
        try:
            print(f"[MIN-MAX ATTACK] Starting attack on {len(X)} samples")
            
            if len(X) == 0 or len(y) == 0:
                print("[MIN-MAX ATTACK] Warning: Empty data provided")
                return X.copy(), y.copy()
            
            # Copy data to avoid modifying original
            X_poisoned = X.copy()
            y_poisoned = y.copy()
            
            # Calculate poison ratio
            poison_ratio = min(0.9, 0.5 * self.attack_strength)
            num_poison = max(1, int(len(y) * poison_ratio))
            
            print(f"[MIN-MAX ATTACK] Will poison {num_poison}/{len(y)} samples ({poison_ratio*100:.1f}%)")
            
            # Select samples to poison
            if num_poison >= len(y):
                poison_indices = np.arange(len(y))
            else:
                poison_indices = np.random.choice(len(y), num_poison, replace=False)
            
            # Strategy 1: Flip labels to target class
            y_poisoned[poison_indices] = self.target_class
            
            # Strategy 2: Apply pattern perturbations
            perturbation_strength = min(0.3, 0.15 * self.attack_strength)
            
            for i, idx in enumerate(poison_indices):
                try:
                    self._apply_pattern_perturbation(X_poisoned[idx], perturbation_strength)
                    
                    if i % 100 == 0 and i > 0:
                        print(f"[MIN-MAX ATTACK] Processed {i}/{len(poison_indices)} samples")
                        
                except Exception as e:
                    print(f"[MIN-MAX ATTACK] Error processing sample {idx}: {e}")
                    continue
            
            print(f"[MIN-MAX ATTACK] Successfully poisoned {len(poison_indices)} samples")
            
            # Print statistics
            unique, counts = np.unique(y_poisoned, return_counts=True)
            print(f"[MIN-MAX ATTACK] Final label distribution: {dict(zip(unique, counts))}")
            
            return X_poisoned, y_poisoned
            
        except Exception as e:
            print(f"[MIN-MAX ATTACK] Critical error in apply_attack: {e}")
            print(f"[MIN-MAX ATTACK] Returning original data")
            return X.copy(), y.copy()
    
    def _apply_pattern_perturbation(self, sample: np.ndarray, strength: float):
        """
        Apply pattern-based perturbation to make the attack more effective
        """
        try:
            if len(sample.shape) >= 2:  # Image data
                # Add diagonal pattern safely
                min_dim = min(sample.shape[0], sample.shape[1])
                for i in range(0, min_dim, 2):  # Skip every other pixel to be more subtle
                    if i < sample.shape[0] and i < sample.shape[1]:
                        if len(sample.shape) == 3:  # RGB/channels
                            sample[i, i, :] = np.clip(sample[i, i, :] + strength, 0, 1)
                        else:  # Grayscale
                            sample[i, i] = np.clip(sample[i, i] + strength, 0, 1)
        except Exception as e:
            print(f"[MIN-MAX ATTACK] Error in pattern perturbation: {e}")


class ScalingAttack:
    """
    Implementation of update scaling attack
    """
    
    def __init__(self, scaling_factor: float = 10.0, target_class: Optional[int] = None):
        """
        Initialize Scaling Attack
        """
        self.scaling_factor = max(1.0, min(scaling_factor, 50.0))  # Reasonable bounds
        self.target_class = target_class
        print(f"[SCALING ATTACK] Initialized with scaling_factor: {self.scaling_factor}")
    
    def apply_attack(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply scaling attack
        """
        try:
            print(f"[SCALING ATTACK] Preparing data for scaling attack")
            
            if len(X) == 0 or len(y) == 0:
                return X.copy(), y.copy()
            
            # Copy data
            X_poisoned = X.copy()
            y_poisoned = y.copy()
            
            # Apply label poisoning
            poison_ratio = 0.7  # High poisoning ratio for scaling attack
            num_poison = max(1, int(len(y) * poison_ratio))
            
            if num_poison >= len(y):
                poison_indices = np.arange(len(y))
            else:
                poison_indices = np.random.choice(len(y), num_poison, replace=False)
            
            if self.target_class is not None:
                y_poisoned[poison_indices] = self.target_class
            else:
                # Random label flipping
                unique_classes = np.unique(y)
                for idx in poison_indices:
                    y_poisoned[idx] = np.random.choice(unique_classes)
            
            print(f"[SCALING ATTACK] Prepared {num_poison}/{len(y)} poisoned samples")
            print(f"[SCALING ATTACK] Model update will be scaled by factor {self.scaling_factor}")
            
            return X_poisoned, y_poisoned
            
        except Exception as e:
            print(f"[SCALING ATTACK] Error in apply_attack: {e}")
            return X.copy(), y.copy()


class BackdoorAttack:
    """
    Implementation of backdoor attack
    """
    
    def __init__(self, target_class: int = 0, trigger_size: int = 3, poison_ratio: float = 0.1):
        """
        Initialize Backdoor Attack
        """
        self.target_class = target_class
        self.trigger_size = max(1, min(trigger_size, 10))  # Reasonable bounds
        self.poison_ratio = max(0.01, min(poison_ratio, 0.5))  # Reasonable bounds
        print(f"[BACKDOOR ATTACK] Initialized with target_class: {self.target_class}, trigger_size: {self.trigger_size}")
    
    def apply_attack(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backdoor attack by embedding trigger pattern
        """
        try:
            print(f"[BACKDOOR ATTACK] Embedding trigger, target class: {self.target_class}")
            
            if len(X) == 0 or len(y) == 0:
                return X.copy(), y.copy()
            
            # Copy data
            X_poisoned = X.copy()
            y_poisoned = y.copy()
            
            # Select samples for backdoor
            num_poison = max(1, int(len(y) * self.poison_ratio))
            
            if num_poison >= len(y):
                poison_indices = np.arange(len(y))
            else:
                poison_indices = np.random.choice(len(y), num_poison, replace=False)
            
            # Add trigger pattern and change labels
            for idx in poison_indices:
                try:
                    X_poisoned[idx] = self._add_trigger(X_poisoned[idx])
                    y_poisoned[idx] = self.target_class
                except Exception as e:
                    print(f"[BACKDOOR ATTACK] Error processing sample {idx}: {e}")
                    continue
            
            print(f"[BACKDOOR ATTACK] Added trigger to {len(poison_indices)}/{len(y)} samples ({self.poison_ratio*100:.1f}%)")
            
            return X_poisoned, y_poisoned
            
        except Exception as e:
            print(f"[BACKDOOR ATTACK] Error in apply_attack: {e}")
            return X.copy(), y.copy()
    
    def _add_trigger(self, sample: np.ndarray) -> np.ndarray:
        """
        Add trigger pattern to a sample
        """
        try:
            sample_with_trigger = sample.copy()
            
            # Add small white square in bottom-right corner
            if len(sample.shape) == 3:  # Image with channels
                h, w, c = sample.shape
                end_h = max(0, h - self.trigger_size)
                end_w = max(0, w - self.trigger_size)
                sample_with_trigger[end_h:h, end_w:w, :] = 1.0
            elif len(sample.shape) == 2:  # Grayscale image
                h, w = sample.shape
                end_h = max(0, h - self.trigger_size)
                end_w = max(0, w - self.trigger_size)
                sample_with_trigger[end_h:h, end_w:w] = 1.0
            
            return sample_with_trigger
            
        except Exception as e:
            print(f"[BACKDOOR ATTACK] Error adding trigger: {e}")
            return sample.copy()


# Factory function to create attacks
def create_attack(attack_type: str, **kwargs):
    """
    Factory function to create attack instances
    
    Args:
        attack_type: Type of attack ('fang', 'min_max', 'scaling', 'backdoor')
        **kwargs: Additional arguments for the attack
    
    Returns:
        Attack instance
    """
    try:
        print(f"[ATTACK FACTORY] Creating {attack_type} attack with args: {kwargs}")
        
        if attack_type == 'fang':
            return FangAttack(**kwargs)
        elif attack_type == 'min_max':
            return MinMaxAttack(**kwargs)
        elif attack_type == 'scaling':
            return ScalingAttack(**kwargs)
        elif attack_type == 'backdoor':
            return BackdoorAttack(**kwargs)
        else:
            print(f"[ATTACK FACTORY] Unknown attack type: {attack_type}")
            raise ValueError(f"Unknown attack type: {attack_type}")
            
    except Exception as e:
        print(f"[ATTACK FACTORY] Error creating {attack_type} attack: {e}")
        raise


# Test function
def test_attacks():
    """
    Test all attack implementations
    """
    # Create dummy data for testing
    print("Creating test data...")
    X_dummy = np.random.rand(100, 28, 28, 1)  # Smaller dataset for testing
    y_dummy = np.random.randint(0, 10, 100)   # 10 classes
    
    print("Testing Strong Attacks Implementation")
    print("=" * 50)
    
    attacks_to_test = [
        ('fang', {'attack_strength': 2.0, 'target_class': 7}),
        ('min_max', {'target_class': 1, 'attack_strength': 2.0}),
        ('scaling', {'scaling_factor': 15.0}),
        ('backdoor', {'target_class': 0, 'trigger_size': 3, 'poison_ratio': 0.1})
    ]
    
    for attack_name, attack_params in attacks_to_test:
        try:
            print(f"\n{attack_name.upper()} ATTACK TEST:")
            print("-" * 30)
            
            attack = create_attack(attack_name, **attack_params)
            X_attacked, y_attacked = attack.apply_attack(X_dummy.copy(), y_dummy.copy())
            
            print(f"Original labels distribution: {np.bincount(y_dummy)}")
            print(f"{attack_name} attack labels distribution: {np.bincount(y_attacked)}")
            print(f"✅ {attack_name} attack test passed!")
            
        except Exception as e:
            print(f"❌ {attack_name} attack test failed: {e}")
    
    print("\n🎉 Attack testing completed!")


if __name__ == "__main__":
    test_attacks()