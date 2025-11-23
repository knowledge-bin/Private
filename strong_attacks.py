# # strong_attacks.py
# import numpy as np
# import tensorflow as tf

# class FangAttack:
#     def __init__(self, attack_strength=2.0, target_class=1):
#         self.lambda_val = attack_strength
#         self.target_class = target_class  # Added target class parameter
    
#     def apply_attack(self, X, y, benign_gradients=None):
#         """Improved Fang attack implementation"""
#         # 1. Strategic label flipping
#         y_attacked = np.full_like(y, self.target_class)
        
#         # 2. Gradient-aligned noise (simulated)
#         if benign_gradients is not None:
#             # Use gradient direction for noise pattern
#             noise_pattern = np.mean(benign_gradients, axis=0)
#             noise = noise_pattern * self.lambda_val
#         else:
#             # Fallback to directional noise
#             noise = np.random.normal(0, 0.15 * self.lambda_val, X.shape)
        
#         X_attacked = X + noise
#         X_attacked = np.clip(X_attacked, 0, 1)
        
#         return X_attacked, y_attacked

# class MinMaxAttack:
#     def __init__(self, target_class=1, epsilon=0.3):
#         self.target_class = target_class
#         self.epsilon = epsilon  # Controls perturbation strength
    
#     def apply_attack(self, X, y):
#         """Improved Min-Max attack implementation"""
#         # 1. Create target class cluster
#         y_attacked = np.full_like(y, self.target_class)
        
#         # 2. Differentiated perturbations
#         mask = (y != self.target_class).reshape(-1, 1, 1, 1)  # For image data
        
#         # For non-target samples: max loss perturbations
#         perturbation = np.random.uniform(-self.epsilon, self.epsilon, X.shape)
#         X_attacked = X + perturbation * mask  # Only perturb non-target samples
        
#         # For target samples: min loss (keep clean)
#         X_attacked = np.clip(X_attacked, 0, 1)
        
#         return X_attacked, y_attacked


# import numpy as np
# from typing import Tuple, Optional
# import tensorflow as tf

# """
# Strong Attacks Implementation for PROFILE
# Implements Fang Attack and Min-Max Attack for stronger poisoning evaluation
# """

# class FangAttack:
#     """
#     Implementation of the Fang Attack from "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning"
    
#     This attack optimizes the poisoned update to maximize the global model's loss
#     while remaining undetected by defense mechanisms.
#     """
    
#     def __init__(self, attack_strength: float = 1.0, target_class: Optional[int] = None):
#         """
#         Initialize Fang Attack
        
#         Args:
#             attack_strength: Multiplier for attack intensity (higher = stronger attack)
#             target_class: Specific class to target (None for untargeted)
#         """
#         self.attack_strength = attack_strength
#         self.target_class = target_class
        
#     def apply_attack(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Apply Fang attack to training data
        
#         Args:
#             X: Training data
#             y: Training labels
            
#         Returns:
#             Tuple of (poisoned_X, poisoned_y)
#         """
#         print(f"[FANG ATTACK] Applying with strength {self.attack_strength}")
        
#         # Copy data to avoid modifying original
#         X_poisoned = X.copy()
#         y_poisoned = y.copy()
        
#         # Get number of classes
#         num_classes = len(np.unique(y))
        
#         # For Fang attack, we manipulate a significant portion of the data
#         poison_ratio = min(0.9, 0.6 * self.attack_strength)  # More aggressive than original
#         num_poison = int(len(y) * poison_ratio)
        
#         # Select samples to poison (random selection)
#         poison_indices = np.random.choice(len(y), num_poison, replace=False)
        
#         # Strategy 1: Aggressive label flipping
#         if self.target_class is not None:
#             # Targeted attack: flip everything to target class
#             y_poisoned[poison_indices] = self.target_class
#             print(f"[FANG ATTACK] Targeting class {self.target_class}")
#         else:
#             # Untargeted attack: flip to wrong classes strategically
#             for idx in poison_indices:
#                 original_class = y[idx]
#                 # Flip to the class that's furthest from original (for maximum confusion)
#                 possible_classes = [c for c in range(num_classes) if c != original_class]
#                 y_poisoned[idx] = np.random.choice(possible_classes)
#             print(f"[FANG ATTACK] Untargeted attack on all classes")
        
#         # Strategy 2: Add noise to input features to make detection harder
#         noise_strength = 0.15 * self.attack_strength  # Increased noise
#         for idx in poison_indices:
#             # Add carefully crafted noise that doesn't make samples obviously wrong
#             noise = np.random.normal(0, noise_strength, X_poisoned[idx].shape)
#             X_poisoned[idx] = np.clip(X_poisoned[idx] + noise, 0, 1)
        
#         # Strategy 3: Gradient-based poisoning (simulate model manipulation)
#         # This is more sophisticated than simple label flipping
#         self._apply_gradient_poisoning(X_poisoned, y_poisoned, poison_indices)
        
#         print(f"[FANG ATTACK] Poisoned {num_poison}/{len(y)} samples ({poison_ratio*100:.1f}%)")
        
#         # Print attack statistics
#         unique, counts = np.unique(y_poisoned, return_counts=True)
#         print(f"[FANG ATTACK] Final label distribution: {dict(zip(unique, counts))}")
        
#         return X_poisoned, y_poisoned
    
#     def _apply_gradient_poisoning(self, X_poisoned: np.ndarray, y_poisoned: np.ndarray, poison_indices: np.ndarray):
#         """
#         Apply gradient-based poisoning to make the attack more sophisticated
#         """
#         # Add adversarial perturbations that are optimized to fool the model
#         perturbation_strength = 0.1 * self.attack_strength
        
#         for idx in poison_indices:
#             # Create adversarial perturbation
#             original_shape = X_poisoned[idx].shape
            
#             # Simple adversarial noise (in practice, this would use gradients)
#             adversarial_noise = np.random.uniform(-perturbation_strength, perturbation_strength, original_shape)
            
#             # Apply noise while keeping values in valid range
#             X_poisoned[idx] = np.clip(X_poisoned[idx] + adversarial_noise, 0, 1)


# class MinMaxAttack:
#     """
#     Implementation of Min-Max attack that tries to maximize the distance 
#     from honest updates while minimizing detection probability.
#     """
    
#     def __init__(self, target_class: int = 1, attack_strength: float = 1.5):
#         """
#         Initialize Min-Max Attack
        
#         Args:
#             target_class: Class to target for the attack
#             attack_strength: Strength of the attack
#         """
#         self.target_class = target_class
#         self.attack_strength = attack_strength
    
#     def apply_attack(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Apply Min-Max attack to training data
#         """
#         print(f"[MIN-MAX ATTACK] Applying with strength {self.attack_strength}, targeting class {self.target_class}")
        
#         # Copy data to avoid modifying original
#         X_poisoned = X.copy()
#         y_poisoned = y.copy()
        
#         # More aggressive poisoning ratio
#         poison_ratio = min(0.95, 0.7 * self.attack_strength)
#         num_poison = int(len(y) * poison_ratio)
        
#         # Select samples to poison
#         poison_indices = np.random.choice(len(y), num_poison, replace=False)
        
#         # Strategy 1: Flip all labels to target class
#         y_poisoned[poison_indices] = self.target_class
        
#         # Strategy 2: Apply sophisticated input perturbations
#         for idx in poison_indices:
#             # Apply pattern-based perturbation
#             self._apply_pattern_perturbation(X_poisoned[idx])
        
#         print(f"[MIN-MAX ATTACK] Poisoned {num_poison}/{len(y)} samples ({poison_ratio*100:.1f}%)")
        
#         # Print attack statistics
#         unique, counts = np.unique(y_poisoned, return_counts=True)
#         print(f"[MIN-MAX ATTACK] Final label distribution: {dict(zip(unique, counts))}")
        
#         return X_poisoned, y_poisoned
    
#     def _apply_pattern_perturbation(self, sample: np.ndarray):
#         """
#         Apply pattern-based perturbation to make the attack more effective
#         """
#         # Add a subtle but consistent pattern
#         if len(sample.shape) == 3:  # Image with channels
#             # Add diagonal pattern
#             for i in range(min(sample.shape[0], sample.shape[1])):
#                 if i < sample.shape[0] and i < sample.shape[1]:
#                     sample[i, i] = np.clip(sample[i, i] + 0.2 * self.attack_strength, 0, 1)
#         elif len(sample.shape) == 2:  # Grayscale image
#             # Add diagonal pattern
#             for i in range(min(sample.shape[0], sample.shape[1])):
#                 sample[i, i] = np.clip(sample[i, i] + 0.2 * self.attack_strength, 0, 1)


# class ScalingAttack:
#     """
#     Implementation of update scaling attack where malicious client 
#     amplifies their contribution to dominate aggregation.
#     """
    
#     def __init__(self, scaling_factor: float = 10.0, target_class: Optional[int] = None):
#         """
#         Initialize Scaling Attack
        
#         Args:
#             scaling_factor: Factor by which to scale the malicious update
#             target_class: Specific class to target (None for untargeted)
#         """
#         self.scaling_factor = scaling_factor
#         self.target_class = target_class
    
#     def apply_attack(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Apply scaling attack (note: actual scaling happens at model update level)
#         """
#         print(f"[SCALING ATTACK] Preparing data with scaling factor {self.scaling_factor}")
        
#         # Copy data
#         X_poisoned = X.copy()
#         y_poisoned = y.copy()
        
#         # Apply label poisoning
#         poison_ratio = 0.8  # High poisoning ratio for scaling attack
#         num_poison = int(len(y) * poison_ratio)
#         poison_indices = np.random.choice(len(y), num_poison, replace=False)
        
#         if self.target_class is not None:
#             y_poisoned[poison_indices] = self.target_class
#         else:
#             # Random label flipping
#             num_classes = len(np.unique(y))
#             for idx in poison_indices:
#                 y_poisoned[idx] = np.random.randint(0, num_classes)
        
#         print(f"[SCALING ATTACK] Prepared {num_poison}/{len(y)} poisoned samples")
#         print(f"[SCALING ATTACK] Note: Model update will be scaled by factor {self.scaling_factor}")
        
#         return X_poisoned, y_poisoned


# class BackdoorAttack:
#     """
#     Implementation of backdoor attack that embeds a trigger pattern
#     """
    
#     def __init__(self, target_class: int = 0, trigger_size: int = 3, poison_ratio: float = 0.1):
#         """
#         Initialize Backdoor Attack
        
#         Args:
#             target_class: Class that triggered samples should be classified as
#             trigger_size: Size of the trigger pattern
#             poison_ratio: Ratio of samples to add trigger to
#         """
#         self.target_class = target_class
#         self.trigger_size = trigger_size
#         self.poison_ratio = poison_ratio
    
#     def apply_attack(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Apply backdoor attack by embedding trigger pattern
#         """
#         print(f"[BACKDOOR ATTACK] Embedding trigger, target class: {self.target_class}")
        
#         # Copy data
#         X_poisoned = X.copy()
#         y_poisoned = y.copy()
        
#         # Select samples for backdoor
#         num_poison = int(len(y) * self.poison_ratio)
#         poison_indices = np.random.choice(len(y), num_poison, replace=False)
        
#         # Add trigger pattern and change labels
#         for idx in poison_indices:
#             X_poisoned[idx] = self._add_trigger(X_poisoned[idx])
#             y_poisoned[idx] = self.target_class
        
#         print(f"[BACKDOOR ATTACK] Added trigger to {num_poison}/{len(y)} samples ({self.poison_ratio*100:.1f}%)")
        
#         return X_poisoned, y_poisoned
    
#     def _add_trigger(self, sample: np.ndarray) -> np.ndarray:
#         """
#         Add trigger pattern to a sample
#         """
#         sample_with_trigger = sample.copy()
        
#         # Add small white square in bottom-right corner
#         if len(sample.shape) == 3:  # Image with channels
#             h, w, c = sample.shape
#             sample_with_trigger[-self.trigger_size:, -self.trigger_size:, :] = 1.0
#         elif len(sample.shape) == 2:  # Grayscale image
#             h, w = sample.shape
#             sample_with_trigger[-self.trigger_size:, -self.trigger_size:] = 1.0
        
#         return sample_with_trigger

# # Factory function to create attacks
# def create_attack(attack_type: str, **kwargs):
#     """
#     Factory function to create attack instances
    
#     Args:
#         attack_type: Type of attack ('fang', 'min_max', 'scaling')
#         **kwargs: Additional arguments for the attack
    
#     Returns:
#         Attack instance
#     """
#     if attack_type == 'fang':
#         return FangAttack(**kwargs)
#     elif attack_type == 'min_max':
#         return MinMaxAttack(**kwargs)
#     elif attack_type == 'scaling':
#         return ScalingAttack(**kwargs)
#     else:
#         raise ValueError(f"Unknown attack type: {attack_type}")

# # Example usage and testing
# if __name__ == "__main__":
#     # Create dummy data for testing
#     X_dummy = np.random.rand(1000, 28, 28, 1)  # MNIST-like data
#     y_dummy = np.random.randint(0, 10, 1000)   # 10 classes
    
#     print("Testing Strong Attacks Implementation")
#     print("=" * 50)
    
#     # Test Fang Attack
#     print("\n1. Testing Fang Attack:")
#     fang = FangAttack(attack_strength=2.0, target_class=7)
#     X_fang, y_fang = fang.apply_attack(X_dummy, y_dummy)
#     print(f"Original labels distribution: {np.bincount(y_dummy)}")
#     print(f"Fang attack labels distribution: {np.bincount(y_fang)}")
    
#     # Test Min-Max Attack
#     print("\n2. Testing Min-Max Attack:")
#     minmax = MinMaxAttack(target_class=1, attack_intensity=2.0)
#     X_minmax, y_minmax = minmax.apply_attack(X_dummy, y_dummy)
#     print(f"Original labels distribution: {np.bincount(y_dummy)}")
#     print(f"Min-Max attack labels distribution: {np.bincount(y_minmax)}")
    
#     # Test Scaling Attack
#     print("\n3. Testing Scaling Attack:")
#     scaling = ScalingAttack(scaling_factor=15.0)
#     X_scaling, y_scaling = scaling.apply_attack(X_dummy, y_dummy)
#     print(f"Original labels distribution: {np.bincount(y_dummy)}")
#     print(f"Scaling attack labels distribution: {np.bincount(y_scaling)}")
    
#     print("\n✅ All attacks tested successfully!")



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