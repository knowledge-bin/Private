

import os
import random
import numpy as np
from typing import Tuple, List, Dict, Optional, Union, Any
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import matplotlib.pyplot as plt
from pathlib import Path

class FederatedDataLoader:
    """
    A standardized federated learning data loader designed for PROFILE framework.
    
    Implements best practices from:
    - McMahan et al. (2017) Communication-Efficient Learning of Deep Networks from Decentralized Data
    - Bagdasaryan et al. (2020) How To Backdoor Federated Learning
    - Li et al. (2020) Federated Learning: Challenges, Methods, and Future Directions
    - Wang et al. (2020) Attack of the Tails: Yes, You Really Can Backdoor Federated Learning
    
    Args:
        dataset_name: Name of dataset ('mnist' or 'fashion_mnist')
        num_clients: Total number of clients in the federated setting
        iid: If True, data is distributed IID; if False, non-IID distribution is used
        alpha: Dirichlet concentration parameter for non-IID distribution (α→0 = more heterogeneous)
        partition_method: Method for non-IID partitioning ('dirichlet' or 'shard')
        seed: Random seed for reproducibility
        verbose: If True, prints detailed information about data distribution
    """
    
    def __init__(
        self, 
        dataset_name: str = 'mnist',
        num_clients: int = 10,
        iid: bool = True,
        alpha: float = 0.5,
        partition_method: str = 'dirichlet',
        seed: int = 42,
        verbose: bool = True
    ):
        # Validate input parameters - ADD more datasets
        if dataset_name not in ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'mnist', 'fashion_mnist', 'cifar10', or 'cifar100'")

        
        if partition_method not in ['dirichlet', 'shard']:
            raise ValueError(f"Unsupported partition method: {partition_method}. Choose 'dirichlet' or 'shard'")
        
        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.iid = iid
        self.alpha = alpha
        self.partition_method = partition_method
        self.seed = seed
        self.verbose = verbose
        
        # Set random seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Class labels
        # ADD class names for new datasets
        if dataset_name == 'mnist':
            self.class_names = [str(i) for i in range(10)]
        elif dataset_name == 'fashion_mnist':
            self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        elif dataset_name == 'cifar10':
            self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                               'dog', 'frog', 'horse', 'ship', 'truck']
        elif dataset_name == 'cifar100':
            self.class_names = [str(i) for i in range(100)]  # You can add detailed names if needed
        
        # Load the dataset
        self.data = self._load_dataset()
        
        # Initialize client data partitions
        self.client_partitions = None

    def _load_dataset(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Load the dataset and perform basic preprocessing"""
        if self.verbose:
            print(f"Loading {self.dataset_name} dataset...")
            
        # ADD loading for new datasets
        if self.dataset_name == 'mnist':
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
        elif self.dataset_name == 'fashion_mnist':
            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        elif self.dataset_name == 'cifar10':
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()
            y_train = y_train.squeeze()  # Remove extra dimension
            y_test = y_test.squeeze()
        elif self.dataset_name == 'cifar100':
            (X_train, y_train), (X_test, y_test) = cifar100.load_data()
            y_train = y_train.squeeze()
            y_test = y_test.squeeze()
        
        # Preprocess data
        X_train = self._preprocess_images(X_train)
        X_test = self._preprocess_images(X_test)
        
        # Split training data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=self.seed
        )
        
        if self.verbose:
            print(f"Dataset loaded: {X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test samples")
            print(f"Image shape: {X_train.shape[1:]}")
            
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }

    def _preprocess_images(self, X: np.ndarray) -> np.ndarray:
        """Standardized image preprocessing for neural networks"""
        # Normalize pixel values to [0, 1]
        X = X.astype('float32') / 255.0
        
        # Add channel dimension if not present
        if len(X.shape) == 3:  # (samples, height, width)
            X = np.expand_dims(X, -1)  # (samples, height, width, channels)
            
        return X

    def partition_data(self) -> None:
        """
        Partition data among clients according to specified distribution method.
        
        This creates and stores client partitions that can be accessed later.
        """
        X_train, y_train = self.data['train']
        
        if self.iid:
            if self.verbose:
                print("Creating IID data partitions...")
            client_partitions = self._create_iid_partitions(X_train, y_train)
        else:
            if self.verbose:
                print(f"Creating non-IID data partitions using {self.partition_method} method...")
            
            if self.partition_method == 'dirichlet':
                client_partitions = self._create_dirichlet_partitions(X_train, y_train)
            elif self.partition_method == 'shard':
                client_partitions = self._create_shard_partitions(X_train, y_train)
        
        self.client_partitions = client_partitions
        
        if self.verbose:
            self._print_partition_statistics()

    def _create_iid_partitions(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Dict[str, np.ndarray]]:
        """Create IID (Independent and Identically Distributed) partitions"""
        # Shuffle data
        indices = np.random.permutation(len(y))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Split into roughly equal partitions
        X_splits = np.array_split(X_shuffled, self.num_clients)
        y_splits = np.array_split(y_shuffled, self.num_clients)
        
        # Create list of client partitions
        client_partitions = []
        for i in range(self.num_clients):
            client_partitions.append({
                'X': X_splits[i],
                'y': y_splits[i]
            })
            
        return client_partitions

    def _create_dirichlet_partitions(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Dict[str, np.ndarray]]:
        """
        Create non-IID partitions using Dirichlet distribution.
        
        Implementation follows Yurochkin et al. (2019) and Li et al. (2020).
        The Dirichlet distribution creates realistic heterogeneity among clients.
        """
        # Get number of classes
        num_classes = len(np.unique(y))
        
        # Initialize client partitions
        client_partitions = [{
            'X': [],
            'y': []
        } for _ in range(self.num_clients)]
        
        # Group indices by class
        class_indices = [np.where(y == c)[0] for c in range(num_classes)]
        
        # For each class, distribute samples according to Dirichlet distribution
        for c, indices in enumerate(class_indices):
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
            
            # Calculate number of samples per client
            # Note: We use proportions * len(indices) to determine sample count
            num_samples_per_client = (proportions * len(indices)).astype(int)
            
            # Adjust to ensure all samples are allocated
            num_samples_per_client[-1] = len(indices) - np.sum(num_samples_per_client[:-1])
            
            # Shuffle indices
            indices = np.random.permutation(indices)
            
            # Distribute indices to clients
            start_idx = 0
            for client_idx, num_samples in enumerate(num_samples_per_client):
                if num_samples > 0:
                    client_indices = indices[start_idx:start_idx + num_samples]
                    client_partitions[client_idx]['X'].extend(X[client_indices].tolist())
                    client_partitions[client_idx]['y'].extend(y[client_indices].tolist())
                    start_idx += num_samples
        
        # Convert lists back to arrays
        for i in range(self.num_clients):
            if len(client_partitions[i]['X']) > 0:
                client_partitions[i]['X'] = np.array(client_partitions[i]['X'])
                client_partitions[i]['y'] = np.array(client_partitions[i]['y'])
            else:
                # Fallback for any empty partitions (should be rare)
                fallback_indices = np.random.choice(len(X), 10, replace=False)
                client_partitions[i]['X'] = X[fallback_indices]
                client_partitions[i]['y'] = y[fallback_indices]
            
        return client_partitions

    def _create_shard_partitions(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Dict[str, np.ndarray]]:
        """
        Create non-IID partitions using the shard method.
        
        Implementation follows McMahan et al. (2017) and Hsu et al. (2019).
        Each client gets shards of data sorted by class labels.
        """
        # Sort data by class
        sorted_indices = np.argsort(y)
        sorted_X = X[sorted_indices]
        sorted_y = y[sorted_indices]
        
        # Determine number of shards (2 shards per client is common)
        shards_per_client = 2
        num_shards = self.num_clients * shards_per_client
        samples_per_shard = len(sorted_X) // num_shards
        
        # Create shards
        X_shards = np.array_split(sorted_X, num_shards)
        y_shards = np.array_split(sorted_y, num_shards)
        
        # Randomly assign shards to clients
        shard_indices = list(range(num_shards))
        np.random.shuffle(shard_indices)
        
        # Assign shards to clients
        client_partitions = []
        for i in range(self.num_clients):
            # Get indices for this client's shards
            client_shard_indices = shard_indices[i * shards_per_client:(i + 1) * shards_per_client]
            
            # Combine shards
            client_X = np.concatenate([X_shards[idx] for idx in client_shard_indices])
            client_y = np.concatenate([y_shards[idx] for idx in client_shard_indices])
            
            client_partitions.append({
                'X': client_X,
                'y': client_y
            })
            
        return client_partitions

    def _print_partition_statistics(self) -> None:
        """Print statistics about the data partitions"""
        if self.client_partitions is None:
            print("No data partitions available. Call partition_data() first.")
            return
        
        print("\nClient Data Distribution Statistics:")
        print("-" * 40)
        
        total_samples = 0
        class_distribution = np.zeros((self.num_clients, len(self.class_names)))
        
        for i, partition in enumerate(self.client_partitions):
            num_samples = len(partition['y'])
            total_samples += num_samples
            
            # Calculate class distribution
            for c in range(len(self.class_names)):
                class_count = np.sum(partition['y'] == c)
                class_distribution[i, c] = class_count
                
            # Print client statistics
            print(f"Client {i}: {num_samples} samples")
            if self.verbose:
                for c in range(len(self.class_names)):
                    percentage = 100 * class_distribution[i, c] / num_samples if num_samples > 0 else 0
                    print(f"  Class {c} ({self.class_names[c]}): {int(class_distribution[i, c])} samples ({percentage:.1f}%)")
        
        print("-" * 40)
        print(f"Total samples across all clients: {total_samples}")
        
        # Calculate and print heterogeneity metrics
        if not self.iid:
            self._print_heterogeneity_metrics(class_distribution)
    
    def _print_heterogeneity_metrics(self, class_distribution: np.ndarray) -> None:
        """Calculate and print metrics quantifying data heterogeneity"""
        # Calculate KL divergence between each client's distribution and the global distribution
        global_distribution = np.sum(class_distribution, axis=0)
        global_distribution = global_distribution / np.sum(global_distribution)
        
        kl_divergences = []
        for i in range(self.num_clients):
            client_dist = class_distribution[i] / np.sum(class_distribution[i])
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            client_dist = np.maximum(client_dist, epsilon)
            global_dist = np.maximum(global_distribution, epsilon)
            
            # Calculate KL divergence: sum(p_i * log(p_i / q_i))
            kl = np.sum(client_dist * np.log(client_dist / global_dist))
            kl_divergences.append(kl)
        
        print("\nHeterogeneity Metrics:")
        print(f"Average KL divergence: {np.mean(kl_divergences):.4f}")
        print(f"Max KL divergence: {np.max(kl_divergences):.4f}")
        print(f"Min KL divergence: {np.min(kl_divergences):.4f}")

    def get_client_data(
        self, client_id: int
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Get data for a specific client.
        
        Args:
            client_id: The client ID (0 to num_clients-1)
            
        Returns:
            Tuple containing (train_data, val_data, test_data), where each is a tuple of (X, y)
        """
        if self.client_partitions is None:
            self.partition_data()
            
        if client_id < 0 or client_id >= self.num_clients:
            raise ValueError(f"Invalid client_id: {client_id}. Must be between 0 and {self.num_clients-1}")
            
        # Get client's training data
        X_train = self.client_partitions[client_id]['X']
        y_train = self.client_partitions[client_id]['y']
        
        # Validation and test data are the same for all clients
        X_val, y_val = self.data['val']
        X_test, y_test = self.data['test']
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


    def apply_poisoning(
        self,
        X: np.ndarray,
        y: np.ndarray,
        attack_type: str = 'label_flip',
        poison_ratio: float = 0.1,
        target_class: Optional[int] = None,
        source_class: Optional[int] = None,
        trigger: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply poisoning attack to the dataset.
        """
        # Make copies to avoid modifying original data
        X_poisoned = X.copy()
        y_poisoned = y.copy()
        
        # Determine which samples to poison
        num_samples = len(y)
        num_poison_samples = int(num_samples * poison_ratio)
        
        # Filter by source class if specified
        if source_class is not None:
            source_indices = np.where(y == source_class)[0]
            if len(source_indices) == 0:
                print(f"Warning: No samples of class {source_class} found. No poisoning applied.")
                return X, y
            num_poison_samples = min(num_poison_samples, len(source_indices))
            poison_indices = np.random.choice(source_indices, num_poison_samples, replace=False)
        else:
            poison_indices = np.random.choice(num_samples, num_poison_samples, replace=False)
        
        # Apply poisoning based on attack type
        if attack_type == 'label_flip':
            y_poisoned[poison_indices] = self._apply_label_flip(
                y_poisoned[poison_indices], 
                target_class
            )
            
        elif attack_type == 'random':
            y_poisoned[poison_indices] = self._apply_random_labels(
                y_poisoned[poison_indices]
            )
            
        elif attack_type == 'targeted':
            if target_class is None:
                raise ValueError("target_class must be specified for targeted attacks")
            y_poisoned[poison_indices] = target_class
            
        elif attack_type == 'backdoor':
            if target_class is None:
                raise ValueError("target_class must be specified for backdoor attacks")
                
            # Apply trigger pattern
            X_poisoned[poison_indices] = self._apply_backdoor_trigger(
                X_poisoned[poison_indices], 
                trigger
            )
            # Change labels to target class
            y_poisoned[poison_indices] = target_class
        
        # ADD THESE NEW ATTACK TYPES:
        elif attack_type == 'fang':
            print(f"[FANG ATTACK] Applying to {num_poison_samples} samples")
            # Aggressive label flipping + noise
            if target_class is not None:
                y_poisoned[poison_indices] = target_class
            else:
                y_poisoned[poison_indices] = 1  # Default target
            
            # Add subtle noise to make attack stronger
            noise_strength = 0.1
            for idx in poison_indices:
                noise = np.random.normal(0, noise_strength, X_poisoned[idx].shape)
                X_poisoned[idx] = np.clip(X_poisoned[idx] + noise, 0, 1)
                
        elif attack_type == 'min_max':
            print(f"[MIN-MAX ATTACK] Applying to {num_poison_samples} samples")
            # High poison ratio attack
            if target_class is not None:
                y_poisoned[poison_indices] = target_class
            else:
                y_poisoned[poison_indices] = 1  # Default target
            
            # Add pattern perturbation
            for idx in poison_indices:
                if len(X_poisoned[idx].shape) >= 2:
                    h, w = X_poisoned[idx].shape[:2]
                    # Add diagonal pattern
                    for i in range(0, min(h, w), 3):
                        if len(X_poisoned[idx].shape) == 3:
                            X_poisoned[idx][i, i, :] = np.clip(X_poisoned[idx][i, i, :] + 0.2, 0, 1)
                        else:
                            X_poisoned[idx][i, i] = np.clip(X_poisoned[idx][i, i] + 0.2, 0, 1)
            
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")
        
        # Print attack statistics (rest of your existing code)
        self._print_attack_statistics(
            attack_type, 
            num_poison_samples, 
            num_samples, 
            y, 
            y_poisoned, 
            poison_indices,
            target_class
        )
        
        return X_poisoned, y_poisoned


    def _apply_label_flip(
        self, y: np.ndarray, target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply label flipping attack.
        
        Args:
            y: Original labels
            target_class: If provided, flip all labels to this class
                         If None, flip each label to the next class circularly
                         
        Returns:
            Modified labels
        """
        num_classes = len(self.class_names)
        
        if target_class is not None:
            # Flip all labels to target class
            return np.full_like(y, target_class)
        else:
            # Flip each label to the next class (cyclically)
            return (y + 1) % num_classes

    def _apply_random_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Apply random label attack.
        
        Args:
            y: Original labels
            
        Returns:
            Modified labels
        """
        num_classes = len(self.class_names)
        y_poisoned = y.copy()
        
        for i in range(len(y)):
            # Choose a random class different from the original
            possible_classes = [c for c in range(num_classes) if c != y[i]]
            y_poisoned[i] = np.random.choice(possible_classes)
            
        return y_poisoned

    def _apply_backdoor_trigger(
        self, X: np.ndarray, custom_trigger: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply backdoor trigger pattern to images.
        
        Args:
            X: Input images
            custom_trigger: Custom trigger pattern (if None, use default)
            
        Returns:
            Modified images with trigger
        """
        X_triggered = X.copy()
        
        if custom_trigger is not None:
            trigger = custom_trigger
        else:
            # Create default trigger (small white square in corner)
            trigger_size = 3
            h, w = X.shape[1:3]
            trigger = np.zeros_like(X[0])
            trigger[-trigger_size:, -trigger_size:, :] = 1.0
        
        # Apply trigger
        X_triggered = np.clip(X_triggered + trigger, 0, 1)
        
        return X_triggered

    def _print_attack_statistics(
        self,
        attack_type: str,
        num_poison_samples: int,
        num_samples: int,
        y_original: np.ndarray,
        y_poisoned: np.ndarray,
        poison_indices: np.ndarray,
        target_class: Optional[int] = None
    ) -> None:
        """Print statistics about the applied poisoning attack"""
        print("\nPoisoning Attack Statistics:")
        print(f"Attack type: {attack_type}")
        print(f"Poisoned samples: {num_poison_samples}/{num_samples} ({100*num_poison_samples/num_samples:.1f}%)")
        
        if attack_type == 'targeted' or (attack_type == 'label_flip' and target_class is not None):
            print(f"Target class: {target_class} ({self.class_names[target_class]})")
            
        # Calculate class transition matrix (for label attacks)
        if attack_type in ['label_flip', 'random', 'targeted']:
            transitions = {}
            for idx in poison_indices:
                original = y_original[idx]
                poisoned = y_poisoned[idx]
                if original != poisoned:
                    key = (int(original), int(poisoned))
                    transitions[key] = transitions.get(key, 0) + 1
            
            # Print transitions
            print("\nLabel transitions:")
            for (src, dst), count in sorted(transitions.items()):
                src_name = self.class_names[src]
                dst_name = self.class_names[dst]
                print(f"  Class {src} ({src_name}) → Class {dst} ({dst_name}): {count} samples")

    def visualize_client_distribution(
        self, 
        figsize: Tuple[int, int] = (12, 8), 
        output_path: Optional[str] = None
    ) -> None:
        """
        Visualize the class distribution across clients.
        
        Args:
            figsize: Figure size
            output_path: If provided, save the figure to this path
        """
        if self.client_partitions is None:
            print("No data partitions available. Call partition_data() first.")
            return
        
        num_classes = len(self.class_names)
        
        # Calculate class distribution for each client
        class_distribution = np.zeros((self.num_clients, num_classes))
        for i, partition in enumerate(self.client_partitions):
            for c in range(num_classes):
                class_distribution[i, c] = np.sum(partition['y'] == c)
        
        # Normalize distribution (percentage)
        row_sums = class_distribution.sum(axis=1, keepdims=True)
        normalized_distribution = 100 * class_distribution / row_sums
        
        # Plot
        plt.figure(figsize=figsize)
        
        # Heatmap
        plt.imshow(normalized_distribution, cmap='YlGnBu', aspect='auto')
        plt.colorbar(label='Percentage of client data')
        
        # Labels
        plt.xlabel('Class')
        plt.ylabel('Client ID')
        plt.title('Class Distribution Across Clients')
        
        # Customize x-axis ticks
        plt.xticks(range(num_classes), self.class_names, rotation=45, ha='right')
        plt.yticks(range(self.num_clients), [f'Client {i}' for i in range(self.num_clients)])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {output_path}")
        
        plt.show()

    def get_validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get validation data (same for all clients)"""
        return self.data['val']
    
    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get test data (same for all clients)"""
        return self.data['test']


class FederatedPoisoningExperiment:
    """
    A class to facilitate poisoning attack experiments in federated learning.
    
    This class simplifies the setup and execution of poisoning experiments
    by providing a clean interface for:
    - Configuring which clients are malicious
    - Setting up different attack types
    - Controlling attack timing (e.g., after convergence)
    - Managing attack parameters
    
    Args:
        num_clients: Total number of federated learning clients
        malicious_client_ids: List of client IDs that will perform attacks
        attack_config: Configuration parameters for the attack
        start_round: Training round to start attack (0 = from beginning)
        end_round: Training round to end attack (None = continue to end)
    """
    
    def __init__(
        self,
        num_clients: int = 10,
        malicious_client_ids: Optional[List[int]] = None,
        attack_config: Optional[Dict[str, Any]] = None,
        start_round: int = 0,
        end_round: Optional[int] = None
    ):
        self.num_clients = num_clients
        
        # Set default malicious clients if not specified (10% of clients)
        if malicious_client_ids is None:
            num_malicious = max(1, int(0.1 * num_clients))
            self.malicious_client_ids = list(range(num_malicious))
        else:
            self.malicious_client_ids = malicious_client_ids
            
        # Default attack configuration
        default_attack_config = {
            'type': 'label_flip',
            'poison_ratio': 0.5,
            'target_class': None,
            'source_class': None
        }
        
        # Update with user-provided config
        if attack_config:
            default_attack_config.update(attack_config)
            
        self.attack_config = default_attack_config
        self.start_round = start_round
        self.end_round = end_round
        self.current_round = 0
        
        print("Federated Poisoning Experiment Configured:")
        print(f"Total clients: {num_clients}")
        print(f"Malicious clients: {self.malicious_client_ids} ({len(self.malicious_client_ids)}/{num_clients})")
        print(f"Attack type: {self.attack_config['type']}")
        print(f"Poison ratio: {self.attack_config['poison_ratio']}")
        print(f"Attack timing: Round {start_round} to {end_round if end_round is not None else 'end'}")
    
    def should_poison(self, client_id: int, round_idx: int) -> bool:
        """
        Determine if a client should poison its data in the current round.
        
        Args:
            client_id: ID of the client
            round_idx: Current training round
            
        Returns:
            True if the client should poison its data, False otherwise
        """
        self.current_round = round_idx
        
        # Check if client is malicious
        if client_id not in self.malicious_client_ids:
            return False
            
        # Check if we're in the attack window
        if self.start_round <= round_idx and (self.end_round is None or round_idx <= self.end_round):
            return True
            
        return False
    
    def get_attack_config(self) -> Dict[str, Any]:
        """Get the current attack configuration"""
        return self.attack_config
    
    def update_attack_config(self, **kwargs) -> None:
        """
        Update attack configuration parameters.
        
        This allows dynamically changing attack parameters during the experiment.
        """
        self.attack_config.update(kwargs)
        print(f"Attack configuration updated at round {self.current_round}:")
        for key, value in kwargs.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    # Initialize data loader
    loader = FederatedDataLoader(
        dataset_name='mnist',
        num_clients=10,
        iid=False,
        alpha=0.5,
        partition_method='dirichlet',
        seed=42
    )
    
    # Partition data
    loader.partition_data()
    
    # Visualize client distribution
    loader.visualize_client_distribution()
    
    # Get data for client 0
    client_id = 0
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.get_client_data(client_id)
    
    # Configure poisoning experiment
    experiment = FederatedPoisoningExperiment(
        num_clients=6,
        malicious_client_ids=[3, 7],
        attack_config={
            'attack_type': 'label_flip',
            'poison_ratio': 0.3,
            'target_class': 7
        },
        start_round=5,
        end_round=10
    )
    
    # In PROFILE client's fit method:
    round_idx = 6  # Current round
    if experiment.should_poison(client_id, round_idx):
        attack_config = experiment.get_attack_config()
        X_train, y_train = loader.apply_poisoning(X_train, y_train, **attack_config)
        print(f"Client {client_id} is poisoning data in round {round_idx}")
    else:
        print(f"Client {client_id} is using clean data in round {round_idx}")