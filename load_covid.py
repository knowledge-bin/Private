


# # # Standard Libraries
# # import os
# # import random
# # from pathlib import Path

# # # Third Party
# # import numpy as np
# # import progressbar
# # from skimage.transform import resize
# # from sklearn.model_selection import train_test_split
# # from tensorflow.keras.datasets import mnist
# # from tensorflow.keras.utils import to_categorical
# # from matplotlib import image as img


# # def load_dataset(dataset_name, num_clients=1, num_evaluators=1, client_id=None, evaluator_id=None, iid=True, alpha=0.5, limit=None, poison_type=None, poison_ratio=0.1, target_label=None):
# #     """
# #     Universal data loader function that loads and preprocesses various datasets.
# #     Supports different datasets like 'mnist' and 'covid'.
# #     Includes options for data poisoning, IID/non-IID distribution, and evaluator support.
    
# #     Parameters:
# #         dataset_name (str): Name of the dataset to load ('mnist', 'covid').
# #         num_clients (int): Total number of clients participating in the federated learning.
# #         num_evaluators (int): Total number of evaluators.
# #         client_id (int, optional): ID of the current client (0 to num_clients-1).
# #         evaluator_id (int, optional): ID of the current evaluator (0 to num_evaluators-1).
# #         iid (bool): Whether to use IID (True) or non-IID (False) data distribution.
# #         alpha (float): Concentration parameter for Dirichlet distribution (used for non-IID).
# #         limit (int, optional): Limit the number of samples loaded. Default is None.
# #         poison_type (str, optional): Type of poisoning attack ('label_flip', 'random_label', 'targeted').
# #         poison_ratio (float): Ratio of data to be poisoned (0 to 1). Default is 0.1.
# #         target_label (int, optional): Target label for targeted attack. Required if poison_type is 'targeted'.

# #     Returns:
# #         Tuple of Numpy arrays: (X_train, y_train), (X_val, y_val), (X_test, y_test)
# #     """
# #     if dataset_name == 'mnist':
# #         (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()
        
# #         # Preprocess data
# #         X_train_full = X_train_full.astype("float32") / 255.0
# #         X_test_full = X_test_full.astype("float32") / 255.0
        
# #         # Expand dimensions
# #         X_train_full = np.expand_dims(X_train_full, -1)
# #         X_test_full = np.expand_dims(X_test_full, -1)
# #     elif dataset_name == 'covid':
# #         (X_train_full, y_train_full), (X_test_full, y_test_full) = load_covid_data(limit)
# #     else:
# #         raise ValueError("Unsupported dataset. Please choose 'mnist' or 'covid'.")

# #     # Split training data into train and validation
# #     X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)

# #     if client_id is not None:
# #         # Distribute training data to clients
# #         if iid:
# #             client_data = create_iid_partition((X_train, y_train), num_clients)
# #         else:
# #             client_data = create_non_iid_partition((X_train, y_train), num_clients, alpha)
        
# #         if 0 <= client_id < num_clients:
# #             X_train, y_train = client_data[client_id]
            
# #             # Apply poisoning if specified
# #             if poison_type:
# #                 X_train, y_train = apply_poisoning(X_train, y_train, poison_type, poison_ratio, target_label)
            
# #             return (X_train, y_train), (X_val, y_val), None
# #         else:
# #             return None
    
# #     elif evaluator_id is not None:
# #         # Distribute test data to evaluators
# #         if iid:
# #             evaluator_data = create_iid_partition((X_test_full, y_test_full), num_evaluators)
# #         else:
# #             evaluator_data = create_non_iid_partition((X_test_full, y_test_full), num_evaluators, alpha)
        
# #         if 0 <= evaluator_id < num_evaluators:
# #             return None, None, evaluator_data[evaluator_id]
# #         else:
# #             return None
    
# #     else:
# #         # Return full dataset for server
# #         return (X_train, y_train), (X_val, y_val), (X_test_full, y_test_full)


# # def apply_poisoning(X, y, poison_type, poison_ratio, target_label=None):
# #     """
# #     Apply poisoning attack to the dataset.
    
# #     Parameters:
# #         X (numpy.ndarray): Input features.
# #         y (numpy.ndarray): Labels.
# #         poison_type (str): Type of poisoning attack ('label_flip', 'random_label', 'targeted').
# #         poison_ratio (float): Ratio of data to be poisoned (0 to 1).
# #         target_label (int, optional): Target label for targeted attack.

# #     Returns:
# #         Tuple of numpy.ndarray: Poisoned X and y
# #     """
# #     num_samples = len(y)
# #     num_poison_samples = int(num_samples * poison_ratio)
# #     poison_indices = np.random.choice(num_samples, num_poison_samples, replace=False)

# #     if poison_type == 'label_flip':
# #         y[poison_indices] = label_flip_attack(y[poison_indices])
# #     elif poison_type == 'random_label':
# #         y[poison_indices] = random_label_attack(y[poison_indices])
# #     elif poison_type == 'targeted':
# #         if target_label is None:
# #             raise ValueError("Target label must be specified for targeted attack.")
# #         y[poison_indices] = targeted_attack(y[poison_indices], target_label)
# #     else:
# #         raise ValueError("Invalid poison type. Choose 'label_flip', 'random_label', or 'targeted'.")

# #     return X, y

# # # def label_flip_attack(labels):
# # #     """
# # #     Implement label flipping attack.
# # #     For binary classification, it flips 0 to 1 and 1 to 0.
# # #     For multi-class, it rotates labels (e.g., 0->1, 1->2, ..., 9->0 for MNIST).
# # #     """
# # #     num_classes = len(np.unique(labels))
# # #     if num_classes == 2:
# # #         return 1 - labels
# # #     else:
# # #         return (labels + 1) % num_classes

# # def label_flip_attack(labels):
# #     """
# #     Implement label flipping attack for CIFAR-10 one-hot encoded labels.
# #     """
# #     num_classes = 10  # CIFAR-10 specific
# #     new_labels = np.zeros_like(labels)
    
# #     # Keep track of flips for monitoring
# #     flip_count = {(i, j): 0 for i in range(num_classes) for j in range(num_classes) if i != j}
    
# #     for i in range(len(labels)):
# #         current_class = np.argmax(labels[i])
# #         # Randomly select a new class different from the current one
# #         new_class = np.random.choice([j for j in range(num_classes) if j != current_class])
# #         new_labels[i, new_class] = 1
# #         flip_count[(current_class, new_class)] += 1
    
# #     # Print flip statistics
# #     print("\nLabel Flip Statistics:")
# #     for (old, new), count in flip_count.items():
# #         if count > 0:
# #             print(f"Class {old} → Class {new}: {count} samples")
    
# #     return new_labels

# # def random_label_attack(labels):
# #     """
# #     Implement random label attack.
# #     Randomly assigns a new label to each sample, ensuring it's different from the original.
# #     """
# #     num_classes = len(np.unique(labels))
# #     new_labels = np.array([np.random.choice([l for l in range(num_classes) if l != label]) for label in labels])
# #     return new_labels

# # def targeted_attack(labels, target_label):
# #     """
# #     Implement targeted attack.
# #     Changes all labels to the specified target label.
# #     """
# #     return np.full_like(labels, target_label)



# # def load_covid_data(limit):
# #     """
# #     Loads COVID and non-COVID images, resizes them, and splits them into training and testing sets.
# #     """
# #     script_dir = os.path.dirname(os.path.realpath(__file__))
# #     covid_path = os.path.join(script_dir, "data", "covid")
# #     non_covid_path = os.path.join(script_dir, "data", "noncovid")
    
# #     if not os.path.exists(covid_path) or not os.path.exists(non_covid_path):
# #         raise FileNotFoundError("COVID-19 data directories not found. Please check your paths.")

# #     covid_images = [os.path.join(covid_path, f) for f in os.listdir(covid_path) if f.endswith('.png')]
# #     non_covid_images = [os.path.join(non_covid_path, f) for f in os.listdir(non_covid_path) if f.endswith('.png')]

# #     # Randomly sample "limit" number of images
# #     if limit is not None:
# #         covid_images = random.sample(covid_images, limit)
# #         non_covid_images = random.sample(non_covid_images, limit)

# #     IMG_SIZE = 128
# #     covid_npy = np.empty((len(covid_images), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
# #     non_covid_npy = np.empty((len(non_covid_images), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

# #     bar = progressbar.ProgressBar(maxval=len(covid_images) + len(non_covid_images), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
# #     bar.start()

# #     for i, _file in enumerate(covid_images):
# #         try:
# #             image_npy = img.imread(_file)
# #             resized_img = resize(image_npy, (IMG_SIZE, IMG_SIZE, 1), anti_aliasing=True)
# #             covid_npy[i] = resized_img
# #         except Exception as e:
# #             print(f"Error processing {_file}: {e}")

# #     for i, _file in enumerate(non_covid_images):
# #         try:
# #             image_npy = img.imread(_file)
# #             resized_img = resize(image_npy, (IMG_SIZE, IMG_SIZE, 1), anti_aliasing=True)
# #             non_covid_npy[i] = resized_img
# #         except Exception as e:
# #             print(f"Error processing {_file}: {e}")

# #     bar.finish()

# #     X = np.concatenate([covid_npy, non_covid_npy])
# #     y = np.concatenate([np.ones(len(covid_npy)), np.zeros(len(non_covid_npy))])

# #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
# #     validation_split = 0.1
# #     val_size = int(len(X_train) * validation_split)
# #     X_val, y_val = X_train[:val_size], y_train[:val_size]
# #     X_train, y_train = X_train[val_size:], y_train[val_size:]

# #     return (X_train, y_train), (X_val, y_val), (X_test, y_test)



# # def load_mnist_data(num_clients=None, client_id=None):
# #     """
# #     Loads and preprocesses the MNIST dataset.
# #     If num_clients and client_id are None, it returns the full dataset for server-side evaluation.
# #     Otherwise, it distributes the data among clients.
# #     """
# #     (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()
    
# #     # Preprocess data
# #     X_train_full = X_train_full.astype("float32") / 255.0
# #     X_test_full = X_test_full.astype("float32") / 255.0
    
# #     # Expand dimensions
# #     X_train_full = np.expand_dims(X_train_full, -1)
# #     X_test_full = np.expand_dims(X_test_full, -1)

# #     if num_clients is None or client_id is None:
# #         # Server-side: return full dataset
# #         print("Server - Full dataset loaded")
# #         print(f"X_train shape: {X_train_full.shape}, y_train shape: {y_train_full.shape}")
# #         print(f"X_test shape: {X_test_full.shape}, y_test shape: {y_test_full.shape}")
# #         return (X_train_full, y_train_full), (X_test_full, y_test_full), (X_test_full, y_test_full)
    
# #     # Client-side: distribute data
# #     samples_per_client = len(X_train_full) // num_clients
# #     start_idx = (client_id % num_clients) * samples_per_client
# #     end_idx = start_idx + samples_per_client

# #     X_train_client = X_train_full[start_idx:end_idx]
# #     y_train_client = y_train_full[start_idx:end_idx]

# #     # Use a portion of the test set for this client
# #     test_samples_per_client = len(X_test_full) // num_clients
# #     start_idx = (client_id % num_clients) * test_samples_per_client
# #     end_idx = start_idx + test_samples_per_client

# #     X_test_client = X_test_full[start_idx:end_idx]
# #     y_test_client = y_test_full[start_idx:end_idx]

# #     # Create a small validation set from the client's training data
# #     X_train_client, X_val_client, y_train_client, y_val_client = train_test_split(
# #         X_train_client, y_train_client, test_size=0.1, random_state=42)

# #     print(f"Client {client_id} - X_train shape: {X_train_client.shape}, y_train shape: {y_train_client.shape}")
# #     print(f"Client {client_id} - X_val shape: {X_val_client.shape}, y_val shape: {y_val_client.shape}")
# #     print(f"Client {client_id} - X_test shape: {X_test_client.shape}, y_test shape: {y_test_client.shape}")

# #     return (X_train_client, y_train_client), (X_val_client, y_val_client), (X_test_client, y_test_client)

# # def load_fashion_mnist_data(num_clients=None, client_id=None):
# #     """
# #     Loads and preprocesses the Fashion MNIST dataset.
# #     If num_clients and client_id are None, it returns the full dataset for server-side evaluation.
# #     Otherwise, it distributes the data among clients.
# #     """
# #     (X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()
    
# #     # Preprocess data
# #     X_train_full = X_train_full.astype("float32") / 255.0
# #     X_test_full = X_test_full.astype("float32") / 255.0
    
# #     # Expand dimensions
# #     X_train_full = np.expand_dims(X_train_full, -1)
# #     X_test_full = np.expand_dims(X_test_full, -1)

# #     # Print class distribution
# #     print("\nClass Distribution:")
# #     class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
# #                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# #     for i, class_name in enumerate(class_names):
# #         train_count = np.sum(y_train_full == i)
# #         test_count = np.sum(y_test_full == i)
# #         print(f"Class {i} ({class_name}): {train_count} train, {test_count} test")

# #     if num_clients is None or client_id is None:
# #         # Server-side: return full dataset
# #         print("\nServer - Full dataset loaded")
# #         print(f"X_train shape: {X_train_full.shape}, y_train shape: {y_train_full.shape}")
# #         print(f"X_test shape: {X_test_full.shape}, y_test shape: {y_test_full.shape}")
# #         return (X_train_full, y_train_full), (X_test_full, y_test_full), (X_test_full, y_test_full)
    
# #     # Client-side: distribute data
# #     samples_per_client = len(X_train_full) // num_clients
# #     start_idx = (client_id % num_clients) * samples_per_client
# #     end_idx = start_idx + samples_per_client

# #     X_train_client = X_train_full[start_idx:end_idx]
# #     y_train_client = y_train_full[start_idx:end_idx]

# #     # Use a portion of the test set for this client
# #     test_samples_per_client = len(X_test_full) // num_clients
# #     start_idx = (client_id % num_clients) * test_samples_per_client
# #     end_idx = start_idx + test_samples_per_client

# #     X_test_client = X_test_full[start_idx:end_idx]
# #     y_test_client = y_test_full[start_idx:end_idx]

# #     # Create a small validation set from the client's training data
# #     X_train_client, X_val_client, y_train_client, y_val_client = train_test_split(
# #         X_train_client, y_train_client, test_size=0.1, random_state=42)

# #     print(f"\nClient {client_id} Data Distribution:")
# #     for i, class_name in enumerate(class_names):
# #         train_count = np.sum(y_train_client == i)
# #         val_count = np.sum(y_val_client == i)
# #         test_count = np.sum(y_test_client == i)
# #         print(f"Class {i} ({class_name}): {train_count} train, {val_count} val, {test_count} test")

# #     print(f"\nClient {client_id} - X_train shape: {X_train_client.shape}, y_train shape: {y_train_client.shape}")
# #     print(f"Client {client_id} - X_val shape: {X_val_client.shape}, y_val shape: {y_val_client.shape}")
# #     print(f"Client {client_id} - X_test shape: {X_test_client.shape}, y_test shape: {y_test_client.shape}")

# #     return (X_train_client, y_train_client), (X_val_client, y_val_client), (X_test_client, y_test_client)


# # def load_mnist_data_non_iid(num_clients=None, client_id=None):
# #     """
# #     Loads and preprocesses the MNIST dataset with non-IID distribution using sharding.
# #     If num_clients and client_id are None, it returns the full dataset for server-side evaluation.
# #     Otherwise, it distributes the data among clients in a non-IID manner.
    
# #     :param num_clients: Total number of clients
# #     :param client_id: ID of the current client (0 to num_clients-1)
# #     """
# #     (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()
    
# #     # Preprocess data
# #     X_train_full = X_train_full.astype("float32") / 255.0
# #     X_test_full = X_test_full.astype("float32") / 255.0
    
# #     # Expand dimensions
# #     X_train_full = np.expand_dims(X_train_full, -1)
# #     X_test_full = np.expand_dims(X_test_full, -1)

# #     if num_clients is None or client_id is None:
# #         # Server-side: return full dataset
# #         print("Server - Full dataset loaded")
# #         print(f"X_train shape: {X_train_full.shape}, y_train shape: {y_train_full.shape}")
# #         print(f"X_test shape: {X_test_full.shape}, y_test shape: {y_test_full.shape}")
# #         return (X_train_full, y_train_full), (X_test_full, y_test_full), (X_test_full, y_test_full)
    
# #     # Client-side: distribute data in a non-IID manner
# #     num_shards = num_clients * 2
# #     num_imgs = len(X_train_full) // num_shards
    
# #     # Sort the data by digit
# #     idx_shard = [i for i in range(num_shards)]
# #     dict_users = {i: np.array([], dtype='int64') for i in range(num_clients)}
# #     idxs = np.arange(len(X_train_full))
# #     labels = y_train_full.flatten()

# #     # Sort the indices by label
# #     idxs_labels = np.vstack((idxs, labels))
# #     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
# #     idxs = idxs_labels[0, :]

# #     # Divide the data into shards and distribute to clients
# #     for i in range(num_clients):
# #         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
# #         idx_shard = list(set(idx_shard) - rand_set)
# #         for rand in rand_set:
# #             dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

# #     # Get data for the specific client
# #     client_idxs = dict_users[client_id]
# #     X_train_client = X_train_full[client_idxs]
# #     y_train_client = y_train_full[client_idxs]

# #     # Create a small validation set from the client's training data
# #     X_train_client, X_val_client, y_train_client, y_val_client = train_test_split(
# #         X_train_client, y_train_client, test_size=0.1, random_state=42)

# #     # Use the full test set for each client (assuming we want to evaluate on the full distribution)
# #     X_test_client = X_test_full
# #     y_test_client = y_test_full

# #     print(f"Client {client_id} - X_train shape: {X_train_client.shape}, y_train shape: {y_train_client.shape}")
# #     print(f"Client {client_id} - X_val shape: {X_val_client.shape}, y_val shape: {y_val_client.shape}")
# #     print(f"Client {client_id} - X_test shape: {X_test_client.shape}, y_test shape: {y_test_client.shape}")
    
# #     # Print class distribution for this client
# #     unique, counts = np.unique(y_train_client, return_counts=True)
# #     print(f"Client {client_id} class distribution: {dict(zip(unique, counts))}")

# #     return (X_train_client, y_train_client), (X_val_client, y_val_client), (X_test_client, y_test_client)





# # def create_iid_partition(data, num_partitions):
# #     X, y = data
# #     partition_size = len(X) // num_partitions
# #     shuffled_indices = np.random.permutation(len(X))
# #     return [(X[shuffled_indices[i*partition_size:(i+1)*partition_size]], 
# #              y[shuffled_indices[i*partition_size:(i+1)*partition_size]]) 
# #             for i in range(num_partitions)]

# # def create_non_iid_partition(data, num_partitions, alpha=0.5):
# #     X, y = data
# #     if len(y.shape) > 1:
# #         y = y.argmax(axis=1)  # Convert one-hot encoded y to class indices
    
# #     num_classes = len(np.unique(y))
# #     class_idxs = [np.where(y == i)[0] for i in range(num_classes)]
    
# #     partitions = [[] for _ in range(num_partitions)]
    
# #     for idxs in class_idxs:
# #         proportions = np.random.dirichlet(np.repeat(alpha, num_partitions))
# #         proportions = (proportions / proportions.sum() * len(idxs)).astype(int)
# #         proportions[-1] = len(idxs) - proportions[:-1].sum()
        
# #         start = 0
# #         for p, num_samples in enumerate(proportions):
# #             partitions[p].extend(idxs[start:start+num_samples])
# #             start += num_samples

# #     return [(X[np.array(partition)], y[np.array(partition)]) for partition in partitions]




# import os
# import random
# import numpy as np
# from typing import Tuple, List, Dict, Optional, Union, Any
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.datasets import mnist, fashion_mnist
# import matplotlib.pyplot as plt
# from pathlib import Path

# class FederatedDataLoader:
#     """
#     A standardized federated learning data loader designed for PROFILE framework.
    
#     Implements best practices from:
#     - McMahan et al. (2017) Communication-Efficient Learning of Deep Networks from Decentralized Data
#     - Bagdasaryan et al. (2020) How To Backdoor Federated Learning
#     - Li et al. (2020) Federated Learning: Challenges, Methods, and Future Directions
#     - Wang et al. (2020) Attack of the Tails: Yes, You Really Can Backdoor Federated Learning
    
#     Args:
#         dataset_name: Name of dataset ('mnist' or 'fashion_mnist')
#         num_clients: Total number of clients in the federated setting
#         iid: If True, data is distributed IID; if False, non-IID distribution is used
#         alpha: Dirichlet concentration parameter for non-IID distribution (α→0 = more heterogeneous)
#         partition_method: Method for non-IID partitioning ('dirichlet' or 'shard')
#         seed: Random seed for reproducibility
#         verbose: If True, prints detailed information about data distribution
#     """
    
#     def __init__(
#         self, 
#         dataset_name: str = 'mnist',
#         num_clients: int = 10,
#         iid: bool = True,
#         alpha: float = 0.5,
#         partition_method: str = 'dirichlet',
#         seed: int = 42,
#         verbose: bool = True
#     ):
#         # Validate input parameters
#         if dataset_name not in ['mnist', 'fashion_mnist']:
#             raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'mnist' or 'fashion_mnist'")
        
#         if partition_method not in ['dirichlet', 'shard']:
#             raise ValueError(f"Unsupported partition method: {partition_method}. Choose 'dirichlet' or 'shard'")
        
#         self.dataset_name = dataset_name
#         self.num_clients = num_clients
#         self.iid = iid
#         self.alpha = alpha
#         self.partition_method = partition_method
#         self.seed = seed
#         self.verbose = verbose
        
#         # Set random seeds for reproducibility
#         np.random.seed(seed)
#         random.seed(seed)
        
#         # Class labels
#         if dataset_name == 'mnist':
#             self.class_names = [str(i) for i in range(10)]
#         elif dataset_name == 'fashion_mnist':
#             self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
#         # Load the dataset
#         self.data = self._load_dataset()
        
#         # Initialize client data partitions
#         self.client_partitions = None

#     def _load_dataset(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
#         """Load the dataset and perform basic preprocessing"""
#         if self.verbose:
#             print(f"Loading {self.dataset_name} dataset...")
            
#         if self.dataset_name == 'mnist':
#             (X_train, y_train), (X_test, y_test) = mnist.load_data()
#         elif self.dataset_name == 'fashion_mnist':
#             (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        
#         # Preprocess data
#         X_train = self._preprocess_images(X_train)
#         X_test = self._preprocess_images(X_test)
        
#         # Split training data into train and validation sets
#         X_train, X_val, y_train, y_val = train_test_split(
#             X_train, y_train, test_size=0.1, random_state=self.seed
#         )
        
#         if self.verbose:
#             print(f"Dataset loaded: {X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test samples")
#             print(f"Image shape: {X_train.shape[1:]}")
            
#         return {
#             'train': (X_train, y_train),
#             'val': (X_val, y_val),
#             'test': (X_test, y_test)
#         }

#     def _preprocess_images(self, X: np.ndarray) -> np.ndarray:
#         """Standardized image preprocessing for neural networks"""
#         # Normalize pixel values to [0, 1]
#         X = X.astype('float32') / 255.0
        
#         # Add channel dimension if not present
#         if len(X.shape) == 3:  # (samples, height, width)
#             X = np.expand_dims(X, -1)  # (samples, height, width, channels)
            
#         return X

#     def partition_data(self) -> None:
#         """
#         Partition data among clients according to specified distribution method.
        
#         This creates and stores client partitions that can be accessed later.
#         """
#         X_train, y_train = self.data['train']
        
#         if self.iid:
#             if self.verbose:
#                 print("Creating IID data partitions...")
#             client_partitions = self._create_iid_partitions(X_train, y_train)
#         else:
#             if self.verbose:
#                 print(f"Creating non-IID data partitions using {self.partition_method} method...")
            
#             if self.partition_method == 'dirichlet':
#                 client_partitions = self._create_dirichlet_partitions(X_train, y_train)
#             elif self.partition_method == 'shard':
#                 client_partitions = self._create_shard_partitions(X_train, y_train)
        
#         self.client_partitions = client_partitions
        
#         if self.verbose:
#             self._print_partition_statistics()

#     def _create_iid_partitions(
#         self, X: np.ndarray, y: np.ndarray
#     ) -> List[Dict[str, np.ndarray]]:
#         """Create IID (Independent and Identically Distributed) partitions"""
#         # Shuffle data
#         indices = np.random.permutation(len(y))
#         X_shuffled = X[indices]
#         y_shuffled = y[indices]
        
#         # Split into roughly equal partitions
#         X_splits = np.array_split(X_shuffled, self.num_clients)
#         y_splits = np.array_split(y_shuffled, self.num_clients)
        
#         # Create list of client partitions
#         client_partitions = []
#         for i in range(self.num_clients):
#             client_partitions.append({
#                 'X': X_splits[i],
#                 'y': y_splits[i]
#             })
            
#         return client_partitions

#     def _create_dirichlet_partitions(
#         self, X: np.ndarray, y: np.ndarray
#     ) -> List[Dict[str, np.ndarray]]:
#         """
#         Create non-IID partitions using Dirichlet distribution.
        
#         Implementation follows Yurochkin et al. (2019) and Li et al. (2020).
#         The Dirichlet distribution creates realistic heterogeneity among clients.
#         """
#         # Get number of classes
#         num_classes = len(np.unique(y))
        
#         # Initialize client partitions
#         client_partitions = [{
#             'X': [],
#             'y': []
#         } for _ in range(self.num_clients)]
        
#         # Group indices by class
#         class_indices = [np.where(y == c)[0] for c in range(num_classes)]
        
#         # For each class, distribute samples according to Dirichlet distribution
#         for c, indices in enumerate(class_indices):
#             # Sample from Dirichlet distribution
#             proportions = np.random.dirichlet(np.repeat(self.alpha, self.num_clients))
            
#             # Calculate number of samples per client
#             # Note: We use proportions * len(indices) to determine sample count
#             num_samples_per_client = (proportions * len(indices)).astype(int)
            
#             # Adjust to ensure all samples are allocated
#             num_samples_per_client[-1] = len(indices) - np.sum(num_samples_per_client[:-1])
            
#             # Shuffle indices
#             indices = np.random.permutation(indices)
            
#             # Distribute indices to clients
#             start_idx = 0
#             for client_idx, num_samples in enumerate(num_samples_per_client):
#                 if num_samples > 0:
#                     client_indices = indices[start_idx:start_idx + num_samples]
#                     client_partitions[client_idx]['X'].extend(X[client_indices].tolist())
#                     client_partitions[client_idx]['y'].extend(y[client_indices].tolist())
#                     start_idx += num_samples
        
#         # Convert lists back to arrays
#         for i in range(self.num_clients):
#             if len(client_partitions[i]['X']) > 0:
#                 client_partitions[i]['X'] = np.array(client_partitions[i]['X'])
#                 client_partitions[i]['y'] = np.array(client_partitions[i]['y'])
#             else:
#                 # Fallback for any empty partitions (should be rare)
#                 fallback_indices = np.random.choice(len(X), 10, replace=False)
#                 client_partitions[i]['X'] = X[fallback_indices]
#                 client_partitions[i]['y'] = y[fallback_indices]
            
#         return client_partitions

#     def _create_shard_partitions(
#         self, X: np.ndarray, y: np.ndarray
#     ) -> List[Dict[str, np.ndarray]]:
#         """
#         Create non-IID partitions using the shard method.
        
#         Implementation follows McMahan et al. (2017) and Hsu et al. (2019).
#         Each client gets shards of data sorted by class labels.
#         """
#         # Sort data by class
#         sorted_indices = np.argsort(y)
#         sorted_X = X[sorted_indices]
#         sorted_y = y[sorted_indices]
        
#         # Determine number of shards (2 shards per client is common)
#         shards_per_client = 2
#         num_shards = self.num_clients * shards_per_client
#         samples_per_shard = len(sorted_X) // num_shards
        
#         # Create shards
#         X_shards = np.array_split(sorted_X, num_shards)
#         y_shards = np.array_split(sorted_y, num_shards)
        
#         # Randomly assign shards to clients
#         shard_indices = list(range(num_shards))
#         np.random.shuffle(shard_indices)
        
#         # Assign shards to clients
#         client_partitions = []
#         for i in range(self.num_clients):
#             # Get indices for this client's shards
#             client_shard_indices = shard_indices[i * shards_per_client:(i + 1) * shards_per_client]
            
#             # Combine shards
#             client_X = np.concatenate([X_shards[idx] for idx in client_shard_indices])
#             client_y = np.concatenate([y_shards[idx] for idx in client_shard_indices])
            
#             client_partitions.append({
#                 'X': client_X,
#                 'y': client_y
#             })
            
#         return client_partitions

#     def _print_partition_statistics(self) -> None:
#         """Print statistics about the data partitions"""
#         if self.client_partitions is None:
#             print("No data partitions available. Call partition_data() first.")
#             return
        
#         print("\nClient Data Distribution Statistics:")
#         print("-" * 40)
        
#         total_samples = 0
#         class_distribution = np.zeros((self.num_clients, len(self.class_names)))
        
#         for i, partition in enumerate(self.client_partitions):
#             num_samples = len(partition['y'])
#             total_samples += num_samples
            
#             # Calculate class distribution
#             for c in range(len(self.class_names)):
#                 class_count = np.sum(partition['y'] == c)
#                 class_distribution[i, c] = class_count
                
#             # Print client statistics
#             print(f"Client {i}: {num_samples} samples")
#             if self.verbose:
#                 for c in range(len(self.class_names)):
#                     percentage = 100 * class_distribution[i, c] / num_samples if num_samples > 0 else 0
#                     print(f"  Class {c} ({self.class_names[c]}): {int(class_distribution[i, c])} samples ({percentage:.1f}%)")
        
#         print("-" * 40)
#         print(f"Total samples across all clients: {total_samples}")
        
#         # Calculate and print heterogeneity metrics
#         if not self.iid:
#             self._print_heterogeneity_metrics(class_distribution)
    
#     def _print_heterogeneity_metrics(self, class_distribution: np.ndarray) -> None:
#         """Calculate and print metrics quantifying data heterogeneity"""
#         # Calculate KL divergence between each client's distribution and the global distribution
#         global_distribution = np.sum(class_distribution, axis=0)
#         global_distribution = global_distribution / np.sum(global_distribution)
        
#         kl_divergences = []
#         for i in range(self.num_clients):
#             client_dist = class_distribution[i] / np.sum(class_distribution[i])
#             # Add small epsilon to avoid division by zero
#             epsilon = 1e-10
#             client_dist = np.maximum(client_dist, epsilon)
#             global_dist = np.maximum(global_distribution, epsilon)
            
#             # Calculate KL divergence: sum(p_i * log(p_i / q_i))
#             kl = np.sum(client_dist * np.log(client_dist / global_dist))
#             kl_divergences.append(kl)
        
#         print("\nHeterogeneity Metrics:")
#         print(f"Average KL divergence: {np.mean(kl_divergences):.4f}")
#         print(f"Max KL divergence: {np.max(kl_divergences):.4f}")
#         print(f"Min KL divergence: {np.min(kl_divergences):.4f}")

#     def get_client_data(
#         self, client_id: int
#     ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
#         """
#         Get data for a specific client.
        
#         Args:
#             client_id: The client ID (0 to num_clients-1)
            
#         Returns:
#             Tuple containing (train_data, val_data, test_data), where each is a tuple of (X, y)
#         """
#         if self.client_partitions is None:
#             self.partition_data()
            
#         if client_id < 0 or client_id >= self.num_clients:
#             raise ValueError(f"Invalid client_id: {client_id}. Must be between 0 and {self.num_clients-1}")
            
#         # Get client's training data
#         X_train = self.client_partitions[client_id]['X']
#         y_train = self.client_partitions[client_id]['y']
        
#         # Validation and test data are the same for all clients
#         X_val, y_val = self.data['val']
#         X_test, y_test = self.data['test']
        
#         return (X_train, y_train), (X_val, y_val), (X_test, y_test)

#     def apply_poisoning(
#         self,
#         X: np.ndarray,
#         y: np.ndarray,
#         attack_type: str = 'label_flip',
#         poison_ratio: float = 0.1,
#         target_class: Optional[int] = None,
#         source_class: Optional[int] = None,
#         trigger: Optional[np.ndarray] = None
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Apply poisoning attack to the dataset.
        
#         Implements multiple attack types from literature:
#         - Label flipping (Tolpegin et al., 2020)
#         - Targeted attack (Bagdasaryan et al., 2020)
#         - Backdoor attack (Wang et al., 2020)
        
#         Args:
#             X: Input data
#             y: Labels
#             attack_type: Type of attack ('label_flip', 'targeted', 'random', 'backdoor')
#             poison_ratio: Fraction of data to poison
#             target_class: Target class for targeted attacks
#             source_class: Only poison samples from this class (if None, poison from all classes)
#             trigger: Custom trigger pattern for backdoor attacks (if None, default is used)
            
#         Returns:
#             Tuple of (poisoned_X, poisoned_y)
#         """
#         # Make copies to avoid modifying original data
#         X_poisoned = X.copy()
#         y_poisoned = y.copy()
        
#         # Determine which samples to poison
#         num_samples = len(y)
#         num_poison_samples = int(num_samples * poison_ratio)
        
#         # Filter by source class if specified
#         if source_class is not None:
#             source_indices = np.where(y == source_class)[0]
#             if len(source_indices) == 0:
#                 print(f"Warning: No samples of class {source_class} found. No poisoning applied.")
#                 return X, y
#             num_poison_samples = min(num_poison_samples, len(source_indices))
#             poison_indices = np.random.choice(source_indices, num_poison_samples, replace=False)
#         else:
#             poison_indices = np.random.choice(num_samples, num_poison_samples, replace=False)
        
#         # Apply poisoning based on attack type
#         if attack_type == 'label_flip':
#             y_poisoned[poison_indices] = self._apply_label_flip(
#                 y_poisoned[poison_indices], 
#                 target_class
#             )
            
#         elif attack_type == 'random':
#             y_poisoned[poison_indices] = self._apply_random_labels(
#                 y_poisoned[poison_indices]
#             )
            
#         elif attack_type == 'targeted':
#             if target_class is None:
#                 raise ValueError("target_class must be specified for targeted attacks")
#             y_poisoned[poison_indices] = target_class
            
#         elif attack_type == 'backdoor':
#             if target_class is None:
#                 raise ValueError("target_class must be specified for backdoor attacks")
                
#             # Apply trigger pattern
#             X_poisoned[poison_indices] = self._apply_backdoor_trigger(
#                 X_poisoned[poison_indices], 
#                 trigger
#             )
#             # Change labels to target class
#             y_poisoned[poison_indices] = target_class
            
#         else:
#             raise ValueError(f"Unknown attack type: {attack_type}")
        
#         # Print attack statistics
#         self._print_attack_statistics(
#             attack_type, 
#             num_poison_samples, 
#             num_samples, 
#             y, 
#             y_poisoned, 
#             poison_indices,
#             target_class
#         )
        
#         return X_poisoned, y_poisoned

#     def _apply_label_flip(
#         self, y: np.ndarray, target_class: Optional[int] = None
#     ) -> np.ndarray:
#         """
#         Apply label flipping attack.
        
#         Args:
#             y: Original labels
#             target_class: If provided, flip all labels to this class
#                          If None, flip each label to the next class circularly
                         
#         Returns:
#             Modified labels
#         """
#         num_classes = len(self.class_names)
        
#         if target_class is not None:
#             # Flip all labels to target class
#             return np.full_like(y, target_class)
#         else:
#             # Flip each label to the next class (cyclically)
#             return (y + 1) % num_classes

#     def _apply_random_labels(self, y: np.ndarray) -> np.ndarray:
#         """
#         Apply random label attack.
        
#         Args:
#             y: Original labels
            
#         Returns:
#             Modified labels
#         """
#         num_classes = len(self.class_names)
#         y_poisoned = y.copy()
        
#         for i in range(len(y)):
#             # Choose a random class different from the original
#             possible_classes = [c for c in range(num_classes) if c != y[i]]
#             y_poisoned[i] = np.random.choice(possible_classes)
            
#         return y_poisoned

#     def _apply_backdoor_trigger(
#         self, X: np.ndarray, custom_trigger: Optional[np.ndarray] = None
#     ) -> np.ndarray:
#         """
#         Apply backdoor trigger pattern to images.
        
#         Args:
#             X: Input images
#             custom_trigger: Custom trigger pattern (if None, use default)
            
#         Returns:
#             Modified images with trigger
#         """
#         X_triggered = X.copy()
        
#         if custom_trigger is not None:
#             trigger = custom_trigger
#         else:
#             # Create default trigger (small white square in corner)
#             trigger_size = 3
#             h, w = X.shape[1:3]
#             trigger = np.zeros_like(X[0])
#             trigger[-trigger_size:, -trigger_size:, :] = 1.0
        
#         # Apply trigger
#         X_triggered = np.clip(X_triggered + trigger, 0, 1)
        
#         return X_triggered

#     def _print_attack_statistics(
#         self,
#         attack_type: str,
#         num_poison_samples: int,
#         num_samples: int,
#         y_original: np.ndarray,
#         y_poisoned: np.ndarray,
#         poison_indices: np.ndarray,
#         target_class: Optional[int] = None
#     ) -> None:
#         """Print statistics about the applied poisoning attack"""
#         print("\nPoisoning Attack Statistics:")
#         print(f"Attack type: {attack_type}")
#         print(f"Poisoned samples: {num_poison_samples}/{num_samples} ({100*num_poison_samples/num_samples:.1f}%)")
        
#         if attack_type == 'targeted' or (attack_type == 'label_flip' and target_class is not None):
#             print(f"Target class: {target_class} ({self.class_names[target_class]})")
            
#         # Calculate class transition matrix (for label attacks)
#         if attack_type in ['label_flip', 'random', 'targeted']:
#             transitions = {}
#             for idx in poison_indices:
#                 original = y_original[idx]
#                 poisoned = y_poisoned[idx]
#                 if original != poisoned:
#                     key = (int(original), int(poisoned))
#                     transitions[key] = transitions.get(key, 0) + 1
            
#             # Print transitions
#             print("\nLabel transitions:")
#             for (src, dst), count in sorted(transitions.items()):
#                 src_name = self.class_names[src]
#                 dst_name = self.class_names[dst]
#                 print(f"  Class {src} ({src_name}) → Class {dst} ({dst_name}): {count} samples")

#     def visualize_client_distribution(
#         self, 
#         figsize: Tuple[int, int] = (12, 8), 
#         output_path: Optional[str] = None
#     ) -> None:
#         """
#         Visualize the class distribution across clients.
        
#         Args:
#             figsize: Figure size
#             output_path: If provided, save the figure to this path
#         """
#         if self.client_partitions is None:
#             print("No data partitions available. Call partition_data() first.")
#             return
        
#         num_classes = len(self.class_names)
        
#         # Calculate class distribution for each client
#         class_distribution = np.zeros((self.num_clients, num_classes))
#         for i, partition in enumerate(self.client_partitions):
#             for c in range(num_classes):
#                 class_distribution[i, c] = np.sum(partition['y'] == c)
        
#         # Normalize distribution (percentage)
#         row_sums = class_distribution.sum(axis=1, keepdims=True)
#         normalized_distribution = 100 * class_distribution / row_sums
        
#         # Plot
#         plt.figure(figsize=figsize)
        
#         # Heatmap
#         plt.imshow(normalized_distribution, cmap='YlGnBu', aspect='auto')
#         plt.colorbar(label='Percentage of client data')
        
#         # Labels
#         plt.xlabel('Class')
#         plt.ylabel('Client ID')
#         plt.title('Class Distribution Across Clients')
        
#         # Customize x-axis ticks
#         plt.xticks(range(num_classes), self.class_names, rotation=45, ha='right')
#         plt.yticks(range(self.num_clients), [f'Client {i}' for i in range(self.num_clients)])
        
#         plt.tight_layout()
        
#         if output_path:
#             plt.savefig(output_path, dpi=300, bbox_inches='tight')
#             print(f"Figure saved to {output_path}")
        
#         plt.show()

#     def get_validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
#         """Get validation data (same for all clients)"""
#         return self.data['val']
    
#     def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
#         """Get test data (same for all clients)"""
#         return self.data['test']


# class FederatedPoisoningExperiment:
#     """
#     A class to facilitate poisoning attack experiments in federated learning.
    
#     This class simplifies the setup and execution of poisoning experiments
#     by providing a clean interface for:
#     - Configuring which clients are malicious
#     - Setting up different attack types
#     - Controlling attack timing (e.g., after convergence)
#     - Managing attack parameters
    
#     Args:
#         num_clients: Total number of federated learning clients
#         malicious_client_ids: List of client IDs that will perform attacks
#         attack_config: Configuration parameters for the attack
#         start_round: Training round to start attack (0 = from beginning)
#         end_round: Training round to end attack (None = continue to end)
#     """
    
#     def __init__(
#         self,
#         num_clients: int = 10,
#         malicious_client_ids: Optional[List[int]] = None,
#         attack_config: Optional[Dict[str, Any]] = None,
#         start_round: int = 0,
#         end_round: Optional[int] = None
#     ):
#         self.num_clients = num_clients
        
#         # Set default malicious clients if not specified (10% of clients)
#         if malicious_client_ids is None:
#             num_malicious = max(1, int(0.1 * num_clients))
#             self.malicious_client_ids = list(range(num_malicious))
#         else:
#             self.malicious_client_ids = malicious_client_ids
            
#         # Default attack configuration
#         default_attack_config = {
#             'type': 'label_flip',
#             'poison_ratio': 0.5,
#             'target_class': None,
#             'source_class': None
#         }
        
#         # Update with user-provided config
#         if attack_config:
#             default_attack_config.update(attack_config)
            
#         self.attack_config = default_attack_config
#         self.start_round = start_round
#         self.end_round = end_round
#         self.current_round = 0
        
#         print("Federated Poisoning Experiment Configured:")
#         print(f"Total clients: {num_clients}")
#         print(f"Malicious clients: {self.malicious_client_ids} ({len(self.malicious_client_ids)}/{num_clients})")
#         print(f"Attack type: {self.attack_config['type']}")
#         print(f"Poison ratio: {self.attack_config['poison_ratio']}")
#         print(f"Attack timing: Round {start_round} to {end_round if end_round is not None else 'end'}")
    
#     def should_poison(self, client_id: int, round_idx: int) -> bool:
#         """
#         Determine if a client should poison its data in the current round.
        
#         Args:
#             client_id: ID of the client
#             round_idx: Current training round
            
#         Returns:
#             True if the client should poison its data, False otherwise
#         """
#         self.current_round = round_idx
        
#         # Check if client is malicious
#         if client_id not in self.malicious_client_ids:
#             return False
            
#         # Check if we're in the attack window
#         if self.start_round <= round_idx and (self.end_round is None or round_idx <= self.end_round):
#             return True
            
#         return False
    
#     def get_attack_config(self) -> Dict[str, Any]:
#         """Get the current attack configuration"""
#         return self.attack_config
    
#     def update_attack_config(self, **kwargs) -> None:
#         """
#         Update attack configuration parameters.
        
#         This allows dynamically changing attack parameters during the experiment.
#         """
#         self.attack_config.update(kwargs)
#         print(f"Attack configuration updated at round {self.current_round}:")
#         for key, value in kwargs.items():
#             print(f"  {key}: {value}")


# # Example usage for PROFILE client:
# if __name__ == "__main__":
#     # Initialize data loader
#     loader = FederatedDataLoader(
#         dataset_name='mnist',
#         num_clients=10,
#         iid=False,
#         alpha=0.5,
#         partition_method='dirichlet',
#         seed=42
#     )
    
#     # Partition data
#     loader.partition_data()
    
#     # Visualize client distribution
#     loader.visualize_client_distribution()
    
#     # Get data for client 0
#     client_id = 0
#     (X_train, y_train), (X_val, y_val), (X_test, y_test) = loader.get_client_data(client_id)
    
#     # Configure poisoning experiment
#     experiment = FederatedPoisoningExperiment(
#         num_clients=10,
#         malicious_client_ids=[3, 7],
#         attack_config={
#             'type': 'label_flip',
#             'poison_ratio': 0.3,
#             'target_class': 7
#         },
#         start_round=5,
#         end_round=10
#     )
    
#     # In PROFILE client's fit method:
#     round_idx = 6  # Current round
#     if experiment.should_poison(client_id, round_idx):
#         attack_config = experiment.get_attack_config()
#         X_train, y_train = loader.apply_poisoning(X_train, y_train, **attack_config)
#         print(f"Client {client_id} is poisoning data in round {round_idx}")
#     else:
#         print(f"Client {client_id} is using clean data in round {round_idx}")




# Standard Libraries
import os
import random
from pathlib import Path

# Third Party
import numpy as np
import progressbar
from matplotlib import image as img
from skimage.transform import resize
from sklearn.model_selection import train_test_split



def load_raw_covid_data(limit):
    """
    Loads COVID and non-COVID to numpy arrays, resized to specific IMG_SIZE
    Images are randomly sampled if provided limit. 
    Split loaded and processed images to training and testing, return four sets.
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the script
    covid_path = os.path.join(script_dir, "data", "covid")
    non_covid_path = os.path.join(script_dir, "data", "noncovid")
    covid_images = [f for f in os.listdir(covid_path) if f.endswith('.png')]
    non_covid_images = [f for f in os.listdir(non_covid_path) if f.endswith('.png')]

    # Randomly samples "limit" amount of images for the train-test phase
    if limit != None:
        covid_images = random.sample(covid_images, limit)
        non_covid_images = random.sample(non_covid_images, limit)

    IMG_SIZE = 128

    # Two empty numpy arrays to store coverted images
    positive_npy = np.empty(
        (len(covid_images), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
    negative_npy = np.empty(
        (len(non_covid_images), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    # start a bar of show percentage of loading data
    covid_bar = progressbar.ProgressBar(maxval=len(covid_images), widgets=[
                                        progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    non_covid_bar = progressbar.ProgressBar(maxval=len(non_covid_images), widgets=[
                                            progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    covid_bar.start()
    # Converting COVID dataset to .npy format
    for i, _file in enumerate(covid_images):
        try:
            image_npy = img.imread(_file)
            positive = resize(
                image_npy, (IMG_SIZE, IMG_SIZE, 1), anti_aliasing=True)
            positive_npy[i] = positive
        except:
            pass
        covid_bar.update(i+1)

    covid_bar.finish()
    print("COVID images converting done")

    non_covid_bar.start()
    # Converting non-COVID dataset to .npy format
    for i, _file in enumerate(non_covid_images):
        try:
            image_npy = img.imread(_file)
            negative = resize(
                image_npy, (IMG_SIZE, IMG_SIZE, 1), anti_aliasing=True)
            negative_npy[i] = negative
        except:
            pass
        non_covid_bar.update(i+1)
    non_covid_bar.finish()
    print("non covid images converting done")

    positive = positive_npy
    positive_labels = ["1" for i in positive]
    negative = negative_npy
    negative_labels = ["0" for i in negative]

    # Joining both datasets and labels
    X = np.concatenate([positive, negative])
    y = np.array((positive_labels + negative_labels), dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test


# Simple COVID Data Wrapper to mimic FederatedDataLoader interface
class CovidDataWrapper:
    def __init__(self, X_train, X_test, y_train, y_test, num_clients=6, client_id=0):
        self.dataset_name = "covid"
        self.num_clients = num_clients
        
        # Split training data among clients
        samples_per_client = len(X_train) // num_clients
        start_idx = client_id * samples_per_client
        end_idx = start_idx + samples_per_client if client_id < num_clients - 1 else len(X_train)
        
        # Assign data to this client
        self.X_train = X_train[start_idx:end_idx]
        self.y_train = y_train[start_idx:end_idx]
        
        # Split client's training data into train/val
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.1, random_state=42
        )
        
        # All clients use same test data
        self.X_test = X_test
        self.y_test = y_test
    
    def get_client_data(self, client_id):
        """Mimic FederatedDataLoader interface"""
        return (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test)
    
    def apply_poisoning(self, X_train, y_train, **kwargs):
        """Simple poisoning for COVID data - just flip some labels"""
        poison_ratio = kwargs.get('poison_ratio', 0.1)
        attack_type = kwargs.get('attack_type', 'label_flip')
        
        X_poisoned = X_train.copy()
        y_poisoned = y_train.copy()
        
        if attack_type == 'label_flip':
            # Randomly select samples to flip
            num_poison = int(len(y_train) * poison_ratio)
            poison_indices = np.random.choice(len(y_train), num_poison, replace=False)
            
            # Flip labels (0->1, 1->0)
            y_poisoned[poison_indices] = 1 - y_poisoned[poison_indices]
            
            print(f"Poisoned {num_poison} samples by flipping labels")
        
        return X_poisoned, y_poisoned