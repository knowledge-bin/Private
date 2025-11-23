# #!/usr/bin/env python3
# # PROFILE_server.py

# # import importlib
# # import math
# # import os
# # import sys
# # import time
# # import random
# # import concurrent.futures
# # import flwr as fl
# # import numpy as np  # Needed for evaluation function
# # from typing import Dict, List, Optional, Tuple
# # from load_covid import load_mnist_data
# # from flwr.common import NDArrays, Scalar, Parameters, GetParametersIns, ReconnectIns
# # from flwr.server.strategy import FedAvg
# # from memory_profiler import memory_usage
# # from rlwe_xmkckks import RLWE
# # from sklearn.metrics import classification_report, log_loss
# # import multiprocessing
# # import threading

# # # Local Imports
# # from load_covid import *
# # # Get absolute paths to let a user run the script from anywhere
# # current_directory = os.path.dirname(os.path.abspath(__file__))
# # parent_directory = os.path.basename(current_directory)
# # working_directory = os.getcwd()
# # sys.path.append(os.path.join(current_directory, '..'))
# # if current_directory == working_directory:
# #     from cnn import CNN
# #     import utils
# # else:
# #     CNN = importlib.import_module(f"{parent_directory}.cnn").CNN
# #     import utils

# # # --- Helper functions for client communication using concurrent futures ---
# # def my_fit_clients(client_instructions, max_workers, timeout):
# #     """Send fit instructions concurrently to clients.
# #        Returns (results, failures) where each result is a tuple (client, fit_res)."""
# #     results = []
# #     failures = []
# #     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
# #         future_to_client = {
# #             executor.submit(lambda pair: (pair[0], pair[0].fit(pair[1], timeout=timeout)), instr): instr
# #             for instr in client_instructions
# #         }
# #         for future in concurrent.futures.as_completed(future_to_client):
# #             try:
# #                 res = future.result()
# #                 results.append(res)
# #             except Exception as e:
# #                 failures.append(e)
# #     return results, failures

# # def my_evaluate_clients(client_instructions, max_workers, timeout):
# #     """Send evaluate instructions concurrently to clients.
# #        Returns (results, failures) where each result is a tuple (client, eval_res)."""
# #     results = []
# #     failures = []
# #     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
# #         future_to_client = {
# #             executor.submit(lambda pair: (pair[0], pair[0].evaluate(pair[1], timeout=timeout)), instr): instr
# #             for instr in client_instructions
# #         }
# #         for future in concurrent.futures.as_completed(future_to_client):
# #             try:
# #                 res = future.result()
# #                 results.append(res)
# #             except Exception as e:
# #                 failures.append(e)
# #     return results, failures

# # def my_reconnect_clients(client_instructions, max_workers, timeout):
# #     """Send reconnect instructions concurrently to clients.
# #        Returns (results, failures) where each result is a tuple (client, disconnect_res)."""
# #     results = []
# #     failures = []
# #     with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
# #         future_to_client = {
# #             executor.submit(lambda pair: (pair[0], pair[0].reconnect(pair[1], timeout=timeout)), instr): instr
# #             for instr in client_instructions
# #         }
# #         for future in concurrent.futures.as_completed(future_to_client):
# #             try:
# #                 res = future.result()
# #                 results.append(res)
# #             except Exception as e:
# #                 failures.append(e)
# #     return results, failures

# # # --- Evaluation Function for Server-side Evaluation ---
# # def get_evaluate_fn(model):
# #     """Return an evaluation function for server-side evaluation."""
# #     # Load MNIST test data to avoid overhead in evaluation
# #     _, _, (X_test, y_test) = load_mnist_data()

# #     def evaluate(
# #         server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
# #     ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
# #         utils.set_model_params(model, parameters)
# #         y_pred = model.model.predict(X_test)
# #         predicted = np.argmax(y_pred, axis=-1)
# #         true_labels = y_test
# #         accuracy = (true_labels == predicted).mean()
# #         loss = log_loss(y_test, y_pred)
# #         print(classification_report(true_labels, predicted))
# #         return loss, {"accuracy": accuracy}
# #     return evaluate

# # def fit_round(server_round: int) -> Dict:
# #     """Send round number to client."""
# #     return {"server_round": server_round}

# # # --- Custom Strategy ---
# # class CustomFedAvg(FedAvg):
# #     def __init__(self, rlwe_instance: RLWE, num_groups: int = 1, *args, **kwargs):
# #         super().__init__(*args, **kwargs)
# #         self.rlwe = rlwe_instance
# #         self.num_groups = num_groups

# # # --- Helper: Group Clients into Buckets ---
# # def group_clients(clients: List[fl.server.client_proxy.ClientProxy], num_groups: int) -> List[List[fl.server.client_proxy.ClientProxy]]:
# #     """Divide clients into num_groups buckets using simple slicing."""
# #     return [clients[i::num_groups] for i in range(num_groups)]

# # # --- Custom Server with Bucketting (Group-Based Aggregation) ---
# # from flwr.server.client_manager import SimpleClientManager
# # from flwr.server.history import History

# # class Server(fl.server.Server):
# #     def __init__(
# #         self,
# #         *,
# #         client_manager: fl.server.client_manager.ClientManager,
# #         strategy: Optional[fl.server.strategy.Strategy] = None,
# #         num_groups: int = 1
# #     ) -> None:
# #         super().__init__(client_manager=client_manager, strategy=strategy)
# #         self.parameters: Parameters = Parameters(tensors=[], tensor_type="numpy.ndarray")
# #         self.max_workers: Optional[int] = None
# #         self.num_groups = num_groups

# #     def set_max_workers(self, max_workers: Optional[int]) -> None:
# #         self.max_workers = max_workers

# #     def set_strategy(self, strategy: fl.server.strategy.Strategy) -> None:
# #         self.strategy = strategy

# #     def client_manager(self) -> fl.server.client_manager.ClientManager:
# #         return self._client_manager

# #     def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
# #         history = History()

# #         print("[Server] Initializing global parameters")
# #         self.parameters = self._get_initial_parameters(timeout=timeout)
# #         print("[Server] Evaluating initial parameters")
# #         res = self.strategy.evaluate(0, parameters=self.parameters)
# #         if res is not None:
# #             print(f"[Server] Initial parameters (loss, metrics): {res[0]}, {res[1]}")
# #             history.add_loss_centralized(server_round=0, loss=res[0])
# #             history.add_metrics_centralized(server_round=0, metrics=res[1])

# #         # RLWE Setup: generate vector_a
# #         self.strategy.rlwe.generate_vector_a()
# #         vector_a = self.strategy.rlwe.get_vector_a().poly_to_list()

# #         # Group clients
# #         all_clients = list(self._client_manager.clients.values())
# #         groups = group_clients(all_clients, self.num_groups)

# #         print(f"[Server] Total clients: {len(all_clients)}, Divided into {self.num_groups} groups")
# #         for idx, group in enumerate(groups):
# #             print(f"[Server] Group {idx} size: {len(group)}")

# #         # Public key aggregation per group
# #         for group_idx, clients_in_group in enumerate(groups):
# #             print(f"\n[Server] Processing public key aggregation for Group {group_idx}")
# #             vector_b_list = []
# #             for c in clients_in_group:
# #                 b = c.request_vec_b(vector_a, timeout=timeout)
# #                 vector_b_list.append(self.strategy.rlwe.list_to_poly(b, "q"))
# #             if vector_b_list:
# #                 allpub = sum(vector_b_list[1:], start=vector_b_list[0]).poly_to_list()
# #                 for c in clients_in_group:
# #                     c.request_allpub_confirmation(allpub, timeout=timeout)
# #                 print(f"[Server] Group {group_idx} public key aggregation completed.")
# #             else:
# #                 print(f"[Server] Warning: Group {group_idx} is empty!")

# #         # Main training rounds: process each group separately.
# #         print("[Server] Starting federated learning rounds")
# #         overall_start = time.time()
# #         for current_round in range(1, num_rounds + 1):
# #             print(f"\n=== Round {current_round} ===")
# #             for group_idx, clients_in_group in enumerate(groups):
# #                 print(f"\n--- Group {group_idx} Training ---")
# #                 res_fit = self.fit_round_on_clients(
# #                     server_round=current_round, timeout=timeout, group_clients=clients_in_group
# #                 )
# #                 if res_fit is None:
# #                     print(f"[Server] Round {current_round}, Group {group_idx}: No clients participated in training.")
# #                     continue

# #                 # Secure weight aggregation per group
# #                 request = "full" if current_round == 1 else "gradient"
# #                 c0sum, c1sum = None, None
# #                 print(f"[Server] Group {group_idx}: Aggregating encrypted parameters.")
# #                 for i, c in enumerate(clients_in_group):
# #                     c0, c1 = c.request_encrypted_parameters(request, timeout=timeout)
# #                     c0p = self.strategy.rlwe.list_to_poly(c0, "q")
# #                     c1p = self.strategy.rlwe.list_to_poly(c1, "q")
# #                     if i == 0:
# #                         c0sum, c1sum = c0p, c1p
# #                     else:
# #                         c0sum += c0p
# #                         c1sum += c1p
# #                 print(f"[Server] Group {group_idx}: Aggregated ciphertext computed.")

# #                 c1sum_list = c1sum.poly_to_list()
# #                 dsum = None
# #                 print(f"[Server] Group {group_idx}: Requesting decryption shares.")
# #                 for i, c in enumerate(clients_in_group):
# #                     d = c.request_decryption_share(c1sum_list, timeout=timeout)
# #                     d_poly = self.strategy.rlwe.list_to_poly(d, "t")
# #                     if i == 0:
# #                         dsum = d_poly
# #                     else:
# #                         dsum += d_poly

# #                 plaintext = c0sum + dsum
# #                 plaintext = plaintext.testing(self.strategy.rlwe.t).poly_to_list()
# #                 avg_weights = [round(w / len(clients_in_group)) for w in plaintext]
# #                 for c in clients_in_group:
# #                     c.request_modelupdate_confirmation(avg_weights, timeout=timeout)
# #                 print(f"[Server] Group {group_idx}: Model update confirmed.")
# #             print(f"[Server] Round {current_round} completed in {time.time() - overall_start:.2f} seconds")
# #         print("[Server] Federated learning complete.")
# #         return history
# #     # Helper: Run fit round on a subset of clients (group_clients)
# #     def fit_round_on_clients(
# #         self,
# #         server_round: int,
# #         timeout: Optional[float],
# #         group_clients: List[fl.server.client_proxy.ClientProxy],
# #     ) -> Optional[Tuple[Optional[Parameters], Dict[str, Scalar], Tuple[List, List]]]:
# #         # Create a temporary client manager with only these clients.
# #         temp_cm = SimpleClientManager()
# #         temp_cm.clients = {str(id(c)): c for c in group_clients}
# #         client_instructions = self.strategy.configure_fit(
# #             server_round=server_round,
# #             parameters=self.parameters,
# #             client_manager=temp_cm,
# #         )
# #         if not client_instructions:
# #             print(f"[Server] fit_round {server_round}: No clients selected for this group; canceling.")
# #             return None
# #         results, failures = my_fit_clients(
# #             client_instructions=client_instructions,
# #             max_workers=self.max_workers,
# #             timeout=timeout,
# #         )
# #         aggregated_result = self.strategy.aggregate_fit(server_round, results, failures)
# #         parameters_aggregated, metrics_aggregated = aggregated_result
# #         print(f"[Server] fit_round {server_round}: Aggregated training results computed for group.")
# #         return parameters_aggregated, metrics_aggregated, (results, failures)

# #     def evaluate_round(
# #         self,
# #         server_round: int,
# #         timeout: Optional[float],
# #     ) -> Optional[Tuple[Optional[float], Dict[str, Scalar], Tuple[List, List]]]:
# #         client_instructions = self.strategy.configure_evaluate(
# #             server_round=server_round,
# #             parameters=self.parameters,
# #             client_manager=self._client_manager,
# #         )
# #         if not client_instructions:
# #             print(f"[Server] evaluate_round {server_round}: No clients selected; canceling.")
# #             return None
# #         results, failures = my_evaluate_clients(
# #             client_instructions,
# #             max_workers=self.max_workers,
# #             timeout=timeout,
# #         )
# #         aggregated_result = self.strategy.aggregate_evaluate(server_round, results, failures)
# #         loss_aggregated, metrics_aggregated = aggregated_result
# #         return loss_aggregated, metrics_aggregated, (results, failures)

# #     def disconnect_all_clients(self, timeout: Optional[float]) -> None:
# #         all_clients = self._client_manager.all()
# #         clients = [all_clients[k] for k in all_clients.keys()]
# #         instruction = ReconnectIns(seconds=None)
# #         client_instructions = [(client_proxy, instruction) for client_proxy in clients]
# #         _ = my_reconnect_clients(
# #             client_instructions=client_instructions,
# #             max_workers=self.max_workers,
# #             timeout=timeout,
# #         )

# #     def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
# #         parameters: Optional[Parameters] = self.strategy.initialize_parameters(
# #             client_manager=self._client_manager
# #         )
# #         if parameters is not None:
# #             print("[Server] Using initial parameters provided by strategy")
# #             return parameters
# #         print("[Server] Requesting initial parameters from one random client")
# #         random_client = self._client_manager.sample(1)[0]
# #         ins = GetParametersIns(config={})
# #         get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
# #         print("[Server] Received initial parameters from one random client")
# #         return get_parameters_res.parameters

# # # --- Main: Start the Server and Measure Memory Usage ---
# # if __name__ == "__main__":
# #     def measure_memory():
# #         start_time = time.time()

# #         # RLWE SETTINGS (dynamically)
# #         WEIGHT_DECIMALS = 8
# #         model = CNN(WEIGHT_DECIMALS)
# #         utils.set_initial_params(model)
# #         params, _ = utils.get_flat_weights(model)
# #         num_weights = len(params)
# #         n = 2 ** math.ceil(math.log2(num_weights))
# #         print(f"n: {n}")

# #         max_weight_value = 10 ** WEIGHT_DECIMALS
# #         num_clients = 2
# #         t = utils.next_prime(num_clients * max_weight_value * 2)
# #         print(f"t: {t}")

# #         q = utils.next_prime(t * 50)
# #         print(f"q: {q}")

# #         std = 3
# #         rlwe = RLWE(n, q, t, std)

# #         # Define number of groups (e.g., 3 for bucketing)
# #         NUM_GROUPS = 2

# #         # Custom Strategy with number of groups passed
# #         strategy = CustomFedAvg(
# #             min_available_clients=2,
# #             min_fit_clients=2,
# #             evaluate_fn=get_evaluate_fn(model),
# #             on_fit_config_fn=fit_round,
# #             rlwe_instance=rlwe,
# #             num_groups=NUM_GROUPS
# #         )

# #         client_manager = fl.server.SimpleClientManager()
# #         server = Server(strategy=strategy, client_manager=client_manager, num_groups=NUM_GROUPS)

# #         fl.server.start_server(
# #             server_address="127.0.0.1:9010",
# #             server=server,
# #             config=fl.server.ServerConfig(num_rounds=5)
# #         )

# #         exec_time = time.time() - start_time
# #         print("Execution time:", exec_time)

# #     mem_usage = memory_usage(measure_memory)
# #     print("Memory usage (in MB):", max(mem_usage))










# # #!/usr/bin/env python3
# # # PROFILE_server.py

# # import socket
# # # Standard Libraries
# # import importlib
# # import math
# # import os
# # import sys
# # import time
# # import random
# # import flwr as fl
# # from typing import Dict, List, Optional, Tuple
# # from load_covid import load_mnist_data
# # from flwr.common import NDArrays, Scalar
# # from flwr.server.strategy import FedAvg
# # from memory_profiler import memory_usage
# # from rlwe_xmkckks import RLWE
# # from sklearn.metrics import classification_report, log_loss
# # import multiprocessing
# # import threading
# # from sklearn.metrics import classification_report, log_loss, precision_score, recall_score, f1_score, confusion_matrix
# # # Local Imports
# # from load_covid import *

# # # Local Imports
# # from load_covid import *
# # # Get absolute paths to let a user run the script from anywhere
# # current_directory = os.path.dirname(os.path.abspath(__file__))
# # parent_directory = os.path.basename(current_directory)
# # working_directory = os.getcwd()
# # # Add parent directory to Python's module search path
# # sys.path.append(os.path.join(current_directory, '..'))
# # # Compare paths
# # if current_directory == working_directory:
# #     from cnn import CNN
# #     import utils
# # else:
# #     # Add current directory to Python's module search path
# #     CNN = importlib.import_module(f"{parent_directory}.cnn").CNN
# #     import utils



# # def fit_round(server_round: int) -> Dict:
# #     """Send round number to client."""
# #     return {"server_round": server_round}


# # def get_evaluate_fn(model):
# #     """Return an evaluation function for server-side evaluation."""
# #     # Load MNIST test data here to avoid the overhead of doing it in 'evaluate' itself
# #     _, _, (X_test, y_test) = load_mnist_data()

# #     # The 'evaluate' function will be called after every round
# #     def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
# #         utils.set_model_params(model, parameters)

# #         y_pred = model.model.predict(X_test)
# #         predicted = np.argmax(y_pred, axis=-1)

# #         # FIXED LINE:
# #         true_labels = y_test  # Remove np.argmax if y_test is already in label form

# #         accuracy = np.equal(true_labels, predicted).mean()
# #         loss = log_loss(y_test, y_pred)

# #         print(classification_report(true_labels, predicted))

# #         return loss, {"accuracy": accuracy}

# #     return evaluate




# # class CustomFedAvg(FedAvg):
# #     def __init__(
# #         self, 
# #         rlwe_instance: RLWE, 
# #         num_groups: int = 1,  # Default number of buckets/groups
# #         eval_fn=None,
# #         *args, 
# #         **kwargs
# #     ):
# #         super().__init__(*args, **kwargs)
# #         self.rlwe = rlwe_instance
# #         self.num_groups = num_groups  # Number of buckets to divide clients into
# #         self.bucket_models = {}  # Store models for each bucket
# #         self.bucket_weights = {}  # Store weights for each bucket
# #         self.evaluate_fn = eval_fn  # Function for evaluating bucket models

# # def fit_round(server_round: int) -> Dict:
# #     """Send round number to client."""
# #     return {"server_round": server_round}
# # def get_evaluate_fn(model):
# #     """Return an evaluation function for server-side evaluation."""
# #     _, _, (X_test, y_test) = load_mnist_data()

# #     def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
# #         utils.set_model_params(model, parameters)
# #         y_pred = model.model.predict(X_test)
# #         predicted = np.argmax(y_pred, axis=-1)
# #         true_labels = np.argmax(y_test, axis=-1)
# #         accuracy = np.equal(true_labels, predicted).mean()
# #         loss = log_loss(true_labels, y_pred)
# #         print(classification_report(true_labels, predicted))
# #         return loss, {"accuracy": accuracy}

# # # Start Flower server for five rounds of federated learning
# # if __name__ == "__main__": 
# #     def measure_memory():
# #         # Measure the execution time
# #         start_time = time.time()

# #         # # RLWE SETTINGS with minimal changes
# #         # WEIGHT_DECIMALS = 4  # Keep your current decimal precision
# #         # model = CNN(WEIGHT_DECIMALS)
# #         # utils.set_initial_params(model)
# #         # params, _ = utils.get_flat_weights(model)
# #         # print("Initial parameters", params[0:20])

# #         # # Step 1: Calculate ring dimension (keep your current approach)
# #         # num_weights = len(params)
# #         # n = 2 ** math.ceil(math.log2(num_weights))
# #         # print(f"n: {n}")

# #         # # Step 2: Choose plaintext modulus t based on our testing
# #         # num_clients = 6        #############################################
        
# #         #   # Your actual client count
# #         # max_weight_value = 10 ** WEIGHT_DECIMALS  # 10,000 for 4 decimal places
# #         # t = utils.next_prime(num_clients * max_weight_value * 4)  # Factor of 4 from our testing
# #         # print(f"t: {t}")

# #         # # Step 3: Choose ciphertext modulus q with optimal ratio to t
# #         # #q = utils.next_prime(t * 800)  # 200x factor was optimal for 5-10 clients in our tests
# #         # q = utils.next_prime(t * num_clients * 100)
# #         # print(f"q: {q}")

# #         # # Step 4: Set noise standard deviation
# #         # std = 4  # Optimal for 5-10 clients in our tests
# #         # print(f"std: {std}")

# #         # # Initialize RLWE with these parameters
# #         # rlwe = RLWE(n, q, t, std)



# #         # Revised RLWE Parameter Calculation

# #         WEIGHT_DECIMALS = 4  # Keep your current decimal precision
# #         model = CNN(WEIGHT_DECIMALS)
# #         utils.set_initial_params(model)
# #         params, _ = utils.get_flat_weights(model)
# #         print("Initial parameters", params[0:20])

# #         # Step 1: Calculate ring dimension (keep your current approach)
# #         num_weights = len(params)
# #         n = 2 ** math.ceil(math.log2(num_weights))
# #         print(f"n: {n}")

# #         # Step 2: Choose plaintext modulus t based on our testing
# #         num_clients = 6  # Your actual client count
# #         max_weight_value = 10 ** WEIGHT_DECIMALS  # 10,000 for 4 decimal places
# #         t = utils.next_prime(num_clients * max_weight_value * 4)  # Factor of 4 from our testing
# #         print(f"t: {t}")

# #         # Step 3: Choose ciphertext modulus q with optimal ratio to t
# #         q = utils.next_prime(t * num_clients * 100)
# #         print(f"q: {q}")

# #         # Step 4: Set noise standard deviation
# #         std = 4  # Optimal for 5-10 clients in our tests
# #         print(f"std: {std}")

# #         # Initialize RLWE with these parameters
# #         rlwe = RLWE(n, q, t, std)       


# #         # Custom Strategy
# #         strategy = CustomFedAvg(
# #             min_available_clients=num_clients,
# #             min_fit_clients=num_clients,             
# #             evaluate_fn=get_evaluate_fn(model),
# #             on_fit_config_fn=fit_round,
# #             rlwe_instance=rlwe,
# #         )

# #         # Define Server and Client Manager
# #         client_manager = fl.server.SimpleClientManager()
# #         server = fl.server.Server(
# #             strategy=strategy,
# #             client_manager=client_manager
# #         )

# #         # Start Server
# #         # Start Server
# #         fl.server.start_server(
# #             server_address="0.0.0.0:8080",
# #             server=server,
# #             config=fl.server.ServerConfig(num_rounds=5),
# #         )


# #         # Calculate the execution time
# #         execution_time = time.time() - start_time

# #         # Print the execution time
# #         print("Execution time:", execution_time)

# #     mem_usage = memory_usage(measure_memory)
# #     print("Memory usage (in MB):", max(mem_usage))






# #new server with new data loader 


# # #!/usr/bin/env python3
# # # PROFILE_server.py

# # import socket
# # # Standard Libraries
# # import importlib
# # import math
# # import os
# # import sys
# # import time
# # import random
# # import flwr as fl
# # import numpy as np
# # from typing import Dict, List, Optional, Tuple
# # # Remove problematic import
# # # from load_covid import load_mnist_data
# # from flwr.common import NDArrays, Scalar
# # from flwr.server.strategy import FedAvg
# # from memory_profiler import memory_usage
# # from rlwe_xmkckks import RLWE
# # from sklearn.metrics import classification_report, log_loss
# # import multiprocessing
# # import threading
# # from sklearn.metrics import classification_report, log_loss, precision_score, recall_score, f1_score, confusion_matrix
# # # Replace with FederatedDataLoader
# # from federated_data_loader import FederatedDataLoader

# # # Get absolute paths to let a user run the script from anywhere
# # current_directory = os.path.dirname(os.path.abspath(__file__))
# # parent_directory = os.path.basename(current_directory)
# # working_directory = os.getcwd()
# # # Add parent directory to Python's module search path
# # sys.path.append(os.path.join(current_directory, '..'))
# # # Compare paths
# # if current_directory == working_directory:
# #     from cnn import CNN
# #     import utils
# # else:
# #     # Add current directory to Python's module search path
# #     CNN = importlib.import_module(f"{parent_directory}.cnn").CNN
# #     import utils


# # def fit_round(server_round: int) -> Dict:
# #     """Send round number to client."""
# #     return {"server_round": server_round}


# # def get_evaluate_fn(model):
# #     """Return an evaluation function for server-side evaluation."""
# #     # Create data loader and get test data
# #     data_loader = FederatedDataLoader(
# #         dataset_name='mnist',
# #         num_clients=6,  # Match your client count
# #         seed=42
# #     )
    
# #     # Get test data
# #     X_test, y_test = data_loader.get_test_data()

# #     # The 'evaluate' function will be called after every round
# #     def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
# #         utils.set_model_params(model, parameters)

# #         y_pred = model.model.predict(X_test)
# #         predicted = np.argmax(y_pred, axis=-1)

# #         # Already fixed:
# #         true_labels = y_test  # The data loader returns labels in the right format

# #         accuracy = np.equal(true_labels, predicted).mean()
# #         loss = log_loss(y_test, y_pred)

# #         print(classification_report(true_labels, predicted))

# #         return loss, {"accuracy": accuracy}

# #     return evaluate


# # class CustomFedAvg(FedAvg):
# #     def __init__(
# #         self, 
# #         rlwe_instance: RLWE, 
# #         num_groups: int = 1,  # Default number of buckets/groups
# #         eval_fn=None,
# #         *args, 
# #         **kwargs
# #     ):
# #         super().__init__(*args, **kwargs)
# #         self.rlwe = rlwe_instance
# #         self.num_groups = num_groups  # Number of buckets to divide clients into
# #         self.bucket_models = {}  # Store models for each bucket
# #         self.bucket_weights = {}  # Store weights for each bucket
# #         self.evaluate_fn = eval_fn  # Function for evaluating bucket models


# # # Start Flower server for five rounds of federated learning
# # if __name__ == "__main__": 
# #     def measure_memory():
# #         # Measure the execution time
# #         start_time = time.time()

# #         # Revised RLWE Parameter Calculation
# #         WEIGHT_DECIMALS = 4  # Keep your current decimal precision
# #         model = CNN(WEIGHT_DECIMALS)
# #         utils.set_initial_params(model)
# #         params, _ = utils.get_flat_weights(model)
# #         print("Initial parameters", params[0:20])

# #         # Step 1: Calculate ring dimension (keep your current approach)
# #         num_weights = len(params)
# #         n = 2 ** math.ceil(math.log2(num_weights))
# #         print(f"n: {n}")

# #         # Step 2: Choose plaintext modulus t based on our testing
# #         num_clients = 6  # Your actual client count
# #         max_weight_value = 10 ** WEIGHT_DECIMALS  # 10,000 for 4 decimal places
# #         t = utils.next_prime(num_clients * max_weight_value * 4)  # Factor of 4 from our testing
# #         print(f"t: {t}")

# #         # Step 3: Choose ciphertext modulus q with optimal ratio to t
# #         q = utils.next_prime(t * num_clients * 100)
# #         print(f"q: {q}")

# #         # Step 4: Set noise standard deviation
# #         std = 4  # Optimal for 5-10 clients in our tests
# #         print(f"std: {std}")

# #         # Initialize RLWE with these parameters
# #         rlwe = RLWE(n, q, t, std)       

# #         # Custom Strategy
# #         strategy = CustomFedAvg(
# #             min_available_clients=num_clients,
# #             min_fit_clients=num_clients,             
# #             evaluate_fn=get_evaluate_fn(model),
# #             on_fit_config_fn=fit_round,
# #             rlwe_instance=rlwe,
# #         )

# #         # Define Server and Client Manager
# #         client_manager = fl.server.SimpleClientManager()
# #         server = fl.server.Server(
# #             strategy=strategy,
# #             client_manager=client_manager
# #         )

# #         # Start Server
# #         fl.server.start_server(
# #             server_address="0.0.0.0:8080",
# #             server=server,
# #             config=fl.server.ServerConfig(num_rounds=10),
# #         )

# #         # Calculate the execution time
# #         execution_time = time.time() - start_time

# #         # Print the execution time
# #         print("Execution time:", execution_time)

# #     mem_usage = memory_usage(measure_memory)
# #     print("Memory usage (in MB):", max(mem_usage))



# # #!/usr/bin/env python3
# # # PROFILE_server.py

# # import socket
# # # Standard Libraries
# # import importlib
# # import math
# # import os
# # import sys
# # import time
# # import random
# # import flwr as fl
# # import numpy as np
# # from typing import Dict, List, Optional, Tuple
# # # Remove problematic import
# # # from load_covid import load_mnist_data
# # from flwr.common import NDArrays, Scalar
# # from flwr.server.strategy import FedAvg
# # from memory_profiler import memory_usage
# # from rlwe_xmkckks import RLWE
# # from sklearn.metrics import classification_report, log_loss
# # import multiprocessing
# # import threading
# # from sklearn.metrics import classification_report, log_loss, precision_score, recall_score, f1_score, confusion_matrix
# # # Replace with FederatedDataLoader
# # from federated_data_loader import FederatedDataLoader

# # # Add these two imports for metrics saving
# # import json
# # import csv

# # # Get absolute paths to let a user run the script from anywhere
# # current_directory = os.path.dirname(os.path.abspath(__file__))
# # parent_directory = os.path.basename(current_directory)
# # working_directory = os.getcwd()
# # # Add parent directory to Python's module search path
# # sys.path.append(os.path.join(current_directory, '..'))
# # # Compare paths
# # if current_directory == working_directory:
# #     from cnn import CNN
# #     import utils
# # else:
# #     # Add current directory to Python's module search path
# #     CNN = importlib.import_module(f"{parent_directory}.cnn").CNN
# #     import utils


# # def fit_round(server_round: int) -> Dict:
# #     """Send round number to client."""
# #     return {"server_round": server_round}


# # def get_evaluate_fn(model):
# #     """Return an evaluation function for server-side evaluation."""
# #     # Create data loader and get test data
# #     data_loader = FederatedDataLoader(
# #         dataset_name='mnist',
# #         num_clients=6,  # Match your client count
# #         seed=42
# #     )
    
# #     # Get test data
# #     X_test, y_test = data_loader.get_test_data()

# #     # The 'evaluate' function will be called after every round
# #     def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
# #         utils.set_model_params(model, parameters)

# #         y_pred = model.model.predict(X_test)
# #         predicted = np.argmax(y_pred, axis=-1)

# #         # Already fixed:
# #         true_labels = y_test  # The data loader returns labels in the right format

# #         accuracy = np.equal(true_labels, predicted).mean()
# #         loss = log_loss(y_test, y_pred)

# #         print(classification_report(true_labels, predicted))

# #         # ADDITION: Save evaluation metrics
# #         eval_metrics = {
# #             'round': server_round,
# #             'server_accuracy': accuracy,
# #             'server_loss': loss,
# #             'timestamp': time.time()
# #         }
# #         save_metrics(eval_metrics, "server_evaluation")

# #         return loss, {"accuracy": accuracy}

# #     return evaluate


# # class CustomFedAvg(FedAvg):
# #     def __init__(
# #         self, 
# #         rlwe_instance: RLWE, 
# #         num_groups: int = 1,  # Default number of buckets/groups
# #         eval_fn=None,
# #         *args, 
# #         **kwargs
# #     ):
# #         super().__init__(*args, **kwargs)
# #         self.rlwe = rlwe_instance
# #         self.num_groups = num_groups  # Number of buckets to divide clients into
# #         self.bucket_models = {}  # Store models for each bucket
# #         self.bucket_weights = {}  # Store weights for each bucket
# #         self.evaluate_fn = eval_fn  # Function for evaluating bucket models


# # # ADDITION: Simple metrics saving function
# # def save_metrics(metrics, metric_type="server_metrics"):
# #     """Save metrics to file with minimal overhead"""
# #     session_id = getattr(save_metrics, 'session_id', f"server_session_{int(time.time())}")
# #     save_metrics.session_id = session_id  # Store as function attribute
    
# #     metrics_dir = f"metrics/{session_id}"
# #     os.makedirs(metrics_dir, exist_ok=True)
    
# #     # Save to JSONL file
# #     jsonl_file = f"{metrics_dir}/{metric_type}.jsonl"
# #     with open(jsonl_file, 'a') as f:
# #         f.write(json.dumps(metrics) + '\n')
    
# #     print(f"Metrics saved to {jsonl_file}")


# # # Start Flower server for five rounds of federated learning
# # if __name__ == "__main__": 
# #     def measure_memory():
# #         # Measure the execution time
# #         start_time = time.time()

# #         # Revised RLWE Parameter Calculation
# #         WEIGHT_DECIMALS = 4  # Keep your current decimal precision
# #         model = CNN(WEIGHT_DECIMALS)
# #         utils.set_initial_params(model)
# #         params, _ = utils.get_flat_weights(model)
# #         print("Initial parameters", params[0:20])

# #         # Step 1: Calculate ring dimension (keep your current approach)
# #         num_weights = len(params)
# #         n = 2 ** math.ceil(math.log2(num_weights))
# #         print(f"n: {n}")

# #         # Step 2: Choose plaintext modulus t based on our testing
# #         num_clients = 6  # Your actual client count
# #         max_weight_value = 10 ** WEIGHT_DECIMALS  # 10,000 for 4 decimal places
# #         t = utils.next_prime(num_clients * max_weight_value * 4)  # Factor of 4 from our testing
# #         print(f"t: {t}")

# #         # Step 3: Choose ciphertext modulus q with optimal ratio to t
# #         q = utils.next_prime(t * num_clients * 100)
# #         print(f"q: {q}")

# #         # Step 4: Set noise standard deviation
# #         std = 4  # Optimal for 5-10 clients in our tests
# #         print(f"std: {std}")

# #         # Initialize RLWE with these parameters
# #         rlwe = RLWE(n, q, t, std)
        
# #         # ADDITION: Save initialization metrics
# #         init_metrics = {
# #             'weight_decimals': WEIGHT_DECIMALS,
# #             'num_weights': num_weights,
# #             'n': n,
# #             'q': int(q),
# #             't': int(t),
# #             'std': std,
# #             'num_clients': num_clients,
# #             'timestamp': time.time()
# #         }
# #         save_metrics(init_metrics, "initialization")

# #         # Custom Strategy
# #         strategy = CustomFedAvg(
# #             min_available_clients=num_clients,
# #             min_fit_clients=num_clients,             
# #             evaluate_fn=get_evaluate_fn(model),
# #             on_fit_config_fn=fit_round,
# #             rlwe_instance=rlwe,
# #         )

# #         # Define Server and Client Manager
# #         client_manager = fl.server.SimpleClientManager()
# #         server = fl.server.Server(
# #             strategy=strategy,
# #             client_manager=client_manager
# #         )

# #         # Start Server
# #         fl.server.start_server(
# #             server_address="0.0.0.0:8080",
# #             server=server,
# #             config=fl.server.ServerConfig(num_rounds=10),
# #         )

# #         # Calculate the execution time
# #         execution_time = time.time() - start_time

# #         # Print the execution time
# #         print("Execution time:", execution_time)
        
# #         # ADDITION: Save final metrics
# #         final_metrics = {
# #             'total_execution_time': execution_time,
# #             'timestamp': time.time()
# #         }
# #         save_metrics(final_metrics, "execution_summary")

# #     mem_usage = memory_usage(measure_memory)
# #     max_mem = max(mem_usage)
# #     print("Memory usage (in MB):", max_mem)
    
# #     # ADDITION: Save memory usage
# #     memory_metrics = {
# #         'max_memory_mb': max_mem,
# #         'timestamp': time.time()
# #     }
# #     save_metrics(memory_metrics, "memory_usage")




#!/usr/bin/env python3
# PROFILE_server.py

import socket
# Standard Libraries
import importlib
import math
import os
import sys
import time
import random
import flwr as fl
import numpy as np
from typing import Dict, List, Optional, Tuple
from flwr.common import NDArrays, Scalar
from flwr.server.strategy import FedAvg
from memory_profiler import memory_usage
from rlwe_xmkckks import RLWE
from sklearn.metrics import classification_report, log_loss
import multiprocessing
import threading
from sklearn.metrics import classification_report, log_loss, precision_score, recall_score, f1_score, confusion_matrix
# Replace with FederatedDataLoader
from federated_data_loader import FederatedDataLoader
import tensorflow as tf
from datetime import datetime
# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("? Memory growth enabled for GPU")
    except RuntimeError as e:
        print("? Failed to set memory growth:", e)
# Add these imports for argument parsing and metrics saving
import json
import csv
import argparse

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


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    print(f"\n{'='*80}")
    print(f"🔄 SERVER: Starting Round {server_round}")
    print(f"{'='*80}")
    return {"server_round": server_round}


def get_evaluate_fn(model, dataset_name='mnist'):
    """Return an evaluation function for server-side evaluation."""
    # Create data loader with selected dataset
    data_loader = FederatedDataLoader(
        dataset_name=dataset_name,  # Use the passed dataset name
        num_clients=10,  # Match your client count
        seed=42
    )
    
    # Get test data
    X_test, y_test = data_loader.get_test_data()

    # The 'evaluate' function will be called after every round
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        print(f"📊 SERVER: Evaluating Round {server_round}...")
        utils.set_model_params(model, parameters)

        y_pred = model.model.predict(X_test)
        predicted = np.argmax(y_pred, axis=-1)

        # Already fixed:
        true_labels = y_test  # The data loader returns labels in the right format

        accuracy = np.equal(true_labels, predicted).mean()
        loss = log_loss(y_test, y_pred)

        print(f"✅ SERVER Round {server_round}: Accuracy={accuracy:.4f}, Loss={loss:.4f}")
        print(classification_report(true_labels, predicted))

        # ADDITION: Save evaluation metrics
        eval_metrics = {
            'round': server_round,
            'server_accuracy': accuracy,
            'server_loss': loss,
            'timestamp': time.time(),
            'dataset': dataset_name  # Add dataset info to metrics
        }
        save_metrics(eval_metrics, "server_evaluation")

        return loss, {"accuracy": accuracy}

    return evaluate



class CustomFedAvg(FedAvg):
    def __init__(
        self, 
        rlwe_instance: RLWE, 
        num_groups: int = 1,  # Default number of buckets/groups
        num_buckets: int = 1,  # ADD THIS LINE
        eval_fn=None,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rlwe = rlwe_instance
        self.num_groups = num_groups  # Number of buckets to divide clients into
        self.num_buckets = num_buckets  # ADD THIS LINE
        self.bucket_models = {}  # Store models for each bucket
        self.bucket_weights = {}  # Store weights for each bucket
        self.evaluate_fn = eval_fn  # Function for evaluating bucket models


# ADDITION: Simple metrics saving function
# ADDITION: Simple metrics saving function
def save_metrics(metrics, metric_type="server_metrics"):
    """Save metrics to file with minimal overhead"""
    # Get experiment parameters from the metrics if available, otherwise use defaults
    dataset = metrics.get('dataset', 'unknown')
    num_clients = metrics.get('num_clients', 'unknown') 
    num_buckets = metrics.get('num_buckets', 'unknown')
    epsilon = metrics.get('epsilon_per_round', 'unknown')
    
    # Get attack_type from args if available, otherwise from metrics
    if 'args' in globals():
        attack_type = globals()['args'].attack_type
    else:
        attack_type = metrics.get('attack_type', 'none')
    
    # Create descriptive session ID with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"dataset_{dataset}_clients_{num_clients}_buckets_{num_buckets}_attack_{attack_type}_eps_{epsilon}_{timestamp}"
    
    # Store session_id and directory as function attributes
    if not hasattr(save_metrics, 'session_id'):
        save_metrics.session_id = session_id
        save_metrics.session_dir = f"metrics/{session_id}"
        print(f"[PROFILE] Creating experiment directory: {session_id}")
    
    session_id = save_metrics.session_id
    metrics_dir = save_metrics.session_dir
    
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Save to JSONL file
    jsonl_file = f"{metrics_dir}/{metric_type}.jsonl"
    with open(jsonl_file, 'a') as f:
        f.write(json.dumps(metrics) + '\n')
    
    # Also save to CSV for easy analysis
    csv_file = f"{metrics_dir}/{metric_type}.csv"
    _append_to_csv(metrics, csv_file)
    
    print(f"Metrics saved to {jsonl_file}")

def _append_to_csv(metrics, csv_file):
    """Helper function to append metrics to CSV file"""
    import csv
    import os
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(csv_file)
    
    # Flatten nested dictionaries for CSV storage
    flat_metrics = _flatten_dict(metrics)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat_metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat_metrics)

def _flatten_dict(d, parent_key='', sep='_'):
    """Flatten nested dictionary for CSV storage"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            # Skip complex nested structures
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='PROFILE Server')
    parser.add_argument('--dataset', type=str, default='cifar10', 
                       choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100'], 
                       help='Dataset to use')
    parser.add_argument('--num_buckets', type=int, default=1,
                       help='Number of buckets for client grouping')
    parser.add_argument('--num_clients', type=int, default=15,
                       help='Number of clients in federated learning')
    parser.add_argument('--num_rounds', type=int, default=50,
                       help='Number of federated learning rounds')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of local training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for local training')
    parser.add_argument('--attack_type', type=str, default='none',
                       choices=['none', 'label_flip', 'targeted', 'random', 'backdoor', 'fang', 'min_max'],
                       help='Type of attack being tested')
    parser.add_argument('--attack_percent', type=int, default=30,
                       help='Percentage of malicious clients')
    parser.add_argument('--num_malicious', type=int, default=6,
                       help='Number of malicious clients')
    parser.add_argument('--poison_ratio', type=float, default=0.0,
                       help='Ratio of data poisoning for attack')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    
    # Ablation study flags
    parser.add_argument('--disable_bucketing', action='store_true',
                       help='Disable bucketing mechanism (ablation study)')
    parser.add_argument('--disable_he', action='store_true',
                       help='Disable homomorphic encryption (ablation study)')
    parser.add_argument('--disable_validation', action='store_true',
                       help='Disable validator ensemble (ablation study)')
    parser.add_argument('--disable_dp', action='store_true',
                       help='Disable differential privacy (ablation study)')
    
    return parser.parse_args()
def validate_security_constraint(num_clients, num_buckets, min_clients_per_bucket=3):
    """Validate that bucket configuration meets security requirements"""
    clients_per_bucket = num_clients // num_buckets
    
    if clients_per_bucket < min_clients_per_bucket:
        print(f"⚠️  SECURITY CONSTRAINT WARNING:")
        print(f"   {clients_per_bucket} clients per bucket < {min_clients_per_bucket} minimum")
        print(f"   Consider reducing buckets or increasing clients")
        # Don't raise error, just warn
    
    return clients_per_bucket >= min_clients_per_bucket


if __name__ == "__main__": 
    # Add argument parsing
    args = parse_args()
    
    def measure_memory():
        start_time = time.time()
        globals()['args'] = args        
        # Measure the execution time

        # Revised RLWE Parameter Calculation
        WEIGHT_DECIMALS = 4
        
        # Initialize model with dataset support
        model = CNN.create_for_dataset(args.dataset, WEIGHT_DECIMALS)
        model.set_initial_params()
        
        # Get flat weights using the model's method
        weights = model.get_weights()
        flat_params = []
        for weight in weights:
            weight_flat = tf.reshape(weight, [-1])
            flat_params.extend(weight_flat.numpy().tolist())
        
        print("Initial parameters", flat_params[0:20])

        # Step 1: Calculate ring dimension
        num_weights = len(flat_params)
        n = 2 ** math.ceil(math.log2(num_weights))
        print(f"n: {n}")

        # CHANGE THIS LINE: Use args.num_clients instead of hardcoded 6
        num_clients = args.num_clients  # CHANGED FROM: num_clients = 6
        
        # ADD VALIDATION
        is_secure = validate_security_constraint(num_clients, args.num_buckets)
        if is_secure:
            print(f"✅ Security constraint satisfied: {num_clients} clients, {args.num_buckets} buckets")
        else:
            print(f"⚠️  Security constraint warning: Consider adjusting configuration")

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
        
        # ADDITION: Save initialization metrics (UPDATE TO INCLUDE num_clients)
        # ADDITION: Save initialization metrics (UPDATE TO INCLUDE all params)
        init_metrics = {
            'weight_decimals': WEIGHT_DECIMALS,
            'num_weights': num_weights,
            'n': n,
            'q': int(q),
            't': int(t),
            'std': std,
            'num_clients': num_clients,
            'num_buckets': args.num_buckets,
            'dataset': args.dataset,
            'epsilon_per_round': 1.0,  # Add your default epsilon here
            'attack_type': args.attack_type,  # Include attack type
            'timestamp': time.time()
        }
        save_metrics(init_metrics, "experiment_config")
        
        # Get the session directory that was created
        session_dir = getattr(save_metrics, 'session_dir', None)

        # Custom Strategy with dataset support
        strategy = CustomFedAvg(
            min_available_clients=num_clients,  # CHANGED FROM hardcoded 6
            min_fit_clients=num_clients,        # CHANGED FROM hardcoded 6
            evaluate_fn=None,  # TEMPORARILY DISABLED - was hanging on model.predict()
            on_fit_config_fn=fit_round,
            rlwe_instance=rlwe,
            num_buckets=args.num_buckets,
        )

        # REST OF THE FUNCTION REMAINS THE SAME...
        # Define Server and Client Manager
        client_manager = fl.server.SimpleClientManager()
        server = fl.server.Server(
            strategy=strategy,
            client_manager=client_manager,
            session_dir=session_dir 
        )

        # Start Server
        fl.server.start_server(
            server_address="0.0.0.0:8081",
            server=server,
            config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        )

        # Calculate the execution time
        execution_time = time.time() - start_time
        print("Execution time:", execution_time)
        
        # ADDITION: Save final metrics
        experiment_summary = {
            'total_execution_time': execution_time,
            'dataset': args.dataset,
            'num_clients': num_clients,
            'num_buckets': args.num_buckets,
            'epsilon_per_round': 1.0,  # Add your epsilon value
            'attack_type': args.attack_type,
            'status': 'completed',
            'timestamp': time.time()
        }
        save_metrics(experiment_summary, "experiment_summary")

    # REST REMAINS THE SAME
    mem_usage = memory_usage(measure_memory)
    max_mem = max(mem_usage)
    print("Memory usage (in MB):", max_mem)
    
    memory_metrics = {
        'max_memory_mb': max_mem,
        'dataset': args.dataset,
        'num_clients': args.num_clients,  # ADD THIS
        'timestamp': time.time()
    }
    save_metrics(memory_metrics, "memory_usage")





# #!/usr/bin/env python3
# # Enhanced PROFILE_server.py with organized metrics

# import socket
# import importlib
# import math
# import os
# import sys
# import time
# import random
# import flwr as fl
# import numpy as np
# from typing import Dict, List, Optional, Tuple
# from flwr.common import NDArrays, Scalar
# from flwr.server.strategy import FedAvg
# from memory_profiler import memory_usage
# from rlwe_xmkckks import RLWE
# from sklearn.metrics import classification_report, log_loss
# import multiprocessing
# import threading
# from sklearn.metrics import classification_report, log_loss, precision_score, recall_score, f1_score, confusion_matrix
# from federated_data_loader import FederatedDataLoader
# import tensorflow as tf
# import json
# import csv
# import argparse
# from datetime import datetime

# # Simple metrics manager (inline to avoid external dependencies)
# class ExperimentMetricsManager:
#     def __init__(self, dataset: str, num_clients: int, num_buckets: int, 
#                  attack_type: str = "none", poison_ratio: float = 0.0, epsilon: float = 1.0):
        
#         # Generate experiment name
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         self.experiment_name = f"profile_{dataset}_c{num_clients}_b{num_buckets}_eps{epsilon:.1f}_{timestamp}"
        
#         if attack_type != "none":
#             self.experiment_name += f"_{attack_type}_p{poison_ratio:.1f}"
        
#         self.experiment_dir = os.path.join("experiments", self.experiment_name)
        
#         # Create directory structure
#         subdirs = ["server_metrics", "client_metrics", "privacy", "detection", "logs"]
#         for subdir in subdirs:
#             os.makedirs(os.path.join(self.experiment_dir, subdir), exist_ok=True)
        
#         # Create individual client folders
#         for i in range(num_clients):
#             os.makedirs(os.path.join(self.experiment_dir, "client_metrics", f"client_{i}"), exist_ok=True)
        
#         # Save config
#         config = {
#             'experiment_name': self.experiment_name,
#             'dataset': dataset, 'num_clients': num_clients, 'num_buckets': num_buckets,
#             'attack_type': attack_type, 'poison_ratio': poison_ratio, 'epsilon': epsilon,
#             'created_at': datetime.now().isoformat()
#         }
        
#         with open(os.path.join(self.experiment_dir, "experiment_config.json"), 'w') as f:
#             json.dump(config, f, indent=4)
        
#         print(f"📁 Experiment Directory: {self.experiment_dir}")
    
#     def save_server_metrics(self, metrics: Dict, filename: str = "server_metrics"):
#         """Save server metrics"""
#         try:
#             metrics = self._convert_numpy(metrics)
#             metrics['timestamp'] = time.time()
            
#             # Save as JSONL
#             jsonl_file = os.path.join(self.experiment_dir, "server_metrics", f"{filename}.jsonl")
#             with open(jsonl_file, 'a') as f:
#                 f.write(json.dumps(metrics) + '\n')
            
#             # Save as CSV
#             csv_file = os.path.join(self.experiment_dir, "server_metrics", f"{filename}.csv")
#             self._save_to_csv(metrics, csv_file)
            
#         except Exception as e:
#             print(f"❌ Error saving server metrics: {e}")
    
#     def save_privacy_metrics(self, metrics: Dict, filename: str = "privacy_metrics"):
#         """Save privacy metrics"""
#         try:
#             metrics = self._convert_numpy(metrics)
#             metrics['timestamp'] = time.time()
            
#             privacy_file = os.path.join(self.experiment_dir, "privacy", f"{filename}.jsonl")
#             with open(privacy_file, 'a') as f:
#                 f.write(json.dumps(metrics) + '\n')
                
#             csv_file = os.path.join(self.experiment_dir, "privacy", f"{filename}.csv")
#             self._save_to_csv(metrics, csv_file)
            
#         except Exception as e:
#             print(f"❌ Error saving privacy metrics: {e}")
    
#     def save_detection_metrics(self, metrics: Dict, filename: str = "detection_metrics"):
#         """Save detection metrics"""
#         try:
#             metrics = self._convert_numpy(metrics)
#             metrics['timestamp'] = time.time()
            
#             detection_file = os.path.join(self.experiment_dir, "detection", f"{filename}.jsonl")
#             with open(detection_file, 'a') as f:
#                 f.write(json.dumps(metrics) + '\n')
                
#             csv_file = os.path.join(self.experiment_dir, "detection", f"{filename}.csv")
#             self._save_to_csv(metrics, csv_file)
            
#         except Exception as e:
#             print(f"❌ Error saving detection metrics: {e}")
    
#     def _convert_numpy(self, obj):
#         """Convert NumPy types for JSON"""
#         if isinstance(obj, dict):
#             return {k: self._convert_numpy(v) for k, v in obj.items()}
#         elif isinstance(obj, list):
#             return [self._convert_numpy(item) for item in obj]
#         elif isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.bool_):
#             return bool(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return obj
    
#     def _save_to_csv(self, metrics: Dict, csv_file: str):
#         """Save to CSV with flattened structure"""
#         try:
#             flat_metrics = self._flatten_dict(metrics)
#             file_exists = os.path.isfile(csv_file)
            
#             with open(csv_file, 'a', newline='') as f:
#                 writer = csv.DictWriter(f, fieldnames=sorted(flat_metrics.keys()))
#                 if not file_exists:
#                     writer.writeheader()
#                 writer.writerow(flat_metrics)
#         except Exception as e:
#             print(f"⚠️ CSV save warning: {e}")
    
#     def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_'):
#         """Flatten nested dict for CSV"""
#         items = []
#         for k, v in d.items():
#             new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
#             if isinstance(v, dict):
#                 items.extend(self._flatten_dict(v, new_key, sep=sep).items())
#             else:
#                 items.append((new_key, str(v)))
#         return dict(items)
    


# # Enable GPU memory growth
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         print("✅ Memory growth enabled for GPU")
#     except RuntimeError as e:
#         print("❌ Failed to set memory growth:", e)

# # Get absolute paths
# current_directory = os.path.dirname(os.path.abspath(__file__))
# parent_directory = os.path.basename(current_directory)
# working_directory = os.getcwd()
# sys.path.append(os.path.join(current_directory, '..'))

# if current_directory == working_directory:
#     from cnn import CNN
#     import utils
# else:
#     CNN = importlib.import_module(f"{parent_directory}.cnn").CNN
#     import utils

# def fit_round(server_round: int) -> Dict:
#     """Send round number to client."""
#     return {"server_round": server_round}

# def get_evaluate_fn(model, metrics_manager, dataset_name='mnist'):
#     """Return evaluation function with metrics integration."""
#     data_loader = FederatedDataLoader(
#         dataset_name=dataset_name,
#         num_clients=6,
#         seed=42
#     )
    
#     X_test, y_test = data_loader.get_test_data()

#     def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
#         utils.set_model_params(model, parameters)

#         y_pred = model.model.predict(X_test)
#         predicted = np.argmax(y_pred, axis=-1)
#         true_labels = y_test

#         accuracy = np.equal(true_labels, predicted).mean()
#         loss = log_loss(y_test, y_pred)

#         print(classification_report(true_labels, predicted))

#         # Save enhanced evaluation metrics
#         eval_metrics = {
#             'round': server_round,
#             'server_accuracy': accuracy,
#             'server_loss': loss,
#             'dataset': dataset_name,
#             'detailed_classification': classification_report(true_labels, predicted, output_dict=True)
#         }
        
#         metrics_manager.save_server_metrics(eval_metrics, "evaluation")
#         return loss, {"accuracy": accuracy}

#     return evaluate

# class CustomFedAvg(FedAvg):
#     def __init__(self, rlwe_instance: RLWE, metrics_manager, num_groups: int = 1,
#                  num_buckets: int = 2, eval_fn=None, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.rlwe = rlwe_instance
#         self.metrics_manager = metrics_manager
#         self.num_groups = num_groups
#         self.num_buckets = num_buckets
#         self.bucket_models = {}
#         self.bucket_weights = {}
#         self.evaluate_fn = eval_fn

# def parse_args():
#     parser = argparse.ArgumentParser(description='Enhanced PROFILE Server')
#     parser.add_argument('--dataset', type=str, default='mnist', 
#                        choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar100'])
#     parser.add_argument('--num_buckets', type=int, default=2)
#     parser.add_argument('--num_clients', type=int, default=6)
#     parser.add_argument('--epsilon', type=float, default=1.0)
#     parser.add_argument('--attack_type', type=str, default='none')
#     parser.add_argument('--poison_ratio', type=float, default=0.0)
#     return parser.parse_args()

# def validate_security_constraint(num_clients, num_buckets, min_clients_per_bucket=3):
#     """Validate bucket configuration"""
#     clients_per_bucket = num_clients // num_buckets
    
#     if clients_per_bucket < min_clients_per_bucket:
#         print(f"⚠️ SECURITY WARNING: {clients_per_bucket} clients per bucket < {min_clients_per_bucket} minimum")
#     else:
#         print(f"✅ Security constraint satisfied: {clients_per_bucket} clients per bucket")
    
#     return clients_per_bucket >= min_clients_per_bucket

# def set_metrics_logger_in_strategy(self):
#     """Set the metrics logger in the strategy's evaluate function"""
#     if hasattr(self.strategy, 'evaluate_fn'):
#         # Create a new evaluate function that uses our metrics logger
#         original_evaluate_fn = self.strategy.evaluate_fn
        
#         def new_evaluate_fn(server_round, parameters, config):
#             result = original_evaluate_fn(server_round, parameters, config)
#             if result and len(result) == 2:
#                 loss, metrics_dict = result
#                 accuracy = metrics_dict.get('accuracy', 0.0)
#                 self.metrics_logger.record_server_evaluation(server_round, accuracy, loss, self.dataset)
#             return result
        
#         self.strategy.evaluate_fn = new_evaluate_fn


# if __name__ == "__main__": 
#     args = parse_args()
    
#     def measure_memory():
#         start_time = time.time()
#         globals()['args'] = args
#         # CREATE ENHANCED METRICS MANAGER
#         metrics_manager = ExperimentMetricsManager(
#             dataset=args.dataset,
#             num_clients=args.num_clients,
#             num_buckets=args.num_buckets,
#             epsilon=args.epsilon,
#             attack_type=args.attack_type,
#             poison_ratio=args.poison_ratio
#         )
        
#         print(f"🔬 Starting PROFILE experiment: {metrics_manager.experiment_name}")

#         # Initialize model and RLWE parameters
#         WEIGHT_DECIMALS = 4
#         model = CNN.create_for_dataset(args.dataset, WEIGHT_DECIMALS)
#         model.set_initial_params()
        
#         weights = model.get_weights()
#         flat_params = []
#         for weight in weights:
#             weight_flat = tf.reshape(weight, [-1])
#             flat_params.extend(weight_flat.numpy().tolist())
        
#         print("Initial parameters", flat_params[0:20])

#         # Calculate RLWE parameters
#         num_weights = len(flat_params)
#         n = 2 ** math.ceil(math.log2(num_weights))
#         print(f"n: {n}")

#         num_clients = args.num_clients
#         is_secure = validate_security_constraint(num_clients, args.num_buckets)

#         max_weight_value = 10 ** WEIGHT_DECIMALS
#         t = utils.next_prime(num_clients * max_weight_value * 4)
#         print(f"t: {t}")

#         q = utils.next_prime(t * num_clients * 100)
#         print(f"q: {q}")

#         std = 4
#         print(f"std: {std}")

#         rlwe = RLWE(n, q, t, std)
        
#         # ENHANCED: Save comprehensive initialization metrics
#         init_metrics = {
#             'weight_decimals': WEIGHT_DECIMALS,
#             'num_weights': num_weights,
#             'n': n,
#             'q': int(q),
#             't': int(t),
#             'std': std,
#             'num_clients': num_clients,
#             'num_buckets': args.num_buckets,
#             'dataset': args.dataset,
#             'security_constraint_satisfied': is_secure,
#             'model_architecture': {
#                 'total_parameters': len(flat_params),
#                 'sample_parameters': flat_params[0:10]
#             }
#         }
#         metrics_manager.save_server_metrics(init_metrics, "initialization")

#         # Create enhanced strategy
#         strategy = CustomFedAvg(
#             min_available_clients=num_clients,
#             min_fit_clients=num_clients,
#             evaluate_fn=get_evaluate_fn(model, metrics_manager, args.dataset),
#             on_fit_config_fn=fit_round,
#             rlwe_instance=rlwe,
#             metrics_manager=metrics_manager,
#             num_buckets=args.num_buckets,
#         )

#         # Privacy tracking setup
#         epsilon_per_round = args.epsilon
#         delta = 1e-5
#         privacy_metrics_list = []

#         # Simulate privacy metrics for each round (you'll integrate this with your actual FL loop)
#         for round_num in range(1, 6):  # 5 rounds for demo
#             # Calculate privacy using different composition methods
#             min_bucket_size = max(1, num_clients // args.num_buckets)
#             sensitivity = 2.0
#             sigma = (sensitivity/min_bucket_size * np.sqrt(2 * np.log(1.25/delta))) / epsilon_per_round
            
#             # Basic composition
#             epsilon_naive = round_num * epsilon_per_round / np.sqrt(min_bucket_size)
            
#             # Moments Accountant (simplified)
#             epsilon_ma = epsilon_per_round * np.sqrt(round_num) / np.sqrt(min_bucket_size)
            
#             # zCDP composition
#             rho_total = round_num * (epsilon_per_round**2 / 2)
#             epsilon_zcdp = rho_total + 2 * np.sqrt(rho_total * np.log(1/delta))
            
#             privacy_metrics = {
#                 'round': round_num,
#                 'epsilon_naive': epsilon_naive,
#                 'epsilon_ma': epsilon_ma,
#                 'epsilon_zcdp': epsilon_zcdp,
#                 'delta': delta,
#                 'min_bucket_size': min_bucket_size,
#                 'avg_bucket_size': min_bucket_size,
#                 'noise_multiplier': sigma / sensitivity,
#                 'avg_noise_level': sigma
#             }
            
#             metrics_manager.save_privacy_metrics(privacy_metrics, "round_privacy")
#             privacy_metrics_list.append(privacy_metrics)
            
#             print(f"[PROFILE] Round {round_num} privacy: ε_MA={epsilon_ma:.4f}, ε_zCDP={epsilon_zcdp:.4f}")
            
#             # Simulate detection metrics
#             detection_metrics = {
#                 'round': round_num,
#                 'accuracy': 0.95 + np.random.normal(0, 0.02),
#                 'precision': 0.94 + np.random.normal(0, 0.03),
#                 'recall': 0.93 + np.random.normal(0, 0.03),
#                 'f1_score': 0.935 + np.random.normal(0, 0.025),
#                 'true_positives': int(45 + np.random.normal(0, 3)),
#                 'false_positives': int(2 + np.random.normal(0, 1)),
#                 'true_negatives': int(48 + np.random.normal(0, 2)),
#                 'false_negatives': int(5 + np.random.normal(0, 2))
#             }
            
#             metrics_manager.save_detection_metrics(detection_metrics, "round_detection")

#         # Setup and start server
#         # client_manager = fl.server.SimpleClientManager()
#         # server = fl.server.Server(
#         #     strategy=strategy,
#         #     client_manager=client_manager
#         # )

#         # AFTER creating the server:
#         client_manager = fl.server.SimpleClientManager()
#         server = fl.server.Server(
#             strategy=strategy,
#             client_manager=client_manager,
#             dataset=args.dataset,
#             num_clients=num_clients,
#             num_buckets=args.num_buckets
#         )



#         print(f"🚀 Starting FL server with {num_clients} clients and {args.num_buckets} buckets")
        
#         fl.server.start_server(
#             server_address="0.0.0.0:8080",
#             server=server,
#             config=fl.server.ServerConfig(num_rounds=5),
#         )

#         execution_time = time.time() - start_time
        
#         # ENHANCED: Save comprehensive final summary
#         final_privacy = privacy_metrics_list[-1] if privacy_metrics_list else {}
#         final_metrics = {
#             'total_execution_time': execution_time,
#             'total_rounds': 5,
#             'dataset': args.dataset,
#             'num_clients': num_clients,
#             'num_buckets': args.num_buckets,
#             'attack_type': args.attack_type,
#             'poison_ratio': args.poison_ratio,
#             'final_privacy_budget': final_privacy.get('epsilon_ma', 0.0),
#             'privacy_composition_method': 'moments_accountant',
#             'security_constraint_satisfied': is_secure,
#             'experiment_success': True,
#             'notes': f"PROFILE experiment completed with {args.dataset} dataset, {args.attack_type} attack"
#         }
        
#         metrics_manager.save_server_metrics(final_metrics, "final_summary")
        
#         print(f"✅ PROFILE experiment completed successfully!")
#         print(f"📊 Total execution time: {execution_time:.2f}s")
#         print(f"🔒 Final privacy budget (ε): {final_privacy.get('epsilon_ma', 0.0):.4f}")
#         print(f"📁 All metrics saved to: {metrics_manager.experiment_dir}")

#     # Measure memory and run experiment
#     mem_usage = memory_usage(measure_memory)
#     max_mem = max(mem_usage)
    
#     print(f"💾 Peak memory usage: {max_mem:.1f} MB")