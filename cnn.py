import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, accuracy_score

class CNN:
    def __init__(self, weight_decimals=8, input_shape=None, num_classes=None):
        self.model = None
        self.WEIGHT_DECIMALS = self.set_weight_decimals(weight_decimals)
        print("*********************RESNET***********************")
        # Set default values if not provided
        if input_shape is None:
            input_shape = (28, 28, 1)  # Default to MNIST
        if num_classes is None:
            num_classes = 10  # Default to 10 classes
            
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def set_weight_decimals(self, weight_decimals):
        if 2 <= weight_decimals <= 8:
            return weight_decimals
        return 8
    
    # NEW: Helper method to build a residual block
    def _resnet_block(self, x, filters, conv_shortcut=False, name=None):
        """A single residual block with optional shortcut convolution."""
        shortcut = x
        if conv_shortcut:
            shortcut = layers.Conv2D(filters, 1, strides=1, name=name + "_shortcut_conv")(shortcut)
            shortcut = layers.BatchNormalization(name=name + "_shortcut_bn")(shortcut)
        
        # First convolution
        x = layers.Conv2D(filters, 3, padding="same", name=name + "_conv1")(x)
        x = layers.BatchNormalization(name=name + "_bn1")(x)
        x = layers.Activation("relu", name=name + "_relu1")(x)
        
        # Second convolution
        x = layers.Conv2D(filters, 3, padding="same", name=name + "_conv2")(x)
        x = layers.BatchNormalization(name=name + "_bn2")(x)
        
        # Add shortcut and activate
        x = layers.add([x, shortcut], name=name + "_add")
        x = layers.Activation("relu", name=name + "_relu2")(x)
        return x

    # NEW: Method to create a lightweight ResNet model
    def _create_resnet_model(self):
        """Creates a lightweight ResNet-style model for larger datasets like CIFAR."""
        inputs = keras.Input(shape=self.input_shape)
        
        # Initial convolution
        x = layers.Conv2D(16, 3, padding="same", name="conv_initial")(inputs)
        x = layers.BatchNormalization(name="bn_initial")(x)
        x = layers.Activation("relu", name="relu_initial")(x)
        
        # First residual block stack
        x = self._resnet_block(x, 16, conv_shortcut=True, name="res_block_1")
        x = self._resnet_block(x, 16, name="res_block_2")
        
        # Second residual block stack
        x = layers.MaxPool2D(pool_size=(2, 2), name="pool1")(x)
        x = self._resnet_block(x, 32, conv_shortcut=True, name="res_block_3")
        x = self._resnet_block(x, 32, name="res_block_4")
        
        # Third residual block stack
        x = layers.MaxPool2D(pool_size=(2, 2), name="pool2")(x)
        x = self._resnet_block(x, 64, conv_shortcut=True, name="res_block_5")
        x = self._resnet_block(x, 64, name="res_block_6")
        
        # Final layers
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(128, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.5, name="dropout")(x)
        outputs = layers.Dense(self.num_classes, activation="softmax", name="output")(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name="resnet_model")
    
    # Existing LeNet-style model builder
    def _create_lenet_model(self):
        """Creates a LeNet-style model for smaller datasets like MNIST."""
        inputs = keras.Input(shape=self.input_shape)
        x = layers.Conv2D(16, (5, 5), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0005))(inputs)
        x = layers.AveragePooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(32, (5, 5), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0005))(x)
        x = layers.AveragePooling2D(pool_size=(2, 2))(x)
        
        # Adapt based on input size for deeper layers
        if self.input_shape[0] > 28:
            x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.0005))(x)
            x = layers.AveragePooling2D(pool_size=(2, 2))(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(120, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(84, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)

    # UPDATED: set_initial_params acts as a dispatcher
    def set_initial_params(self):
        if self.input_shape == (32, 32, 3):
            print("Creating ResNet-style model for CIFAR-10.")
            self._create_resnet_model()
        else:
            print("Creating LeNet-style model for MNIST/Fashion-MNIST.")
            self._create_lenet_model()

        self.model.compile(
            loss=keras.losses.sparse_categorical_crossentropy,
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            metrics=['accuracy']
        )
    
    # Model factory method
    @staticmethod
    def create_for_dataset(dataset_name, weight_decimals=8):
        """Factory method to create CNN model for specific datasets"""
        configs = {
            'mnist': {'input_shape': (28, 28, 1), 'num_classes': 10},
            'fashion_mnist': {'input_shape': (28, 28, 1), 'num_classes': 10},
            'cifar10': {'input_shape': (32, 32, 3), 'num_classes': 10},
            'cifar100': {'input_shape': (32, 32, 3), 'num_classes': 100}
        }
        
        if dataset_name not in configs:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        config = configs[dataset_name]
        return CNN(
            weight_decimals=weight_decimals,
            input_shape=config['input_shape'],
            num_classes=config['num_classes']
        )
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=1):
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=1
        )
        return {
            'loss': history.history['loss'][-1],
            'accuracy': history.history['accuracy'][-1],
            'val_loss': history.history['val_loss'][-1],
            'val_accuracy': history.history['val_accuracy'][-1]
        }
    
    def get_weights(self):
        weights = self.model.get_weights()
        clipped_weights = []
        for w in weights:
            w_clipped = np.clip(w, -1.0, 1.0)
            w_scaled = tf.math.round(w_clipped * 10**self.WEIGHT_DECIMALS)
            clipped_weights.append(w_scaled)
        return clipped_weights
    
    def set_weights(self, parameters):
        try:
            scaled_weights = [w / 10**self.WEIGHT_DECIMALS for w in parameters]
            self.model.set_weights(scaled_weights)
        except Exception as e:
            print(f"Error in set_weights: {e}")
            print("Parameter shapes:")
            for i, p in enumerate(parameters):
                print(f"  {i}: {p.shape}")
    
    def evaluate(self, X_test, y_test):
        try:
            y_pred = self.model.predict(X_test)
            
            if np.isnan(y_pred).any():
                print("WARNING: NaN values detected in predictions. Using fallback metrics calculation.")
                self._check_model_weights()
                loss = 999
                accuracy = 0.1
                precision = 0.1
                recall = 0.1
                f1 = 0.1
                return loss, accuracy, precision, recall, f1
            
            predicted = np.argmax(y_pred, axis=-1)
            accuracy = accuracy_score(y_test, predicted)
            loss = log_loss(y_test, y_pred)
            precision = precision_score(y_test, predicted, average='weighted')
            recall = recall_score(y_test, predicted, average='weighted')
            f1 = f1_score(y_test, predicted, average='weighted')
            return loss, accuracy, precision, recall, f1
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return 999, 0.1, 0.1, 0.1, 0.1
    
    def _check_model_weights(self):
        weights = self.model.get_weights()
        print("MODEL WEIGHTS SUMMARY (due to NaN detection):")
        for i, w in enumerate(weights):
            contains_nan = np.isnan(w).any()
            print(f"Layer {i}: shape={w.shape}, contains_nan={contains_nan}, min={np.min(w)}, max={np.max(w)}")

    def flatten_list(self, nested_list):
        flattened = []
        for item in nested_list:
            if isinstance(item, (list, np.ndarray)):
                flattened.extend(self.flatten_list(item.tolist() if isinstance(item, np.ndarray) else item))
            else:
                flattened.append(item)
        return flattened

    def unflatten_list(self, flat_list, shapes):
        unflattened_list = []
        index = 0
        for shape in shapes:
            size = np.prod(shape)
            unflattened_list.append(np.array(flat_list[index:index + size]).reshape(shape))
            index += size
        return unflattened_list










# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import regularizers
# from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Dropout, Dense, Flatten, MaxPool2D
# from tensorflow.keras.models import Model
# from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, accuracy_score
# class CNN:
#     def __init__(self, weight_decimals=8, input_shape=(32, 32, 3), num_classes=10):
#         self.model = None
#         self.WEIGHT_DECIMALS = self.set_weight_decimals(weight_decimals)
#         self.input_shape = input_shape
#         self.num_classes = num_classes
#         self.set_initial_params()
#         self.print_architecture()

#     def set_weight_decimals(self, weight_decimals):
#         if 2 <= weight_decimals <= 8:
#             return weight_decimals
#         return 8



#     def set_initial_params(self):
#         inputs = Input(shape=self.input_shape)
#         x = Conv2D(32, kernel_size=3, activation="relu", kernel_regularizer=regularizers.l2(0.01))(inputs)
#         x = BatchNormalization()(x)
#         x = MaxPool2D(pool_size=(2, 2))(x)
#         x = Conv2D(64, kernel_size=3, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
#         x = BatchNormalization()(x)
#         x = MaxPool2D(pool_size=(2, 2))(x)
#         x = Flatten()(x)
#         x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
#         x = Dropout(0.5)(x)
#         outputs = Dense(self.num_classes, activation="softmax")(x)

#         self.model = Model(inputs=inputs, outputs=outputs)
#         self.model.compile(
#             loss=keras.losses.categorical_crossentropy,
#             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#             metrics=["accuracy"],
#         )

#     def print_architecture(self):
#         self.model.summary()
#         print(f"Input shape: {self.input_shape}")
#         print(f"Number of classes: {self.num_classes}")

#     # The rest of the methods remain the same
#     def fit(self, X_train, y_train, X_val, y_val, epochs=1):
#         history = self.model.fit(
#             X_train, y_train,
#             epochs=epochs,
#             validation_data=(X_val, y_val),
#             verbose=1
#         )
        
#         # Return the metrics from the last epoch
#         return {
#             'loss': history.history['loss'][-1],
#             'accuracy': history.history['accuracy'][-1],
#             'val_loss': history.history['val_loss'][-1],
#             'val_accuracy': history.history['val_accuracy'][-1]
#         }

#     def get_weights(self):
#         weights = self.model.get_weights()
#         scaled_weights = [tf.math.round(w * 10**self.WEIGHT_DECIMALS) for w in weights]
#         return scaled_weights

#     def set_weights(self, parameters):
#         scaled_weights = [w / 10**self.WEIGHT_DECIMALS for w in parameters]
#         self.model.set_weights(scaled_weights)


#     def unflatten_list(self, flat_list, shapes):
#         unflattened_list = []
#         index = 0
#         for shape in shapes:
#             size = np.prod(shape)
#             unflattened_list.append(np.array(flat_list[index:index + size]).reshape(shape))
#             index += size
#         return unflattened_list