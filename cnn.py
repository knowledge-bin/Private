# # Standard Libraries
# import numpy as np

# # Third Party Imports
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import regularizers
# from tensorflow.keras.layers import BatchNormalization, Conv2D, Dropout, Dense, Flatten, MaxPool2D
# from tensorflow.keras.models import Sequential
# from sklearn.metrics import log_loss
# # import torch



# class CNN():
#     def __init__(self, weight_decimals=8):
#         self.model = None
#         self.WEIGHT_DECIMALS = self.set_weight_decimals(weight_decimals)

#     def set_weight_decimals(self, weight_decimals):
#         if 2 <= weight_decimals <= 8:
#             return weight_decimals
#         return 8

#     def set_initial_params(self):
#         """
#         This function creates and compiles a cnn model using the tensorflow keras api 
#         First three convolutional layers, each followed by batch normalization and max pooling. 
#             - each layer uses 5x5 filters and a ReLU activation function, with L2 regularization. 
#             - first, second, and third layers contain 32, 16, and 32 filters respectively.
#         After convolutional layers we have a Flatten layer to convert 3D output to 1D
#         After is a fully connected Dense layer with 200 nodes
#             - uses a ReLU activation function and L2 regularization. 
#             - has a dropout rate of 0.5 to prevent overfitting.
#         Final layer also a Dense layer with 2 nodes to representing the two output classes.
#             - uses a softmax activation function for multiclass classification
#         Model is compiled with Adam Optimizer
#         """
#         model = Sequential()
#         # convulutional layer
#         model.add(
#             Conv2D(
#                 32,
#                 kernel_size=5,
#                 activation="relu",
#                 input_shape=(128, 128, 1),
#                 kernel_regularizer=regularizers.l2(0.01)
#             )
#         )
#         # Normalising after activation
#         model.add(BatchNormalization())
#         model.add(MaxPool2D(pool_size=(2, 2)))
#         model.add(Conv2D(16, kernel_size=5, activation="relu",
#                   kernel_regularizer=regularizers.l2(0.01)))
#         model.add(BatchNormalization())
#         model.add(MaxPool2D(pool_size=(2, 2)))
#         model.add(Conv2D(32, kernel_size=5, activation="relu",
#                   kernel_regularizer=regularizers.l2(0.01)))
#         model.add(BatchNormalization())
#         model.add(MaxPool2D(pool_size=(2, 2)))
#         model.add(Flatten())
#         # fully connected layer
#         model.add(Dense(200, activation="relu",
#                   kernel_regularizer=regularizers.l2(0.01)))
#         model.add(Dropout(0.5))

#         # output
#         model.add(Dense(2, activation="softmax"))
#         model.compile(
#             loss=keras.losses.sparse_categorical_crossentropy,
#             optimizer=tf.keras.optimizers.legacy.Adam(
#                 learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0),
#             metrics=["accuracy"],
#         )
#         self.model = model

#     # def fit(self, X_train, y_train, X_val, y_val, epochs=20, workers=4):
#     #     """
#     #     Takes in epochs and workers and starts training the model 
#     #     ----------
#     #     epochs : int
#     #         The number of epochs in training
#     #     workers : int
#     #         The number of workers in the training
#     #     """
#     #     tf.keras.preprocessing.image.ImageDataGenerator(
#     #         featurewise_center=True,
#     #         featurewise_std_normalization=True,
#     #         rotation_range=20,
#     #         width_shift_range=0.2,
#     #         height_shift_range=0.2,
#     #         horizontal_flip=True,
#     #         validation_split=0.2)

#     #     tf.keras.callbacks.EarlyStopping(
#     #         monitor='val_loss',
#     #         min_delta=0.05,
#     #         patience=2,
#     #         verbose=1,
#     #         restore_best_weights=True)

#     #     self.model.fit(
#     #         X_train,
#     #         y_train,
#     #         epochs=epochs,
#     #         workers=workers,
#     #         validation_data=(X_val, y_val),
#     #     )

#     def fit(self, X_train, y_train, X_val, y_val, epochs=10, workers=3, batch_size=32):
#             # Data augmentation
#             datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#                 featurewise_center=True,
#                 featurewise_std_normalization=True,
#                 rotation_range=20,
#                 width_shift_range=0.2,
#                 height_shift_range=0.2,
#                 horizontal_flip=True
#             )
#             datagen.fit(X_train)

#             # Early stopping
#             early_stopping = tf.keras.callbacks.EarlyStopping(
#                 monitor='val_loss',
#                 min_delta=0.05,
#                 patience=3,  # Increased patience for better training
#                 verbose=1,
#                 restore_best_weights=True
#             )

#             # Learning rate scheduler
#             lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
#                 monitor='val_loss',
#                 factor=0.5,
#                 patience=2,
#                 min_lr=1e-6,
#                 verbose=1
#             )

#             # Class weights (adjust according to your class distribution)
#             class_weight = {0: 1.0, 1: 2.0}  # Example weights

#             # Compile model with adjusted learning rate
#             self.model.compile(
#                 loss=keras.losses.sparse_categorical_crossentropy,
#                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Reduced learning rate
#                 metrics=["accuracy"]
#             )

#             # Fit model
#             self.model.fit(
#                 datagen.flow(X_train, y_train, batch_size=batch_size),
#                 epochs=epochs,
#                 workers=workers,
#                 validation_data=(X_val, y_val),
#                 callbacks=[early_stopping, lr_scheduler],
#                 class_weight=class_weight
#             )



#     def get_weights(self):
#         """
#         returns the weights of the model
#         """
#         # return self.model.get_weights()
#         weights = self.model.get_weights()
#         scaled_weights = [tf.math.round(w * 10**self.WEIGHT_DECIMALS) for w in weights]
#         return scaled_weights

#     def set_weights(self, parameters):
#         """
#         Sets the weight of the model
#         """
#         # self.model.set_weights(parameters)
#         scaled_weights = [w / 10**self.WEIGHT_DECIMALS for w in parameters]
#         self.model.set_weights(scaled_weights)

#     def evaluate(self, X_test, y_test):
#         """
#         Evaluates the accuracy and gives a classification report of the result from the CNN model
#         """
#         y_pred = self.model.predict(X_test)
#         predicted = np.argmax(y_pred, axis=-1)
#         accuracy = np.equal(y_test, predicted).mean()
#         loss = log_loss(y_test, y_pred)

#         return loss, accuracy
    

#     def flatten_list(self, nested_list):
#         """
#         Takes nested list of tensors from cnn models and flattens it into a single list of elements.
#         """
#         flattened = []
#         for item in nested_list:
#             #if torch.is_tensor(item):
#             #    flattened.extend(self.flatten_list(item.tolist() if isinstance(item, np.ndarray) else item))
#             if isinstance(item, (list, np.ndarray)):
#                 flattened.extend(self.flatten_list(item.tolist() if isinstance(item, np.ndarray) else item))
#             else:
#                 flattened.append(item)
#         return flattened

#     def unflatten_list(self, flat_list, shapes):
#         """
#         Reshapes flattened list back to a nested list of tensors for a cnn model.
#         """
#         unflattened_list = []
#         index = 0
#         for shape in shapes:
#             size = np.prod(shape)
#             unflattened_list.append(np.array(flat_list[index:index + size]).reshape(shape))
#             index += size
#         return unflattened_list










# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import regularizers
# from tensorflow.keras.layers import BatchNormalization, Conv2D, Dropout, Dense, Flatten, MaxPool2D
# from tensorflow.keras.models import Sequential
# from sklearn.metrics import log_loss

# class CNN:
#     def __init__(self, weight_decimals=8):
#         self.model = None
#         self.WEIGHT_DECIMALS = self.set_weight_decimals(weight_decimals)

#     def set_weight_decimals(self, weight_decimals):
#         if 2 <= weight_decimals <= 8:
#             return weight_decimals
#         return 8

#     def set_initial_params(self):
#         self.model = Sequential()
#         self.model.add(Conv2D(32, kernel_size=3, activation="relu", input_shape=(28, 28, 1), kernel_regularizer=regularizers.l2(0.01)))
#         self.model.add(BatchNormalization())
#         self.model.add(MaxPool2D(pool_size=(2, 2)))
#         self.model.add(Conv2D(64, kernel_size=3, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
#         self.model.add(BatchNormalization())
#         self.model.add(MaxPool2D(pool_size=(2, 2)))
#         self.model.add(Flatten())
#         self.model.add(Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
#         self.model.add(Dropout(0.5))
#         self.model.add(Dense(10, activation="softmax"))
#         self.model.compile(
#             loss=keras.losses.sparse_categorical_crossentropy,
#             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#             metrics=["accuracy"],
#         )

#     def fit(self, X_train, y_train, X_val, y_val, epochs=20):
#         self.model.fit(
#             X_train, y_train,
#             epochs=epochs,
#             validation_data=(X_val, y_val),
#         )

#     def get_weights(self):
#         weights = self.model.get_weights()
#         scaled_weights = [tf.math.round(w * 10**self.WEIGHT_DECIMALS) for w in weights]
#         return scaled_weights

#     def set_weights(self, parameters):
#         scaled_weights = [w / 10**self.WEIGHT_DECIMALS for w in parameters]
#         self.model.set_weights(scaled_weights)

#     def evaluate(self, X_test, y_test):
#         y_pred = self.model.predict(X_test)
#         predicted = np.argmax(y_pred, axis=-1)
#         accuracy = np.equal(y_test, predicted).mean()
#         loss = log_loss(y_test, y_pred)
#         precision = precision_score(y_test, predicted, average='weighted')
#         recall = recall_score(y_test, predicted, average='weighted')
#         f1 = f1_score(y_test, predicted, average='weighted')
#         return loss, accuracy, precision, recall, f1

#     def flatten_list(self, nested_list):
#         flattened = []
#         for item in nested_list:
#             if isinstance(item, (list, np.ndarray)):
#                 flattened.extend(self.flatten_list(item.tolist() if isinstance(item, np.ndarray) else item))
#             else:
#                 flattened.append(item)
#         return flattened

#     def unflatten_list(self, flat_list, shapes):
#         unflattened_list = []
#         index = 0
#         for shape in shapes:
#             size = np.prod(shape)
#             unflattened_list.append(np.array(flat_list[index:index + size]).reshape(shape))
#             index += size
#         return unflattened_list


# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import regularizers
# from tensorflow.keras.layers import BatchNormalization, Conv2D, Dropout, Dense, Flatten, MaxPool2D
# from tensorflow.keras.models import Sequential
# from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, accuracy_score

# class CNN:
#     def __init__(self, weight_decimals=8):
#         self.model = None
#         self.WEIGHT_DECIMALS = self.set_weight_decimals(weight_decimals)

#     def set_weight_decimals(self, weight_decimals):
#         if 2 <= weight_decimals <= 8:
#             return weight_decimals
#         return 8

#     def set_initial_params(self):
#         self.model = Sequential()
#         self.model.add(Conv2D(32, kernel_size=3, activation="relu", input_shape=(28, 28, 1), kernel_regularizer=regularizers.l2(0.01)))
#         self.model.add(BatchNormalization())
#         self.model.add(MaxPool2D(pool_size=(2, 2)))
#         self.model.add(Conv2D(64, kernel_size=3, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
#         self.model.add(BatchNormalization())
#         self.model.add(MaxPool2D(pool_size=(2, 2)))
#         self.model.add(Flatten())
#         self.model.add(Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
#         self.model.add(Dropout(0.5))
#         self.model.add(Dense(10, activation="softmax"))
#         self.model.compile(
#             loss=keras.losses.sparse_categorical_crossentropy,
#             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#             metrics=["accuracy"],
#         )

#     def fit(self, X_train, y_train, X_val, y_val, epochs=20):
#         self.model.fit(
#             X_train, y_train,
#             epochs=epochs,
#             validation_data=(X_val, y_val),
#         )

#     def get_weights(self):
#         weights = self.model.get_weights()
#         scaled_weights = [tf.math.round(w * 10**self.WEIGHT_DECIMALS) for w in weights]
#         return scaled_weights

#     def set_weights(self, parameters):
#         scaled_weights = [w / 10**self.WEIGHT_DECIMALS for w in parameters]
#         self.model.set_weights(scaled_weights)

#     def evaluate(self, X_test, y_test):
#         y_pred = self.model.predict(X_test)
#         predicted = np.argmax(y_pred, axis=-1)
#         accuracy = accuracy_score(y_test, predicted)
#         loss = log_loss(y_test, y_pred)
#         precision = precision_score(y_test, predicted, average='weighted')
#         recall = recall_score(y_test, predicted, average='weighted')
#         f1 = f1_score(y_test, predicted, average='weighted')
#         return loss, accuracy, precision, recall, f1

#     def flatten_list(self, nested_list):
#         flattened = []
#         for item in nested_list:
#             if isinstance(item, (list, np.ndarray)):
#                 flattened.extend(self.flatten_list(item.tolist() if isinstance(item, np.ndarray) else item))
#             else:
#                 flattened.append(item)
#         return flattened

#     def unflatten_list(self, flat_list, shapes):
#         unflattened_list = []
#         index = 0
#         for shape in shapes:
#             size = np.prod(shape)
#             unflattened_list.append(np.array(flat_list[index:index + size]).reshape(shape))
#             index += size
#         return unflattened_list




# #WORKING FOR MNIST 33333333333333333333333333333333333333333333333333333333333333333

# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import regularizers
# from tensorflow.keras.layers import Input, BatchNormalization, Conv2D, Dropout, Dense, Flatten, MaxPool2D
# from tensorflow.keras.models import Model
# from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, accuracy_score

# class CNN:
#     def __init__(self, weight_decimals=8):
#         self.model = None
#         self.WEIGHT_DECIMALS = self.set_weight_decimals(weight_decimals)

#     def set_weight_decimals(self, weight_decimals):
#         if 2 <= weight_decimals <= 8:
#             return weight_decimals
#         return 8

#     def set_initial_params(self):
#         inputs = Input(shape=(28, 28, 1))
#         x = Conv2D(32, kernel_size=3, activation="relu", kernel_regularizer=regularizers.l2(0.01))(inputs)
#         x = BatchNormalization()(x)
#         x = MaxPool2D(pool_size=(2, 2))(x)
#         x = Conv2D(64, kernel_size=3, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
#         x = BatchNormalization()(x)
#         x = MaxPool2D(pool_size=(2, 2))(x)
#         x = Flatten()(x)
#         x = Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
#         x = Dropout(0.5)(x)
#         outputs = Dense(10, activation="softmax")(x)

#         self.model = Model(inputs=inputs, outputs=outputs)
#         self.model.compile(
#             loss=keras.losses.sparse_categorical_crossentropy,
#             optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#             metrics=["accuracy"],
#         )

#     # The rest of the methods remain the same
#     # def fit(self, X_train, y_train, X_val, y_val, epochs=20):
#     #     self.model.fit(
#     #         X_train, y_train,
#     #         epochs=epochs,
#     #         validation_data=(X_val, y_val),
#     #     )

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

#     def evaluate(self, X_test, y_test):
#         y_pred = self.model.predict(X_test)
#         predicted = np.argmax(y_pred, axis=-1)
#         accuracy = accuracy_score(y_test, predicted)
#         loss = log_loss(y_test, y_pred)
#         precision = precision_score(y_test, predicted, average='weighted')
#         recall = recall_score(y_test, predicted, average='weighted')
#         f1 = f1_score(y_test, predicted, average='weighted')
#         return loss, accuracy, precision, recall, f1

#     def flatten_list(self, nested_list):
#         flattened = []
#         for item in nested_list:
#             if isinstance(item, (list, np.ndarray)):
#                 flattened.extend(self.flatten_list(item.tolist() if isinstance(item, np.ndarray) else item))
#             else:
#                 flattened.append(item)
#         return flattened

#     def unflatten_list(self, flat_list, shapes):
#         unflattened_list = []
#         index = 0
#         for shape in shapes:
#             size = np.prod(shape)
#             unflattened_list.append(np.array(flat_list[index:index + size]).reshape(shape))
#             index += size
#         return unflattened_list
    
#     #33333333333333333333333333333333333333333333333333333333333333333

#NEW RESNET with mnist it works but predicton is NaN
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, regularizers
# from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, accuracy_score

# class CNN:
#     def __init__(self, weight_decimals=8):
#         self.model = None
#         self.WEIGHT_DECIMALS = self.set_weight_decimals(weight_decimals)
    
#     def set_weight_decimals(self, weight_decimals):
#         if 2 <= weight_decimals <= 8:
#             return weight_decimals
#         return 8
        
#     def resnet_block(self, x, filters, kernel_size=3, stride=1, conv_shortcut=False):
#         """A residual block similar to ResNet but lightweight"""
#         shortcut = x
        
#         if conv_shortcut:
#             shortcut = layers.Conv2D(filters, 1, strides=stride)(shortcut)
#             shortcut = layers.BatchNormalization()(shortcut)
        
#         # First convolution
#         x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
        
#         # Second convolution
#         x = layers.Conv2D(filters, kernel_size, padding='same')(x)
#         x = layers.BatchNormalization()(x)
        
#         # Add shortcut to output
#         x = layers.add([x, shortcut])
#         x = layers.Activation('relu')(x)
        
#         return x
        
#     def set_initial_params(self):
#         """Create a lightweight ResNet-inspired model for MNIST"""
#         inputs = keras.Input(shape=(28, 28, 1))
        
#         # Initial convolution
#         x = layers.Conv2D(16, 3, padding='same')(inputs)
#         x = layers.BatchNormalization()(x)
#         x = layers.Activation('relu')(x)
        
#         # First stack of residual blocks (16 filters)
#         x = self.resnet_block(x, 16, conv_shortcut=True)
#         x = self.resnet_block(x, 16)
        
#         # Second stack of residual blocks (32 filters)
#         x = self.resnet_block(x, 32, stride=2, conv_shortcut=True)
#         x = self.resnet_block(x, 32)
        
#         # Global average pooling and dense layer
#         x = layers.GlobalAveragePooling2D()(x)
#         x = layers.Dense(64, activation='relu')(x)
#         x = layers.Dropout(0.5)(x)
#         outputs = layers.Dense(10, activation='softmax')(x)
        
#         self.model = keras.Model(inputs=inputs, outputs=outputs)
#         self.model.compile(
#             loss=keras.losses.sparse_categorical_crossentropy,
#             optimizer=keras.optimizers.Adam(learning_rate=0.001),
#             metrics=['accuracy']
#         )
    
#     def fit(self, X_train, y_train, X_val, y_val, epochs=1):
#         history = self.model.fit(
#             X_train, y_train,
#             epochs=epochs,
#             batch_size=32,
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
    
#     def evaluate(self, X_test, y_test):
#         loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
#         # Additional metrics for detailed evaluation
#         y_pred = self.model.predict(X_test)
#         predicted = np.argmax(y_pred, axis=-1)
#         precision = precision_score(y_test, predicted, average='weighted')
#         recall = recall_score(y_test, predicted, average='weighted')
#         f1 = f1_score(y_test, predicted, average='weighted')
        
#         return loss, accuracy, precision, recall, f1

###############################################################################
#VVVVVVGGGGGGGGGGGGGGGG
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, regularizers
# from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, accuracy_score

# class CNN:
#     def __init__(self, weight_decimals=8):
#         self.model = None
#         self.WEIGHT_DECIMALS = self.set_weight_decimals(weight_decimals)
    
#     def set_weight_decimals(self, weight_decimals):
#         if 2 <= weight_decimals <= 8:
#             return weight_decimals
#         return 8
    
#     def set_initial_params(self):
#         """Create a LeNet-5 style model without BatchNormalization for MNIST"""
#         inputs = keras.Input(shape=(28, 28, 1))
        
#         # First convolutional layer
#         x = layers.Conv2D(16, (5, 5), padding='same', activation='relu',
#                          kernel_regularizer=regularizers.l2(0.0005))(inputs)
#         x = layers.AveragePooling2D(pool_size=(2, 2))(x)
        
#         # Second convolutional layer
#         x = layers.Conv2D(32, (5, 5), padding='same', activation='relu',
#                          kernel_regularizer=regularizers.l2(0.0005))(x)
#         x = layers.AveragePooling2D(pool_size=(2, 2))(x)
        
#         # Fully connected layers
#         x = layers.Flatten()(x)
#         x = layers.Dense(120, activation='relu', 
#                          kernel_regularizer=regularizers.l2(0.0005))(x)
#         x = layers.Dropout(0.3)(x)
#         x = layers.Dense(84, activation='relu',
#                          kernel_regularizer=regularizers.l2(0.0005))(x)
#         x = layers.Dropout(0.3)(x)
#         outputs = layers.Dense(10, activation='softmax')(x)
        
#         self.model = keras.Model(inputs=inputs, outputs=outputs)
#         self.model.compile(
#             loss=keras.losses.sparse_categorical_crossentropy,
#             optimizer=keras.optimizers.Adam(learning_rate=0.0005),
#             metrics=['accuracy']
#         )
    
#     def fit(self, X_train, y_train, X_val, y_val, epochs=1):
#         history = self.model.fit(
#             X_train, y_train,
#             epochs=epochs,
#             batch_size=32,
#             validation_data=(X_val, y_val),
#             verbose=1
#         )
        
#         return {
#             'loss': history.history['loss'][-1],
#             'accuracy': history.history['accuracy'][-1],
#             'val_loss': history.history['val_loss'][-1],
#             'val_accuracy': history.history['val_accuracy'][-1]
#         }
    
#     def get_weights(self):
#         weights = self.model.get_weights()
#         # Add value clipping to prevent extreme values
#         clipped_weights = []
#         for w in weights:
#             # Clip to a reasonable range before scaling
#             w_clipped = np.clip(w, -1.0, 1.0)
#             w_scaled = tf.math.round(w_clipped * 10**self.WEIGHT_DECIMALS)
#             clipped_weights.append(w_scaled)
#         return clipped_weights
    
#     def set_weights(self, parameters):
#         try:
#             # Add safety measures when converting back from integer to float
#             scaled_weights = []
#             for w in parameters:
#                 w_float = w / 10**self.WEIGHT_DECIMALS
#                 # Additional safety check
#                 if np.isnan(w_float).any() or np.isinf(w_float).any():
#                     raise ValueError("NaN or Inf values detected in parameters")
#                 scaled_weights.append(w_float)
#             self.model.set_weights(scaled_weights)
#         except Exception as e:
#             print(f"Error in set_weights: {e}")
#             # Print the shapes of parameters for debugging
#             print("Parameter shapes:")
#             for i, p in enumerate(parameters):
#                 print(f"  {i}: {p.shape}")
    
#     def evaluate(self, X_test, y_test):
#         try:
#             # Let's be extra careful about NaN values in predictions
#             y_pred = self.model.predict(X_test)
            
#             # Check for NaN values
#             if np.isnan(y_pred).any():
#                 print("WARNING: NaN values detected in predictions. Using fallback metrics calculation.")
#                 # Detailed weights analysis for debugging
#                 self._check_model_weights()
                
#                 # Use a fallback prediction
#                 loss = 999
#                 accuracy = 0.1  # approximate class distribution for MNIST
#                 precision = 0.1
#                 recall = 0.1
#                 f1 = 0.1
#                 return loss, accuracy, precision, recall, f1
            
#             predicted = np.argmax(y_pred, axis=-1)
#             accuracy = accuracy_score(y_test, predicted)
#             loss = log_loss(y_test, y_pred)
#             precision = precision_score(y_test, predicted, average='weighted')
#             recall = recall_score(y_test, predicted, average='weighted')
#             f1 = f1_score(y_test, predicted, average='weighted')
#             return loss, accuracy, precision, recall, f1
            
#         except Exception as e:
#             print(f"Error during evaluation: {e}")
#             return 999, 0.1, 0.1, 0.1, 0.1
    
#     def _check_model_weights(self):
#         """Analyze model weights for debugging purposes"""
#         weights = self.model.get_weights()
#         print("MODEL WEIGHTS SUMMARY (due to NaN detection):")
#         for i, w in enumerate(weights):
#             contains_nan = np.isnan(w).any()
#             print(f"Layer {i}: shape={w.shape}, contains_nan={contains_nan}, min={np.min(w)}, max={np.max(w)}")

#     def flatten_list(self, nested_list):
#         flattened = []
#         for item in nested_list:
#             if isinstance(item, (list, np.ndarray)):
#                 flattened.extend(self.flatten_list(item.tolist() if isinstance(item, np.ndarray) else item))
#             else:
#                 flattened.append(item)
#         return flattened

#     def unflatten_list(self, flat_list, shapes):
#         unflattened_list = []
#         index = 0
#         for shape in shapes:
#             size = np.prod(shape)
#             unflattened_list.append(np.array(flat_list[index:index + size]).reshape(shape))
#             index += size
#         return unflattened_list
###############################################################################
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, regularizers
# from sklearn.metrics import log_loss, precision_score, recall_score, f1_score, accuracy_score

# class CNN:
#     def __init__(self, weight_decimals=8, input_shape=None, num_classes=None):
#         self.model = None
#         self.WEIGHT_DECIMALS = self.set_weight_decimals(weight_decimals)
        
#         # Set default values if not provided
#         if input_shape is None:
#             input_shape = (28, 28, 1)  # Default to MNIST
#         if num_classes is None:
#             num_classes = 10  # Default to 10 classes
            
#         self.input_shape = input_shape
#         self.num_classes = num_classes
    
#     def set_weight_decimals(self, weight_decimals):
#         if 2 <= weight_decimals <= 8:
#             return weight_decimals
#         return 8
    
#     def set_initial_params(self):
#         """Create a model that adapts to different image sizes and number of classes"""
#         inputs = keras.Input(shape=self.input_shape)
        
#         # First convolutional layer
#         x = layers.Conv2D(16, (5, 5), padding='same', activation='relu',
#                          kernel_regularizer=regularizers.l2(0.0005))(inputs)
#         x = layers.AveragePooling2D(pool_size=(2, 2))(x)
        
#         # Second convolutional layer
#         x = layers.Conv2D(32, (5, 5), padding='same', activation='relu',
#                          kernel_regularizer=regularizers.l2(0.0005))(x)
#         x = layers.AveragePooling2D(pool_size=(2, 2))(x)
        
#         # Adapt based on input size
#         if self.input_shape[0] > 28:  # For larger images like CIFAR
#             x = layers.Conv2D(64, (3, 3), padding='same', activation='relu',
#                              kernel_regularizer=regularizers.l2(0.0005))(x)
#             x = layers.AveragePooling2D(pool_size=(2, 2))(x)
        
#         # Fully connected layers
#         x = layers.Flatten()(x)
#         x = layers.Dense(120, activation='relu', 
#                          kernel_regularizer=regularizers.l2(0.0005))(x)
#         x = layers.Dropout(0.3)(x)
#         x = layers.Dense(84, activation='relu',
#                          kernel_regularizer=regularizers.l2(0.0005))(x)
#         x = layers.Dropout(0.3)(x)
        
#         # Output layer with adaptive number of classes
#         outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
#         self.model = keras.Model(inputs=inputs, outputs=outputs)
#         self.model.compile(
#             loss=keras.losses.sparse_categorical_crossentropy,
#             optimizer=keras.optimizers.Adam(learning_rate=0.0005),
#             metrics=['accuracy']
#         )
    
#     # Model factory method to help with easy creation
#     @staticmethod
#     def create_for_dataset(dataset_name, weight_decimals=8):
#         """Factory method to create CNN model for specific datasets"""
#         configs = {
#             'mnist': {
#                 'input_shape': (28, 28, 1),
#                 'num_classes': 10
#             },
#             'fashion_mnist': {
#                 'input_shape': (28, 28, 1),
#                 'num_classes': 10
#             },
#             'cifar10': {
#                 'input_shape': (32, 32, 3),
#                 'num_classes': 10
#             },
#             'cifar100': {
#                 'input_shape': (32, 32, 3),
#                 'num_classes': 100
#             }
#         }
        
#         if dataset_name not in configs:
#             raise ValueError(f"Unsupported dataset: {dataset_name}")
        
#         config = configs[dataset_name]
#         return CNN(
#             weight_decimals=weight_decimals,
#             input_shape=config['input_shape'],
#             num_classes=config['num_classes']
#         )
    
#     # Rest of the methods remain the same
#     def fit(self, X_train, y_train, X_val, y_val, epochs=1):
#         history = self.model.fit(
#             X_train, y_train,
#             epochs=epochs,
#             batch_size=32,
#             validation_data=(X_val, y_val),
#             verbose=1
#         )
        
#         return {
#             'loss': history.history['loss'][-1],
#             'accuracy': history.history['accuracy'][-1],
#             'val_loss': history.history['val_loss'][-1],
#             'val_accuracy': history.history['val_accuracy'][-1]
#         }
    
#     def get_weights(self):
#         weights = self.model.get_weights()
#         # Add value clipping to prevent extreme values
#         clipped_weights = []
#         for w in weights:
#             # Clip to a reasonable range before scaling
#             w_clipped = np.clip(w, -1.0, 1.0)
#             w_scaled = tf.math.round(w_clipped * 10**self.WEIGHT_DECIMALS)
#             clipped_weights.append(w_scaled)
#         return clipped_weights
    
#     def set_weights(self, parameters):
#         try:
#             # Add safety measures when converting back from integer to float
#             scaled_weights = []
#             for w in parameters:
#                 w_float = w / 10**self.WEIGHT_DECIMALS
#                 # Additional safety check
#                 if np.isnan(w_float).any() or np.isinf(w_float).any():
#                     raise ValueError("NaN or Inf values detected in parameters")
#                 scaled_weights.append(w_float)
#             self.model.set_weights(scaled_weights)
#         except Exception as e:
#             print(f"Error in set_weights: {e}")
#             # Print the shapes of parameters for debugging
#             print("Parameter shapes:")
#             for i, p in enumerate(parameters):
#                 print(f"  {i}: {p.shape}")
    
#     def evaluate(self, X_test, y_test):
#         try:
#             # Let's be extra careful about NaN values in predictions
#             y_pred = self.model.predict(X_test)
            
#             # Check for NaN values
#             if np.isnan(y_pred).any():
#                 print("WARNING: NaN values detected in predictions. Using fallback metrics calculation.")
#                 # Detailed weights analysis for debugging
#                 self._check_model_weights()
                
#                 # Use a fallback prediction
#                 loss = 999
#                 accuracy = 0.1  # approximate class distribution for MNIST
#                 precision = 0.1
#                 recall = 0.1
#                 f1 = 0.1
#                 return loss, accuracy, precision, recall, f1
            
#             predicted = np.argmax(y_pred, axis=-1)
#             accuracy = accuracy_score(y_test, predicted)
#             loss = log_loss(y_test, y_pred)
#             precision = precision_score(y_test, predicted, average='weighted')
#             recall = recall_score(y_test, predicted, average='weighted')
#             f1 = f1_score(y_test, predicted, average='weighted')
#             return loss, accuracy, precision, recall, f1
            
#         except Exception as e:
#             print(f"Error during evaluation: {e}")
#             return 999, 0.1, 0.1, 0.1, 0.1
    
#     def _check_model_weights(self):
#         """Analyze model weights for debugging purposes"""
#         weights = self.model.get_weights()
#         print("MODEL WEIGHTS SUMMARY (due to NaN detection):")
#         for i, w in enumerate(weights):
#             contains_nan = np.isnan(w).any()
#             print(f"Layer {i}: shape={w.shape}, contains_nan={contains_nan}, min={np.min(w)}, max={np.max(w)}")

#     def flatten_list(self, nested_list):
#         flattened = []
#         for item in nested_list:
#             if isinstance(item, (list, np.ndarray)):
#                 flattened.extend(self.flatten_list(item.tolist() if isinstance(item, np.ndarray) else item))
#             else:
#                 flattened.append(item)
#         return flattened

#     def unflatten_list(self, flat_list, shapes):
#         unflattened_list = []
#         index = 0
#         for shape in shapes:
#             size = np.prod(shape)
#             unflattened_list.append(np.array(flat_list[index:index + size]).reshape(shape))
#             index += size
#         return unflattened_list



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

#     def evaluate(self, X_test, y_test):
#         y_pred = self.model.predict(X_test)
#         predicted = np.argmax(y_pred, axis=-1)
#         true_labels = np.argmax(y_test, axis=-1)  # Assuming y_test is one-hot encoded
#         accuracy = accuracy_score(true_labels, predicted)
#         loss = log_loss(y_test, y_pred)
#         precision = precision_score(true_labels, predicted, average='weighted')
#         recall = recall_score(true_labels, predicted, average='weighted')
#         f1 = f1_score(true_labels, predicted, average='weighted')
#         return loss, accuracy, precision, recall, f1
#     def flatten_list(self, nested_list):
#         flattened = []
#         for item in nested_list:
#             if isinstance(item, (list, np.ndarray)):
#                 flattened.extend(self.flatten_list(item.tolist() if isinstance(item, np.ndarray) else item))
#             else:
#                 flattened.append(item)
#         return flattened

#     def unflatten_list(self, flat_list, shapes):
#         unflattened_list = []
#         index = 0
#         for shape in shapes:
#             size = np.prod(shape)
#             unflattened_list.append(np.array(flat_list[index:index + size]).reshape(shape))
#             index += size
#         return unflattened_list