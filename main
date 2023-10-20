import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalMaxPooling1D, MaxPooling1D, Conv1D, Dropout, BatchNormalization
from keras.optimizers import Adam

# Load the CSV data
df = pd.read_csv('dataset_file1.csv')

df = df.sample(frac=1)

# Split the data into features and target
X = df.drop('target', axis=1).values
y = df['target'].values

# Reshape input to be 2D [samples, features]
X = X.reshape((X.shape[0], -1))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

num_features = X_train.shape[1]
print(num_features)

# Convolutional Neural Network for regression
model = Sequential()
model.add(Conv1D(128, 3, activation='relu', input_shape=(num_features, 1)))
model.add(BatchNormalization())
model.add(MaxPooling1D(3))
model.add(Conv1D(256, 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(256, 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(256, 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv1D(256, 3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(3))
model.add(GlobalMaxPooling1D())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2)) 
model.add(Dense(1))  


# Stampa il modello
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
model.summary()

import tensorflow.keras as tfk
from tensorflow.keras.callbacks import ModelCheckpoint

def create_callbacks() :
    callbacks = []

    # Early Stopping -----------------------------------------------------
    es_callback = tfk.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=30)
    callbacks.append(es_callback)
    
    # Checkpointer
    checkpointer = ModelCheckpoint(filepath='./checkpoint/modelCNN2.h5', verbose=1, 
                                    save_best_only=True, monitor = "val_loss", mode = "auto",)
    callbacks.append(checkpointer)
    
    # Learning Rate Scheduler --------------------------------------------
    # LRS_callback = tfk.callbacks.ReduceLROnPlateau(monitor='val_accuracy', mode='max', patience=5, factor=0.5, min_lr=1e-5)
    # callbacks.append(LRS_callback)
    
    return callbacks

adam = Adam(learning_rate=0.0001)

# Compile the model with accuracy metric
model.compile(loss='mae', optimizer=adam)

# Fit the model
history = model.fit(X_train, y_train, epochs=300, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False, callbacks = create_callbacks())

import matplotlib.pyplot as plt
# Plot the loss
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
# Scatter
plt.scatter(y_test, loss)
plt.ylabel('loss')
plt.xlabel('y_loss')
plt.legend()
plt.show()
