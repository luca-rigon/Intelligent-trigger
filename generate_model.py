import os
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D


FCN = True

#load data
#Training data: 75k signal files, 75k noise files
x = np.load("training_data.npy").astype(np.float32) #shape: (150000, 256, 1)
n_samples = x.shape[1]
n_channels = x.shape[2]
n_classes = 2



# define labels with one hot encoding, (1,0) -> noise, (0,1) -> signal
y_train = np.zeros((x.shape[0], 2))
y_train[:75000, 1] = 1
y_train[75000:, 0] = 1

BATCH_SIZE = 32
EPOCHS = 20
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=1)]

if FCN == True:
#Train Fully connected network
    x_train = np.reshape(x, (x.shape[0], -1))  #for fc network: shape is (150000, 256)
    model = keras.Sequential(
    [
        Dense(64, activation="relu", input_dim=x_train.shape[-1]),
        Dense(64, activation="relu"),
        Dense(n_classes, activation="softmax"),
    ]
)
else:
    #1D - Convolutional Network
    x_train = np.expand_dims(x, axis=-1) #for cnn network (150000, 256, 1)
    model = Sequential(
    [
        Conv1D(50, 50, activation='relu', input_shape=(n_samples, n_channels)), # n_filters, n_width
        Conv1D(75, 50, activation='relu'), # n_filters, n_width
        Dropout(0.5),
        #MaxPooling1D(pool_size=(2)),
        Flatten(),
        #Dense(200, activation="relu"),
        #Dense(50, activation="relu"),
        Dense(n_classes, activation='softmax')
    ]
)


# train the network
model.compile(optimizer='Adam',
          loss='binary_crossentropy',
          metrics=['accuracy'])


print(model.summary())


model.fit(x_train,y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          callbacks=callbacks_list,
          validation_split=0.2,
          verbose=1)


model.save('model_fc_2l_64'.h5)