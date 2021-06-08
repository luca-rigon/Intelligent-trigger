# Compatibility layer between Python 2 and Python 3
#original file that trains the networks
import os
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, Activation, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D

noise = np.load("data/data_noise_3.0SNR_1ch_0000.npy")
signal = np.load("data/data_signal_B_1ch_0000.npy")
n_classes = 2

x = np.vstack((noise, signal))  # shape is (200000, 1, 256)
# x = np.reshape(x, (x.shape[0], -1))  # shape is (200000, 1024)


x2 = np.swapaxes(x, 1, 2)  # (200000, 256, 1)
n_samples = x2.shape[1]
n_channels = x2.shape[2]

x_1D = np.reshape(x, (x.shape[0], -1))  # shape is (200000, 256)
x3 = np.expand_dims(x2, axis=-1)

# define labels with one hot encoding, (1,0) -> noise, (0,1) -> signal
y = np.zeros((200000, 2))
y[:100000, 0] = 1
y[100000:, 1] = 1

y2 = np.zeros(200000)
y2[:100000] = 1

BATCH_SIZE = 32
EPOCHS = 50
callbacks_list = [
#     keras.callbacks.ModelCheckpoint(
#         filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
#         monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=1)
]

# define a neural network with fully connected layers:

# fully connected layouts
model_fc_1layer64 = Sequential()
model_fc_1layer64.add(Dense(64, input_dim=x_1D.shape[-1]))
model_fc_1layer64.add(Activation('relu'))
model_fc_1layer64.add(Dense(2))
model_fc_1layer64.add(Activation('softmax'))

# define a neutral network with convolutional layers:

# conv1D network without global avg pooling
model_cnn_1layer10_10 = Sequential()
model_cnn_1layer10_10.add(Conv2D(10, (10, 1), activation='relu', input_shape=(n_samples, n_channels, 1)))  # n_filters, n_width
model_cnn_1layer10_10.add(Dropout(0.5))
model_cnn_1layer10_10.add(MaxPooling2D(pool_size=(10, 1)))
model_cnn_1layer10_10.add(Reshape((np.prod(model_cnn_1layer10_10.layers[-1].output_shape[1:]),)))  # equivalent to Flatten
model_cnn_1layer10_10.add(Dense(n_classes, activation='softmax'))

# # train the fully connected network
model_fc_1layer64.compile(optimizer='Adam',
          loss='binary_crossentropy',
          metrics=['accuracy'])

print(model_fc_1layer64.summary())
model_fc_1layer64.fit(x_1D, y,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          callbacks=callbacks_list,
          validation_split=0.2,
          verbose=1)
model_fc_1layer64.save(f'models/model_fc_1layer64_test.h5')

# train the convolutional network
model_cnn_1layer10_10.compile(optimizer='Adam',
          loss='binary_crossentropy',
          metrics=['accuracy'])

print(model_cnn_1layer10_10.summary())
model_cnn_1layer10_10.fit(x3,
          y,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          callbacks=callbacks_list,
          validation_split=0.2,
          verbose=1)
model_cnn_1layer10_10.save(f'models/model_cnn_1layer10_10_test.h5')
