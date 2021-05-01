import tensorflow as tf
from tensorflow import keras

model_name = 'model_fc_4l_64'
model = keras.models.load_model(f'models/'+model_name+'.h5', compile=False)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open(model_name+'.tflite', 'wb') as f:
  f.write(tflite_model)