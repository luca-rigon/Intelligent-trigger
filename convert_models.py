import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model


#generate random data in the shape of the input tensor as a representative dataset, for network quantization
def representative_dataset():
    for _ in range(200):
      data = np.random.uniform(-0.1, 0.1, (1, 256))  #FCN = (1, 256); CNN = (1, 256, 1)
      yield [data.astype(np.float32)]

      
      
      
#Convert model into TensorFlow - Lite model, supporting float32-operations
def convert_float(model):
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  return tflite_model
      
      
      
#convert to quantized model - See https://www.tensorflow.org/lite/performance/post_training_quantization
def convert_int(model):
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_dataset
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.int8  # or tf.uint8
  converter.inference_output_type = tf.int8  # or tf.uint8
  converter.experimental_new_converter = False
  
  tflite_model = = converter.convert()
  return tflite_model
  
  

  
  
  
quant = False #wether we want to quantize the model or not

model_path = "/home/luca/Documents/Project/models"
names = ['model_conv1D_1l_5_5', 'model_conv1D_1l_10_5', 'model_conv1D_1l_15_5', 'model_conv1D_2l_10_5', 'model_conv1D_2l_15_5', 'model_conv1D_2l_5_10', 'model_conv1D_1l_15_10', 'model_conv1D_2l_10_10', 'model_conv1D_2l_20_10', 'model_conv1D_2l_10_20']


for n in range(len(names)):
  model_name = names[n]
  model = load_model(f"{model_path}/{model_name}.h5")
  print("Converting model)
  

  # Convert the model
  if quant:
    print('Quantizing model...', model_name)
    tflite_quant_model = convert_int(model)
    # Save the model.
    with open(f'{model_name}_quantized.tflite', 'wb') as f:
     f.write(tflite_quant_model)

  else:
    print('Convert float32-model...', model_name)
    tflite_float_model = convert_float(model)
    # Save the model.
    with open(f'{model_name}.tflite', 'wb') as f:
     f.write(tflite_float_model)


print("Done!")
