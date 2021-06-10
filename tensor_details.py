import os
import numpy as np
from termcolor import cprint
import tflite_runtime.interpreter as tflite
# -------

#Script which returns a list of quantization parameters (scale factors, zeropoint) for all the tensors of a network


#make a list of models we want to print the details of
model_path = '/home/mendel/inference_tests/quantized/'
models = [ 'model_conv1D_1l_5_1']#'model_fc_1l_16', 'model_fc_1l_32', 'model_fc_4l_32', 'model_fc_1l_64', 'model_fc_4l_64', 'model_fc_1l_128', 'model_fc_4l_128', 'model_fc_1l_256', 'model_fc_4l_200', 'model_fc_4l_256', 'model_fc_4l_350']
n_models = len(models)

scale_factors = []
zeropoints = []
layer_names = []


for m in range(n_models):
    times = []

    model_name = models[m]
    cprint("Show details for: "+model_name, "yellow")

    
    # Load model, initialize interpreter 
    path_to_file = model_path + model_name + '_quantized.tflite'
    interpreter = tflite.Interpreter(path_to_file)
    interpreter.allocate_tensors()


    # Get input tensor.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    details = interpreter.get_tensor_details()
    input_shape = input_details[0]['shape']
    print("\nModel loaded, input_shape:",input_shape)
    print("\nInput details:", input_details)

    print("\nNumber of tensors: ", len(details))
    for i in range(len(details)):
        s = details[i]['quantization'][0]
        z = details[i]['quantization'][1]
        name = details[i]['name']

        scale_factors.append(s)
        zeropoints.append(z)
        layer_names.append(name)
        print(f'\nQuantization parameters for Tensor ', name, '-', details[i]['index'],':')
        print( 'q = ', s, 'z = ', z)


