# Imports
import os
import time
import numpy as np
from termcolor import cprint

from pycoral.utils import edgetpu
from pycoral.adapters import common
# -------


#Inference on TPU: find time_per_inference vs 'FLOPS; use converted models (quantized & compiled for edgetpu)

#Settings: FCN/CNN, quantized, set path
var = 'CNN'
models = []
model_path = '/home/mendel/inference_tests/edgeTPU/'
plots_dir = "plots"


# Load test file data
#data = np.zeros((950000, 256, 1))
data = np.load('/home/mendel/inference_tests/testing_data_300k.npy').astype(np.int8)
loaded_events = data.shape[0]
n_samples = data.shape[1]
n_channels = data.shape[-1]


#Reshape data shape, according to cnn or fcn network

if var == 'FCN':
    print('\nReshape for fully connected network')
    data = np.reshape(data, (loaded_events, -1))  #for fc network: shape is (number_events, 256)
    models = ['model_fc_1l_16', 'model_fc_4l_16', 'model_fc_1l_32', 'model_fc_4l_32', 'model_fc_1l_64', 'model_fc_4l_64', 'model_fc_1l_128', 'model_fc_4l_128', 'model_fc_1l_256', 'model_fc_4l_200', 'model_fc_4l_256', 'model_fc_4l_350']
else:
    print('\nConvolutional networks:')
    #data = np.expand_dims(data, axis=-1) #for cnn network: shape is (number_events, 256, n_channels, 1) 
    models = ['model_conv1D_1l_5_5', 'model_conv1D_1l_10_5', 'model_conv1D_1l_15_5', 'model_conv1D_2l_10_5', 'model_conv1D_2l_15_5', 'model_conv1D_2l_5_10', 'model_conv1D_1l_15_10', 'model_conv1D_2l_10_10', 'model_conv1D_2l_20_10']#, 'model_conv1D_2l_10_20']

n_models = len(models)
print("data.shape", data.shape)


# Make sure saved_models folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)



times_mean = []
times_std = []
medians = []


for m in range(n_models):
    times = []

    model_name = models[m]
    cprint("Starting inference test for: "+model_name, "yellow")

    
    # Load model, initialize interpreter - Add Edge TPU; https://coral.ai/docs/edgetpu/tflite-python/?fbclid=IwAR15uMuMw096qEZFbex5BXumbObDn9dMQakk25ZFqCiatJX48D6NGvUFXyw#update-existing-tf-lite-code-for-the-edge-tpu
    path_to_file = model_path + model_name + '_quantized_edgetpu.tflite'
    interpreter = edgetpu.make_interpreter(path_to_file)
    interpreter.allocate_tensors()

    # Get input tensor.
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    print("Model loaded, input_shape:",input_shape)
    print("Input details:", input_details)

    


    #N = int(loaded_events/10000) #divide #events by N*10'000 sets
    N = int(loaded_events/5)

    # Make pedictions for each event and time over 10'000 inferences
    for i in range(N):
        if i % 6000 == 0:
            print(f"Iteration {i}/{N}...")
            print('Time: ', np.mean(times))
        data_tmp = data[i, :]
        data_tmp = data_tmp[np.newaxis, :]
        
        

        #data_tmp = np.array(data_tmp, dtype=np.float32)
        data_tmp = np.array(data_tmp, dtype=np.int8)

        t0 = time.time()

        interpreter.set_tensor(input_details[0]['index'], data_tmp)
        interpreter.invoke()


        t = time.time() - t0

        if  i != 0:
            times.append(t)

        
    #print(times)

    mean = np.mean(times)
    median = np.median(times)
    std = np.std(times)

    times_mean.append(mean)
    times_std.append(std)
    medians.append(median)

    pred_rate = 1/mean
    pred_std = pred_rate**2*std

    print(f'Average time per prediction = ({mean*10**3} pm {std*10**3}) ms')
    print(f'Prediction rate = ({pred_rate} pm {pred_std}) pred/s')
    print('Median = ', median)


np.save(f'{plots_dir}/{var}_inference_tpu_API_mean.npy', times_mean)
np.save(f'{plots_dir}/{var}_inference_tpu_API_std.npy', times_std)
np.save(f'{plots_dir}/{var}_inference_tpu_API_std.npy', medians)

    
cprint("Inference test for Coral done!", "green")
