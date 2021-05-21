#Code Sigfrid
# Imports
import os
import time
import numpy as np
#from matplotlib import pyplot as plt
from termcolor import cprint
import tflite_runtime.interpreter as tflite
# -------

time_tot = time.time()

#Inference on TPU: find time_per_inference vs 'FLOPS; use converted models (floats)

#Settings: CPU/TPU, quantized, normal, set path
var = 'FCN'
chip = 'cpu'
quant = False
models = []
folder = ''
suffix = '.tflite'

if chip == 'cpu':
    if quant == False:
        folder = 'converted/'
        print('Inference for Normal models on CPU')
    else:
        folder = 'quantized/'
        suffix = '_quantized.tflite'
        print('Inference for quantized models on CPU')
else:
    folder = 'edgeTPU/'
    suffix = '_quantized_edgetpu.tflite'
    print('Inference for quantized models on TPU')


model_path = '/home/mendel/inference_tests/'+folder
plots_dir = "plots"



# Load test file data
#data = np.zeros((950000, 256, 1))
data = np.load('/home/mendel/inference_tests/testing_data_300k.npy').astype(np.float32)
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


for m in range(n_models):
    times = []

    model_name = models[m]
    cprint("Starting inference test for: "+model_name, "yellow")

    
    # Load model, initialize interpreter - Add Edge TPU; https://coral.ai/docs/edgetpu/tflite-python/?fbclid=IwAR15uMuMw096qEZFbex5BXumbObDn9dMQakk25ZFqCiatJX48D6NGvUFXyw#update-existing-tf-lite-code-for-the-edge-tpu
    path_to_file = model_path + model_name + suffix
    interpreter = tflite.Interpreter(path_to_file,
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    interpreter.allocate_tensors()


    # Get input tensor.
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    print("\nModel loaded, input_shape:",input_shape)
    print("\nInput details:", input_details)


    interpreter.resize_tensor_input(0, [1, n_samples])
    interpreter.allocate_tensors()

    #N = int(loaded_events/10000) #divide #events by N*10'000 sets
    N = loaded_events

    # Make pedictions for each event and time over 10'000 inferences
    for i in range(N):
        if i%25000==0:
            print(f"Iteration {i}/{N}...")
            print('Average time: ', np.mean(times))     

        
        event_i = data[i, :]  # ---> change this for conv network; add ,:,:]
        event_i = np.expand_dims(event_i, axis=0)

        t0 = time.time()
        interpreter.set_tensor(input_details[0]['index'], event_i)
        interpreter.invoke()
        t = (time.time() - t0)
        
        times.append(t)
        
    #print(times)

    mean = np.mean(times)
    std = np.std(times)

    times_mean.append(mean)
    times_std.append(std)

    pred_rate = 1/mean
    pred_std = pred_rate**2*std

    print(f'Average time per prediction = ({mean*10**3} pm {std*10**3}) ms')
    print(f'Prediction rate = ({pred_rate} pm {pred_std}) pred/s')


np.save(f'{plots_dir}/{var}_inference_float_{chip}_mean.npy', times_mean)
np.save(f'{plots_dir}/{var}_inference_float_{chip}_std.npy', times_std)

    

print(f'\nTotal time elapsed: {time.time()-time_tot}')
cprint("Inference test for Fully connected Network done!", "green")