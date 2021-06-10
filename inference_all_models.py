import os
import time
import numpy as np
#from matplotlib import pyplot as plt
from termcolor import cprint
import tflite_runtime.interpreter as tflite

# -------


#Run model inference: as a function of FLOPS on the DevBoard

#Settings: CPU/TPU, quantized, normal, set path
var = 'CNN'         #FCN or CNN
chip = 'cpu'        #cpu or tpu
quant = False        #quantized model or unquantized
models = []
folder = ''
suffix = '.tflite'
call_delegate = False   #activate the TPU-delegate, run on TPU
type = np.int8  #format of the data, depending iff float32-model, or int8-model

if chip == 'cpu':
    if quant == False:
        folder = 'converted/'
        type = np.float32
        print('Inference for Normal models on CPU')
    else:
        folder = 'quantized/'
        suffix = '_quantized.tflite'
        print('Inference for quantized models on CPU')
else:
    folder = 'edgeTPU/'
    suffix = '_quantized_edgetpu.tflite'
    call_delegate = True
    print('Inference for quantized models on TPU')


model_path = '/home/mendel/inference_tests/'+folder
plots_dir = "plots_new"



# Load test file data: 300k events
data = np.load('/home/mendel/inference_tests/testing_data_300k.npy').astype(type)
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
    #data = np.expand_dims(data, axis=-1) #for conv2D network: shape is (number_events, 256, n_channels, 1) 
    models = ['model_conv1D_2l_10_20']#'model_conv1D_1l_5_1', 'model_conv1D_2l_10_20']#['model_conv2D_1l_5_5', 'model_conv2D_1l_10_5', 'model_conv2D_1l_15_5', 'model_conv2D_2l_10_5', 'model_conv2D_2l_15_5', 'model_conv2D_2l_5_10', 'model_conv2D_1l_15_10', 'model_conv2D_2l_10_10', 'model_conv2D_2l_20_10']#, 'model_conv1D_2l_10_20']
    #models = ['model_conv1D_1l_5_5', 'model_conv1D_1l_10_5', 'model_conv1D_1l_15_5', 'model_conv1D_2l_10_5', 'model_conv1D_2l_15_5', 'model_conv1D_2l_5_10', 'model_conv1D_1l_15_10', 'model_conv1D_2l_10_10', 'model_conv1D_2l_20_10']

n_models = len(models)
print("data.shape", data.shape)


# Make sure saved_models folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)



times_mean = []
times_std = []

index = np.arange(0,loaded_events,10000) #choose indices to infere over
print(index)


for m in range(n_models):
    times = []

    model_name = models[m]
    cprint("Starting inference test for: "+model_name, "yellow")

    
    # Load model, initialize interpreter - Add Edge TPU; https://coral.ai/docs/edgetpu/tflite-python/?fbclid=IwAR15uMuMw096qEZFbex5BXumbObDn9dMQakk25ZFqCiatJX48D6NGvUFXyw#update-existing-tf-lite-code-for-the-edge-tpu
    path_to_file = model_path + model_name + suffix

    if call_delegate:
        interpreter = tflite.Interpreter(path_to_file, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    else:
        interpreter = tflite.Interpreter(path_to_file)
    
    interpreter.allocate_tensors()


    # Get input tensor.
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    print("\nModel loaded, input_shape:",input_shape)
    print("\nInput details:", input_details)


    


    #N = int(loaded_events/10000) #divide #events by N*10'000 sets
    N = int(loaded_events/10000)

    

    # Make pedictions for each event and time over 10'000 inferences
    for i in range(N):
        print(f"Iteration {i}/{N}...")
        print('Time: ', np.mean(times))

        u = index[i]
        event_i = data[u, :, :]  # ---> change this for conv network; add ,:,:]
        event_i = np.expand_dims(event_i, axis=0)
      
        t0 = time.time()
        for k in range(10000):
            interpreter.set_tensor(input_details[0]['index'], event_i)
            interpreter.invoke()
            if k == 0:
                t0 = time.time()
        
        t = (time.time() - t0)/9999

        if  i != 0:
            times.append(t)
        print(f'time inference at {u} = {t}')
        
    #print(times)

    mean = np.mean(times)
    std = np.std(times)

    times_mean.append(mean)
    times_std.append(std)

    pred_rate = 1/mean
    pred_std = pred_rate**2*std

    print(f'Average time per prediction = ({mean*10**3} pm {std*10**3}) ms')
    print(f'Prediction rate = ({pred_rate} pm {pred_std}) pred/s')
    print(f'Median = {np.median(times)}')


np.save(f'{plots_dir}/{var}_inference_float_{chip}_mean.npy', times_mean)
np.save(f'{plots_dir}/{var}_inference_float_{chip}_std.npy', times_std)

    
cprint("\nInference test for Fully connected Network done!", "green")