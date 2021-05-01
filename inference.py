#Code Sigfrid
# Imports
import os
import time
import numpy as np
#from matplotlib import pyplot as plt
from termcolor import cprint
from tflite_runtime.interpreter import Interpreter 
# -------

# Constants
plots_dir = "plots"
model_name = 'model_fc_2l_64'
models_path = '/home/mendel/inference_tests/converted'

# Make sure saved_models folder exists
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

cprint("Starting inference test for Fully connected Network, 2 layers, length: 64", "yellow")

# Load model
path_to_model = f'{models_path}/{model_name}.tflite'
interpreter = Interpreter(path_to_model)
interpreter.allocate_tensors()

# Get input tensor.
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
print("Model loaded, input_shape:",input_shape)
print("Input details:", input_details)

# Load test file data
#data = np.zeros((950000, 256, 1))
data = np.load('/home/mendel/inference_tests/testing_data_300k.npy').astype(np.float32)
loaded_events = data.shape[0]
n_samples = data.shape[1]
n_channels = data.shape[-1]

if model_name[6]+model_name[7] == 'fc':
  print('reshape for fully connected network')
  data = np.reshape(data, (data.shape[0], -1))  #for fc network: shape is (950000, 256)
else:
  print('reshape for convolutional network')
  data = np.expand_dims(x_1, axis=-1) #for cnn network



# Amount of times to do 1-inferences:
times_mean = []
times_std = []

batch_sizes = np.logspace(np.log10(5), np.log10(loaded_events/10.0), num=20, dtype=int)
# Remove duplicates
batch_sizes = list(set(batch_sizes))

print("Batch sizes: ", batch_sizes)
print("loaded_events/10.0", loaded_events/10.0)
print("data.shape", data.shape)



for batch_size in batch_sizes:
    times = []

    N = min(30, int(np.floor((loaded_events-1)/batch_size)))
    print("This time, N = ", N)
    # Make pedictions and time it
    for i in range(N):
        print(f"On step {i}/{N}...")
        data_tmp = data[(i)*batch_size+1:(i+1)*batch_size, :]  # ---> change this for conv network; add ,:,:]
        #data_tmp = data_tmp[np.newaxis, :, :, :]
        print("Shape of data_tmp:", data_tmp.shape)

        data_tmp = np.array(data_tmp, dtype=np.float32)

        if i == 0:
            interpreter.resize_tensor_input(0, [data_tmp.shape[0], n_samples])   #0, [data_tmp.shape[0], 5, 512, 1] not sure about this
            interpreter.allocate_tensors()

        t0 = time.time()

        interpreter.set_tensor(input_details[0]['index'], data_tmp)

        interpreter.invoke()

        t = time.time() - t0
        if i != 0:
            times.append(t)

    print(times)

    mean = np.mean(times)
    std = np.std(times)

    times_mean.append(mean)
    times_std.append(std)

print(times_mean)
print(times_std)

#fig, ax = plt.subplots(1,1)

#ax.set_xscale("log")
#ax.set_yscale("log")

#ax.errorbar(batch_sizes, times_mean, fmt="o", yerr=times_std)

#ax.set(title='Time per inference over events per inference')
#ax.set(xlabel="Events per inference")
#ax.set(ylabel="Time per inference (s)")
# plt.xlabel("Batch size")
# plt.ylabel("Time")
#plt.savefig(f"{plots_dir}/{model_name}_inference_test.png")

with open(f'{plots_dir}/{model_name}_inference_test.npy', 'wb') as f:
    np.save(f, batch_sizes)
    np.save(f, times_mean)
    np.save(f, times_std)

cprint("Inference test for Fully connected Network done!", "green")