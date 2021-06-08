import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
print('TensorFlow:', tf.version)


#Function that calculates the number of flops of a model, given its path
#https://github.com/tensorflow/tensorflow/issues/32809

def get_flops(path_to_model):
    model = tf.keras.models.load_model(path_to_model)

    forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])

    graph_info = profile(forward_pass.get_concrete_function().graph,
                        options=ProfileOptionBuilder.float_operation())

    #Consider nuumber of flops: use //2 if only considering multiply accumulates (MACC)
    flops = graph_info.total_float_ops 
    print('Flops: {:,}'.format(flops))

    return flops



# .... Define your model here .... 
#Make list of the model names (fully connected ones and convolutional ones

models_fc = ['model_fc_1l_16', 'model_fc_4l_16', 'model_fc_1l_32', 'model_fc_4l_32', 'model_fc_1l_64', 'model_fc_4l_64', 'model_fc_1l_128', 'model_fc_4l_128', 'model_fc_1l_256', 'model_fc_4l_200', 'model_fc_4l_256', 'model_fc_4l_350']

models_cnn = [ 'model_conv1D_1l_5_1','model_conv1D_1l_5_5', 'model_conv1D_1l_10_5', 'model_conv1D_1l_15_5', 'model_conv1D_2l_10_5', 'model_conv1D_2l_15_5', 'model_conv1D_2l_5_10', 'model_conv1D_1l_15_10', 'model_conv1D_2l_10_10', 'model_conv1D_2l_10_20', 'model_conv1D_2l_20_10']


flops_fc = []
flops_cnn = []

#get flops for the FCNs
for i in range(len(models_fc)):
    model_name = models_fc[i]
    tot_flops = get_flops(f'model_path/{model_name}.h5')
    flops_fc.append(tot_flops)

#get flops for the CNNs
for i in range(len(models_cnn)):
    model_name = models_cnn[i]
    tot_flops = get_flops(f'model_path/{model_name}.h5')
    flops_cnn.append(tot_flops)

   



print('FLOPS for the fully connected networks: ', flops_fc)
print('FLOPS for the fully connected networks: ', flops_cnn)
