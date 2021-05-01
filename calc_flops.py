import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def flops_fc(n, l):  # n: number of layers, l: length of layers
    return (2*256-1)*l + (n-1)*(2*l-1)*l + (2*l-1)*2

k=10

l1 = k**2*247*10*2
l2 = l1 + k**2*238*10*2
l3 = l2 + k**2*229*10*2
l4 = l3 + k**2*220*10*2

#print(l1 + (2*2470-1)*2, l2 + (2380*2-1)*2, l3 + (2290*2-1)*2, l4 + (2200*2-1)*2)
#print(k**2*237*10*2 + k**2*218*10*2 + (2*2180-1)*2)

f5 = k**2*247*5*2 + k**2*238*5*2
f10 = f5*2
f15 = f5*3
f20 = f5*4

#print(f5 + (2*1190-1)*2, f10 + (2*2380-1)*2, f15 + (2*3570-1)*2, f20 + (2*4760-1)*2)

print(l2 + (2*470-1)*2)

#model = keras.models.load_model(f'models/model_cnn_1layer10_10.h5', compile=False)
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()

# Save the model.
#with open('model.tflite', 'wb') as f:
 # f.write(tflite_model)



length_layers_fc = np.array([16, 32, 64, 128, 256])
number_of_layers_fc = np.arange(1,5)
print(length_layers_fc, number_of_layers_fc)

table = np.zeros((len(number_of_layers_fc), len(length_layers_fc)))


for i,n in enumerate(number_of_layers_fc):
    print(n)
    for j,l in enumerate(length_layers_fc):
        table[i,j] = flops_fc(n,l)
        print(flops_fc(n,l))

print(table)

