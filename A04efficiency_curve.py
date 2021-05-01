from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras

signal = np.load("data/data_signal_B_1ch_0000.npy").astype(np.float32)

noise = np.zeros((1000000, 1, 256), dtype=np.float32)
for i in range(10):
    noise[(i) * 100000:(i + 1) * 100000] = np.load(f"data/data_noise_3.0SNR_1ch_{i:04d}.npy").astype(np.float32)

x = np.vstack((signal, noise))
x = np.swapaxes(x, 1, 2)
n_samples = x.shape[1]
n_channels = x.shape[2]
x3 = np.expand_dims(x, axis=-1)
y_test = np.zeros((x.shape[0], 2))
y_test[:100000, 1] = 1
y_test[100000:, 0] = 1

model = keras.models.load_model(f'models/model_cnn_1layer10_10.h5', compile=False)
y_pred = model.predict(np.reshape(x, (x.shape[0], -1)))

yy = np.log10(np.linspace(10 ** 0.001, 10 ** 0.99999, 1000))
n = np.zeros_like(yy)
s = np.zeros_like(yy)
for iT, threshold in enumerate(yy):
    eff_signal = np.sum((y_pred[:100000, 1] > threshold) == True) / 100000
    s[iT] = eff_signal
    eff_noise = np.sum((y_pred[100000:, 1] > threshold) == False) / 1000000
    if(eff_noise < 1):
        reduction_factor = np.log10(1 / (1 - eff_noise))
    else:
        reduction_factor = np.log10(1000000)
    n[iT] = 10 ** reduction_factor

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.plot(s, n)
ax.set_xlabel("signal efficiency")
ax.set_ylabel("noise reduction factor")
ax.semilogy(True)
fig.tight_layout()
fig.savefig("plots/efficiency_curve_cnn.png")
plt.show()
