import os
import numpy as np
import matplotlib.pyplot as plt

#Plot average inference time for 30 events (10000 repetitions) vs FLOPs-changes according to used network

path = "/home/luca/Documents/Project/results_inference"

times_mean_fc_int = np.load(f"{path}/FCN_inference_int_cpu_mean.npy").astype(np.float32)
times_std_fc_int = np.load(f"{path}/FCN_inference_int_cpu_std.npy").astype(np.float32)

times_mean_cnn_int = np.load(f"{path}/CNN_inference_int_cpu.npy").astype(np.float32)
times_std_cnn_int = np.load(f"{path}/CNN_inference_int_cpu.npy").astype(np.float32)



times_mean_fc_float = np.load(f"{path}/FCN_inference_float_cpu_mean.npy").astype(np.float32)
times_std_fc_float = np.load(f"{path}/FCN_inference_float_cpu_std.npy").astype(np.float32)

times_mean_cnn_float = np.load(f"{path}/CNN_inference_float_cpu_mean.npy").astype(np.float32)
times_std_cnn_float = np.load(f"{path}/CNN_inference_float_cpu_std.npy").astype(np.float32)



times_mean_fc_tpu = np.load(f"{path}/FCN_inference_int_tpu_mean.npy").astype(np.float32)
times_std_fc_tpu = np.load(f"{path}/FCN_inference_int_tpu_std.npy").astype(np.float32)

times_mean_cnn_tpu = np.load(f"{path}/CNN_inference_int_tpu_mean.npy").astype(np.float32)
times_std_cnn_tpu = np.load(f"{path}/CNN_inference_int_tpu_std.npy").astype(np.float32)






models_fc = ['model_fc_1l_16', 'model_fc_4l_16', 'model_fc_1l_32', 'model_fc_4l_32', 'model_fc_1l_64', 'model_fc_4l_64', 'model_fc_1l_128', 'model_fc_4l_128', 'model_fc_1l_256', 'model_fc_4l_200', 'model_fc_4l_256', 'model_fc_4l_350']

models_cnn = ['model_conv1D_1l_5_1','model_conv1D_1l_5_5', 'model_conv1D_1l_10_5', 'model_conv1D_1l_15_5', 'model_conv1D_1l_15_10', 'model_conv1D_2l_5_10', 'model_conv1D_2l_10_5', 'model_conv1D_2l_10_10', 'model_conv1D_2l_10_20', 'model_conv1D_2l_20_10', 'model_conv1D_fcn_4l']

FLOPS_fc = np.array([8284, 9868, 16556, 22796, 33100, 57868, 66188, 164876, 132364, 344012, 526348, 917012, 1706812])
FLOPS_cnn = np.array([8972, 18912, 37812, 56712, 92637, 150897, 288132, 539782, 980082, 2031552])


#calculate prediction rates + error propagation
pred_rate_fc_int = 1/times_mean_fc_int
pred_std_fc_int = pred_rate_fc_int**2*times_std_fc_int

pred_rate_cnn_int = 1/times_mean_cnn_int
pred_std_cnn_int = pred_rate_cnn_int**2*times_std_cnn_int



pred_rate_fc_float = 1/times_mean_fc_float
pred_std_fc_float = pred_rate_fc_float**2*times_std_fc_float

pred_rate_cnn_float = 1/times_mean_cnn_float
pred_std_cnn_float = pred_rate_cnn_float**2*times_std_cnn_float



pred_rate_fc_tpu = 1/times_mean_fc_tpu
pred_std_fc_tpu = pred_rate_fc_tpu**2*times_std_fc_tpu

pred_rate_cnn_tpu = 1/(times_mean_cnn_tpu)
pred_std_cnn_tpu = pred_rate_cnn_tpu**2*times_std_cnn_tpu



#Plot Pred Rate vs FLOPS CPU
fig, ax = plt.subplots(1,1)
ax.set_xscale("log")
ax.set_yscale("log")

ax.errorbar(FLOPS_fc, pred_rate_fc_float, fmt="o", yerr=pred_std_fc_float, label = 'fcn: float32')
ax.errorbar(FLOPS_cnn, pred_rate_cnn_float, fmt="*", yerr=pred_std_cnn_float, label = 'cnn: float32')

ax.errorbar(FLOPS_fc, pred_rate_fc_int, fmt="o", yerr=pred_std_fc_int, label = 'fcn: int8')
ax.errorbar(FLOPS_cnn, pred_rate_cnn_int, fmt="*", yerr=pred_std_cnn_int, label = 'cnn: int8')


ax.set(title='Average prediction rate per number of FLOPs')
ax.set(xlabel="Floating point operations/s")
ax.set(ylabel="Predictions/s")
#ax.set_xlim(0, 100000)
#ax.set_ylim(0, 0.002)
ax.legend()
plt.savefig(f"rate_vs_flops_all_cpu.png", bbox_inches="tight", dpi=125)
plt.show()



#Plot Pred Rate vs FLOPS TPU vs CPU, int8
fig, ax = plt.subplots(1,1)
ax.set_xscale("log")
ax.set_yscale("log")

ax.errorbar(FLOPS_fc, pred_rate_fc_int, fmt="go", yerr=pred_std_fc_int, label = 'fcn, int8: CPU')
ax.errorbar(FLOPS_fc, pred_rate_fc_tpu, fmt="o", yerr=pred_std_fc_tpu, label = 'fcn, int8: TPU')

ax.errorbar(FLOPS_cnn[:], pred_rate_cnn_int[:], fmt="r*", yerr=pred_std_cnn_int[:], label = 'cnn, int8: CPU')
ax.errorbar(FLOPS_cnn[:], pred_rate_cnn_tpu[:], fmt="*", yerr=pred_std_cnn_tpu[:], label = 'cnn, int8: TPU')



ax.set(title='Average prediction rate per number of FLOPs')
ax.set(xlabel="Floating point operations/s")
ax.set(ylabel="Predictions/s")
#ax.set_xlim(0, 100000)
#ax.set_ylim(0, 0.002)
ax.legend()
plt.savefig(f"rate_vs_flops_tpu_vs_cpu.png", bbox_inches="tight", dpi=125)
plt.show()


print(f'Highest rate FCN, normal: {models_fc[np.argmax(pred_rate_fc_float)]} - {np.max(pred_rate_fc_float)} pm {pred_std_fc_float[np.argmax(pred_rate_fc_float)]}')
print(f'Highest rate FCN, quantized: {models_fc[np.argmax(pred_rate_fc_int)]} - {np.min(pred_rate_fc_int)} pm {pred_std_fc_int[np.argmin(pred_rate_fc_int)]}')
print('Average rate FCN, TPU:', np.mean(pred_rate_fc_tpu), 'pm', np.std(pred_rate_fc_tpu)/np.sqrt(len(pred_rate_fc_tpu)))

print(f'\nHighest rate CNN, normal: {models_cnn[np.argmax(pred_rate_cnn_float)]} - {np.max(pred_rate_cnn_float)} pm {pred_std_cnn_float[np.argmax(pred_rate_cnn_float)]}')
print(f'Highest rate CNN, quantized: {models_cnn[np.argmax(pred_rate_cnn_int)]} - {np.max(pred_rate_cnn_int)} pm {pred_std_cnn_int[np.argmax(pred_rate_cnn_int)]}')
print(f'Highest rate CNN, TPU: {models_cnn[np.argmax(pred_rate_cnn_tpu)]} - {np.max(pred_rate_cnn_tpu)} pm {pred_std_cnn_tpu[np.argmax(pred_rate_cnn_tpu)]}')
