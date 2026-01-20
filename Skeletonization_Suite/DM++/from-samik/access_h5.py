import numpy as np
import h5py


mba = h5py.File('dmnet_membrane_MBA.hdf5', 'r')

print("Keys: ",list(mba.keys()))

model_weights=mba['model_weights']
optimizer_weights=mba['optimizer_weights']

print("model_weights keys: ",list(model_weights.keys()))
print("optimizer_weights keys: ",list(optimizer_weights.keys()))

conv2d_1=model_weights['conv2d_1']
print("conv2d_1 keys: ",list(conv2d_1.keys()))

conv2d_1_2=conv2d_1['conv2d_1']
print("conv2d_1_2 keys: ",conv2d_1_2.keys())

bias_0=conv2d_1_2['bias:0']
print("bias_0 shape: ",bias_0.shape)
print("kernel:0 shape: ",conv2d_1_2['kernel:0'].shape)