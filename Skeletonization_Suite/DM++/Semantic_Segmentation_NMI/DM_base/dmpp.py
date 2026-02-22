import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model, Model, save_model
# import Image
import numpy as np
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.models
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten, merge, UpSampling2D, Reshape, BatchNormalization
from keras.layers import Input, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.ops import array_ops
from scipy.linalg._expm_frechet import vec
from tensorflow.python.framework import ops
from tensorflow.python.framework.op_def_library import _Flatten, _IsListValue
from keras.callbacks import TensorBoard, ModelCheckpoint
from modelUnet import *
from data import *
import cv2
import tensorflow as tf
from skimage import transform
from scipy import ndimage
import numpy as np
from scipy.ndimage import zoom
from skimage.io import imread
from keras.callbacks import TensorBoard
from scipy import misc
from createNetR import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)

model = dmnet('/home/pratik/DM++/from-samik/dmnet_membrane_MBA.hdf5')


def sigmoid(x):
    return 1 / 1 + np.exp(-x)

def testImages(img, dm):
    # model_path='/home/pratik/DM++/from-samik/dmnet_membrane_MBA.hdf5'
    # model = dmnet(model_path)
    # print("Using Model situated at: ",model_path)
    
    X = []
    Y = []
    img = img.astype('float32')
    #dm = imread(name2 + "/" + f1[:-8]+'.tif').astype('float32')
    dm = dm.astype('float32')
    img = img / 255.
    dm = dm / 255.
    print(img.max(), dm.max())       # if img.max():

    X_arr = np.asarray(img)
    X_arr = X_arr[..., np.newaxis]
    X_arr = X_arr[np.newaxis, ...]
    
    Y_arr = np.asarray(dm)
    Y_arr = Y_arr[..., np.newaxis]
    Y_arr = Y_arr[np.newaxis, ...]

    out_img = model.predict([X_arr, Y_arr])

    print(out_img.min(), out_img.max())
    output_img = np.squeeze(out_img[0]) * 255. #* 100000.
    output_img=np.uint8(output_img)

    return output_img