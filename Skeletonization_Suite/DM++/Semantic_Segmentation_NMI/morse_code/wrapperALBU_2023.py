import numpy as np
import keras
# import matplotlib.pyplot as plt
import tensorflow as tf
import keras.models
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
# from tensorflow.python.ops import array_ops
# from scipy.linalg._expm_frechet import vec
# from tensorflow.python.framework import ops
# from tensorflow.python.framework.op_def_library import _Flatten, _IsListValue
from keras.callbacks import TensorBoard, ModelCheckpoint
# from modelUnet import *
# from data import *
import cv2
import tensorflow as tf
# from skimage import transform
from scipy import ndimage
import numpy as np
from scipy.ndimage import zoom
from keras.callbacks import TensorBoard
from scipy import misc
import os
# import 
import albu_dingkang
from multiprocessing import Pool
# from torch.multiprocessing import Pool, Process, set_start_method, freeze_support

# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)

#import dm
# from albu import test_eval
# from dmp import tst

eps = 0.0001
# filePath = '/home/pratik/DM++/data_albu/'
# # filePath = '/nfs/data/main/M32/Samik/180830/180830_JH_WG_Fezf2LSLflp_CFA_female_processed/TrainingDataProofread/small_train/train/images/'
# fileList1 = os.listdir(filePath)
# outDir = '/home/pratik/DM++/pred/'
# # outDir = '/nfs/data/main/M32/Samik/180830/180830_JH_WG_Fezf2LSLflp_CFA_female_processed/TrainingDataProofread/small_train/train/pred/'
# os.system("mkdir " + outDir)
# #fileList1 ='StitchedImage_Z052_L001.tif'
# #print(fileList1)
# for fichier in fileList1[:]: # filelist[:] makes a copy of filelist.
#     if not(fichier.endswith(".tif")):
#         fileList1.remove(fichier)
# #print(fileList1)



def testImages(img):
    count = 0
    print("------------------>")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tile=img
    op = albu_dingkang.predict(models_albu, tile, image_type='8_bit_RGB')
    op = np.uint8(op)
    print(op.max(), op.min())

    return op


models_albu = albu_dingkang.read_model([os.path.join('/nfs/data/main/M32/Samik/NMI_models/ALBU_models/MBA_RT/', 'fold{}_best.pth'.format(i)) for i in range(4)])
