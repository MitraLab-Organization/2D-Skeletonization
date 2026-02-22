import numpy as np
from createNetR import *

def testImages(img, dm, model):
    # if np.sum(dm):
    #     print("dm input before: ",np.min(dm),np.max(dm))
    img = img.astype('float32')
    dm = dm.astype('float32')
    img = img / 255.
    # if np.sum(dm):
    #     print("albu input: ",img.min(), img.max())
    #     print("dm input after: ",np.min(dm),np.max(dm))
    X_arr = np.asarray(img)
    X_arr = X_arr[..., np.newaxis]
    X_arr = X_arr[np.newaxis, ...]
    
    Y_arr = np.asarray(dm)
    Y_arr = Y_arr[..., np.newaxis]
    Y_arr = Y_arr[np.newaxis, ...]

    out_img = model.predict([X_arr, Y_arr])
    # out_img = sigmoid(out_img)
    img_out = np.uint8(np.squeeze(out_img[0]) * 255.) #* 100000.
    # print("img_out: ",img_out.min(), img_out.max())

    return img_out

