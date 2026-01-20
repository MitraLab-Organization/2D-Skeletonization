from skimage.io import imread
import os
import cv2
import tifffile as tiff
import numpy as np

def imread_fast(img_path):
    img_path_C= img_path.replace("&", "\&")
    print(img_path_C)
    base_C = os.path.basename(img_path_C)
    print(base_C)
    base_C = base_C[0:-4]
    print(base_C)
    base = os.path.basename(img_path)
    print(base)
    base = base[0:-4]
    print(base)
    # print("./kdu_expand -i "+img_path_C+" -o temp/"+base_C+".tif -num_threads 1")
    # os.system("cd /home/samik/v7_A_6-01832N/bin/Linux-x86-64-gcc/")
    os.system("kdu_expand -i "+img_path_C+" -o temp/"+base_C+".tif -num_threads 32")
    # print(err_code)
    # os.system("cd /home/samik/Mask_RCNN/samples/nucleus")
    # img = imread('temp/'+base+'.tif')
    # img=cv2.imread('temp/'+base+'.tif',cv2.IMREAD_UNCHANGED)
    img=tiff.imread('temp/'+base+'.tif')
    # os.system("rm temp/"+base_C+'.tif')
    return img

input_image='dataset/1/PMD1229_F54_1_0160_likelihood_green.jp2'
img=imread_fast(input_image)
print("dim of img: ",img.shape)
# print("Sum of red ch: ",np.sum(img[:,:,0]))
# print("Sum of green ch: ",np.sum(img[:,:,1]))
# print("Sum of blue ch: ",np.sum(img[:,:,2]))