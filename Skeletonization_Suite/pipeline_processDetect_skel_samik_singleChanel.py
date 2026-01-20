dm_base='DM++/Semantic_Segmentation_NMI/DM_base'
morse_code='DM++/Semantic_Segmentation_NMI/morse_code'
dm2d_code='DM_2D_code'

import shutil
import os
os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'
import sys
import numpy as np
import cv2
from PIL import Image
import multiprocessing
import math
import time
# import tifffile as tiff
from functools import wraps
import numpy as np
import subprocess as sp
import sys
import time
# from skimage.io import imsave, imread
from albu_calculations_singleChanel import *
from dmpp_calculations import *
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Manager
from numba import cuda
from DM2D_Pipeline_Tiled import *
sys.path.append(dm2d_code)
import DiMo2d as dm

sys.path.append(dm_base)
from createNetR import *

sys.path.append(morse_code)
import albu_dingkang
import new_dm_mba
import tsting_single_cal


def mask(org_img):
    scaling_factor=100
    # img = cv2.cvtColor(np.uint8(org_img), cv2.COLOR_BGR2GRAY)
    img=np.uint8(org_img)
    img_dim = img.shape
    down_size = (img_dim[1]//scaling_factor, img_dim[0]//scaling_factor)

    # down sample the image 
    down_size_img = cv2.resize(img, down_size, interpolation=cv2.INTER_AREA)

    # Otsu's thresholding
    _, binary_img = cv2.threshold(down_size_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove non-brain portion
    kernel = np.ones((10, 10), np.uint8)
    binary_img_no = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # Filling holes
    kernel = np.ones((32, 32), np.uint8)
    binary_image_no_holes = cv2.morphologyEx(binary_img_no, cv2.MORPH_CLOSE, kernel)

    # up sample the image 
    up_size_img_norm = cv2.resize(binary_image_no_holes, (img_dim[1], img_dim[0]), interpolation=cv2.INTER_CUBIC)
    
    # binarizing the upsampled image
    _, up_size_img_norm_bin = cv2.threshold(up_size_img_norm, 5, 255, cv2.THRESH_BINARY)
    up_size_img_norm_bin = up_size_img_norm_bin.astype(np.uint8)

    return up_size_img_norm_bin

def kakadu_image_read(input_image):
    # Replaced kakadu with cv2
    img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {input_image}")
    
    # Ensure 3 dimensions if expected by downstream code (H, W, C)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
    elif len(img.shape) == 3 and img.shape[2] == 3:
         # Convert BGR to RGB if necessary
         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         
    return img

def imwrite_fast(img_path, opImg):
    # Replaced kakadu with cv2
    cv2.imwrite(img_path, opImg)

def unpack(func):
    @wraps(func)
    def wrapper(arg_tuple):
        return func(*arg_tuple)
    return wrapper

@unpack
def dm_fn(tile,id,temp_dir):
    persistence_th=128
    t_arr=np.asarray(tile)
    t_arr_th = np.where(t_arr>5, 1, 0)

    if np.sum(t_arr_th):
        dm_op = new_dm_mba.dm_cal(tile,id,persistence_th,temp_dir)
    else:
        # print("xxxxxxxxxx-->",id)
        dm_op=np.zeros_like(tile)
    return dm_op


# ========== Reading Image =========== #
def main(input_image_path,json_out_dir,brain_no,section_num,albu_models,model_dmpp,temp_dir,scratch_dir,json_out_dir_temp):

    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)
    
    if not os.path.exists(scratch_dir):
        os.mkdir(scratch_dir)

    a=time.time()
    print("##----Reading image----##")


    print(input_image_path)
    # img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    img = kakadu_image_read(input_image_path)
    print("Dimension of the input image: ",img.shape)
    width,height,channel=img.shape
    print(img.dtype)
    print(img.max())
    # return removed

    # Getting mask
    mask_image=mask(img[:,:,0])
    # maskB = np.ones((width,height),dtype='uint8')
    # maskB = maskB / maskB.max()
    # mask_image = np.uint8(maskB) * 25
    b=time.time()
    print("Time to read image: ",b-a," Seconds")
    print("------------Reading Image Completed------------")


    # ==================== Tiling ================= #
    print("------------  Tiling Started  ------------\n")
    a=time.time()
    id = []
    tile = []
    temp_dir_list=[]
    count = 0

    for row in range(0, width-511, 512):
        for column in range(0, height-511, 512):
            if np.sum(mask_image[row:row+512,column:column+512]):
                tile.append(img[row:row+512,column:column+512,0])
                id.append(count)
                count = count + 1
                temp_dir_list.append(temp_dir)


    total_tiles=count
    print("Total tiles: ",total_tiles)
    print("------------Tiling Completed------------")

    # ============ dm ================= #
    print("------------DM Started------------\n")

    argList = zip(tile,id,temp_dir_list)
    max_cpu=multiprocessing.cpu_count()
    p = multiprocessing.Pool(max_cpu-5)
    # p = multiprocessing.Pool(5)
    dm_opL = p.map(dm_fn, iterable=argList)
    p.close()
    p.join()


    b=time.time()
    print("Time to execute DM: ",b-a," Seconds")

    print("------------DM Completed------------")
    shutil.rmtree(temp_dir)
    # =============================== #
            
    print("------------Starting  ------------")
    a=time.time()
    albu_out=albu_cal(width,height,total_tiles,tile,mask_image,albu_models)

    ALBU_out=np.zeros((width,height),dtype=np.uint8)
    count=0
    for row in range(0, width-511, 512):
        for column in range(0, height-511, 512):
            if np.sum(mask_image[row:row+512,column:column+512]):
                ALBU_out[row:row+512,column:column+512]=albu_out[:,:,count]
                count = count + 1
                
    likelihood_image = ALBU_out
    likelihood_image[likelihood_image<40] = 0
    lkl_path = f"{json_out_dir}/lkl/"
    if not os.path.exists(lkl_path):
        os.mkdir(lkl_path)
    lkl_image=f"{lkl_path}/{brain_no}_{section_num}.jpg"
    cv2.imwrite(lkl_image, likelihood_image)

    b=time.time()
    print("Time to execute : ",b-a," Seconds")
    print("------------Completed-------------")
    
    
    #------------------DEBUG-----------------------------#
    # return removed
    #------------------DEBUG-----------------------------#
    

    #------------------DM2D-----------------------------#
    print("-----------------DM2D Started----------------------")
    a=time.time()
    division_x=16
    division_y=16
    
    _, likelihood_image_bin = cv2.threshold(likelihood_image, 20, 255, cv2.THRESH_BINARY)

    ve_persistence_threshold=0
    et_persistence_threshold=0
    bit_depth = 8 # bit depth of the input images (should be 8 or 16-bit)
    background_pixel_val = 0 # background pixel values for real-world neuron fragments

    DM2D_Pipeline(likelihood_image,likelihood_image_bin,division_x,division_y,ve_persistence_threshold,et_persistence_threshold,json_out_dir,json_out_dir_temp,scratch_dir)
    rename_command=f"mv {json_out_dir}/merged_geojson.json {json_out_dir}/{brain_no}_{section_num}.json"
    os.system(rename_command)


    b=time.time()
    print("Time to execute DM2D: ",b-a," Seconds")
    print("-----------------DM2D Completed----------------------")
    print(">>>>> Saved JSON files: ",f"{json_out_dir}/{brain_no}_{section_num}.json")

    json_file_path = f"{json_out_dir}/{brain_no}_{section_num}.json"
    skel_bin_path = f"{json_out_dir}/mask/"
    if not os.path.exists(skel_bin_path):
        os.mkdir(skel_bin_path)
    skel_bin_path_file =  f"{json_out_dir}/mask/{brain_no}_{section_num}.jpg"

    
    with open(json_file_path) as f:
        gj = geojson.load(f)

    total_segments=len(gj['features'])
    background_image=np.zeros_like(img)

    for i in range(0,total_segments):
        x1=gj['features'][i]['geometry']['coordinates'][0][0]
        y1=-gj['features'][i]['geometry']['coordinates'][0][1]
        x2=gj['features'][i]['geometry']['coordinates'][1][0]
        y2=-gj['features'][i]['geometry']['coordinates'][1][1]
        cv2.line(background_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1, lineType=cv2.LINE_AA)
    
    cv2.imwrite(skel_bin_path_file, background_image)
    print(">>>>> Saved Binary files: ",f"{skel_bin_path_file}")

    shutil.rmtree(scratch_dir)

if __name__ == '__main__':
    main()
