dm_base='DM++/Semantic_Segmentation_NMI/DM_base'
morse_code='DM++/Semantic_Segmentation_NMI/morse_code'
dm2d_code='DM_2D_code'

import shutil
import os
os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))  # Allow very large images
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
    print(f"Reading {input_image} with glymur...")
    try:
        import glymur
        jp2 = glymur.Jp2k(input_image)
        img = jp2[:]
    except Exception as e:
        print(f"glymur failed: {e}, falling back to cv2")
        img = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)
        if img is not None and len(img.shape) == 3 and img.shape[2] == 3:
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        raise ValueError(f"Failed to read image: {input_image}")
    
    # Ensure 3 dimensions if expected by downstream code (H, W, C)
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=2)
         
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
def main(input_image_path,json_out_dir,brain_no,section_num,albu_models,model_dmpp,temp_dir,scratch_dir,json_out_dir_temp,ve_persistence_threshold=0,et_persistence_threshold=0,norm_factor=16,use_mask=True, bypass_albu=False):

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
    if use_mask:
        print("Computing tissue mask via Otsu thresholding...")
        mask_image=mask(img[:,:,0])
    else:
        print("Skipping mask (using full image)...")
        maskB = np.ones((width,height),dtype='uint8')
        maskB = maskB / maskB.max()
        mask_image = np.uint8(maskB) * 255
    b=time.time()
    print("Time to read image: ",b-a," Seconds")
    print("------------Reading Image Completed------------")

    if bypass_albu:
        # ---- Bypass ALBU: treat input as pre-computed likelihood image ----
        print("------------Bypassing ALBU Inference------------")
        likelihood_image = img[:,:,0] if len(img.shape) == 3 else img
        lkl_path = f"{json_out_dir}/lkl/"
        if not os.path.exists(lkl_path):
            os.mkdir(lkl_path)
        lkl_image=f"{lkl_path}/{brain_no}_{section_num}.jpg"
        cv2.imwrite(lkl_image, likelihood_image)
    else:
        # ---- Full pipeline: Tiling + DM + ALBU ----

        # ==================== Tiling ================= #
        print("------------  Tiling Started  ------------\n")
        a=time.time()
        id = []
        tile = []
        temp_dir_list=[]
        count = 0

        # Pad to nearest multiple of 512 so edge tiles aren't dropped
        pad_y = (512 - (width % 512)) % 512
        pad_x = (512 - (height % 512)) % 512
        padded_width = width + pad_y
        padded_height = height + pad_x
        print(f"Original dims: {width}x{height}, Padded dims: {padded_width}x{padded_height}")

        img_padded = cv2.copyMakeBorder(img[:,:,0], 0, pad_y, 0, pad_x, cv2.BORDER_REFLECT_101)
        mask_padded = cv2.copyMakeBorder(mask_image, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=0)

        for row in range(0, padded_width-511, 512):
            for column in range(0, padded_height-511, 512):
                if np.sum(mask_padded[row:row+512,column:column+512]):
                    tile.append(img_padded[row:row+512,column:column+512])
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

        print("------------Starting ALBU Inference ------------")
        a=time.time()
        albu_out=albu_cal(padded_width,padded_height,total_tiles,tile,mask_padded,albu_models,norm_factor=norm_factor)

        ALBU_out=np.zeros((padded_width,padded_height),dtype=np.uint8)
        count=0
        for row in range(0, padded_width-511, 512):
            for column in range(0, padded_height-511, 512):
                if np.sum(mask_padded[row:row+512,column:column+512]):
                    ALBU_out[row:row+512,column:column+512]=albu_out[:,:,count]
                    count = count + 1

        # Crop back to original dimensions
        likelihood_image = ALBU_out[:width, :height]

        # likelihood_image[likelihood_image<40] = 0
        lkl_path = f"{json_out_dir}/lkl/"
        if not os.path.exists(lkl_path):
            os.mkdir(lkl_path)
        lkl_image=f"{lkl_path}/{brain_no}_{section_num}.jpg"
        cv2.imwrite(lkl_image, likelihood_image)

        b=time.time()
        print("Time to execute ALBU: ",b-a," Seconds")
        print("------------ALBU Completed-------------")
    
    
    #------------------DEBUG-----------------------------#
    # return removed
    #------------------DEBUG-----------------------------#
    

    #------------------DM2D-----------------------------#
    print("-----------------DM2D Started----------------------")
    a=time.time()
    division_x=16
    division_y=16
    
    _, likelihood_image_bin = cv2.threshold(likelihood_image, 20, 255, cv2.THRESH_BINARY)

    print(f"  VE persistence: {ve_persistence_threshold}, ET persistence: {et_persistence_threshold}")
    bit_depth = 8 # bit depth of the input images (should be 8 or 16-bit)
    background_pixel_val = 0 # background pixel values for real-world neuron fragments

    DM2D_Pipeline(likelihood_image,likelihood_image_bin,division_x,division_y,ve_persistence_threshold,et_persistence_threshold,json_out_dir,json_out_dir_temp,scratch_dir)
    # Use shutil.move instead of os.system to handle special characters in filenames (e.g., &)
    src_json = os.path.join(json_out_dir, "merged_geojson.json")
    dst_json = os.path.join(json_out_dir, f"{brain_no}_{section_num}.json")
    shutil.move(src_json, dst_json)


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
    background_image=np.zeros((width, height), dtype=np.uint8)

    for i in range(0,total_segments):
        x1=gj['features'][i]['geometry']['coordinates'][0][0]
        y1=-gj['features'][i]['geometry']['coordinates'][0][1]
        x2=gj['features'][i]['geometry']['coordinates'][1][0]
        y2=-gj['features'][i]['geometry']['coordinates'][1][1]
        cv2.line(background_image, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1, lineType=cv2.LINE_AA)
    
    cv2.imwrite(skel_bin_path_file, background_image)
    print(">>>>> Saved Binary files: ",f"{skel_bin_path_file}")

    shutil.rmtree(scratch_dir)

if __name__ == '__main__':
    main()
