dm2d_code='DM_2D_code'

import shutil
import os
os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(pow(2, 40))  # Allow very large images
import sys
import numpy as np
import cv2
from PIL import Image
import math
import time
# import tifffile as tiff
import numpy as np
import subprocess as sp
import sys
import time
# from skimage.io import imsave, imread
from albu_calculations_singleChanel import *
from concurrent.futures import ThreadPoolExecutor
from numba import cuda
from DM2D_Pipeline_Tiled import *
sys.path.append(dm2d_code)
import DiMo2d as dm


def mask(org_img):
    """Binary tissue mask via Otsu thresholding on a downsampled image, with
    morphological cleanup to remove speckle and fill holes.

    Downsample target and kernel sizes scale with the input image (rather
    than a fixed absolute pixel count) so this works on both full whole-brain
    scans and small crops: a fixed 100x downsample with a fixed 10x10 opening
    kernel collapses a 1000x1000 crop to a 10x10 intermediate -- the same
    size as the kernel -- which erases the mask entirely regardless of
    actual tissue content.
    """
    img=np.uint8(org_img)
    img_dim = img.shape

    # Downsample so the shorter side is ~200px (matching the original
    # design's typical result at its old fixed 100x factor on whole-brain
    # scans), without shrinking small images below their native size.
    target_short_side = 200
    scale = min(1.0, target_short_side / min(img_dim[0], img_dim[1]))
    down_size = (max(1, int(img_dim[1] * scale)), max(1, int(img_dim[0] * scale)))

    # down sample the image
    down_size_img = cv2.resize(img, down_size, interpolation=cv2.INTER_AREA)

    # Otsu's thresholding
    _, binary_img = cv2.threshold(down_size_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Kernel sizes as a fraction of the downsampled image (same ratios as
    # the original fixed 10/32 against its typical ~200px downsampled size)
    # so opening/closing stays proportionally meaningful at any input scale.
    short_side = min(down_size)
    open_k = max(1, round(short_side * 0.05))
    close_k = max(1, round(short_side * 0.16))

    # Remove non-brain portion
    kernel = np.ones((open_k, open_k), np.uint8)
    binary_img_no = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # Filling holes
    kernel = np.ones((close_k, close_k), np.uint8)
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


def tile_starts(dim, tile_size=512):
    """Sliding-window tile start positions covering the full dimension.

    range(0, dim-tile_size+1, tile_size) alone silently drops any remainder
    past the last full-size tile (e.g. for dim=1000, tile_size=512, only
    start 0 is produced -- pixels 512-999 are never tiled at all). This adds
    an edge-aligned final tile (which may overlap the previous one) so the
    full image gets covered instead of just its first tile_size x tile_size
    corner.
    """
    starts = list(range(0, dim - tile_size + 1, tile_size))
    if not starts:
        return [0]
    if starts[-1] + tile_size < dim:
        starts.append(dim - tile_size)
    return starts

# ========== Reading Image =========== #
def compute_likelihood(input_image_path, json_out_dir, brain_no, section_num, albu_models, norm_factor=16, likelihood_threshold=40, use_mask=True):
    """Read a whole-section image, tile it, and run ALBU to produce a stitched
    likelihood image. Saves it to {json_out_dir}/lkl/{brain_no}_{section_num}.jpg
    and returns the array."""

    a=time.time()
    print("##----Reading image----##")


    print(input_image_path)
    # img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    img = kakadu_image_read(input_image_path)
    print("Dimension of the input image: ",img.shape)
    width,height,channel=img.shape
    print(img.dtype)
    print(img.max())

    # Getting mask. Raw pixel values (often 16-bit, e.g. PMD ~500, STP ~29000)
    # must be brought into 0-255 range the same way ALBU's own input is
    # normalized (see albu_cal) before Otsu thresholding, or the mask ends up
    # thresholding on wrapped-around noise instead of actual tissue contrast.
    if use_mask:
        print("Computing tissue mask via Otsu thresholding...")
        normalized_channel = np.uint8(img[:,:,0] // norm_factor)
        mask_image = mask(normalized_channel)
        if not np.any(mask_image):
            # Otsu-style bulk tissue/background separation doesn't fit every
            # image -- e.g. sparse, low-intensity thin-fiber signal can fail
            # to produce any foreground at all. An empty mask would silently
            # skip every tile and produce a blank likelihood image, which is
            # worse than just not masking, so fall back to the full frame.
            print("Warning: tissue mask came back empty, falling back to full frame")
            maskB = np.ones((width,height),dtype='uint8')
            maskB = maskB / maskB.max()
            mask_image = np.uint8(maskB) * 255
    else:
        maskB = np.ones((width,height),dtype='uint8')
        maskB = maskB / maskB.max()
        mask_image = np.uint8(maskB) * 255
    b=time.time()
    print("Time to read image: ",b-a," Seconds")
    print("------------Reading Image Completed------------")


    # ==================== Tiling ================= #
    print("------------  Tiling Started  ------------\n")
    a=time.time()
    tile_all_channels = []
    count = 0

    for row in tile_starts(width):
        for column in tile_starts(height):
            if np.sum(mask_image[row:row+512,column:column+512]):
                # ALBU gets every real channel available (e.g. PMD's red+green+blue),
                # not just channel 0 -- see albu_cal() for how this is populated.
                tile_all_channels.append(img[row:row+512,column:column+512,:])
                count = count + 1


    total_tiles=count
    print("Total tiles: ",total_tiles)
    print("------------Tiling Completed------------")

    print("------------Starting  ------------")
    a=time.time()
    albu_out=albu_cal(width,height,total_tiles,tile_all_channels,mask_image,albu_models,norm_factor=norm_factor)

    ALBU_out=np.zeros((width,height),dtype=np.uint8)
    count=0
    for row in tile_starts(width):
        for column in tile_starts(height):
            if np.sum(mask_image[row:row+512,column:column+512]):
                ALBU_out[row:row+512,column:column+512]=albu_out[:,:,count]
                count = count + 1

    likelihood_image = ALBU_out
    # ALBU's raw output is a smooth field where every pixel has some small
    # positive value. Clipping low-confidence background to zero matches the
    # reference likelihood tiles this pipeline is meant to reproduce, and the
    # exact cutoff materially affects downstream precision/recall -- see
    # test_dmnet_fusion_investigation/11_paper_reproduction_evaluation for
    # the sweep this default (and the per-mode presets) were picked from.
    likelihood_image[likelihood_image < likelihood_threshold] = 0
    lkl_path = f"{json_out_dir}/lkl/"
    if not os.path.exists(lkl_path):
        os.mkdir(lkl_path)
    lkl_image=f"{lkl_path}/{brain_no}_{section_num}.jpg"
    cv2.imwrite(lkl_image, likelihood_image)

    b=time.time()
    print("Time to execute : ",b-a," Seconds")
    print("------------Completed-------------")

    return likelihood_image


def skeletonize_likelihood(likelihood_image, json_out_dir, json_out_dir_temp, scratch_dir, brain_no, section_num, ve_persistence_threshold=0, et_persistence_threshold=0):
    """Threshold a likelihood image and run DM2D to produce a skeleton JSON
    ({json_out_dir}/{brain_no}_{section_num}.json) and a binary mask image
    ({json_out_dir}/mask/{brain_no}_{section_num}.jpg). Returns the JSON path."""

    if not os.path.exists(scratch_dir):
        os.mkdir(scratch_dir)

    width, height = likelihood_image.shape

    #------------------DM2D-----------------------------#
    print("-----------------DM2D Started----------------------")
    a=time.time()
    division_x=16
    division_y=16

    _, likelihood_image_bin = cv2.threshold(likelihood_image, 20, 255, cv2.THRESH_BINARY)

    print(f"  VE persistence: {ve_persistence_threshold}, ET persistence: {et_persistence_threshold}")

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

    return json_file_path


def main(input_image_path,json_out_dir,brain_no,section_num,albu_models,scratch_dir,json_out_dir_temp,ve_persistence_threshold=0,et_persistence_threshold=0,norm_factor=16,likelihood_threshold=40,use_mask=True):
    likelihood_image = compute_likelihood(input_image_path, json_out_dir, brain_no, section_num, albu_models, norm_factor=norm_factor, likelihood_threshold=likelihood_threshold, use_mask=use_mask)
    return skeletonize_likelihood(likelihood_image, json_out_dir, json_out_dir_temp, scratch_dir, brain_no, section_num, ve_persistence_threshold=ve_persistence_threshold, et_persistence_threshold=et_persistence_threshold)


if __name__ == '__main__':
    main()
