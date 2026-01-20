import sys
import os

# Use absolute path for DM_2D_code to ensure correct import regardless of CWD
dm2d_code = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DM_2D_code')
sys.path.append(dm2d_code)
import DiMo2d as dm

import multiprocessing
import numpy as np
import os
import geojson
import shutil
from functools import wraps

def unpack(func):
    @wraps(func)
    def wrapper(arg_tuple):
        return func(*arg_tuple)
    return wrapper

@unpack   # Comment when not using multiprocessing
def dm2d_cal(input_image, binary_image, ve_persistence_threshold, et_persistence_threshold, json_out_dir, json_filename, scratch_root):

    # Use passed scratch directory
    if not os.path.exists(scratch_root):
        os.makedirs(scratch_root, exist_ok=True)

    scratch_dir = f'{scratch_root}/{json_filename}'
    if not os.path.exists(scratch_dir):
        os.mkdir(scratch_dir)

    [input_image_crop,crop_coordinates,dipha_input,dipha_thresh_edges,dipha_edges_txt,vert_txt]=dm.compute_persistence_single_channel(input_image,scratch_dir)

    [dimo_vert,dimo_edge,uncropped_dimo_vert,no_dup_crossed_edge]=dm.generate_morse_graphs(dipha_edges_txt,vert_txt,crop_coordinates,binary_image,scratch_dir,ve_persistence_threshold,et_persistence_threshold)
    # [dimo_vert,dimo_edge,uncropped_dimo_vert,crossed_vert, crossed_edge,no_dup_crossed_edge]=dm.generate_morse_graphs(dipha_edges_txt,vert_txt,crop_coordinates,binary_image,scratch_dir,ve_persistence_threshold,et_persistence_threshold)
    [paths,haircut_edge]=dm.postprocess_graphs(no_dup_crossed_edge,uncropped_dimo_vert,scratch_dir,ve_persistence_threshold,et_persistence_threshold)

    dm.cshl_post_results(uncropped_dimo_vert,haircut_edge,json_out_dir,json_filename,ve_persistence_threshold,et_persistence_threshold)

    shutil.rmtree(scratch_dir)

# ... (dm2d_cal_noMP and merge_json unchanged) ...

def DM2D_Pipeline(input_image,binary_image,division_x,division_y,ve_persistence_threshold,et_persistence_threshold,json_out_dir,json_out_dir_temp,scratch_dir):

    print(input_image.shape)
    y,x=input_image.shape
    print("Dim of input image: ",x,y)
    # json_out_dir_temp=f"{json_out_dir}/temp"
    if(not os.path.exists(json_out_dir_temp)):
        os.mkdir(json_out_dir_temp)

    tile_size_x=x//division_x
    tile_size_y=y//division_y
    total_tiles=(division_y)*(division_x)
    print("Dividing the input image into ",total_tiles,"tiles for processing")
   
    tiles=[]
    binary_tiles=[]
    ve_pers_thrs=[]
    et_pers_thrs=[]
    json_dir_list=[]
    count_list=[]
    scratch_dir_list=[] # New list for scratch dir
    count = 0

    for h in range(0, tile_size_y*division_y, tile_size_y):
        for w in range(0, tile_size_x*division_x, tile_size_x):
            s=np.sum(binary_image[h:h+tile_size_y,w:w+tile_size_x])
            if s>0:
                tiles.append(input_image[h:h+tile_size_y,w:w+tile_size_x])
                binary_tiles.append(binary_image[h:h+tile_size_y,w:w+tile_size_x])
                ve_pers_thrs.append(ve_persistence_threshold)
                et_pers_thrs.append(et_persistence_threshold)
                json_dir_list.append(json_out_dir_temp)
                count_list.append(str(count))
                scratch_dir_list.append(scratch_dir) # Add to list
                # print("Tile no: ",count)
            count=count+1             

    argList = zip(tiles,binary_tiles,ve_pers_thrs,et_pers_thrs,json_dir_list,count_list,scratch_dir_list)
    max_cpu=multiprocessing.cpu_count()
    p = multiprocessing.Pool(max_cpu-5)
    p.map(dm2d_cal, iterable=argList)
    p.close()
    p.join()

    merge_json(json_out_dir,json_out_dir_temp,x,y,division_x,division_y) # Make json_out_dir empty before running the code


def DM2D_Pipeline_without_multiprocessing(input_image,binary_image,division_x,division_y,ve_persistence_threshold,et_persistence_threshold,json_out_dir):

    print(input_image.shape)
    y,x=input_image.shape
    print("Dim of input image: ",x,y)
    json_out_dir_temp=f"{json_out_dir}/temp"
    if(not os.path.exists(json_out_dir_temp)):
        os.mkdir(json_out_dir_temp)

    print("***not using multiprocessing****")
   
    dm2d_cal_noMP(input_image,binary_image,ve_persistence_threshold,et_persistence_threshold,json_out_dir_temp,0)

    merge_json(json_out_dir,json_out_dir_temp,x,y,division_x,division_y) # Make json_out_dir empty before running the code
