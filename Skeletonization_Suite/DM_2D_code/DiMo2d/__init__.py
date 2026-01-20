##Pratik

import os
from multiprocessing import Pool
from functools import partial
from matplotlib import image as mpimg
import cv2
import numpy as np
import csv
from geojson import Feature, FeatureCollection, LineString
import geojson as gjson
from math import fabs
from shutil import copyfile
import subprocess
import ctypes
import tempfile
import struct
import time
import timeit

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

def crop_channel(image):
    
    #x, y = np.nonzero(image)
    non_zero_indices = np.argwhere(image != 0)
    x, y = non_zero_indices[:, 0], non_zero_indices[:, 1]

    print('check:', len(x), len(y))
    if len(x) == len(y) == 0:
        raise ValueError("No non-zero elements found in the image.")
        return

    # if np.max(image) == 31:
    #     raise ValueError("Maximum value in the image is 31. Cannot proceed with cropping.")
    #     # return
    
    #print('cropping')
    xl, xr = x.min(), x.max()
    yl, yr = y.min(), y.max()
    image_crop = image[xl:xr+1, yl:yr+1]
    
    crop_coordinates=(str(xl) + ' ' + str(xr+1) + ' ' + str(yl) + ' ' + str(yr+1) + '\n')

    print("Size of image crop: ",image_crop.shape)
    return image_crop, crop_coordinates

def write_dipha_input_file(image):
    
    
    # save_image_input
    data = 255 - image;
        
    # Creating data for writing
    dipha_input=np.zeros((np.size(data)+6),dtype=np.double);
    
    data=data.astype(dtype=np.double)
    #data_flat=data.flatten('F') # Column major order
    data_flat=data.ravel('F')
    
    dipha_input[6:np.size(data)+6]=data_flat
    
    total_elements="%x" % np.size(data) 
    for i in range(1,17-len(total_elements)):
      total_elements='0' + total_elements
    
    total_elements_arranged=''
    for i in range(0,16,2):
      total_elements_arranged = total_elements_arranged + total_elements[14-i:16-i]
    
    dimension="%x" % np.size(np.shape(data))
    for i in range(1,17-len(dimension)):
      dimension='0' + dimension
    
    dimension_arranged=''
    for i in range(0,16,2):
      dimension_arranged = dimension_arranged + dimension[14-i:16-i]
      
    dim1="%x" % np.shape(data)[0] 
    for i in range(1,17-len(dim1)):
      dim1='0' + dim1
    
    dim1_arranged=''
    for i in range(0,16,2):
      dim1_arranged = dim1_arranged + dim1[14-i:16-i]
      
    dim2="%x" % np.shape(data)[1] 
    for i in range(1,17-len(dim2)):
      dim2='0' + dim2
    
    dim2_arranged=''
    for i in range(0,16,2):
      dim2_arranged = dim2_arranged + dim2[14-i:16-i]   
    
    #dipha_input_bytes=dipha_input.tobytes()
    dipha_input_bytes=memoryview(dipha_input).tobytes()
    
    dipha_input_bytes_modified=bytes.fromhex('0046D7E001000000') + bytes.fromhex('0100000000000000') + bytes.fromhex(str(total_elements_arranged)) + bytes.fromhex(str(dimension_arranged)) + bytes.fromhex(str(dim1_arranged)) + bytes.fromhex(str(dim2_arranged)) + dipha_input_bytes[48:]
    
    dipha_input = np.frombuffer(dipha_input_bytes_modified, dtype=np.double)
    
    return dipha_input

def run_dipha_persistence(image,dipha_input,scratch_dir,mpi_threads=1):

    nx, ny = image.shape
    
     ## Using tempfile
#    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as dipha_input_file:
#      dipha_input.tofile(dipha_input_file,format="%f")
#      dipha_filename = dipha_input_file.name
#      
#    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as diagram_file:
#      diagram_filename = diagram_file.name
#      
#    with tempfile.NamedTemporaryFile(mode='wb', delete=False) as dipha_edge_file:
#      dipha_edge_filename = dipha_edge_file.name
    if not os.path.exists(scratch_dir):
       os.mkdir(scratch_dir)

    dif=f"{scratch_dir}/dipha_inputfile"
    with open(dif,"wb") as di:
      di.write(memoryview(dipha_input).tobytes())
      
    dipha_filename=f"{scratch_dir}/dipha_inputfile"
    diagram_filename=f"{scratch_dir}/diagram_filename"
    dipha_edge_filename=f"{scratch_dir}/dipha_edge_filename"
      
    dipha_executable = os.path.join("DM_2D_code", "DiMo2d", "code", "dipha-2d-thresh", "build", "dipha")
    if not os.path.exists(dipha_executable):
       # Fallback to absolute path if needed, or assume it's relative to CWD
       pass

    # Use direct execution instead of mpiexec to avoid MPI issues in Docker
    subprocess.run([dipha_executable, "--upper_dim", "2", dipha_filename, diagram_filename, dipha_edge_filename, str(nx), str(ny)])
    # os.system(dipha_command)

       
    #diagram_bin=np.fromfile(diagram_filename,dtype=np.double)
    dipha_thresh_edges=np.fromfile(dipha_edge_filename,dtype=np.double)
    
    # os.remove(diagram_filename)
    # os.remove(dipha_edge_filename)
    # os.remove(dipha_filename)

    return dipha_thresh_edges

def convert_persistence_diagrams(dipha_thresh_edges):
    
    dipha_thresh_edges_bytes=dipha_thresh_edges.tobytes()

    # load_persistence_diagram
    # if dipha_thresh_edges_bytes[0:8]!=bytes.fromhex('0046D7E001000000'):  # hexadecimal of 8067171840 = 0x0046D7E001000000
    #     raise Exception("input is not a DIPHA file")

    # if dipha_thresh_edges_bytes[8:16]!=bytes.fromhex('0200000000000000'):  
    #     raise Exception("input is not a persistence_diagram file")

    dipha_number=struct.unpack('<q', dipha_thresh_edges_bytes[0:8])[0]
    if dipha_number != 8067171840:
       raise Exception("input is not a DIPHA file")
    
    file_no=struct.unpack('<q', dipha_thresh_edges_bytes[8:16])[0]
    if file_no != 2:
       raise Exception("input is not a persistence_diagram file")
    
    num_pairs=struct.unpack('<q', dipha_thresh_edges_bytes[16:24])[0]
    print("num of pairs: ",num_pairs)

    # num_str=dipha_thresh_edges_bytes[16:24] 

    # num_hex = ''
    # i = len(num_str)-1
    # print("num_str: ",i)
    # for d in range(1, 9):
    #     hex_substr = num_str[i] 
    #     print(hex_substr)
    #     try:
    #         x = hex(hex_substr)[2:]  
    #         print("x: ",x)
    #         num_hex += str(x)  
    #     except ValueError:
    #         pass  
    #     i -= 1
    
    # num_pairs=int(num_hex,16) 
    
    everts=dipha_thresh_edges[4:(3*num_pairs)+5:3]
    everts=everts.astype(np.int64)

    pers=dipha_thresh_edges[5:(3*num_pairs)+6:3]  
    pers=pers.astype(np.int64)

    start=24
    bverts=np.zeros(pers.shape[0]) 
  
    for i in range(0,pers.shape[0]): 
      bverts[i]=struct.unpack('<q', dipha_thresh_edges_bytes[start:start+8])[0]
      start=start+24
    bverts=bverts.astype(np.int64) 

    bverts=bverts.astype(str)
    everts=everts.astype(str)
    pers=pers.astype(str)

    bverts=np.char.add(bverts,' ') 
    #bverts = np.core.defchararray.add(bverts, ' ')
    everts=np.char.add(everts,' ')
    #everts = np.core.defchararray.add(everts, ' ')
    pers=np.char.add(pers,'\n')
    #pers = np.core.defchararray.add(pers, '\n')
    
    dipha_edges_txt = np.core.defchararray.add(bverts, np.core.defchararray.add(everts, pers)) # 1.8 secs
    
    dipha_edges_txt_bytes=dipha_edges_txt.tobytes()   # 0.24

    #dipha_edges_txt_bytes=dipha_edges_txt_bytes.replace(b'\x00',b'')    
    dipha_edges_txt_bytes = dipha_edges_txt_bytes.translate(None, b'\x00')

    dipha_edges_txt=dipha_edges_txt_bytes.decode('utf-8')

    return dipha_edges_txt

def __generate_vert(i, j):
    return f"{i} {j} {image[i, j]}\n"

def write_vertex_file(image):
    nx, ny = image.shape
    
    # i_array=np.tile(np.arange(0, nx), ny)
    # i_array=i_array.astype(str)
    # i_array=np.char.add(i_array,' ')
   
    # j_array=np.repeat(np.arange(0,ny),nx)
    # j_array=j_array.astype(str)
    # j_array=np.char.add(j_array,' ')
   
    # image_array=image.flatten('F')
    # image_array=image_array.astype(str)
    # image_array=np.char.add(image_array,'\n')
   
    # vert_txt = np.core.defchararray.add(i_array, np.core.defchararray.add(j_array, image_array))
    # vert_txt_bytes=vert_txt.tobytes()
    # vert_txt_bytes=vert_txt_bytes.replace(b'\x00',b'')
    # vert_txt = vert_txt_bytes.decode('utf-8')
    
    # vert=[]
    # for j in range(ny):
    #     #temp=[]
    #     for i in range(nx):
    #         k=image[i,j]
    #         q=str(i) + " " + str(j) + " " + str(k) + "\n"
    #         #temp.append(q)
    #         vert.append(q)
    #     #vert.append(temp)

    vert = [f"{i} {j} {image[i, j]}\n" for j in range(ny) for i in range(nx)]
    
    
    vert_txt=''.join(vert)

    return vert_txt

def compute_persistence_single_channel(input_image,scratch_dir):
    [input_image_crop,crop_coordinates]=crop_channel(input_image)
    dipha_input=write_dipha_input_file(input_image_crop)
    dipha_thresh_edges=run_dipha_persistence(input_image_crop,dipha_input,scratch_dir,mpi_threads=1)
    dipha_edges_txt=convert_persistence_diagrams(dipha_thresh_edges)
    vert_txt=write_vertex_file(input_image_crop)

    return input_image_crop,crop_coordinates,dipha_input,dipha_thresh_edges,dipha_edges_txt,vert_txt

def run_graph_reconstruction(dipha_edges_txt,vert_txt,scratch_dir,ve_persistence_threshold=0,et_persistence_threshold=64):

    ## Using temp files
#    with tempfile.NamedTemporaryFile(mode='w', delete=False) as vert_txt_file:
#      vert_txt_file.write(vert_txt)
#      vert_txt_filename = vert_txt_file.name
#
#    with tempfile.NamedTemporaryFile(mode='w', delete=False) as dipha_edges_txt_file:
#      dipha_edges_txt_file.write(dipha_edges_txt)
#      dipha_edges_txt_filename = dipha_edges_txt_file.name
#      
#    with tempfile.NamedTemporaryFile(mode='w', delete=False) as dimo_out_file:
#      dimo_out_filename = dimo_out_file.name
     
    ## Using the shared memory in RAM  
    if not os.path.exists(scratch_dir):
       os.mkdir(scratch_dir)

    vf=f"{scratch_dir}/vert_txtfile"
    with open(vf,"w") as vtf:
      vtf.write(vert_txt)
    
    diedtf=f"{scratch_dir}/dipha_edges_txt_file"
    with open(diedtf,"w") as detf:
      detf.write(dipha_edges_txt)
      
    vert_txt_filename=f"{scratch_dir}/vert_txtfile"
    dipha_edges_txt_filename=f"{scratch_dir}/dipha_edges_txt_file"
    dimo_out_filename=f"{scratch_dir}/dimo"
        
    morse_command = 'DM_2D_code/DiMo2d/code/dipha-output-2d-ve-et-thresh/a.out ' + vert_txt_filename + ' ' + dipha_edges_txt_filename + ' ' + str(ve_persistence_threshold) + ' ' + str(et_persistence_threshold) + ' ' + dimo_out_filename

    os.system(morse_command)
    
    dimo_vert_filename=dimo_out_filename + "dimo_vert.txt"
    dimo_edge_filename=dimo_out_filename + "dimo_edge.txt"

    with open(dimo_vert_filename,"r") as dvf:
      dimo_vert=dvf.read()
      
    with open(dimo_edge_filename,"r") as dedf:
      dimo_edge=dedf.read()
    
    # os.remove(vert_txt_filename)
    # os.remove(dipha_edges_txt_filename)
    # os.remove(dimo_vert_filename)
    # os.remove(dimo_edge_filename)
    
    return dimo_vert,dimo_edge

def shift_vertex_coordinates(crop_coordinates, dimo_vert):
  
    crop_coordinates=crop_coordinates.split("\n")[0]
    crop_split=crop_coordinates.split(" ")
    
    dimo_vert_split=dimo_vert.split("\n")
    size=len(dimo_vert_split)
    
    uncropped=[]
    for j in range(size-1):
      q=dimo_vert_split[j]
      w=q.split(" ")
      c=str(int(w[0])+ int(crop_split[0]))+' '+str(int(w[1])+int(crop_split[2]))+' '+w[2]+'\n'
      uncropped.append(c)
      
    uncropped_dimo_vert=''.join(uncropped)
    
    return uncropped_dimo_vert

def intersect_morse_graph_with_binary_output(binary_image,ve_persistence_threshold, et_persistence_threshold,dimo_edge,uncropped_dimo_vert):

    uncropped_dimo_vert=uncropped_dimo_vert.split("\n")
    
    vert_index_dict = {}
    v_ind = 0
    for i in range(len(uncropped_dimo_vert)-1):
        v = uncropped_dimo_vert[i].split(" ")
        val = binary_image[int(v[0]), int(v[1])]
        
        if val == 255:
            vert_index_dict[i] = v_ind
            v_ind += 1
            continue
        assert (val == 0)
        vert_index_dict[i] = -1

    crossed_vert=[]
    for i in range(len(uncropped_dimo_vert)-1):
      if vert_index_dict[i] == -1:
        continue
      v = uncropped_dimo_vert[i]
      crossed_vert.append(str(v[0]) + ' ' + str(v[1]) + ' ' + v[2] + '\n')
           
    crossed_vert=''.join(crossed_vert)
    dimo_edge=dimo_edge.split("\n")

    crossed_edge=[]
    
    for j in range(len(dimo_edge)-1):
      e=dimo_edge[j].split(" ")
      if vert_index_dict[int(e[0])] == -1 or vert_index_dict[int(e[1])] == -1:
        continue
      crossed_edge.append(str(vert_index_dict[int(e[0])]) + ' ' + str(vert_index_dict[int(e[1])]) + '\n')

    crossed_edge=''.join(crossed_edge)
      
    return crossed_vert,crossed_edge

def remove_duplicate_edges(dimo_edge, ve_persistence_threshold=0, et_persistence_threshold=64):
     
      dimo_edge=dimo_edge.split("\n")
      no_dup_crossed_edge=set()
      
      for i in range(len(dimo_edge)-1):
        v=dimo_edge[i].split(" ")
        v0=int(v[0])
        v1=int(v[1])
        
        if v0 < v1:
          vmin = v0
          vmax = v1
        else:
          vmin = v1
          vmax = v0
        if (vmin, vmax) not in no_dup_crossed_edge:
          no_dup_crossed_edge.add((vmin, vmax))
      
      no_dup_cross=[]
      for e in no_dup_crossed_edge:
        no_dup_cross.append(str(e[0]) + ' ' + str(e[1]) + '\n')
        
      no_dup_crossed_edge=''.join(no_dup_cross)  
      
      return no_dup_crossed_edge


def generate_morse_graphs(dipha_edges_txt,vert_txt,crop_coordinates,binary_image,scratch_dir,ve_persistence_threshold=0, et_persistence_threshold=64):
    [dimo_vert,dimo_edge]=run_graph_reconstruction(dipha_edges_txt,vert_txt,scratch_dir,ve_persistence_threshold=ve_persistence_threshold,et_persistence_threshold=et_persistence_threshold)
    uncropped_dimo_vert=shift_vertex_coordinates(crop_coordinates, dimo_vert)
    # [crossed_vert,crossed_edge]=intersect_morse_graph_with_binary_output(binary_image, ve_persistence_threshold, et_persistence_threshold,dimo_edge,uncropped_dimo_vert)
    # no_dup_crossed_edge=remove_duplicate_edges(crossed_edge, ve_persistence_threshold, et_persistence_threshold)
    no_dup_crossed_edge=remove_duplicate_edges(dimo_edge, ve_persistence_threshold, et_persistence_threshold)

    # return dimo_vert,dimo_edge,uncropped_dimo_vert,crossed_vert,crossed_edge,no_dup_crossed_edge
    return dimo_vert,dimo_edge,uncropped_dimo_vert,no_dup_crossed_edge

def non_degree_2_paths(no_dup_crossed_edge, dimo_vert,scratch_dir,ve_persistence_threshold, et_persistence_threshold):
#    image_output_dir = os.path.join(input_dir, os.path.splitext(image_filename)[0]) \
#        + '/' + str(ve_persistence_threshold) + '_' + str(et_persistence_threshold) + '/'
#    command = './DiMo2d/code/paths_src/a.out ' + image_output_dir
#    os.system(command)

# Files require:  no-dup-crossed-edge.txt  and  dimo_vert.txt
    directory=f"{scratch_dir}/"
    if not os.path.exists(scratch_dir):
       os.mkdir(scratch_dir)

    no=f"{scratch_dir}/no-dup-crossed-edge.txt"
    with open(no,"w") as ndce:
      ndce.write(no_dup_crossed_edge)
    
    dim=f"{scratch_dir}/dimo_vert.txt"
    with open(dim,"w") as dvt:
      dvt.write(dimo_vert) 
      
    command = 'DM_2D_code/DiMo2d/code/paths_src/a.out ' + directory
    os.system(command)
    
    pth=f"{scratch_dir}/paths.txt"
    with open(pth,"r") as pt:
      paths=pt.read()

    # os.remove(f"{scratch_dir}/no-dup-crossed-edge.txt")
    # os.remove(f"{scratch_dir}/dimo_vert.txt")
    # os.remove(f"{scratch_dir}/paths.txt")
       
    return paths


def haircut(dimo_vert,paths,ve_persistence_threshold, et_persistence_threshold):
    verts_lines=dimo_vert.split("\n")

    verts=[]
    for i in range(len(verts_lines)-1):
        v=verts_lines[i]
        line=v.split(" ")
        verts.append([int(line[0]),int(line[1]),int(line[2])])
    
    paths1=paths.split("\n")

    paths2=[]
    for i in paths1:
        k=i.split(" ")
        k.pop()
        paths2.append(k)
    
    paths3 = [[int(n) for n in c] for c in paths2]
    
    valid_paths = []
    
    for i in range(len(paths3)-1):
      '''
      vals = [verts[v][2] for v in p]
      if min(vals) < 50:
          continue
      '''
      p=paths3[i]
      valid_paths.append(p)

    degrees = {}
    for p in valid_paths:
        if p[0] not in degrees.keys():
            degrees[p[0]] = 1
        else:
            degrees[p[0]] += 1
        if p[len(p) - 1] not in degrees.keys():
            degrees[p[len(p) - 1]] = 1
        else:
            degrees[p[len(p) - 1]] += 1

    output_edge=[]
    
    for i in range(len(valid_paths)):
    
        p = valid_paths[i]
        if len(p) < 2:
            output_edge.append(paths[i] + '\n')
            print('less than 2')
            continue
    
        # print(len(verts), p[0], p[1])

        if verts[p[0]][0] == verts[p[1]][0]:
            direction = 1
        else:
            assert (verts[p[0]][1] == verts[p[1]][1])
            direction = 0
        delta = 0
        for j in range(1, len(p)):
            if verts[p[j - 1]][0] == verts[p[j]][0]:
                current_direction = 1
            else:
                assert (verts[p[j - 1]][1] == verts[p[j]][1])
                current_direction = 0
            if current_direction == direction:
                continue
            direction = current_direction
            delta += 1

        first_degree = degrees[p[0]]
        second_degree = degrees[p[len(p) - 1]]

            # haircut
        if delta <= 1 and (first_degree == 1 or second_degree == 1) and (first_degree > 2 or second_degree > 2):
            continue
  
        for j in range(len(p) - 1):
            output_edge.append(str(p[j]) + ' ' + str(p[j + 1]) + '\n')
    
    haircut_edge=''.join(output_edge)
    
    return haircut_edge


def postprocess_graphs(no_dup_crossed_edge, uncropped_dimo_vert,scratch_dir,ve_persistence_threshold=0, et_persistence_threshold=64):
    paths=non_degree_2_paths(no_dup_crossed_edge, uncropped_dimo_vert,scratch_dir,ve_persistence_threshold, et_persistence_threshold)
    # paths=non_degree_2_paths(no_dup_crossed_edge, dimo_vert,scratch_dir,ve_persistence_threshold, et_persistence_threshold)
    haircut_edge=haircut(uncropped_dimo_vert,paths,ve_persistence_threshold, et_persistence_threshold)
    return paths,haircut_edge


def cshl_align_coordinates_with_webviewer(dimo_vert):
#    image_output_dir = os.path.join(input_dir, os.path.splitext(image_filename)[0]) \
#                       + '/' + str(ve_persistence_threshold) + '_' + str(et_persistence_threshold) + '/'
#    # input_filename = os.path.join(image_output_dir, 'crossed-vert.txt')
#    input_filename = os.path.join(image_output_dir, 'dimo_vert.txt')
#    output_filename = os.path.join(image_output_dir, 'json-vert.txt')

    x_vals = []
    y_vals = []
    dimo_vert=dimo_vert.split("\n")
    json_vert=[]
    
    #print("dimo_vert: ",dimo_vert[0:4])
    for i in range(len(dimo_vert)-1):
        row=dimo_vert[i]
        row=row.split(" ")
        raw_y = int(row[1])
        raw_x = int(row[0])

        x = raw_y
        y = - raw_x

        x_vals.append(x)
        y_vals.append(y)

        json_vert.append(str(y) + ' ' + str(x) + ' 0 0' + '\n')
        
    json_vert=''.join(json_vert)

    return json_vert

def __read_ve(json_vert, haircut_edge):
    VX, VY, VZ = 1, 1, 1
    nodes, edges = [], []
    
    json_vert_line=json_vert.split("\n")
    json_vert_line=json_vert_line[:len(json_vert_line)-1]
    
    for line in json_vert_line:
        line_s=line.split(" ")
        node = [float(x) for x in line_s[:3]]
        node[0], node[1] = node[1] * VX, node[0] * VY
        node[2] = node[2] * VZ
        node = tuple(node)
        nodes.append(node)
            
    haircut_edge_line=haircut_edge.split("\n")
    haircut_edge_line=haircut_edge_line[:len(haircut_edge_line)-1]
    
    for line in haircut_edge_line:
        edge = tuple([int(x) for x in line.strip().split()[:2]])
        edges.append(edge)
        '''
        if len(edges) % 100000 == 0:
            print(len(edges))
            sys.stdout.flush()
        '''
    return nodes, edges

def __in_between(z, uz, vz, eps=1e-6):
    max_uv = max(uz, vz)
    min_uv = min(uz, vz)
    return ((min_uv < z + 0.5)) and ((max_uv > z - 0.5))

def __segment(u, v, z):
    if fabs(u[2] - v[2]) < 1e-5:
        # return (u[0]+(u[0] - v[0])/4, u[1]+(u[1] - v[1])/4, v[0]-(u[0] - v[0])/4, v[1]-(u[1] - v[1])/4)
        return (u[0], u[1], v[0], v[1])
    z_top = z - 0.5
    z_down = z + 0.5
    if u[2] > v[2]:
        u, v = v, u
    ru, rv = list(u), list(v)
    if u[2] < z_top:
        scale = (z_top - u[2]) / (v[2] - u[2])
        ru[0] = scale * (v[0] - u[0]) + u[0]
        ru[1] = scale * (v[1] - u[1]) + u[1]
    if v[2] > z_down:
        scale = (v[2] - z_down) / (v[2] - u[2])
        rv[0] = v[0] - scale * (v[0] - u[0])
        rv[1] = v[1] - scale * (v[1] - u[1])
    # return (ru[0]+(ru[0] - rv[0])/4, ru[1]+(ru[1] - rv[1])/4, rv[0]-(ru[0] - rv[0])/4, rv[1]-(ru[1] - rv[1])/4)
    return (ru[0], ru[1], rv[0], rv[1])

def __get_all_segs(nodes, edges, z_range):
    print(len(edges), len(nodes))
    seg_all = [[] for i in range(z_range + 1)]
    max_density = 0.0
    for z in range(z_range):
        # print(z)
        # sys.stdout.flush()
        for e in range(len(edges)):
            edge = edges[e]
            u = nodes[edge[0]]
            v = nodes[edge[1]]
            if __in_between(z, u[2], v[2]):
                seg = __segment(u, v, z)
                density = 1
                # density = get_density(seg, cloud_list[z])
                seg_all[z].append((seg, density, e))  # seg = (x1, y1, x2, y2), density, id(e))
                max_density = max(max_density, density)
    return max_density, seg_all

def __make_geojson(json_filename,seg_all, z_range, dir_path, max_density, ind_array=None, scale=1, max_width=10):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    if ind_array is None:
        ind_array = [i for i in range(z_range)]
    for z in range(z_range):
        features = []
        # json_filename = '{:04d}.json'.format(ind_array[z - 1])
        json_filename_x = f'{json_filename}.json'
        output_file = os.path.join(dir_path, json_filename_x)
        print("output_file: ",output_file)
        for seg in seg_all[z]:
            seg_rescale = [x * scale for x in seg[0]]
            features.append(Feature(id=seg[2], geometry=LineString(
                [(seg_rescale[0], seg_rescale[1]), (seg_rescale[2], seg_rescale[3])]),
                                    properties={"stroke-width": 1}))
        with open(output_file, 'w') as file:
            file.write(gjson.dumps(FeatureCollection(features), sort_keys=True))


def convert_morse_graphs_to_geojson(json_filename,json_vert,haircut_edge,ve_persistence_threshold, et_persistence_threshold,json_out_dir):
#    image_output_dir = os.path.join(input_dir, os.path.splitext(image_filename)[0]) \
#                       + '/' + str(ve_persistence_threshold) + '_' + str(et_persistence_threshold) + '/'
#
#    # flag_image = True
#    # cloud_file = ''
#
#    file_vert = os.path.join(image_output_dir, 'json-vert.txt')
#    file_edge = os.path.join(image_output_dir, 'haircut-edge.txt')
#    dir_name = os.path.dirname(file_vert)
    z_range = 1
    length, width = 24000, 24000
    # print(file_vert, file_edge, z_range, length, width)
    # sys.stdout.flush()
    nodes, edges = __read_ve(json_vert, haircut_edge)

    # print('Get segments')
    # sys.stdout.flush()
    max_density, seg_all = __get_all_segs(nodes, edges, z_range)
    
    __make_geojson(json_filename, seg_all, z_range, json_out_dir, max_density)



def __single_move_geojson_to_folder(input_dir, output_dir, ve_persistence_threshold, et_persistence_threshold, image_filename):
    image_output_dir = os.path.join(input_dir, os.path.splitext(image_filename)[0]) \
                       + '/' + str(ve_persistence_threshold) + '_' + str(et_persistence_threshold) + '/'
    json_filename = os.path.join(image_output_dir, 'GeoJson/0000.json')

    output_filename = os.path.join(output_dir, os.path.splitext(image_filename)[0] + '.json')
    copyfile(json_filename, output_filename)


def move_geojsons_to_folder(input_dir, output_dir, ve_persistence_threshold=0, et_persistence_threshold=64, threads=1):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    image_filenames = [listing for listing in os.listdir(input_dir)]
    image_filenames.sort()

    pool = Pool(threads)
    pool.map(partial(__single_move_geojson_to_folder, input_dir, output_dir, ve_persistence_threshold, et_persistence_threshold), image_filenames)
    pool.close()
    pool.join()


def cshl_post_results(uncropped_dimo_vert, haircut_edge,json_out_dir,json_filename,ve_persistence_threshold=0, et_persistence_threshold=64):
    json_vert=cshl_align_coordinates_with_webviewer(uncropped_dimo_vert)
    convert_morse_graphs_to_geojson(json_filename,json_vert,haircut_edge,ve_persistence_threshold, et_persistence_threshold,json_out_dir)
    #move_geojsons_to_folder(input_dir, output_dir, ve_persistence_threshold, et_persistence_threshold, threads)

# def dm2d_pipeline(input_image,binary_image,ve_persistence_threshold,et_persistence_threshold,json_out_dir,json_filename,scratch_dir):
#    [input_image_crop,crop_coordinates]=crop_channel(input_image)
#    dipha_input=write_dipha_input_file(input_image_crop)
#    dipha_thresh_edges=run_dipha_persistence(input_image_crop,dipha_input,scratch_dir,mpi_threads=1)
#    dipha_edges_txt=convert_persistence_diagrams(dipha_thresh_edges)
#    vert_txt=write_vertex_file(input_image_crop)
   
#    [dimo_vert,dimo_edge]=run_graph_reconstruction(dipha_edges_txt, vert_txt,scratch_dir,ve_persistence_threshold=0,et_persistence_threshold=64)
#    uncropped_dimo_vert=shift_vertex_coordinates(crop_coordinates, dimo_vert)
#    [crossed_vert,crossed_edge]=intersect_morse_graph_with_binary_output(binary_image, ve_persistence_threshold, et_persistence_threshold,dimo_edge,uncropped_dimo_vert)
#    no_dup_crossed_edge=remove_duplicate_edges(dimo_edge, ve_persistence_threshold, et_persistence_threshold)
   
#    paths=non_degree_2_paths(no_dup_crossed_edge, dimo_vert,scratch_dir,ve_persistence_threshold, et_persistence_threshold)
#    haircut_edge=haircut(dimo_vert,paths,ve_persistence_threshold, et_persistence_threshold)
   
#    json_vert=cshl_align_coordinates_with_webviewer(dimo_vert)
#    convert_morse_graphs_to_geojson(json_filename,json_vert,haircut_edge,ve_persistence_threshold, et_persistence_threshold,json_out_dir)
