import pipeline_processDetect_skel_samik_singleChanel_lkl as p
import os
import subprocess
import re
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)
from keras import backend as K
import sys
import time

morse_code='DM++/Semantic_Segmentation_NMI/morse_code'
sys.path.append(morse_code)
import albu_dingkang

dm_base='DM++/Semantic_Segmentation_NMI/DM_base'
sys.path.append(dm_base)
from createNetR import *

ip_dir = sys.argv[1]
op_dir = sys.argv[2]

models_albu = albu_dingkang.read_model([os.path.join('models/', 'fold{}_best.pth'.format(i)) for i in range(4)])
model_dmpp = dmnet('DM++/from-samik/dmnet_membrane_MBA.hdf5')
brain_dir = f"{ip_dir}/"
temp_dir=f"tmp/"
scratch_dir=f"scratch_1/"
json_out_dir = f"{op_dir}"
if(not os.path.exists(json_out_dir)):
    os.mkdir(json_out_dir)
json_out_dir_temp = f"{json_out_dir}/{temp_dir}"


current_dir = os.getcwd()
os.chdir(brain_dir)

list_command = 'ls *\&*F*lossless.jp2'
sections = subprocess.check_output(list_command, shell=True, universal_newlines=True)
sections_list = sections.split("\n")
sections_list=sections_list[:-1]

os.chdir(current_dir)
current_dir = os.getcwd()
print("current directory: ",current_dir)
print("temporary directory: ",temp_dir)
count=0

for image in sections_list:
    a=time.time()
    count=count+1
    input_image_path=f"{brain_dir}/{image}"
    
    
    pattern_f = r'F\d+'
    # pattern_lossy = r'PMD\d+_\d+_\d+_lossy'

    match_f = re.search(pattern_f, image)
    print(image)
    list_str = image.split('_')
    print(list_str)
    brain_no = list_str[1]
    
    print(brain_no)
    if match_f:# and match_lossy:
        section_num = int(list_str[3].split('.')[0])
        print(section_num)
        if section_num: #==58 or section_num==80 or section_num==115 or section_num==156:
            print("-------=========xxxxxxxxxxxxxxx==========-----")
            print(count)
            print("Input Image: ",image)
            print("Section number: ",section_num)

            p.main(input_image_path,json_out_dir,brain_no,section_num,models_albu,model_dmpp,scratch_dir,temp_dir,json_out_dir_temp)
    else:
        print("Pattern not found in the image string. Getting next Image-->")

    b=time.time()
    print("Time to Process: ",b-a," Seconds")

print("-------------COMPLETED-----------------")
