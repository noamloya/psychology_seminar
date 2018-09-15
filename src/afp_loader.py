import numpy as np
import fnmatch
import os

def get_data_from_directory(path='../resources/s1_rest', pattern='a*.BLN', init_idx=3, afp_fixed_len=50):

    file_list = fnmatch.filter(os.listdir(path), pattern)
    print("file list len:", len(file_list))
    subj_to_afp = {}
    min_afp_length = 1000
    max_afp_length = 0
    for file in file_list:
        f = open(os.path.join(path, file))
        subj_id = int(file.split('_')[0][1:])
        afp = [float(line.strip('\n')) for line in f.readlines()]
        afp = np.array(afp[init_idx:init_idx+afp_fixed_len])
        subj_to_afp[subj_id] = afp
    return subj_to_afp, min_afp_length, max_afp_length


subj_to_afp, min_afp_length, max_afp_length = get_data_from_directory()

print (min_afp_length, max_afp_length)