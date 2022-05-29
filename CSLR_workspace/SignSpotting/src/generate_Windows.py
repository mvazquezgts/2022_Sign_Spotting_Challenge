import os, sys
import tqdm
import argparse
import shutil
from internal.feeder import Feeder
import numpy as np

def create_folder(folder):
    print ('create_folder: {}'.format(folder))    
    try:
        os.makedirs(folder)    
        print("Directory " , folder ,  " Created ")
    except FileExistsError:
        print("Directory " , folder ,  " already exists")
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder) 
        print("Directory " , folder ,  " reset")
       
def main(args):
       
    folder_in = args.input
    folder_out = args.output
    window_size = args.window_size
    window_stride = args.window_stride
    type_features = args.type_features
    
    feeder = Feeder(type_features, window_size = window_size, window_stride=window_stride) 
    
    for type_features_idx in type_features:
        folder_in_idx = os.path.join(folder_in, type_features_idx)
        folder_out_idx = os.path.join(folder_out, type_features_idx)
        create_folder(folder_out_idx)
        
        print ('LIST_FILES')
        list_files = os.listdir(folder_in_idx)
        print (list_files)
        print (len(list_files))
        
        for file_idx in list_files:
            print ('Processing: {}'.format(file_idx))
            filepath_in = os.path.join(folder_in_idx, file_idx)
            filepath_out = os.path.join(folder_out_idx, file_idx)
            
            data_in = feeder.getItems(filepath_in)
            np.save(filepath_out, data_in)
            
    config_info_file = open(os.path.join(folder_out, 'setting.txt'), 'a')
    config_info_file.write(str(feeder)+'\r\n')
    config_info_file.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, default='', type=str)
    parser.add_argument('--type_features', nargs="+", required=True)
    parser.add_argument('--window_size', required=False, default=40, type=int)
    parser.add_argument('--window_stride', required=False, default=5, type=int)
    arg = parser.parse_args()
    main(arg)
    