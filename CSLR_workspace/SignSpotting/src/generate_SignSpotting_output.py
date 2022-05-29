import os, sys
import tqdm
import argparse
import shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, '.')
from internal.generate_output import Generate_Output

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
    threshold = args.threshold
    min_margin_is_sign = args.min_margin_is_sign
    min_margin_is_same = args.min_margin_is_same
    fps = args.fps
    windows_size = args.windows_size
    window_stride = args.window_stride

    print('folder_in: {}'.format(folder_in))
    print('folder_out: {}'.format(folder_out))

    create_folder(folder_out)

    print ('LIST_FILES')
    list_files = os.listdir(folder_in)
    print (list_files)
    print (len(list_files))
    
    generate_output = Generate_Output(threshold, min_margin_is_sign, min_margin_is_same, fps, windows_size, window_stride)
    
    config_info_file = open(os.path.join(folder_out, 'setting.txt'), 'a')
    config_info_file.write(str(generate_output)+'\r\n')
    config_info_file.close()
    
    print (generate_output)
    
    for file_idx in list_files:
        if '.npy' in file_idx:
            print ('Processing: {}'.format(file_idx))
            filepath_in = os.path.join(folder_in, file_idx)
            filepath_out_npy = os.path.join(folder_out, file_idx)
            filepath_out_txt = os.path.join(folder_out, Path(file_idx).stem +'.txt')
            
            data_in = np.load(filepath_in)
            
            generate_output.generateOutput(data_in, filepath_out_txt)
            
            np.save(filepath_out_npy, data_in)
            
def main2(args):
    
    folder_in = args.input
    folder_out = args.output
    threshold = args.threshold
    min_margin_is_sign = args.min_margin_is_sign
    min_margin_is_same = args.min_margin_is_same
    fps = args.fps
    windows_size = args.windows_size
    window_stride = args.window_stride
    type_features = args.type_features
    weigths = args.weigths
    
    print('folder_in: {}'.format(folder_in))
    print('folder_out: {}'.format(folder_out))
    print('threshold: {}'.format(threshold))
    print('min_margin_is_sign: {}'.format(min_margin_is_sign))
    print('min_margin_is_same: {}'.format(min_margin_is_same))
    print('fps: {}'.format(fps))
    print('windows_size: {}'.format(windows_size))
    print('window_stride: {}'.format(window_stride))
    print('type_features: {}'.format(type_features))
    print('weigths: {}'.format(weigths))

    create_folder(folder_out)

    print ('LIST_FILES')
    list_files = os.listdir(os.path.join(folder_in, type_features[0]))
    print (list_files)
    print (len(list_files))
    
    generate_output = Generate_Output(threshold, min_margin_is_sign, min_margin_is_same, fps, windows_size, window_stride, type_features, weigths)
    config_info_file = open(os.path.join(folder_out, 'setting.txt'), 'a')
    config_info_file.write(str(generate_output)+'\r\n')
    config_info_file.close()
    
    for file_idx in list_files:
        if '.npy' in file_idx:
            print ('Processing: {}'.format(file_idx))
            filepath_out_npy = os.path.join(folder_out, file_idx)
            # filepath_out_raw_npy = os.path.join(folder_out, 'raw_'+file_idx)
            filepath_out_txt = os.path.join(folder_out, Path(file_idx).stem +'.txt')
            
            list_data_in = []
            for type_features_idx in type_features:
                filepath_in = os.path.join(folder_in, type_features_idx, file_idx)
                list_data_in.append(np.load(filepath_in))
            
            data_in = generate_output.combineOutput(list_data_in)
            # np.save(filepath_out_raw_npy, data_in)

            print ('end: {}'.format(data_in.shape))
            arr_output = generate_output.generateOutput(data_in, filepath_out_txt)
            np.save(filepath_out_npy, arr_output)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ISLR_output')
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, default='', type=str)
    parser.add_argument('--threshold', required=False, default=0.5, type=float)
    parser.add_argument('--min_margin_is_sign', required=False, default=2, type=int)
    parser.add_argument('--min_margin_is_same', required=False, default=2, type=int)
    parser.add_argument('--fps', required=False, default=25, type=int)
    parser.add_argument('--windows_size', required=False, default=40, type=int)
    parser.add_argument('--window_stride', required=False, default=5, type=int)
    parser.add_argument("--type_features", nargs="+", required=True)
    parser.add_argument("--weigths", nargs="+", required=False)
    
    arg = parser.parse_args()
    main2(arg)
        