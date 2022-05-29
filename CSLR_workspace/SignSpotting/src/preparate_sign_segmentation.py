import os
import argparse
import tqdm
from pathlib import Path
import tools

import pickle
import numpy as np

PADDING_FIX = 9
FPS = 25

def detect_pull_up_down(value1, value2):
    if (value1 == value2):
        return 0
    elif (value1 == 0):
        return 1
    elif (value1 == 1):
        return -1

def main(args):

    folder_in = args.input
    folder_out = args.output

    print('LIST output files of the segmentation process')
    list_files = os.listdir(folder_in)
    print(list_files)
    print(len(list_files))
    tools.create_folder(folder_out, reset=False)

    for predictions_file in tqdm.tqdm(list_files):

        if 'predictions.pkl' in predictions_file:
            print('Processing: {}'.format(predictions_file))
            filepath_in = os.path.join(folder_in, predictions_file)

            predictions_file = predictions_file.replace('_predictions.pkl', '')

            filepath_out = os.path.join(folder_out, predictions_file+'.npy')
            filepath_out_text = os.path.join(
                folder_out, predictions_file+'.txt')
            # file_text = open(filepath_out, 'w')

            count = 0
            start = 0
            end = 0
            new_dict = pickle.load(open(filepath_in, 'rb'))

            arr_detections = []
            for data_idx in range(len(new_dict)-1):
                # print (new_dict[data_idx])
                if detect_pull_up_down(new_dict[data_idx], new_dict[data_idx+1]) == 0:
                    if (count > 0):
                        count = count + 1
                elif detect_pull_up_down(new_dict[data_idx], new_dict[data_idx+1]) == -1:
                    count = count + 1
                    start = data_idx
                elif detect_pull_up_down(new_dict[data_idx], new_dict[data_idx+1]) == 1:
                    if (count > 0):
                        end = start + count
                        print('start: {} & end: {}'.format(start, end))
                        arr_detections.append(
                            [-1, int(start + PADDING_FIX), int(end + PADDING_FIX)])
                        count = 0
            arr_detections_npy = np.asarray(arr_detections)
            np.save(filepath_out, arr_detections_npy)

            fps_ms = 1000/FPS
            with open(filepath_out_text, 'w') as f:
                for arr_idx in arr_detections_npy:
                    start = int(arr_idx[1] * fps_ms)
                    end = int(arr_idx[2] * fps_ms)
                    line = '{},{},{}'.format('None', start, end)
                    f.write(line+'\n')
                f.close()

            # file_text.close()
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, default='', type=str)
    arg = parser.parse_args()
    main(arg)
