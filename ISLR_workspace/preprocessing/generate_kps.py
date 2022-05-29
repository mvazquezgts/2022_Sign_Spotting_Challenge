import numpy as np
import os, sys
import tqdm
import argparse
import shutil

sys.path.append('..')
from preprocessing.src.gen_keypoints import GenKeypointsMediapipeC4 as Keypoints
from preprocessing.src.gen_keypoints import GenKeypointsOpenpose as KeypointsOpenpose


def create_folder(folder, reset = False):
    print ('create_folder: {}'.format(folder))    
    try:
        os.makedirs(folder)    
        print("Directory " , folder ,  " Created ")
    except FileExistsError:
        print("Directory " , folder ,  " already exists")
        if (reset):
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder) 
            print("Directory " , folder ,  " reset")
        else:
            print ('NO RESET')


def main(args):

    gen_keypoints = Keypoints()
    gen_keypoints_openpose = KeypointsOpenpose(args.gpu)
    folder_videos = args.input
    folder_out = args.output
    use_openpose = args.op
    
    
    print ('LIST_VIDEOS')
    list_files = os.listdir(folder_videos)
    print (list_files)
    print (len(list_files))

    create_folder(folder_out, reset = False)
    list_files_out = os.listdir(folder_out)
    
    print (list_files_out)

    
    for file_idx in tqdm.tqdm(list_files):

        if '.mp4' in file_idx:
            filename = os.path.splitext(os.path.basename(file_idx))[0]
            print ('filename: {}'.format(filename))
            
            if filename+'.npy' not in list_files_out:
                
                video_path = os.path.join(folder_videos, file_idx)
                out_path = os.path.join(folder_out, filename+'.json')
                
                if not os.path.isfile(out_path):
                    
                    if use_openpose:
                        out_path = os.path.join(folder_out, filename+'.json')
                        gen_keypoints_openpose.genKeypoints(video_path, out_path)
                    else: 
                        out_path = os.path.join(folder_out, filename+'.npy')
                        keypoints = gen_keypoints.genKeypoints(video_path)
                        np.save(out_path, keypoints)
            else:
                print ('no process')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, default='', type=str)
    parser.add_argument('--op', action='store_true')
    parser.add_argument('--gpu', required=False, default=0, type=int)

    arg = parser.parse_args()
    main(arg)
