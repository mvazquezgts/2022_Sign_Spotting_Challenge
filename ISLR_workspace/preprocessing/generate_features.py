import numpy as np
import os, sys
import tqdm
import argparse
import shutil

sys.path.append('..')
from preprocessing.src.gen_features import MediapipeOptions
from preprocessing.src.gen_features import GenFeaturesMediapipeC4 as Features

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


def main(arg):
    folder = arg.folder
    type_kps = arg.type_kps
    mscaleX = arg.mscaleX
    adjustZ = arg.adjustZ
    noframeslimit = arg.noframeslimit
    max_frames = arg.max_frames
    list_videos = os.listdir(os.path.join(folder, 'kps'))
    print (len(list_videos))

    folder_out_joints = os.path.join(folder, 'joints_'+type_kps)
    folder_out_bones = os.path.join(folder, 'bones_'+type_kps)
    folder_out_joints_motion = os.path.join(folder, 'joints_motion_'+type_kps)
    folder_out_bones_motion = os.path.join(folder, 'bones_motion_'+type_kps)

    create_folder(folder_out_joints)
    create_folder(folder_out_bones)
    create_folder(folder_out_joints_motion)
    create_folder(folder_out_bones_motion)

    if (type_kps=='C3_xyc'):
        genFeatures = Features(MediapipeOptions.XYC, max_frames, mscaleX, adjustZ, noframeslimit)
    elif (type_kps=='C3_xyz'):
        genFeatures = Features(MediapipeOptions.XYZ, max_frames, mscaleX, adjustZ, noframeslimit)
    elif (type_kps=='C4_xyzc'):
        genFeatures = Features(MediapipeOptions.XYZC, max_frames, mscaleX, adjustZ, noframeslimit)

    
    for video in tqdm.tqdm(list_videos):
        filename = os.path.splitext(os.path.basename(video))[0]
        file_path = os.path.join(folder, 'kps', filename+'.npy')
        keypoints = np.load(file_path)
        
        
        data_joints, data_bones, data_joints_motion, data_bones_motion = genFeatures.getFeatures(keypoints)
        out_path = os.path.join(folder_out_joints, filename+'.npy')
        np.save(out_path, data_joints)
        out_path = os.path.join(folder_out_bones, filename+'.npy')
        np.save(out_path, data_bones)
        out_path = os.path.join(folder_out_joints_motion, filename+'.npy')
        np.save(out_path, data_joints_motion)
        out_path = os.path.join(folder_out_bones_motion, filename+'.npy')
        np.save(out_path, data_bones_motion)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True, type=str)
    parser.add_argument('--type_kps', required=True, default='', type=str)
    parser.add_argument('--max_frames', required=False, default=-1, type=int)
    parser.add_argument('--mscaleX', required=False, default=1, type=float)
    parser.add_argument('--adjustZ', required=False, default=False, type=bool)
    parser.add_argument('--noframeslimit', required=False, default=False, type=bool)
    arg = parser.parse_args()
    print (arg)
    main(arg)