import argparse
import os
import shutil
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as f
import torchvision.transforms as transforms
from collections import OrderedDict
import datetime

from PIL import Image
import numpy as np
import cv2

import cv2
import mediapipe as mp
import os, sys
import shutil
import datetime

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

index_mirror = np.concatenate([
                [1,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16],
                [21,22,23,18,19,20],
                np.arange(40,23,-1), np.arange(50,40,-1),
                np.arange(51,55), np.arange(59,54,-1),
                [69,68,67,66,71,70], [63,62,61,60,65,64],
                np.arange(78,71,-1), np.arange(83,78,-1),
                [88,87,86,85,84,91,90,89],
                np.arange(113,134), np.arange(92,113)
                ]) - 1
assert(index_mirror.shape[0] == 133)
#multi_scales = [512,640]
multi_scales = [512]

#MAX_FRAMES = 157
MAX_FRAMES = 200000


class GenKeypointsMediapipe():
    def __init__(self):
        self.holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True, model_complexity=2)
        
    def genKeypoints(self, video_path, output_img=None):
        filename = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        count = 0
        
        print ('genKeypoints: {}'.format(video_path))
        
        output_list = []
        count = 0
        while cap.isOpened():
            count = count + 1
            success, image = cap.read()
            if not success:
                break
            if count%1 != 0:
                continue
            image = cv2.flip(image, 1)
            image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_temp.flags.writeable = False
            results = self.holistic.process(image_temp)

            idx_kps_pose = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]
            kps_frame = np.zeros((53, 3))
            for idx, idx_mediapipe in enumerate(idx_kps_pose):
                if results.pose_landmarks:
                    kps_frame[idx][0] = results.pose_landmarks.landmark[idx_mediapipe].x
                    kps_frame[idx][1] = results.pose_landmarks.landmark[idx_mediapipe].y
                    kps_frame[idx][2] = results.pose_landmarks.landmark[idx_mediapipe].visibility
        
            
            for idx in range(21):
                if results.left_hand_landmarks:
                    kps_frame[idx+len(idx_kps_pose)][0] = results.left_hand_landmarks.landmark[idx].x
                    kps_frame[idx+len(idx_kps_pose)][1] = results.left_hand_landmarks.landmark[idx].y
                    kps_frame[idx+len(idx_kps_pose)][2] = 1
                    
                if results.right_hand_landmarks:
                    kps_frame[idx+len(idx_kps_pose)+21][0] = results.right_hand_landmarks.landmark[idx].x
                    kps_frame[idx+len(idx_kps_pose)+21][1] = results.right_hand_landmarks.landmark[idx].y
                    kps_frame[idx+len(idx_kps_pose)+21][2] = 1
                    
            #print(kps_frame.shape)
            output_list.append(kps_frame)
        
        output_list = np.array(output_list)
        print ('Read nº frames: {}'.format(output_list.shape))
        

        if (output_list.shape[0] > MAX_FRAMES):
            print ('Recortar el final, pues excede el maximo')
            output_list = output_list[0:MAX_FRAMES]

        return output_list


class GenKeypointsMediapipeC4():
    def __init__(self):
        self.holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, smooth_landmarks=True, model_complexity=2)
        
    def genKeypoints(self, video_path, output_img=None):
        filename = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        count = 0
        
        print ('genKeypoints: {}'.format(video_path))
        
        output_list = []
        count = 0
        while cap.isOpened():
            count = count + 1
            success, image = cap.read()
            if not success:
                break
            if count%1 != 0:
                continue
            image = cv2.flip(image, 1)
            image_temp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_temp.flags.writeable = False
            results = self.holistic.process(image_temp)

            idx_kps_pose = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16]
            kps_frame = np.zeros((53, 4))

            #idx_kps_pose = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]
            #kps_frame = np.zeros((55, 4))

            for idx, idx_mediapipe in enumerate(idx_kps_pose):
                if results.pose_landmarks:
                    kps_frame[idx][0] = results.pose_landmarks.landmark[idx_mediapipe].x
                    kps_frame[idx][1] = results.pose_landmarks.landmark[idx_mediapipe].y
                    kps_frame[idx][2] = results.pose_landmarks.landmark[idx_mediapipe].z
                    kps_frame[idx][3] = results.pose_landmarks.landmark[idx_mediapipe].visibility
        
            
            for idx in range(21):
                if results.left_hand_landmarks:
                    kps_frame[idx+len(idx_kps_pose)][0] = results.left_hand_landmarks.landmark[idx].x
                    kps_frame[idx+len(idx_kps_pose)][1] = results.left_hand_landmarks.landmark[idx].y
                    kps_frame[idx+len(idx_kps_pose)][2] = results.left_hand_landmarks.landmark[idx].z
                    kps_frame[idx+len(idx_kps_pose)][3] = 1
                    
                if results.right_hand_landmarks:
                    kps_frame[idx+len(idx_kps_pose)+21][0] = results.right_hand_landmarks.landmark[idx].x
                    kps_frame[idx+len(idx_kps_pose)+21][1] = results.right_hand_landmarks.landmark[idx].y
                    kps_frame[idx+len(idx_kps_pose)+21][2] = results.right_hand_landmarks.landmark[idx].z
                    kps_frame[idx+len(idx_kps_pose)+21][3] = 1
                    
            #print(kps_frame.shape)
            output_list.append(kps_frame)
        
        output_list = np.array(output_list)
        print ('Read nº frames: {}'.format(output_list.shape))
        

        if (output_list.shape[0] > MAX_FRAMES):
            print ('Recortar el final, pues excede el maximo')
            output_list = output_list[0:MAX_FRAMES]

        return output_list    
            
class GenKeypointsOpenpose():
    def __init__(self, gpu_id = 0 ):
        print ('init GenKeypointsOpenpose')
        self.gpu_id = gpu_id
        
    def genKeypoints(self, video_path, output_json):
        print ('genKeypoints')
        openpose_bin_path_and_configuration = 'CUDA_VISIBLE_DEVICES='+str(self.gpu_id)+' /home/gts/projects/mvazquez/openpose/build/examples/openpose/openpose.bin --net_resolution "512x512" --display 0 --scale_number 4 --scale_gap 0.25 --hand --hand_scale_number 6 --hand_scale_range 0.4 --render-pose=0 --model_folder /home/gts/projects/mvazquez/openpose/models'
        # openpose_bin_path_and_configuration = 'CUDA_VISIBLE_DEVICES='+gpu+' /home/gts/projects/mvazquez/openpose/build/examples/openpose/openpose.bin --display 0 --render-pose=0 --hand --model_folder /home/gts/projects/mvazquez/openpose/models'
        command = openpose_bin_path_and_configuration+" --video "+video_path+" --write_json "+output_json+"/"
        
        if os.path.exists(output_json):
            print ("Remove folder: "+output_json) 
            shutil.rmtree(output_json)
            
        print ("Create folder: "+output_json) 
        os.makedirs(output_json)
        
        print(command)
        os.system(command)
        
        print ('end')
        