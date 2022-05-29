import os, sys
import errno
import cv2
import math
import torchvision
from torchvision.models import detection
from PIL import Image, ImageDraw
import shutil
import uuid
import cv2
import numpy as np


RESOLUTION_SUPPORTED=['512x512', '256x256', '1920x1080','800x600']
METHODS_SUPPORTED=['CENTERED_CUT']
    
class PreprocessVideo():
    def __init__(self, method, resolution, detection_iter= 3, margin=0.1):
        
        self.margin=margin
        
        # Default params
        if method == None:
            print ('Use default')
            method = METHODS_SUPPORTED[0]
        if resolution == None:
            print ('Use default')
            resolution = RESOLUTION_SUPPORTED[3]
        
        self.detection_iter = detection_iter
        
        
        if method in METHODS_SUPPORTED : 
            self.method = method
        else:
            raise InterruptedError(errno.EINTR, os.strerror(errno.EINTR),'Method selected not supported')
        
        if resolution in RESOLUTION_SUPPORTED : 
            self.selected_resolution = resolution
            self.selected_width = int(self.selected_resolution.split('x')[0])
            self.selected_height = int(self.selected_resolution.split('x')[1])
            self.aspect_ratio = self.selected_width/self.selected_height
        else:
            raise InterruptedError(errno.EINTR, os.strerror(errno.EINTR),'Resolution selected not supported')
        
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        
    def execute(self, input_video_path, out_video_path):
        # ffmpeg -i in.mp4 -filter:v "crop=out_w:out_h:x:y" out.mp4
        # 1920x1080    Screen Width: 1920 pixels   Screen Height: 1080 pixels
        
        video_cv2 = cv2.VideoCapture(input_video_path)
        width = video_cv2.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_cv2.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_scale = width/height
        
        if self.method == METHODS_SUPPORTED[0]:
            
            
            if ( actual_scale > self.aspect_ratio ) :
                # quitar width
                h = height
                w = h * self.aspect_ratio
                
            elif ( actual_scale < self.aspect_ratio ) :
                # quitar height
                w = width
                h = w * self.aspect_ratio
                
            else:
                w = width
                h = height
            
            x_center = width/2
            y_center = height/2 
                    
            x = int(x_center - w/2)
            y = int(y_center - h/2)
            w = int(w)
            h = int(h)
            
        
        cmd = 'ffmpeg -i {} -filter:v "crop={}:{}:{}:{},scale={}:{}" {}'.format(input_video_path, w, h, x, y, self.selected_width, self.selected_height, out_video_path)
        os.system(cmd) 
        
    def execute_with_BBox(self, input_video_path, out_video_path, bbox):
        
        bbox_x = math.floor(bbox[0])
        bbox_y = math.floor(bbox[1])
        bbox_x2 = math.ceil(bbox[2])
        bbox_y2 = math.ceil(bbox[3])
        
        bbox_w = bbox_x2 - bbox_x
        bbox_h = bbox_y2 - bbox_y
        
        cmd = 'ffmpeg -y -i {} -filter:v "crop={}:{}:{}:{},scale={}:{}:force_original_aspect_ratio=1,pad={}:{}:(ow-iw)/2:(oh-ih)/2" {}'.format(input_video_path, bbox_w, bbox_h, bbox_x, bbox_y, self.selected_width, self.selected_height, self.selected_width, self.selected_height, out_video_path)
        
        
        print(cmd)
        os.system(cmd) 
        
    def get_bbox_signer(self, img_path, threshold):
        img = Image.open(img_path)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        img = transform(img)
        pred = self.model([img])
        pred_class = pred[0]['labels'].numpy()
        pred_score = pred[0]['scores'].detach().numpy()
        pred_boxes = pred[0]['boxes'].detach().numpy()
        indices = [idx for idx in range(len(pred_class)) if pred_class[idx] == 1] # person
        indices = [idx for idx in indices if pred_score[idx] > threshold]

        return pred_boxes[indices][0]
    
    def get_frame_size(self, img_path):
        print ('get get_frame_size')
        img = Image.open(img_path)
        w, h =  img.size
        return w, h
    
    def increase_bbox_limit_from_video(self, bbox_limit, w_max, h_max):
        
        print ('antes: {}'.format(bbox_limit))
        
        inc_margin = int(h_max * self.margin)
        
        for idx in range (2):
            bbox_limit[idx] = bbox_limit[idx] - inc_margin
            bbox_limit[idx+2] = bbox_limit[idx+2] + (inc_margin*2)
            
        if bbox_limit[0] < 0:
            bbox_limit[0] = 0
        
        if bbox_limit[1] < 0:
            bbox_limit[1] = 0
            
        if bbox_limit[2] > w_max:
            bbox_limit[2] = w_max
        
        if bbox_limit[3] > h_max:
            bbox_limit[3] = h_max
            
        print ('despues: {}'.format(bbox_limit))
            
        return bbox_limit
    
    def change_video_fps(self, input_video_path, maxfps):
        
        print ('input_video_path: {}'.format(input_video_path))

        head, tail = os.path.split(input_video_path)
        temp_input_video_path = os.path.join(head, 'temp_'+tail)
        
        cam = cv2.VideoCapture(input_video_path)
        fps = cam.get(cv2.CAP_PROP_FPS)
        if fps > maxfps:
            cmd =  'ffmpeg -i {} -filter:v fps=fps={} {}'.format(input_video_path, maxfps, temp_input_video_path)
            os.system(cmd)
        else:
            cmd =  'cp {} {}'.format(input_video_path, temp_input_video_path)
            os.system(cmd)
            
        return temp_input_video_path

    def get_bbox_limit_from_video(self, input_video_path):
        
        temporal_folder = 'temporal_frames_{}'.format(str(uuid.uuid4()))
        try:
            os.mkdir(temporal_folder)
            print("Directory " , temporal_folder ,  " Created ") 
        except FileExistsError:
            print("Directory " , temporal_folder ,  " already exists")

        # cmd = 'ffmpeg -i {} -vf fps=0.2 {}/img%03d.jpg'.format(input_video_path, temporal_folder)
        
        
        cmd = 'ffmpeg -i {} -vf fps=25 {}/img%03d.jpg'.format(input_video_path, temporal_folder)
        os.system(cmd)

        bbox_limit = [2000,2000,0,0]
        list_files = os.listdir(temporal_folder)
        
        # iterations = len(list_files) / 10
        
        counter_bbox_processed = 0
        iterations = len(list_files) / self.detection_iter
        files_indices = np.arange(0, len(list_files), int(iterations), dtype=int)
        
        for index in files_indices:
            fidx = list_files[index]
            print (fidx)
            filepath = os.path.join(temporal_folder, fidx)
            try:
                bbox = self.get_bbox_signer(filepath, 0.9)
                for idx in range(2):
                    if bbox[idx]<bbox_limit[idx]: 
                        bbox_limit[idx] = bbox[idx]
                    if bbox[idx+2]>bbox_limit[idx+2]: 
                        bbox_limit[idx+2] = bbox[idx+2]
                    counter_bbox_processed = counter_bbox_processed + 1
            except:
                print ('jump')
        
        if counter_bbox_processed < 3:   # No encontró un bbox correcto. Volver a ejecutarlo pero ahora cogiendo todos los frames para analizar el bbox.
            print ('VOLVER A REVISAR Y PASAR POR TODOS LOS FRAMES: {}'.format(int(len(list_files)/2)))
            for index in range(int(len(list_files)/2)):
                fidx = list_files[index*2]
                print (fidx)
                filepath = os.path.join(temporal_folder, fidx)
                try:
                    bbox = self.get_bbox_signer(filepath, 0.9)
                    for idx in range(2):
                        if bbox[idx]<bbox_limit[idx]: 
                            bbox_limit[idx] = bbox[idx]
                        if bbox[idx+2]>bbox_limit[idx+2]: 
                            bbox_limit[idx+2] = bbox[idx+2]
                except:
                    print ('jump')
            
                
        w, h = self.get_frame_size(os.path.join(temporal_folder, list_files[0]))
        # if bbox_limit[0] > h:
        #     bbox_limit[0] = 0
        #     bbox_limit[2] = h
        # if bbox_limit[1] > w:
        #     bbox_limit[1] = 0
        #     bbox_limit[3] = w
            
        print ('bbox_limit: ')
        print(bbox_limit)
        shutil.rmtree(temporal_folder)
        
        return self.increase_bbox_limit_from_video(bbox_limit, w, h)        
        #return bbox_limit
    
    
    #bb = get_prediction(path_image, 0.5, show=True)