from textwrap import indent
from webvtt import WebVTT, Caption
import os
import math
import pympi

class Generate_Subtitles:
    def __init__(self, list_labels):
        print ('Generate_Visualization')
        self.list_labels =list_labels
        
        

    def generateSubtitles(self, input_filter_file_path, input_file_path, filepath_gt, output_folder_path):
        #print (input_file_path)
        #print ('folder gt: {}'.format(filepath_gt))
        
        filename = os.path.basename(input_file_path).split('.')[0]
        
        self.vtt = WebVTT()
        self.eafob = pympi.Elan.Eaf()
        self.eafob.add_tier('gt')
        self.eafob.add_tier('predictions')
        self.eafob.add_tier('filter')
        # self.eafob.add_tier('segmentador')
        # self.eafob.add_tier('filter_segmentador')
        self.eafob.add_linked_file('{}.mp4'.format(filename),mimetype='video/mp4')
        # self.eafob.add_linked_file('{}.wav'.format(filename),mimetype='audio/x-wav')
        
        # MEDIA_URL="file:///G:/Mi unidad/Anotación é-Saúde/U-px3QJ6XSU.mp4" MIME_TYPE="video/mp4"
        
        try:
            file_gt = open(filepath_gt, 'r')
            for line in file_gt.readlines():
                entry = line.strip().split(',')
                self.add_caption_eaf('gt', entry[0], int(entry[1]), int(entry[2]))
            #print ('Insert file_gt data')
        except:
            pass
            #print (' No filepath_gt: {}'.format(filepath_gt))
        
        # Using readlines()
        try:
            file_input = open(input_file_path, 'r')
            for line in file_input.readlines():
                entry = line.strip().split(',')
                self.add_caption_eaf('predictions', entry[0], int(entry[1]), int(entry[2]))
                #self.add_caption_vtt(entry[0], int(entry[1]), int(entry[2]))
            #print ('Insert file_input data')
        except:
            pass
            #print (' No input_file_path: {} '.format(input_file_path))
            
        try:
            file_filter = open(input_filter_file_path, 'r')
            for line in file_filter.readlines():
                entry = line.strip().split(',')
                self.add_caption_eaf('filter', entry[0], int(entry[1]), int(entry[2]))
                #self.add_caption_vtt(entry[0], int(entry[1]), int(entry[2]))
            #print ('Insert file_filter data')
        except:
            pass
           #print (' No input_filter_file_path: {} '.format(input_filter_file_path))
           
        
        # input_predictions_txt = os.path.join(output_folder_path, 'arr_detections.txt')
        # try:
        #     file_outputs = open(input_predictions_txt, 'r')
        #     for line in file_outputs.readlines():
        #         entry = line.strip().split(',')
        #         self.add_caption_eaf('segmentador', "None", int(entry[1]), int(entry[2]))
        # except:
        #    print (' No input_filter_file_path: {} '.format(input_predictions_txt))
           
        # input_filter2_file_path = os.path.join(output_folder_path, 'filter2_U-px3QJ6XSU.txt')
        # try:
        #     file_outputs = open(input_filter2_file_path, 'r')
        #     for line in file_outputs.readlines():
        #         entry = line.strip().split(',')
        #         self.add_caption_eaf('filter_segmentador', entry[0], int(entry[1]), int(entry[2]))
        # except:
        #    print (' No input_filter_file_path: {} '.format(input_filter2_file_path))
            
            
        #self.vtt.save(os.path.join(output_folder_path, filename))
        #self.vtt.save_as_srt(os.path.join(output_folder_path, filename))
        
        self.eafob.to_file(os.path.join(output_folder_path, filename+'.eaf'))
        
    def add_caption_vtt(self, text, frame_start, frame_end):
        
        caption = Caption(
            self.convert_ms_to_time(frame_start),
            self.convert_ms_to_time(frame_end),
            self.list_labels[int(text)]
        )
        self.vtt.captions.append(caption)
        
    def add_caption_eaf(self,tier, text, frame_start, frame_end):
        if text=='None':
            self.eafob.add_annotation(tier, frame_start, frame_end, value='None')
        else:
            self.eafob.add_annotation(tier, frame_start, frame_end, value=self.list_labels[int(text)])
    
    
    def convert_ms_to_time(self, milliseconds):
            
        hours = math.floor(milliseconds/3600000)
        milliseconds = milliseconds - (3600000 * hours)
        minutes = math.floor(milliseconds/60000)
        milliseconds = milliseconds - (60000 * minutes)
        seconds = math.floor(milliseconds/1000)
        milliseconds = milliseconds - (1000 * seconds)
        
        str_out = '{}:{}:{}.{}'.format(str(hours).zfill(2), str(minutes).zfill(2), str(seconds).zfill(2), str(milliseconds).zfill(3))
        # print (str_out) 
        return str_out
        