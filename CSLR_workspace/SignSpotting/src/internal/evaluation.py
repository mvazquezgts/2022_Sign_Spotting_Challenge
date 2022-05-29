import os
import fnmatch
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from pathlib import Path



class Generate_Evaluation:
    def __init__(self, numberClasses):
        print('Generate_Evaluation. Number classes: {}'.format(numberClasses))
        self.numberClasses = numberClasses

    def extract_data_from_txt(self, filepath):
        data = []

        with open(filepath) as f:
            contents = f.readlines()

        for line in contents:
            try:
                line_data = line.strip().split(',')
                entry = []
                entry.append(int(line_data[0]))
                entry.append(int(line_data[1]))
                entry.append(int(line_data[2]))
                data.append(entry)
            except:
                print('discard')

        return np.asarray(data)
    
    
    # def extract_performances(self, data_gt, data_eval, threshold_iou):
    #     tp = fp = fn = 0
        
    #     if (data_gt.shape[0] == 0 and data_eval.shape[0] == 0):
    #         return tp, fp, fn
            
    #     if (data_gt.shape[0] == 0):
    #         fp = data_eval.shape[0]
    #         return tp, fp, fn

    #     if (data_eval.shape[0] == 0):
    #         fn = data_gt.shape[0]
    #         return tp, fp, fn

    #     for eval_idx in data_eval:
    #         data_eval = np.delete(eval_idx, 0)
    #         detections = 0
            
    #         for gt_idx in data_gt:
    #             data_gtx = np.delete(gt_idx, 0)
    #             if (self.get_temporal_iou_1d(data_eval, data_gtx) >= threshold_iou):
    #                 detections = detections + 1
                    
    #         if detections > 0:
    #             tp = tp + 1
    #             fp = fp + (detections - 1) # Detectado para el mismo gt varios candidatos iguales por lo que sólo será válido uno y el resto se catalogarán como fp.
    #         else:
    #             fp = fp + 1

    #     fn = data_gt.shape[0] - tp
    #     return tp, fp, fn
    
    
    def extract_performances(self, data_gt, data_eval, threshold_iou):
        tp = fp = fn = 0
        if (data_gt.shape[0] == 0 and data_eval.shape[0] == 0):
            return tp, fp, fn
            
        if (data_gt.shape[0] == 0):
            fp = data_eval.shape[0]
            return tp, fp, fn

        if (data_eval.shape[0] == 0):
            fn = data_gt.shape[0]
            return tp, fp, fn
        
        for gt_idx in data_gt:
            class_id = gt_idx[0]
            data_gt_idx = np.delete(gt_idx, 0)            
            detections = 0
            for eval_idx in data_eval:
                if (eval_idx[0] == class_id):
                    data_eval_idx = np.delete(eval_idx, 0)
                    if (self.get_temporal_iou_1d(data_eval_idx, data_gt_idx) >= threshold_iou):
                        detections = detections + 1
            if detections > 0:
                tp = tp + 1
        
        fn = data_gt.shape[0] - tp
        fp = data_eval.shape[0] - tp

        return tp, fp, fn
    
    

    def calculate_metrics(self, tp, fp, fn, show=False):

        if (tp+fp) > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
            
        if (tp+fn) > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
            
        if (precision+recall) > 0:
            f1 = (2 * (precision * recall)) / (precision + recall)
        else:
            f1 = 0
        
        if show:
            print('tp: {}'.format(tp))
            print('fp: {}'.format(fp))
            print('fn: {}'.format(fn))
            print ('********************************************************************************')
            
            
            print('precision: {}'.format(precision))
            print('recall: {}'.format(recall))
            print('f1: {}'.format(f1))
            print ('********************************************************************************')

        return precision, recall, f1

    def write_metrics_global(self, output_filepath, avg_precision, avg_recall, avg_f1, avg_precision_per_file, avg_recall_per_file, avg_f1_per_file):
        with open(output_filepath, 'w') as f:
            f.write('GLOBAL AVERAGE\r\n')
            f.write('Precision: {}\r\n'.format(avg_precision))
            f.write('Recall: {}\r\n'.format(avg_recall))
            f.write('F1: {}\r\n'.format(avg_f1))
            
            f.write('------------------------------ \r\n')
            
            f.write('AVERAGE per file\r\n')
            f.write('Precision: {}\r\n'.format(avg_precision_per_file))
            f.write('Recall: {}\r\n'.format(avg_recall_per_file))
            f.write('F1: {}\r\n'.format(avg_f1_per_file))
            
            f.close()
    
    def write_metrics_global(self, output_filepath, avg_precision, avg_recall, avg_f1):
        with open(output_filepath, 'w') as f:
            f.write('GLOBAL AVERAGE\r\n')
            f.write('Precision: {}\r\n'.format(avg_precision))
            f.write('Recall: {}\r\n'.format(avg_recall))
            f.write('F1: {}\r\n'.format(avg_f1))  
            f.close()        
    

    def write_metrics_file(self, output_filepath, tp, fp, fn, precision, recall, f1):
        with open(output_filepath, 'w') as f:
            f.write('tp: {}\r\n'.format(tp))
            f.write('fp: {}\r\n'.format(fp))
            f.write('fn: {}\r\n'.format(fn))
            f.write('-------------------\r\n')
            f.write('Precision: {}\r\n'.format(precision))
            f.write('Recall: {}\r\n'.format(recall))
            f.write('F1: {}\r\n'.format(f1))
            f.close()

    def get_temporal_iou_1d(self, v1, v2):
        earliest_start = min(v1[0],v2[0])
        latest_start = max(v1[0],v2[0])
        earliest_end = min(v1[1],v2[1])
        latest_end = max(v1[1],v2[1]) 
        iou = (earliest_end - latest_start) / (latest_end - earliest_start)
        return 0 if iou < 0 else iou
