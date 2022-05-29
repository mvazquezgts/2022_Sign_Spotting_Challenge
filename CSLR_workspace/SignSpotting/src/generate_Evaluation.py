import os, sys
import tqdm
import argparse
import shutil
import numpy as np
import tools
import pickle

# sys.path.insert(0, '/home/temporal2/mvazquez/CSLR_workspace/sign_spotting/src')
sys.path.insert(0, '.')
from internal.visualization import Generate_Visualization
from internal.evaluation import Generate_Evaluation
from internal.substitles import Generate_Subtitles



def create_folder(folder, remove_old = True):
    print ('create_folder: {}'.format(folder))    
    try:
        os.makedirs(folder)    
        print("Directory " , folder ,  " Created ")
    except FileExistsError:
        print("Directory " , folder ,  " already exists")
        if (remove_old == True):
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder) 
            print("Directory " , folder ,  " reset")

    
def main(args):
       
    folder_in =args.input
    folder_out = args.output
    folder_gt = args.gt
    labels = args.labels
    vis = args.vis
    metric = args.metric
    subs = args.subs
    #threshold_iou = args.threshold_iou
    
    threshold_iou_min = args.threshold_iou_min
    threshold_iou_max = args.threshold_iou_max
    threshold_iou_step = args.threshold_iou_step
    if threshold_iou_max > threshold_iou_min:
        threshold_iou = np.arange(start=threshold_iou_min, stop=threshold_iou_max+threshold_iou_step-0.01, step=threshold_iou_step)
    else:
        threshold_iou = np.array([threshold_iou_min])
    
    
    print('folder_in: {}'.format(folder_in))
    print('folder_out: {}'.format(folder_out))
    print('folder_gt: {}'.format(folder_gt))
    print('labels: {}'.format(labels))
    print('vis: {}'.format(vis))
    print('metric: {}'.format(metric))
    print('threshold_iou: {}'.format(threshold_iou))
    
    tools.create_folder(folder_out, reset=False)
    
    
    list_words = tools.get_list_words(labels)
    gen_vis = Generate_Visualization()
    gen_eval = Generate_Evaluation(len(list_words))
    gen_subs = Generate_Subtitles(list_words)

                    
    if (vis == True):
        list_files_in = os.listdir(folder_in)
        print (list_files_in)
        print (len(list_files_in))
        
        for file_idx in list_files_in:
            if '.npy' in file_idx:
                
                filepath_in = os.path.join(folder_in, file_idx)
                data_in = np.load(filepath_in)
                data_in = data_in[:,0,:]

                gen_vis.generateCMAP(data_in, file_idx, folder_out)
                          
        
    file_pkl_result_out = os.path.join(folder_out, 'predictions.pkl')  
    results_out = {}
          
    if (metric == True):
        
        folder_out_metrics = os.path.join(folder_out, 'metrics')
        tools.create_folder(folder_out_metrics, reset=False)
        
        print ('calculate metrics')
        list_files_gt = os.listdir(folder_gt)
     
        counter_total = 0
        counter_precision_per_file = 0
        counter_recall_per_file = 0
        counter_f1_per_file = 0
        counter_files_no_error = 0
        
        global_tp = 0
        global_fp = 0
        global_fn = 0
        
        for file_idx in list_files_gt:
            if '.txt' in file_idx:
                #counter_total = counter_total + 1
                try:
                    filepath_gt = os.path.join(folder_gt, file_idx)
                    filepath_in = os.path.join(folder_in, 'filter_'+file_idx)
                    filepath_out = os.path.join(folder_out_metrics, file_idx)
                    
                    data_gt = gen_eval.extract_data_from_txt(filepath_gt)
                    data_in = gen_eval.extract_data_from_txt(filepath_in)
                    
                    results_out[file_idx.replace('.txt','')] = data_in
                    
                    data_performances = np.array([gen_eval.extract_performances(data_gt, data_in, iou_idx) for iou_idx in threshold_iou])
                    
                    for data_performances_idx in data_performances:
                        
                        tp = data_performances_idx[0]
                        fp = data_performances_idx[1]
                        fn = data_performances_idx[2]
                        
                        global_tp = global_tp + tp
                        global_fp = global_fp + fp
                        global_fn = global_fn + fn
                    
                    precision, recall, f1 = gen_eval.calculate_metrics(tp, fp, fn)
                    gen_eval.write_metrics_file(filepath_out, tp, fp, fn, precision, recall, f1)
                    
                    counter_precision_per_file = counter_precision_per_file + precision
                    counter_recall_per_file = counter_recall_per_file + recall
                    counter_f1_per_file = counter_f1_per_file + f1
                    counter_files_no_error = counter_files_no_error + 1
                    counter_total = counter_total + 1
                    
                except:
                    print ('problem access data into file: {}'.format(filepath_in))    
        
        avg_precision, avg_recall, avg_f1 = gen_eval.calculate_metrics(global_tp, global_fp, global_fn, show=False)
            
        print ('********************************************************************************')
        print ('TOTAL_SAMPLES: {}'.format(counter_total))
        print ('GLOBAL PERFOMANCES')
        print (' -- COUNTER Files without error: {}'.format(counter_files_no_error))
            
        print (' -- global_tp: {}'.format(str(global_tp).replace('.',',')))
        print (' -- global_fp: {}'.format(str(global_fp).replace('.',',')))
        print (' -- global_fn: {}'.format(str(global_fn).replace('.',',')))
            
        print (' -- global_precision: {}'.format(str(avg_precision).replace('.',',')))
        print (' -- global_recall: {}'.format(str(avg_recall).replace('.',',')))
        print (' -- global_f1: {}'.format(str(avg_f1).replace('.',',')))
        print ('********************************************************************************')
            
        filepath_global_out = os.path.join(folder_out, 'global_results.txt')
        gen_eval.write_metrics_global(filepath_global_out, avg_precision, avg_recall, avg_f1)
             
        if (counter_total == 0):
            print ('No gt data. Remove argument --metric')
        
    else:
        list_files_gt = os.listdir(folder_gt)
        
        for file_idx in list_files_gt:
            if '.mp4' in file_idx:
                file_in = file_idx.replace('.mp4','')
                try:
                    filepath_in = os.path.join(folder_in, 'filter_'+file_in+'.txt')
                    data_in = gen_eval.extract_data_from_txt(filepath_in)
                    results_out[file_in] = data_in
                    
                except:
                    print ('problem access data into file: {}'.format(filepath_in))         
    
    # save pickle
    with open(file_pkl_result_out, 'wb') as f:
        pickle.dump(results_out, f)
            
            
    if (subs == True):
        
        print ('Generate eaf files!!!')
            
        folder_out_subs = os.path.join(folder_out, 'subs')
        tools.create_folder(folder_out_subs, reset=False)
            
        list_files_gt = os.listdir(folder_gt)
        #print (list_files_gt)
        print (len(list_files_gt))
            
        for file_idx in list_files_gt:
            if '.txt' in file_idx: 
                # print ('process: {}'.format(file_idx))
                filepath_gt = os.path.join(folder_gt, file_idx)
                filepath_in = os.path.join(folder_in, file_idx)
                filepath_filter_in = os.path.join(folder_in, 'filter_'+file_idx)
                gen_subs.generateSubtitles(filepath_filter_in, filepath_in, filepath_gt, folder_out_subs)
                

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generate Result Output')
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, default='', type=str)
    parser.add_argument('--gt', required=True, default='', type=str)
    parser.add_argument('--labels', required=True, default='', type=str)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--metric', action='store_true')
    parser.add_argument('--subs', action='store_true')
    #parser.add_argument('--threshold_iou', required=True, default=0.5, type=float)
    parser.add_argument('--threshold_iou_min', required=True, default=0.4, type=float)
    parser.add_argument('--threshold_iou_max', required=True, default=0.75, type=float)
    parser.add_argument('--threshold_iou_step', required=True, default=0.05, type=float)
    arg = parser.parse_args()
    main(arg)
    