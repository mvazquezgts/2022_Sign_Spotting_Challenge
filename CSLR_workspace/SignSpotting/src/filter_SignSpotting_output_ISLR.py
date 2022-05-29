import os, sys
import tqdm
import argparse
import shutil
import numpy as np
from tqdm import tqdm
import tools

from internal.feeder import Feeder
sys.path.insert(0, '../../../ISLR_workspace/msg3d')
from msg3d_processor import MSG3D_Processor
         
def main(args, parser):
    input_folder_results = args.input_results
    input_folder_features = args.input_features
    # output_folder_results = args.output_results
    type_features = args.type_features
    folder_models = args.folder_models
    threshold = args.threshold
    labels = args.labels
    result_size_fixed = args.result_size_fixed
    use_context = args.use_context
    
    list_labels = tools.get_list_words(labels)
    number_classes = len(list_labels)
    
    print('input_folder_results: {}'.format(input_folder_results))
    print('input_folder_features: {}'.format(input_folder_features))
    # print('output_folder_results: {}'.format(output_folder_results))
    print('type_features: {}'.format(type_features))
    print('number_classes: {}'.format(number_classes))
    print('folder_models: {}'.format(folder_models))
    print('threshold: {}'.format(threshold))
    print('labels: {}'.format(labels))
    print('result_size_fixed: {}'.format(result_size_fixed))
    print('use_context: {}'.format(use_context))
    
    list_models = []
    list_folders_out = []
    list_folders_out_fc = []
    list_folders_out_global_average = []
    
    for type_features_idx in type_features:
        list_models.append(MSG3D_Processor(parser, os.path.join(folder_models, type_features_idx+'.yaml'), number_classes=number_classes, result_size_fixed= result_size_fixed))
        # list_models.append(EfficientGCN_Processor(parser, os.path.join(folder_models, type_features_idx+'.yaml')))
        
        filepath_out = os.path.join(input_folder_results,type_features_idx)
        list_folders_out.append(filepath_out)
        tools.create_folder(filepath_out)
        
        filepath_out_fc = os.path.join(input_folder_results+'_RAW', 'out_fc' ,type_features_idx)
        list_folders_out_fc.append(filepath_out_fc)
        tools.create_folder(filepath_out_fc)
        
        filepath_out_global_average = os.path.join(input_folder_results+'_RAW','out_global_average', type_features_idx)
        list_folders_out_global_average.append(filepath_out_global_average)
        tools.create_folder(filepath_out_global_average)
        
        
    list_files = os.listdir(input_folder_results)
    for file_idx in list_files:
        if '.npy' in file_idx and 'raw' not in file_idx:
            print (file_idx)
            filepath_npy = os.path.join(input_folder_results, file_idx)
            print('filepath_npy: {}'.format(filepath_npy))
            data_results_prefilter = np.load(filepath_npy)
            
            list_data_in = []
            for type_features_idx in type_features:
                filepath_features_npy = os.path.join(input_folder_features, type_features_idx, file_idx)
                list_data_in.append(np.load(filepath_features_npy))
            
            filter_boolean=[]
            filename_text_results = file_idx.replace('.npy','.txt')
            filename_text_filter = 'filter_{}'.format(filename_text_results)
            
            filepath_text_results = os.path.join(input_folder_results, filename_text_results)
            filepath_text_filter = os.path.join(input_folder_results, filename_text_filter)
            
            
            arr_save_output_softmax = np.zeros((len(type_features), len(data_results_prefilter), number_classes))
            arr_save_output_fc = np.zeros((len(type_features), len(data_results_prefilter), number_classes))
            arr_save_output_global_average = np.zeros((len(type_features), len(data_results_prefilter), 384))
            
            for data_idx, data_result_idx in enumerate(data_results_prefilter):
                if (data_result_idx[2] - data_result_idx[1] > 5):
                    
                    arr_output_softmax = []
                    for feature_idx, type_features_idx in enumerate(type_features):
                        
                        data_feed_in = list_data_in[feature_idx][:,data_result_idx[1]:data_result_idx[2],:,:]  
                        if use_context:                                                                    # get context - get output_frames_fixed frames, sign in the middle.
                            duration_result = data_result_idx[2] - data_result_idx[1]
                            if duration_result < result_size_fixed:
                                center = int(data_result_idx[1] + duration_result / 2)
                                start = int(center - result_size_fixed/2)
                                if (start < 0):
                                    start = 0
                                end = start + result_size_fixed
                                data_feed_in = list_data_in[feature_idx][:,start:end,:,:] 
                        
                        index, prob, output_softmax, output_fc, out_global_average = list_models[feature_idx].inference_softmax2(data_feed_in)
                        arr_output_softmax.append(output_softmax)
                        
                        arr_save_output_softmax[feature_idx][data_idx] = output_softmax.cpu().detach().numpy()
                        arr_save_output_fc[feature_idx][data_idx] = output_fc.cpu().detach().numpy()
                        arr_save_output_global_average[feature_idx][data_idx] = out_global_average.cpu().detach().numpy()
                    
                    if (len(arr_output_softmax)>1):   # JOINST & BONES - COMBINE
                        combi = (arr_output_softmax[0] + arr_output_softmax[1]) / 2
                    else:  # ONLY ONE INPUT
                        combi = arr_output_softmax[0]
                       
                    prob, index = combi.topk(5)
                    #print ('class_id: {} - class prob: {} '.format(index, prob, prob))
                    prob = prob.cpu().numpy()[0][0]
                    index = index.cpu().numpy()[0][0]
                    
                    if prob > threshold:
                        filter_boolean.append(index)
                    else:
                        filter_boolean.append(-1)
                else:
                    filter_boolean.append(-1)
                    
            print ("FILTER_BOOLEAN: {}".format(filter_boolean))
            
            
            # save files
            for idx in range(len(type_features)):
                filepath_out = os.path.join(list_folders_out[idx],file_idx)
                np.save(filepath_out, np.array(arr_save_output_softmax[idx]))
                
                filepath_out_fc = os.path.join(list_folders_out_fc[idx],file_idx)
                np.save(filepath_out_fc, np.array(arr_save_output_fc[idx]))
                
                filepath_out_global_average = os.path.join(list_folders_out_global_average[idx],file_idx)
                np.save(filepath_out_global_average, np.array(arr_save_output_global_average[idx]))
                
            
            
            with open(filepath_text_filter,'w') as f:
                file_outputs = open(filepath_text_results, 'r')
                for idx, line in enumerate(file_outputs.readlines()):
                    if (filter_boolean[idx] > -1):
                        lines_strip = line.split(',')
                        new_line = '{},{},{}'.format(filter_boolean[idx], lines_strip[1], lines_strip[2])
                        f.write(new_line)
                
                file_outputs.close()
                f.close()


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='ISLR_output')
    parser.add_argument('--input_results', required=True, type=str)
    parser.add_argument('--input_features', required=True, type=str)
    #parser.add_argument('--output_results', required=True, type=str)
    parser.add_argument('--type_features', nargs="+", required=True)
    parser.add_argument('--labels', required=True, type=str)
    parser.add_argument('--folder_models', required=True, type=str)
    parser.add_argument('--threshold', required=True, default=0.5, type=float)
    parser.add_argument('--result_size_fixed', required=True, default=25, type=int)  # default frames into filter 25 frames = 1 sec.
    parser.add_argument('--use_context', action='store_true')
    
    # Herencia from msg3d processor.
    parser.add_argument(
        '--model',
        default=None,
        required=False,
        help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        required=False,
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        required=False,
        help='the weights for network initialization')
    parser.add_argument(
        '--half',
        action='store_true',
        required=False,
        help='Use half-precision (FP16) training')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        required=False,
        help='the indexes of GPUs for training or testing')
    
    arg = parser.parse_args()
    main(arg, parser)
