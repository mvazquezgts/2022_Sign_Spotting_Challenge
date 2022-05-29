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
       
def main(args, parser):
       
    folder_in = args.input
    folder_out = args.output
    type_features = args.type_features
    labels=args.labels
    folder_models = args.folder_models
    batch_size = args.batch_size
    raw_data_boolean = args.raw
    
    print('folder_in: {}'.format(folder_in))
    print('folder_out: {}'.format(folder_out))
    print('type_features: {}'.format(type_features))
    print('folder_models: {}'.format(folder_models))
    
    list_labels = tools.get_list_words(labels)
    number_classes = len(list_labels)
    
    for type_features_idx in type_features:
        
        model_config = os.path.join(folder_models, type_features_idx+'.yaml')
        folder_in_idx = os.path.join(folder_in, type_features_idx)
        folder_out_idx = os.path.join(folder_out, type_features_idx)
        
        print ('model config: {}'.format(model_config))
           
        model = MSG3D_Processor(parser, model_config, number_classes = number_classes)
        create_folder(folder_out_idx)
        
        if (raw_data_boolean):
            folder_out_raw_idx = os.path.join(folder_out+'_RAW')
            folder_out_raw_idx_out_fc = os.path.join(folder_out_raw_idx , 'out_fc', type_features_idx)
            create_folder(folder_out_raw_idx_out_fc)
            folder_out_raw_idx_out_global_average = os.path.join(folder_out_raw_idx , 'out_global_average', type_features_idx)
            create_folder(folder_out_raw_idx_out_global_average)
            

        print ('LIST_FILES')
        list_files = os.listdir(folder_in_idx)
        print (list_files)
        print (len(list_files))
        
        for file_idx in tqdm(list_files):
            if '.npy' in file_idx:
                print ('Processing: {}'.format(file_idx))
                filepath_in = os.path.join(folder_in_idx, file_idx)
                filepath_out = os.path.join(folder_out_idx, file_idx)
                
                data_in = np.load(filepath_in)
                print ('feeder_out_pre_feed_model: {}'.format(data_in.shape))
                
                arr_output_fc = []
                arr_output_global_average = []
                arr_output_softmax = []
                number_samples = data_in.shape[0]
                
                for batch_idx in tqdm(range(int(number_samples/batch_size))):
                    start = batch_idx * batch_size
                    end = start + batch_size
                    _, _, output_softmax, output_fc, out_global_average = model.inference_softmax_with_batch_size(data_in[start:end])
                    for out_idx in range(batch_size):
                        arr_output_softmax.append(output_softmax[out_idx].cpu().detach().numpy())
                        arr_output_fc.append(output_fc[out_idx].cpu().detach().numpy())
                        arr_output_global_average.append(out_global_average[out_idx].cpu().detach().numpy())
                
                if number_samples%batch_size > 0:
                    start = int(number_samples/batch_size) * batch_size
                    end = start + number_samples%batch_size
                    _, _, output_softmax, output_fc, out_global_average = model.inference_softmax_with_batch_size(data_in[start:end])
                    for out_idx in range(number_samples%batch_size):
                        arr_output_softmax.append(output_softmax[out_idx].cpu().detach().numpy())
                        arr_output_fc.append(output_fc[out_idx].cpu().detach().numpy())
                        arr_output_global_average.append(out_global_average[out_idx].cpu().detach().numpy())
                
                np.save(filepath_out, np.array(arr_output_softmax))
        
                if (raw_data_boolean):
                    filepath_out_fc = os.path.join(folder_out_raw_idx_out_fc, file_idx)
                    np.save(filepath_out_fc, np.array(arr_output_fc))
                    
                    filepath_out_global_average = os.path.join(folder_out_raw_idx_out_global_average, file_idx)
                    np.save(filepath_out_global_average, np.array(arr_output_global_average))
        


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ISLR_output')
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, default='', type=str)
    parser.add_argument('--type_features', nargs="+", required=True)
    parser.add_argument('--labels', required=True, type=str)
    parser.add_argument('--folder_models', required=True, type=str)
    parser.add_argument('--batch_size', required=False, default=1, type=int)
    parser.add_argument('--raw', action='store_true')
    
    
    # Herencia "msg3d processor".
    parser.add_argument(
        '--model',
        default=None,
        help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--half',
        action='store_true',
        help='Use half-precision (FP16) training')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    
    arg = parser.parse_args()
    main(arg, parser)