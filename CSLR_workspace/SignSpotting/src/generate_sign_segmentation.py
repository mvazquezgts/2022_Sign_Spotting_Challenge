import os
import argparse
import tqdm
import tools

def main(args):
       
    folder_in = args.input
    folder_out = args.output
    path_segmenter = args.path_segmenter
    
    print ('LIST videos to process')
    list_files = os.listdir(folder_in)
    print (list_files)
    print (len(list_files))
    tools.create_folder(folder_out, reset=False)
        
    os.chdir(path_segmenter)
    
    for file_idx in tqdm.tqdm(list_files): 
        if 'mp4' in file_idx:
            
            path_file_video = os.path.join(folder_in, file_idx)
            cmd = 'python demo/demo.py --video_path {} --save_segments --save_path {}'.format(path_file_video, folder_out)
            print (cmd)
            os.system(cmd)
            
            cmd = 'mv {}/predictions.pkl {}/{}_predictions.pkl'.format(folder_out, folder_out, file_idx.replace('.mp4', ''))
            print (cmd)
            os.system(cmd)
            
            cmd = 'rm {}/probabilities.pkl'.format(folder_out)
            print (cmd)
            os.system(cmd)
            
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, default='', type=str)
    parser.add_argument('--path_segmenter', required=True, type=str)
    arg = parser.parse_args()
    main(arg)
