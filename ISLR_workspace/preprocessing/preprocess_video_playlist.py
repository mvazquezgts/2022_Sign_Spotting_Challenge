import os, sys
import tqdm
import argparse
import shutil
import utils

sys.path.append('..')
from preprocessing.src.preprocess_video import PreprocessVideo as PreprocessVideo


def main(args):
       
    folder_videos = args.input
    folder_out = args.output
    resolution = args.resolution
    method = args.method
    detection_iter = args.detection_iter
    
    print ('method: {}'.format(method))
    print ('resolution: {}'.format(resolution))
    preprocessVideo = PreprocessVideo(method, resolution, detection_iter=detection_iter)
    
    print ('LIST_VIDEOS')
    list_videos = os.listdir(folder_videos)
    print (list_videos)
    print (len(list_videos))
    utils.create_folder(folder_out, reset=False)

    for video in tqdm.tqdm(list_videos):  
        if 'mp4'in video:
            input_video_path = os.path.join(folder_videos, video)
            out_video_path = os.path.join(folder_out, video)
            
            print ('input_video_path: {}'.format(input_video_path))
            if not os.path.isfile(out_video_path):
            
                try:
                    
                    temp_video_path = preprocessVideo.change_video_fps(input_video_path, 30) 
                    bbox = preprocessVideo.get_bbox_limit_from_video(temp_video_path)
                    preprocessVideo.execute_with_BBox(temp_video_path, out_video_path, bbox)
                    os.remove(temp_video_path)
                    
                except:
                    print ('exception detected')
            
            # preprocessVideo.execute(input_video_path, out_video_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, default='', type=str)
    parser.add_argument('--resolution', required=False, type=str)
    parser.add_argument('--method', required=False, type=str)
    parser.add_argument('--detection_iter', default=5, required=False, type=str)
    arg = parser.parse_args()
    main(arg)

