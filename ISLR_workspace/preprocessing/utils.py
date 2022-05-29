import os
import datetime
import pympi
import numpy as np
import shutil

    
def create_folder(folder, reset=False):
    print ('create_folder: {}'.format(folder))    
    try:
        os.makedirs(folder)    
        print("Directory " , folder ,  " Created ")
    except FileExistsError:
        print("Directory " , folder ,  " already exists")
        
        if reset==True:
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder) 
            print("Directory " , folder ,  " reset")
        
def parser_file(filepath):
    print ('parser_file')
    removes = []
    with open(filepath, errors="ignore") as f:
        contents = f.readlines()
        try:
            for line in contents:
                if not '##' in line:
                    removes.append(line.strip())
        except:
            pass
    return removes  


def parser_file_into_dict(filepath):
    print ('parser_corrections_file')
    corrections_dict = {}
    with open(filepath) as f:
        contents = f.readlines()
        try:
            for line in contents:
                if not '##' in line:
                    elements = line.split(',')
                    corrections_dict[elements[0]] = elements[1].strip()
        except:
            pass
    return corrections_dict 


def format_milliseconds_HH_MM_SS_MS(milliseconds):
    return datetime.timedelta(milliseconds=int(milliseconds))


def mv_file(path_file_in, path_file_out):
    cmd = 'mv {} {}'.format(path_file_in, path_file_out)
    print ('CMD: {}'.format(cmd))
    os.system(cmd)


def segmentation_video(filepath_in, filepath_out, start, end, idx):
    
    video_path='{}.mp4'.format(filepath_in)
    out_video_path='{}_{}.mp4'.format(filepath_out, idx)
    
    start = start * 1e-3
    end = end * 1e-3
    dur = end - start
    
    if dur>0:
        cmd = 'ffmpeg -i {} -ss {} -t {} {} -loglevel quiet'.format(video_path,  start, dur, out_video_path)
    else:
        cmd = 'ffmpeg -i {} -ss {} {} -loglevel quiet'.format(video_path,  start, out_video_path)
    print (cmd)
    
    os.system(cmd)
    
def segmentation_eaf(filepath_in, filepath_out, start, end, idx):
    
    eaf_path='{}.eaf'.format(filepath_in)
    out_eaf_path='{}_{}.eaf'.format(filepath_out, idx)
    
    eafob = pympi.Elan.Eaf(eaf_path)
    eafob_new = pympi.Elan.Eaf(eaf_path)
    eafob_new.remove_linked_files(mimetype='video/mp4')
    
    relpath = './{}_{}.mp4'.format(os.path.basename(filepath_in), idx)
    eafob_new.add_linked_file('{}.mp4'.format(relpath),relpath=relpath, mimetype='video/mp4')
    
    ort_tier_names=list(eafob.get_tier_names())
    
    for tier_idx in ort_tier_names:

        eafob_new.remove_tier(tier_idx,clean=True)
        eafob_new.add_tier(tier_idx)
        
        for annotation in eafob.get_annotation_data_for_tier(tier_idx):
            
            if (annotation[0]>start):
                if (annotation[0]<end or end<0):
                    eafob_new.add_annotation(tier_idx, annotation[0] - start, annotation[1] - start, value=annotation[2])
            
    eafob_new.to_file(out_eaf_path)

def get_annotations_time_range(filepath_in, start, end):
    eaf_path='{}.eaf'.format(filepath_in)
    eafob = pympi.Elan.Eaf(eaf_path)
    
    intervals = []
    
    for annotation in eafob.get_annotation_data_between_times(list(eafob.get_tier_names())[0], start, end):
        entry = np.array([annotation[2],int(annotation[0]), int(annotation[1])])
        intervals.append(entry)
    
    return np.asarray(intervals)



def get_all_annotations(filepath_in, list_labels = None):
    eaf_path='{}.eaf'.format(filepath_in)
    eafob = pympi.Elan.Eaf(eaf_path)
    
    intervals = []
    
    for annotation in eafob.get_annotation_data_for_tier(list(eafob.get_tier_names())[0]):
        label = annotation[2].replace('*','')
        if list_labels == None:
            entry = np.array([label,int(annotation[0]), int(annotation[1])])
        else:
            entry = np.array([list_labels.index(label),int(annotation[0]), int(annotation[1])])
            
        intervals.append(entry)
    
    return np.asarray(intervals).astype(np.int32)