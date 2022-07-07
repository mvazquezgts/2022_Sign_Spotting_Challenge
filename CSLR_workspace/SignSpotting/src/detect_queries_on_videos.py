'''
The query detection system uses the Python package libfmp https://github.com/meinardmueller/libfmp
The code is based on some of the FMP notebooks offer a collection of educational material closely following the textbook Fundamentals of Music Processing (FMP) [Muller, FMP, Springer 2021].
Meinard Muller and Frank Zalkow. libfmp: A Python Package for Fundamentals of Music Processing. Journal of Open Source Software (JOSS), 6(63), 2021.
Meinard Muller. Fundamentals of Music Processing  Using Python and Jupyter Notebooks. Springer Verlag, 2nd edition, 2021.
'''

import sys
import os
import glob
import argparse
import numpy as np
import scipy.spatial
from numba import jit
import pandas as pd
import pickle
from tqdm import tqdm

sys.path.append('..')

import libfmp.c3
import libfmp.c7

sys.path.insert(0, '.')

## Joints
FOLDER_Joints_DATA_SET='out_global_average/joints_C3_xyz'
#Bones
FOLDER_Bones_DATA_SET='out_global_average/bones_C3_xyz'

ms_per_frame=40


def create_folder(folder):
    print ('create_folder: {}'.format(folder))    
    try:
        os.makedirs(folder)    
        print("Directory " , folder ,  " created ")
    except FileExistsError:
        print("Directory " , folder ,  " already exists")

def check_matrices(M1, M2, label='Matrices'):
    if (M1.shape != M2.shape):
        print(label, 'have different shape!', (M1.shape, M2.shape))
    elif not np.allclose(M1, M2):
        print(label, 'are numerical different!')
    else:
        print(label, 'are equal.')


def check_if_folder_is_empty(folder):
    if os.path.isdir(folder) and os.path.exists(folder):
        if len(os.listdir(folder)) == 0:
            print('Directory {} is empty'.format(folder))
            return True
        else:    
            #print('Directory {} is not empty'.format(folder))
            return False
    else:
        print('Directory {} does not exist'.format(folder))
        return True

def compute_matching_function_dtw(X, Y, stepsize=2):
    """
    based on Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        X (np.ndarray): Query feature sequence (given as K x N matrix)
        Y (np.ndarray): Database feature sequence (given as K x M matrix)
        stepsize (int): Parameter for step size condition (1 or 2) (Default value = 2)

    Returns:
        Delta (np.ndarray): DTW-based matching function
        C (np.ndarray): Cost matrix
        D (np.ndarray): Accumulated cost matrix
    """
    C = libfmp.c3.compute_cost_matrix(X, Y, 'cosine') #Cost metric, a valid strings for scipy.spatial.distance.cdist
    if stepsize == 1:
        #Given the cost matrix, compute the accumulated cost matrix for
        #subsequence dynamic time warping with step sizes {(1, 0), (0, 1), (1, 1)}"
        D = libfmp.c7.compute_accumulated_cost_matrix_subsequence_dtw(C)
    if stepsize == 2:
        #Given the cost matrix, compute the accumulated cost matrix for
        #subsequence dynamic time warping with step sizes {(1, 1), (2, 1), (1, 2)}"
        D = libfmp.c7.compute_accumulated_cost_matrix_subsequence_dtw_21(C)
    N, M = C.shape
    Delta = D[-1, :] / N
    return Delta, C, D


def matches_dtw(pos, D, stepsize=2):
    """Derives matches from positions for DTW-based strategy

    Based on Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        pos (np.ndarray): End positions of matches
        D (np.ndarray): Accumulated cost matrix
        stepsize (int): Parameter for step size condition (1 or 2) (Default value = 2)

    Returns:
        matches (np.ndarray): Array containing matches (start, end)
    """
    matches = np.zeros((len(pos), 2)).astype(int)
    for k in range(len(pos)):
        t = pos[k]
        matches[k, 1] = t
        if stepsize == 1:
            P = libfmp.c7.compute_optimal_warping_path_subsequence_dtw(D, m=t)
        if stepsize == 2:
            P = libfmp.c7.compute_optimal_warping_path_subsequence_dtw_21(D, m=t)
        s = P[0, 1]
        matches[k, 0] = s
    return matches



def compute_matching_function_DTW_subsequence(X, Y, stepsize=2, tau=0.2, num=5):
    #X query: numpy ndarray
    #Y video: numpy ndarray
    N, M = X.shape[1], Y.shape[1]
    Delta, C, D = compute_matching_function_dtw(X, Y, stepsize=stepsize)
    pos = libfmp.c7.mininma_from_matching_function(Delta, rho=2*N//3, tau=tau, num=num)
    matches = matches_dtw(pos, D, stepsize=stepsize)
    return C, Delta, pos, D, matches


def main(args):
    
    folder_in_videos = args.input_videos
    folder_in_queries = args.input_queries
    folder_out = args.output
    file_query_limits = args.file_query_limits
    fps = args.fps
    type_features = args.type_features
    
    print('folder_in_videos: {}'.format(folder_in_videos))
    print('folder_in_queries: {}'.format(folder_in_queries))
    print('folder_out: {}'.format(folder_out))
    print('file_query_limits: {}'.format(file_query_limits))
    print('fps: {}'.format(fps))
    print('type_features: {}'.format(type_features))

    create_folder(folder_out)

    filepath_out = os.path.join(folder_out, 'predictions.pkl')

    query_limits=pd.read_table(file_query_limits, sep=';')

    data_dir_query_joints = os.path.join(folder_in_queries, FOLDER_Joints_DATA_SET)
    data_dir_query_bones = os.path.join(folder_in_queries, FOLDER_Bones_DATA_SET)
    data_dir_video_joints = os.path.join(folder_in_videos, FOLDER_Joints_DATA_SET)
    data_dir_video_bones = os.path.join(folder_in_videos, FOLDER_Bones_DATA_SET)

    if check_if_folder_is_empty(data_dir_query_joints):
        print('execute main_step1_track2.py')
        sys.exit()
    if check_if_folder_is_empty(data_dir_query_bones):
        print('execute main_step1_track2.py')
        sys.exit()
    if check_if_folder_is_empty(data_dir_video_joints):
        print('execute main_step1_track2.py')
        sys.exit()
    if check_if_folder_is_empty(data_dir_video_bones):
        print('execute main_step1_track2.py')
        sys.exit()


    print ('LIST_QUERY_FILES')
    list_files = os.listdir(os.path.join(folder_in_queries, FOLDER_Joints_DATA_SET))
    print (list_files)
    print (len(list_files))
    
    outdata=[]
    predictions={}
    idclass_offset=query_limits['CLASS'][0] 
    for fquery in os.listdir(data_dir_query_joints):
        if fquery.endswith('.npy'):
            fnQ = os.path.join(data_dir_query_joints, fquery)
            query = os.path.splitext(fquery)[0].split('_')[1]
            query_class=int(query[1:])
            # Crop query frames
            X1 = np.load(fnQ) # query
            Xj = np.squeeze(X1).T
            ini_frame=round(query_limits['BEGIN'][query_class-idclass_offset]/ms_per_frame)
            end_frame=round(query_limits['END'][query_class-idclass_offset]/ms_per_frame)
            Xj=Xj[:,ini_frame:end_frame]
            # rows-> feature values, cols-> frames
            [rowsq,cols]=Xj.shape
            N = Xj.shape[1]
            fnQ = os.path.join(data_dir_query_bones, fquery)
            # Crop query frames 
            X1 = np.load(fnQ) # query
            Xb = np.squeeze(X1).T
            Xb=Xb[:,ini_frame:end_frame]
            X=np.concatenate((Xj,Xb))
            # rows-> feature values, cols-> frames
            for fnV in glob.iglob(data_dir_video_joints+'/*'+query+'*.npy', recursive=True):
                fvideo = os.path.basename(fnV)
                Y1 = np.load(fnV) # Video to search in
                Yj = np.squeeze(Y1).T
                M = Yj.shape[1]
                fnVb = os.path.join(data_dir_video_bones, fvideo)
                Y1 = np.load(fnVb) # Video to search in
                Yb = np.squeeze(Y1).T
                Y=np.concatenate((Yj,Yb))
                #print('=== Query X: ', query, '; Video Y:', os.path.basename(fnV),' ===')
                C, Delta, pos, D, matches = compute_matching_function_DTW_subsequence(X, Y, stepsize=2,num=1)
                for (s, t) in matches:
                    outdata.append([s,t,Delta[s],Delta[t]])
                if matches.shape[0] > 0:
                    predictions[os.path.splitext(fvideo)[0]]=[[query_class, matches[0][0]*ms_per_frame, matches[0][1]*ms_per_frame]]
                else:
                    predictions[os.path.splitext(fvideo)[0]]=[]

    with open(filepath_out, 'wb') as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ISLR_output')
    parser.add_argument('--input_videos', required=True, type=str)
    parser.add_argument('--input_queries', required=True, type=str)
    parser.add_argument('--output', required=True, default='', type=str)
    parser.add_argument('--file_query_limits', required=True, default='', type=str)
    parser.add_argument('--fps', required=False, default=25, type=int)
    parser.add_argument("--type_features", nargs="+", required=True)
    
    arg = parser.parse_args()
    main(arg)

