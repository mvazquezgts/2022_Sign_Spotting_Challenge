import os
import fnmatch
import numpy as np 
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from pathlib import Path

class Generate_Visualization:
    def __init__(self):
        print ('Generate_Visualization')

    def generateCMAP(self, data_in, filename, output_folder_path):
        data_in_shape = data_in.shape
        w = int(data_in_shape[1])
        h = int(data_in_shape[0]/4)
        max = 65500
        if ( w > max):
            w = max
        if ( h > max):
            h = max

        fig, _ = plt.subplots(figsize=(w, h))
        cmap = sns.color_palette("YlOrBr", as_cmap=True)
        sns.heatmap(data_in, vmin=0.2, vmax=2, cmap=cmap, linewidth=0.2, annot=True, fmt=".2f")
        plt.show()
        
        filename_without_extesion = Path(filename).stem
        full_img_out_file = os.path.join(output_folder_path, filename_without_extesion+'.png')
        fig.savefig(full_img_out_file)
        
    def generateCurve(self, data_in, filename, output_folder_path):
        data_in_shape = data_in.shape
        w = int(data_in_shape[1])
        h = int(data_in_shape[0]/4)
        max = 65500
        if ( w > max):
            w = max
        if ( h > max):
            h = max

        fig, _ = plt.subplots(figsize=(w, h))
        cmap = sns.color_palette("YlOrBr", as_cmap=True)
        sns.heatmap(data_in, vmin=0.2, vmax=2, cmap=cmap, linewidth=0.2, annot=True, fmt=".2f")
        plt.show()
        
        filename_without_extesion = Path(filename).stem
        full_img_out_file = os.path.join(output_folder_path, filename_without_extesion+'.png')
        fig.savefig(full_img_out_file)