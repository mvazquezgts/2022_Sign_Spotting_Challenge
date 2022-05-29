import os, sys
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
       
def get_list_words(filepath_labels):
    list_words=[]
    with open(filepath_labels) as f:
        contents = f.readlines()
        for line in contents:
            list_words.append(line.strip().split(',')[0])
    return list_words


def remove_asterisk(filename):
    return filename.replace('*', '')