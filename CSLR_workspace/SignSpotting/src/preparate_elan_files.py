import pympi
import os
import argparse
import tqdm
from pathlib import Path
import tools


def main(args):
       
    folder_in = args.input
    folder_out = args.output
    file_labels = args.labels
    
    try:
        list_words = tools.get_list_words(file_labels)
    except:
        print ('NO LABELS')
        list_words = []
    
    print ('LIST Elan files')
    list_elan = os.listdir(folder_in)
    print (list_elan)
    print (len(list_elan))
    tools.create_folder(folder_out, reset=False)
    

    for elan_file in tqdm.tqdm(list_elan): 
        
        if '.eaf' in elan_file:
            
            # class_id = int(elan_file.split('_')[-2].replace('s',''))
            # print(class_id)
            
            print ('Processing: {}'.format(elan_file))
            filepath_in = os.path.join(folder_in, elan_file)
            filepath_out = os.path.join(folder_out, Path(elan_file).stem+'.txt')
            
            eafob = pympi.Elan.Eaf(filepath_in)
            ort_tier_names=list(eafob.get_tier_names())
            file_text = open(filepath_out, 'w')
            
            for annotation in eafob.get_annotation_data_for_tier(ort_tier_names[0]):
                # print (annotation)
                try:
                    label = tools.remove_asterisk(annotation[2])
                    label_idx = list_words.index(label)
                    
                    #if label_idx == class_id:
                        # print('Found: {}'.format(label))
                        
                    if len(list_words) > 0:
                        entry= '{},{},{}'.format(label_idx,annotation[0],annotation[1])
                    else:
                        entry= '{},{},{}'.format(label,annotation[0],annotation[1])
                    # print(entry)
                    file_text.write(entry+'\r\n')
                except:
                    print ('except - label: {}'.format(label))
                    print('PROBLEM!!!!')
                    pass
            file_text.close()
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, default='', type=str)
    parser.add_argument('--labels', required=False, default='', type=str)
    arg = parser.parse_args()
    main(arg)
