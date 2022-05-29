import os
import numpy as np 

class Generate_Output:
    def __init__(self,threshold, min_margin_is_sign, min_margin_is_same, fps, windows_size, window_stride, type_features, weigths_in):
        self.threshold = threshold
        self.min_margin_is_sign = min_margin_is_sign
        self.min_margin_is_same = min_margin_is_same
        self.fps = fps
        self.ms = 1000/self.fps
        self.windows_size = windows_size
        self.window_stride = window_stride
        self.type_features = type_features
        self.weigths_in = weigths_in
        print ('generate_output')
        
    
    # --threshold 0.6 --min_margin_is_sign 5 --min_margin_is_same 5 --fps 25 --windows_size 30 --window_stride 2
    
    def __str__ (self):
        str_out = 'Generate_Output - Setting. \r\n \
            threshold: {} \r\n \
            min_margin_is_sign: {} \r\n \
            min_margin_is_same: {} \r\n \
            fps: {} \r\n \
            windows_size: {} \r\n \
            window_stride: {}\r\n ---------------------------------------------------------- \r\n type_features :{}  \r\n weigths_in :{}  \r\n '.format(self.threshold, self.min_margin_is_sign, self.min_margin_is_same, self.fps, self.windows_size, self.window_stride, self.type_features, self.weigths_in)
        return str_out


    def combineOutput(self, list_data_in):
        # print (len(list_data_in))
        # print (list_data_in[0].shape)
        result = np.zeros(list_data_in[0].shape)
        for arr_idx in list_data_in:
            result = result + arr_idx
        result = result / len(list_data_in)
        
        return result
        
        
        
    def generateOutput(self, data_in, filepath_output):
        arr_output = []
        with open(filepath_output,'w') as f:
            for idx in range (data_in.shape[2]):
                arr = self.class_id_output(data_in, idx)
                for arr_idx in arr:
                    start = int((arr_idx[0] * self.ms * self.window_stride))
                    # end = int((arr_idx[1] * self.ms * self.window_stride ) + (self.windows_size * self.ms))
                    end = int((arr_idx[1] * self.ms * self.window_stride ) + ((self.windows_size/2) * self.ms))
                    line = '{},{},{}'.format(idx, start,end)
                    print (line)
                    f.write(line+'\n')
                    
                    entry = np.array([idx, arr_idx[0], (arr_idx[1] + self.windows_size)])
                    arr_output.append(entry)
            f.close()
        return arr_output

    def class_id_output (self, data, class_id):
        data_idx = data[:,0,class_id]
        data_indices_possitive = data_idx > self.threshold
        data_indices_possitive = self.remove_blank_space(data_indices_possitive)
        possitive_intervals = self.get_intervals_possitives(data_indices_possitive)

        return possitive_intervals
    
    def remove_blank_space(self, data):
        # print ('remove_blank_space')
        for idx in range(len(data)-1-self.min_margin_is_same):
            if (data[idx] == True and data[idx+1] == False):
                state = False
                for idx2 in range(self.min_margin_is_same):
                    if data[idx+2+idx2]==True:
                        state = True
                data[idx+1] = state
        return data

    def get_intervals_possitives(self, data):
        arr_intervals = []
        start = -1
        end = -1
        for idx in range(len(data)):
            if (data[idx]==True and start==-1):
                start = idx
            if (start>-1):
                if (data[idx]==True):
                    end = idx
                else:
                    if ((end-start)>self.min_margin_is_sign):
                        entry=[]
                        entry.append(start)
                        entry.append(end) 
                        arr_intervals.append(entry)
                    start = -1
                    end = -1
        return arr_intervals