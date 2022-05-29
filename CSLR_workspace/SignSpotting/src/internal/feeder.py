from enum import Enum
import os
import numpy as np

# partitioner_config= {
#     'window_size' : 40,
#     'window_stride' : 5,
# }
    
class Feeder:
    def __init__(self, type_features, window_size = 40, window_stride = 5):
        print ('feeder')
        self.type_features = type_features
        
        self.window_size = window_size
        self.window_stride = window_stride

        print ("*********************************************************************")
        print ('Init Feeder')
        print ('type_features: {}'.format(self.type_features))
        print ('window_size: {}'.format(self.window_size))
        print ('window_stride: {}'.format(self.window_stride))
        print ("*********************************************************************")


    def __len__(self):
         return len(self.list_files)
     
    def __str__ (self):
        str_out = 'Feeder - Setting & Window Config. \r\n Type_features: {} \r\n Size: {} & Stride: {}'.format(self.type_features, self.window_size, self.window_stride)
        return str_out

    def getItems(self, filepath):
        data = np.load(filepath)
        print ('Data input: {}'.format(data.shape))
        items = self.get_slicing_windows(data, self.window_size, self.window_stride)
        print ('Slicing Windods: {}'.format(items.shape))
        return items

    def get_slicing_windows(self, data, L, S ):  # Window len = L, Stride len/stepsize = S
        
        # INPUT_SIZE = 20
        C, T, K, M = data.shape
        
        rest = T - L
        n_window = 0
        if (rest > 0):
            n_window = int( rest / S)

        result = np.zeros ([n_window + 1, C, L, K, M])
        # assert L <= INPUT_SIZE , "Window size incorrect, is more large than {}".format(INPUT_SIZE)

        result[0, :, 0:L, :, :] = data[:, 0:L, :, :]
        for i in range (n_window):
            start = S + (S * i)
            end = L + S + (S * i)
            result[1 +  i, :, 0:L, :, :] = data[:, start:end, :, :]

        return result