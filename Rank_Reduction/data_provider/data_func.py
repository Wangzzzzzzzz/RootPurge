import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List
import matplotlib.pyplot as plt





class Dataset_Function_MC_Core:
    def __init__(self, functions: List, random_generator = None, x_start = 0, x_end = 100, x_space = 0.01, flag = "train", scale=True, size=None):
        


        if size == None:
            self.seq_len = 96
            self.label_len = 0
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.scale = scale
        self.f = functions
        self.x_start = x_start
        self.x_end = x_end
        self.x_space = x_space
        self.total_len = len(np.arange(start=x_start, stop=x_end, step=x_space))
        self.random_generator = random_generator

        self.__get_data()

    def __get_data(self):
        self.scaler = StandardScaler()
        x = np.arange(start=self.x_start, stop = self.x_end, step=self.x_space)

        data_raw= []
        for i in range(len(self.f)):
            if self.random_generator is None:
                random_term = 0
            else:
                random_term = self.random_generator(len(x))
            data_raw.append(self.f[i](x)+random_term)
        data_raw = np.array(data_raw).T
        
        b1s = [0, int(self.total_len * 0.5), int(self.total_len * 0.5) + int(self.total_len * 0.25)]
        b2s = [int(self.total_len * 0.5), int(self.total_len * 0.5) + int(self.total_len * 0.25), self.total_len]

        if self.scale:
            train_data = data_raw[b1s[0]:b2s[1]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(data_raw)
        else:
            data = data_raw

        self.y_all = data
        self.x_all = x
        self.b1s, self.b2s = b1s, b2s

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def plot(self):
        y_ = self.y_all.T
        x_ = self.x_all
        # Create a plot
        for idx, y in enumerate(y_):
            plt.figure(figsize=(10, 6))  # You can adjust the figure size as needed
            
            b1_train, b2_train = self.b1s[0], self.b2s[0]
            b1_val, b2_val = self.b1s[1], self.b2s[1]
            b1_test, b2_test = self.b1s[2], self.b2s[2]
            
            plt.plot(x_[b1_train:b2_train], y[b1_train:b2_train], label='train')
            plt.plot(x_[b1_val:b2_val], y[b1_val:b2_val], label='val')
            plt.plot(x_[b1_test:b2_test], y[b1_test:b2_test], label='test')

            # Add titles and labels
            plt.title('Plot of the Function')
            plt.xlabel('x')
            plt.ylabel('f(x)')
        
            # Add a legend
            plt.legend()
            
            # Show grid
            plt.grid(True)


class Dataset_Function_MC(Dataset):
    def __init__(self, data_core:Dataset_Function_MC_Core, flag: str):
        flag = flag.lower()
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}

        self.set_type = type_map[flag]
        self.data_core = data_core
        border1 = self.data_core.b1s[self.set_type]
        border2 = self.data_core.b2s[self.set_type]
        
        self.data_x = self.data_core.y_all[border1:border2]
        self.data_y = self.data_core.y_all[border1:border2]


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.data_core.seq_len
        r_begin = s_end - self.data_core.label_len
        r_end = r_begin + self.data_core.label_len + self.data_core.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.data_core.seq_len - self.data_core.pred_len + 1