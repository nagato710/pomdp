'''
Format data list
'''

import numpy as np

scale_tar = 1
scale_ob = 1 
class dataset():
    def __init__(self,data):

        self.drag = []
        self.state_index = []
        self.put = []
        self.state = []
        self.target = []
        self.rotate_list = []

        for i in range(2):
            self.drag.append(float(data[i])*10)


        for i in range(3):
            self.put.append(float(data[i + 2])/scale_ob)

        for i in range(4):
            if i != 3:
                self.target.append(float(data[i + 5])/scale_tar)
            else:
                self.rotate_list.append(float(data[i + 5])/scale_tar)

           

        self.state.append(self.target)
        
        for i in range(5):
            state = []
            state.append(i)
            for j in range(4):
                    state.append(float(data[i*4 + 9 + j])/scale_ob)
            
            self.state_index.append(state)
        # 状態の一意性が故にソートする
        self.state_index.sort(key = lambda x: x[3])

        for i in range(5):
            state = []
            for j in range(5):
                if j >= 1 and j <= 3:
                    state.append(self.state_index[i][j])
                if j == 4:
                    self.rotate_list.append(self.state_index[i][j])
                
            self.state.append(state)


    def get_dataset(self):
        
        data_array_putting = []
        data_array_position = []

        for i in range(3):
            data_array_putting .append(self.put[i])

        for i in range(6):
            for j in range(3):
                data_array_position.append(self.state[i][j])

        return np.array(data_array_putting+data_array_position)
        #return np.array(data_array_putting), np.array(data_array_position)

class test_data():
    def __init__(self,data,input):

        self.state = []
        target = []

        for i in range(3):
            target.append(float(data[29 + i])/scale_tar)
        
        self.state.append(target)
            
        # 入力データに合わせてソートする
        for i in range(5):
            state = []
            for j in range(3):
                state.append(float(data[input.state_index[i][0]*4 + 33 + j])/scale_ob)
            
            self.state.append(state)
     
    def get_testdata(self):
        
        data_array = []

        for i in range(6):
            for j in range(3):
                data_array.append(self.state[i][j])

        return np.array(data_array)    