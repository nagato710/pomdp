import pomdp_model
import csv
import numpy as np
import GPy
import pandas as pd
import matplotlib.pyplot as plt



class dataset():
    def __init__(self,data):

        self.drag = []
        self.state = []
        self.put = []

        # データを引き出した方向，抑えた位置，物体の位置を入力していく
        for i in range(2):
            self.drag.append(float(data[i]))

        for i in range(3):
            self.put.append(float(data[i + 2]))

        
        for i in range(6):
            state = []
            for j in range(4):
                state.append(float(data[i*4 + 6 + j]))
            
            self.state.append(state)
        # 状態の一意性が故にソートする
        self.state.sort(key = lambda x: x[2])

    def get_dataset(self):
        
        data_array = []

        for i in range(6):
            for j in range(4):
                data_array.append(self.state[i][j])
                
        for i in range(3):
            data_array.append(self.put[i])

        for i in range(2):
            data_array.append(self.drag[i])

        return np.array(data_array)


class test_data():
    def __init__(self,data):

        self.state = []

        # データを引き出した方向，抑えた位置，物体の位置を入力していく
        
        for i in range(6):
            state = []
            for j in range(4):
                state.append(float(data[i*4 + 29 + j]))
            
            self.state.append(state)
        # 状態の一意性が故にソートする
        self.state.sort(key = lambda x: x[2])
    
    def get_testdata(self):
        
        data_array = []

        for i in range(6):
            for j in range(4):
                data_array.append(self.state[i][j])

        return np.array(data_array)  

def change_state(Xtest,data_type = 1):

    Target = []
    Obj = []

    print("WEEEEEEEEEEEEEEE")
    
    if data_type == 1:
        for i in range(4):
            print(Xtest[i+5])
            Target.append(Xtest[i + 5])
        Target.append((8,15))
        
        for i in range(5):
            Obj_2 = []
            for j in range(4):
                Obj_2.append(Xtest[i*4 + j + 9])
            Obj_2.append((8,14))
            Obj.append(Obj_2)
    
    if data_type == 2:
        for i in range(4):
            Target.append(Xtest[i])
        Target.append((8,15))
        
        for i in range(5):
            Obj_2 = []
            for j in range(4):
                Obj_2.append(Xtest[i*4 + j + 4])
            Obj_2.append((8,14))
            Obj.append(Obj_2)

    return pomdp_model.State(Target,Obj,5)  

if __name__ == '__main__':

    # kernel = GPy.kern.Matern52(29, ARD=True)
    kernel = GPy.kern.RBF(29,useGPU=True)
    f = open('incrementdata.csv','r')
    reader = csv.reader(f)
    Xdata = []
    Ydata = []

    Xtest = []
    Ytest = []
    count = 0

    for row in reader:

        if count >= 8002 and count < 8003:
            data = dataset(row)
            result = test_data(row)

            Xtest.append(data.get_dataset())
            Ytest.append(result.get_testdata())

        if count >= 8003:
            break

        count += 1

    model2 = GPy.core.model.Model.load_model("/home/nagato/gaussian_process/model.zip")
    # model.load_model("/home/nagato/gaussian_process/model.zip",data = True)
    x_pred = np.array(Xtest)
    print(model2)
    # print(model.save_model("model"))
    # y_qua_pred = model2
    y_qua_pred = model2.predict_quantiles(x_pred, quantiles=(2.5, 97.5))[0]

    s1 = change_state(Xtest,1)
    s2_correct = change_state(Ytest,2)
    s2_predict = change_state(y_qua_pred,2)

    s1.visualization()
    s2_correct.visualization()
    s2_predict.visualization()

    f.close()