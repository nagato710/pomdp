import pomdp_model
import csv
import numpy as np
# import GPy
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math




# 学習済みのモデルに対して，取得したデータを入力してどのような遷移が起こっているかを確認するためのプログラム
datanum = 100

class dataset():
    def __init__(self,data):

        self.drag = []
        self.state_index = []
        self.put = []
        self.state = []
        self.target = []

        # データを引き出した方向，抑えた位置，物体の位置を入力していく
        # for i in range(2):
        #     self.drag.append(float(data[i]))

        for i in range(3):
            self.put.append(float(data[i + 2]))

        for i in range(4):
            if i != 3:
                self.target.append(float(data[i + 5]))
            if i == 3:
                self.target.append(float(data[i + 5]))

        self.state.append(self.target)
        
        for i in range(5):
            state = []
            state.append(i)
            for j in range(4):
                if j != 3:
                    state.append(float(data[i*4 + 9 + j]))
                if j == 3:
                    state.append(float(data[i*4 + 9 + j]))
            
            self.state_index.append(state)
        # 状態の一意性が故にソートする
        self.state_index.sort(key = lambda x: x[3])

        for i in range(5):
            state = []
            for j in range(5):
                if j >= 1:
                    state.append(self.state_index[i][j])
            self.state.append(state)


    def get_dataset(self):
        
        data_array = []

        # for i in range(2):
        #     data_array.append(self.drag[i])

        for i in range(3):
            data_array.append(self.put[i])

        for i in range(6):
            for j in range(4):
                data_array.append(self.state[i][j])

        return np.array(data_array)



class test_data():
    def __init__(self,data,input):

        self.state = []
        target = []

        for i in range(4):
            if i != 3:
                target.append(float(data[29 + i]))
            if i == 3:
                target.append(float(data[29 + i]))
        
        self.state.append(target)
            
        # 入力データに合わせてソートする
        for i in range(5):
            state = []
            for j in range(4):
                if j != 3:
                    state.append(float(data[input.state_index[i][0]*4 + 33 + j]))
                if j == 3:
                    state.append(float(data[input.state_index[i][0]*4 + 33 + j]))
            
            self.state.append(state)
     
    def get_testdata(self):
        
        data_array = []

        for i in range(6):
            for j in range(4):
                data_array.append(self.state[i][j])

        return np.array(data_array)    


#　arrayで入力されたデータを状態に合う形に変形する

def change_state(Xtest,rotate_list = None,data_type = 1,index = 0):
    # data_type:csvファイルには一列に状態とアクション，次状態を記録してしまっている．
    # そのため　data_type = 1 では状態とアクションを出力する
    #           data_type = 2 では次状態のみを出力とする 
    # index :何個目の入力データか表示している

    Target = []
    Obj = []
    drag = []
    push = []

    print("Xtestのデータを表示します")
    print(Xtest)
    
    # data_type == 1のときは状態と行動も出力される
    if data_type == 1:
        
        rot_count = 0
        for i in range(3):
            push.append(Xtest[index][i]*10)

        for i in range(4):
            if i < 3:
                Target.append(Xtest[index][i + 3]*10)
            else:
                Target.append(Xtest[index][i + 3])
                rot_count += 1
        Target.append((8,15))
        
        for i in range(5):
            Obj_2 = []
            for j in range(4):
                if j < 3:
                    Obj_2.append(Xtest[index][i*4 + j + 7]*10)
                else:
                    Obj_2.append(Xtest[index][i*4 + j + 7])
                    rot_count += 1
            Obj_2.append((8,14))
            Obj.append(Obj_2)
        
        print("=========================")
        print(Target,Obj)
        
        return push,pomdp_model.State(Obj,Target,5)  

    # data_type == ２のときは状態だけ出力される
    if data_type == 2:
        rot_count = 0
        for i in range(4):
            if i < 3:
                Target.append(Xtest[index][i]*10)
            else:
                Target.append(rotate_list[rot_count])
                rot_count += 1
        Target.append((8,15))
        
        for i in range(5):
            Obj_2 = []
            for j in range(4):
                if j < 3:
                    Obj_2.append(Xtest[index][i*3 + j + 3]*10)
                else:
                    Obj_2.append(rotate_list[rot_count])
                    rot_count += 1
            Obj_2.append((8,14))
            Obj.append(Obj_2)
        
        return pomdp_model.State(Obj,Target,5)  

def evaluate_prid(Ytest,Yprd):

    error = 0
    for i in range(len(Ytest)):
        for j in range(6):
            error1 = 0
            for k in range(3):
                error1 += pow( (Ytest[i][j*3 + k] - Yprd[i][j*3 + k]) , 2)
            
            error += math.sqrt(error1)

    error = error/(len(Ytest)*6)

    print("それぞれの箱のズレの平均値を表示")
    print(error*10,"cm")

def visual_change(ypre,yresult):

    color_list = []
    for i in range(6):
        dist = 0
        for j in range(3):
            dist += pow(ypre[j*4 + i] - yresult[j*4 + i],2)
        dist = math.sqrt(dist)
        color_list.append((dist*255,dist*255,dist*255))

    return color_list

      


if __name__ == '__main__':

    # 自分で状態を入力するパターン
    Obj = []
    Target = []
    # Target = pomdp_model.set_Obj(Obj)

    # s1 = pomdp_model.State(Obj,Target,5)
   
    # f = open('simulator_new_2cm_action_all_normalized3.csv','r')
    f = open('testdata_new_2cm_action_normalized3.csv','r')
    reader = csv.reader(f)
    Xdata = []
    Ydata = []

    Xtest = []
    Ytest = []
    rotate_list = []
    count = 0
    drag = []

    for row in reader:

        if count >= datanum and count < datanum + 1:
            data = dataset(row)
            result = test_data(row,data)
            Xtest.append(data.get_dataset())
            Ytest.append(result.get_testdata())

        if count >= datanum + 1:
            break

        count += 1

    push,s1 = change_state(Xtest,rotate_list,1,0)

    print("入力データから状態だけを表示します")
    print(s1.target,s1.Obj)


    action1 = pomdp_model.Action('a16')
    action2 = pomdp_model.Action('a17')
    action3 = pomdp_model.Action('a18')
    action4 = pomdp_model.Action('a19')
    action5 = pomdp_model.Action('a20')
    
    sample = pomdp_model.transitionmodel(5)

    s2 = sample.sample(s1,action1)
    s2_ob = pomdp_model.Observation(s2)
    s2_ob.correct_observ()
    print("ターゲットの オクルージョン比を表示します")
    print(s2_ob.occl_target)


    s3 = sample.sample(s1,action2)


    s4 = sample.sample(s1,action3)
    s4_ob = pomdp_model.Observation(s2)
    s4_ob.correct_observ()
    print("ターゲットの オクルージョン比を表示します")
    print(s4_ob.occl_target)


    s5 = sample.sample(s1,action4)
    s5_ob = pomdp_model.Observation(s5)
    s5_ob.correct_observ()
    print("ターゲットの オクルージョン比を表示します")
    print(s5_ob.occl_target)


    s6 = sample.sample(s1,action5)
    s6_ob = pomdp_model.Observation(s6)
    s6_ob.correct_observ()
    print("ターゲットの オクルージョン比を表示します")
    print(s6_ob.occl_target)

    s1.visualization(action1.drag,action1.push,name="s1+a6")
    s2.visualization()
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    s1.visualization(action2.drag,action2.push,name="s1+a7")
    s3.visualization()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    s1.visualization(action3.drag,action3.push,name="s1+a8")
    s4.visualization()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    s1.visualization(action4.drag,action4.push,name="s1+a9")
    s5.visualization()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    s1.visualization(action5.drag,action5.push,name="s1+a10")
    s6.visualization()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print("変化後を表示します")
    # print(s1.target,s1.Obj)
