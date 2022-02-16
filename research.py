
from pickle import TRUE
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    # Remove "Russia" from MyList 
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


import make_initial_belief2
import random
#from PIL.Image import NONE
import cv2
from matplotlib.pyplot import imshow
import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
import re
# import GPy
import copy
# from pomdp_model2 import Obj, Target
import transition_NN2 as tNN
import pomdp_py
import graspbytakasu
from pomdp_py.utils import TreeDebugger
import time
fflag = False 
# from state_change import evaluate_prid
posy = 7 

# 作業空間の大きさを600×600の画像上で表現
w = 1000
h = 1000
# kernel = GPy.kern.RBF(29,useGPU=True)
drag_len = 2
load_models = []

obj_num = 6

target_sizew = 8
target_sizeh = 15

particles_copy = []
scale_tar = 1
scale_ob = 1 
threshold= 1000000
count_sim = 0


rotate_mat = np.array([[0,1,0],
                      [-1,0,0],
                      [0,0,1],
                    ])


def check_nearid(Obj,push):

    for i in range (3):
        push[i] = push[i]
        # print("行動から抑える位置を出力します")
        # print(push[i])

    dis = math.sqrt(pow( (Obj[0][0] - push[0]) , 2) + pow( (Obj[0][1] - push[1]) , 2) + pow( (Obj[0][2] - push[2]) , 2) )
    id = 0
    for i in range(len(Obj)):
        if math.sqrt(pow( (Obj[i][0] - push[0]) , 2) + pow( (Obj[i][1] - push[1]) , 2) + pow( (Obj[i][2] - push[2]) , 2) ) < dis:
            dis = math.sqrt(pow( (Obj[i][0] - push[0]) , 2) + pow( (Obj[i][1] - push[1]) , 2) + pow( (Obj[i][2] - push[2]) , 2) )
            id = i
        
    return id

class RewardModel(pomdp_py.RewardModel):

    def __init__(self,num_obj):
        self.num_obj = num_obj

    def sample(self, state, action, next_state, normalized=False, **kwargs):
        # deterministic
        reward = 0
        w1 = 0.05
        w2 = 0.05
        w3 = 0.05 
        w4 = 10
        w5 = 1
        w6 = 0.001
        global count_sim 

        count_sim  += 1
        # print("action time is ...",count_sim)

        # print("action_number is ...",action.numint)
        if state.terminal == True:
            if reward == 0:
                print("There is no reward...1")
            return reward
        state.occlu_check()
        next_state.occlu_check()
        if action.model_num < 16:
            # print(state.Obj)
            # print(next_state.Obj_prev)

            # 物体の動きによる報酬
            # TODO対応させた箱同士での動きを計算するようにする
            for i in range(len(state.Obj)):
                reward -= w1*pow(state.Obj[i][0] - next_state.Obj_prev[i][0],2) + w2*pow(state.Obj[i][1] - next_state.Obj_prev[i][1],2) + w3*pow(state.Obj[i][2] - next_state.Obj_prev[i][2],2)
            # print("move_object reward is ...",reward)
            # オクルージョンの改善量による報酬

        
            reward -= w4 * ( state.occl_target - next_state.occl_target )
            # print("occulusion_object reward is ...",reward)

            # graspabilityによる評価値をここに書く
            
            if state.visible_area[0] >= threshold  and  next_state.visible_area[0] >= threshold:
                img1,img2 = state.get_depth_image()
        
                img3,img4 = next_state.get_depth_image()
                if cv2.countNonZero(img1) != 0 and cv2.countNonZero(img2) != 0 and cv2.countNonZero(img3) != 0 and cv2.countNonZero(img4) != 0:

                    Wc1 = graspbytakasu.calcWt2(img1,img2)
                    _,maxgrasp_s =graspbytakasu.graspability(img2,Wc1)
                    Wc2 = graspbytakasu.calcWt2(img3,img4)
                    _,maxgrasp_ns =graspbytakasu.graspability(img4,Wc2)

                    # print("s1 = ",maxgrasp_s)
                    # print("s2 = ",maxgrasp_ns)
                    # print("graspability reward is ...",-w6*(maxgrasp_ns - maxgrasp_s))
                    reward -=  w6*(maxgrasp_s - maxgrasp_ns)             
            
        if action.model_num == 16:
            reward += w5(1/(math.e(-state.occl_target - 0.5) + 1)-0.5)
            # print("grasp_reward is  ... ",reward)

        # print("\n")

        if reward == 0:
            print("There is no reward...2")
        
        return reward


class Action(pomdp_py.Action):
    def __init__(self,name):

        self.name = name
        # print(type(self.name))
        # print("action name is ...",self.name)
        numbers = re.sub(r'[^0-9]', '', name)
        self.numint = int(numbers) 
        numint = int(numbers) 
        # print("アクションナンバーを表示します")
        # print(numint)
        self.model_num = 0

        # １から５番のIDをを振っている（抑える箱のID）
        self.put_id = numint % 5
    
        # actionに応じたIDを振っている
        if numint >= 1 and numint <= 5:
            self.drag = (1.000000,0.000000)
            self.model_num = 0
        if numint >= 6 and numint <= 10:
            self.drag = (0.923879,0.382683)
            self.model_num = 1
        if numint >= 11 and numint <= 15:
            self.drag = (0.707107,0.707107)
            self.model_num = 2
        if numint >= 16 and numint <= 20:
            self.drag = (0.382683,0.923879)
            self.model_num = 3
        if numint >= 21 and numint <= 25:
            self.drag = (0.000000,1.000000)
            self.model_num = 4
        if numint >= 26 and numint <= 30:
            self.drag = (-0.382683,0.923880)
            self.model_num = 5
        if numint >= 31 and numint <= 35:
            self.drag = (-0.707107,0.707107)
            self.model_num = 6
        if numint >= 36 and numint <= 40:
            self.drag = (-0.923879,0.382683)
            self.model_num = 7
        if numint >= 41 and numint <= 45:
            self.drag = (-1.000000,0.000000)
            self.model_num = 8
        if numint >= 46 and numint <= 50:
            self.drag = (-0.923879,-0.382683)
            self.model_num = 9
        if numint >= 51 and numint <= 55:
            self.drag = (-0.707107,-0.707107)
            self.model_num = 10
        if numint >= 56 and numint <= 60:
            self.drag = (-0.382684,-0.923879)
            self.model_num = 11
        if numint >= 61 and numint <= 65:
            self.drag = (0.000000,-1.000000)
            self.model_num = 12
        if numint >= 66 and numint <= 70:
            self.drag = (0.382684,-0.923879)
            self.model_num = 13
        if numint >= 71 and numint <= 75:
            # 5000
            self.drag = (0.707107,-0.707107)
            self.model_num = 14
        if numint >= 76 and numint <= 80:
            self.drag = (0.923880,-0.382683)
            self.model_num = 15
        if numint == 81: #把持動作
            self.drag = None
            self.model_num = 16
     
    def __hash__(self):
        return hash(self.name)
    def __eq__(self, other):
        if isinstance(other, Action):
            return self.name == other.name
        elif type(other) == str:
            return self.name == other
    def __str__(self):
        return self.name
    def __repr__(self):
        return "Action(%s)" % self.name

    def push_pos(self,state):
        
        self.push = []

        self.push.append(state.Obj[self.put_id][0])
        self.push.append(state.Obj[self.put_id][1])
        self.push.append(state.Obj[self.put_id][2])

# モデルから出力された状態を状態に沿った形に変換する
def change_classstate(Xtest,rotate_list = None,data_type = 1,index = 0, boxsize = None):

    Target = []
    Obj = []
    # print(Xtest)
    count = 0
    # data_type == 1のときは状態と行動も出力される
    # if data_type == 1:

    #     for i in range(4):
    #         if i < 3:
    #             Target.append(Xtest[index][i + 5]*10)
    #         else:
    #             Target.append(rotate_list[count])
    #             count += 1

    #     Target.append((8,15))
        
    #     for i in range(5):
    #         Obj_2 = []
    #         for j in range(4):
    #             if j < 3:
    #                 Obj_2.append(Xtest[index][i*4 + j + 9]*10)
    #             else:
    #                 Obj_2.append(rotate_list[count])
    #                 count += 1
    #         Obj_2.append((boxsize[i][0],boxsize[i][1]))
    #         Obj.append(Obj_2)
        
    #     return drag,push,State(Obj,Target,5)  

    # data_type == ２のときは状態だけ出力される
    if data_type == 2:
        for i in range(4):
            if i < 3:
                Target.append(Xtest[index][i]*10*scale_ob)
            else:
                Target.append(rotate_list[count])
                count += 1
        Target.append((8,15))
        
        for i in range(5):
            Obj_2 = []
            for j in range(4):
                if j < 3:
                    Obj_2.append(Xtest[index][i*3 + j + 3]*10*scale_tar)
                else:
                    Obj_2.append(rotate_list[count])
                    count += 1
            Obj_2.append((boxsize[i][0],boxsize[i][1]))
            Obj.append(Obj_2)

        # print("変化後の状態を表示します")
        # print(Target,Obj)
        
        return State(Obj,Target,5) 

class State(pomdp_py.State):
    # 状態を入力するクラス__init__で状態を初期化
    def __init__(self, Obj,target, N = 5):
        """
        (tuple): Obj:((x_1,y_1,z_1,yaw_1),.....(x_N,y_N,z_N,,yaw_N)

        """
        self.Obj_prev = Obj
        self.target = target
        self.Obj = []
        height = []
        self.terminal = False
        count = 0

        for i in range(len(Obj)):
            if Obj[i] != None:
                self.Obj.append(Obj[i])
            else:
                count += 1
              
        self.Obj.sort(key = lambda x: x[2])

        # print(count)
        for i in range(count):
            # print("AAAAAAAAAAAAAAA")
            self.Obj.append(None)
        # print("state obj")
        # print(self.Obj)
        # アンカーボックスのIDからサイズを出力
        self.size_list = []
        # print(self.Obj)
        for i in range(len(self.Obj)):
            if self.Obj[i] == None:
                self.size_list.append(None)
                continue
            self.size_list.append((self.Obj[i][4][0],self.Obj[i][4][1]))
    
    def __hash__(self):
        return hash((self.target[0],self.Obj[0][0],self.Obj[3][0]))

    def __eq__(self, other):
        if isinstance(other, State):
            return self.target[0] == other.target[0]\
                and self.Obj[0][0] == other.Obj[0][0]\
                and self.Obj[3][0] == other.Obj[3][0]
        else:
            return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "State(%s | %s)" % (str(self.target), str(self.Obj))

    # 回転させた矩形をimgに描写してからリターン
    def rotatedRectangle(self,img, rotatedRect, color):
        # print(rotatedRect)
        (x,y), (width, height), angle = rotatedRect
        angle = math.radians(angle)
        
    
        # 回転する前の矩形の頂点
        pt1_1 = (int(x + width / 2), int(y + height / 2))
        pt2_1 = (int(x + width / 2), int(y - height / 2))
        pt3_1 = (int(x - width / 2), int(y - height / 2))
        pt4_1 = (int(x - width / 2), int(y + height / 2))
        
    
        # 変換行列
        t = np.array([[np.cos(angle),   -np.sin(angle), x-x*np.cos(angle)+y*np.sin(angle)],
                        [np.sin(angle), np.cos(angle),  y-x*np.sin(angle)-y*np.cos(angle)],
                        [0,             0,              1]])
        
        tmp_pt1_1 = np.array([[pt1_1[0]], [pt1_1[1]], [1]])
        tmp_pt1_2 = np.dot(t, tmp_pt1_1)
        pt1_2 = (int(tmp_pt1_2[0][0]), int(tmp_pt1_2[1][0]))
    
        tmp_pt2_1 = np.array([[pt2_1[0]], [pt2_1[1]], [1]])
        tmp_pt2_2 = np.dot(t, tmp_pt2_1)
        pt2_2 = (int(tmp_pt2_2[0][0]), int(tmp_pt2_2[1][0]))
    
        tmp_pt3_1 = np.array([[pt3_1[0]], [pt3_1[1]], [1]])
        tmp_pt3_2 = np.dot(t, tmp_pt3_1)
        pt3_2 = (int(tmp_pt3_2[0][0]), int(tmp_pt3_2[1][0]))
    
        tmp_pt4_1 = np.array([[pt4_1[0]], [pt4_1[1]], [1]])
        tmp_pt4_2 = np.dot(t, tmp_pt4_1)
        pt4_2 = (int(tmp_pt4_2[0][0]), int(tmp_pt4_2[1][0]))
        
    
        points = np.array([pt1_2, pt2_2, pt3_2, pt4_2])
        cv2.fillConvexPoly(img, points, color)

        return img

    def get_depth_image(self):

        img = np.ones((w,h,3),np.uint8)
        img2 = np.ones((w,h,3),np.uint8)

        parm = 20
        # 対象物の位置を表示
        rotate = ((self.target[0]*20,-self.target[1]*20+h),(20*8,20*15),-self.target[3]*180/math.pi) 
        img = self.rotatedRectangle(img,rotate,(self.target[2]*255/parm,self.target[2]*255/parm ,self.target[2]*255/parm ))
        
        box_list = []
        # print(self.Obj)
        for i in range(len(self.Obj)):

            # 座標系を合わせるために x + 500 -y + 500  -θ することで対象物中心の座標系から表現 
            # x座標，y座標　20画素で1cmを表現する
            rotate = ((self.Obj[i][0]*20,-self.Obj[i][1]*20+h),(self.size_list[i][0]*20,self.size_list[i][1]*20),-self.Obj[i][3]*180/math.pi) 

            img2 = self.rotatedRectangle(img2,rotate,(self.Obj[i][2]*255/parm ,self.Obj[i][2]*255/parm ,self.Obj[i][2]*255/parm ))

        img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  
        img_gray_tar = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_tar = cv2.subtract(img_gray_tar ,img_gray)

        # cv2.imshow("A",img_gray)
        # cv2.imshow("B",img_gray_tar)
        # cv2.waitKey(0)
        return img_gray,img_gray_tar



    def visualization(self,drag = None,push = None,name = None):

        img = np.ones((w,h,3),np.uint8)*255

        # 対象物の位置を表示
        rotate = ((self.target[0]*20,-self.target[1]*20+h),(20*8,20*15),-self.target[3]*180/math.pi) 
        img = self.rotatedRectangle(img,rotate,(0,0,255))
        id = 100

        # print("各物体の位置を表示します")
        # print(self.Obj)
        if push != None:
            id = check_nearid(self.Obj,push)
            # print("抑えた物体の位置を表示します")
            # print(self.Obj[id])

        box_list = []
        # print(self.Obj)
        for i in range(len(self.Obj)):

            # 座標系を合わせるために x + 500 -y + 500  -θ することで対象物中心の座標系から表現 
            # x座標，y座標　20画素で1cmを表現する
            rotate = ((self.Obj[i][0]*20,-self.Obj[i][1]*20+h),(self.size_list[i][0]*20,self.size_list[i][1]*20),-self.Obj[i][3]*180/math.pi) 
            if id != i:
                img = self.rotatedRectangle(img,rotate,(25*(i+1),25*(i+1),25*(i+1)))
            else:
                img = self.rotatedRectangle(img,rotate,(255,0,0))

        if drag != None:
            pt1 = (int(self.target[0]*20), int(-self.target[1]*20+h))
            pt2 = (int(self.target[0]*20 + drag[0]*50),int(-self.target[1]*20+h - drag[1]*50))

            cv2.arrowedLine(img,pt1,pt2,(0,255,0),thickness=2,tipLength=0.2)
        
        if name != None:
            cv2.imshow(name + 'state',img)

        if name == None:
            cv2.imshow('state',img)
        return img
    
    def occlu_check(self):
    
        img_list = []
        img =  np.zeros((w,h,3),np.uint8)
        self.img_target = np.zeros((w,h,3),np.uint8)
        rotate = ((self.target[0]*20,- self.target[1]*20+h),(8*20,15*20),-self.target[3]*180/math.pi) 
        img2 = cv2.cvtColor(self.rotatedRectangle(img,rotate,(255,255,255)),cv2.COLOR_BGR2GRAY)
        self.img_target = img2
        self.occl_list = []
        self.visible_area = []

        for i in range(len(self.Obj)):
            img =  np.zeros((w,h,3),np.uint8)
            rotate = ((self.Obj[i][0]*20,-self.Obj[i][1]*20+h),(self.size_list[i][0]*20,self.size_list[i][1]*20),-self.Obj[i][3]*180/math.pi) 
            img2 = cv2.cvtColor(self.rotatedRectangle(img,rotate,(255,255,255)),cv2.COLOR_BGR2GRAY)
            img_list.append(img2)

        N = len(self.Obj)
    
        for i in range(N):

            img = img_list[N-1-i]
            whitePixels = np.count_nonzero(img)

            # 上にある物体による遮蔽を計算
            for j in range(i):
                img = cv2.subtract(img ,img_list[N-i+j])
            
            whitePixels_diff = np.count_nonzero(img)
            self.visible_area.append(whitePixels_diff)
            self.occl_list.append(whitePixels_diff/whitePixels)

        self.occl_list.reverse()
        img = self.img_target
        whitePixels = np.count_nonzero(img)

        for i in range(len(self.Obj)):
            img = cv2.subtract(img ,img_list[i])
        
        whitePixels_diff = np.count_nonzero(img)
        #　self.occl_targetには見えている面積が格納されている
        self.occl_target = whitePixels_diff/whitePixels
        self.visible_area.append(whitePixels_diff)
        self.visible_area.reverse()

        # print("target_occusion is ...",self.occl_target)
        # print("other Object _occusion is ...",self.occl_list)
        # print("visible area is ...",self.visible_area)

        return

class SampleAction(Action):
    def __init__(self):
        super().__init__("sample")

class CheckAction(Action):
    def __init__(self, id):
        self.rock_id = id
        super().__init__("check-%d" % self.id)

def get_state(sample):
    
    Obj = []
    Target = []
    num = 5

    for i in range(4):
        Target.append(sample[0][i])

    Target.append((sample[0][4],sample[0][5]))

    for j in range(num):
        Obj2 = []
        if sample[j+1] == None:
            Obj.append(Obj2)
            continue
        for i in range(4):
            Obj2.append(sample[j+1][i])

        Obj2.append((sample[j+1][4],sample[j+1][5]))

        Obj.append(Obj2)
    
    return Obj, Target
    
class Observation(pomdp_py.Observation):
    # img_list：各物体の二値画像
    # state：この状態に対しての観測
    # occl_list：各物体のオクルージョン比率
    # im_bbox：各物体のBBOX
    # correct_ob：観測集合を出力（計算上）
    def __init__(self, Obj,target, N = 5):
        """
        (tuple): Obj:((x_1,y_1,z_1,yaw_1),.....(x_N,y_N,z_N,,yaw_N)

        """
        if Obj != None and target != None:
            self.Obj_pos = []
            self.tar_pos = target
            count = 0
            
            # print(Obj)
            for i in range(N):
                if len(Obj[i]) != 0:
                    self.Obj_pos.append(Obj[i])
                if len(Obj[i]) == 0:
                    count += 1
            # print(self.Obj_pos)
            self.Obj_pos.sort(key = lambda x: x[2])
            for i in range(count):
                self.Obj_pos.append(None)
            
            # アンカーボックスのIDからサイズを出力
            self.size_list = []
            for i in range(len(self.Obj_pos)):
                if self.Obj_pos[i] == None:
                    self.size_list.append(None)
                    continue
                self.size_list.append((self.Obj_pos[i][4][0],self.Obj_pos[i][4][1]))

            state = State(self.Obj_pos,target)

            self.img_list = [] 
            self.state = state
            
            img =  np.zeros((w,h,3),np.uint8)
            self.img_target = np.zeros((w,h,3),np.uint8)
            rotate = ((state.target[0]*20,-state.target[1]*20+h),(8*20,15*20),-state.target[3]*180/math.pi) 
            img2 = cv2.cvtColor(state.rotatedRectangle(img,rotate,(255,255,255)),cv2.COLOR_BGR2GRAY)
            self.img_target = img2

            # print(state.size_list)
            for i in range(N):
                if state.size_list[i] == None:
                    self.img_list.append(None)
                    continue
                img =  np.zeros((w,h,3),np.uint8)
                rotate = ((state.Obj[i][0]*20,-state.Obj[i][1]*20+h),(state.size_list[i][0]*20,state.size_list[i][1]*20),-state.Obj[i][3]*180/math.pi) 
                img2 = cv2.cvtColor(state.rotatedRectangle(img,rotate,(255,255,255)),cv2.COLOR_BGR2GRAY)
                self.img_list.append(img2)
        else:
            self.tar_pos = []
            self.Obj_pos = []
            self.Obj_pos.append(tuple([10,10,10,0*math.pi/9,(8,8)]))
            self.Obj_pos.append(tuple([10,10,10,0*math.pi/9,(8,8)]))
            self.Obj_pos.append(tuple([10,10,10,0*math.pi/9,(8,8)]))
            self.Obj_pos.append(tuple([10,10,10,0*math.pi/9,(8,8)]))
            self.Obj_pos.append(tuple([10,10,10,0*math.pi/9,(8,8)]))

            self.tar_pos.append((10,10,10,0,(8,8)))

    # ある状態におけるそれぞれのバウンディングボックスを出力します
    def check_observation(self,N = 5):

        # for i in range(N):
        #     cv2.imshow("check_state",self.img_list[i])
            cv2.waitKey(0)

    # pointsに白黒画像の白色の部分を格納
    def return_whitepoints(self,img):
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        points = cv2.findNonZero(img)

        

        return points

    # 離散化していない観測集合を出力
    def get_result(self,rect,box,number):
 
        x = ((box[0][0] + box[1][0] + box[2][0] + box[3][0])/4)/20
        y = -((box[0][1] + box[1][1] + box[2][1] + box[3][1])/4 - h)/20
        z = self.state.Obj[number][2]

        width = rect[1][0]/20
        height = rect[1][1]/20

        # theta = rect[2]*math.pi/180
        theta = -rect[2]
        if(theta < 0):
            theta = theta + 180
        
        if(theta > 180):
            theta = theta - 180

        # theta = theta*math.pi/180

        return (x,y,z,theta,width,height)

    
    # 一意のBBOXになるように正規化する
    def normalized_observ(self,not_normalized_ob):

        # not_normalized_ob = list(not_normalized_ob1)
        
        for i in range(len(not_normalized_ob)):
            if not_normalized_ob[i] == None:
                continue
            not_normalized_ob[i] = list(not_normalized_ob[i])

        
        # print(not_normalized_ob)


        for i in range(len(not_normalized_ob)):
            if not_normalized_ob[i] == None:
                continue

            if not_normalized_ob[i][4] > not_normalized_ob[i][5]:

                change = not_normalized_ob[i][4]
                not_normalized_ob[i][4] = not_normalized_ob[i][5]
                not_normalized_ob[i][5] = change
                not_normalized_ob[i][3] = not_normalized_ob[i][3] - 90
            
            if int(not_normalized_ob[i][4]) == int(not_normalized_ob[i][5]):
                not_normalized_ob[i][3] = not_normalized_ob[i][3] - 90

        
            if(not_normalized_ob[i][3] < 0):
                not_normalized_ob[i][3] = not_normalized_ob[i][3] + 180
            
            if(not_normalized_ob[i][3] > 180):
                not_normalized_ob[i][3] = not_normalized_ob[i][3] - 180
            

        for i in range(len(not_normalized_ob)):
            if not_normalized_ob[i] == None:
                continue
            not_normalized_ob[i][3] = not_normalized_ob[i][3]*math.pi/180



        return not_normalized_ob
    
    def normalized_observ2(self,not_normalized_ob):

        # not_normalized_ob = list(not_normalized_ob1)
        
        for i in range(len(not_normalized_ob)):
            if not_normalized_ob[i] == None:
                continue
            not_normalized_ob[i] = list(not_normalized_ob[i])
        
        for i in range(len(not_normalized_ob)):
            if not_normalized_ob[i] == None:
                continue
            not_normalized_ob[i][3] =  not_normalized_ob[i][3]*180/math.pi

        for i in range(len(not_normalized_ob)):
            if not_normalized_ob[i] == None:
                continue
            if not_normalized_ob[i][4] > not_normalized_ob[i][5]:

                change = not_normalized_ob[i][4]
                not_normalized_ob[i][4] = not_normalized_ob[i][5]
                not_normalized_ob[i][5] = change
                not_normalized_ob[i][3] = not_normalized_ob[i][3] - 90
            
            if int(not_normalized_ob[i][4]) == int(not_normalized_ob[i][5]):
                not_normalized_ob[i][3] = not_normalized_ob[i][3] - 90


            if(not_normalized_ob[i][3] < 0):
                not_normalized_ob[i][3] = not_normalized_ob[i][3] + 180
            
            if(not_normalized_ob[i][3] > 180):
                not_normalized_ob[i][3] = not_normalized_ob[i][3] - 180
            

        for i in range(len(not_normalized_ob)):
            if not_normalized_ob[i] == None:
                continue
            not_normalized_ob[i][3] = not_normalized_ob[i][3]*math.pi/180



        return not_normalized_ob

    # 間違えて作った．またいつか使う日まで．．
    # 観測から中心位置を表示する関数
    # def get_center(self,box,center_list,num):
        
    #     x = 0
    #     y = 0

    #     # print(self.Obj_pos)
    #     pos = copy.copy(self.Obj_pos)
    #     pos.reverse()
    
    #     for i in range(len(box)):
    #         x += box[i][0]
    #         y += box[i][0]
        
    #     x = x/len(box)
    #     y = y/len(box)
    #     z = pos[num][2]


    #     center_list.append((x,y,z))

    # ある状態の正解の観測結果を出力します
    def correct_observ(self,N = 5):

        correct_list = []
        # それぞれのオブジェクトのオクルージョン比率を格納
        self.occl_list = []
        self.obj_area = []

        self.im_bbox =  np.zeros((w,h,3),np.uint8)
        self.correct_ob = []
        not_nomalized_ob =[]
        center_list = []

        #DEBUG MODE
        # self.state.visualization()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        
        for i in range(N):

            img = self.img_list[N-1-i]
            whitePixels = np.count_nonzero(img)

            # 上にある物体による遮蔽を計算
            for j in range(i):
                img = cv2.subtract(img ,self.img_list[N-i+j])
            
            
            # print(whitePixels)
            point = self.return_whitepoints(img)
            if point is None:
                not_nomalized_ob.append(None)
                self.obj_area.append(0)
                self.occl_list.append(0)
                correct_list.append(img)
                continue
                
            # pointsに白黒画像の白色の部分を格納
            # points = np.array(point)
            # print("points are .....")
            # print(points)
            # rectにはBBOXの左上の座標，wとh,回転角度を入力
            rect = cv2.minAreaRect(np.float32(point))
            box = cv2.boxPoints(rect)
            # boxの角の点４つを入力する
            box = np.int0(box)
            # self.get_center(box,center_list,i)
            not_nomalized_ob.append(self.get_result(rect,box,N-1-i))
            self.im_bbox = cv2.drawContours(self.im_bbox,[box],0,(255*random.random(),255*random.random(),255*random.random()),2)

            whitePixels_diff = np.count_nonzero(img)
            self.obj_area.append(whitePixels_diff)
            self.occl_list.append(whitePixels_diff/whitePixels)
            correct_list.append(img)

        # print("各物体の中心糸を表示します",center_list)

        img = self.img_target
        whitePixels = np.count_nonzero(img)

        for i in range(N):
            img = cv2.subtract(img ,self.img_list[i])
        
        whitePixels_diff = np.count_nonzero(img)
        
        points = np.array(self.return_whitepoints(img))
        self.obj_area.append(whitePixels_diff)
        # rectにはBBOXの左上の座標，wとh,回転角度を入力
        # cv2.imshow("win",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(points)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        # boxの角の点４つを入力する
        box = np.int0(box)
        # print("QQQQQQQQQ")
        # print(rect)
        self.im_bbox = cv2.drawContours(self.im_bbox,[box],0,(255*random.random(),255*random.random(),255*random.random()),2)
        self.occl_target = whitePixels_diff/whitePixels

        # cv2.imshow("correct",self.im_bbox)
        # cv2.waitKey(0)

        self.obj_area.reverse()
        self.correct_ob = self.normalized_observ(not_nomalized_ob)
        self.correct_ob.reverse()
        # print("正解の観測データを出力します")
        # print(self.correct_ob)

        return self.correct_ob
        
        # for i in range(N):
        #     cv2.imshow("check_state",correct_list[i])
        #     print("オクルージョンの割合を表示します")
        #     print(self.occl_list[i])
        #     cv2.waitKey(0)

    # ノイズが乗った出力を表示します
    # オクルージョン，センサとの距離に応じてノイズを出力するようにプログラム

    def sample_observ(self):
        
        occ = list(reversed(self.occl_list))
        w1 = 1 
        w2 = 1
        w3 = 1
        theta = 1
        sample = []
        sample_img =  np.zeros((self.im_bbox.shape[0],self.im_bbox.shape[1],3),np.uint8)
        N = len(self.state.Obj)

          
        # x = w1 * np.random.normal(0,1-self.occl_target)
        # y = w1 * np.random.normal(0,1-self.occl_target)
        # z = w1 * np.random.normal(0,1-self.occl_target)

        # width = w2 * np.random.normal(0,1-self.occl_target)
        # height = w2 * np.random.normal(0,1-self.occl_target)

        # theta = w3 * np.random.normal(0,1-self.occl_target)
        x = 0
        y = 0
        z = 0

        width = 0
        height = 0

        theta = 0
        # 整数型に丸めている．観測の離散化のため
        # sample.append(( round(self.state.target[0] + x ), round(self.state.target[1] + y) , round(self.state.target[2] + z ),round(self.state.target[3] + theta),round(self.state.target[4][0]) ,round(self.state.target[4][1])))
        sample.append(( self.state.target[0] + x , self.state.target[1] + y , self.state.target[2] + z ,self.state.target[3] + theta,self.state.target[4][0] ,self.state.target[4][1]))



        for i in range(len(self.state.Obj)):
            if self.correct_ob[i] == None:
                sample.append(None)
                continue
        
            # x = w1 * np.random.normal(0,0)
            # y = w1 * np.random.normal(0,0)
            # z = w1 * np.random.normal(0,0)

            # width = w2 * np.random.normal(0,0)
            # height = w2 * np.random.normal(0,0)

            # theta = w3 * np.random.normal(0,0)
            x = 0
            y = 0
            z = 0

            width = 0
            height = 0

            theta = 0
            sample.append(( self.correct_ob[i][0] + x , self.correct_ob[i][1] + y , self.correct_ob[i][2] + z ,self.correct_ob[i][3] + theta,self.correct_ob[i][4] + width,self.correct_ob[i][5] + height ))
            # print("角度を表示します")
            # print(self.correct_ob[i][3])
        rect = []

        for i in range(len(sample)):
            if sample[i] == None:
                rect.append(None)
                continue
            # 角度が何故かおかしい．．
            (w,h) = ( sample[i][4]*20,sample[i][5]*20)
            (x,y) = ( sample[i][0]*20,-sample[i][1]*20 + self.im_bbox.shape[1] )
            theta = -sample[i][3]*180/math.pi
            rect.append(((x,y),(w,h),theta))
            box = cv2.boxPoints(rect[i])
            # boxの角の点４つを入力する
            box = np.int0(box)
            sample_img = cv2.add(sample_img,cv2.drawContours(sample_img,[box],0,(255*random.random(),255*random.random(),255*random.random()),2))

        sample = self.normalized_observ2(sample)
        
        for i in range(len(sample)):
            if sample[i] == None:
                continue

            sample[i] = list(sample[i])
            for j in range(len(sample[i])):
                if j != 3:

                    # 整数型に丸めている．観測の離散化のため
                    # sample[i][j] = int(sample[i][j])
                    sample[i][j] = sample[i][j]
                else:
                    sample[i][j] = sample[i][j]

        # cv2.imshow("sample_image",sample_img)
        # cv2.waitKey(0)

        Obj,target = get_state(sample)

        return Observation(Obj,target)
    
    def __hash__(self):
        return hash((self.tar_pos[0],self.Obj_pos[0][0],self.Obj_pos[3][0]))

    def __eq__(self, other):
        if isinstance(other, Observation):
            return self.tar_pos[0] == other.tar_pos[0]\
                and self.Obj_pos[0][0] == other.Obj_pos[0][0]\
                and self.Obj_pos[3][0] == other.Obj_pos[3][0]
        else:
            return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "Obs(%s | %s)" % (str(self.tar_pos), str(self.Obj_pos))

class ObservationModel(pomdp_py.ObservationModel):
    
    def __init__(self,num):
        self.n = num
    
    def sample(self, next_state, action, argmax=False):

        if next_state.terminal == False:

            Obj = next_state.Obj
            target = next_state.target
            # print("Observation is ...")
            # print(Obj,target)
            x =  Observation(Obj,target)
            x.correct_observ(5)
            obs = x.sample_observ()

            return obs
        else:
            return Observation(None,None)

def set_Obj(Obj):

    Obj.append(tuple([8,5+posy,3,0*math.pi/9,(8,13)]))
    Obj.append(tuple([24,6+posy,3,0*math.pi/9,(8,14)]))
    Obj.append(tuple([15,5+posy,6,0*math.pi/9,(8,14)]))
    Obj.append(tuple([20,6+posy,7,0*math.pi/9,(8,12)]))
    Obj.append(tuple([16,6+posy,10,0*math.pi/9,(8,13)]))

    return ((16,6.5+posy,2,0,(8,15)))

class PolicyModel(pomdp_py.RolloutPolicy):
    """Simple policy model according to problem description."""
    def __init__(self, n):
        # check_actions = set({CheckAction(rock_id) for rock_id in range(k)})
        # self._move_actions = {MoveEast, MoveWest, MoveNorth, MoveSouth}
        # self._other_actions = {SampleAction()} | check_actions
        # self._all_actions = self._move_actions | self._other_actions
        self._n = n

    def argmax(self, state, normalized= True, **kwargs):
        """Returns the most likely reward"""
        # print("JYYYYY")
        return 10
    def get_nextstate(self,state,action):
        target = state.target
        target = list(target)
        target[0] = target[0] + action.drag[0]
        target[1] = target[1] + action.drag[1]
        return State(state.Obj,target)

    def get_all_actions(self, state,**kwargs):

        # 多分，ここに状態を入力して行動どの行動を実行可能であるかを出力する関数を作る
        state.occlu_check()
        can_push = []
        action_id_list = []
        action_list = []
        drag_dir = 16
        
        img1,img2 = state.get_depth_image()
        if cv2.countNonZero(img1) > threshold and cv2.countNonZero(img2) > threshold:
            Wc = graspbytakasu.calcWt2(img1,img2)
            grasp,maxgrasp =graspbytakasu.graspability(img2,Wc)
            # graspabilityの結果を画像として確認したいときにはここをTRUEに変更
            debugmade = False
            if debugmade == True:
                if grasp[0] > 100:
                    alpha = 100
                    point = np.array(grasp[2])
                    theta = -(grasp[1] + 90)
                    vdp = np.array([int(alpha*math.cos(math.radians(theta))), int(-alpha*math.sin(math.radians(theta)))])
                    gp1 = point + vdp
                    gp2 = point - vdp
                    cv2.line(img1, (int(gp1[1]), int(gp1[0])), (int(gp2[1]), int(gp2[0])), (255, 255, 255), 3)
                    cv2.circle(img1, (int(gp1[1]), int(gp1[0])), 10, (255, 255, 255), -1)
                    cv2.circle(img1, (int(gp2[1]), int(gp2[0])), 10, (255, 255, 255), -1)
                    cv2.circle(img1, (int(grasp[2][1]), int(grasp[2][0])), 10, (255, 255, 255), -1)
                cv2.imshow('image', img1)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
            if maxgrasp > 100:
                # print("##################################")
                action_list.append(Action("a81"))

        for i in range (6):
            if state.visible_area[i] > 10000:
                can_push.append(i)

        state.occlu_check()
        drag_list = []
        for i in range(16):
            a = Action(str((i*5 + 1)))
            nexts = self.get_nextstate(state,a)
            nexts.occlu_check()

            if nexts.occl_target - state.occl_target > 0:
                drag_list.append(i*5 + 1)

        for j in range(len(can_push)):
            for i in range (len(drag_list)):
                action_id_list.append(drag_list[i] + can_push[j])
     
       
        for i in range(len(action_id_list)):
            # TODO 修正の必要あり
            if str(action_id_list[i]) != "81":
                action_list.append(Action("a" + str(action_id_list[i])))
    
        
        # print(action_list)
        return action_list

    def rollout(self, state, history=None):
        global fflag 
        fflag = True
        print("random rollout now")
        # random.sample(self.get_all_actions(state=state), 1)[0]は行動の名前がランダムに出力されている.
        # get_all_actionsで可能なすべてのアクションに限定し，その中からのランダムな行動を選択している
        return random.sample(self.get_all_actions(state=state), 1)[0]

def check_region(state):
    flag = False
    
    for i in range(2):      
        if state.target[i] < 0 and  state.target[i] > 30:
            flag = True
            
    for i in range(len(state.Obj)):
        for j in range(2):
            if state.Obj[i][j] < 0 and state.Obj[i][j] > 30:
                flag = True
    
    return flag





class TransitionModel(pomdp_py.TransitionModel):
    def __init__(self,n = 5):
        
        self.obj_num = n
    
    # def check_state(self,state):
    #     print(pomdp_py.Particles.particles)
    
    # 状態，行動，アクションから，次の行動を出力する関数
    def sample(self,state,action):
        global fflag

        if fflag == False:
            print("not randomrollout")
        
        if fflag == True:
            print("randomrollout")
            fflag = False

        flag = False
        spre = []
        rotate_list = []
        # self.check_state(state)
        if action.model_num < 16:
            action.push_pos(state)
            # print("変化前の状態を表示します")
            # print(state.target,state.Obj)

            # for i in range(2):
            #     spre.append(action.drag[i]*drag_len/10)

            for i in range(3):
                spre.append(action.push[i]/10/scale_ob)

            for i in range(4):
                if i != 3:
                    spre.append(state.target[i]/10/scale_tar)
                else:
                    rotate_list.append(state.target[3])
            
            for i in range(len(state.Obj)):
                for j in range(4):
                    if j != 3:
                        # print(state.Obj[i][j]/10/scale_ob)
                        spre.append(state.Obj[i][j]/10/scale_ob)
                    else:
                        rotate_list.append(state.Obj[i][j])

            test = []
            test.append(spre)
            
            # print("type_check")
            # print(test)
            snext = tNN.eval(test,action.model_num)
            # print("type_check")
            # print(snext)
            state_nex =change_classstate(snext,rotate_list,2,0,state.size_list)
            flag = check_region(state_nex)
            state_nex.terminal = False    

        # TODO　ここから下で把持したときに次の状態がなくなるようなプログラムを書かなければいけない
        if action.model_num == 16:
            print("action 81 is selected")
            state_nex = state
            state_nex.terminal = True

        return state_nex
        
        
    
class SampleProblem(pomdp_py.POMDP):

    def __init__(self,init_state, obj_num , init_belief):
        
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(obj_num),
                               TransitionModel(obj_num),
                               ObservationModel(obj_num),
                               RewardModel(obj_num))
        env = pomdp_py.Environment(init_state,
                                   TransitionModel(obj_num),
                                   RewardModel(obj_num))
        # self._rock_locs = rock_locs
        super().__init__(agent, env, name="SampleProblem") 


# 現在の信念から平均的な状態を変換する
def belief_to_state(particles):

    target_list = []
    object_list = []

    for i in range(len(particles)):
        s = particles[i]
        target_list.append(s.target)
        object_list.append(s.Obj)
    
    center_posx = 0
    center_posy = 0
    center_posz = 0
    
    for i in range(len(target_list)):

        center_posx  += target_list[i][0]
        center_posy  += target_list[i][1]
        center_posz  += target_list[i][2]
    
    center_posx = center_posx/len(target_list)
    center_posy = center_posy/len(target_list)
    center_posz = center_posz/len(target_list)

    Target = (center_posx,center_posy,center_posz,target_list[0][3],(target_sizew,target_sizeh))

    center_posx = 0
    center_posy = 0
    center_posz = 0
    Obj = []

    for i in range(obj_num-1):

        center_posx = 0
        center_posy = 0
        center_posz = 0
        obj = []
        wid = 0
        hei = 0

        for j in range(len(object_list)):
            center_posx += object_list[j][i][0]
            center_posy += object_list[j][i][1]
            center_posz += object_list[j][i][2]
            wid += object_list[j][i][4][0]
            hei += object_list[j][i][4][1]
        

        center_posx = center_posx/len(object_list)
        center_posy = center_posy/len(object_list)
        center_posz = center_posz/len(object_list)
        wid = wid/len(object_list)
        hei = hei/len(object_list)
        obj = (center_posx,center_posy,center_posz,object_list[0][i][3],(wid,hei))
        Obj.append(obj)

    print(Obj,Target)
    
    return State(Obj,Target)

def get_conerpoint(rotatedRect):
        # print(rotatedRect)
        (x,y), (width, height), angle = rotatedRect
        angle = math.radians(angle)
        
    
        # 回転する前の矩形の頂点
        pt1_1 = (int(x + width / 2), int(y + height / 2))
        pt2_1 = (int(x + width / 2), int(y - height / 2))
        pt3_1 = (int(x - width / 2), int(y - height / 2))
        pt4_1 = (int(x - width / 2), int(y + height / 2))
        
    
        # 変換行列
        t = np.array([[np.cos(angle),   -np.sin(angle), x-x*np.cos(angle)+y*np.sin(angle)],
                        [np.sin(angle), np.cos(angle),  y-x*np.sin(angle)-y*np.cos(angle)],
                        [0,             0,              1]])
        
        tmp_pt1_1 = np.array([[pt1_1[0]], [pt1_1[1]], [1]])
        tmp_pt1_2 = np.dot(t, tmp_pt1_1)
        pt1_2 = (int(tmp_pt1_2[0][0]), int(tmp_pt1_2[1][0]))
    
        tmp_pt2_1 = np.array([[pt2_1[0]], [pt2_1[1]], [1]])
        tmp_pt2_2 = np.dot(t, tmp_pt2_1)
        pt2_2 = (int(tmp_pt2_2[0][0]), int(tmp_pt2_2[1][0]))
    
        tmp_pt3_1 = np.array([[pt3_1[0]], [pt3_1[1]], [1]])
        tmp_pt3_2 = np.dot(t, tmp_pt3_1)
        pt3_2 = (int(tmp_pt3_2[0][0]), int(tmp_pt3_2[1][0]))
    
        tmp_pt4_1 = np.array([[pt4_1[0]], [pt4_1[1]], [1]])
        tmp_pt4_2 = np.dot(t, tmp_pt4_1)
        pt4_2 = (int(tmp_pt4_2[0][0]), int(tmp_pt4_2[1][0]))
        
    
        points = np.array([pt1_2, pt2_2, pt3_2, pt4_2])

        return points

def intersect(p1, p2, p3, p4):

    print(p1)
    print(p2)
    print(p3)
    print(p4)

    tc1 = (p1[0] - p2[0]) * (p3[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p3[0])
    tc2 = (p1[0] - p2[0]) * (p4[1] - p1[1]) + (p1[1] - p2[1]) * (p1[0] - p4[0])
    td1 = (p3[0] - p4[0]) * (p1[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p1[0])
    td2 = (p3[0] - p4[0]) * (p2[1] - p3[1]) + (p3[1] - p4[1]) * (p3[0] - p2[0])
    
    if tc1*tc2<0 and td1*td2<0:
        return True
    else:
        return False
     

def decide_support_position(state,action):

    id = action.put_id
    direction = action.drag

    x = state.Obj[id][0]
    y = state.Obj[id][1]
    z = state.Obj[id][2]

    theta = state.Obj[id][3]

    wid = state.Obj[id][4][0]
    hei = state.Obj[id][4][1]

    rotate = ((x,y),(wid,hei),theta*180/math.pi) 
    # self.rotatedRectangle(img,rotate,(255,255,255))
    point = get_conerpoint(rotate)

    target_point = np.array([x,y])
    target_point2 = np.array([x + direction[0] *100 ,y + direction[1]*100])
    support_point = []
    for i in range(4):
        if i == 0:
            if intersect(target_point,target_point2,point[0],point[1]):
                support_point.append((point[0][0] + point[1][0])/2)
                support_point.append((point[0][1] + point[1][1])/2)
        if i == 1:
            if intersect(target_point,target_point2,point[1],point[2]):
                support_point.append((point[1][0] + point[2][0])/2)
                support_point.append((point[1][1] + point[2][1])/2)
        if i == 2:   
            if intersect(target_point,target_point2,point[2],point[3]):
                support_point.append((point[2][0] + point[3][0])/2)
                support_point.append((point[2][1] + point[3][1])/2)
        if i == 3:
            if intersect(target_point,target_point2,point[3],point[0]):
                support_point.append((point[3][0] + point[0][0])/2)
                support_point.append((point[3][1] + point[0][1])/2)
    
    support_point[0] += direction[0]*5
    support_point[1] += direction[1]*5
 

    print("support position is ...",support_point)

    return support_point

def decide_action(action,particles):
    
    print("direction of dragging is ...")
    print(action.drag)

    print("pushing position is ...")
    print(action.push)
    # パーティクルの平均状態を出力する
    s = belief_to_state(particles)
    s.visualization(name="belief")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    support = decide_support_position(s,action)

    direction = np.array([action.drag[0],action.drag[1],1])
    supporting = np.array([support[0],support[1],action.push[2]])
    sliding = make_initial_belief2.sliding

    # make_initial_beliefのrotate_objectと同じ値にする必要あり 
    # posx = 45
    # posy = -10

    # direction[0] = direction[0] - posx
    # direction[1] = direction[1] - posy

    # direction = np.dot(rotate_mat,direction)


    # supporting[0] = supporting[0] - posx
    # supporting[1] = supporting[1] - posy

    # supporting = np.dot(rotate_mat,supporting)


    # sliding[0] = sliding[0] - posx
    # sliding[1] = sliding[1] - posy

    # sliding = np.dot(rotate_mat,sliding)

    # make_initial_beliefのget_state と同じ値にする必要あり 
    x = -40
    y = 25
    # 机の高さ分プラスする
    z = -64

    direction[0] = (direction[0] - x)/100
    direction[1] = (direction[1] - y)/100
    direction[2] = (direction[2] - z)/100

    supporting[0] = (supporting[0] - x)/100
    supporting[1] = (supporting[1] - y)/100
    supporting[2] = (supporting[2] - z)/100

    sliding[0] = (sliding[0] - x )/100
    sliding[1] = (sliding[1] - y )/100
    sliding[2] = (sliding[2] - z )/100


    direction = np.transpose(direction)
    supporting = np.transpose(supporting)
    sliding = np.transpose(sliding)

    np.savetxt('direction.csv', direction,fmt='%12.8f',delimiter=",")
    np.savetxt('supporting.csv', supporting,fmt='%12.8f', delimiter=",")
    np.savetxt('sliding.csv', sliding,fmt='%12.8f', delimiter=",")


def test_planner(sample, planner, nsteps=3, discount=0.95):
    
    gamma = 1.0
    total_reward = 0
    total_discounted_reward = 0

    for i in range(nsteps):
        print("==== Step %d ====" % (i+1))
        # ここにagentを入力するだけ？
        # 行動決定！
        start = time.time()

        action = planner.plan(sample.agent)
        action.push_pos(sample.env.state)
        elapsed_time = time.time() - start
        print ("time:{0}".format(elapsed_time) + "[sec]")
        
        print("action name is ...",action.name)

        sample.env.state.visualization(action.drag,action.push,name="s1+a7")

        decide_action(action,sample.agent.belief.particles)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        TreeDebugger(sample.agent.tree).pp



        # 真の状態を出力
        true_state = copy.deepcopy(sample.env.state)
        # 真の状態が変化
        env_reward = sample.env.state_transition(action, execute=True)
        true_next_state = copy.deepcopy(sample.env.state)
        true_next_state.visualization(name = "s8")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 観測が得られる
        real_observation = sample.env.provide_observation(sample.agent.observation_model,
                                                              action)

        # 信念の更新
        sample.agent.update_history(action, real_observation)
        planner.update(sample.agent, action, real_observation)
        total_reward += env_reward
        total_discounted_reward += env_reward * gamma
        gamma *= discount
        print("True state: %s" % true_state)
        print("Action: %s" % str(action))
        print("Observation: %s" % str(real_observation))
        print("Reward: %s" % str(env_reward))
        print("Reward (Cumulative): %s" % str(total_reward))
        print("Reward (Cumulative Discounted): %s" % str(total_discounted_reward))
        if isinstance(planner, pomdp_py.POUCT):
            print("__num_sims__: %d" % planner.last_num_sims)
            print("__plan_time__: %.5f" % planner.last_planning_time)
        if isinstance(planner, pomdp_py.PORollout):
            print("__best_reward__: %d" % planner.last_best_reward)
        print("World:")
        # rocksample.print_state()

        if sample.env.state.terminal == True:
            break

    return total_reward, total_discounted_reward

def make_noise(Obj):

    
    x = np.random.normal(0,1)/10
    y = np.random.normal(0,1)/10
    z = np.random.normal(0,1)/10

    Obj.append(tuple([8 + x,5 + y + posy,3 + z,0*math.pi/9,(8,13)]))

    x = np.random.normal(0,1)/10
    y = np.random.normal(0,1)/10
    z = np.random.normal(0,1)/10

    Obj.append(tuple([24 + x,6 + y + posy,3 + z,0*math.pi/9,(8,14)]))

    x = np.random.normal(0,1)/10
    y = np.random.normal(0,1)/10
    z = np.random.normal(0,1)/10

    Obj.append(tuple([15 + x,5 + y + posy,6 + z,0*math.pi/9,(8,14)]))

    x = np.random.normal(0,1)/10
    y = np.random.normal(0,1)/10
    z = np.random.normal(0,1)/10

    Obj.append(tuple([20 + x,6 + y + posy,7 + z,0*math.pi/9,(8,12)]))

    x = np.random.normal(0,1)/10
    y = np.random.normal(0,1)/10
    z = np.random.normal(0,1)/10

    Obj.append(tuple([16 + x,6 + y + posy,10 + z,0*math.pi/9,(8,13)]))

    x = np.random.normal(0,1)/10
    y = np.random.normal(0,1)/10
    z = np.random.normal(0,1)/10

    return ((16 + x,6.5 + y + posy,2 + z,0,(8,15)))
    

# この関数で始めにnum_particlesの数だけ状態を作成する
# TODO
# 考えられるばら積みを作成するアルゴリズムを完成させる
def init_particles_belief(num_particles, belief="uniform"):
    num_particles = 200
    particles = []
    
    for _ in range(num_particles):
        if belief == "uniform":
            rocktypes = []
            # for i in range(k):
                # rocktypes.append(RockType.random())
            # rocktypes = tuple(rocktypes)
            Obj = []
            Target = []
            Target = make_noise(Obj)
            s = State(Obj,Target)
        particles.append(s)
        particles_copy.append(s)
    init_belief = pomdp_py.Particles(particles)
    return init_belief

def get_particleType(particle,num_particles):

    noise = num_particles - len(particle)
    particle_pomdp = []

    for _ in range(noise):

        num = random.randint(0,len(particle)-1)
        particle_noise = [] 
        target = list(copy.deepcopy(particle[num].target))
        Obj = list(copy.deepcopy(particle[num].Obj))

        # print(target,Obj)
        target[0] = target[0] + np.random.normal(0,1)/10
        target[1] = target[1] + np.random.normal(0,1)/10
        target[2] = target[2] + np.random.normal(0,1)/10

        # print("targer particle is ....")
        # print(target)

        # particle_noise.append(particle[num][0] + np.random.normal(0,1)/10,particle[num][1] + np.random.normal(0,1)/10,particle[num][1] + np.random.normal(0,1)/10)
        # particle_pomdp[1] = particle[num][1] + np.random.normal(0,1)/10
        # particle_pomdp[2] = particle[num][1] + np.random.normal(0,1)/10
        s = State(Obj,target)
        # s.visualization()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        particle_pomdp.append(s)
    
    init_belief = pomdp_py.Particles(particle_pomdp)

    return init_belief

def main():

    # Obj = []
    # Target = set_Obj(Obj)
    # init_state = State(Obj,Target,5)
    obj_num = 6
    

    # make_initial_belief2

    # init_state.visualization(name = "init_state")
    # cv2.waitKey(0)
    

    # num_particles = 1000

    # particle = make_initial_belief2.init_particles_belief()
    # init_belief = get_particleType(particle,num_particles)
    # init_state = particle[0]
    
    # for i in range(len(particle)):
    #     particle[i].visualization(name = "start"+ str(i))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # print("len particles is ....")
    # print(len(particle))

    # # for i in range(len(particle)):
    # #     particle[i].visualization(name = "s"+ str(i))
    # #     cv2.waitKey(0)
    # #     cv2.destroyAllWindows()

    # # init_belief = init_particles_belief(num_particles, belief="uniform")
    # # init_belief = make_initial_belief2.init_particles_belief()
    # problem = SampleProblem(init_state, obj_num ,init_belief)

    # # print(problem.agent.belief)
    
    # print("*** Testing POMCP ***")
    # pomcp = pomdp_py.POMCP(max_depth=0, discount_factor=1,
    #                        num_sims=100, exploration_const=3,
    #                        rollout_policy=problem.agent.policy_model,
    #                        num_visits_init=0)
    
    # # print(problem.agent.belief.particles[0])

    # tt, ttd = test_planner(problem, pomcp, nsteps=3, discount=0.95)



    Obj = []
    Target = set_Obj(Obj)
    obj_num = 6
    init_state = State(Obj,Target,5)

    # init_state.visualization(name = "init_state")

    num_particles = 1000
    init_belief = init_particles_belief(num_particles, belief="uniform")

    problem = SampleProblem(init_state, obj_num ,init_belief)
    
    print("*** Testing POMCP ***")
    pomcp = pomdp_py.POMCP(max_depth=1, discount_factor=1,
                           num_sims=100, exploration_const=3,
                           rollout_policy=problem.agent.policy_model,
                           num_visits_init=0)

    tt, ttd = test_planner(problem, pomcp, nsteps=3, discount=0.95)

if __name__ == '__main__':
    
    main()
    
