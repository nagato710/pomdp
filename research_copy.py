import random
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

# from state_change import evaluate_prid


# 作業空間の大きさを600×600の画像上で表現
w = 600
h = 600
# kernel = GPy.kern.RBF(29,useGPU=True)
drag_len = 2
load_models = []

scale_tar = 1
scale_ob = 1 

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
        w1 = 1
        w2 = 1
        w3 = 1  
        w4 = 1
        w5 = 1
        w6 = 1
        state.occlu_check()
        next_state.occlu_check()
        if action.model_num < 16:
            # 物体の動きによる報酬
            for i in range(len(state.Obj)):
                reward -= w1*pow(state.Obj[i][0] - next_state.Obj[i][0],2) + w2*pow(state.Obj[i][1] - next_state.Obj[i][1],2) + w3*pow(state.Obj[i][2] - next_state.Obj[i][2],2)
            
            # オクルージョンの改善量による報酬
            reward = w4 * (next_state.occlu_target - state.occl_target)

            # graspabilityによる評価値をここに書く
            if state.visible_area[0] < 1000  or  next_state.visible_area[0] < 1000:
                reward -= 100
            if state.visible_area[0] >= 1000  and  next_state.visible_area[0] >= 1000:
                img1,img2 = state.get_depth_image()
                Wc1 = graspbytakasu.calcWt2(img1,img2)
                _,maxgrasp_s =graspbytakasu.graspability(img2,Wc1)

                img3,img4 = state.get_depth_image()
                Wc2 = graspbytakasu.calcWt2(img3,img4)
                _,maxgrasp_ns =graspbytakasu.graspability(img4,Wc2)

                reward -=  w6*(maxgrasp_ns - maxgrasp_s)  
            
        if action.model_num == 16:
            reward = w5/(math.e(-state.occl_target) + 1)
        
        return reward


class Action(pomdp_py.Action):
    def __init__(self,name):

        self.name = name
        numbers = re.sub(r'[^0-9]', '', name)
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

        print("変化後の状態を表示します")
        print(Target,Obj)
        
        return State(Obj,Target,5) 

class State(pomdp_py.State):
    # 状態を入力するクラス__init__で状態を初期化
    def __init__(self, Obj,target, N = 5):
        """
        (tuple): Obj:((x_1,y_1,z_1,yaw_1),.....(x_N,y_N,z_N,,yaw_N)

        """
        self.target = target
        self.Obj = list(Obj)
        height = []
        N = 5
        for i in range(N):
            height.append(self.Obj[i][2])
        
        self.Obj.sort(key = lambda x: x[2])
        # アンカーボックスのIDからサイズを出力
        self.size_list = []
        for i in range(len(self.Obj)):
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
        return "State(%s | %s | %s)" % (str(self.target), str(self.Obj))

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
        self.occl_target = whitePixels_diff/whitePixels
        self.visible_area.append(whitePixels_diff)
        self.visible_area.reverse()

        print("target_occusion is ...",self.occl_target)
        print("other Object _occusion is ...",self.occl_list)
        print("visible area is ...",self.visible_area)

        return

class SampleAction(Action):
    def __init__(self):
        super().__init__("sample")

class CheckAction(Action):
    def __init__(self, id):
        self.rock_id = id
        super().__init__("check-%d" % self.id)

class Observation(pomdp_py.Observation):
    def __init__(self, Obj,target, N = 5):
        """
        (tuple): Obj:((x_1,y_1,z_1,yaw_1),.....(x_N,y_N,z_N,,yaw_N)

        """
        self.tar_pos = target
        self.Obj_pos = Obj
        
        self.Obj_pos.sort(key = lambda x: x[2])
        # アンカーボックスのIDからサイズを出力
        self.size_list = []
        for i in range(len(self.Obj_pos)):
            self.size_list.append((self.Obj_pos[i][4][0],self.Obj_pos[i][4][1]))
    
    def __hash__(self):
        return hash((self.tar_pos, self.Obj_pos))

    def __eq__(self, other):
        if isinstance(other, State):
            return self.tar_pos == other.tar_pos\
                and self.Obj_pos == other.Obj_pos
        else:
            return False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "State(%s | %s | %s)" % (str(self.tar_pos), str(self.Obj_pos))


class ObservationModel(pomdp_py.ObservationModel):
    # img_list：各物体の二値画像
    # state：この状態に対しての観測
    # occl_list：各物体のオクルージョン比率
    # im_bbox：各物体のBBOX
    # correct_ob：観測集合を出力（計算上）


    def __init__(self,state,N = 5):

        self.img_list = [] 
        self.state = state
        img =  np.zeros((w,h,3),np.uint8)
        self.img_target = np.zeros((w,h,3),np.uint8)
        rotate = ((state.target[0]*20,-state.target[1]*20+h),(8*20,15*20),-state.target[3]*180/math.pi) 
        img2 = cv2.cvtColor(state.rotatedRectangle(img,rotate,(255,255,255)),cv2.COLOR_BGR2GRAY)
        self.img_target = img2

        for i in range(N):
            img =  np.zeros((w,h,3),np.uint8)
            rotate = ((state.Obj[i][0]*20,-state.Obj[i][1]*20+h),(state.size_list[i][0]*20,state.size_list[i][1]*20),-state.Obj[i][3]*180/math.pi) 
            img2 = cv2.cvtColor(state.rotatedRectangle(img,rotate,(255,255,255)),cv2.COLOR_BGR2GRAY)
            self.img_list.append(img2)
        
    # ある状態におけるそれぞれのバウンディングボックスを出力します
    def check_observation(self,N = 5):

        for i in range(N):
            cv2.imshow("check_state",self.img_list[i])
            cv2.waitKey(0)

    # pointsに白黒画像の白色の部分を格納
    def return_whitepoints(self,img):
        width = img.shape[0]
        height = img.shape[1]
        points = []
        for i in range(width):
            for j in range(height):
                if img[i,j] != 0:
                    points.append([j,i])
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
            not_normalized_ob[i] = list(not_normalized_ob[i])

        
        # print(not_normalized_ob)


        for i in range(len(not_normalized_ob)):
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
            not_normalized_ob[i][3] = not_normalized_ob[i][3]*math.pi/180



        return not_normalized_ob
    
    def normalized_observ2(self,not_normalized_ob):

        # not_normalized_ob = list(not_normalized_ob1)
        
        for i in range(len(not_normalized_ob)):
            not_normalized_ob[i] = list(not_normalized_ob[i])
        
        for i in range(len(not_normalized_ob)):
            not_normalized_ob[i][3] =  not_normalized_ob[i][3]*180/math.pi

        for i in range(len(not_normalized_ob)):
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
            not_normalized_ob[i][3] = not_normalized_ob[i][3]*math.pi/180



        return not_normalized_ob
    
    # ある状態の正解の観測結果を出力します
    def correct_observ(self,N = 5):

        correct_list = []
        # それぞれのオブジェクトのオクルージョン比率を格納
        self.occl_list = []

        self.im_bbox =  np.zeros((w,h,3),np.uint8)
        self.correct_ob = []
        not_nomalized_ob =[]
        
        
        
        for i in range(N):

            img = self.img_list[N-1-i]
            whitePixels = np.count_nonzero(img)

            # 上にある物体による遮蔽を計算
            for j in range(i):
                img = cv2.subtract(img ,self.img_list[N-i+j])
            
            # pointsに白黒画像の白色の部分を格納
            points = np.array(self.return_whitepoints(img))
            # rectにはBBOXの左上の座標，wとh,回転角度を入力
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            # boxの角の点４つを入力する
            box = np.int0(box)
            not_nomalized_ob.append(self.get_result(rect,box,N-1-i))
            self.im_bbox = cv2.drawContours(self.im_bbox,[box],0,(255*random.random(),255*random.random(),255*random.random()),2)

            whitePixels_diff = np.count_nonzero(img)
            self.occl_list.append(whitePixels_diff/whitePixels)
            correct_list.append(img)


        img = self.img_target
        whitePixels = np.count_nonzero(img)

        for i in range(N):
            img = cv2.subtract(img ,self.img_list[i])
        
        whitePixels_diff = np.count_nonzero(img)
        
        points = np.array(self.return_whitepoints(img))
        # rectにはBBOXの左上の座標，wとh,回転角度を入力
        
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
        w3 = 0.05
        theta = 1
        sample = []
        sample_img =  np.zeros((self.im_bbox.shape[0],self.im_bbox.shape[1],3),np.uint8)
        N = len(self.state.Obj)

          
        x = w1 * np.random.normal(0,1-self.occl_target)
        y = w1 * np.random.normal(0,1-self.occl_target)
        z = w1 * np.random.normal(0,1-self.occl_target)

        width = w2 * np.random.normal(0,1-self.occl_target)
        height = w2 * np.random.normal(0,1-self.occl_target)

        theta = w3 * np.random.normal(0,1-self.occl_target)

        sample.append(( self.state.target[0] + x , self.state.target[1] + y , self.state.target[2] + z ,self.state.target[3] + theta,self.state.target[4][0] ,self.state.target[4][1]))



        for i in range(len(self.state.Obj)):
        
            x = w1 * np.random.normal(0,0)
            y = w1 * np.random.normal(0,0)
            z = w1 * np.random.normal(0,0)

            width = w2 * np.random.normal(0,0)
            height = w2 * np.random.normal(0,0)

            theta = w3 * np.random.normal(0,0)
            sample.append(( self.correct_ob[i][0] + x , self.correct_ob[i][1] + y , self.correct_ob[i][2] + z ,self.correct_ob[i][3] + theta,self.correct_ob[i][4] + width,self.correct_ob[i][5] + height ))
            # print("角度を表示します")
            # print(self.correct_ob[i][3])
        rect = []
        for i in range(len(sample)):
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
            sample[i] = list(sample[i])
            for j in range(len(sample[i])):
                if j != 3:
                    sample[i][j] = int(sample[i][j])
                else:
                    sample[i][j] = sample[i][j]

        cv2.imshow("sample_image",sample_img)
        cv2.waitKey(0)

        return Observation(sample)
    
    def sample(self, next_state, action, argmax=False):
        x =  ObservationModel(next_state)
        x.correct_observ(5)
        obs = x.sample_observ()

        return obs

def set_Obj(Obj):

    Obj.append(tuple([8,5,3,0*math.pi/9,(8,13)]))
    Obj.append(tuple([24,6,3,0*math.pi/9,(8,14)]))
    Obj.append(tuple([15,5,6,0*math.pi/9,(8,14)]))
    Obj.append(tuple([20,6,7,0*math.pi/9,(8,12)]))
    Obj.append(tuple([16,6,10,0*math.pi/9,(8,13)]))

    return ((16,6.5,2,0,(8,15)))

class PolicyModel(pomdp_py.RolloutPolicy):
    """Simple policy model according to problem description."""
    def __init__(self, n):
        # check_actions = set({CheckAction(rock_id) for rock_id in range(k)})
        # self._move_actions = {MoveEast, MoveWest, MoveNorth, MoveSouth}
        # self._other_actions = {SampleAction()} | check_actions
        # self._all_actions = self._move_actions | self._other_actions
        self._n = n

    # def argmax(self, state, normalized=False, **kwargs):
    #     """Returns the most likely reward"""
    #     raise NotImplementedError

    def get_all_actions(self, state):

        # 多分，ここに状態を入力して行動どの行動を実行可能であるかを出力する関数を作る
        state.occlu_check()
        can_push = []
        action_id_list = []
        action_list = []
        drag_dir = 16
        img1,img2 = state.get_depth_image()
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
            action_list.append("a81")

        for i in range (6):
            if state.visible_area[i] > 10000:
                can_push.append(i)
   
        for j in range(len(can_push)):
            for i in range (drag_dir):
                action_id_list.append(i*5 + can_push[j])
        
        print("we can carry out action that number is",action_id_list)
       
        for i in range(len(action_id_list)):
            action_list.append(Action("a" + str(action_id_list[i])))

        return action_list

    def rollout(self, state, history=None):
        # random.sample(self.get_all_actions(state=state), 1)[0]は行動の名前がランダムに出力されている.
        # get_all_actionsで可能なすべてのアクションに限定し，その中からのランダムな行動を選択している
        return random.sample(self.get_all_actions(state=state), 1)[0]


class TransitionModel(pomdp_py.TransitionModel):
    def __init__(self,n = 5):
        
        self.obj_num = n
    
    # 状態，行動，アクションから，次の行動を出力する関数
    def sample(self,state,action):
        
        spre = []
        rotate_list = []

        action.push_pos(state)
        print("変化前の状態を表示します")
        print(state.target,state.Obj)

        # for i in range(2):
        #     spre.append(action.drag[i]*drag_len/10)

        for i in range(3):
            spre.append(action.push[i]/10/scale_ob)

        for i in range(4):
            if i != 3:
                spre.append(state.target[i]/10/scale_tar)
            else:
                rotate_list.append(state.target[3])
        
        for i in range(self.obj_num):
            for j in range(4):
                if j != 3:
                    spre.append(state.Obj[i][j]/10/scale_ob)
                else:
                    rotate_list.append(state.Obj[i][j])
            
        test = []
        test.append(spre)
        
        print("type_check")
        print(test)
        snext = tNN.eval(test,action.model_num)
        print("type_check")
        print(snext)
        state_nex =change_classstate(snext,rotate_list,2,0,state.size_list)

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

def test_planner(sample, planner, nsteps=3, discount=0.95):
    
    gamma = 1.0
    total_reward = 0
    total_discounted_reward = 0

    for i in range(nsteps):
        print("==== Step %d ====" % (i+1))
        # ここにagentを入力するだけ？
        # 行動決定！
        action = planner.plan(sample.agent)
        # pomdp_py.visual.visualize_pouct_search_tree(rocksample.agent.tree,
        #                                             max_depth=5, anonymize=False)

        # 真の状態を出力
        true_state = copy.deepcopy(sample.env.state)
        # 真の状態が変化
        env_reward = sample.env.state_transition(action, execute=True)
        true_next_state = copy.deepcopy(sample.env.state)

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

        if sample.in_exit_area(sample.env.state.position):
            break

    return total_reward, total_discounted_reward

def make_noise(Obj):

    
    x = np.random.normal(0,1)
    y = np.random.normal(0,1)
    z = np.random.normal(0,1)

    Obj.append(tuple([8 + x,5 + y,3 + z,0*math.pi/9,(8,13)]))

    x = np.random.normal(0,1)
    y = np.random.normal(0,1)
    z = np.random.normal(0,1)

    Obj.append(tuple([24 + x,6 + y,3 + z,0*math.pi/9,(8,14)]))

    x = np.random.normal(0,1)
    y = np.random.normal(0,1)
    z = np.random.normal(0,1)

    Obj.append(tuple([15 + x,5 + y,6 + z,0*math.pi/9,(8,14)]))

    x = np.random.normal(0,1)
    y = np.random.normal(0,1)
    z = np.random.normal(0,1)

    Obj.append(tuple([20 + x,6 + y,7 + z,0*math.pi/9,(8,12)]))

    x = np.random.normal(0,1)
    y = np.random.normal(0,1)
    z = np.random.normal(0,1)

    Obj.append(tuple([16 + x,6 + y,10 + z,0*math.pi/9,(8,13)]))

    x = np.random.normal(0,1)
    y = np.random.normal(0,1)
    z = np.random.normal(0,1)

    return ((16 + x,6.5 + y,2 + z,0,(8,15)))
    

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
        particles.append(State(Obj,Target))
    init_belief = pomdp_py.Particles(particles)
    return init_belief


def main():
    Obj = []
    Target = set_Obj(Obj)
    obj_num = 6
    init_state = State(Obj,Target,5)

    num_particles = 200
    init_belief = init_particles_belief(num_particles, belief="uniform")

    problem = SampleProblem(init_state, obj_num ,init_belief)
    

    print("*** Testing POMCP ***")
    pomcp = pomdp_py.POMCP(max_depth=20, discount_factor=0.95,
                           num_sims=10000, exploration_const=20,
                           rollout_policy=problem.agent.policy_model,
                           num_visits_init=1)

    tt, ttd = test_planner(problem, pomcp, nsteps=100, discount=0.95)



if __name__ == '__main__':

    main()
    