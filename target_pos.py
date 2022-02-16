
from PIL.Image import NONE
from matplotlib.pyplot import flag
from numpy.lib.function_base import append
import pyrealsense2 as rs
import numpy as np
import cv2
import numpy as np
import math
import sys
import random
from motpy.core import Detection, Track, setup_logger
from motpy.testing_viz import draw_detection, draw_track
from motpy import Detection, MultiObjectTracker
import time
import depth_cap
# ========
# from pomdp_model2 import Observation 
import research
import copy
import itertools
import make_initial_belief2


# 作業空間の大きさを600×600の画像上で表現
w = 1000
h = 1000
sys.setrecursionlimit(30000)#反復回数の上限を10000回に変更
extend_condition = 2 #輝度に対する閾値(これが小さいと細かくセグメンテーションされる)
threshold = 1000 #面積に対する閾値
thresholdbig = 20000
N = 6
grid_size = 0.1
weight1 = 1
weight2 = 1 

tar_size1 = 15
tar_size2 = 8
carib_mat = np.array([[-0.999880,-0.011667, 0.010202 , 0.534507],
                    [-0.012136, 0.998813 , -0.047167, -0.021096],
                    [-0.009640, -0.047285, -0.998835, 1.481784],
                    [0.000000,0.000000,0.000000,1.000000]
                    ])
   
class Observation():
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
            self.tar_pos = target
            self.Obj_pos = Obj
            
            self.Obj_pos.sort(key = lambda x: x[2])
            # アンカーボックスのIDからサイズを出力
            self.size_list = []
            for i in range(len(self.Obj_pos)):
                self.size_list.append((self.Obj_pos[i][4][0],self.Obj_pos[i][4][1]))

            state = State(Obj,target)

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

    # 間違えて作った．またいつか使う日まで．．
    # 観測から中心位置を表示する関数

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
            # self.get_center(box,center_list,i)
            not_nomalized_ob.append(self.get_result(rect,box,N-1-i))
            self.im_bbox = cv2.drawContours(self.im_bbox,[box],0,(255*random.random(),255*random.random(),255*random.random()),2)

            whitePixels_diff = np.count_nonzero(img)
            self.obj_area.append(whitePixels_diff)
            self.occl_list.append(whitePixels_diff/whitePixels)
            correct_list.append(img)

        img = self.img_target
        whitePixels = np.count_nonzero(img)

        for i in range(N):
            img = cv2.subtract(img ,self.img_list[i])
        
        whitePixels_diff = np.count_nonzero(img)
        
        points = np.array(self.return_whitepoints(img))
        self.obj_area.append(whitePixels_diff)
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

                    # 整数型に丸めている．観測の離散化のため
                    # sample[i][j] = int(sample[i][j])
                    sample[i][j] = sample[i][j]
                else:
                    sample[i][j] = sample[i][j]

        # cv2.imshow("sample_image",sample_img)
        # cv2.waitKey(0)

        Obj,target = research.get_state(sample)

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


class State():
    # 状態を入力するクラス__init__で状態を初期化
    def __init__(self, Obj,target, N = 5):
        """
        (tuple): Obj:((x_1,y_1,z_1,yaw_1),.....(x_N,y_N,z_N,,yaw_N)

        """
        print("____________________________________")
        print(target)

        tar = list(target)
        self.Obj_prev = Obj
        
        self.Obj = list(Obj)
        height = []
        self.terminal = False

        if tar[4][0] < tar[4][1]:
            self.target_wid  = target[4][1]
            self.target_height  = target[4][0]
                
        else:
            self.target_wid  = target[4][0]
            self.target_height  = target[4][1] 
            tar[3] = tar[3] + math.pi/2
        
        self.target = tar

        # for i in range(len(self.Obj)):
        #     height.append(self.Obj[i][2])
        
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
        rotate = ((self.target[0]*20,-self.target[1]*20+h),(20*self.target_wid,20*self.target_height),-self.target[3]*180/math.pi) 
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

    def visualization(self,name = None):

        img = np.ones((w,h,3),np.uint8)*255

        # 対象物の位置を表示
        # print("pos is ...",self.target[0]*20,-self.target[1]*20+h)
        rotate = ((self.target[0]*20,-self.target[1]*20+h),(20*self.target_height,20*self.target_wid),-self.target[3]*180/math.pi) 
        img = self.rotatedRectangle(img,rotate,(0,0,255))
        id = 100

        # print("各物体の位置を表示します")
        # print(self.Obj)

        box_list = []
        # print(self.Obj)
        for i in range(len(self.Obj)):

            # 座標系を合わせるために x + 500 -y + 500  -θ することで対象物中心の座標系から表現 
            # x座標，y座標　20画素で1cmを表現する
            rotate = ((self.Obj[i][0]*20,-self.Obj[i][1]*20+h),(self.size_list[i][0]*20,self.size_list[i][1]*20),-self.Obj[i][3]*180/math.pi) 
            img = self.rotatedRectangle(img,rotate,(25*(i+1),25*(i+1),25*(i+1)))
        
        if name != None:
            cv2.imshow(name + 'state',img)

        if name == None:
            cv2.imshow('state',img)
        return img

    
    def occlu_check(self):
    
        img_list = []
        img =  np.zeros((w,h,3),np.uint8)
        self.img_target = np.zeros((w,h,3),np.uint8)
        # rotate = ((self.target[0]*20,- self.target[1]*20+h),(self.target_height*20,self.target_wid*20),-self.target[3]*180/math.pi)
        rotate = ((self.target[0]*20,- self.target[1]*20+h),(self.target_height*20,self.target_wid*20),-self.target[3]*180/math.pi)  
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
            # cv2.imshow("target",img)
            # cv2.waitKey(0)
            # print(whitePixels)
            

            # 上にある物体による遮蔽を計算
            for j in range(i):
                img = cv2.subtract(img ,img_list[N-i+j])
            
            whitePixels_diff = np.count_nonzero(img)
            self.visible_area.append(whitePixels_diff)
            if whitePixels != 0: 
                self.occl_list.append(whitePixels_diff/whitePixels)
            
            self.occl_list.append(1)

        self.occl_list.reverse()
        img = self.img_target
        whitePixels = np.count_nonzero(img)

        for i in range(len(self.Obj)):
            img = cv2.subtract(img ,img_list[i])
        
        whitePixels_diff = np.count_nonzero(img)
        #　self.occl_targetには見えている面積が格納されている
        self.picel_target =  whitePixels_diff
        self.occl_target = whitePixels_diff/whitePixels
        self.visible_area.append(whitePixels_diff)
        self.visible_area.reverse()

        # print("target_occusion is ...",self.occl_target)
        # print("other Object _occusion is ...",self.occl_list)
        # print("visible area is ...",self.visible_area)

        return

# 回転させた矩形をimgに描写してからリターン
def Rectangle(img, rotatedRect, color):
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


def make_rect(target,Obj,name = None):

    img = np.ones((w,h,3),np.uint8)*255

    # 対象物の位置を表示
    print("pos is ...",target[0]*20,-target[1]*20+h)
    rotate = ((target[0]*20,-target[1]*20+h),(20*8,20*15),-target[3]*180/math.pi) 
    img = Rectangle(img,rotate,(0,0,255))
    id = 100

    box_list = []
    print(Obj)
    for i in range(len(Obj)):

        # 座標系を合わせるために x + 500 -y + 500  -θ することで対象物中心の座標系から表現 
        # x座標，y座標　20画素で1cmを表現する
        
        rotate = ((Obj[i][0]*20,-Obj[i][1]*20+h),(Obj[i][4][0]*20,Obj[i][4][1]*20),-Obj[i][3]*180/math.pi) 
        img = Rectangle(img,rotate,(25*(i+1),25*(i+1),25*(i+1)))
    
    if name != None:
        cv2.imshow(name + 'state',img)

    if name == None:
        cv2.imshow('state',img)
    return img

def get_bigger_state1(obj,tar,num,flag):
    
    obj_list = copy.deepcopy(obj)
    tar_copy = list(tar)
    
    # print("+++++++++++++++++++++++++++++")

    # 回転前の4つの頂点を格納
    pt1_1 = list((int(tar[0]*20 + tar[4][0]*20 / 2), int(-tar[1]*20 + h + tar[4][1]*20 / 2)))
    pt2_1 = list((int(tar[0]*20 + tar[4][0]*20 / 2), int(-tar[1]*20 + h - tar[4][1]*20 / 2)))
    pt3_1 = list((int(tar[0]*20 - tar[4][0]*20 / 2), int(-tar[1]*20 + h - tar[4][1]*20 / 2)))
    pt4_1 = list((int(tar[0]*20 - tar[4][0]*20 / 2), int(-tar[1]*20 + h + tar[4][1]*20 / 2)))

    # print("four positon is ...")
    # print(pt1_1)
    # print(pt2_1)
    # print(pt3_1)
    # print(pt4_1)

    if flag == 2:
        height = tar_size1 - tar_copy[4][0]
        width = tar_size2 - tar_copy[4][1]
        tar_copy[3] = tar_copy[3] + math.pi/2
        
        
    
    if flag == 1:
        height = tar_size1 - tar_copy[4][0]
        width = tar_size2 - tar_copy[4][1]
        # tar_copy[3] = tar_copy[3] + math.pi/2
        

    a = width
    b = height

    # print("len is ...")
    # print(a)
    # print(b)

    # 伸ばしたことによって変わる座標を格納
    if num == 1: 
        # 右上の点を基準に長方形を大きくする
        pt1_1[1] -= b * 20

        pt3_1[0] -= a * 20
        
        pt4_1[0] -= a * 20
        pt4_1[1] -= b * 20

    if num == 2:
        # 右したの点を基準に長方形を大きくする
        pt1_1[1] -= b * 20
        pt1_1[0] += a * 20

        pt2_1[0] += a * 20
        
        pt4_1[1] -= b * 20

    if num == 3:
        # 左下の点を基準に長方形を大きくする
        pt1_1[0] += a * 20


        pt2_1[0] += a * 20
        pt2_1[1] += b * 20
        
        pt3_1[0] += b * 20

    # num== 0のときだけ調整しているので，他のところも調整する．TODO
    if num == 0:  
        # 左上の点を基準に長方形を大きくする
        pt4_1[0] -= a * 20

        pt2_1[1] += b * 20
        # pt2_1[1] -= -b * 20 + h

        pt3_1[0] -= a * 20
        pt3_1[1] += b * 20
    

    x = ( pt1_1[0] + pt2_1[0] + pt3_1[0] + pt4_1[0] ) /4
    y = ( pt1_1[1] + pt2_1[1] + pt3_1[1] + pt4_1[1] ) /4


    t = np.array([[np.cos(tar[3]),   -np.sin(tar[3]), x-x*np.cos(tar[3])+y*np.sin(tar[3])],
                    [np.sin(tar[3]), np.cos(tar[3]),  y-x*np.sin(tar[3])-y*np.cos(tar[3])],
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

    # print("four positon is ...")
    # print(pt1_2)
    # print(pt2_2)
    # print(pt3_2)
    # print(pt4_2)


    x2 = ( pt1_2[0] + pt2_2[0] + pt3_2[0] + pt4_2[0] ) /4
    y2 = ( pt1_2[1] + pt2_2[1] + pt3_2[1] + pt4_2[1] ) /4

    if tar_copy[4][0] + a > tar_copy[4][1] + b:
        wid = tar_copy[4][1] + a
        hei = tar_copy[4][0] + b
    else :
        wid = tar_copy[4][0] + b
        hei = tar_copy[4][1] + a

    belief_tar = (x2/20,-(y2 - h)/20,tar_copy[2],tar_copy[3] ,(hei,wid))

    # countcm大きな物体を作成してもしそれの一辺3cm以下だったらNoneを返す
    return belief_tar

# 物体の縦の大きさだけ変更したときに行けるかどうか確認するプログラム
def check_a(obj,tar,obs):

    obj_list = copy.deepcopy(obj) 
    

    # 回転前の4つの頂点を格納
    pt1_1 = list((int(obj[i][0]*20 + obj[i][4][0]*20 / 2), int(-obj[i][1]*20 + h + obj[i][4][1]*20 / 2)))
    pt2_1 = list((int(obj[i][0]*20 + obj[i][4][0]*20 / 2), int(-obj[i][1]*20 + h - obj[i][4][1]*20 / 2)))
    pt3_1 = list((int(obj[i][0]*20 - obj[i][4][0]*20 / 2), int(-obj[i][1]*20 + h - obj[i][4][1]*20 / 2)))
    pt4_1 = list((int(obj[i][0]*20 - obj[i][4][0]*20 / 2), int(-obj[i][1]*20 + h + obj[i][4][1]*20 / 2)))
    count  = 0

    # 伸ばしたことによって変わる座標を格納
    for x in range(2):
        if x == 0:
            # 右上の点を基準に長方形を大きくする
            pt3_1[0] -= a * 20       
            pt4_1[0] -= a * 20

        if x == 1:
            # 右したの点を基準に長方形を大きくする
            pt1_1[0] += a * 20
            pt2_1[0] += a * 20

        
        x = ( pt1_1[0] + pt2_1[0] + pt3_1[0] + pt4_1[0] ) /4
        y = ( pt1_1[1] + pt2_1[1] + pt3_1[1] + pt4_1[1] ) /4


        t = np.array([[np.cos(obj[i][3]),   -np.sin(obj[i][3]), x-x*np.cos(obj[i][3])+y*np.sin(obj[i][3])],
                        [np.sin(obj[i][3]), np.cos(obj[i][3]),  y-x*np.sin(obj[i][3])-y*np.cos(obj[i][3])],
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


        x2 = ( pt1_2[0] + pt2_2[0] + pt3_2[0] + pt4_2[0] ) /4
        y2 = ( pt1_2[1] + pt2_2[1] + pt3_2[1] + pt4_2[1] ) /4

        
        if obj[i][4][0] + a > obj[i][4][1]:
            wid = obj[i][4][1]
            hei = obj[i][4][0] + a
        else :
            wid = obj[i][4][0] + a
            hei = obj[i][4][1]

        belief_obj = (x/20,-(y2 - h)/20,obj[i][2],obj[i][3],(wid,hei))
        obj_list[i] = belief_obj
        # print("object_list is ...",obj_list)
        # s1 = State(obj_list,tar)
        # s1.visualization(name ="A")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        flag1 = True 

        for j in range(5):
            if abs(o.obj_area[j] - obs.obj_area[j]) > 1000:
                flag1 = False
        if flag1 == False:
            count += 1
    
    if count == 2:
        return False
    else: 
        return True

def check_b(obj,i,b,tar,obs):
    obj_list = copy.deepcopy(obj) 

    # 回転前の4つの頂点を格納
    pt1_1 = list((int(obj[i][0]*20 + obj[i][4][0]*20 / 2), int(-obj[i][1]*20 + h + obj[i][4][1]*20 / 2)))
    pt2_1 = list((int(obj[i][0]*20 + obj[i][4][0]*20 / 2), int(-obj[i][1]*20 + h - obj[i][4][1]*20 / 2)))
    pt3_1 = list((int(obj[i][0]*20 - obj[i][4][0]*20 / 2), int(-obj[i][1]*20 + h - obj[i][4][1]*20 / 2)))
    pt4_1 = list((int(obj[i][0]*20 - obj[i][4][0]*20 / 2), int(-obj[i][1]*20 + h + obj[i][4][1]*20 / 2)))
    count  = 0

    # 伸ばしたことによって変わる座標を格納
    for x in range(2):
        if x == 0:
            # 右上の点を基準に長方形を大きくする
            pt1_1[1] += 20*b
            pt4_1[1] += 20*b

        if x == 1:
            # 右したの点を基準に長方形を大きくする
            pt2_1[1] += 20*b
            pt3_1[1] += 20*b

        
        x = ( pt1_1[0] + pt2_1[0] + pt3_1[0] + pt4_1[0] ) /4
        y = ( pt1_1[1] + pt2_1[1] + pt3_1[1] + pt4_1[1] ) /4


        t = np.array([[np.cos(obj[i][3]),   -np.sin(obj[i][3]), x-x*np.cos(obj[i][3])+y*np.sin(obj[i][3])],
                        [np.sin(obj[i][3]), np.cos(obj[i][3]),  y-x*np.sin(obj[i][3])-y*np.cos(obj[i][3])],
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


        x2 = ( pt1_2[0] + pt2_2[0] + pt3_2[0] + pt4_2[0] ) /4
        y2 = ( pt1_2[1] + pt2_2[1] + pt3_2[1] + pt4_2[1] ) /4

        
        if obj[i][4][0] > obj[i][4][1] + b:
            hei = obj[i][4][0]
            wid = obj[i][4][1] + b
        else :
            wid = obj[i][4][0]
            hei = obj[i][4][1] + b

        belief_obj = (x/20,-(y2 - h)/20,obj[i][2],obj[i][3],(wid,hei))
        obj_list[i] = belief_obj
        # print("object_list is ...",obj_list)

        # s1 = State(obj_list,tar)
        # s1.visualization(name= "aAA")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        flag1 = True 

        for j in range(5):
            if abs(o.obj_area[j] - obs.obj_area[j]) >0:
                flag1 = False
        if flag1 == False:
            count += 1
    
    if count == 2:
        return False
    else: 
        return True
        
    
def get_obj_num(tar,obj2):

    s = State(obj2,tar)
    img = s.visualization(name ="3")
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    count = np.unique(img).size - 1
    
    if count == 6:
        return True
    if count != 6:
        return False   

def get_target_pose(s):

    tar = s.target
    obj = s.Obj

    pixcel_list = []
    pixcel = s.picel_target
    belief_tar = []


    # s.visualization(name = "s1_visual")
    # cv2.waitKey(0)

    # 1cm 大きくした時の矩形を書くプログラム
    # obj2に大きくした時の物体の位置を表示するようにプログラムを書く
    # 上にある物体iから順に探索 
    
    for x in range(4):

        target = get_bigger_state1(obj,tar,x,1)
        belief_tar.append(target)

        s1 =  State(obj,target)
        s1.occlu_check()
        s1.visualization(name=str(x))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pixcel_list.append(abs(pixcel - s1.picel_target))


    for x in range(4):

        target = get_bigger_state1(obj,tar,x,2)
        
        belief_tar.append(target)
        print("target is ....")
        print(target)
        s1 =  State(obj,target)
        s1.occlu_check()
        s1.visualization(name=str(x))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        pixcel_list.append(abs(pixcel - s1.picel_target))

    
    index = pixcel_list.index(min(pixcel_list))
    print("pixcel list is ....")
    print(pixcel_list)
    print("index is ....")
    print(index)

    return belief_tar[index]

   