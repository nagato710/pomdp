import random
import cv2
from matplotlib.pyplot import imshow
import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt


# 作業空間の大きさを600×600の画像上で表現
w = 600
h = 600

class State():
    # 状態を入力するクラス__init__で状態を初期化
    def __init__(self, Obj,target, N = 5):
        """
        (tuple): Obj:((x_1,y_1,z_1,yaw_1),.....(x_N,y_N,z_N,,yaw_N)

        """
        self.target = Target
        self.Obj = Obj
        height = []

        
        for i in range(N):
           height.append(Obj[i][2])
        
        # 高さが低い物体から順に入れていく
        # height.sort()
        
        # for i in range(N):
        #     for j in range(N):
        #         if height[i] == Obj[j][2]:
        #             self.Obj.append(Obj[j])
        self.Obj.sort(key = lambda x: x[2])
        # アンカーボックスのIDからサイズを出力
        self.size_list = []
        for i in range(len(self.Obj)):
            
            self.size_list.append((self.Obj[i][4][0]*20,self.Obj[i][4][1]*20))
            


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


    def visualization(self):

        img = np.zeros((w,h,3),np.uint8)

        # 対象物の位置を表示
        rotate = ((w/2,h/2),(20*8,20*15),0) 
        img = self.rotatedRectangle(img,rotate,(0,0,255))


        box_list = []
        print(self.Obj)
        for i in range(len(self.Obj)):

            # 座標系を合わせるために x + 500 -y + 500  -θ することで対象物中心の座標系から表現 
            # x座標，y座標　20画素で1cmを表現する
            rotate = ((self.Obj[i][0]*20+w/2,-self.Obj[i][1]*20+h/2),(self.size_list[i][0],self.size_list[i][1]),-self.Obj[i][3]*180/math.pi) 
            img = self.rotatedRectangle(img,rotate,(255*random.random(),255*random.random(),255*random.random()))
            cv2.imshow('state',img)
            cv2.waitKey(0)

class Observation:
    # img_list：各物体の二値画像
    # state：この状態に対しての観測
    # occl_list：各物体のオクルージョン比率
    # im_bbox：各物体のBBOX
    # correct_ob：観測集合を出力（計算上）
    # 

    def __init__(self,state,N = 5):

        self.img_list = [] 
        self.state = state
        img =  np.zeros((w,h,3),np.uint8)
        self.img_target = np.zeros((w,h,3),np.uint8)
        rotate = ((state.target[0][0]*20+w/2,-state.target[0][1]*20+h/2),(8*20,15*20),-state.target[0][3]*180/math.pi) 
        img2 = cv2.cvtColor(state.rotatedRectangle(img,rotate,(255,255,255)),cv2.COLOR_BGR2GRAY)
        self.img_target = img2

        for i in range(N):
            img =  np.zeros((w,h,3),np.uint8)
            rotate = ((state.Obj[i][0]*20+w/2,-state.Obj[i][1]*20+h/2),(state.size_list[i][0],state.size_list[i][1]),-state.Obj[i][3]*180/math.pi) 
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
 
        x = ((box[0][0] + box[1][0] + box[2][0] + box[3][0])/4 - w/2)/20
        y = -((box[0][1] + box[1][1] + box[2][1] + box[3][1])/4 - h/2)/20
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
        self.im_bbox = cv2.drawContours(self.im_bbox,[box],0,(255*random.random(),255*random.random(),255*random.random()),2)
        self.occl_target = whitePixels_diff/whitePixels
        

        
        self.correct_ob = self.normalized_observ(not_nomalized_ob)
        self.correct_ob.reverse()
        print("正解の観測データを出力します")
        print(self.correct_ob)
        
        
        
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
        print(self.state.target)

        sample.append(( self.state.target[0][0] + x , self.state.target[0][1] + y , self.state.target[0][2] + z ,self.state.target[0][3] + theta,self.state.target[0][4][0] ,self.state.target[0][4][1]))



        for i in range(len(self.state.Obj)):
        
            
            x = w1 * np.random.normal(0,1-occ[i])
            y = w1 * np.random.normal(0,1-occ[i])
            z = w1 * np.random.normal(0,1-occ[i])

            width = w2 * np.random.normal(0,1-occ[i])
            height = w2 * np.random.normal(0,1-occ[i])

            theta = w3 * np.random.normal(0,1-occ[i])

            sample.append(( self.correct_ob[i][0] + x , self.correct_ob[i][1] + y , self.correct_ob[i][2] + z ,self.correct_ob[i][3] + theta,self.correct_ob[i][4] + width,self.correct_ob[i][5] + height ))

        rect = []

        for i in range(len(sample)):
            
            (w,h) = (sample[i][4]*20,sample[i][5]*20)
            (x,y) = (sample[i][0]*20 + self.im_bbox.shape[0]/2,-sample[i][1]*20 + self.im_bbox.shape[1]/2)
            theta = -sample[i][3]*180/math.pi
            rect.append(((x,y),(w,h),theta))
            box = cv2.boxPoints(rect[i])
            # boxの角の点４つを入力する
            box = np.int0(box)
            sample_img = cv2.add(sample_img,cv2.drawContours(sample_img,[box],0,(255*random.random(),255*random.random(),255*random.random()),2))


        

        for i in range(len(sample)):
            sample[i] = list(sample[i])
            for j in range(len(sample[i])):
                sample[i][j] = int(sample[i][j])

        print("ランダムにずらした観測結果を描写します")
        print(sample)
                




        cv2.imshow("sample_image",sample_img)
        cv2.waitKey(0)

        return sample 

def set_Obj(Obj,Target):

    Target.append(tuple([0,0,0,0,(8,15)]))
    
    Obj.append(tuple([-2,-2,4,0*math.pi/9,(8,13)]))
    Obj.append(tuple([8,0,3,0*math.pi/9,(8,14)]))
    Obj.append(tuple([0,-5,5,0*math.pi/9,(8,14)]))
    Obj.append(tuple([-9,0,3,0*math.pi/9,(8,12)]))
    Obj.append(tuple([-6,-5,4,0*math.pi/9,(8,13)]))


if __name__ == '__main__':

    Obj = []
    Target = []
    set_Obj(Obj,Target)

    s1 = State(Obj,Target,5)
    print("ここで状態を出力します")
    print((s1.target,s1.Obj))
    s1.visualization()

    

    o1 = Observation(s1,5)
    o1.correct_observ(5)
    o1.sample_observ()
    o1.sample_observ()
    cv2.imshow("press_bottan",o1.im_bbox)
    cv2.waitKey(0)
    print("a")
    
# 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