import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    # Remove "Russia" from MyList 
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from PIL.Image import NONE
from matplotlib.pyplot import flag
from numpy.lib.function_base import append
import pyrealsense2 as rs
import numpy as np
import cv2
import numpy as np
import math
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
import target_pos

# 作業空間の大きさを600×600の画像上で表現
w = 1000
h = 1000
sys.setrecursionlimit(30000)#反復回数の上限を10000回に変更
extend_condition = 2 #輝度に対する閾値(これが小さいと細かくセグメンテーションされる)
threshold = 1000 #面積に対する閾値
thresholdbig = 20000
N = 6
grid_size = 1
weight1 = 1
weight2 = 1 

sliding = np.array([0,0,0])

carib_mat = np.array([[-0.999880,-0.011667, 0.010202 , 0.534507],
                    [-0.012136, 0.998813 , -0.047167, -0.021096],
                    [-0.009640, -0.047285, -0.998835, 1.481784],
                    [0.000000,0.000000,0.000000,1.000000]
                    ])

rotate_mat = np.array([[0,-1,0],
                      [1,0,0],
                      [0,0,1],
                    ])
class Belief():

    def __init__(self):
        self.height = 480*weight1
        self.width = 640*weight2

    #白黒画像をカラー画像に変換
    def gray2color(self,gray_img):
        height, width = gray_img.shape
        color_img = np.zeros((height, width, 3)) #色情報を3次元にして作成
        for i in range(0, height):
            for j in range(0, width):
                luminosity = gray_img[i][j]
                color_img[i][j] = [luminosity, luminosity, luminosity]
        return color_img

    #近傍点を取得
    def get_neighbour(self,poi):
        neighbour = []
        x0, y0 = poi[0], poi[1]
        for i in range(-1, 2):
            for j in range(-1, 2):
                #poi(注目点)の座標は格納しない
                if (i, j) == (0, 0):
                    continue
                x, y = x0 + i, y0 + j
                if x < self.height and y < self.width and x > 0 and y > 0:
                    if self.img[x][y] > 0:#画像サイズ内かつ画素値が0より大きい
                        neighbour.append([x, y])#neighborに格納
        return neighbour

    #輝度値の比較
    def compare_luminosity(self,neighbour, poi):
        region = []
        poi_luminosity = self.img_copy[poi[0]][poi[1]]#poi(注目点)の画素値をコピーした画像から取得(オリジナル画像は画素値は消されている)
        for xy in neighbour:
            x, y = xy[0], xy[1]
            neighbour_luminosity = self.img_copy[x][y]#近傍点の画素値をコピーした画像から取得
            if np.abs(int(poi_luminosity) - int(neighbour_luminosity)) < extend_condition:#画素値の差分と輝度値に対する閾値との比較
                region.append([x, y])#regionに近傍点の座標を格納
                self.img[x][y] = 0#格納した点の画素値を消す
        return region

    #領域成長
    def region_growing(self,prepartial_region, region):
        if len(prepartial_region) < 0:#prepartial_regionに何も入っていなければ終了
            return prepartial_region, region
        #prepartial_regionに格納されている座標を取り出す
        poi = prepartial_region[0]
        neighbour = self.get_neighbour(poi)#poiの近傍点を取得する
        if len(neighbour) == 0:#neighborに何も入っていなければ次のpoiに移る
            prepartial_region.remove(poi)
            return prepartial_region, region
        partial_region = self.compare_luminosity(neighbour, poi)#近傍点と注目点との画素値を比較
        prepartial_region.remove(poi)
        prepartial_region.extend(partial_region)
        region.extend(partial_region)#得られたpartial_regionをregionに加える
        return prepartial_region, region

    #視覚化
    def visualize_result(self,region_list):
        color_img = self.gray2color(self.img)#画像をカラー画像に変更する
        rect = []
        r_rot = []
        for i in range(0, len(region_list)):
            blue = random.random()#青色を0〜1の中でランダムに設定
            green = random.random()#緑色を0〜1の中でランダムに設定
            red = random.random()#赤色を0〜1の中でランダムに設定
            matforec = []
            for xy in region_list[i]:
                x, y = xy[0], xy[1]
                matforec.append(xy)
                color_img[x][y] = [blue, green, red]#各領域ごとに異なる色を指定
        #画像の表示
            retval = cv2.boundingRect(np.array(matforec))
            # Define boxsize
            a = np.array(matforec)*1
            rect_rot = cv2.minAreaRect( a[:,[1,0] ] )
            rect.append(retval)
            r_rot.append(rect_rot)

        # print(rect)
        color_img = cv2.resize(color_img,dsize=(int(color_img.shape[1]*1),int(color_img.shape[0]*1)))
        cv2.imshow('image', color_img)
        cv2.waitKey(0)
        return rect , r_rot


    #seedを探索
    def search_seed(self):
        nozeros = np.where(self.img > 0)#画素値が0出ない画素を探索
        region_list = []
        while len(nozeros[0]) > 0:
            seed = [[nozeros[0][0], nozeros[1][0]]]#nozerosの座標を適当に取り出す
            self.img[seed[0][0]][seed[0][1]] = 0#取り出した画素の画素値を消す
            region = []
            while len(seed) > 0:
                seed, region = self.region_growing(seed, region)#seedに対して領域成長を行う
            nozeros = np.where(self.img > 0)#画素値が0出ない画素を探索
            if len(region) > threshold:#面積が閾値以上であれば格納
                if len(region) < thresholdbig:
                    region_list.append(region)
        rect , r_rot= self.visualize_result(region_list)#視覚化する
        return (region_list, rect , r_rot)

    def make_detect_class(self,rect):
        out_detections = []
        for i in range(len(rect)):
            # if ( (abs((rect[i][1] + rect[i][3])*2 - rect[i][1]*2) < 400)):
            #     print(i)
            # Define boxsize 
            out_detections.append(Detection(box=[rect[i][1]*1,rect[i][0]*1 , (rect[i][1] + rect[i][3])*1, (rect[i][0] + rect[i][2])*1]))

        return out_detections

    def get_center(self,box):
        
        return  ((box[0][0] + box[1][0] + box[2][0] + box[3][0])/4 , (box[0][1] + box[1][1] + box[2][1] + box[3][1])/4)

    def check_color(self,box , color):

        x = (box[0][0] + box[1][0] + box[2][0] + box[3][0])/4
        y = (box[0][1] + box[1][1] + box[2][1] + box[3][1])/4
        
        
        minl =  math.sqrt(math.pow( abs(x - color[0][1][0] ),2) +(math.pow( abs(y - color[0][1][1] ),2)))
        flag = 0

        for i in range( len(color) -1):

            if( minl > math.sqrt(math.pow( (x - color[i+1][1][0] ),2) +(math.pow( (y - color[i+1][1][1] ),2)))):
                    minl = math.sqrt(math.pow( (x - color[i+1][1][0] ),2) + (math.pow( (y - color[i+1][1][1] ),2)))
                    flag = i+1
        
        return flag

    # BBOXのサイズを出力する関数
    def get_box_point(self,point_list,depth_frame,color_intr,index,center_list_im):
        
        # print("取得する三次元の画素は．．．") 
        # print(point_list[index][0][0], point_list[index][0][1])

        # print(center_list_im[index][0],center_list_im[index][1])
        point_list2 = list(point_list)

        

        dist = depth_frame.get_distance(point_list[index][0][0], point_list[index][0][1])
        point_I = rs.rs2_deproject_pixel_to_point(color_intr , [point_list[index][0][0],point_list[index][0][1]], dist)
        # point_I[1] = -point_I[1]

        if point_I[0] == 0 and point_I[1] == 0 and point_I[2] == 0:
            x = center_list_im[index][1] -  point_list[index][0][0]
            y = center_list_im[index][0] -  point_list[index][0][1]
            root = math.sqrt(pow(x,2) + pow(y,2))
            x = x/root
            y = y/root
            count = 1
            while(1):
                a = int(point_list[index][0][0] + x*count)
                b = int(point_list[index][0][1] + y*count)
                dist = depth_frame.get_distance(a, b)
                point_I = rs.rs2_deproject_pixel_to_point(color_intr , [a,b], dist)
                # point_I[1] = -point_I[1]
                count += 1
                if point_I[0] != 0 and point_I[1] != 0 and point_I[2] != 0:
                    break
      
        point_I = np.append(point_I,1)
        point3d_1 = np.dot(carib_mat,point_I)


        

        # print("取得する三次元の画素は．．．") 
        # print(point_list[index][1][0], point_list[index][1][1])
 
        dist2 = depth_frame.get_distance(point_list[index][1][0], point_list[index][1][1])
        point_II = rs.rs2_deproject_pixel_to_point(color_intr , [point_list[index][1][0],point_list[index][1][1]], dist2)
        # point_II[1] = -point_II[1]

        if point_II[0] == 0 and point_II[1] == 0 and point_II[2] == 0:
            x = center_list_im[index][1] -  point_list[index][1][0]
            y = center_list_im[index][0] -  point_list[index][1][1]
            root = math.sqrt(pow(x,2) + pow(y,2))
            x = x/root
            y = y/root
            count = 1
            while(1):
                a = int(point_list[index][1][0] + x*count)
                b = int(point_list[index][1][1] + y*count)
                dist2 = depth_frame.get_distance(a, b)
                point_II = rs.rs2_deproject_pixel_to_point(color_intr , [a,b], dist2)
                # point_II[1] = -point_II[1]
                count += 1
               
                if point_II[0] != 0 and point_II[1] != 0 and point_II[2] != 0:
                    break
        point_II = np.append(point_II,1)
        point3d_2 = np.dot(carib_mat,point_II)
        print(point_II)
        print("point2")
        print(point3d_2)
        
        dist3 = depth_frame.get_distance(point_list[index][2][0], point_list[index][2][1])
        point_III = rs.rs2_deproject_pixel_to_point(color_intr , [point_list[index][2][0],point_list[index][2][1]], dist3)
        # point_III[1] = -point_III[1]
        if point_III[0] == 0 and point_III[1] == 0 and point_III[2] == 0:
            x = center_list_im[index][1] -  point_list[index][2][0]
            y = center_list_im[index][0] -  point_list[index][2][1]
            root = math.sqrt(pow(x,2) + pow(y,2))
            x = x/root
            y = y/root
            count = 1
            # print("gaso")
            # print(x,y)
            while(1):
                a = int(point_list[index][2][0] + x*count)
                b = int(point_list[index][2][1] + y*count)
                dist3 = depth_frame.get_distance(a, b)
                point_III = rs.rs2_deproject_pixel_to_point(color_intr , [a,b], dist3)
                # point_III[1] = -point_III[1]
                count += 1
                if count == 2: 
                    print("gaso")
                    print(a,b)
                if point_III[0] != 0 and point_III[1] != 0 and point_III[2] != 0:
                    break


        point_III = np.append(point_III,1)
        point3d_3 = np.dot(carib_mat,point_III)
        print(point_III)
        print("point3")
        print(point3d_3)
 
        lenx = math.sqrt( pow( (point3d_1[0] - point3d_2[0]) , 2) + pow( (point3d_1[1] - point3d_2[1]) , 2) )
        leny = math.sqrt( pow( (point3d_2[0] - point3d_3[0]) , 2) + pow( (point3d_2[1] - point3d_3[1]) , 2) )
        print((lenx*100,leny*100))

        return (lenx*100,leny*100)
    
    def get_box_center(self,point_list,depth_frame,color_intr,index,center_list_im2):

        
        dist = depth_frame.get_distance(int(point_list[index][1]), int(point_list[index][0]))
        point = rs.rs2_deproject_pixel_to_point(color_intr , [int(point_list[index][1]), int(point_list[index][0])], dist)
        # point[1] = -point[1]
        # if point_I[0] != 0 and point_I[1] != 0 and point_I[2] != 0:
        #     break
        if point[0] == 0 and point[1] == 0 and point[2] == 0:
            # print("No pointcloud")
            x = point_list[index][0] - center_list_im2[index][1]
            y = point_list[index][1] - center_list_im2[index][0] 
            root = math.sqrt(pow(x,2) + pow(y,2))
            x = x/root
            y = y/root
            count = 1
            while(1):
                a = int(point_list[index][0] + x*count)
                b = int(point_list[index][1] + y*count)
                dist = depth_frame.get_distance(b,a)
                point = rs.rs2_deproject_pixel_to_point(color_intr , [b,a], dist)
                # point[1] = -point[1]
                count += 1
                if point[0] != 0 and point[1] != 0 and point[2] != 0:
                    break
        
        point = np.append(point,1)
        # print(point)
        point = np.dot(carib_mat,point)
        
        # dist = depth_frame.get_distance(int(point_list[index][1]), int(point_list[index][0]))
        # point = rs.rs2_deproject_pixel_to_point(color_intr , [point_list[index][1],point_list[index][0]], dist)

        return(point)

    def get_box_center2(self,point_list,depth_frame,color_intr,index,center_list_im2):

        print("pointlist is .....")
        print(int(point_list[index][1]), int(point_list[index][0]))
        
        dist = depth_frame.get_distance(int(point_list[index][0]), int(point_list[index][1]))
        point = rs.rs2_deproject_pixel_to_point(color_intr , [int(point_list[index][0]), int(point_list[index][1])], dist)
        # point[1] = -point[1]
        # if point_I[0] != 0 and point_I[1] != 0 and point_I[2] != 0:
        #     break
        if point[0] == 0 and point[1] == 0 and point[2] == 0:
            if index == 0:  
                print("______________________________")
                print(point_list)
                print("No pointcloud")
            x = point_list[index][0] - center_list_im2[index][0]
            y = point_list[index][1] - center_list_im2[index][1] 
            root = math.sqrt(pow(x,2) + pow(y,2))
            x = x/root
            y = y/root
            count = 1
            while(1):
                a = int(point_list[index][0] + x*count)
                b = int(point_list[index][1] + y*count)
                dist = depth_frame.get_distance(a,b)
                if index == 0:  
                    print("______________________________")
                    # print(point_list)
                    print(a,b)
                point = rs.rs2_deproject_pixel_to_point(color_intr , [a,b], dist)
                # point[1] = -point[1]
                count += 1
                if point[0] != 0 or point[1] != 0 or point[2] != 0:
                    break
        
        point = np.append(point,1)
        print(point)
        point = np.dot(carib_mat,point)
        
        # dist = depth_frame.get_distance(int(point_list[index][1]), int(point_list[index][0]))
        # point = rs.rs2_deproject_pixel_to_point(color_intr , [point_list[index][1],point_list[index][0]], dist)

        return(point)

    def get_center2(self,r_rot):

        count = 0
        x = 0
        y = 0

        for i in range( len( r_rot )):
            x = x + r_rot[i][0]
            y = y + r_rot[i][1]
            count += 1

        x = x/count
        y = y/count

        return (x,y)

    def make_belief(self):
        
        self.img = depth_cap.get_depth_image()
        config = rs.config()
        #config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        #config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        #
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        #
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # ストリーミング開始
        pipeline = rs.pipeline()
        profile = pipeline.start(config)

        # Alignオブジェクト生成
        align_to = rs.stream.color
        align = rs.align(align_to)
        start = time.time()

        try:
            while True:

                # フレーム待ち(Color & Depth)
                frames = pipeline.wait_for_frames()

                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                self.depth_frame = aligned_frames.get_depth_frame()
                self.color_intr = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()
                elapsed_time = time.time() - start

                if not self.depth_frame or not color_frame:
                    continue

                #imageをnumpy arrayに
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(self.depth_frame.get_data())

                #depth imageをカラーマップに変換
                depth_colormap1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
                depth_colormap = cv2.cvtColor(depth_colormap1, cv2.COLOR_BGR2GRAY) 
                #画像表示
                color_image_s = cv2.resize(color_image, (640, 480))
                depth_colormap_s = cv2.resize(depth_image, (640, 480))
                # images = np.hstack((color_image_s, depth_colormap_s))
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', depth_colormap)

                if elapsed_time > 2:
                    img = cv2.resize(self.img,dsize=(int(depth_colormap_s.shape[1]*weight1),int(depth_colormap_s.shape[0]*weight2)))
                    self.img = img.astype(np.float64)
                    self.img_copy = self.img.copy()

                    cv2.imshow("img1",self.img)
                    cv2.imshow("img2",self.img_copy)
                    break

                if cv2.waitKey(1) & 0xff == 27:#ESCで終了
                    cv2.destroyAllWindows()
                    break

        finally:
            #ストリーミング停止
            pipeline.stop()  

        (region_list, rectregion , r_rot) = self.search_seed()
        
        detections = self.make_detect_class(rectregion)


        rotate_list = []
        point_list = []
        center_list_im = []
        center_list_im2 = []
        center_list = []
        center_list2 = []
        center =[]

        for i in range(len(region_list)):
            center_list_im.append(self.get_center2(region_list[i]))

        for i in range(len(r_rot)):
            chan = tuple(r_rot[i])
            box = cv2.boxPoints(chan)
            box = np.int0(box)
            center_list_im2.append(self.get_center(box))
            point_list.append(((box[0][0],box[0][1]),(box[1][0],box[1][1]),(box[2][0],box[2][1])))
            rotate_list.append(chan[2])

            color_image_s = cv2.drawContours(color_image_s,[box],0,(255,255,255),5)
        

        # point_listにはBBOXの頂点
        # rotate_listには回転が含まれている

        # print("center_list show ...")
        # print(center_list_im)

        # print(center_list_im2)

        BBox_size = []

        # print(point_list)
        for i in range( len(point_list) ):

            BBox_size.append(self.get_box_point(point_list,self.depth_frame,self.color_intr ,i,center_list_im))
            center_list.append(self.get_box_center(center_list_im,self.depth_frame,self.color_intr ,i,center_list_im2))
            center_list2.append(self.get_box_center2(center_list_im2,self.depth_frame,self.color_intr ,i,center_list_im))
            center.append((center_list2[i][0],center_list2[i][1],center_list[i][2]))
        
      
        # for i in range( len(point_list) ):
        print("中心座標を表示します")
        print(center)
        
        simple_state = self.get_state(center,rotate_list,BBox_size)
        # simple_state.visualization(name = "nagato_debug")
        
        cv2.imshow('NOT_normal_state', color_image_s)
        cv2.waitKey(0)   

        return simple_state

# 観測結果から一番簡単な予測を作成する
    def get_state(self,center,rotate_list,BBox_size):

        Target = []
        Obj_center = []
        Obj_rotate = []
        Obj_size = []
        Obj = []

        x = -40
        y = 25

        target_id = 5
        for i in range(len(center[target_id])):
            if i == 0:
                Target.append(center[target_id][i]*100+x)
            elif i == 1:
                Target.append(center[target_id][i]*100+y)
            else:
                Target.append(center[target_id][i]*100-64)

        global sliding
        sliding = np.array([center[target_id][0]*100+x,center[target_id][1]*100+y,center[target_id][2]*100-64])

        # BBOXの大きさを一意に決定する
        if BBox_size[target_id][0] < BBox_size[target_id][1]:
            Target.append(rotate_list[target_id]*math.pi/180  +  math.pi/2 )
            Target.append((BBox_size[target_id][0],BBox_size[target_id][1]))
        else:
            Target.append(rotate_list[target_id]*math.pi/180 )
            Target.append((BBox_size[target_id][1],BBox_size[target_id][0]))

        print("target is ...")
        print(Target)
        # 他物体の入力
        for i in range(len(center)):
            if i != target_id:
                obj_center =[]
                for j in range(len(center[0])):
                    if j == 0:
                        obj_center.append(center[i][j]*100 + x)
                    elif j == 1:
                        obj_center.append(center[i][j]*100 + y)
                    else:
                        obj_center.append(center[i][j]*100 - 64)
                Obj_center.append(obj_center)
 
        print(Obj_center)
        
        # TODO
        # for i in range(len(rotate_list)):
        #     if i != target_id:
        #         Obj_rotate.append(rotate_list[i]*math.pi/180 +  math.pi/2)
        print(Obj_rotate)
        
        for i in range(len(BBox_size)):
            if i != target_id:
                if BBox_size[i][0] < BBox_size[i][1]:
                    Obj_rotate.append(rotate_list[i]*math.pi/180+  math.pi/2)
                    Obj_size.append((BBox_size[i][0],BBox_size[i][1]))
                else:
                    Obj_rotate.append(rotate_list[i]*math.pi/180 )
                    Obj_size.append((BBox_size[i][1],BBox_size[i][0]))
                    # Obj_rotate.append(rotate_list[i]*math.pi/180 +  math.pi/2)

        print(Obj_size)
        for i in range(len(BBox_size)-1):
            print((Obj_center[i][0],Obj_center[i][1],Obj_center[i][2],Obj_rotate[i],Obj_size[i]))
            Obj.append((Obj_center[i][0],Obj_center[i][1],Obj_center[i][2],Obj_rotate[i],Obj_size[i]))

        print("obj and target")
        print(Obj,Target)

        return target_pos.State(Obj,Target,len(Obj)+1)

def observe_to_state(obs):
    
    Target = []

    Obj = []
    # print(obs)
    # for i in range(4):
    #         Target.append(obs[0][i])

    # Target.append((obs[0][4],obs[0][5]))

    # for i in range( N-1 ):
    #     Obj2 = []
    #     for j in range(4):
    #         if j != 3:
    #             Obj2.append(obs[i + 1][j])
    #         if j == 3:
    #             Obj2.append(obs[i + 1][j])
    #     Obj2.append((obs[i + 1][4],obs[i + 1][5]))
    #     Obj.append(Obj2)


    return research.State(obs.Obj_pos,obs.tar_pos) 

class State():
    # 状態を入力するクラス__init__で状態を初期化
    def __init__(self, Obj,target, N = 5):
        """
        (tuple): Obj:((x_1,y_1,z_1,yaw_1),.....(x_N,y_N,z_N,,yaw_N)

        """
        self.Obj_prev = Obj
        self.target = target
        self.Obj = list(Obj)
        height = []
        self.terminal = False

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



    def visualization(self,name = None):

        img = np.ones((w,h,3),np.uint8)*255

        # 対象物の位置を表示
        # print("pos is ...",self.target[0]*20,-self.target[1]*20+h)
        rotate = ((self.target[0]*20,-self.target[1]*20+h),(20*8,20*15),-self.target[3]*180/math.pi) 
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

def get_bigger_state1(obj,tar,i,a,b,num):
    
    obj_list = copy.deepcopy(obj)
    # print(obj_list)

    # 回転前の4つの頂点を格納
    pt1_1 = list((int(obj[i][0]*20 + obj[i][4][0]*20 / 2), int(-obj[i][1]*20 + h + obj[i][4][1]*20 / 2)))
    pt2_1 = list((int(obj[i][0]*20 + obj[i][4][0]*20 / 2), int(-obj[i][1]*20 + h - obj[i][4][1]*20 / 2)))
    pt3_1 = list((int(obj[i][0]*20 - obj[i][4][0]*20 / 2), int(-obj[i][1]*20 + h - obj[i][4][1]*20 / 2)))
    pt4_1 = list((int(obj[i][0]*20 - obj[i][4][0]*20 / 2), int(-obj[i][1]*20 + h + obj[i][4][1]*20 / 2)))

    # print("pos_obj1 is ...")
    # print(pt1_1)
    # print(pt2_1)
    # print(pt3_1)
    # print(pt4_1)
    # print("yokohaba is ...",a)
    # print("tatehaba is ...",b)

    # 伸ばしたことによって変わる座標を格納
    if num == 1:
        # print("yokohaba is ...",a)
        # print("tatehaba is ...",b)
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
    
    # print("pos_obj1 is ...")
    # print(pt1_1)
    # print(pt2_1)
    # print(pt3_1)
    # print(pt4_1)
    

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

    if obj[i][4][0] + a > obj[i][4][1] + b:
        wid = obj[i][4][1] + b
        hei = obj[i][4][0] + a
    else :
        wid = obj[i][4][0] + a
        hei = obj[i][4][1] + b

    belief_obj = (x/20,-(y2 - h)/20,obj[i][2],obj[i][3],(wid,hei))
    obj_list[i] = copy.deepcopy(belief_obj)
    
    # if num == 3 :
    #     s1 = State(obj_list,tar)
    #     s1.visualization(name = "num2")
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # countcm大きな物体を作成してもしそれの一辺3cm以下だったらNoneを返す
    return belief_obj,obj_list

# 物体の縦の大きさだけ変更したときに行けるかどうか確認するプログラム
def check_a(obj,i,a,tar,obs):

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
        o = Observation(obj_list,tar)
        o.correct_observ(5)
        o.sample_observ()
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
        o = Observation(obj_list,tar)
        o.correct_observ(5)
        o.sample_observ()
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

def get_init_particle(obs):

    obs.correct_observ(5)
    tar = obs.tar_pos
    obj = obs.Obj_pos
  
    particles = []

    Obj1 = []
    Obj2 = []
    Obj3 = []
    Obj4 = []
    Obj5 = []

    obj1_2 = []
    obj2_2 = []
    obj3_2 = []
    obj4_2 = []
    obj5_2 = []

    for i in range(5):
        if i == 0:
            Obj1.append(tuple(obj[i]))
        if i == 1:
            Obj2.append(tuple(obj[i]))
        if i == 2:
            Obj3.append(tuple(obj[i]))
        if i == 3:
            Obj4.append(tuple(obj[i]))
        if i == 4:
            Obj5.append(tuple(obj[i]))

    # 1cm 大きくした時の矩形を書くプログラム
    # obj2に大きくした時の物体の位置を表示するようにプログラムを書く
    # 上にある物体iから順に探索

    for i in range(N-1):
        a = 0 
        # b = 0
        while(1): 
            # 左上の点を基準に大きくしていく作業
            a += grid_size
            b = 0 
            
            # belief_objに大きくした物体を格納する
            # obj2に大きくした物体を含む集合を格納
            
            flag_list = []

            while(1):
                # どの点を基準に大きくするか
                for  x in range(4):
                    if flag_list.count(x) != 0: 
                        continue
                    
                    belief_obj,obj2 = get_bigger_state1(obj,tar,i,a,b,x)

                    if get_obj_num(tar,obj2) == True:
                        o = Observation(obj2,tar)
                        o.correct_observ(5)
                        o.sample_observ()

                        flag1 = True 

                        # for m in range(len(o.obj_area)):
                        #     print("area is ...",o.obj_area[m])
                        #     print("area2 is ...",obs.obj_area[m])

                        if abs(o.obj_area[i+1] - obs.obj_area[i+1]) > 0:
                            flag_list.append(x)
                            flag1 = False
                                       
                        if flag1:
                            if i == 0:
                                Obj1.append(tuple(belief_obj))
                            if i == 1:
                                Obj2.append(tuple(belief_obj))
                            if i == 2:
                                Obj3.append(tuple(belief_obj))
                            if i == 3:
                                Obj4.append(tuple(belief_obj))
                            if i == 4:
                                Obj5.append(tuple(belief_obj))

                # print("len_list is ...",len(flag_list))

                b += grid_size   
                if len(flag_list) == 4:
                    # print("aaaaaaaaaaaaaa")
                    break

            if check_a(obj,i,a,tar,obs) == False:
                # if check_b(obj,i,b,tar,obs) == False:
                break
                    
        # print("=============================--")
        
        
        obj2_1 = list(set(Obj1))
        obj2_2 = list(set(Obj2))
        obj2_3 = list(set(Obj3))
        obj2_4 = list(set(Obj4))
        obj2_5 = list(set(Obj5))
        # print(obj2_1)
        # print(obj2_2)
        # print(obj2_3) 
        # print(obj2_4) 
        # print(obj2_5)  
    
    Obj_belief = []
    Obj_belief = list(itertools.product(obj2_1,obj2_2,obj2_3,obj2_4,obj2_5))
    # print(len(Obj_belief))
    cv2.destroyAllWindows()
    for i in range(len(Obj_belief)):
        particles.append(research.State(Obj_belief[i],tar))


    return particles

def rotate_object(tar,obj):
    
    target = list(tar)
    object = list(obj)
    Object = []

    posx = 45
    posy = -10

    pos_tar = np.array([tar[0],tar[1],tar[2]])
    tar_rotate = np.dot(rotate_mat,pos_tar)

    
    target[0] = tar_rotate[0] + posx
    target[1] = tar_rotate[1] + posy
    target[2] = tar_rotate[2]
    target[3] = target[3] + math.pi/2

    obj_pos = []

    print("before ...")
    print(obj[0])

    for i in range(len(obj)):
        
        Obj = list(obj[i])

        pos = np.array([Obj[0],Obj[1],Obj[2]])
        obj_pos = np.dot(rotate_mat,pos)
        
        Obj[0] = obj_pos[0] + posx
        Obj[1] = obj_pos[1] + posy
        Obj[2] = obj_pos[2] 
        Obj[3] = Obj[3] + math.pi/2
        Object.append(Obj)
    
    print("afeter ...")
    print(Object[0])
    
    return target,Object



def init_particles_belief():
    
    belief = Belief()
    s = belief.make_belief()
    s.visualization()
    s.occlu_check()
    cv2.waitKey(0)

    # 対象物のどこに指を置くかを指定するための変数
    
    

    tar = target_pos.get_target_pose(s)
    obj = s.Obj

    # print(tar)

    # TODO rotate 90 rotate_mat
    # target,Object = rotate_object(tar,obj)

    target = tar
    Object = obj
    
    s1 = State(Object,target)
    s1.visualization(name="start")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # print("target_obj occlusion is  ....")
    # print(s.picel_target)
    # cv2.waitKey(0)

    obs = Observation(Object,target)
    
    print("real Ovrse is  ", obs)
    particle = get_init_particle(obs)

    return particle
    
if __name__ == "__main__":

    init_particles_belief()

   

   

    
   