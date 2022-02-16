# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import pyrealsense2 as rs
import time
import cv2
import numpy as np
from open3d import *

w = 480
h = 640

carib_mat = np.array([[-0.999880,-0.011667, 0.010202 , 0.534507],
                    [-0.012136, 0.998813 , -0.047167, -0.021096],
                    [-0.009640, -0.047285, -0.998835, 1.481784],
                    [0.000000,0.000000,0.000000,1.000000]
                    ])

def get_box_point(depth_frame,color_intr ):

    dist = []
    dist2 = []
    for i in range(w):
        for j in range(h):
            dist_a = depth_frame.get_distance(j, i)
            point_I = rs.rs2_deproject_pixel_to_point(color_intr , [j,i], dist_a)
            # print(point_I)
            point = np.append(point_I,1)
            # point[1] = -point[1]
            # point[2] = -point[2]
            point3d = np.dot(carib_mat,point)
            # print(point3d)

            dist.append(((i,j),point3d[2]))
            dist2.append(point3d[2])
    
    # print(dist2)
    cv2.waitKey(0)
    max_dep = max(l for l in dist2 if l < 1.48)
    min_dep = min(l for l in dist2 if l > 0.68)
    # min_dep = min(dist2)
    # print("hyouji")
    # print(max_dep)
    # print(min_dep)

    img =  np.zeros((w,h,1),np.uint8)
    for i in range(len(dist)):
        if max_dep >= dist[i][1] and min_dep <= dist[i][1]:
            x = dist[i][0][0]
            y = dist[i][0][1]
            
            # print((dist[i][1] - min_dep) * 255/(max_dep - min_dep))
            img[x,y] = (dist[i][1] - min_dep) * 255/(max_dep - min_dep)
    

    
    cv2.imshow("img1",img)
    cv2.waitKey(0)

    return img


def get_depth_image():

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # ストリーミング開始
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # Alignオブジェクト生成
    align_to = rs.stream.color
    align = rs.align(align_to)


    while True:
        # フレーム待ち(Color & Depth)
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        color_intr = rs.video_stream_profile(profile.get_stream(rs.stream.color)).get_intrinsics()

        if not depth_frame or not color_frame:
            continue

        #imageをnumpy arrayに
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())


        #depth imageをカラーマップに変換
        depth_colormap1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.08), cv2.COLORMAP_JET)
        depth_colormap = cv2.cvtColor(depth_colormap1, cv2.COLOR_BGR2GRAY) 
        #画像表示
        color_image_s = cv2.resize(color_image, (640, 480))
        depth_colormap_s = cv2.resize(depth_image, (640, 480))

        img = cv2.resize(depth_colormap_s,dsize=(int(depth_colormap_s.shape[1]*0.5),int(depth_colormap_s.shape[0]*0.5)))
        img = img.astype(np.float64)
        height, width = img.shape #画像サイズの取得
        img_copy = img.copy() #画像のコピー
        
        img = get_box_point(depth_frame,color_intr)
        ret, img1 = cv2.threshold(img, 20, 255, cv2.THRESH_TOZERO)
        cv2.imwrite("depth_img.png",img1)

        return img1

        cv2.imshow('frame', color_image_s)
        if cv2.waitKey(1) & 0xff == 27:#ESCで終了
            cv2.destroyAllWindows()
            break

def main():
    get_depth_image()

if __name__ == "__main__":

    main()
