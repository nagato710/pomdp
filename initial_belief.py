import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import pyrealsense2 as rs
import numpy as np
import cv2
import numpy as np
import math
import sys
import random


def get_box_point(x,y,depth_frame,color_intr ):

    dist = depth_frame.get_distance(x, y)
    point_I = rs.rs2_deproject_pixel_to_point(color_intr , [80,80], dist)
    print("point", point_I)



def main():

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

        
        
        get_box_point(80,80,depth_frame,color_intr)

        

        cv2.imshow('frame', color_image_s)
        if cv2.waitKey(1) & 0xff == 27:#ESCで終了
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':

    main()
    