# -*- coding: utf-8 -*-
from __future__ import print_function
import copy
import os
import shutil
import argparse
import numpy as np
import sys
from PIL import Image
from PIL import ImageDraw
import math
import cv2
#from shapely.geometry import LineString
from scipy import interpolate
import matplotlib.pyplot as plt
# from skimage.morphology import skeletonize
# from skimage import data
from scipy import ndimage
# from skimage.morphology import medial_axis
# from skimage.util import invert
from scipy.ndimage import rotate
from scipy import ndimage
import csv
import pprint
import time
def cv2pil(image):
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image
def graspability(mask, img_mask):
    
    Wc = img_mask
    Wt = mask
 
    TEMPLATE_SIZE = 400
    GaussianKernelSize = 301
    GaussianSigma = 35
    Ht_original = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE))
    Hc_original = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE))
    HAND_THICKNESS_X = 2#シミュレーション時は2だが、実際のハンドに合わせて15mm//(おそらく横幅)
    HAND_THICKNESS_Y = 5#シミュレーション時は5だが、実際のハンドに合わせて25mm//(おそらく縦幅)
    BEFORE_TO_AFTER = 500/100#//おそらく奥行き
    HAND_WIDTH = 50
    L1x = int((TEMPLATE_SIZE / 2) - (((HAND_WIDTH / 2) + HAND_THICKNESS_X) * BEFORE_TO_AFTER))
    L2x = int((TEMPLATE_SIZE / 2) - (((HAND_WIDTH / 2) + 0) * BEFORE_TO_AFTER))
    L3x = int((TEMPLATE_SIZE / 2) - (((HAND_WIDTH / 2) + 0) * BEFORE_TO_AFTER))
    L4x = int((TEMPLATE_SIZE / 2) - (((HAND_WIDTH / 2) + HAND_THICKNESS_X) * BEFORE_TO_AFTER))
    R1x = int((TEMPLATE_SIZE / 2) + (((HAND_WIDTH / 2) + 0) * BEFORE_TO_AFTER))
    R2x = int((TEMPLATE_SIZE / 2) + (((HAND_WIDTH / 2) + HAND_THICKNESS_X) * BEFORE_TO_AFTER))
    R3x = int((TEMPLATE_SIZE / 2) + (((HAND_WIDTH / 2) + HAND_THICKNESS_X) * BEFORE_TO_AFTER))
    R4x = int((TEMPLATE_SIZE / 2) + (((HAND_WIDTH / 2) + 0) * BEFORE_TO_AFTER))
    L1y = int((TEMPLATE_SIZE / 2) - ((HAND_THICKNESS_Y / 2) * BEFORE_TO_AFTER))
    L2y = int((TEMPLATE_SIZE / 2) - ((HAND_THICKNESS_Y / 2) * BEFORE_TO_AFTER))
    L3y = int((TEMPLATE_SIZE / 2) + ((HAND_THICKNESS_Y / 2) * BEFORE_TO_AFTER))
    L4y = int((TEMPLATE_SIZE / 2) + ((HAND_THICKNESS_Y / 2) * BEFORE_TO_AFTER))
    R1y = int((TEMPLATE_SIZE / 2) - ((HAND_THICKNESS_Y / 2) * BEFORE_TO_AFTER))
    R2y = int((TEMPLATE_SIZE / 2) - ((HAND_THICKNESS_Y / 2) * BEFORE_TO_AFTER))
    R3y = int((TEMPLATE_SIZE / 2) + ((HAND_THICKNESS_Y / 2) * BEFORE_TO_AFTER))
    R4y = int((TEMPLATE_SIZE / 2) + ((HAND_THICKNESS_Y / 2) * BEFORE_TO_AFTER))
    #左上の角と右下の角を指定すると四角形が描かれる
    cv2.rectangle(Hc_original, (L1x, L1y), (L3x, L3y), (255, 255, 255), -1)
    cv2.rectangle(Hc_original, (R1x, R1y), (R3x, R3y), (255, 255, 255), -1)
    cv2.rectangle(Ht_original, (L2x, L2y), (R4x, R4y), (255, 255, 255), -1)
    

    Ht_original = cv2pil(Ht_original)
    Hc_original = cv2pil(Hc_original)
    # うまく行かないときにここをTRUEにして確認する
    debugmade = False
    if debugmade == True:
        cv2.imshow("Wc",Wc)
        cv2.imshow("Wt",Wt)
        cv2.imshow("Ht",np.array(Ht_original))
        cv2.imshow("Hc",np.array(Hc_original))
        cv2.waitKey()
    
    optimal_grasp = [[0] for i in range(3)]
    # for rotate in np.arange (0, 180, 10.0):
    for rotate in np.arange (0, 20, 5.0):
        Ht_rotate = Ht_original.rotate(rotate)
        Hc_rotate = Hc_original.rotate(rotate)
        T = cv2.filter2D(Wt, -1, np.array(Ht_rotate))
        C = cv2.filter2D(Wc, -1, np.array(Hc_rotate))
        Cbar = 255 - C
        T_and_Cbar = T & Cbar#//共通部分の作成
        G = cv2.GaussianBlur(T_and_Cbar, (GaussianKernelSize, GaussianKernelSize), GaussianSigma, GaussianSigma) #//ガウシアンフィルタをかける(白い部分からの距離に応じた平滑化)
        # cv2.imshow("T",T)
        # cv2.imshow("C",C)
        # cv2.imshow("T_and_Cbar",T_and_Cbar)
        # cv2.imshow("G", G)
        cv2.waitKey()
        max_graspability = np.amax(G)
        # print(max_graspability)
        if optimal_grasp[0] < max_graspability:
            optimal_grasp[0] = max_graspability
            optimal_grasp[1] = rotate
            index = np.where(G >= optimal_grasp[0])
            # print(index)
            optimal_grasp[2] = [index[0][0], index[1][0]]
    return optimal_grasp,max_graspability

def calcWt2(mask,img_mask):

    Wc = np.ones((mask.shape[0],mask.shape[1]),np.uint8)
    ave = 0
    count = 0
    # print(img_mask)
    count = cv2.countNonZero(img_mask)
    ave =  img_mask[img_mask > 0].sum()/count

    # print("threshhold is ...",ave)

    ret, Wc = cv2.threshold(mask, ave-1, 255, cv2.THRESH_BINARY)
                
    return Wc



def calcWt(mask,img_mask):

    empty_list = [0]*256
    
    for i in range(0,mask.shape[0]):
        for j in range(0,mask.shape[1]):
            if mask[i][j]  == 255:
                empty_list[img_mask[i][j]] += 1
    
    empty_list2 = copy.copy(empty_list)
    empty_list2.sort()
    
   
    count = 0
    threshhold = 0
    threshhold2 = 0
    for i in range(0,255):
        for j in range(0,255):
            if empty_list[j] ==  empty_list2[255-i]  and (abs(j - threshhold*0.95)<5 or i == 0):
                threshhold  = j
                if count == 0:
                    threshhold2 = j
                if threshhold2 > j:
                    threshhold2 = j
                count += 1
                # print(j)
            if empty_list2[255-i] == 0:
                break

    

    # print(threshhold)
    
    new_mask = img_mask.copy()
    
    for i in range(0,mask.shape[0]):
        for j in range(0,mask.shape[1]):
            if threshhold2  > img_mask[i][j]:
                new_mask[i][j] = 0
    
    for i in range(0,mask.shape[0]):
        for j in range(0,mask.shape[1]):
            if new_mask[i][j] !=0:
                new_mask[i][j] = 255

    return new_mask



if __name__ == "__main__":
    # mask = cv2.imread('/home/nagato/graspability/graspability3.png',0)
    # img_mask = cv2.imread('/home/nagato/graspability/graspability4.png',0)
    # img = cv2.imread('/home/nagato/graspability/graspability4.png')
    
    mask = cv2.imread('/home/nagato/graspability/collision4.png',0)
    img_mask = cv2.imread('/home/nagato/graspability/collision3.png',0)
    img = cv2.imread('/home/nagato/graspability/collision3.png')

    
    Wc = calcWt2(img_mask,mask)
    s1 = time.time()
    
    # cv2.imshow("Wc",Wc)
    # cv2.waitKey(0)
    
    grasp,_ = graspability(mask, Wc)
    s2 = time.time()
    print(grasp)
    if grasp[0] > 80:
        alpha = 100
        point = np.array(grasp[2])
        theta = -(grasp[1] + 90)
        vdp = np.array([int(alpha*math.cos(math.radians(theta))), int(-alpha*math.sin(math.radians(theta)))])
        gp1 = point + vdp
        gp2 = point - vdp
        cv2.line(img, (int(gp1[1]), int(gp1[0])), (int(gp2[1]), int(gp2[0])), (255, 255, 255), 3)
        cv2.circle(img, (int(gp1[1]), int(gp1[0])), 10, (255, 255, 255), -1)
        cv2.circle(img, (int(gp2[1]), int(gp2[0])), 10, (255, 255, 255), -1)
        cv2.circle(img, (int(grasp[2][1]), int(grasp[2][0])), 10, (255, 255, 255), -1)

   
    print("grasp time is ...",s2 - s1)
    
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()