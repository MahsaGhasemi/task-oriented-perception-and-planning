import setup_path
import airsim

import cv2
import time
import sys
import math
import numpy as np
from numpy.linalg import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pprint
import tf
import transformations
import os
import random

def noised_label(img_rgb, noise):
    segment = np.zeros((img_rgb.shape[0],img_rgb.shape[1],3))
    for i in range(img_rgb.shape[0]):
        for j in range(img_rgb.shape[1]):
            red = img_rgb[i,j,0]
            green = img_rgb[i,j,1]
            blue = img_rgb[i,j,2]

            if red == 55 and green == 181 and blue == 57:
                noisy = random.choice(noise)
                segment[i,j,0] = 1 - noisy
                segment[i,j,1] = noisy
            elif red == 153 and green == 108 and blue == 6:
                noisy = random.choice(noise)
                segment[i,j,1] = 1 - noisy
                segment[i,j,0] = noisy
            elif red == 112 and green == 105 and blue == 191:
                noisy = random.choice(noise)
                segment[i,j,2] = 1 - noisy
                segment[i,j,0] = noisy
    return segment

def point_cloud_gen(camera_info, rawImage):
    max_depth = 600 # Default Max Distance Field in UE4

    # Intrinsic Matrix for 512x512 resolution

    K = np.array([[64.,   0., 64.],
                       [  0., 64., 64.],
                       [  0.,   0.,   1.]])

    K = np.matmul(K,np.array([[0,0,-1],
                                        [0,1,0],
                                        [1,0,0]])) # match the coordinate system

    K_inv = np.linalg.inv(K) # reflect the image back to the frame

    pointsU = np.zeros((128,128))
    pointsV = np.zeros((128,128))
    points1 = np.ones((128,128))

    for i in range(pointsU.shape[0]):
      for j in range(pointsU.shape[1]):
        pointsU[i,j] = i
        pointsV[i,j] = j

    pointsUV = np.stack((pointsU,pointsV),axis=-1)
    pointsUV1 = np.stack((pointsU,pointsV,points1),axis=-1)

    w_val = camera_info.pose.orientation.w_val
    x_val = camera_info.pose.orientation.x_val
    y_val = camera_info.pose.orientation.y_val
    z_val = camera_info.pose.orientation.z_val
    x_cor = camera_info.pose.position.x_val
    y_cor = camera_info.pose.position.y_val
    z_cor = camera_info.pose.position.z_val*(-1)

    array = np.asarray(rawImage.image_data_float)
    depth = array.reshape((-1))
    depth3 = np.stack((depth,depth,depth),axis=-1)
    proj = np.matmul(K_inv,pointsUV1.reshape((-1,3)).T).T
    drone3d = np.multiply(proj,depth3) # point cloud without the extrinsic matrix

    drone3d1 = np.append(drone3d,np.ones((drone3d.shape[0],1)),1)
    trans = transformations.translation_matrix([x_cor, y_cor, z_cor])
    rot = transformations.quaternion_matrix([w_val, x_val, y_val, z_val])
    transform = transformations.concatenate_matrices(trans,rot)
    T = transform
    drone3d1world = np.matmul(T,drone3d1.T).T
    points = drone3d1world[:,:3] # point cloud

    return points

def gridworld_gen(labeled_point_cloud,grid_space,x_state_num,y_state_num):
    gridworld_0 = np.zeros((x_state_num*y_state_num,2))
    gridworld_1 = np.ones((x_state_num*y_state_num,1))
    gridworld = np.concatenate((gridworld_1,gridworld_0), axis = 1)
    prob_dict = {}
    x_length = (x_state_num-1) * grid_space
    y_length = (y_state_num-1) * grid_space
    for i in range(labeled_point_cloud.shape[0]):
        if labeled_point_cloud[i,0] > 0 and labeled_point_cloud[i,0] < x_length and labeled_point_cloud[i,1] > 0 and labeled_point_cloud[i,1] < y_length and labeled_point_cloud[i,2] > 5 and labeled_point_cloud[i,2] < 1000:
            for j in range(x_state_num*y_state_num):
                if labeled_point_cloud[i,0] + 2.5 > (j % x_state_num)*grid_space and labeled_point_cloud[i,0] + 2.5 < ((j % x_state_num)+1)*grid_space and labeled_point_cloud[i,1] + 2.5 > (j // x_state_num)*grid_space and labeled_point_cloud[i,1] + 2.5 < ((j // x_state_num)+1)*grid_space and (labeled_point_cloud[i,4] > 0.1 or labeled_point_cloud[i,5] > 0.1):
                    if not str(j) in prob_dict.keys():
                        prob_dict[str(j)] = {}
                        prob_dict[str(j)]['obstacle_prob'] = []
                        prob_dict[str(j)]['target_prob'] = []
                    if labeled_point_cloud[i,4] > 0.1:
                        prob_dict[str(j)]['obstacle_prob'].append(labeled_point_cloud[i,4])
                    else:
                        prob_dict[str(j)]['target_prob'].append(labeled_point_cloud[i,5])
                    break
    for grid in prob_dict:
        average_target = 0
        average_obstacle = 0
        if len(prob_dict[str(grid)]['target_prob']) != 0:
            average_target = np.average(prob_dict[str(grid)]['target_prob'])
        if len(prob_dict[str(grid)]['obstacle_prob']) != 0:
            average_obstacle = np.average(prob_dict[str(grid)]['obstacle_prob'])
        gridworld[int(grid),0] = gridworld[int(grid),0] - float(average_target+average_obstacle)
        gridworld[int(grid),1] = average_obstacle
        gridworld[int(grid),2] = float(average_target)
    return gridworld

def savePointCloud(image, fileName):
    color = (0,253,0)
    rgb = "%d %d %d" % color
    f = open(fileName, "w")
    for x in range(image.shape[0]):
        pt = image[x]
        f.write("%f %f %f %s\n" % (pt[0], pt[1], pt[2], rgb))
    f.close()
