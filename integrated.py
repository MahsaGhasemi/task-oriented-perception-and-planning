# for planner
import setup_path
import sys
import time
from math import isclose
import numpy as np
import copy
import itertools
from scipy.special import comb
from scipy.stats import entropy
from scipy.spatial import distance
import random
import gurobipy as grb
import pandas as pd
import matplotlib.pyplot as plt
from partial_semantics import *

# for airsim
import setup_path
import airsim

import cv2
import math
from numpy.linalg import linalg
from mpl_toolkits.mplot3d import Axes3D
import pprint
import tf
import transformations
import os
from utils import *
import tempfile
import json
from scipy import misc
from compute_all_vis import *


##############################################################################
# Airsim initialization
noise = [0.01, 0.05, 0.1, 0.25]
grid_space = 5
x_state_num = 37
y_state_num = 24

# Starting the AirSim Client
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, "Drone1")
client.armDisarm(True, "Drone1")

# Giving the object ID's for different meshes

client.simSetSegmentationObjectID("[\w]*", 0, True);
client.simSetSegmentationObjectID("BG_Building[\w]*", 1, True);
client.simSetSegmentationObjectID("Target_Building[\w]*", 2, True);

airsim.wait_key('Press any key to takeoff')
client.takeoffAsync(vehicle_name="Drone1").join()

target = 0 # At the beginning reaching the target is FALSE
agent_x = 0
agent_y = 0
##############################################################################

# define simulation parameters
n_iter = 1
infqual_hist_all = []
risk_hist_all = []
timestep_all = []
plan_count_all = []
task_flag_all = []

for iter in range(n_iter):

    # create problem setting
    model = MDP('gridworld')
    model.semantic_representation(prior_belief='noisy-ind') # changed for scenario 2
    perc_flag = True
    bayes_flag = True
    replan_flag = True
    div_test_flag = True
    act_info_flag = True
    spec_true = [[],[]]
    for s in range(len(model.states)):
        if model.label_true[s,0] == True:
            spec_true[0].append(s)
        if model.label_true[s,1] == True:
            spec_true[1].append(s)

    visit_freq = np.ones(len(model.states))

##############################################################################

    # simulation results
    term_flag = False
    task_flag = False
    timestep = 0
    max_timestep = 250
    plan_count = 0
    div_thresh = 0.001
    n_sample = 10
    risk_thresh = 0.1
    state_hist = []
    state_hist.append(model.init_state)
    action_hist = [[],[]] # [[chosen action],[taken action]]
    infqual_hist = []
    infqual_hist.append(info_qual(model.label_belief))
    risk_hist = []

    f1 = client.moveToPositionAsync(agent_x*grid_space, agent_y*grid_space, -30, 8, vehicle_name="Drone1")
    f1.join()

    while not term_flag:

        if perc_flag:
            # estimate map
            label_est = estimate_AP(model.label_belief, method='risk-averse')
            spec_est = [[],[]]
            for s in range(len(model.states)):
                if label_est[s,0] == True:
                    spec_est[0].append(s)
                if label_est[s,1] == True:
                    spec_est[1].append(s)
            print("obstacle:   ",spec_est[0])
            print("target:     ",spec_est[1])

        if replan_flag or (not replan_flag and plan_count==0):
            # find an optimal policy
            (vars_val, opt_policy) = verifier(copy.deepcopy(model), spec_est)
            print(opt_policy[0:20])
            plan_count += 1

        if act_info_flag:
            # risk evaluation
            prob_sat = stat_verifier(model,state_hist[-1],opt_policy,spec,n_sample)
            risk = np.abs(vars_val[state_hist[-1]] - prob_sat); print(vars_val[state_hist[-1]],prob_sat)
            risk_hist.append(risk)
            print("Risk due to Perception Uncertainty:   ",risk)

            # perception policy
            if risk > risk_thresh:
                # implement perception policy
                timestep += 1
                state = state_hist[-1]
                action = 0
            else:
                pass
        timestep += 1
        print("Timestep:   ",timestep)
        state = state_hist[-1]
        opt_act = opt_policy[state]
        if 0 in opt_act and len(opt_act)>1:
            opt_act = opt_act[1:]

        action = np.random.choice(opt_act)

        action_hist[0].append(action)
        next_state = np.random.choice(model.states, p=model.transitions[state,action])
        # identify taken action
        for a in model.actions[model.enabled_actions[state]]:
            if model.action_effect(state,a)[0] == next_state:
                action_taken = a
        action_hist[1].append(action_taken)
        state_hist.append(next_state)

############################################################################## uncommented for scenario 2
        # get new information
        # (obs,p_obs_model) = obs_modeling(model)
        #
        # # update belief
        # next_label_belief = belief_update(model.label_belief, obs,
        #                                  p_obs_model, bayes_flag)

############################################################################### commented for scenario 2
        # while target == 0:
        for i in range(1):
            # time.sleep(5)
            response = client.simGetImages([airsim.ImageRequest("9", airsim.ImageType.Segmentation, False, False),
            airsim.ImageRequest("10", airsim.ImageType.Segmentation, False, False),
            airsim.ImageRequest("11", airsim.ImageType.Segmentation, False, False),
            airsim.ImageRequest("12", airsim.ImageType.Segmentation, False, False),
            airsim.ImageRequest("0", airsim.ImageType.Scene)])
            airsim.write_file('png.png', response[4].image_data_uint8)
            img1d_0 = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8) # get numpy array
            img1d_1 = np.frombuffer(response[1].image_data_uint8, dtype=np.uint8)
            img1d_2 = np.frombuffer(response[2].image_data_uint8, dtype=np.uint8)
            img1d_3 = np.frombuffer(response[3].image_data_uint8, dtype=np.uint8)
            img_rgb_0 = img1d_0.reshape(response[0].height, response[0].width, 4) # reshape array to 3 channel image array H X W X 3
            img_rgb_1 = img1d_1.reshape(response[1].height, response[1].width, 4)
            img_rgb_2 = img1d_2.reshape(response[2].height, response[2].width, 4)
            img_rgb_3 = img1d_3.reshape(response[3].height, response[3].width, 4)
            labeled_img_0 = noised_label(img_rgb_0, noise)
            labeled_img_1 = noised_label(img_rgb_1, noise)
            labeled_img_2 = noised_label(img_rgb_2, noise)
            labeled_img_3 = noised_label(img_rgb_3, noise)
            camera_info_0 = client.simGetCameraInfo("5")
            camera_info_1 = client.simGetCameraInfo("6")
            camera_info_2 = client.simGetCameraInfo("7")
            camera_info_3 = client.simGetCameraInfo("8")

            rawImage = client.simGetImages([airsim.ImageRequest("5", airsim.ImageType.DepthPlanner, True),
            airsim.ImageRequest("6", airsim.ImageType.DepthPlanner, True),
            airsim.ImageRequest("7", airsim.ImageType.DepthPlanner, True),
            airsim.ImageRequest("8", airsim.ImageType.DepthPlanner, True)])

            point_cloud_0 = point_cloud_gen(camera_info_0,rawImage[0])
            point_cloud_1 = point_cloud_gen(camera_info_1,rawImage[1])
            point_cloud_2 = point_cloud_gen(camera_info_2,rawImage[2])
            point_cloud_3 = point_cloud_gen(camera_info_3,rawImage[3])
            point_cloud = np.concatenate((point_cloud_0,point_cloud_1,point_cloud_2,point_cloud_3),axis = 0)
            savePointCloud(point_cloud_0, "cloud.asc")
            stack_0 = np.concatenate((point_cloud_0,labeled_img_0.reshape((-1,3))),axis = 1)
            stack_1 = np.concatenate((point_cloud_1,labeled_img_1.reshape((-1,3))),axis = 1)
            stack_2 = np.concatenate((point_cloud_2,labeled_img_2.reshape((-1,3))),axis = 1)
            stack_3 = np.concatenate((point_cloud_3,labeled_img_3.reshape((-1,3))),axis = 1)
            labeled_point_cloud = np.concatenate((stack_0,stack_1,stack_2,stack_3),axis = 0)
            gridworld = gridworld_gen(labeled_point_cloud,grid_space,x_state_num,y_state_num)
            busy = []
            for i in range(gridworld.shape[0]):
                if gridworld[i,1] > 0.5 or gridworld[i,2] > 0.5:
                    busy.append(i)
            # compute visibility for each state
            notVis = compute_visibility_for_all(busy, y_state_num, x_state_num, agent_y, agent_x, radius = 100)
            notVis = list(set(notVis).difference(set(busy)))

            df = pd.DataFrame(gridworld.reshape((24,37,3))[:,:,2])
            filepath = 'my_excel_file5.xlsx'
            df.to_excel(filepath, index=False)

            # update belief
            next_label_belief = copy.deepcopy(model.label_belief)
            visit_freq_next = copy.deepcopy(visit_freq) + 1
            for s in notVis:
                visit_freq_next[s] -= 1
            for s in model.states:
                # update for 'obstacle'
                next_label_belief[s,0] = (next_label_belief[s,0]*visit_freq[s] + gridworld[s,1]) / visit_freq_next[s]
                # update for 'target'
                next_label_belief[s,1] = (next_label_belief[s,1]*visit_freq[s] + gridworld[s,2]) / visit_freq_next[s]
            visit_freq = copy.deepcopy(visit_freq_next)

##############################################################################
        # move to next state
        if len(state_hist) > 1:
            if next_state != state_hist[-2]:
                next_state = model.state_mapping[next_state]
                print(next_state)
                f1 = client.moveToPositionAsync(next_state[1]*grid_space, next_state[0]*grid_space, -30, 2, vehicle_name="Drone1")
        else:
            next_state = model.state_mapping[next_state]
            print(next_state)
            f1 = client.moveToPositionAsync(next_state[1]*grid_space, next_state[0]*grid_space, -30, 2, vehicle_name="Drone1")
        # f1.join()
        time.sleep(5)

        # divergence test on belief
        if div_test_flag:
            div = info_div(model.label_belief,next_label_belief)
            print("Belief Divergence:   ",div)
            if info_div(model.label_belief,next_label_belief) > div_thresh:
                replan_flag = True
            else:
                replan_flag = False
        model.label_belief = np.copy(next_label_belief)
        infqual_hist.append(info_qual(model.label_belief))

        # check task realization
        if model.label_true[state_hist[-1],0] == True:
            term_flag = True
            print("at a state with an obstacle")

        if model.label_true[state_hist[-1],1] == True:
            task_flag = True
            term_flag = True
            print("at a target state")

        if timestep == max_timestep:
            term_flag = True
            print("timestep exceeded the maximum limit")

    print("Number of Time Steps:   ",timestep)
    print("Number of Replanning:   ",plan_count)

##############################################################################
    # exit AirSim
    airsim.wait_key('Press any key to reset to original state')

    client.armDisarm(False)
    client.reset()

    # that's enough fun for now. let's quit cleanly
    client.enableApiControl(False)

##############################################################################
    infqual_hist_all.append(infqual_hist)
    risk_hist_all.append(risk_hist)
    timestep_all.append(timestep)
    plan_count_all.append(plan_count)
    task_flag_all.append(int(task_flag))

task_rate = np.mean(task_flag_all)
