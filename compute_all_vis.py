import os
import sys
import code

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skfmm import distance
import argparse
import pandas as pd

from vis2d import vis2d


def compute_visibility(phi, psi, x0, dx):
    # assumes x0 is [n,2] matrix, each row is an observing location
    psi_current = vis2d(phi, np.round(x0[-1,:]/dx).astype(int))
    psi_current = distance(psi_current,dx)
    psi = np.maximum(psi, psi_current)
    return psi, psi_current


def img2obj(image):
    # from image, compute obstacles using row major indexing
    h,w = image.shape[:2]
    obj = []

    # assuming
    for i in range(h):
        for j in range(w):
            if image[i,j] < 254:
                obj.append(i*w + j)
    return obj


def obj2img(obj, h, w):

    # convert to image for computation
    image = 255 * np.ones((h,w))

    numObj = len(obj)

    for ind in obj:
        i = ind // w
        j = ind % w
        image[i,j] = 0

    return image


def createCircle(center, radius, x, y):
    dx = x[0,1] - x[0,0]
    mask = (x-center[1])**2 + (y-center[0])**2 <= radius**2
    mask = 1*mask # convert to number
    return mask

def compute_visibility_for_all(obj, h, w, pos_x, pos_y, radius = np.inf):
    image = obj2img(obj, h, w)
    image = image * 1.0/image.max() # rescale to [0,1]
    dx = 1.0/h
    phi = distance( (image-.5)*2*dx, dx ) # convert to signed distance function

    notVis = {}  # dictionary containing {position: [non visible positions] }

    # create a grid for circular mask
    x = np.linspace(0,w-1,w)
    y = np.linspace(0,h-1,h)
    x,y = np.meshgrid(x,y)

    mask = createCircle([pos_x,pos_y], radius, x, y)
    psi = vis2d(phi, [pos_x,pos_y])
    psi = 1*(psi>0) * mask
    I,J = np.where(psi==0)  # subscripts of obstacles
    occluded = (I*w + J).tolist()
    notVis = set(occluded)
    return notVis

if __name__ == '__main__':

    pos_x = 0
    pos_y = 35
    WS = pd.read_excel('GridWorld_Labels.xlsx',index_col=None, header=None)
    WS_np = np.array(WS)
    h,w = WS_np.shape[:2]
    WS_1d = WS_np.flatten()
    obj = []
    for i in range(WS_1d.shape[0]):
        if WS_1d[i] == 1:
            obj.append(i)

    # compute visibility for each state
    notVis = compute_visibility_for_all(obj, h, w, pos_x, pos_y, radius = 100)
