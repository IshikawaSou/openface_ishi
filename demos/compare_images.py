#!/usr/bin/env python2
# -*-coding:utf-8 -*-
import cv2
import numpy as np
np.set_printoptions(precision=2)

import sys
import os
import openface
import glob
import itertools

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

class Rep:

def __init__(self):

self.align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
self.net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'), 96)
 
def get_rep(self,imgPath):

bgrImg = cv2.imread(imgPath)
rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

bb = self.align.getLargestFaceBoundingBox(rgbImg)
alignedFace = self.align.align(96, rgbImg, bb,landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

rep = self.net.forward(alignedFace)
return rep

if __name__=="__main__":

reps = Rep()
img1,img2 = sys.argv[1],sys.argv[2]

img1_rep, img2_rep = reps.get_rep(img1),reps.get_rep(img2)

d = img1_rep - img2_rep
print(np.dot(d,d))