# -*- coding: utf-8 -*-

import argparse
import cv2
import glob
import code
import numpy as np
import sys
from timeit import default_timer as timer
import os
import cPickle as pickle

print 'REMEMBER TO SET THE SHAPE CORRECTLY'

shape = (297, 406)

zSelect = sys.argv[1]

nSelect = sys.argv[2]

blobArray = pickle.load(open('blobArray.p', 'rb'))

listofpixels = blobArray[zSelect][nSelect]

image = np.zeros(shape, np.uint16)

for pixel in listofpixels:
	image[pixel] = 5555

cv2.imshow('blob',image)
cv2.waitKey()
