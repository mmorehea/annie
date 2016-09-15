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

print 'REMEMBER TO SET THE IMAGE SHAPE CORRECTLY'

shape = (297, 406)

zSelect = int(sys.argv[1])

nSelect = int(sys.argv[2])

blobDict = pickle.load(open('pickles/blobDict' + str(zSelect-1) + '.p', 'rb'))

listofpixels = blobDict[nSelect]

image = np.zeros(shape, np.uint16)

for pixel in listofpixels:
	image[pixel] = 99999

cv2.imshow('blob',image)
cv2.waitKey()
code.interact(local=locals())
