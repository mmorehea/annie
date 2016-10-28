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

class Blob():
	def __init__(self, imageCount, n, listofpixels, color, listofpixels_foundbelow, color_foundbelow, nFromPrevSlice, zValue, skipped):
		self.imageCount = imageCount
		self.n = n
		self.listofpixels = listofpixels
		self.color = color
		self.listofpixels_foundbelow = listofpixels_foundbelow
		self.color_foundbelow = color_foundbelow
		self.nFromPrevSlice = nFromPrevSlice
		self.zValue = zValue
		self.skipped = skipped

		self.centroid = findCentroid(listofpixels)
		self.centroid_foundbelow = findCentroid(listofpixels_foundbelow)
		self.percent_overlap_foundbelow = testOverlap(set(self.listofpixels), set(self.listofpixels_foundbelow))

print 'REMEMBER TO SET THE IMAGE SHAPE CORRECTLY'

shape = (286, 424)

pickleFolder = 'picklesLR/'

zSelect = int(sys.argv[1])

nSelect = int(sys.argv[2])

blobList = pickle.load(open(pickleFolder + 'blobList' + str(zSelect-1) + '.p', 'rb'))

blob = next((b for b in blobList if b.n == nSelect), None)

if blob == None:
	print 'This blob does not exist.'
else:

	image1 = np.zeros(shape, np.uint16)
	image2 = np.zeros(shape, np.uint16)

	for pixel in blob.listofpixels:
		image1[pixel] = 99999
	for pixel in blob.listofpixels_foundbelow:
		image2[pixel] = 99999

	print '\n'
	print 'n from previous slice: ' + str(blob.nFromPrevSlice)
	print 'zValue: ' + str(blob.zValue)
	print 'Skipped: ' + str(blob.skipped)
	print 'Percent overlap: ' + str(blob.percent_overlap_foundbelow)
	print '\n'

	cv2.imshow('foundbelow ' + str(blob.imageCount + 1) + ' ' + str(blob.n), image2)
	cv2.imshow('listofpixels ' + str(blob.imageCount + 1) + ' ' + str(blob.n),image1)
	cv2.waitKey()
	code.interact(local=locals())
