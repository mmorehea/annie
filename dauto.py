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



def waterShed(blob, imageD):
	img16 = imageD
	for pixel in blob:
		img16[pixel] = 99999
	img8 = (img16/256).astype('uint8')
	w = np.where(img8 != 0)
	initblob = zip(w[0], w[1])

	contours = cv2.findContours(img8.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[1]
	blobs = []
	for cnt in contours:

		mask = np.zeros(img8.shape,np.uint8)
		cv2.drawContours(mask,[cnt],0,255,-1)
		pixelpoints = np.transpose(np.nonzero(mask))
		blobs.append([(x[0],x[1]) for x in pixelpoints])

		# convert back to row, column and store as list of points
		# cnt = [(x[0][1], x[0][0]) for x in cnt]
		# newconts.append(cnt)

	if len(blobs) == 1 and blobs[0] != initblob: # causes significant speed hit
		newblob = list(set(initblob) - set(blobs[0]))
		blobs.append(newblob)


	return blobs

def display(blob, imageD):
	img = imageD
	for pixel in blob:
		img[pixel] = 99999

	cv2.imshow('display',img)
	cv2.waitKey()


def erode(blob, imageD):
	img16 = imageD
	for pixel in blob:
		img16[pixel] = 99999
	img8 = (img16/256).astype('uint8')

	kernel = np.ones((5,5), np.uint8)
	erosion = cv2.erode(img8,kernel,iterations=1)

	w = np.where(erosion != 0)
	result = zip(w[0],w[1])
	return result


print 'REMEMBER TO SET THE IMAGE SHAPE CORRECTLY'

global shape

shape = (286, 424)

pickleFolder = 'pickles1/'

zSelect = int(sys.argv[1])

nSelect = int(sys.argv[2])

blobList = pickle.load(open(pickleFolder + 'blobList' + str(zSelect-1) + '.p', 'rb'))

blob = next((b for b in blobList if b.n == nSelect), None)

imageD = np.zeros(shape, np.uint16)

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
