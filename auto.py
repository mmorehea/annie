# -*- coding: utf-8 -*-

import argparse
import cv2
import glob
import code
import numpy as np
import sys
from timeit import default_timer as timer
import os
from itertools import cycle
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import cPickle as pickle

np.set_printoptions(threshold=np.inf)

class blob():
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




def findBB(dlist):
	xs = dlist[0]
	ys = dlist[1]

	return min(xs), max(xs), min(ys), max(ys)

def buildColorMap(img):
	colorMap = {0: 0}
	x, y = img.shape
	counter = 0
	uniqueValues = sorted(np.unique(img))
	for each in uniqueValues:
			if each in colorMap.values():
				continue
			else:
				counter += 1
				colorMap[counter] = each
	#print colorMap
	return colorMap

# /*
# ██     ██  █████  ████████ ███████ ██████  ███████ ██   ██ ███████ ██████
# ██     ██ ██   ██    ██    ██      ██   ██ ██      ██   ██ ██      ██   ██
# ██  █  ██ ███████    ██    █████   ██████  ███████ ███████ █████   ██   ██
# ██ ███ ██ ██   ██    ██    ██      ██   ██      ██ ██   ██ ██      ██   ██
#  ███ ███  ██   ██    ██    ███████ ██   ██ ███████ ██   ██ ███████ ██████
# */
def waterShed(img16):
	img8 = (img16/256).astype('uint8')

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

	return blobs

#  /*
# ███████ ██ ███    ██ ██████   ██████ ███████ ███    ██ ████████ ██████   ██████  ██ ██████
# ██      ██ ████   ██ ██   ██ ██      ██      ████   ██    ██    ██   ██ ██    ██ ██ ██   ██
# █████   ██ ██ ██  ██ ██   ██ ██      █████   ██ ██  ██    ██    ██████  ██    ██ ██ ██   ██
# ██      ██ ██  ██ ██ ██   ██ ██      ██      ██  ██ ██    ██    ██   ██ ██    ██ ██ ██   ██
# ██      ██ ██   ████ ██████   ██████ ███████ ██   ████    ██    ██   ██  ██████  ██ ██████
# */
def findCentroid(listofpixels):
	if len(listofpixels) == 0:
		return (0,0)
	rows = [p[0] for p in listofpixels]
	cols = [p[1] for p in listofpixels]
	try:
		centroid = int(round(np.mean(rows))), int(round(np.mean(cols)))
	except:
		# code.interact(local=locals())
		centroid = (0,0)
	return centroid

# /*
# ███████ ██ ███    ██ ██████  ███    ██ ███████  █████  ██████  ███████ ███████ ████████
# ██      ██ ████   ██ ██   ██ ████   ██ ██      ██   ██ ██   ██ ██      ██         ██
# █████   ██ ██ ██  ██ ██   ██ ██ ██  ██ █████   ███████ ██████  █████   ███████    ██
# ██      ██ ██  ██ ██ ██   ██ ██  ██ ██ ██      ██   ██ ██   ██ ██           ██    ██
# ██      ██ ██   ████ ██████  ██   ████ ███████ ██   ██ ██   ██ ███████ ███████    ██
# */
def findNearest(img, startPoint):
	directions = cycle([[0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1]])
	increment = 0
	cycleCounter = 0
	distance = [0,0]

	while True:
		direction = directions.next()

		for i in [0,1]:
			if direction[i] > 0:
				distance[i] = direction[i] + increment
			elif direction[i] < 0:
				distance[i] = direction[i] - increment
			else:
				distance[i] = direction[i]

		checkPoint = (startPoint[0] + distance[0],startPoint[1] + distance[1])

		cycleCounter += 1
		if cycleCounter % 8 == 0:
			increment += 1

		# print cycleCounter

		try:
			if img[checkPoint] > 0:
				break
		except:
			#code.interact(local=locals())
			break

	return checkPoint

#  /*
#  ██████  ███████ ████████ ███████ ███████ ███████ ██████  ██████  ██ ██   ██ ███████ ██
# ██       ██         ██    ██      ██      ██      ██   ██ ██   ██ ██  ██ ██  ██      ██
# ██   ███ █████      ██    ███████ █████   █████   ██   ██ ██████  ██   ███   █████   ██
# ██    ██ ██         ██         ██ ██      ██      ██   ██ ██      ██  ██ ██  ██      ██
#  ██████  ███████    ██    ███████ ███████ ███████ ██████  ██      ██ ██   ██ ███████ ███████
# */
def getSeedPixel(centroid, img, color):
	shouldSkip = False
	seedpixel = (0,0)


	if img[centroid] == 0:
		seedpixel = findNearest(img, centroid)
	else:
		seedpixel = centroid
	try:
		if img[seedpixel] == color:
			shouldSkip = True
			# print 'Same color! Skipping color ' + color
	except:
		shouldSkip = True
		# print 'Index out of bounds'

	return shouldSkip, seedpixel


# /*
# ████████ ███████ ███████ ████████  ██████  ██    ██ ███████ ██████  ██       █████  ██████
#    ██    ██      ██         ██    ██    ██ ██    ██ ██      ██   ██ ██      ██   ██ ██   ██
#    ██    █████   ███████    ██    ██    ██ ██    ██ █████   ██████  ██      ███████ ██████
#    ██    ██           ██    ██    ██    ██  ██  ██  ██      ██   ██ ██      ██   ██ ██
#    ██    ███████ ███████    ██     ██████    ████   ███████ ██   ██ ███████ ██   ██ ██
# */

def testOverlap(setofpixels1, setofpixels2):

	set_intersection = setofpixels1 & setofpixels2
	set_union = setofpixels1 | setofpixels2

	percent_overlap = float(len(set_intersection)) / len(set_union)

	return percent_overlap

# /*
# ██████  ███████  ██████ ███████ ███████  █████  ██████   ██████ ██   ██
# ██   ██ ██      ██      ██      ██      ██   ██ ██   ██ ██      ██   ██
# ██████  █████   ██      ███████ █████   ███████ ██████  ██      ███████
# ██   ██ ██      ██           ██ ██      ██   ██ ██   ██ ██      ██   ██
# ██   ██ ███████  ██████ ███████ ███████ ██   ██ ██   ██  ██████ ██   ██
# */
def recSearch(pixel, img, color):
	front = [pixel]
	found = [pixel]
	foundGrid = np.zeros((img.shape[0], img.shape[1]))
	foundGrid[pixel[0], pixel[1]] = 1
	counter = 0
	while len(front) > 0:
		fronty = front
		front = []
		for each in fronty:
			pixel = each
			searchPixels = [[pixel[0]+1, pixel[1]], [pixel[0]-1, pixel[1]], [pixel[0], pixel[1]+1], [pixel[0], pixel[1]-1]]
			#code.interact(local=locals())
			for neighbor in searchPixels:
				if neighbor[0] not in range(img.shape[0]) or neighbor[1] not in range(img.shape[1]):
					#print "hit border, skipping"
					continue
				#code.interact(local=locals())
				if img[neighbor[0], neighbor[1]] == color and foundGrid[neighbor[0], neighbor[1]] == 0 and neighbor not in front:
					#code.interact(local=locals())
					front.append([neighbor[0], neighbor[1]])
					foundGrid[neighbor[0], neighbor[1]] = 1
					counter = counter + 1

					#found.append([neighbor[0], neighbor[1]])
	found = np.where(foundGrid == 1)
	found = zip(found[0],found[1])
	return found



# /*
# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████
# */

dirr = sys.argv[1]
# stopAt = 200

# load the image, clone it, and setup the mouse callback function
list_of_image_paths = sorted(glob.glob(dirr +'*'))

images = []
for i, path in enumerate(list_of_image_paths):
	im = cv2.imread(path, -1)i + 1) + '/' + str(len(list_of_image_paths))

start = timer()

zTracker = {}
# for each blob, stores 1) a zvalue indicating how far up it is connected to other blobs and 2) the blob that connected to it in the previous slice
for imageCount, image1 in enumerate(images):

	colorMap = buildColorMap(image1)

	# Omitting the first one because it's just 0 mapped to 0
	colorVals = colorMap.values()[1:]

	numberOfColors = len(colorVals)

	blobList = []
	# stores for each blob in a given slice 1) the blob pixels 2) instantaneous z value and 3) n for the blob from previous slice that connected to it
	for n, color1 in enumerate(colorVals):

		where = np.where(image1 == color1)
		listofpixels1 = zip(list(where[0]), list(where[1]))

		try:
			image2 = images[imageCount + 1]
		except:
			continue

		centroid1 = findCentroid(listofpixels1)

		if centroid1 in zTracker.keys():
			nFromPrevSlice = zTracker[centroid1][0]
			zValue = zTracker[centroid1][1]
		else:
			nFromPrevSlice = None
			zValue = 0
			zTracker[centroid1] = [nFromPrevSlice, zValue]

		shouldSkip, seedpixel = getSeedPixel(centroid1, image2, color1)

		if shouldSkip:
			del zTracker[centroid1]
			listofpixels2 = []
			color2 = 0
			blobList.append(blob(imageCount, n, listofpixels1, color1, listofpixels2, color2, nFromPrevSlice, zValue, shouldSkip))
			continue

		setofpixels1 = set(listofpixels1)

		color2 = image2[seedpixel]
		whereColor = np.where(image2==color2)
		listofpixels2 = zip(whereColor[0],whereColor[1])
		setofpixels2 = set(listofpixels2)


		percent_overlap = testOverlap(setofpixels1, setofpixels2)
		overlap_threshold = 0.75

		centroid2 = findCentroid(listofpixels2)

		# cv2.circle(image1, (seedpixel[1], seedpixel[0]), 1, int(color2), -1)
		# cv2.putText(image1, str(n), (centroid1[1],centroid1[0]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, int(color2), 1,cv2.LINE_AA)
		# cv2.line(image1, (centroid1[1],centroid1[0]), (seedpixel[1], seedpixel[0]), int(color2), 1)

		#
		# if centroid2 in zTracker.keys():
		# 	if zTracker[centroid1][0] < zTracker[centroid2][0]:
		# 		del zTracker[centroid1]
		# 		continue

		if percent_overlap == 0:
			del zTracker[centroid1]
			shouldSkip = True
			blobList.append(blob(imageCount, n, listofpixels1, color1, listofpixels2, color2, nFromPrevSlice, zValue, shouldSkip))
			continue
		elif percent_overlap > overlap_threshold:
			for pixel in listofpixels2:
				image2[pixel] = color1
			pop = zTracker.pop(centroid1)
			zTracker[centroid2] = [n, pop[1] + 1]
			blobList.append(blob(imageCount, n, listofpixels1, color1, listofpixels2, color2, nFromPrevSlice, zValue, shouldSkip))
			# pop = zTracker.pop(centroid1)
			# zTracker[centroid2] = [pop[0] + 1, n]
		else:
			imageD = np.zeros(image1.shape, np.uint16)
			for pixel in listofpixels2:
				imageD[pixel] = color2


			subBlobs = waterShed(imageD)
			if len(subBlobs) > 1:
				percent_overlap = 0

				for b in subBlobs:
					pt = testOverlap(setofpixels1, set(b))
					if pt > percent_overlap:
						percent_overlap = pt
						listofpixels2 = b
						setofpixels2 = set(b)
						centroid2 = findCentroid(listofpixels2)
				if percent_overlap == 0:
					del zTracker[centroid1]
					shouldSkip = True
					blobList.append(blob(imageCount, n, listofpixels1, color1, listofpixels2, color2, nFromPrevSlice, zValue, shouldSkip))
					continue
				elif percent_overlap > overlap_threshold:
					for pixel in listofpixels2:
						image2[pixel] = color1
					pop = zTracker.pop(centroid1)
					zTracker[centroid2] = [n, pop[1] + 1]
					blobList.append(blob(imageCount, n, listofpixels1, color1, listofpixels2, color2, nFromPrevSlice, zValue, shouldSkip))
				else:
					del zTracker[centroid1]
					shouldSkip = True
					blobList.append(blob(imageCount, n, listofpixels1, color1, listofpixels2, color2, nFromPrevSlice, zValue, shouldSkip))
					continue
			else:
				del zTracker[centroid1]
				shouldSkip = True
				blobList.append(blob(imageCount, n, listofpixels1, color1, listofpixels2, color2, nFromPrevSlice, zValue, shouldSkip))
				continue


	for blob in blobList:
		cv2.putText(image1, blob.n, (blob.centroid1[1], blob.centroid1[0]), cv2.FONT_HERSHEY_COMPLEX_SMALL,  0.6, int(blob.color_foundbelow), 1,cv2.LINE_AA)
	cv2.imwrite('littleresult4/' + list_of_image_paths[imageCount][list_of_image_paths[imageCount].index('/')+1:], image1)
	pickle.dump(blobList, open('picklesLR5/blobList' + str(imageCount) + '.p', 'wb'))



code.interact(local=locals())
