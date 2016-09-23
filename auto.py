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
	def __init__(self, imageCount, n, listofpixels, color, listofpixels_foundbelow, color_foundbelow, nFromPrevSlice, zValue):
		self.imageCount = imageCount
		self.n = n
		self.listofpixels = listofpixels
		self.color = color
		self.listofpixels_foundbelow = listofpixels_foundbelow
		self.color_foundbelow = color_foundbelow
		self.nFromPrevSlice = nFromPrevSlice
		self.zValue = zValue

		self.centroid = findCentroid(listofpixels)
		self.centroid_foundbelow = findCentroid(listofpixels_foundbelow)




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
# ███████ ██████  ██      ██ ████████ ██████  ██       ██████  ██████  ███████
# ██      ██   ██ ██      ██    ██    ██   ██ ██      ██    ██ ██   ██ ██
# ███████ ██████  ██      ██    ██    ██████  ██      ██    ██ ██████  ███████
#      ██ ██      ██      ██    ██    ██   ██ ██      ██    ██ ██   ██      ██
# ███████ ██      ███████ ██    ██    ██████  ███████  ██████  ██████  ███████
# */
def splitBlobs(blobList, listofpix, blobs):
	blobToSplit = next((x for x in blobList if x.listofpixels == listofpix), None)
	index = blobList.index(blobToSplit)

	for i, blob in enumerate(blobs):
		newblob = blob(blobToSplit.imageCount, )
		blobList.insert(index + i,blob)
	blobList.remove(blobToSplit)



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
for imageCount, image in enumerate(images):

	colorMap = buildColorMap(image)

	# Omitting the first one because it's just 0 mapped to 0
	colorVals = colorMap.values()[1:]

	numberOfColors = len(colorVals)

	blobList = []
	# stores for each blob in a given slice 1) the blob pixels 2) instantaneous z value and 3) n for the blob from previous slice that connected to it
	for n, color in enumerate(colorVals):

		where = np.where(image == color)
		listofpixels = zip(list(where[0]), list(where[1]))


		blob = blob(imageCount, n, listofpixels, color, [], 0, 0, 0)
		blobList.append(blob)

	pickle.dump(blobList, open('pickles/blobList' + str(imageCount) + '.p', 'wb'))

pickleGlob = glob.glob('pickles/*.p')
for imageCount, pickledBlobList in enumerate(pickleGlob):

	print '\n'
	print imageCount
	end = timer()
	print(end - start)
	print '\n'

	blobList = pickle.load(open(pickledBlobList, 'rb'))
	blobList_below = pickle.load(open(pickledGlob[imageCount+1], 'rb'))

	for blob in blobList:

		try:
			image2 = images[imageCount + 1]
		except:
			continue


		shouldSkip, seedpixel = getSeedPixel(blob.centroid, image2, blob.color)

		if shouldSkip:
			continue

		setofpixels1 = set(blob.listofpixels)

		color2 = image2[seedpixel]
		whereColor = np.where(image2==color2)
		listofpixels2 = zip(whereColor[0],whereColor[1])
		setofpixels2 = set(listofpixels2)

		blob_foundbelow = next((x for x in blobList_below if x.listofpixels == listofpixels2), None)
		if blob_foundbelow == None:
			code.interact(local=locals())


		percent_overlap = testOverlap(setofpixels1, setofpixels2)


		# cv2.circle(image1, (seedpixel[1], seedpixel[0]), 1, int(color2), -1)
		# cv2.putText(image1, str(n), (centroid1[1],centroid1[0]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, int(color2), 1,cv2.LINE_AA)
		# cv2.line(image1, (centroid1[1],centroid1[0]), (seedpixel[1], seedpixel[0]), int(color2), 1)

		#
		# if centroid2 in zTracker.keys():
		# 	if zTracker[centroid1][0] < zTracker[centroid2][0]:
		# 		del zTracker[centroid1]
		# 		continue

		if percent_overlap == 0:
			continue
		elif percent_overlap > 0.75:
			blob_foundbelow.color = color
			for pixel in blob_foundbelow.listofpixels:
				image2[pixel] = color
			# pop = zTracker.pop(centroid1)
			# zTracker[centroid2] = [pop[0] + 1, n]
		else:
			imageD = np.zeros(image1.shape, np.uint16)
			for pixel in blob_foundbelow.listofpixels:
				imageD[pixel] = color2

			# labels = waterShed(imageD, color2)



			blobs = waterShed(imageD)
			percent_overlap = 0



			if len(blobs) > 1:
				splitBlobs(blobList_below, blob_foundbelow.listofpixels, blobs)

			for b in blobs:
				pt = testOverlap(setofpixels1, set(b))
				if pt > percent_overlap:
					percent_overlap = pt
					setofpixels2 = set(b)

			blob_foundbelow.color = color
			for pixel in setofpixels2:
				image2[pixel] = color

			# pop = zTracker.pop(centroid1)
			# zTracker[centroid2] = [pop[0] + 1, n]

			# imageF = np.zeros(image1.shape, np.uint16)
			# for pixel in setofpixels1:
			# 	imageF[pixel] = color
			#
			# imageB = np.zeros(image1.shape, np.uint16)
			# for pixel in setofpixels2:
			# 	imageB[pixel] = color
			#
			# cv2.imshow('F', imageF)
			# cv2.imshow('D', imageD)
			# cv2.imshow('B', imageB)
			# cv2.waitKey()
			#
			# code.interact(local=locals())


		# cnt = np.array([[each] for each in listofpixels1],dtype='float32')

		# ctr = np.array(cnt).reshape((-1,1,2)).astype(np.int32)
		# cv2.drawContours(image1, [ctr], 0, 255, 3)

		# display_image1 = cv2.resize(image1, (0,0), fx=0.5, fy=0.5)
		# display_img1 = cv2.resize(img1, (0,0), fx=0.8, fy=0.8)

		# code.interact(local=locals())
	drawStack.append(drawList)
	pickle.dump(blobDict, open('picklesLR5/blobDict' + str(imageCount) + '.p', 'wb'))

for i, drawList in enumerate(drawStack):
	cv2.putText(images[i], str(drawList[]), (centroid1[1],centroid1[0]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, int(color2), 1,cv2.LINE_AA)
	cv2.imwrite('littleresult5/' + list_of_image_paths[imageCount][list_of_image_paths[imageCount].index('/')+1:], image1)
	print 'Saved image ' + str(i + 1) + '/' + str(len(drawStack))


code.interact(local=locals())


#while True:
	#cv2.imshow("image", image)
	#key = cv2.waitKey(20) & 0xFF
#	code.interact(local=locals())
