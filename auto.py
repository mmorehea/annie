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

np.set_printoptions(threshold=np.inf)

def findBB(dlist):
	xs = dlist[0]
	ys = dlist[1]

	return min(xs), max(xs), min(ys), max(ys)

def buildColorMap(img):
	colorMap = {0: 0}
	x, y = img.shape
	counter = 0
	uniqueValues = np.unique(img)
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
	rows = [p[0] for p in listofpixels]
	cols = [p[1] for p in listofpixels]
	try:
		centroid = int(round(np.mean(rows))), int(round(np.mean(cols)))
	except:
		code.interact(local=locals())
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
	im = cv2.imread(path, -1)
	images.append(im)
	print 'Loaded image ' + str(i + 1) + '/' + str(len(list_of_image_paths))

start = timer()

for imageCount, image1 in enumerate(images):


	print '\n'
	print imageCount
	end = timer()
	print(end - start)
	print '\n'


	colorMap = buildColorMap(image1)

	# Omitting the first one because it's just 0 mapped to 0
	colorVals = colorMap.values()[1:]

	numberOfColors = len(colorVals)


	# For filtering the shapes by size and writing to another folder
	# toRemove = []
	# for color in colorVals:
	# 	where = np.where(image1 == color)
	# 	listofpixels1 = zip(list(where[0]), list(where[1]))
	# 	setofpixels1 = set(listofpixels1)
	# 	if len(setofpixels1) > 700:
	# 		toRemove.append(color)
	# 		for pixel in setofpixels1:
	# 			image1[pixel] = 0
	# for each in toRemove:
	# 	colorVals.remove(each)
	# cv2.imwrite('littlecrop/' + list_of_image_paths[imageCount][list_of_image_paths[imageCount].index('/')+1:], image1)
	# continue



	for n, color in enumerate(colorVals):
		# print 'Image ' + str(imageCount + 1) + '/' + str(len(images)) + ', '+ 'Color ' + str(n + 1) + '/' + str(len(colorVals))

		where = np.where(image1 == color)
		listofpixels1 = zip(list(where[0]), list(where[1]))
		setofpixels1 = set(listofpixels1)

		centroid1 = findCentroid(listofpixels1)

		# Makes a purple centroid
		# A cv point is defined by column, row, opposite to a numpy array
		# cv2.circle(image1, (centroid1[1], centroid1[0]), 5, 7283, -1)

		# Need condition for when its the same color

		try:
			image2 = images[imageCount + 1]
		except:
			continue


		shouldSkip, seedpixel = getSeedPixel(centroid1, image2, color)

		if shouldSkip:
			continue

		color2 = image2[seedpixel]
		whereColor = np.where(image2==color2)
		listofpixels2 = zip(whereColor[0],whereColor[1])
		setofpixels2 = set(listofpixels2)

		percent_overlap = testOverlap(setofpixels1, setofpixels2)


		# cv2.circle(image1, (seedpixel[1], seedpixel[0]), 4, 6383, -1)

		if percent_overlap > 0.75:
			for pixel in setofpixels2:
				image2[pixel] = color
		else:
			imageD = np.zeros((image1.shape[0], image1.shape[1]))
			for pixel in setofpixels2:
				imageD[pixel] = color2

			# labels = waterShed(imageD, color2)

			# imageF = np.zeros((image1.shape[0], image1.shape[1]))
			# for pixel in setofpixels1:
			# 	imageF[pixel] = color

			# if imageCount == 0 and n == 95:
				# cv2.imshow('F', imageF)
				# cv2.imshow('D', imageD)
				# cv2.waitKey()
			blobs = waterShed(imageD)
			percent_overlap = 0

			# NEED TO REGULATE FOR BLOBS BEING < length 2

			for blob in blobs:
				pt = testOverlap(setofpixels1, set(blob))
				if pt > percent_overlap:
					percent_overlap = pt
					setofpixels2 = set(blob)


			for pixel in setofpixels2:
				image2[pixel] = color

		# cnt = np.array([[each] for each in listofpixels1],dtype='float32')

		# ctr = np.array(cnt).reshape((-1,1,2)).astype(np.int32)
		# cv2.drawContours(image1, [ctr], 0, 255, 3)

		# display_image1 = cv2.resize(image1, (0,0), fx=0.5, fy=0.5)
		# display_img1 = cv2.resize(img1, (0,0), fx=0.8, fy=0.8)

		# code.interact(local=locals())
	cv2.imwrite('result5/' + list_of_image_paths[imageCount][list_of_image_paths[imageCount].index('/')+1:], image1)

code.interact(local=locals())


#while True:
	#cv2.imshow("image", image)
	#key = cv2.waitKey(20) & 0xFF
#	code.interact(local=locals())
