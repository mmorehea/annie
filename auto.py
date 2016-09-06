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

def waterShedSearch(searchSpace, img, color):
	viablePixels = np.where(searchSpace == color)
	startX = viablePixels[0][0]
	startY = viablePixels[1][0]
	return recSearch([startX, startY], img, color)

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
	centroid = int(round(np.mean(rows))), int(round(np.mean(cols)))
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
def getSeedPixel(centroid1, images, imageCount, color, zspace):
	shouldSkip = False
	seedpixel = (0,0)
	try:
		img = images[imageCount + zspace]
	except:
		shouldSkip = True
		print 'Index out of bounds'
		return shouldSkip, seedpixel

	if img[centroid1] == 0:
		seedpixel = findNearest(img, centroid1)
	else:
		seedpixel = centroid1
	try:
		if img[seedpixel] == color:
			shouldSkip = True
			print 'Same color! Skipping color ' + color
	except:
		shouldSkip = True
		print 'Index out of bounds'

	return shouldSkip, seedpixel


# /*
# ████████ ███████ ███████ ████████  ██████  ██    ██ ███████ ██████  ██       █████  ██████
#    ██    ██      ██         ██    ██    ██ ██    ██ ██      ██   ██ ██      ██   ██ ██   ██
#    ██    █████   ███████    ██    ██    ██ ██    ██ █████   ██████  ██      ███████ ██████
#    ██    ██           ██    ██    ██    ██  ██  ██  ██      ██   ██ ██      ██   ██ ██
#    ██    ███████ ███████    ██     ██████    ████   ███████ ██   ██ ███████ ██   ██ ██
# */

def testOverlap(setofpixels1, image2, seedpixel):
	color2 = image2[seedpixel]
	whereColor = np.where(image2==color2)
	listofpixels2 = zip(whereColor[0],whereColor[1])
	setofpixels2 = set(listofpixels2)

	set_intersection = setofpixels1 & setofpixels2
	set_union = setofpixels1 | setofpixels2

	percent_overlap = float(len(set_intersection)) / len(set_union)

	return percent_overlap, setofpixels2

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

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dir", required=True, help="Path to the directory")
# args = vars(ap.parse_args())

dirr = sys.argv[1]

# load the image, clone it, and setup the mouse callback function
list_of_image_paths = sorted(glob.glob(dirr +'*'))

images = []
for i, path in enumerate(list_of_image_paths):
	im = cv2.imread(path, -1)
	images.append(im)
	print 'Loaded image ' + str(i + 1) + '/' + str(len(list_of_image_paths))

start = timer()

# stopAt = 50
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

	# image1 = np.zeros(img1.shape, np.uint8)


	# For filtering the shapes by size and writing to another folder
	# toRemove = []
	# for color in colorVals:
	# 	where = np.where(image1 == color)
	# 	listofpixels1 = zip(list(where[0]), list(where[1]))
	# 	setofpixels1 = set(listofpixels1)
	# 	if len(setofpixels1) < 12000:
	# 		toRemove.append(color)
	# 		for pixel in setofpixels1:
	# 			image1[pixel] = 0
	# for each in toRemove:
	# 	colorVals.remove(each)
	# cv2.imwrite('crop2/' + list_of_image_paths[imageCount][list_of_image_paths[imageCount].index('/')+1:], image1)
	# continue



	for n, color in enumerate(colorVals):
		print 'Image ' + str(imageCount + 1) + '/' + str(len(images)) + ', '+ 'Color ' + str(n + 1) + '/' + str(len(colorVals))

		# print color


		where = np.where(image1 == color)
		listofpixels1 = zip(list(where[0]), list(where[1]))
		setofpixels1 = set(listofpixels1)


		centroid1 = findCentroid(listofpixels1)

		# Makes a purple centroid
		# A cv point is defined by column, row, opposite to a numpy array
		# cv2.circle(image1, (centroid1[1], centroid1[0]), 5, 7283, -1)

		# Need condition for when its the same color

		percent_overlap = 0
		zspace = 0

		while percent_overlap < 0.5:
			zspace += 1

			if zspace > 7:
				shouldSkip = True
			else:
				# If it can't find a color below, it will skip that color
				shouldSkip, seedpixel = getSeedPixel(centroid1, images, imageCount, color, zspace)
				# cv2.circle(image1, (seedpixel[1], seedpixel[0]), 5, 6383, -1)

			if shouldSkip:
				break

			image2 = images[imageCount + zspace]
			percent_overlap, setofpixels2 = testOverlap(setofpixels1, image2, seedpixel)

			# if percent_overlap < 0.1:
			# 	shouldSkip = True
			# 	break

			# print 'Percent overlap: ' + str(percent_overlap)
			# if zspace > 1:
			# 	print '\tzspace: ' + str(zspace)
				# code.interact(local=locals())

		if shouldSkip:
			continue

		# print image2[list(setofpixels2)[0]]
		for pixel in setofpixels2:
			image2[pixel] = color
		# print image2[list(setofpixels2)[0]]



		# Might want to interpolate here, but unsure of implementation




		# cnt = np.array([[each] for each in listofpixels1],dtype='float32')

		# ctr = np.array(cnt).reshape((-1,1,2)).astype(np.int32)
		# cv2.drawContours(image1, [ctr], 0, 255, 3)

		# display_image1 = cv2.resize(image1, (0,0), fx=0.5, fy=0.5)
		# display_img1 = cv2.resize(img1, (0,0), fx=0.8, fy=0.8)

		# code.interact(local=locals())
	cv2.imwrite('result/' + list_of_image_paths[imageCount][list_of_image_paths[imageCount].index('/')+1:], image1)

code.interact(local=locals())


#while True:
	#cv2.imshow("image", image)
	#key = cv2.waitKey(20) & 0xFF
#	code.interact(local=locals())
