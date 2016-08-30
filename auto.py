import argparse
import cv2
import glob
import code
import numpy as np
import sys
from timeit import default_timer as timer

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


def findCentroid(listofpixels):
	Xs = [x[0] for x in listofpixels]
	Ys = [y[1] for y in listofpixels]
	centroid = (round(np.mean(Xs)), round(np.mean(Ys)))
	return centroid

def findMedian(listofpixels):
	Xs = [x[0] for x in listofpixels]
	Ys = [y[1] for y in listofpixels]
	median = (np.median(Xs),np.median(Ys))
	return median

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


# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dir", required=True, help="Path to the directory")
# args = vars(ap.parse_args())

dirr = sys.argv[1]

# load the image, clone it, and setup the mouse callback function
list_of_images = sorted(glob.glob(dirr +'*'))
start = timer()
largestSize = 0
for imageCount in xrange(len(list_of_images) - 1):
	print imageCount
	end = timer()
	print(end - start)
	imgPath1 = list_of_images[imageCount]
	imgPath2 = list_of_images[imageCount+1]

	img1 = cv2.imread(imgPath1, -1)
	img2 = cv2.imread(imgPath2, -1)
	newImg = np.copy(img2)
	if img1.shape != img2.shape:
		print "THE IMAGES ARE THE WRONG SIZE"
		break


	colorMap = buildColorMap(img1)

	# Omitting the first one because it's just 0 mapped to 0
	colorVals = colorMap.values()[1:]

	numberOfColors = len(colorVals)
	colorCount = 1

	image1 = np.zeros(img1.shape, np.uint8)

	for n, color in enumerate(colorVals):



		where = np.where(img1 == color)
		listofpixels1 = zip(list(where[0]), list(where[1]))
		setofpixels1 = set(listofpixels1)

		centroid1 = findCentroid(listofpixels1)
		median1 = findMedian(listofpixels1)


		cnt = np.array([[each] for each in listofpixels1],dtype='float32')

		ctr = np.array(cnt).reshape((-1,1,2)).astype(np.int32)
		cv2.drawContours(image1, [ctr], 0, 255, 3)


		code.interact(local=locals())

		if img2[centroid1] == 0:
			if img2[median1] == 0:
				print 'Found 0, skipping color ' + str(color) 
				continue
			else:
				seedpixel = median1
		else:
			seedpixel = centroid1

	
		listofpixels2 = recSearch(seedpixel, img2, img2[seedpixel])
		setofpixels2 = set(listofpixels2)

		set_intersection = setofpixels1 & setofpixels2
		set_union = setofpixels1 | setofpixels2
		# average_length = (len(setofpixels1) + len(setofpixels2)) / 2
		percent_overlap = float(len(set_intersection)) / len(set_union)

		#print 'first set: ' + str(len(setofpixels1))
		#print 'second set: ' + str(len(setofpixels2))
		#print 'intersection: ' + str(len(set_intersection))
		print 'overlap (' + str(n) + '/' + str(numberOfColors) + '): ' + str(percent_overlap)
		#print '\n'

	

		continue









		pixelpoints1 = np.where(img1 == color)
		pixelpoints1 = zip(pixelpoints1[0],pixelpoints1[1])
		#cnt = np.array([[each] for each in pixelpoints1],dtype='float32')

		#ctr = np.array(cnt).reshape((-1,1,2)).astype(np.int32)
		#cv2.drawContours(img1, [ctr], 0, 255, 3)

		for each in pixelpoints1: image1[each] = 255
		#code.interact(local=locals())



		# if colorCount % 100 == 0:
		# 	print str(colorCount) + ' / ' + str(numberOfColors)

		colorCount += 1
		#print "Color is " + str(color)
		firstshape = np.where(img1 == color)
		try:
			minX, maxX, minY, maxY = findBB(firstshape)
		except:
			continue
		deltaX = maxX - minX
		deltaY = maxY - minY
		# minX = deltaX * .25 + minX
		# minY = deltaY * .25 + minY
		# maxX = maxX - deltaX * .25
		# maxY = maxY - deltaY * .25
		#code.interact(local=locals())
		searchSpace = img2[minX:maxX, minY:maxY]
		#t = False
		if searchSpace.size < 50:
			continue
		searchSpaceFlat = np.ndarray.flatten(searchSpace)
		searchSpaceFlat = filter(lambda a: a != 0, searchSpaceFlat)
		if len(searchSpaceFlat) == 0:

			#print 'No Object Found'
			continue
		else:
			counts = np.bincount(searchSpaceFlat)
			mode = np.argmax(counts)
			#print "Mode is " + str(mode)
			if mode == 0:
				print "woah"
			pixelsMatch = np.where(img2 == mode)
			#newXs = pixelsMatch[0]
			#newYs = pixelsMatch[1]
			#code.interact(local=locals())
			#pixelsMatch = recSearch([newXs[0], newYs[0]], img2, mode)
			#print len(pixelsMatch)
			#code.interact(local=locals())
			try:
				newImg[pixelsMatch] = color
			except IndexError:
				continue

	code.interact(local=locals())

	cv2.imwrite(imgPath2, newImg)


#while True:
	#cv2.imshow("image", image)
	#key = cv2.waitKey(20) & 0xFF
#	code.interact(local=locals())
