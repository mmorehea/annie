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

def getSeedPixel(centroid1, median1, images, color, zspace):
	shouldSkip = False
	img = images[imageCount + zspace]

	if img[centroid1] == 0:
			if img[median1] == 0:
				print 'Found 0, skipping color ' + str(color) 
				shouldSkip = True
			else:
				seedpixel = median1
		else:
			seedpixel = centroid1

	return shouldSkip, seedpixel



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
list_of_image_paths = sorted(glob.glob(dirr +'*'))

images = []
for path in list_of_image_paths:
	im = cv2.imread(path)
	images.append(im)

start = timer()

for imageCount, image in enumerate(images):
	print imageCount
	end = timer()
	print(end - start)
	

	colorMap = buildColorMap(image)

	# Omitting the first one because it's just 0 mapped to 0
	colorVals = colorMap.values()[1:]

	numberOfColors = len(colorVals)

	# image1 = np.zeros(img1.shape, np.uint8)

	for n, color in enumerate(colorVals):

		zspace = 1

		where = np.where(img1 == color)
		listofpixels1 = zip(list(where[0]), list(where[1]))
		setofpixels1 = set(listofpixels1)

		centroid1 = findCentroid(listofpixels1)
		median1 = findMedian(listofpixels1)



		# Need a condition for if the color is already the same?

		percent_overlap = 1

		while percent_overlap < 0.5:

			shouldSkip, seedpixel = getSeedPixel(centroid1, median1, images, color, zspace)
			if shouldSkip:
				break

			percent_overlap, setofpixels2 = testOverlap(setofpixels1, imageCount + zspace, seedpixel)
			zspace += 1


	else:

		for pixel in setofpixels2:
			images[imageCount + zspace] = color


		# cnt = np.array([[each] for each in listofpixels1],dtype='float32')

		# ctr = np.array(cnt).reshape((-1,1,2)).astype(np.int32)
		# cv2.drawContours(image1, [ctr], 0, 255, 3)

		# display_image1 = cv2.resize(image1, (0,0), fx=0.5, fy=0.5)
		# display_img1 = cv2.resize(img1, (0,0), fx=0.8, fy=0.8)		

		# code.interact(local=locals())

		

	
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
