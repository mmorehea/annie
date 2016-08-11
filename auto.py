import argparse
import cv2
import glob
import code
import numpy as np
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
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True, help="Path to the directory")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
list_of_images = sorted(glob.glob(args["dir"] +'*'))
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
	numberOfColors = len(colorMap.values()[1:])
	colorCount = 1
	for color in colorMap.values()[1:]:
		# if colorCount % 100 == 0:
		# 	print str(colorCount) + ' / ' + str(numberOfColors)

		colorCount += 1
		#print "Color is " + str(color)
		firstshape = np.where(img1 == color)
		minX, maxX, minY, maxY = findBB(firstshape)
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
			#if t:
			#	code.interact(local=locals())
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

	cv2.imwrite(imgPath2, newImg)


#while True:
	#cv2.imshow("image", image)
	#key = cv2.waitKey(20) & 0xFF
#	code.interact(local=locals())
