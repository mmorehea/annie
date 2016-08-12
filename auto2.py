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



	#contours,hierarchy = cv2.findContours(image1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours(image1, contours, -1, 255, 3)

	cntList1 = []
	cntColorDict1 = {}

	for color in colorMap.values()[1:]:

		image1 = np.zeros(img1.shape, np.uint8)

		image2 = np.zeros(img2.shape, np.uint8)

		pixelpoints1 = np.where(img1 == color)
		pixelpoints1 = zip(pixelpoints1[0],pixelpoints1[1])
		#cnt = np.array([[each] for each in pixelpoints1],dtype='float32')

		#ctr = np.array(cnt).reshape((-1,1,2)).astype(np.int32)
		#cv2.drawContours(img1, [ctr], 0, 255, 3)

		for each in pixelpoints1: image1[each] = 255


		pixelpoints2 = np.where(img2 == color)
		pixelpoints2 = zip(pixelpoints2[0],pixelpoints2[1])

		for each in pixelpoints2: image2[each] = 255

		kernel = np.ones((5,5),np.uint8)
		image1 = cv2.erode(image1,kernel,iterations = 1)
		image1 = cv2.dilate(image1,kernel,iterations = 1)


		contours, hierarchy = cv2.findContours(np.copy(image1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# contour = contours[0]

		# cntColorDict1[contour] = color

		# cntList1.append(contour)
		if len(contours) > 1:
			code.interact(local=locals())



		# if colorCount % 100 == 0:
		# 	print str(colorCount) + ' / ' + str(numberOfColors)

		# colorCount += 1
		# #print "Color is " + str(color)
		# firstshape = np.where(img1 == color)
		# try:
		# 	minX, maxX, minY, maxY = findBB(firstshape)
		# except:
		# 	continue
		# deltaX = maxX - minX
		# deltaY = maxY - minY
		# minX = deltaX * .25 + minX
		# minY = deltaY * .25 + minY
		# maxX = maxX - deltaX * .25
		# maxY = maxY - deltaY * .25
		#code.interact(local=locals())
		#searchSpace = img2[minX:maxX, minY:maxY]
		#t = False
		# if searchSpace.size < 50:
		# 	continue
		# searchSpaceFlat = np.ndarray.flatten(searchSpace)
		# searchSpaceFlat = filter(lambda a: a != 0, searchSpaceFlat)
		# if len(searchSpaceFlat) == 0:

			#print 'No Object Found'
		# 	continue
		# else:
		# 	counts = np.bincount(searchSpaceFlat)
		# 	mode = np.argmax(counts)
		# 	#print "Mode is " + str(mode)
			# if mode == 0:
			# 	print "woah"
			# pixelsMatch = np.where(img2 == mode)
			# #newXs = pixelsMatch[0]
			#newYs = pixelsMatch[1]
			#code.interact(local=locals())
			#pixelsMatch = recSearch([newXs[0], newYs[0]], img2, mode)
			#print len(pixelsMatch)
			#code.interact(local=locals())
	# 		try:
	# 			newImg[pixelsMatch] = color
	# 		except IndexError:
	# 			continue

	# cv2.imwrite(imgPath2, newImg)


#while True:
	#cv2.imshow("image", image)
	#key = cv2.waitKey(20) & 0xFF
#	code.interact(local=locals())
