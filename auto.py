# USAGE
# python click_and_crop.py --image jurassic_park_kitchen.jpg

# import the necessary packages
import argparse
import cv2
import glob
import code
import numpy as np
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not

#def findNextLabel(img1, img2):
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


def recSearch(pixel, img, color, listOfPixels, listOfPixelsToSearch):

	if pixel not in listOfPixels:
		listOfPixels.append(pixel)
		listOfPixelsToSearch.append(pixel)

	while len(listOfPixelsToSearch) > 0:
		pixel = listOfPixelsToSearch.pop()
		searchPixels = [[pixel[0]+1, pixel[1]], [pixel[0]-1, pixel[1]], [pixel[0], pixel[1]+1], [pixel[0], pixel[1]-1]]
		count = 0
		for each in searchPixels:
			if img[each] == color and each not in listOfPixels:
				listOfPixels.append(each)
				listOfPixelsToSearch.append(each)
			else:
				count += 1
		if count == 4:
			continue

		listOfPixelsToSearch.remove(pixel)








# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True, help="Path to the directory")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
list_of_images = sorted(glob.glob(args["dir"] +'*'))

for imageCount in xrange(len(list_of_images) - 1):
	imgPath1 = list_of_images[imageCount]
	imgPath2 = list_of_images[imageCount+1]

	img1 = cv2.imread(imgPath1, -1)
	img2 = cv2.imread(imgPath2, -1)

	colorMap = buildColorMap(img1)

	for color in colorMap.values()[1:]:
		firstshape = np.where(img1 == color)
		minX, maxX, minY, maxY = findBB(firstshape)

		searchSpace = img2[minX:maxX, minY:maxY]
		if searchSpace.size < 17:
			continue
		searchSpaceFlat = np.ndarray.flatten(searchSpace)
		searchSpaceFlat = filter(lambda a: a != 0, searchSpaceFlat)
		if len(searchSpaceFlat) == 0:
			print 'No Object Found'
			continue
		else:
			counts = np.bincount(searchSpaceFlat)
			mode = np.argmax(counts)

			pixelsMatch = np.where(img2[minX:maxX, minY:maxY] == mode)
			newXs = pixelsMatch[0] + minX
			newYs = pixelsMatch[1] + minY
			pixelsMatch = [[newXs], [newYs]]
			#code.interact(local=locals())
			img2[pixelsMatch] = color

cv2.imwrite("test.tiff", img2)


#while True:
	#cv2.imshow("image", image)
	#key = cv2.waitKey(20) & 0xFF
#	code.interact(local=locals())
