import argparse
import cv2
import glob
import code
import numpy as np
from timeit import default_timer as timer
from skimage import measure
import matplotlib.pyplot as plt

def buildColorMap(img):
	colorMap = {0: 0}
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


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", required=True, help="Path to the directory")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
list_of_images = sorted(glob.glob(args["dir"] +'*'))
start = timer()


imgPath1 = list_of_images[0]
imgs = cv2.imread(imgPath1, -1)

for imageCount in xrange(len(list_of_images)):
    if imageCount == 0:
        continue
    print 'importing image #' + str(imageCount)
    imgPath1 = list_of_images[imageCount]
    img1 = cv2.imread(imgPath1, -1)
    imgs = np.dstack((imgs, img1))

colorMap = buildColorMap(imgs)

code.interact(local=locals())
