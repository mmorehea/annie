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
import random


def findBBDimensions(listofpixels):
	if len(listofpixels) == 0:
		return None
	else:
		xs = [x[0] for x in listofpixels]
		ys = [y[1] for y in listofpixels]

		minxs = min(xs)
		maxxs = max(xs)

		minys = min(ys)
		maxys = max(ys)

		dx = max(xs) - min(xs)
		dy = max(ys) - min(ys)


		return [minxs, maxxs, minys, maxys], [dx, dy]


# /*
# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████
# */

################################################################################
# SETTINGS
label_and_collect_info = False # Takes a lot more time but labels all blobs and collects info on each for use with dauto.py, good for testing.
write_images_to = 'littleresult/'
write_pickles_to = 'pickles3/blobList' # Only matters if label_and_collect_info is true
indices_of_slices_to_be_removed = []
################################################################################

dirr = sys.argv[1]

list_of_image_paths = sorted(glob.glob(dirr +'*'))

list_of_image_paths = [i for j, i, in enumerate(list_of_image_paths) if j not in indices_of_slices_to_be_removed]

images = []
for i, path in enumerate(list_of_image_paths):
	im = cv2.imread(path, -1)
	images.append(im)
	print 'Loaded image ' + str(i + 1) + '/' + str(len(list_of_image_paths))

start = timer()


for image in images:

	colorVals = [c for c in np.unique(image) if c!=0]

	blobs = []
	for color in colorVals:
		wblob = np.where(image==color)
		blob = zip(wblob[0], wblob[1])
		blobs.append(blob)

	blobs = sorted(blobs, key=len)

	for startBlob in blobs:

		box, dimensions = findBBDimensions(startBlob)

		process = [[startBlob]]

		zspace = 1
		while terminate == False:

			image2 = images[imageCount + zspace]

			view = image2[box[0]:box[1], box[2]:box[3]]

			colorstocheck = [c for c in np.unique(view) if c != 0]

			blobstocheck = []
			for c in colorstocheck:
				wb = np.where(image == c)
				b = zip(wb[0],wb[1])
				blobstocheck.append(b)

			blobsfound = []
			for b in blobstocheck:
				if len(set(startBlob) & set(b)) > 10:
					for pixel in b:
						image2[pixel] = 0


					blobsfound.append(b)

			process.append(blobsfound)
