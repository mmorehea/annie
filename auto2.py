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
import collections
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage


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

def testOverlap(setofpixels1, setofpixels2):

	set_intersection = setofpixels1 & setofpixels2

	set_union = setofpixels1 | setofpixels2

	percent_overlap = float(len(set_intersection)) / len(set_union)

	return percent_overlap

def orderByPercentOverlap(blobs, reference):
	overlapList = []
	for blob in blobs:
		overlapList.append((testOverlap(set(reference),set(blob)), blob))


	overlapList = sorted(overlapList,key=lambda o: o[0])[::-1]
	orderedBlobs = [l[1] for l in overlapList]
	overlapVals = [l[0] for l in overlapList]

	return orderedBlobs, overlapVals

def waterShed(blob, shape):
	img = np.zeros(shape, np.uint16)
	img[zip(*blob)] = 99999

	D = ndimage.distance_transform_edt(img)
	mindist = 7
	labels = [1,2,3,4]
	while len(np.unique(labels)) > 3:
		mindist += 1
		localMax = peak_local_max(D, indices=False, min_distance=mindist, labels=img)

		markers = ndimage.label(localMax, structure=np.ones((3,3)))[0]
		labels = watershed(-D, markers, mask=img)

	subBlobs = []
	for label in np.unique(labels):
		if label == 0:
			continue
		ww = np.where(labels==label)
		bb = zip(ww[0], ww[1])
		subBlobs.append(bb)

	return subBlobs

def display(blob):

	img = np.zeros(shape, np.uint16)
	for pixel in blob:
		img[pixel] = 99999

	cv2.imshow(str(random.random()),img)
	cv2.waitKey()


# /*
# ███    ███  █████  ██ ███    ██
# ████  ████ ██   ██ ██ ████   ██
# ██ ████ ██ ███████ ██ ██ ██  ██
# ██  ██  ██ ██   ██ ██ ██  ██ ██
# ██      ██ ██   ██ ██ ██   ████
# */

################################################################################
# SETTINGS
minimum_process_length = 0
write_images_to = 'littleresult/'
write_pickles_to = 'pickles/object'
trace_objects = True
build_resultStack = True
load_stack_from_pickle_file = False
indices_of_slices_to_be_removed = []
################################################################################

dirr = sys.argv[1]

list_of_image_paths = sorted(glob.glob(dirr +'*'))

list_of_image_paths = [i for j, i, in enumerate(list_of_image_paths) if j not in indices_of_slices_to_be_removed]

shape = cv2.imread(list_of_image_paths[0],-1).shape

start = timer()

if trace_objects:

	chainLengths = []

	images = []
	for i, path in enumerate(list_of_image_paths):
		im = cv2.imread(path, -1)
		images.append(im)
	print 'Loaded ' + str(len(images)) + ' images.'

	imageArray = np.dstack(images)

	colorList = []
	for z in xrange(imageArray.shape[2]):
		colorList.extend([c for c in np.unique(imageArray[:,:,z]) if c!=0])
	colorList = list(set(colorList))

	objectCount = -1
	for z in xrange(imageArray.shape[2]):
		###Testing###
		if z != 0:
			continue
		#############
		image = imageArray[:,:,z]

		colorVals = [c for c in np.unique(image) if c!=0]
		###Testing###
		colorVals = [6228, 5724]
		#############

		blobs = []
		for color in colorVals:
			wblob = np.where(image==color)
			blob = zip(wblob[0], wblob[1])
			blobs.append(blob)

		blobs = sorted(blobs, key=len)

		###Testing###
		# testblob = blobs[94]
		# blobs = [testblob]
		#############

		for i, startBlob in enumerate(blobs):
			# print str(i+1) + '/' + str(len(blobs))

			box, dimensions = findBBDimensions(startBlob)

			color1 = image[startBlob[0]]

			startZ = z

			process = [startBlob]

			image[zip(*startBlob)] = 0

			zspace = 0
			d = 0
			terminate = False
			currentBlob = startBlob

			while terminate == False:

				zspace += 1
				blobsfound = []

				try:
					image2 = imageArray[:,:,z+zspace]
				except:
					terminate = True
					s = '0'
					continue

				window = image2[box[0]:box[1], box[2]:box[3]]

				organicWindow = image2[zip(*currentBlob)]
				frequency = collections.Counter(organicWindow).most_common()

				# if z+zspace >= 74:
				# 	img = np.zeros(shape, np.uint16)
				# 	img[zip(*currentBlob)] = 99999
				# 	code.interact(local=locals())


				if frequency[0][0] == 0:
					if d > 10:
						terminate = True
						while d > 0:
							del process[-1]
							d -= 1
						continue
					else:
						process.append([])
						d += 1
						continue

				for each in frequency:
					if each[0] == 0:
						continue
					clr, freq = each
					break

				q = np.where(image2 == clr)
				blob2 = zip(q[0],q[1])

				overlap = testOverlap(set(currentBlob), set(blob2))
				coverage = freq / float(len(organicWindow))

				if coverage > 0.75:
					if overlap > 0.75:
						blobsfound.append(blob2)
					elif overlap > 0.5 and d > 3:
						blobsfound.append(blob2)
					elif overlap > 0.1:
						subBlobs = waterShed(blob2, shape)
						subBlobs, overlapVals = orderByPercentOverlap(subBlobs, currentBlob)
						blobsfound.append(subBlobs[0])
					else:
						process.append([])
						continue
				else:
					blobsfound.append(blob2)



				if terminate == False:

					currentBlob = []
					for b in blobsfound:
						currentBlob += b

					#Probably need to do the stuff below when I terminate as well
					color1 = image2[currentBlob[0]]

					image2[zip(*currentBlob)] = 0

					process.append(currentBlob)

					box,dimensions = findBBDimensions(currentBlob)

					d = 0


			if len(process) > minimum_process_length:
				objectCount += 1

				color = colorList[objectCount]

				print '\n'
				print objectCount
				end = timer()
				print(end - start)
				print '\n'

				chainLengths.append((objectCount, color, len(process)))
				pickle.dump((startZ, process, color), open(write_pickles_to + str(objectCount) + '.p', 'wb'))

	print 'Number of chains: ' + str(len(chainLengths))
	print 'Average chain length: ' + str(sum([x[0] for x in chainLengths])/len(chainLengths))
	# print s

	if os.path.exists('summary.txt'):
		os.remove('summary.txt')

	chainLengths = sorted(chainLengths)[::-1]

	with open('summary.txt','w') as f:
		for i,each in enumerate(chainLengths):
			f.write(str(chainLengths[i][0]) + ' ' + str(chainLengths[i][1]) + ' ' + str(chainLengths[i][2]) + '\n')


if build_resultStack:

	picklePaths = sorted(glob.glob(write_pickles_to + '*.p'))

	if load_stack_from_pickle_file:
		resultArray, startO = pickle.load(open('resultArraySave.p', 'rb'))
	else:
		resultArray = np.zeros((shape[0], shape[1], len(list_of_image_paths)), np.uint16)
		startO = 0


	for o, path in enumerate(picklePaths):
		if o < startO:
			continue

		startZ, process, color = pickle.load(open(path, 'rb'))

		for z in xrange(resultArray.shape[2]):
			img = resultArray[:,:,z]

			if z < startZ:
				continue

			if z >= startZ + len(process):
				continue

			img[zip(*process[z - startZ])] = color

		pickle.dump((resultArray, o), open('resultArraySave.p,','wb'))

		print '\n'
		print 'Built object ' + str(o+1) + '/' + str(len(picklePaths))
		end = timer()
		print(end - start)
		print '\n'

	for z in xrange(resultArray.shape[2]):
		image = resultArray[:,:,z]
		cv2.imwrite(write_images_to + list_of_image_paths[z][list_of_image_paths[z].index('/')+1:], image)
