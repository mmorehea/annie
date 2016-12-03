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

def changeColor(colorSet):

	newcolor = max(colorSet) + 20
	if newcolor > 2**16:
		newcolor = random.choice(list(set(range(max(vals))[50:]) - set(vals)))

	return newcolor

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

def removeFromStack(image,blob):
	for pixel in blob:
		image[pixel] = 0

	return image

def display(blob):

	img = np.zeros(shape, np.uint16)
	for pixel in blob:
		img[pixel] = 99999

	cv2.imshow(str(random.random()),img)
	cv2.waitKey()


# img = np.zeros(shape, np.uint16)
# for pixel in blob:
# 	img[pixel] = 99999
# cv2.imshow('display',img)
# cv2.waitKey()

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
write_images_to = 'single/'
write_pickles_to = 'singlepickle/object'
trace_objects = True
build_resultStack = True
load_stack_from_pickle_file = True
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
		print 'Loaded image ' + str(i + 1) + '/' + str(len(list_of_image_paths))

	imageArray = np.dstack(images)

	objectCount = 0
	for z in xrange(imageArray.shape[2]):
		###Testing###
		if z != 0:
			continue
		#############
		image = imageArray[:,:,z]

		colorVals = [c for c in np.unique(image) if c!=0]

		blobs = []
		for color in colorVals:
			wblob = np.where(image==color)
			blob = zip(wblob[0], wblob[1])
			blobs.append(blob)

		blobs = sorted(blobs, key=len)

		###Testing###
		testblob = blobs[94]
		blobs = [testblob]
		#############

		for i, startBlob in enumerate(blobs):
			# print str(i+1) + '/' + str(len(blobs))

			resultArray = np.zeros(imageArray.shape,np.uint16)

			box, dimensions = findBBDimensions(startBlob)

			color = image[startBlob[0]]
			color1 = color

			startZ = z

			process = [startBlob]

			image = removeFromStack(image, startBlob)

			zspace = 0
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

				if frequency[0][1] / float(len(organicWindow)) > 0.8:
					if frequency[0][0] == 0:
						terminate = True
						continue
					else:
						h = np.where(image2 == frequency[0][0])
						blobsfound.append(zip(h[0],h[1]))
				else:
					# code.interact(local=locals())
					h = np.where(image2 == frequency[0][0])
					blobsfound.append(zip(h[0],h[1]))

				for blob in blobsfound:
					if len(blob) >len(currentBlob) and testOverlap(set(currentBlob), set(blob)) < 0.05:
						process.append([])
						continue

				# colorstocheck = [c for c in np.unique(window) if c != 0]
				#
				# blobstocheck = []
				# for c in colorstocheck:
				# 	wb = np.where(image2 == c)
				# 	b = zip(wb[0],wb[1])
				# 	blobstocheck.append(b)
				#
				# blobsfound = []
				#
				# ###Testing###
				# # if zspace == 14:
				# # 	code.interact(local=locals())
				# #############
				#
				# if len(blobstocheck) == 0:
				# 	terminate = True
				# 	s = '1'
				# 	# print '\t' + str(zspace)
				# elif len(blobstocheck) == 1:
				# 	olap = testOverlap(set(currentBlob), set(blobstocheck[0]))
				# 	if olap > 0.33:
				# 		blobsfound.append(blobstocheck[0])
				# 	elif olap < 0.05 and len(blobstocheck[0]) > len(currentBlob):
				# 		continue
				# 	else:
				# 		terminate = True
				# 		s = '2'
				# 		# print '\t' + str(zspace)
				# else:
				# 	blobstocheck, overlapVals = orderByPercentOverlap(blobstocheck, currentBlob)
				# 	olap = testOverlap(set(currentBlob), set(blobstocheck[0]))
				#
				# 	if olap > 0.33:
				# 		blobsfound.append(blobstocheck[0])
				# 		for b in blobstocheck[1:]:
				# 			if testOverlap(set(currentBlob), set(blobstocheck[0] + b)) > overlapVals[0]:
				# 				blobsfound.append(b)
				# 	else:
				# 		# code.interact(local=locals())
				# 		terminate = True
				# 		s = '3'
				# 		# print '\t' + str(zspace)

				if terminate == False:

					currentBlob = []
					for b in blobsfound:
						currentBlob += b

					#Probably need to do the stuff below when I terminate as well
					color1 = image2[currentBlob[0]]

					image2 = removeFromStack(image2, currentBlob)

					process.append(currentBlob)

					box,dimensions = findBBDimensions(currentBlob)


			if len(process) > minimum_process_length:
				objectCount += 1

				for i, blob in enumerate(process):
					resultArray[:,:,i][zip(*blob)] = objectCount

				print '\n'
				print objectCount
				end = timer()
				print(end - start)
				print '\n'

				chainLengths.append((len(process), color))
				pickle.dump((resultArray, startZ, startZ + len(process)), open(write_pickles_to + str(objectCount) + '.p', 'wb'))

	print 'Number of chains: ' + str(len(chainLengths))
	print 'Average chain length: ' + str(sum([x[0] for x in chainLengths])/len(chainLengths))
	# print s

	if os.path.exists('summary.txt'):
		os.remove('summary.txt')

	chainLengths = sorted(chainLengths)[::-1]

	with open('summary.txt','w') as f:
		for i,each in enumerate(chainLengths):
			f.write(str(chainLengths[i][1]) + ' ' + str(chainLengths[i][0]) + '\n')


if build_resultStack:

	picklePaths = sorted(glob.glob(write_pickles_to + '*.p'))

	colorSet = []

	if load_stack_from_pickle_file:
		resultStack = []
		for i in xrange(len(list_of_image_paths)):
			blankImg = np.zeros(shape, np.uint16)
			resultStack.append(blankImg)
		startO = 0
	else:
		resultStack, startO = pickle.load(open('resultStackSave.p', 'rb'))

	for o, path in enumerate(picklePaths):
		if o < startO:
			continue


		startZ, process, color = pickle.load(open(path, 'rb'))

		if color in colorSet:
			color = changeColor(colorSet)
		else:
			colorSet.append(color)


		for z, img in enumerate(resultStack):

			if z < startZ:
				continue

			if z >= startZ + len(process):
				continue

			for pixel in process[z - startZ]:
				img[pixel] = color

		pickle.dump((resultStack, o), open('resultStackSave.p,','wb'))

		print '\n'
		print 'Built object ' + str(o+1) + '/' + str(len(picklePaths))
		end = timer()
		print(end - start)
		print '\n'

	for z, image in enumerate(resultStack):
		cv2.imwrite(write_images_to + list_of_image_paths[z][list_of_image_paths[z].index('/')+1:], image)
