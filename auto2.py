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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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

def findCentroid(listofpixels):
	if len(listofpixels) == 0:
		return (0,0)
	rows = [p[0] for p in listofpixels]
	cols = [p[1] for p in listofpixels]
	try:
		centroid = int(round(np.mean(rows))), int(round(np.mean(cols)))
	except:
		# code.interact(local=locals())
		centroid = (0,0)
	return centroid

def getMeasurements(blob, shape):
	img = np.zeros(shape, np.uint16)
	img[zip(*blob)] = 1
	per = []
	for p in blob:
		x = p[0]
		y = p[1]
		q = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]

		edgePoint = False
		for each in q:
			try:
				if img[each] == 0:
					edgePoint = True
			except IndexError:
				edgePoint = True
		if edgePoint:
			per.append(p)
	return len(blob), len(per)

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

def findNearest(img, startPoint):
	directions = cycle([[0,1], [1,1], [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1]])
	increment = 0
	cycleCounter = 0
	distance = [0,0]

	if img[startPoint] > 0:
		return startPoint

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


def blobMerge(blob1, blob2, imshape):
	# from http://stackoverflow.com/questions/14730340/find-the-average-vector-shape
	if len(blob2) > len(blob1):
		blob1, blob2 = blob2, blob1

	blob1 = upperLeftJustify(blob1)
	blob2 = upperLeftJustify(blob2)

	startImg = np.zeros(imshape, np.uint16)
	startImg[zip(*blob2)] = 99999

	mergedBlob = []
	for point in blob1:
		near = findNearest(startImg, point)
		size = ((len(blob1)**0.5) + (len(blob2)**0.5))/2

		if point[0] == near[0]:
			verticalDistance = point[1] - near[1]
			if verticalDistance > 0:
				newpoint = (near[0], near[1] + 0.5 * size)
			elif verticalDistance < 0:
				newpoint = (near[0], near[1] - 0.5 * size)
			else:
				newpoint = (near[0],near[1])

		elif point[1] == near[1]:
			horizontalDistance = point[0] - near[0]
			if horizontalDistance > 0:
				newpoint = (near[0] + 0.5 * size, near[1])
			elif horizontalDistance < 0:
				newpoint = (near[0] - 0.5 * size, near[1])
			else:
				newpoint = (near[0],near[1])

		else:
			slope = float(point[1] - near[1]) / (point[0] - near[0])
			dist = 0.5 * size
			x = (dist**2/(1+slope**2))**0.5
			y = slope * x
			if point[0] < near[0]:
				x = 0-x
			if point[1] < near[1]:
				y=0-y
			newpoint = (int(near[0] + x), int(near[1] + y))

		mergedBlob.append(newpoint)
		if point == (0,0):
			code.interact(local=locals())

	return mergedBlob

def upperLeftJustify(blob):
	box, dimensions = findBBDimensions(blob)
	transformedBlob = []
	for point in blob:
		transformedPoint = (point[0] - box[0], point[1] - box[2])
		transformedBlob.append(transformedPoint)

	return transformedBlob

def upperRightJustify(blob, shape):
	box, dimensions = findBBDimensions(blob)
	transformedBlob = []
	for point in blob:
		transformedPoint = (point[0] - box[0], point[1] + shape[1] - dimensions[1] - 10)
		transformedBlob.append(transformedPoint)

	return transformedBlob

def topJustify(blob, shape):
	box, dimensions = findBBDimensions(blob)
	transformedBlob = []
	for point in blob:
		transformedPoint = (point[0] - box[0],point[1] + 0.5 * shape[1] - 0.5 * dimensions[1])
		transformedBlob.append(transformedPoint)

	return transformedBlob

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
write_images_to = '1r/'
write_pickles_to = 'p/object'
trace_objects = True
build_resultStack = True
load_stack_from_pickle_file = False
indices_of_slices_to_be_removed = []
################################################################################
def main():
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
			colorVals = [5749]
			# 6228, 5724, 7287, 9632, 2547
			# 5724 @ 880: 6758, @817: 5749
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
				# square = []
				# for row in xrange(200):
				# 	for col in xrange(200):
				# 		    square.append((row,col))
				# center = (100,100)
				# circle = []
				# for row in xrange(200):
				# 	for col in xrange(200):
				# 		if (((row-center[0])**2) + ((col-center[1])**2))**0.5 <= 100:
				# 			circle.append((row,col))
				# imm = np.zeros((1000,1000))
				# im2 = imm.copy()
				# im3 = imm.copy()
				# merged = blobMerge(square,circle,imm.shape)
				#
				# circle = topJustify(circle, imm.shape)
				# merged = upperRightJustify(merged, imm.shape)
				# imm[zip(*square)] = 99999
				# imm[zip(*circle)] = 99999
				# imm[zip(*merged)] = 99999
				#
				# cv2.imshow('a',imm)
				# cv2.waitKey()
				#
				#
				# code.interact(local=locals())
				# xs = []
				# ys = []
				# xslopes = []
				# yslopes = []
				measurementsList = []
				coverage2List = []
				coverage2Deviance = []
				noncircularityList = []
				# print str(i+1) + '/' + str(len(blobs))

				box, dimensions = findBBDimensions(startBlob)

				color1 = image[startBlob[0]]
				ogcolor = color1

				centroid1 = findCentroid(startBlob)

				startZ = z

				process = [startBlob]

				image[zip(*startBlob)] = 0

				zspace = 0
				d = 0
				terminate = False
				currentBlob = startBlob

				while terminate == False:
					measurements = getMeasurements(currentBlob, shape)
					noncircularity = (measurements[1]**2)/(4*3.1415926) - measurements[0]
					if len(noncircularityList) > 10:
						if abs(noncircularity) - abs(np.mean(np.array(noncircularityList[-10:]))) > 5 * np.std(np.array(noncircularityList[-10:])):
							terminate = True
							# code.interact(local=locals())
							continue
					# if len(noncircularityList) > 50:
					# 	if abs(noncircularity) - abs(np.mean(np.array(noncircularityList[-50:]))) > 5 * np.std(np.array(noncircularityList[-50:])):
					# 		terminate = True
					# 		continue
					measurementsList.append(measurements)
					noncircularityList.append(noncircularity)

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



					if frequency[0][0] == 0 and len(frequency) == 1:
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
					centroid2 = findCentroid(blob2)

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
							for i, sb in enumerate(subBlobs):
								if overlapVals[i] > 0.1:
									blobsfound.append(sb)
							if len(blobsfound) == 0:
								try:
									blobsfound.append(subBlobs[0])
								except:
									blobsfound.append(blob2)
						else:
							process.append([])
							continue
					else:
						blobsfound.append(blob2)

					freq2 = len(set(currentBlob) & set(blobsfound[0]))
					coverage2 = freq2 / float(len(blob2))
					measurementsList.append(coverage-coverage2)

					if zspace > 10:
						coverage2Deviance.append(coverage2 - float(sum(coverage2List[-10:]))/10)
					elif zspace > 50:
						coverage2Deviance.append(coverage2 - float(sum(coverage2List[-50:]))/50)
					else:
						coverage2Deviance.append(0)

					coverage2List.append(coverage2)

					# code.interact(local=locals())

					if terminate == False:

						newBlob = []
						for b in blobsfound:
							newBlob += b

						# if zspace == 1:
						# 	averageBlob = blobMerge(currentBlob, newBlob, shape)
						# else:
						# 	zz = averageBlob
						# 	averageBlob = blobMerge(averageBlob, newBlob, shape)
						# 	averageBlob += topJustify(zz, shape)
						# 	averageBlob += upperRightJustify(newBlob, shape)


						#Probably need to do the stuff below when I terminate as well
						color1 = image2[newBlob[0]]

						image2[zip(*newBlob)] = 0

						process.append(newBlob)

						box,dimensions = findBBDimensions(newBlob)

						d = 0

						centroid1 = findCentroid(newBlob)

						currentBlob = newBlob



				if len(process) > minimum_process_length:
					# fig = plt.figure()
					# ax = fig.gca(projection='3d')
					# xs = np.array(xs)
					# ys = np.array(ys)
					# zs = np.array(range(imageArray.shape[2]-1)[::-1])
					#
					#
					# ax.plot(xs,ys,zs)

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

		# diffarray = [(measurement[1]**2)/(4*3.1415926) - measurement[0] for measurement in measurementsList]
		#
		# ratarray = [float(measurement[1])/measurement[0] for measurement in measurementsList]

		# plt.figure(1)
		# plt.subplot(211)
		# plt.plot(zip(*measurementsList)[0])
		# plt.subplot(212)
		# plt.plot(zip(*measurementsList)[1])
		# plt.figure(2)
		# plt.plot(diffarray)
		# plt.figure(3)
		# plt.plot(ratarray)
		# plt.show()
		plt.figure(1)
		plt.plot(coverage2List)
		plt.figure(2)
		plt.plot(coverage2Deviance)
		plt.show()
		# code.interact(local=locals())


if __name__ == "__main__":
	main()
