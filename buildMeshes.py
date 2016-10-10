import argparse
import cv2
import os
import glob
import code
import numpy as np
from timeit import default_timer as timer
from skimage import measure
import matplotlib.pyplot as plt
import tifffile

def writeOBJ(filepath, verts, faces):
	with open(filepath, 'w') as f:
	    f.write("# OBJ file\n")
	    for v in verts:
		f.write("v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + " \n")
	    for p in faces:
		f.write("f")
		for i in p:
		    f.write(" %d" % (i + 1))
		f.write("\n")

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
ap.add_argument("-i", "--img", required=True, help="Path to the multipage tiff")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
#list_of_images = sorted(glob.glob(args["dir"] +'*'))
start = timer()

impath = 'maher_results1.tif'

imgs = tifffile.imread(impath)

print "loaded image stack"
#imgPath1 = list_of_images[0]
#imgs = cv2.imread(imgPath1, -1)

#for imageCount in xrange(len(list_of_images)):
#    if imageCount == 0:
#        continue
#    print 'importing image #' + str(imageCount)
#    imgPath1 = list_of_images[imageCount]
#    img1 = cv2.imread(imgPath1, -1)
#    imgs = np.dstack((imgs, img1))
print "building colormap"
colorMap = buildColorMap(imgs)
allSizes = []

objs_processed = glob.glob("./meshes/*")
print "beginning search for objs"
count = 0
folder = "c1"
os.mkdir("./meshes/" + folder)
for each in colorMap.keys()[1:]:
	if each % 21 == 0:
		count += 1
		folder = "c" + str(count)
		os.mkdir("./meshes/" + folder)
	first = np.where(imgs == colorMap[each])

	xMax = np.amax(first[0])
	yMax = np.amax(first[1])
	zMax = np.amax(first[2])
	xMin = np.amin(first[0])
	yMin = np.amin(first[1])
	zMin = np.amin(first[2])
	s = zMax- zMin
	allSizes.append(s)
	if s < 400:
		continue
	print "found suitable object: " + str(each)
	g = np.zeros((xMax+1,yMax+1,zMax+1))
	g[first] = 1
	verts, faces = measure.marching_cubes(g, 0)
	transVerts = []
	for v in verts:
		transVerts.append([v[0] + xMin, v[1] + yMin, v[2] + zMin])
	writeOBJ('meshes/' + folder + "/wholevcn_" + folder + "_input" + str(count) + '.obj', transVerts, faces)
	print "done building obj"


code.interact(local=locals())
