import argparse
import cv2
import glob
import code
import numpy as np
from timeit import default_timer as timer
from skimage import measure
import matplotlib.pyplot as plt

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
ap.add_argument("-d", "--dir", required=True, help="Path to the directory")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
#list_of_images = sorted(glob.glob(args["dir"] +'*'))
start = timer()

impath = 'vcn_tracked_tiffs1.tif'

imgs = tifffile.imread(impath)
#imgPath1 = list_of_images[0]
#imgs = cv2.imread(imgPath1, -1)

#for imageCount in xrange(len(list_of_images)):
#    if imageCount == 0:
#        continue
#    print 'importing image #' + str(imageCount)
#    imgPath1 = list_of_images[imageCount]
#    img1 = cv2.imread(imgPath1, -1)
#    imgs = np.dstack((imgs, img1))

colorMap = buildColorMap(imgs)
allSizes = []

objs_processed = glob.glob("./meshes/*")

for each in colorMap.keys()[1:]:
	if "./meshes/" + str(each) + ".obj" in objs_processed:
		continue
	first = np.where(imgs == colorMap[each])
	if len(first[0])*len(first[1])*len(first[2]) < 400:
		continue
	xMax = np.amax(first[0])
	yMax = np.amax(first[1])
	zMax = np.amax(first[2])
	xMin = np.amin(first[0])
	yMin = np.amin(first[1])
	zMin = np.amin(first[2])
	s = (xMax-xMin) * (yMax -yMin) * (zMax- zMin)
	allSizes.append(s)
	if s < 1000000:
		continue
	print each
	g = np.zeros((xMax+1,yMax+1,zMax+1))
	g[first] = 1
	verts, faces = measure.marching_cubes(g, 0)
	transVerts = []
	for v in verts:
		transVerts.append([v[0] + xMin, v[1] + yMin, v[2] + zMin])
	writeOBJ('meshes/' + str(each) + '.obj', transVerts, faces)

code.interact(local=locals())
