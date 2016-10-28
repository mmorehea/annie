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


def makeGood(ids, imgs):
	good = []
	lengthIDS = len(ids)
	firstImg = imgs[0]
	lastImg = imgs[498]

	#code.interact(local=locals())
	
	for each in ids:
		if each == 0:
			continue
		print "processing " + str(each) + " of " + str(lengthIDS)
		first = np.where(firstImg == each)
		last = np.where(lastImg == each)
		if len(first[0]) == 0:
			continue
		if len(last[0]) == 0:
			continue
		#code.interact(local=locals())

		x1Max = np.amax(first[0])
		x1Min = np.amin(first[0])
		
		x2Max = np.amax(last[0])
		x2Min = np.amin(last[0])
		if (x1Max-x1Min) < 450 & (x2Max -x2Min) < 450:
			good.append(each)
	return good


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

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--img", 	required=True, help="Path to the multipage tiff")
args = vars(ap.parse_args())

impath = args['img']

imgs = tifffile.imread(impath)

ids = np.unique(imgs)

#code.interact(local=locals())

theGood = makeGood(ids, imgs)

#code.interact(local=locals())
print len(theGood)

for ii,each in enumerate(theGood):
	
	first = np.where(imgs == each)

	xMax = np.amax(first[0])
	yMax = np.amax(first[1])
	zMax = np.amax(first[2])
	xMin = np.amin(first[0])
	yMin = np.amin(first[1])
	zMin = np.amin(first[2])
	
	g = np.zeros((xMax+1,yMax+1,zMax+1))
	g[first] = 1
	verts, faces = measure.marching_cubes(g, 0)
	transVerts = []
	for v in verts:
		transVerts.append([v[0] + xMin, v[1] + yMin, v[2] + zMin])
	writeOBJ('meshes/' + "/wholevcn_" + "_input" + str(ii) + '.obj', transVerts, faces)
	print "done building obj"


#code.interact(local=locals())
