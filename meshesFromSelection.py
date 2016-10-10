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
from numpy import genfromtxt

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
ap.add_argument("-i", "--img", required=True, help="Path to the multipage tiff")
ap.add_argument("-c", "--csv", required=True, help="CSV file to extract")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
#list_of_images = sorted(glob.glob(args["dir"] +'*'))
start = timer()

impath = args['img']
csv = args['csv']

my_data = genfromtxt(csv, delimiter=',')
imgs = tifffile.imread(impath)

surfacesToMesh = np.unique(my_data)
folder = "result"
print "entering loop"
for each in surfacesToMesh[2:]:
	
	first = np.where(imgs == int(each))
	if len(first[0]) == 0:
		continue
	#code.interact(local=locals())
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
	writeOBJ('meshes/' + "/wholevcn_" + folder + "_input" + str(each) + '.obj', transVerts, faces)
	print "done building obj " + str(each)


code.interact(local=locals())
