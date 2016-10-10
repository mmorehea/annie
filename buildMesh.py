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


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--img", required=True, help="Path to the multipage tiff")
ap.add_argument("-v", "--val", required=True, help="Pixel Value to extract")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
#list_of_images = sorted(glob.glob(args["dir"] +'*'))
start = timer()
#code.interact(local=locals())
impath = args['img']

val = int(args['val'])

imgs = tifffile.imread(impath)

#code.interact(local=locals())
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


print "beginning search for objs"
count = 0
folder = "c1"



first = np.where(imgs == val)
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
writeOBJ('meshes/' + "wholevcn_" + folder + "_input" + str(val) + '.obj', transVerts, faces)
print "done building obj"


code.interact(local=locals())
