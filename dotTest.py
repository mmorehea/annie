import argparse
import cv2
import glob
import code
import numpy as np

def recSearch(pixel, img, color):
    searchPixels = [[pixel[0]+1, pixel[1]], [pixel[0]-1, pixel[1]], [pixel[0], pixel[1]+1], [pixel[0], pixel[1]-1]]
    front = [pixel]
    found = [pixel]

    while len(front) > 0:
        #print front
        fronty = front
        front = []
        for each in fronty:
            pixel = each

            searchPixels = [[pixel[0]+1, pixel[1]], [pixel[0]-1, pixel[1]], [pixel[0], pixel[1]+1], [pixel[0], pixel[1]-1]]

            for neighbor in searchPixels:
                #code.interact(local=locals())
                if img[neighbor[0], neighbor[1]] == color and neighbor not in front and neighbor not in found:
                    front.append([neighbor[0], neighbor[1]])
                    found.append([neighbor[0], neighbor[1]])
        #code.interact(local=locals())
    return found


img1 = cv2.imread("dot.tif", -1)
print img1
print np.amin(img1)
s = recSearch([16,16], img1, 0)
print s
