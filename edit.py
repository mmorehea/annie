# USAGE
# python click_and_crop.py --image jurassic_park_kitchen.jpg

# import the necessary packages
import argparse
import cv2
import code

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
keepgoing = True

def click(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
	#print x, y
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		keepgoing = False

	global keepgoing




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click)

# keep looping until the 'q' key is pressed
while keepgoing:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(20) & 0xFF


print refPt[0]
print image[refPt[0]]

# if there are two reference points, then crop the region of interest
# from teh image and display it



# close all open windows
cv2.destroyAllWindows()
