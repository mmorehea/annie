import numpy as np
import cv2
import code
import argparse
from Tkinter import *
from PIL import Image, ImageTk
import glob

def callback(event):
    print "called the callback!"
    rgb = "#%02x%02x%02x" % (255, 0, 0)
    status['bg'] = rgb
    print event.x, event.y

def main():
	basewidth = 600

	image_list = sorted(glob.glob('img/*.tif'))
	with Image.open(image_list[0]) as img:
		wpercent = (basewidth / float(img.size[0]))
		hsize = int((float(img.size[1]) * float(wpercent)))

	root = Tk()

	frame = Frame(root)
	frame.pack()


	scrollbar = Scrollbar(frame)
	

	listbox = Listbox(root, yscrollcommand=scrollbar.set)

	w = Canvas(frame, width = 600, height = 560)

	w.bind("<Button-1>", callback)

	object_list = []

	for image_name in image_list:

		with Image.open(image_name) as img:

			img = img.resize((basewidth, hsize), Image.ANTIALIAS)

			img = ImageTk.PhotoImage(img)
			#code.interact(local=locals())
			object_list.append(w.create_image((0,0), image = img, anchor = NW))
			print object_list

	for each in object_list:
		listbox.insert(END, each)

	scrollbar.config(command=listbox.yview)

	scrollbar.pack(side=RIGHT, fill=Y)

	listbox.pack(side=LEFT, fill=BOTH, expand=1)

	w.pack()

	#code.interact(local=locals())

	# displayNum = 1

	# while True:
	# 	for imNum in object_list:
	# 		if imNum != displayNum:
	# 			w.itemconfig(imNum, state=HIDDEN)
	# code.interact(local=locals())

	global status
	status = Label(root, text="cool status bar", bd=1, relief=SUNKEN, anchor=W)

	status.pack(side=BOTTOM, fill=X)

	mainloop()




if __name__ == '__main__':
    main()
