# -*- coding: utf-8 -*-

## Add path to library (just for examples; you do not need this)

import argparse
import numpy as np
import cv2
import code
import tifffile
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
from PyQt4 import QtCore
import pyqtgraph as pg
import os
import glob
import sys
import inspect
import subprocess


# app = QtGui.QApplication([])
#
# ## Create window with ImageView widget
# win = QtGui.QMainWindow()
# win.resize(800,800)
# imv = pg.ImageView()
# win.setCentralWidget(imv)
# win.show()
# win.setWindowTitle('pyqtgraph example: ImageView')
#
# ## Create random 3D data set with noisy signals


# print "Loading tiff stack"
# impath = args["img"]
#
# imgs = tifffile.imread(impath)
# zz, yy, xx = imgs.shape
# for i in range(zz):
#     data = imgs[i,:,:]
#     ## Display the data and assign each frame a time value from 1.0 to 3.0
#     imv.setImage(data, xvals=np.linspace(1., 3., data.shape[0]))
def main():
    app = QtGui.QApplication([])
    w = Widget()
    w.show()
    w.raise_()
    app.exec_()

def getImages():
    impath = sys.argv[1]
    images = []
    if os.path.isdir(impath):
        image_paths = sorted(glob.glob(impath + "*"))
        images = []
        for path in image_paths:
            image = QtGui.QImage(path,'tif')
            images.append(image)
        return images
    else:
        img = tifffile.imread(impath)
        zz, yy, xx = img.shape
        print "Loaded image of size z:" + str(zz) + " y:" + str(yy) + " x:" + str(xx)
        for i in range(zz):
            images.append(buildQtImg(img[i,:,:]))
        return images

def mouseMoved(evt):
    pos = evt[0]  ## using signal proxy turns original arguments into a tuple
    if p1.sceneBoundingRect().contains(pos):
        mousePoint = vb.mapSceneToView(pos)
        index = int(mousePoint.x())
        if index > 0 and index < len(data1):
            label.setText("<span style='font-size: 12pt'>x=%0.1f,   <span style='color: red'>y1=%0.1f</span>,   <span style='color: green'>y2=%0.1f</span>" % (mousePoint.x(), data1[index], data2[index]))
        vLine.setPos(mousePoint.x())
        hLine.setPos(mousePoint.y())

def getStack():
    impath = sys.argv[1]

    if os.path.isdir(impath):
        images = glob.glob(impath + "*")
        first = True
        for ii, each in enumerate(sorted(images)):
            if ii % 100 == 0:
                print ii
            if first:
                img = tifffile.imread(each)
                xx, yy = img.shape
                imgStack = np.zeros(len(images), xx, yy)
                imgStack[ii, :, :] = img
            else:
                i = tifffile.imread(each)
                imgStack[ii, :, :] = i

        return imgStack
    else:
        img = tifffile.imread(impath)
        zz, yy, xx = img.shape
        print "Loaded image of size z:" + str(zz) + " y:" + str(yy) + " x:" + str(xx)
        return img

def buildQtImg(img):
    height, width = img.shape
    bytesPerLine = width
    qImg = QtGui.QImage(img.data, width, height, bytesPerLine)
    return qImg

def makeDisplayFile(path):
    img = tifffile.imread(path)

    xx, yy = img.shape
    newImg = np.zeros((xx, yy, 3))
    u = []
    for r in xrange(img.shape[0]):
        for c in xrange(img.shape[1]):
            if img[r,c] not in u:
                u.append(img[r,c])

    number_of_colors = len(u)

    with open('colors.txt') as f:
        colors = f.readlines()

    colors = [[int(x) for x in c[:-1].split(',')] for c in colors]

    map_16_to_8 = dict([(x,y) for x,y in zip(u, colors)])
    map_16_to_8[0] = [0, 0, 0]

    for c in map_16_to_8.keys():
        for point in zip(np.where(img==c)[0],np.where(img==c)[1]):
            newImg[point] = map_16_to_8[c]

    if not os.path.exists('display'):
        os.mkdir('display')

    newPath = 'display/' + path[path.index('/') + 1:]

    cv2.imwrite(newPath, newImg)
    print 'Writing ' + newPath

    return newPath

def mapConvert(slide):
    code.interact(local=locals())
    slide = tifffile.imread(slide)
    xx, yy = slide.shape
    newSlide = np.zeros((xx, yy, 3))
    u = []
    for r in xrange(slide.shape[0]):
        for c in xrange(slide.shape[1]):
            if slide[r,c] not in u:
                u.append(slide[r,c])

    number_of_colors = len(u)


    #subprocess.Popen(['./glasbey.py', str(number_QtCoreof_colors), 'colors.txt'])

    with open('colors.txt') as f:
        colors = f.readlines()

    colors = [[int(x) for x in c[:-1].split(',')] for c in colors]

    map_16_to_8 = dict([(x,y) for x,y in zip(u, colors)])

    for c in map_16_to_8.keys():
        for point in zip(np.where(slide==c)[0],np.where(slide==c)[1]):
            newSlide[point] = map_16_to_8[c]

    code.interact(local=locals())

    return newSlide

def makeAndSetColorTable(images):
    with open('colors.txt') as f:
        colors = f.readlines()
    #code.interact(local=locals())
    #colorVector = QtCore.Qt.QVector(len(colors))
    #colorVector = []
    colors = [[int(x) for x in c[:-1].split(',')] for c in colors]
    #code.interact(local=locals())
    colorVector = colors
    #code.interact(local=locals())
    #for color in colors:
    #    colorVector.append(QtGui.QColor.qRgb(color[0],color[1],color[2]))
    code.interact(local=locals())
    for image in images:
        image.setColorTable(colorVector)
        #code.interact(local=locals())
    return images

class Widget(QtGui.QWidget):

    def __init__(self, parent=None):
        super(Widget, self).__init__(parent)
        self.resize(1200,1200)
        self.layout = QtGui.QVBoxLayout(self)

        self.scene = QtGui.QGraphicsScene(self)
        self.view = QtGui.QGraphicsView(self.scene)
        self.layout.addWidget(self.view)

        self.image = QtGui.QGraphicsPixmapItem()
        self.scene.addItem(self.image)
        self.view.centerOn(self.image)

        # Changed this to be a list of tuples
        # self._images = [(i, makeDisplayFile(i)) for i in getImages()]
        self._images = makeAndSetColorTable(getImages())

        self.slider = QtGui.QSlider(self)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        # max is the last index of the image list
        self.slider.setMaximum(len(self._images)-1)
        self.layout.addWidget(self.slider)

        self.sliderIndex = 0
        self.sliderMoved(self.sliderIndex)
        self.slider.sliderMoved.connect(self.sliderMoved)

        self.image.mousePressEvent = self.pixelSelect
        self.lastPixelPicked = []
        self.lastColorPicked = []

    def sliderMoved(self, val):
        print "Slider moved to:", val
        try:
            self.sliderIndex = val
            self.image.setPixmap(QtGui.QPixmap(self._images[val]))

        except IndexError:
            print "Error: No image at index", val


    def pixelSelect( self, event ):
            position = QtCore.QPoint( event.pos().x(),  event.pos().y())
            #print self.lastPixelPicked
            self.lastPixelPicked = [position.x(), position.y()]
            c = self.image.pixmap().toImage().pixel(position.x(), position.y())
            colors = QtGui.QColor(c).getRgbF()
            print "(%s,%s) = %s" % (position.x(), position.y(), colors)
            mask1 = self.image.pixmap().createMaskFromColor(QtGui.QColor(c), QtCore.Qt.MaskOutColor)
            cover = QtGui.QGraphicsPixmapItem()



            #color = QtGui.QColor.fromRgb(self.image.pixel( position ) )
            #if color.isValid():
            #    rgbColor = '('+str(color.red())+','+str(color.green())+','+str(color.blue())+','+str(color.alpha())+')'
            #    self.setWindowTitle( 'Pixel position = (' + str( event.pos().x() ) + ' , ' + str( event.pos().y() )+ ') - Value (R,G,B,A)= ' + rgbColor)
            #else:
            #self.setWindowTitle( 'Pixel position = (' + str( event.pos().x() ) + ' , ' + str( event.pos().y() )+ ') - color not valid')

if __name__ == "__main__":
    main()
