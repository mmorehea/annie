# -*- coding: utf-8 -*-

## Add path to library (just for examples; you do not need this)

import argparse
import numpy as np
import cv2
import code
import tifffile
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import os
import glob


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
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img", required=True, help="Path to tiff stack.")
    args = vars(ap.parse_args())
    impath = args["img"]
    images = []
    if os.path.isdir(impath):
        images = glob.glob(impath + "*")
        return sorted(images)
    else:
        img = tifffile.imread(impath)
        zz, yy, xx = img.shape
        print "Loaded image of size z:" + str(zz) + " y:" + str(yy) + " x:" + str(xx)
        for i in range(zz):
            images.append(QtGui.QPixmap(img[i,:,:]))
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

        self._images = getImages()

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

            #color = QtGui.QColor.fromRgb(self.image.pixel( position ) )
            #if color.isValid():
            #    rgbColor = '('+str(color.red())+','+str(color.green())+','+str(color.blue())+','+str(color.alpha())+')'
            #    self.setWindowTitle( 'Pixel position = (' + str( event.pos().x() ) + ' , ' + str( event.pos().y() )+ ') - Value (R,G,B,A)= ' + rgbColor)
            #else:
            #self.setWindowTitle( 'Pixel position = (' + str( event.pos().x() ) + ' , ' + str( event.pos().y() )+ ') - color not valid')

if __name__ == "__main__":
    main()
