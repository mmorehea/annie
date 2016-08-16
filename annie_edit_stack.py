import argparse
import matplotlib
matplotlib.use('GTKAgg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import code
import tifffile
from matplotlib.widgets import Slider, Button, RadioButtons
from timeit import default_timer as timer

def on_mousescroll(self, event):
    if event.button == 'down':
        axplot = fig.add_axes([0.07,0.25,0.90,0.70])
        axplot.imshow(imgs[current+1, :, :])
    elif event.button == 'up':
        self.prev_plot()
    else:
        return
    self.fig.canvas.draw()



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img", required=True, help="Path to tiff stack")
    args = vars(ap.parse_args())

    x = np.linspace(0, 10, 100)

    print "Loading tiff stack"
    impath = args["img"]

    imgs = tifffile.imread(impath)
    zz, yy, xx = imgs.shape

    print "Loaded image of size z:" + str(zz) + " y:" + str(yy) + " x:" + str(xx)
    axes = AxesSequence()
    for i in range(zz):
        axes.new(imgs[i, :, :], i)


    axes.show()
    # fig = plt.figure()
    #
    # axplot = fig.add_axes([0.07,0.25,0.90,0.70])
    # axplot.imshow(imgs[0, :, :])
    # fig.canvas.mpl_connect('scroll_event', on_mousescroll)
    #
    # fig.show()
    # plt.show()


class AxesSequence(object):
    """Creates a series of axes in a figure where only one is displayed at any
    given time. Which plot is displayed is controlled by the arrow keys."""
    def __init__(self):
        self.fig = plt.figure()
        self.axes = {}
        self._i = 0 # Currently displayed axes index
        self._n = 0 # Last created axes index
        self.fig.canvas.mpl_connect('scroll_event', self.on_mousescroll)


    def new(self, slice, sliceNum):
        # The label needs to be specified so that a new axes will be created
        # instead of "add_axes" just returning the original one.
        print self._n
        ax = self.fig.add_axes([0.15, 0.1, 0.8, 0.8], visible=False, label=self._n)
        ax.imshow(slice)
        ax.set_title(sliceNum)
        self.axes[self._n] = ax
        self._n += 1

        return ax

    def on_mousescroll(self, event):
        start = timer()
        if event.button == 'down':
            self.next_plot()
        elif event.button == 'up':
            self.prev_plot()
        else:
            return
        self.fig.canvas.draw()
        end = timer()
        print "mousescroll " + str(end - start)

    def next_plot(self):
        start = timer()
        if self._i <= len(self.axes)-2:
            self.axes[self._i].set_visible(False)
            self.axes[self._i+1].set_visible(True)
            self._i += 1
        end = timer()
        print "nextplot " + str(end - start)

    def prev_plot(self):
        if self._i > 0:
            self.axes[self._i].set_visible(False)
            self.axes[self._i-1].set_visible(True)
            self._i -= 1

    def show(self):
        self.axes[0].set_visible(True)
        plt.show()

if __name__ == '__main__':
    main()
