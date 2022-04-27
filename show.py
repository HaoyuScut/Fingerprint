import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from finger import *
import cv2
import numpy as np
from scipy import ndimage
from scipy import signal
from math import *
import scipy

class MyWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

    def normalise(img):
        normed = (img - np.mean(img)) / (np.std(img))
        return normed

    def ridge_segment(self,im, blksze, thresh):  # img,16,0.1

        rows, cols = im.shape

        im = self.normalise(im)  # normalise to get zero mean and unit standard deviation 归一化？
        # imshow("norm",im)

        new_rows = np.int(blksze * np.ceil((np.float(rows)) / (np.float(blksze))))
        new_cols = np.int(blksze * np.ceil((np.float(cols)) / (np.float(blksze))))

        padded_img = np.zeros((new_rows, new_cols))
        stddevim = np.zeros((new_rows, new_cols))

        padded_img[0:rows][:, 0:cols] = im

        for i in range(0, new_rows, blksze):
            for j in range(0, new_cols, blksze):
                block = padded_img[i:i + blksze][:, j:j + blksze]

                stddevim[i:i + blksze][:, j:j + blksze] = np.std(block) * np.ones(block.shape)

        stddevim = stddevim[0:rows][:, 0:cols]

        mask = stddevim > thresh

        mean_val = np.mean(im[mask])

        std_val = np.std(im[mask])

        normim = (im - mean_val) / (std_val)
        # imshow("norm",normim)

        return (normim, mask)

    def ridge_orient(self,im, gradientsigma, blocksigma, orientsmoothsigma):
        rows, cols = im.shape
        # Calculate image gradients.
        sze = np.fix(6 * gradientsigma)
        if np.remainder(sze, 2) == 0:
            sze = sze + 1

        gauss = cv2.getGaussianKernel(np.int(sze), gradientsigma)
        f = gauss * gauss.T

        fy, fx = np.gradient(f)  # Gradient of Gaussian

        # Gx = ndimage.convolve(np.double(im),fx);
        # Gy = ndimage.convolve(np.double(im),fy);

        Gx = signal.convolve2d(im, fx, mode='same')
        Gy = signal.convolve2d(im, fy, mode='same')

        Gxx = np.power(Gx, 2)
        Gyy = np.power(Gy, 2)
        Gxy = Gx * Gy

        # Now smooth the covariance data to perform a weighted summation of the data.

        sze = np.fix(6 * blocksigma)

        gauss = cv2.getGaussianKernel(np.int(sze), blocksigma)
        f = gauss * gauss.T

        Gxx = ndimage.convolve(Gxx, f)
        Gyy = ndimage.convolve(Gyy, f)
        Gxy = 2 * ndimage.convolve(Gxy, f)

        # Analytic solution of principal direction
        denom = np.sqrt(np.power(Gxy, 2) + np.power((Gxx - Gyy), 2)) + np.finfo(float).eps

        sin2theta = Gxy / denom  # Sine and cosine of doubled angles
        cos2theta = (Gxx - Gyy) / denom

        if orientsmoothsigma:
            sze = np.fix(6 * orientsmoothsigma)
            if np.remainder(sze, 2) == 0:
                sze = sze + 1
            gauss = cv2.getGaussianKernel(np.int(sze), orientsmoothsigma)
            f = gauss * gauss.T
            cos2theta = ndimage.convolve(cos2theta, f)  # Smoothed sine and cosine of
            sin2theta = ndimage.convolve(sin2theta, f)  # doubled angles

        orientim = np.pi / 2 + np.arctan2(sin2theta, cos2theta) / 2
        return orientim

    def ridge_freq(self,im, mask, orient, blksze, windsze, minWaveLength, maxWaveLength):
        rows, cols = im.shape
        freq = np.zeros((rows, cols))

        for r in range(0, rows - blksze, blksze):
            for c in range(0, cols - blksze, blksze):
                blkim = im[r:r + blksze][:, c:c + blksze]
                blkor = orient[r:r + blksze][:, c:c + blksze]

                freq[r:r + blksze][:, c:c + blksze] = self.frequest(blkim, blkor, windsze, minWaveLength, maxWaveLength)

        freq = freq * mask
        freq_1d = np.reshape(freq, (1, rows * cols))
        ind = np.where(freq_1d > 0)

        ind = np.array(ind)
        ind = ind[1, :]

        non_zero_elems_in_freq = freq_1d[0][ind]

        meanfreq = np.mean(non_zero_elems_in_freq)
        medianfreq = np.median(non_zero_elems_in_freq)  # does not work properly
        return freq, meanfreq

    def frequest(self,im, orientim, windsze, minWaveLength, maxWaveLength):
        rows, cols = np.shape(im)

        # Find mean orientation within the block. This is done by averaging the
        # sines and cosines of the doubled angles before reconstructing the
        # angle again.  This avoids wraparound problems at the origin.

        cosorient = np.mean(np.cos(2 * orientim))
        sinorient = np.mean(np.sin(2 * orientim))
        orient = atan2(sinorient, cosorient) / 2

        # Rotate the image block so that the ridges are vertical

        # ROT_mat = cv2.getRotationMatrix2D((cols/2,rows/2),orient/np.pi*180 + 90,1)
        # rotim = cv2.warpAffine(im,ROT_mat,(cols,rows))
        rotim = scipy.ndimage.rotate(im, orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3, mode='nearest')

        # Now crop the image so that the rotated image does not contain any
        # invalid regions.  This prevents the projection down the columns
        # from being mucked up.

        cropsze = int(np.fix(rows / np.sqrt(2)))
        offset = int(np.fix((rows - cropsze) / 2))
        rotim = rotim[offset:offset + cropsze][:, offset:offset + cropsze]

        # Sum down the columns to get a projection of the grey values down
        # the ridges.

        proj = np.sum(rotim, axis=0)
        dilation = scipy.ndimage.grey_dilation(proj, windsze, structure=np.ones(windsze))

        temp = np.abs(dilation - proj)

        peak_thresh = 2

        maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
        maxind = np.where(maxpts)

        rows_maxind, cols_maxind = np.shape(maxind)

        # Determine the spatial frequency of the ridges by divinding the
        # distance between the 1st and last peaks by the (No of peaks-1). If no
        # peaks are detected, or the wavelength is outside the allowed bounds,
        # the frequency image is set to 0

        if cols_maxind < 2:
            freqim = np.zeros(im.shape)
        else:
            NoOfPeaks = cols_maxind
            waveLength = (maxind[0][cols_maxind - 1] - maxind[0][0]) / (NoOfPeaks - 1)
            if waveLength >= minWaveLength and waveLength <= maxWaveLength:
                freqim = 1 / np.double(waveLength) * np.ones(im.shape)
            else:
                freqim = np.zeros(im.shape)

        return freqim

    def select(self):
        global img
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.tif;;*.png;;All Files(*)")
        img = cv2.imdecode(np.fromfile(imgName, dtype=np.uint8), cv2.IMREAD_COLOR)
        print(type(img))
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图

        # car_width = self.lb_car.height() / QtGui.QPixmap(imgName).height() * QtGui.QPixmap(imgName).width()
        jpg = QtGui.QPixmap(imgName).scaled(self.pic_origin.width(), self.pic_origin.height())
        self.pic_origin.setPixmap(jpg)



    def enhance(self):
        global img
        blksze = 16
        thresh = 0.1
        normim, mask = self.ridge_segment(img, blksze, thresh)  # normalise the image and find a ROI
        # imshow("norm", normim)

        gradientsigma = 1
        blocksigma = 7
        orientsmoothsigma = 7
        orientim = self.ridge_orient(normim, gradientsigma, blocksigma, orientsmoothsigma)  # find orientation of every pixel
        # imshow("orient", orientim)

        blksze = 38
        windsze = 5
        minWaveLength = 5
        maxWaveLength = 15
        freq, medfreq = self.ridge_freq(normim, mask, orientim, blksze, windsze, minWaveLength,
                                   maxWaveLength)  # find the overall frequency of ridges
        # imshow("freq", freq)

        freq = medfreq * mask
        kx = 0.65
        ky = 0.65
        newim = self.ridge_filter(normim, orientim, freq, kx, ky)  # create gabor filter and do the actual filtering
        # imshow("new",newim)

        img = 255 * (newim >= -3)


        print(1)

    def thinning(self):
        print(1)

    def feature(self):
        print(1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())