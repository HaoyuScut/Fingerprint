import numpy as np
import scipy
import cv2
from scipy import ndimage, signal
from math import *

def image_enhance(img):

    blksze = 16
    thresh = 0.1
    normim, mask = ridge_segment(img, blksze, thresh)  # normalise the image and find a ROI
    # cv2.imshow("norm", normim)

    gradientsigma = 1
    blocksigma = 7
    orientsmoothsigma = 7
    orientim = ridge_orient(normim, gradientsigma, blocksigma, orientsmoothsigma)  # find orientation of every pixel
    # cv2.imshow("orient", orientim)

    # 绘制方向场图片
    img_d = np.copy(normim)
    ori = np.copy(orientim)
    img_draw = draw_orient(img_d, ori, 16)
    # cv2.imshow('img_draw',img_draw)

    blksze = 38
    windsze = 5
    minWaveLength = 5
    maxWaveLength = 15
    freq, meanfreq = ridge_freq(normim, mask, orientim, blksze, windsze, minWaveLength,
                               maxWaveLength)  # find the overall frequency of ridges
    #freq： 整体频率 meanfreq:平均频率
    # cv2.imshow("freq", freq)

    # 以平均值频率*ROI得到基于ROI的平均频率场
    freq = meanfreq * mask
    # cv2.imshow("ROI_freq", freq)
    kx = 0.65
    ky = 0.65
    newim = ridge_filter(normim, orientim, freq, kx, ky)  # 创建gabor过滤器并进行实际的过滤
    cv2.imshow("new",newim)
    # cv2.waitKey(0)
    # print(newim)
    print(newim.dtype)

    img = 255 * (newim >= -3)
    print(img.dtype)
    # print(type(img))
    # print(img)
    # cv2.imshow("new", img)
    return img

# Z-score标准化（0-1标准化）方法，通过原始数据均值和单位标准差进行归一化
# 经过处理的数据符合标准正态分布，即均值为0，标准差为1。
def normalise(img):
    normed = (img - np.mean(img)) / (np.std(img))
    return normed

def ridge_segment(im, blksze, thresh):  # img,16,0.1

    rows, cols = im.shape #获取图像的长和宽,row：行数，col：列数

    im = normalise(im)  # Z-score标准化
    # cv2.imshow("normalise",im)
    # print("img_shape: ", im.shape)

    #np.ceil 向上取整函数,将长和宽变为blksze的整数倍
    new_rows = np.int(blksze * np.ceil((np.float(rows)) / (np.float(blksze))))
    new_cols = np.int(blksze * np.ceil((np.float(cols)) / (np.float(blksze))))

    padded_img = np.zeros((new_rows, new_cols))
    # print("padded_img.shape: ",padded_img.shape)
    stddevim = np.zeros((new_rows, new_cols))

    #将im的参数传入到（304,304）的padded_img矩阵，多的地方补零
    #X[:,m:n] ，取二维数组的第m列到第n-1列所有行的数据,
    # 本文代码相当于做了两次切片，第一次选出padded_img的前rows行，第二次行不变，选出前cols列
    padded_img[0:rows][:, 0:cols] = im
    # print("padded_img: ", padded_img)

    #指纹图像分成16x16大小的块
    for i in range(0, new_rows, blksze):
        for j in range(0, new_cols, blksze):
            block = padded_img[i:i + blksze][:, j:j + blksze]

            #np.std()函数 被用来计算沿指定轴的标准差。 返回数组元素的标准差
            stddevim[i:i + blksze][:, j:j + blksze] = np.std(block) * np.ones(block.shape)

    stddevim = stddevim[0:rows][:, 0:cols]

    # 根据标准差和阈值的对比得出感兴趣区域
    mask = stddevim > thresh

    mean_val = np.mean(im[mask])

    std_val = np.std(im[mask])

    normim = (im - mean_val) / (std_val)
    # cv2.imshow("norm",normim)

    return (normim, mask)


def ridge_orient(im, gradientsigma, blocksigma, orientsmoothsigma): #img,1,7,7

    #  Calculate image gradients.
    # np.fix:四舍五入数组
    #一般高斯核尺寸通过计算得到：6*sigma+1 要保证尺寸的宽度和高度都为奇数
    sze = np.fix(6 * gradientsigma)
    # 返回两个数组arr1和arr2之间的除法元素余数
    if np.remainder(sze, 2) == 0:
        sze = sze + 1  # sze = 7.(浮点数)

    #高斯滤波器系数矩阵。getGaussianKernel() 函数计算并返回维度为 ksize×1 的高斯滤波器系数矩阵：
    #生成一维高斯核，（7，1）
    gauss = cv2.getGaussianKernel(np.int(sze), gradientsigma)
    #生成二维高斯核
    f = gauss * gauss.T

    fy, fx = np.gradient(f)  # Gradient of Gaussian

    # Gx = ndimage.convolve(np.double(im),fx);
    # Gy = ndimage.convolve(np.double(im),fy);


    #对图像的单通道进行卷积操作，计算图像的7*7 的Sobel算子模板
    Gx = signal.convolve2d(im, fx, mode='same')
    Gy = signal.convolve2d(im, fy, mode='same')

    # power(x, y) 函数，计算 x 的 y 次方。
    Gxx = np.power(Gx, 2)
    Gyy = np.power(Gy, 2)
    Gxy = Gx * Gy

    # Now smooth the covariance data to perform a weighted summation of the data.
    #现在平滑协方差数据，执行数据加权求和 高斯低通滤波器
    sze = np.fix(6 * blocksigma)

    gauss = cv2.getGaussianKernel(np.int(sze), blocksigma)
    f = gauss * gauss.T

    Gxx = ndimage.convolve(Gxx, f)
    Gyy = ndimage.convolve(Gyy, f)
    Gxy = 2 * ndimage.convolve(Gxy, f)

    # Analytic solution of principal direction，计算出近似梯度
    # 参考https://blog.csdn.net/great_yzl/article/details/119709699
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

# 方向场可视化
def draw_orient(img_d,ori,blksze):
    rows, cols = img_d.shape  # 获取图像的长和宽,row：行数，col：列数

    for r in range(0, rows - blksze, blksze):
        for c in range(0,cols - blksze, blksze):
            draw_img = img_d
            tmp = ori[r][c]
            cx = r + blksze/2
            cy = c + blksze/2

            x0 = int(cy - np.cos(tmp) * 5)
            y0 = int(cx - np.sin(tmp) * 5)

            x1 = int(cy + np.cos(tmp) * 5)
            y1 = int(cx + np.sin(tmp) * 5)

            point1 = (x0,y0)
            point2 = (x1,y1)
            point_color = (255, 255, 255)  # BGR
            thickness = 1
            lineType = 8
            cv2.line(draw_img, point1, point2, point_color, thickness, lineType)

    return draw_img



# 求频率场 输入： im,masl,orient,38,5,5,15
def ridge_freq(im, mask, orient, blksze, windsze, minWaveLength, maxWaveLength):
    rows, cols = im.shape
    freq = np.zeros((rows, cols))

    # 指纹图像分成16x16大小的块，得到每一块的像素值和方向场
    for r in range(0, rows - blksze, blksze):
        for c in range(0, cols - blksze, blksze):
            blkim = im[r:r + blksze][:, c:c + blksze]
            blkor = orient[r:r + blksze][:, c:c + blksze]
            # 求取每一小块的freq
            freq[r:r + blksze][:, c:c + blksze] = frequest(blkim, blkor, windsze, minWaveLength, maxWaveLength)

    # 根据ROI位置得到对应ROI的freq (300,300)
    freq = freq * mask
    # print("freq_shape:",freq.shape)
    #转为一维数组 (1,90000)
    freq_1d = np.reshape(freq, (1, rows * cols))
    # print("freq_1d_shape:", freq_1d.shape)
    # np.where(freq_1d > 0) ：返回freq_1d中大于0元素的索引 得到（2，31768）的数组，第一行全为0
    ind = np.where(freq_1d > 0)

    ind = np.array(ind)
    # print("ind_shape:", ind.shape)

    # 只留第二行的内容，即freq_1d中大于0元素的索引
    ind = ind[1, :]

    # 得到freq_1d中大于0元素的集合（31768,）
    non_zero_elems_in_freq = freq_1d[0][ind]

    # 得到freq_1d中大于0元素的均值
    meanfreq = np.mean(non_zero_elems_in_freq)
    medianfreq = np.median(non_zero_elems_in_freq)  # does not work properly
    return freq, meanfreq


def frequest(im, orientim, windsze, minWaveLength, maxWaveLength):
    rows, cols = np.shape(im)

    # 找到块内的平均方向。这是通过对两倍角的正弦和余弦求平均值来完成的，然后再重新构造这个角。这避免了在原点处出现问题。

    cosorient = np.mean(np.cos(2 * orientim))
    sinorient = np.mean(np.sin(2 * orientim))
    orient = atan2(sinorient, cosorient) / 2

    # 旋转图像块，使脊线垂直

    # ROT_mat = cv2.getRotationMatrix2D((cols/2,rows/2),orient/np.pi*180 + 90,1)
    # rotim = cv2.warpAffine(im,ROT_mat,(cols,rows))
    rotim = scipy.ndimage.rotate(im, orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3, mode='nearest')

    # 现在对图像进行裁剪，使旋转后的图像不包含任何无效区域。这样可以防止柱下的投影被弄脏。

    cropsze = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - cropsze) / 2))
    rotim = rotim[offset:offset + cropsze][:, offset:offset + cropsze]

    # 对列求和，得到灰度值沿脊的投影。

    proj = np.sum(rotim, axis=0)
    dilation = scipy.ndimage.grey_dilation(proj, windsze, structure=np.ones(windsze))

    temp = np.abs(dilation - proj)

    peak_thresh = 2

    maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
    maxind = np.where(maxpts)

    rows_maxind, cols_maxind = np.shape(maxind)

    # 通过将第1个峰和最后一个峰之间的距离除以(峰数-1)来确定脊的空间频率。
    # 如果没有峰值检测到，或波长超出允许的范围，频率图像设置为0

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


def ridge_filter(im, orient, freq, kx, ky):
    angleInc = 3
    im = np.double(im)
    rows, cols = im.shape
    newim = np.zeros((rows, cols))

    freq_1d = np.reshape(freq, (1, rows * cols))
    ind = np.where(freq_1d > 0)

    ind = np.array(ind)
    ind = ind[1, :]

    # 将频率数组四舍五入到0.01，以减少我们必须处理的不同频率的数量。（同频率场操作）

    non_zero_elems_in_freq = freq_1d[0][ind]
    # 四舍五入到小数点后两位
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100

    #对于一维数组或者列表，unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表
    unfreq = np.unique(non_zero_elems_in_freq)

    # 生成对应于这些不同的频率和方向在'angleInc'增量的滤波器
    #sigmax sigmay分别是沿着x和y轴的高斯包络常量。
    sigmax = 1 / unfreq[0] * kx
    sigmay = 1 / unfreq[0] * ky

    sze = np.int(np.round(3 * np.max([sigmax, sigmay]))) #sze：13
    # numpy.meshgrid()——生成网格点坐标矩阵,即横坐标矩阵X中的每个元素，与纵坐标矩阵Y中对应位置元素
    # 生成两个（27,27）的x、y矩阵
    x, y = np.meshgrid(np.linspace(-sze, sze, (2 * sze + 1)), np.linspace(-sze, sze, (2 * sze + 1)))

    reffilter = np.exp(-(((np.power(x, 2)) / (sigmax * sigmax) + (np.power(y, 2)) / (sigmay * sigmay)))) * np.cos(
        2 * np.pi * unfreq[0] * x)  # this is the original gabor filter

    filt_rows, filt_cols = reffilter.shape

    angleRange = np.int(180 / angleInc)

    gabor_filter = np.array(np.zeros((angleRange, filt_rows, filt_cols)))

    for o in range(0, angleRange):
        # Generate rotated versions of the filter.  Note orientation
        # image provides orientation *along* the ridges, hence +90
        # degrees, and imrotate requires angles +ve anticlockwise, hence
        # the minus sign.

        rot_filt = scipy.ndimage.rotate(reffilter, -(o * angleInc + 90), reshape=False)
        gabor_filter[o] = rot_filt

    # Find indices of matrix points greater than maxsze from the image
    # boundary

    maxsze = int(sze)

    temp = freq > 0
    validr, validc = np.where(temp)

    temp1 = validr > maxsze
    temp2 = validr < rows - maxsze
    temp3 = validc > maxsze
    temp4 = validc < cols - maxsze

    final_temp = temp1 & temp2 & temp3 & temp4

    finalind = np.where(final_temp)

    # Convert orientation matrix values from radians to an index value
    # that corresponds to round(degrees/angleInc)

    maxorientindex = np.round(180 / angleInc)
    orientindex = np.round(orient / np.pi * 180 / angleInc)

    # do the filtering

    for i in range(0, rows):
        for j in range(0, cols):
            if orientindex[i][j] < 1:
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if orientindex[i][j] > maxorientindex:
                orientindex[i][j] = orientindex[i][j] - maxorientindex
    finalind_rows, finalind_cols = np.shape(finalind)
    sze = int(sze)
    for k in range(0, finalind_cols):
        r = validr[finalind[0][k]]
        c = validc[finalind[0][k]]

        img_block = im[r - sze:r + sze + 1][:, c - sze:c + sze + 1]

        newim[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r][c]) - 1])

    return newim



