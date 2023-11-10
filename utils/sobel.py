"""
@file sobel_demo.py
@brief Sample code using Sobel and/or Scharr OpenCV functions to make a simple Edge Detector
"""
import sys
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
def main(src):
    # window_name = ('Sobel Demo - Simple Edge Detector')
    scale = 1
    delta = 0
    ddepth = cv.CV_16S
    #
    # if len(argv) < 1:
    #     print('Not enough parameters')
    #     print('Usage:\nmorph_lines_detection.py < path_to_image >')
    #     return -1

    # Load the image
    # src = cv.imread(argv, cv.IMREAD_COLOR)
    # Check if image is loaded fine
    # if src is None:
    #     print('Error opening image: ' + argv[0])
    #     return -1

    src = cv.GaussianBlur(src, (3, 3), 0)
    # label=cv.imread('C:\\Users\\auy200\\Desktop\\active_learning\\codes\\pytorch-segmentation\\zurich_000008_000019_gtFine_color.png')
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # Gradient-Y
    # grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # cv.imshow(window_name, grad)
    # cv.waitKey(0)
    grad[grad<20]=0
    grad[grad>200]=0
    grad[grad!=0]=1
    return grad
    # cv.imshow(window_name, grad*255)
    # cv.waitKey(0)
def create_labels(grad,label):
    edges=label.copy()
    edges[grad==0]=255
    centers=label.copy()
    centers[grad==1]=255
    # plt.subplot(2,1,1)
    # plt.imshow(edges)
    # plt.subplot(2,1,2)
    # plt.imshow(centers)
    # plt.show()
    # cv.imwrite('C:\\Users\\auy200\\Desktop\\active_learning\\codes\\pytorch-segmentation\\center.png', centers)
    # cv.imwrite('C:\\Users\\auy200\\Desktop\\active_learning\\codes\\pytorch-segmentation\\boundaries.png', edges)
    return edges,centers

