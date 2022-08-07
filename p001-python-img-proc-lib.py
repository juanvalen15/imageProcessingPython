from skimage import io
img1 = io.imread("images/Osteosarcoma_01.tif")

import cv2
img2 = cv2.imread("images/Osteosarcoma_01.tif")

import numpy as np
a = np.ones((5,5))

from matplotlib import pyplot as plt 
plt.imshow(img1)