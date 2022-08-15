from matplotlib import pyplot as plt

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

import numpy as np
a = np.array(x)
b = np.array(y)
plt.plot(a, b)

import cv2
gray_img = cv2.imread('images/sandstone.tif', 0)

plt.imshow(gray_img, cmap="gray")
plt.show()

plt.hist(gray_img.flat, bins=100, range=(0,255))
plt.show()