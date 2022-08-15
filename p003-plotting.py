from matplotlib import pyplot as plt

x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

import numpy as np
a = np.array(x)
b = np.array(y)
plt.plot(a, b)

import cv2
gray_img = cv2.imread('images/sandstone.tif', 0)

fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)
ax1.imshow(gray_img, cmap="gray")

ax2 = fig.add_subplot(1,2,2)
ax2.hist(gray_img.flat, bins=100, range=(0,255))
