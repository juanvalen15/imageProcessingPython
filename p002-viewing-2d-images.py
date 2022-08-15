from skimage import io
import matplotlib.pyplot as plt
import cv2

img = io.imread("images/Osteosarcoma_01.tif")
img_gray = io.imread("images/Osteosarcoma_01.tif", as_gray=True)

gray_img = cv2.imread("images/Osteosarcoma_01.tif", 0)
color_img = cv2.imread("images/Osteosarcoma_01.tif", 1)

#io.imshow(img)
#plt.imshow(img)


'''
plt.imshow(img_gray)
plt.imshow(img_gray, cmap="hot")
plt.imshow(img_gray, cmap="jet")
plt.imshow(img_gray, cmap="Blues")
'''

fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img_gray, cmap='hot')
ax1.title.set_text('1st')

ax1 = fig.add_subplot(2,2,2)
ax1.imshow(img_gray, cmap='jet')
ax1.title.set_text('2nd')

ax1 = fig.add_subplot(2,2,3)
ax1.imshow(img_gray, cmap='gray')
ax1.title.set_text('3rd')

ax1 = fig.add_subplot(2,2,4)
ax1.imshow(img_gray, cmap='nipy_spectral')
ax1.title.set_text('4th')

img_RGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imshow("pic from skimage import", img_RGB)
cv2.imshow("color pic from opencv", color_img)
cv2.imshow("gray pic from opencv", gray_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


