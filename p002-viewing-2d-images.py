from skimage import io
import matplotlib.pyplot as plt

img = io.imread("images/Osteosarcoma_01.tif")


#io.imshow(img)

#plt.imshow(img)


img_gray = io.imread("images/Osteosarcoma_01.tif", as_gray=True)
plt.imshow(img_gray)
plt.imshow(img_gray, cmap="hot")
plt.imshow(img_gray, cmap="jet")