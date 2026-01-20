import cv2
import numpy as np

# Read the image using cv2.imread
x = cv2.imread('raw/PTM837_114_3073_10241.tif')

# Extract the first color channel (red channel)
y = x[:, :, 2]

# Save 'y' as a TIF file
cv2.imwrite('data/PTM837_114_3073_10241_single.tif', y)
