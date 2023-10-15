import rasterio as rst
from rasterio.plot import show

import cv2
import numpy as np
from DE import DEcluster
#image = cv2.imread("image samples/images/lena1.tif")

img = rst.open("image samples/L3/l3f45q1408mar16/L3-NF45Q14-109-057-08Mar16-BAND2.tif")
image = img.read()

Z = image.reshape((-1,1))
print(Z.shape)
Z = np.float32(Z)

de = DEcluster(n_clusters= 8,n_particles=50, data=Z)
res = de.start(iteration=500)
res2 = res.reshape((image.shape))
show(res2)
"""cv2.imshow('res2',res2)
cv2.waitKey(10000)
cv2.destroyAllWindows()

cv2.imwrite("result.jpeg", res2)
print(image.shape)"""
