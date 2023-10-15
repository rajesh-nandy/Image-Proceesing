import rasterio as rst
from rasterio.plot import show

import cv2
import numpy as np
from GA import GAcluster
#image = cv2.imread("sir doc/images/barb.tif")

img = rst.open("image samples/L3/l3f45q1408mar16/L3-NF45Q14-109-057-08Mar16-BAND2.tif")


image = img.read()

Z = image.reshape((-1,1))
print(Z.shape)
Z = np.float32(Z)

ga = GAcluster(n_clusters= 5,n_particles=10, data=Z)
res = ga.start(iteration=25)
res2 = res.reshape((image.shape))
show(res2)
"""cv2.imshow('res2',res2)
cv2.waitKey(10000)
cv2.destroyAllWindows()

cv2.imwrite("result.jpeg", res2)
print(image.shape)"""
