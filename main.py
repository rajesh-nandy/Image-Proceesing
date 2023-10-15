from PS import PSOClustering
import cv2
import numpy as np

import rasterio as rst
from rasterio.plot import show

img = rst.open("/tst/Programs/sir doc/L3/l3f45q1408mar16/FAA_UTM18N_NAD83.tif")
#image = cv2.imread("/tst/Programs/sir doc/images/barb.tif")
image = img.read()

Z = image.reshape((-1,1))
print(Z.shape)
# convert to np.float32
Z = np.float32(Z)

pso = PSOClustering(n_clusters= 5, n_particles=10, data=Z)
res = pso.start(iteration=5)
res2 = res.reshape((image.shape))
show(res2)
"""cv2.imshow('res2',res2)
cv2.waitKey(10000)
cv2.destroyAllWindows()

cv2.imwrite("result.jpeg", res2)
print(image.shape)"""





