from PS import PSOClustering
import cv2
import numpy as np

import rasterio as rst
from rasterio.plot import show

img = rst.open("/tst/Programs/sir doc/L3/l3f45q1408mar16/FAA_UTM18N_NAD83.tif")
#image = cv2.imread("/tst/Programs/sir doc/images/barb.tif")
image = img.read()
#image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)

Z = image.reshape((-1,1))
print(Z.shape)
# convert to np.float32
list_output = []
# convert to float32
Z = cp.float32(Z)
Z = cp.asarray(Z)

def cp_warmup():
  a = cp.ones((1000,1000))
  b = cp.ones((1000,1000))
  c = cp.dot(a,b)



pso = None
best_val = cp.inf
best_res = None
progress_old = None
chk_old = []
mean_val_old = 0
for i in range(50):
  pso = PSOClustering(n_clusters= 5, n_particles=20, data=Z)

  cp_warmup()

  res, gb_val = pso.start(iteration=1000)
  print(gb_val)
  chk_old.append(gb_val)
  mean_val_old = mean_val_old + gb_val
  if gb_val < best_val:
    best_val = gb_val
    best_res = res.copy()
    progress_old = pso.progress



res2 = res.reshape((image.shape))
show(res2)
"""cv2.imshow('res2',res2)
cv2.waitKey(10000)
cv2.destroyAllWindows()

cv2.imwrite("result.jpeg", res2)
print(image.shape)"""





