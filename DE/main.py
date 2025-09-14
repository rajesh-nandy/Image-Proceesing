import rasterio as rst
from rasterio.plot import show
import cv2
import numpy as np
import cupy as cp
from DE import DEcluster
#image = cv2.imread("image samples/images/lena1.tif")

img = rst.open("image samples/L3/l3f45q1408mar16/L3-NF45Q14-109-057-08Mar16-BAND2.tif")
image = img.read()
#image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)  
Z = image.reshape((-1,1))
print(Z.shape)
Z = np.float32(Z)



def cp_warmup():
  a = cp.ones((1000,1000))
  b = cp.ones((1000,1000))
  c = cp.dot(a,b)
cp_warmup()

de = None
best_val = cp.inf
best_res_old = None
progress_old = None
chk_old = []
for _ in range(1):
  de = DEcluster(n_clusters= 5, n_particles=20, data=Z)
  res, gb_val = de.start(iteration=1000)
  print(gb_val)
  chk_old.append(gb_val)
  if gb_val < best_val:
    best_val = gb_val
    best_res_old = res.copy()
    progress_old = de.progress

res2 = res.reshape((image.shape))
show(res2)
"""cv2.imshow('res2',res2)
cv2.waitKey(10000)
cv2.destroyAllWindows()

cv2.imwrite("result.jpeg", res2)
print(image.shape)"""
