from sklearn import svm
import pandas as pd
import numpy as nm
import cv2 as cv
from sklearn import decomposition as dec

import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fields = nm.genfromtxt('data/fer2013.csv',delimiter=',',dtype=None)

images=[]

for i in range(1,len(fields)):
    images.append(nm.array(fields[i][1].split(" ")).astype(nm.float))

images = nm.array(images)

start_time = time.time()

kpca = dec.KernelPCA(kernel="rbf", gamma=10)
images = kpca.fit_transform(images)

print('kpca - ', time.time() - start_time)


start_time = time.time()
clf = svm.SVC(kernel='rbf', gamma=2)
clf.fit(images[:2000], fields[1:2001,0].astype(nm.float))
print('SVC - ', time.time() - start_time)


print clf.predict(images[2000:2200]) - fields[2001:2201,0].astype(nm.float)

