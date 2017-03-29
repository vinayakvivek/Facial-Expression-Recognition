from sklearn import svm
import pandas as pd
import numpy as nm
import cv2 as cv
from sklearn import decomposition as dec

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fields = nm.genfromtxt('data/fer2013.csv',delimiter=',',dtype=None)

images=[]

for i in range(1,len(fields)):
    images.append(nm.array(fields[i][1].split(" ")).astype(nm.float))

images=nm.array(images)
pca=dec.PCA(n_components=3)
pca.fit(images)

images=pca.transform(images)


fig=plt.figure()

ax = fig.add_subplot(111, projection='3d')


clf = svm.SVC(kernel='rbf', gamma=2)
clf.fit(images[:2000], fields[1:2001,0].astype(nm.float))

print clf.predict(images[2000:2200])- fields[2001:2201,0].astype(nm.float)

