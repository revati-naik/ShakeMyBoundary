import cv2
import numpy as np 

import matplotlib.pyplot as plt 
from scipy.signal import convolve2d
from sklearn.cluster import KMeans



def color(src):
    img = cv2.imread(src)
    img_flat = img.reshape(img.shape[0]*img.shape[1],3)
    km = KMeans(n_clusters=16).fit_predict(img_flat)
    km1 = km.reshape(img.shape[0]*img.shape[1],1)
    t3 = np.zeros((img.shape[0],img.shape[1]), dtype='int')
    print(t3.shape)
    m = 0
    n = 0
    for i in range(0,km1.shape[0]):
        if (n==(img.shape[1])):
            m+=1
            n=0
        t3[m,n] = km1[i] 
        n+=1
       
    return t3