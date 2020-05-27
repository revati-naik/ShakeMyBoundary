import cv2
import numpy as np 

import matplotlib.pyplot as plt 
from scipy.signal import convolve2d
from sklearn.cluster import KMeans


def brightness(src):
    img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    img_flat = img.reshape(img.shape[0]*img.shape[1],1)
    km = KMeans(n_clusters=16).fit_predict(img_flat)
    t3 = np.zeros((img.shape[0],img.shape[1]), dtype='int')
    print(t3.shape)
    m = 0
    n = 0
    for i in range(0,km.shape[0]):
        if (n==(img.shape[1])):
            m+=1
            n=0
        t3[m,n] = km[i] 
        n+=1
       
    return t3

def testMain():
	pass


if __name__ == '__main__':
	testMain()