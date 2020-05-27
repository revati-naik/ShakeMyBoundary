import cv2
import numpy as np 

import matplotlib.pyplot as plt 
from scipy.signal import convolve2d
from sklearn.cluster import KMeans


def texton(src, dog, lm, gabor):
	img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
	img_1 = np.zeros([img.shape[0], img.shape[1], 104]) 
	count = 0

	for i in range(0, 32):
		img_1[:,:,count] = convolve2d(img, dog[:,:,i], mode="same")
		count += 1

	for j in range(0, 48):
		img_1[:,:,count] = convolve2d(img, lm[:,:,j], mode="same")
		count += 1
	
	for k in range(0, 24):
		img_1[:,:,count] = convolve2d(img, gabor[:,:,k], mode="same")
		count += 1
	
	img_flat = img_1.reshape(img.shape[0]*img.shape[1], 104)
	km = KMeans(n_clusters=64).fit_predict(img_flat)
	km1 = km.reshape(154401, 1)
	# print(type(km1))
	t3 = np.zeros((img.shape[0],img.shape[1]), dtype='int')
	print(t3.shape)
	m = 0
	n = 0
	for i in range(0,km1.shape[0]):
		if (n==(img.shape[1])):
			#change rows and cols to form t3 like an image matrix, with shape mxn with every pixel as the value from kmeans (1D)
			m+=1
			n=0
		t3[m,n] = km1[i] 
		n+=1
	   
	return t3

def testMain():
	pass




if __name__ == '__main__':
	testMain()