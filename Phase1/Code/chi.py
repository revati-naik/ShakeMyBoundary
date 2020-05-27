import cv2
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import convolve2d
from sklearn.cluster import KMeans



def chi(img,k, hd1, hd2, hd3):
    
    chi_sum = np.zeros([img.shape[0],img.shape[1]])
    g_i = np.zeros([img.shape[0],img.shape[1]])
    h_i = np.zeros([img.shape[0],img.shape[1]]) 
    
    for i in range(0,4):
        print(i)
        for index in range(0,k):
            temp = np.zeros([img.shape[0],img.shape[1]])
            for m in range(0,img.shape[0]):
                for n in range(0,img.shape[1]):
                    if (img[m][n] == index):
                        temp[m][n] = 1
                    else:
                        temp[m][n] = 0
            g_i = convolve2d(temp,hd1[:,:,(2*i)],mode="same")
            h_i = convolve2d(temp,hd1[:,:,((2*i) +1)],mode="same")
            a = (g_i-h_i)*(g_i-h_i)/(g_i+h_i)
            where = np.isnan(a)
            a[where]=0
            chi_sum = chi_sum + a
    for i in range(0,4):
        print(i)
        for index in range(0,k):
            temp = np.zeros([img.shape[0],img.shape[1]])
            for m in range(0,img.shape[0]):
                for n in range(0,img.shape[1]):
                    if (img[m][n] == index):
                        temp[m][n] = 1
                    else:
                        temp[m][n] = 0
            g_i = convolve2d(temp,hd2[:,:,(2*i)],mode="same")
            h_i = convolve2d(temp,hd2[:,:,((2*i) +1)],mode="same")
            a = (g_i-h_i)*(g_i-h_i)/(g_i+h_i)
            where = np.isnan(a)
            a[where]=0
            chi_sum = chi_sum + a
    for i in range(0,4):
        print(i)
        for index in range(0,k):
            temp = np.zeros([img.shape[0],img.shape[1]])
            for m in range(0,img.shape[0]):
                for n in range(0,img.shape[1]):
                    if (img[m][n] == index):
                        temp[m][n] = 1
                    else:
                        temp[m][n] = 0
            g_i = convolve2d(temp,hd3[:,:,(2*i)],mode="same")
            h_i = convolve2d(temp,hd3[:,:,((2*i) +1)],mode="same")
            a = (g_i-h_i)*(g_i-h_i)/(g_i+h_i)
            where = np.isnan(a)
            a[where]=0
            chi_sum = chi_sum + a

    return chi_sum/2
