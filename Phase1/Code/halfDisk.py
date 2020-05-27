import cv2
import numpy as np 

def halfDisk(s):
    half = np.zeros([2*(3+(2*s)) +1, 2*(3+(2*s)) + 1 ,16])
    x,y = np.ogrid[-(3+(2*s)):((3+(2*s))+1),-(3+(2*s)):((3+(2*s))+1)]
    mask = (x*x + y*y <=((3+(2*s))+1)*((3+(2*s))+1))
    box_l = np.zeros(((2*(3+(2*s))+1),(2*(3+(2*s))+1)))
    box_r = np.zeros(((2*(3+(2*s))+1),(2*(3+(2*s))+1)))
    for i in range(0,(2*(3+(2*s))+1)):
        for j in range(0,(2*(3+(2*s))+1)):
            if j<(3+(2*s)):
                box_l[i][j] = True
            else:
                box_l[i][j] = False
    for i in range(0,(2*(3+(2*s))+1)):
        for j in range(0,(2*(3+(2*s))+1)):
            if j>(3+(2*s)):
                box_r[i][j] = True
            else:
                box_r[i][j] = False
    for i in range(0,8):
        rows,cols = box_l.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2),22.5*i,1)
        dst = cv2.warpAffine(box_l,M,(cols,rows))
        dst_l = np.logical_and(mask,dst)
        half[:,:,2*i] = dst_l
        dst = cv2.warpAffine(box_r,M,(cols,rows))
        dst_r = np.logical_and(mask,dst)
        half[:,:,(2*i+1)] = dst_r
    return half