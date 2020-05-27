import sys
import cv2
import math
import numpy as np 
import matplotlib.pyplot as plt 

def DoG(s, o):
	# sobelx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)
	# sobely = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)

	# new_img_x = cv2.filter2D(image, -1, sobel_filter)
	# new_img_y = cv2.filter2D(image, -1, np.flip(sobel_filter.T, axis=0))


	# plt.subplot(221),plt.imshow(sobelx,cmap = 'gray')
	# plt.title('sobelx'), plt.xticks([]), plt.yticks([])

	# plt.subplot(222),plt.imshow(sobely,cmap = 'gray')
	# plt.title('sobelx'), plt.xticks([]), plt.yticks([])


	# plt.subplot(223),plt.imshow(new_img_x,cmap = 'gray')
	# plt.title('New_Img_x'), plt.xticks([]), plt.yticks([])


	# plt.subplot(224),plt.imshow(new_img_y,cmap = 'gray')
	# plt.title('New_Img_Y'), plt.xticks([]), plt.yticks([])

	# plt.show()

	
	
	gauss = np.zeros([11,11,32])
	# print(gauss.shape)

	theta = 2*np.pi / (o-1)
	count = 0

	for i in range(1, s+1):
		angle = 0
		for j in range(1, o+1):
			gaussian_matrix = gaussianMatrix(m=5, n=5, sigma_x=(0.5+0.5*i)*np.sqrt(2), sigma_y=(0.5+0.5*i)*np.sqrt(2))
			# print(gaussian_matrix.shape)
			sobel_filter = np.array([[-1, 0,  1],
							[-2, 0, 2], 
							[-1, 0, 1]])
	

			Gx = convolveOnImage(img=gaussian_matrix, kernel=sobel_filter)
			Gy = convolveOnImage(img=gaussian_matrix, kernel=np.flip(sobel_filter.T, axis=0))

			DoG = Gx*math.cos(angle) + Gy*math.sin(angle)
			gauss[:,:,count] = DoG

			count += 1
			angle += theta

	return gauss


def sobelFilter():
	sobel_filter = np.array([[-1, 0,  1],
							[-2, 0, 2], 
							[-1, 0, 1]])
	
	return sobel_filter

def convolve(img, x, y, kernel):
	sum = 0
	for i in range(0, kernel.shape[0]):
		for j in range(0, kernel.shape[1]):
			sum += img[x+i-1, y+j-1]*kernel[i,j]
	return sum

def convolveOnImage(img, kernel):
	# print("img.shape", img.shape)
	img_new = np.zeros(img.shape)

	for x in range(0, img.shape[0]):
		for y in range(0, img.shape[1]):
			#avoid padding in the img
			if (x!=0 and y!=0 and x!=img.shape[0]-1 and y!=img.shape[1]-1):
				img_new[x,y] = convolve(img, x, y, kernel)
	return img_new

def gaussianMatrix(m, n, sigma_x, sigma_y):
	x,y = np.ogrid[-m:m+1,-n:n+1]
	return ((1/np.sqrt(2*np.pi*sigma_x*sigma_y))*(np.exp((-x*x/(2*sigma_x*sigma_x))-(y*y/(sigma_y*sigma_y*2)))))
	
def testMain():
	# read image 
	img = cv2.imread('../BSDS500/Images/1.jpg')
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# cv2.imshow("img", img)
	# cv2.waitKey(0)
	# s = input("Enter scale factor: ") 
	# o = input("no. of orientations: ")
	# sys.exit(0)
	# d =DoG(2, 16)
	# # print('d= ',d)
	# fig = plt.figure(figsize=(16,2))
	# fig.tight_layout()
	# for i in range(0,32):
	# 	sub = fig.add_subplot(2,16,i+1)
	# 	sub.imshow(d[:,:,i], cmap = "gray", interpolation="nearest")
	# 	plt.axis('off')
	# plt.show()
	# plt.savefig('DoG_5_5.png')

if __name__ == '__main__':
	# # for i in range(1,3):
	# 	d = gaussianMatrix((6),(6),(0.5+0.5*i)*np.sqrt(2),(0.5+0.5*i)*np.sqrt(2))	
	# # 	print(d)
	# # 	print("===============")
	testMain()