import cv2
import numpy as np
import matplotlib.pyplot as plt


def gabor(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    xmax = 8
    ymax = 8
    xmin = -xmax
    ymin = -ymax
    y_ = np.arange(ymin, ymax + 1)
    x_ = np.arange(xmin, xmax + 1)
    (x, y) = np.meshgrid(x_, y_)

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


def gaborFilter():
    gab = np.zeros([17,17,24])
    scales = np.sqrt(2)*np.array([1.5, 2, 3])
    o = 8
    count = 0
    for s in scales:
        angle = np.pi/2
        a = np.pi/o
        for i in range(0,o):
            gab[:,:,count] = gabor(s,(angle+(i*a)),(4),0, 1)
            count+=1
    return gab

def testMain():
    fig2 = plt.figure(figsize=(8,3))
    fig2.tight_layout()
    g = gaborFilter()
    for i in range(0,24):
        sub2 = fig2.add_subplot(3,8,i+1)
        sub2.imshow(g[:,:,i], cmap="gray")
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    testMain()