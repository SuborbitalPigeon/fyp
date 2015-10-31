import cv2
import numpy as np

def file_to_matrix(filename):
    with open(filename) as f:
        data = f.readlines()

    data = [line.split() for line in data]
    return np.array(data[:3], dtype=float) # strip the last newline

if __name__ == '__main__':
    cv2.namedWindow("img")
    cv2.namedWindow("img2")
    cv2.namedWindow("img2h")

    img = cv2.imread('bark/img1.ppm')
    cv2.imshow("img", img)

    img2 = cv2.imread('bark/img2.ppm')
    cv2.imshow("img2", img2)

    mat = file_to_matrix('bark/H1to2p')
    img2h = cv2.warpPerspective(img, mat, (800, 600))
    cv2.imshow("img2h", img2h)
    
    cv2.waitKey(0)
