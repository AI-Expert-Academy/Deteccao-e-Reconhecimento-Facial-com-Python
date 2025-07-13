import cv2
import face_recognition
import imutils
import numpy as np

foto_eu = cv2.imread('eu.jpg')
foto_eu_nparray = np.array(foto_eu)
print(foto_eu_nparray.shape)
foto_eu_nparray = imutils.resize(foto_eu_nparray, width=300)
print(foto_eu_nparray.shape)
cv2.imwrite('eu_test_resize.jpg', foto_eu_nparray)