import cv2, numpy as np
img = cv2.imread('mouth.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
np.place(img, img==np.array([7, 12, 22], dtype=np.uint8), np.array([255,255,255], dtype=np.uint8))
img[:, :, img==np.array([7, 12, 22])] = np.array([255,255,255])
print(img)