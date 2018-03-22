import numpy as np
import cv2
import matplotlib.pyplot as plt

class CoinDetector:

	def __init__(self):
		pass

	def detect(self, roi):

		print roi.shape

		roi = roi[900:990, 2300:2400,  :]
		hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
		print roi
		plt.imshow(roi)
		plt.show()

		gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		# plt.imshow(gray,  cmap='gray')
		# plt.show()

		gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
		# plt.imshow(gray_blur,  cmap='gray')
		# plt.show()

		thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
		cv2.THRESH_BINARY_INV, 11, 1)

		plt.imshow(thresh,  cmap='gray')
		plt.show()

		kernel = np.ones((3, 3), np.uint8)
		closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
		kernel, iterations=4)

		cont_img = closing.copy()
		print type(cont_img), cont_img.shape

		plt.imshow(cont_img,  cmap='gray')
		plt.show()

		image, contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_NONE)

		print "coutours", len(contours)

		for cnt in contours:	
			area = cv2.contourArea(cnt)
			print 'area', area, len(cnt)
			# if area < 2000 or area > 4000:
			# 	continue

			if len(cnt) < 5:
			 	continue

			ellipse = cv2.fitEllipse(cnt) 
			plt.subplot(131)
			cv2.ellipse(roi, ellipse, (0,255,0), 2) 
			plt.imshow(closing) 
			plt.subplot(132)
			plt.imshow(thresh) 
			plt.subplot(133)
			plt.imshow(roi)
			
			plt.show()

		return None

