import numpy as np

class ImageFrame(object):

	def __init__(self, image, sil, RT, P, K, im_path, sil_path):
		self.image = image
		self.silhouette = sil
		self.RT = RT
		self.projection = P
		self.K = K
		self.image_path = im_path
		self.sil_path = sil_path

	def rot(self):
		if self.RT is not None:
			return self.RT[0:3,0:3]


	def translation(self):
		if self.RT is not None:
			return self.RT[0:3,3]


	def get_camera_direction(self):
		x = np.array([self.image.shape[1] / 2,
		     self.image.shape[0] / 2,
		     1]);
		X = np.linalg.solve(self.K, x)
		X = self.rot().transpose().dot(X)
		return X / np.linalg.norm(X)
