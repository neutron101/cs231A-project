import numpy as np

class AttributeGenerator:

	def __init__(self, RTs, Ks, Ps):
		self._RTs = RTs
		self._Ks = Ks
		self._Ps = Ps

	def generate(self):
		self._update_intrinsics()
		self._update_translation_and_rotation()
		self._updateProjection()


	def use_default_translation_and_rotation(self):
		self.rotation_translation = self._RTs

	def _updateProjection(self):
		self.projection = []

		for i in range(len(self.rotation_translation)):
			kr = self.intrinsics[i].dot(self.rotation_translation[i][0:3,0:3].transpose())
			kt = np.reshape(self.intrinsics[i].dot(-1*self.rotation_translation[i][0:3,0:3].transpose().dot(self.rotation_translation[i][0:3,3])), [3,1])

			new_projection = np.hstack((kr, kt))
			self.projection.append(new_projection)
			

	def _update_translation_and_rotation(self):

		self.rotation_translation = []

		base_rot = self._RTs[0][0:3,0:3] 
		base_trans = self._RTs[0][0:3,3]

		for i in range(0, len(self._RTs)):
			current = np.linalg.inv(base_rot).dot(self._RTs[i][0:3,0:3])
			t = np.linalg.inv(current).dot(self._RTs[i][0:3,3] - base_trans)

			rt = np.hstack((current, np.reshape(t, [3,1])))
			
			self.rotation_translation.append(rt)

	def _update_intrinsics(self):
		self.intrinsics = self._Ks




		