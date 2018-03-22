import utils
import os

class ImageAttributeProcessor:

	def __init__(self, base_location='../data'):
		self.base_location = base_location

	def process_all(self, target_name):
		utils.load_data(os.path.join(self.base_location, target_name))

	def process_single_target(self, target_name):
		utils.load_data(self.base_location, [target_name])