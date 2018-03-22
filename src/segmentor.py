import utils
from inference_mine import infer

class Segmentor:

	def __init__(self):
		pass

	def segment_images_at_path(self, read_image_path, sil_save_path):
		print "img path", read_image_path, "save path", sil_save_path
		infer(read_image_path, sil_save_path)

