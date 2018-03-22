
import pickle as pk
import os
import numpy as np
import utils
from imageframe import ImageFrame

class ImageData:

	def __init__(self, frames, volume):
		self.frames = frames
		self.volume = volume

		self.pixel_width = None
		self.scale = None

	def __init__(self, image_data_path):
		self.volume = {}

		images, im_paths = utils.load_images_of_type(image_data_path)
		silhouettes, sil_paths = utils.load_np_data(image_data_path, 'silhouettes')

		assert np.array_equal(len(images), len(silhouettes)), "Size and number of images must match the silhouettes" 

		volume, scale, pixel_width = FramePropertyWriter.read(image_data_path)

		index = 1
		for i in range(len(images)):
			self.addFrame(images[i], silhouettes[i], utils.strip_file_name(im_paths[i]), utils.strip_file_name(sil_paths[i]))
			index+=1

		self.volume = volume
		self.pixel_width = float(pixel_width) if pixel_width is not None else pixel_width
		self.scale = float(scale) if scale is not None else scale

	def addFrame(self, image, silhouette, image_path, sil_path):
		if not hasattr(self, 'frames'):
			self.frames = []
		self.frames.append(ImageFrame(image, silhouette, None, None, None, image_path, sil_path))


class FramePropertyWriter:

	@classmethod
	def save(self, filepath, volume, scale=None, pixel_width=None):

		data = {
			'volume' : volume,
			'scale' : scale,
			'pixel_width' : pixel_width
		}
		with open(os.path.join(filepath, 'data'), 'wb') as f:
			pk.dump(data, f, pk.HIGHEST_PROTOCOL)

	@classmethod
	def read(self, filepath):

		volume = None
		pixel_width = None
		scale = None

		with open(os.path.join(filepath, 'data'), 'rb') as f:
			data = pk.load(f)	

			volume = data['volume'] 
			scale = data['scale']
			pixel_width = data['pixel_width']

		return volume, scale, pixel_width