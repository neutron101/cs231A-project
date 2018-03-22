import math
import numpy as np
import matplotlib.pyplot as plt
from seg_utils import meanshift_segmentation, kmeans_segmentation
from skimage.color import convert_colorspace 
from settings import debug

###############################################################################
class Method:

	def __init__(self, func, desc, k):
		self.func = lambda data, features: func(data, features, k)
		self.desc = desc


###############################################################################
class VolumeEstimatorByAggregation:

	KMEANS = Method(kmeans_segmentation, 'KMEANS', 2)
	MEANSHIFT = Method(meanshift_segmentation, 'MEANSHIFT', .25)

	def __init__(self, pixel_width, pixel_scale):
		self.pixel_width = pixel_width
		self.pixel_scale = pixel_scale

	def with_kmeans(self):
		self.method = VolumeEstimatorByAggregation.KMEANS
		return self

	def with_meanshift(self):
		self.method = VolumeEstimatorByAggregation.MEANSHIFT
		return self	

	def estimate(self, image_data):

		refined_sils = []
		for frame in image_data.frames[0:2]:
			box = self.bounding_box(frame.silhouette)

			if debug():
				print "Bounding box [minx miny width height]", box

			sil = self.refine_sil(frame.image, frame.silhouette, box, frame.image_path)
			
			refined_sils.append(sil)

		total_pixels = self.agg_pixels(refined_sils)
		volume = self.calc_vol(total_pixels, \
			image_data.scale if image_data.scale is not None else self.pixel_scale, \
			image_data.pixel_width if image_data.pixel_width is not None else self.pixel_width)

		return volume

	# Rerturns [miny minx height width]
	def bounding_box(self, sil):
		white_pixels = np.where(sil>0)

		box = np.array([white_pixels[1].min(), white_pixels[0].min(), white_pixels[1].max()-white_pixels[1].min(), white_pixels[0].max()-white_pixels[0].min()])

		return box

	def refine_sil(self, im, sil, b_box, img_name):

		focus_im = im[b_box[1]:b_box[1]+b_box[3], b_box[0]:b_box[0]+b_box[2], :]
		focus_sil = sil[b_box[1]:b_box[1]+b_box[3], b_box[0]:b_box[0]+b_box[2]]


		#focus_im = convert_colorspace(focus_im, fromspace='RGB', tospace='HSV')
		features = self._featurize(focus_im)

		im_cl_assign = self.method.func(focus_im, features)

		cl_values = im_cl_assign[np.where(focus_sil>0)]
		cl_values = np.unique(cl_values, return_counts=True)

		indexes = np.where(im_cl_assign==cl_values[0][cl_values[1].argmax()])
		sil_index_y = indexes[0] + b_box[1]
		sil_index_x = indexes[1] + b_box[0]

		new_sil = np.copy(sil)
		new_sil[sil_index_y,sil_index_x] = 1

		if debug():			
			fig = plt.figure(figsize=(5,5))
			fig.subplots_adjust(wspace=0.01, hspace=0.01)

			plt.subplot(221)
			plt.imshow(focus_im)
			plt.axis('off')
			plt.title(img_name)

			plt.subplot(222)
			plt.title(self.method.desc)
			plt.imshow(im_cl_assign)
			plt.axis('off')

			plt.subplot(223)
			plt.imshow(sil)
			plt.axis('off')

			plt.subplot(224)
			plt.imshow(new_sil)
			plt.axis('off')	

			plt.show()

		return new_sil

	def agg_pixels(self, sils):
		
		pixel_count = 0
		if len(sils) <= 0:
			raise ValueError("expected 2 silhouettes for pixel calculation")

		sil1 = np.where(sils[0] > 0)
		sil2 = np.where(sils[1] > 0)

		rows1 = zip(sil1[0], sil1[1])
		rows2 = zip(sil2[0], sil2[1])

		sil2_ht = np.unique(sil2[0]).shape[0]
		rows2_sorted_by_ht = sorted(rows2, key=lambda x:x[0])

		depths = np.zeros([sil2_ht])

		old_h = rows2_sorted_by_ht[0][0]
		d = 0
		for h,w in rows2_sorted_by_ht:
			if old_h < h:
				d+=1 
				old_h=h

			depths[d]+=1

		rows1_sorted_by_ht = sorted(rows1, key=lambda x:x[0])
		old_h = rows1_sorted_by_ht[0][0]
		d = 0
		for h,w in rows1_sorted_by_ht:
			if old_h < h:
				d+=1 
				old_h=h

			if d >= depths.shape[0]:
				break

			pixel_count+=depths[d] 

		return pixel_count


	def calc_vol(self, no_of_pixels, scale_a, pixel_width):
		return no_of_pixels * np.power(float(scale_a) * float(pixel_width), 3)

	def _featurize(self, img):
		features = np.zeros((img.shape[0] * img.shape[1], 3))
		for row in xrange(img.shape[0]):
		    for col in xrange(img.shape[1]):
		        features[row*img.shape[1] + col, :] = np.array([
		            img[row, col, 0], img[row, col, 1], img[row, col, 2]])
		features_normalized = features / features.max(axis = 0)

		return features_normalized
    


