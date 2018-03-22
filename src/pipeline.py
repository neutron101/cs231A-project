from volume_estimator import VolumeEstimator
from volume_estimator_by_aggregation import VolumeEstimatorByAggregation
import utils
from imagedata import ImageData
from stats_collector import StatsCollector
from image_attribute_processor import ImageAttributeProcessor
import os
from settings import debug


class Pipeline:

	def __init__(self, pipeline_setup):
		self.pipeline_setup = pipeline_setup

	def run(self):

		image_processor = ImageAttributeProcessor(self.pipeline_setup.base_dir)

		self.volume_estimator = self.createVolumeEstimatorByAggregation()		
		self.stats = StatsCollector()

		if self.pipeline_setup.category is not None:
			image_processor.process_single_target(self.pipeline_setup.category)
			self.run_instance(os.path.join(self.pipeline_setup.base_dir, self.pipeline_setup.category), self.pipeline_setup.category)
		else:
			for category_name in utils.dirs_at_dir(self.pipeline_setup.base_dir):
				image_processor.process_all(category_name)	

				for instance_path in utils.dirpath_at_dir(os.path.join(self.pipeline_setup.base_dir, category_name)):
					self.run_instance(instance_path, category_name)
		
		self.stats.print_stats()


	def run_instance(self, category_path, category):

		print 'Estimating volume for path', category_path

		try:
		 	image_data = ImageData(category_path)

		 	instance = category_path.split('/')[-2] if category_path.endswith('/') else category_path.split('/')[-1]
		 	self.stats.add_true_for_instance(category, instance, image_data.volume['value'])

		 	volume = self.volume_estimator.with_kmeans().estimate(image_data)
		 	self.stats.add_predicted_for_instance(category, instance, self.volume_estimator.method.desc, volume)

		 	if debug():
			 	print u'Predicted volume for target %s = %3.9f %s\u00B3 with %s.' % (instance, volume, image_data.volume['units'], self.volume_estimator.method.desc)

		 	volume = self.volume_estimator.with_meanshift().estimate(image_data)
		 	self.stats.add_predicted_for_instance(category, instance, self.volume_estimator.method.desc, volume)

		 	if debug():
			 	print u'Predicted volume for target %s = %3.9f %s\u00B3 with %s.' % (instance, volume, image_data.volume['units'], self.volume_estimator.method.desc)

		except Exception as e:
			print "Exception occurred dring volume estimation, skipping image at", category_path, e
				

	def createVolumeEstimatorByAggregation(self):
		return VolumeEstimatorByAggregation(self.pipeline_setup.pixel_width, self.pipeline_setup.scale)


	def createVolumeEstimator(self):
		return VolumeEstimator()
