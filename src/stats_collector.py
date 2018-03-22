import numpy as np
import math
from sympy import pretty_print as pp, latex

class StatsCollector:

	TRUE_INDEX = '__TRUE__'

	def __init__(self):
		self.data = {}

	def add_true_for_instance(self, category, instance, value):
		data = self.get_instance_data(category, instance)
		data[StatsCollector.TRUE_INDEX] = value


	def add_predicted_for_instance(self, category, instance, label, value):
		data = self.get_instance_data(category, instance)
		data[label] = value


	def get_instance_data(self, category, instance):
		if category not in self.data:
			self.data[category] = {}
		cat_data = self.data[category]

		if instance not in cat_data:
			cat_data[instance] = {}

		data = cat_data[instance]
		return data


	def print_stats(self):
		
		print (50 * "-")
		print (50 * "-")

		print "Category \t\t Method \t\t Count \t\t True Mean \t\t Predicted Mean \t\t Variance \t\t Mean volume error"

		for category, instance in self.data.iteritems():

			count = self.count(category)
			means = self.mean(category)
			variances = self.variance(category)
			mean_metric_errors = self.mean_metric_error(category)

			print_format = "%s \t\t\t %s \t\t %d \t\t\t %3.8f \t %3.8f \t\t\t %3.8f \t %3.8f"
			for label in means.keys():	
				print print_format % (category, \
					label, \
					count[label], \
					self.mean(category, true_value=True)[StatsCollector.TRUE_INDEX], \
					means[label], \
					variances[label], \
					mean_metric_errors[label])


	def mean(self, category, instance=None, true_value=False):

		method_labels, data = self.get_data(category, instance, true_value)
		return dict(zip(method_labels, data.mean(axis=0))) if data.shape[0] > 0 else None

	def variance(self, category, instance=None, true_value=False):
		method_labels, data = self.get_data(category, instance, true_value)
		return dict(zip(method_labels, data.var(axis=0))) if data.shape[0] > 0 else None

	def count(self, category, instance=None, true_value=False):
		method_labels, data = self.get_data(category, instance, true_value)
		return dict(zip(method_labels, np.repeat(data.shape[0], len(method_labels)))) if data.shape[0] > 0 else None

	def mean_metric_error(self, category, instance=None):
		method_labels, data = self.get_data(category, instance)
		tr_label, tr_data = self.get_data(category, instance, True)

		if len(tr_data.shape) == 1:
			tr_data = np.array([tr_data])

		a_data = data-tr_data.transpose()
		b_data = a_data/tr_data.transpose()
		c_data = b_data.mean(axis=0)

		return dict(zip(method_labels, c_data))
	

	def get_data(self, category, instance=None, true_value=False):
		
		data = None
		labels = None

		if instance is None:
			if category in self.data:

				no_inst = len(self.data[category].keys())
				if no_inst > 0:

					no_labels = 1
					labels = set([StatsCollector.TRUE_INDEX])

					if not true_value:
						no_labels = len(self.data[category][self.data[category].keys()[0]])
						labels = set(self.data[category][self.data[category].keys()[0]].keys()) - labels

					data = np.zeros([no_inst, no_labels])
					
					for i, d in enumerate(self.data[category].values()):
						for j, l in enumerate(labels):
							data[i,j] = d[l]

		return labels, data

