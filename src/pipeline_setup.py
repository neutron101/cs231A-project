class PipelineSetup:

	_PIXEL_WIDTH = 5.5458515194e-5
	_SCALE = 4.05

	def __init__(self, base_dir, category=None):
		self.base_dir = base_dir
		self.category = category

		self.pixel_width = PipelineSetup._PIXEL_WIDTH
		self.scale = PipelineSetup._SCALE