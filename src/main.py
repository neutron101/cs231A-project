import utils
from pipeline import Pipeline
from pipeline_setup import PipelineSetup 
from settings import debug, on, off

def train_pipeline():

	on()
	setup = PipelineSetup('../data/', 'train')	
	return Pipeline(setup)


def test_pipeline():

	on()
	setup = PipelineSetup('../data/test')
	return Pipeline(setup)


if __name__ == '__main__':

	off()

	training_pipeline = test_pipeline()
	training_pipeline.run()