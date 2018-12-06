import pandas as pd

def import_data():
	train_data = pd.read_csv('./data/train_set.csv')
	test_data = pd.read_csv('./data/test_set.csv')
	sample_data = pd.read_csv('./data/sample_submission.csv')
	print('Training dataset:', train_data.head(), 'Test dataset:', test_data.head(), 'Sample submission:', sample_data.head(), sep='\n')
	return train_data, test_data, sample_data
