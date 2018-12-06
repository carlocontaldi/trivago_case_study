import sys
import datetime
from code import *
import pickle

def main(test_size=0.1, dev_size=0.1):
	train_data, test_data, sample_data = import_data.import_data()
	eda.eda(train_data, test_data, sample_data)
	train_data, *prior = preprocess.preprocess_2(train_data)
	# test_data = preprocess.preprocess_2(test_data, testing=True, prior=prior)
	(X_train_dev, y_train_dev), (X_test, y_test) = split.split_3(train_data, test_size)
	model = train.train_10(X_train_dev, y_train_dev, dev_size=dev_size, n_folds=5)
	y_pred_test = evaluate.eval_3(X_test, y_test, model)
	with open('y_pred_test.pkl', 'wb') as f:
		pickle.dump(y_pred_test, f)		# useful to evaluate statistical significance
	# evaluate.test(model, train_data, test_data)

if __name__ == "__main__":
	
	# Define Logger
	class Logger(object):
		def __init__(self):
			self.terminal = sys.stdout
			self.log = open("logfile.log", "w")

		def write(self, message):
			self.terminal.write(message)
			self.log.write(message)

		def flush(self):
			#this flush method is needed for python 3 compatibility.
			#this handles the flush command by doing nothing.
			pass

	sys.stdout = Logger()
	start_time = datetime.datetime.now()
	print('Main execution started at:', start_time)
	main()
	end_time = datetime.datetime.now()
	print('Main execution ended at: {}.\nTotal execution duration: {}.'.format(end_time, end_time-start_time))
