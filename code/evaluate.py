from code import train, visualize
import numpy as np
import pandas as pd
from sklearn.base import clone

def wmse(y_true, y_pred):
	"""Compute WMSE metric."""
	w = 1 + np.log(1+y_true)
	return np.sum(w * (y_true-y_pred)**2) / (len(w) * np.sum(w))

def wmse_log(y_true, y_pred):
	"""Compute WMSE metric on log-transformed n_clicks."""
	y_pred = np.expm1(y_pred)
	w = 1 + np.log(1+y_true)
	return np.sum(w * (y_true-y_pred)**2) / (len(w) * np.sum(w))

# Iteration 12-13
def eval_3(X_test, y_test, model):
	print('Evaluate')
	X_test = model[0].transform(X_test)
	y_pred_test = model[1].predict(X_test)
	y_pred_test = np.clip(np.round(y_pred_test), 0, None)
	print('Random Baseline WMSE:', round(wmse(y_test, np.ones(y_test.shape)*np.mean(y_test)), 5))
	print('Test WMSE:', round(wmse(y_test, y_pred_test), 5))
	visualize.eval_regression(y_test, y_pred_test, len(y_test))
	return y_pred_test

# Iteration 3-11
def eval_2(X_test, y_test, model):
	print('Evaluate')
	y_pred_test = model.predict(X_test)
	y_pred_test = np.clip(np.round(np.expm1(y_pred_test)), 0, None)
	print('Random Baseline WMSE:', round(wmse(y_test, np.ones(y_test.shape)*np.mean(y_test)), 5))
	print('Test WMSE:', round(wmse(y_test, y_pred_test), 5))
	visualize.eval_regression(y_test, y_pred_test, len(y_test))
	return y_pred_test

# Iteration 2
def eval_1(X_test, y_test, model):
	print('Evaluate')
	y_pred_test = model.predict(X_test)
	y_pred_test = np.round(np.expm1(y_pred_test))
	print('Test WMSE:', round(wmse(y_test, y_pred_test), 5))
	return y_pred_test

# Iteration 1
def eval_0(X_test, y_test, model):
	print('Evaluate')
	y_pred_test = model.predict(X_test)
	y_pred_test = np.round(np.maximum(y_pred_test, 0))
	print('Test WMSE:', round(wmse(y_test, y_pred_test), 5))
	return y_pred_test

# def test(model, train_data, test_data):
# 	"""Train the model on available train_data, then make predictions on test_data and produce the submission file."""
# 	hotel_ids = test_data['hotel_id']
# 	test_data.drop(columns=['hotel_id'], inplace=True)
# 	X_train = np.array(train_data.drop(columns=['n_clicks']), dtype=float)
# 	y_train = np.array(train_data['n_clicks'], dtype=float)
# 	X_test = np.array(test_data, dtype=float)
# 	model = clone(model)
# 	y_train = np.log1p(y_train)
# 	model.fit(X_train, y_train)
# 	y_pred = model.predict(X_test)
# 	y_pred = np.round(np.expm1(y_pred))
# 	df = pd.DataFrame(
# 		{'hotel_id' : hotel_ids,
# 		 'n_clicks' : y_pred}
# 	)
# 	df.to_csv('submission.csv', index=False)
