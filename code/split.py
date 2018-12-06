import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from code import visualize

# Iteration 13
def split_3(df, test_size=0.1, oversampling_ratio=1):
	print('\nSplit - Train&Dev Size = ', 1-test_size, ' , Test Size = ', test_size, '.', sep='')
	vcs_one = df['city_id'].value_counts()
	vcs_one = np.array(vcs_one[vcs_one==1].index)
	df['city_id'][df['city_id'].isin(vcs_one)]=-1
	df_train_dev, df_test = train_test_split(df, test_size=test_size, shuffle=True, stratify=df['city_id'], random_state=0)
	n = round(oversampling_ratio*len(df_train_dev))
	df_sampled = df_train_dev.sample(n=n, replace=True, weights=1+np.log1p(df_train_dev['n_clicks']), random_state=0)
	df_train_dev = pd.concat([df_train_dev, df_sampled])
	X_train_dev = np.array(df_train_dev.drop(columns=['n_clicks']), dtype=float)
	y_train_dev = np.array(df_train_dev['n_clicks'], dtype=float)
	X_test = np.array(df_test.drop(columns=['n_clicks']), dtype=float)
	y_test = np.array(df_test['n_clicks'], dtype=float)
	print('n_clicks summaries in train and test sets:', pd.DataFrame(y_train_dev).describe(), pd.DataFrame(y_test).describe(), sep='\n')
	visualize.compare_last_quantiles(pd.Series(y_train_dev), pd.Series(y_test), 'n_clicks Balance Assessment', 'Training & Development Set', 'Hold-Out Test Set')
	print('city_ids summaries in train and test sets:', pd.Series(X_train_dev[:, 0]).value_counts().describe(), pd.Series(X_test[:, 0]).value_counts().describe(), sep='\n')
	visualize.compare_last_quantiles(pd.Series(X_train_dev[:, 0]).value_counts(), pd.Series(X_test[:, 0]).value_counts(), 'city_id Balance Assessment', 'Training & Development Set', 'Hold-Out Test Set')
	return (X_train_dev, y_train_dev), (X_test, y_test)

# Iteration 10-12
def split_2(df, test_size=0.1, oversampling_ratio=1):
	print('\nSplit - Train&Dev Size = ', 1-test_size, ' , Test Size = ', test_size, '.', sep='')
	df_test = df.sample(frac=test_size, replace=False, random_state=0)
	df_train_dev = df[~df.index.isin(df_test.index)]
	n = round(oversampling_ratio*len(df_train_dev))
	df_sampled = df_train_dev.sample(n=n, replace=True, weights=1+np.log1p(df_train_dev['n_clicks']), random_state=0)
	df_train_dev = pd.concat([df_train_dev, df_sampled])
	X_train_dev = np.array(df_train_dev.drop(columns=['n_clicks']), dtype=float)
	y_train_dev = np.array(df_train_dev['n_clicks'], dtype=float)
	X_test = np.array(df_test.drop(columns=['n_clicks']), dtype=float)
	y_test = np.array(df_test['n_clicks'], dtype=float)
	print('n_clicks summaries in train and test sets:', pd.DataFrame(y_train_dev).describe(), pd.DataFrame(y_test).describe(), sep='\n')
	visualize.compare_last_quantiles(pd.Series(y_train_dev), pd.Series(y_test), 'n_clicks Balance Assessment', 'Training & Development Set', 'Hold-Out Test Set')
	return (X_train_dev, y_train_dev), (X_test, y_test)

# Iteration 10
def split_1(df, test_size=0.1):
	print('\nSplit - Train&Dev Size = ', 1-test_size, ' , Test Size = ', test_size, '.', sep='')
	test_len = round(test_size*len(df))
	bins = np.arange(6)
	y_binned = np.digitize(np.log1p(df['n_clicks']), bins)
	partitions = [df.loc[y_binned==b+1, :] for b in bins]
	bin_sz = min([len(p) for p in partitions])
	ss_df = pd.DataFrame(data=None, columns=df.columns)
	for p in partitions[1:]:
		ss_df = pd.concat([ss_df, p.sample(n=bin_sz, replace=False, random_state=0)])
	X = np.array(ss_df.drop(columns=['n_clicks']), dtype=float)
	y = np.array(ss_df['n_clicks'], dtype=float)
	X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=0)
	print('n_clicks summaries in train and test sets:', pd.DataFrame(y_train_dev).describe(), pd.DataFrame(y_test).describe(), sep='\n')
	visualize.compare_last_quantiles(pd.Series(y_train_dev), pd.Series(y_test), 'n_clicks Balance Assessment', 'Training & Development Set', 'Hold-Out Test Set')
	return (X_train_dev, y_train_dev), (X_test, y_test)

# Iteration 1-9
def split_0(df, test_size=0.1):
	print('\nSplit - Train&Dev Size = ', 1-test_size, ' , Test Size = ', test_size, '.', sep='')
	test_len = round(test_size*len(df))
	X = np.array(df.drop(columns=['n_clicks']), dtype=float)
	y = np.array(df['n_clicks'], dtype=float)
	X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=0)
	print('n_clicks summaries in train and test sets:', pd.DataFrame(y_train_dev).describe(), pd.DataFrame(y_test).describe(), sep='\n')
	visualize.compare_last_quantiles(pd.Series(y_train_dev), pd.Series(y_test), 'n_clicks Balance Assessment', 'Training & Development Set', 'Hold-Out Test Set')
	return (X_train_dev, y_train_dev), (X_test, y_test)
