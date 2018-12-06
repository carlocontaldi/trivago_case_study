import numpy as np
import pandas as pd
import random
from code import visualize

# Iteration 9-13
def preprocess_2(df, testing=False, prior=None, imputation_strategy=0):
	# Convert all entries to float
	df = df.apply(pd.to_numeric, errors='coerce', downcast='float')
	if not testing:
		# Drop entries with NaNs in hotel_id and city_id
		df.dropna(subset=['hotel_id', 'city_id'], inplace=True)
		# Drop hotel_id column
		df.drop(columns=['hotel_id'], inplace=True)
		# Drop entries with NaNs in relevant features and out-of-expected-domain entries
		df = df[
		(df['content_score']>=0) &
		(df['content_score']<=100) &
		(df['n_images']>=-1) &
		(df['distance_to_center']>=0) &
		((df['avg_rating'].isnull()) | (df['avg_rating']>=0)) &
		((df['avg_rating'].isnull()) | (df['avg_rating']<=100)) &
		(df['stars']>=0) &
		(df['stars']<=5) &
		(df['n_reviews']>=0) &
		(df['avg_rank']>=1) &
		(df['avg_price']>0) &
		(df['avg_saving_percent']>=0) &
		(df['avg_saving_percent']<100) &
		(df['n_clicks']>=0)
		]
	else:
		feat_list = ['hotel_id', 'content_score', 'n_images', 'distance_to_center', 'stars', 'n_reviews', 'avg_rank', 'avg_price', 'avg_saving_percent']
		df.loc[:, feat_list] = df.loc[:, feat_list].fillna(0)
	# Impute missing values in avg_rating and stars
	if imputation_strategy==0:
		if not testing:
			avg_rating_rounded_mean = round(df['avg_rating'][df['avg_rating'].notnull()].mean())
		else:
			avg_rating_rounded_mean = prior[0]
		nan_avg_rating_mask = df['avg_rating'].isnull()
		df['avg_rating'][nan_avg_rating_mask] = avg_rating_rounded_mean
		df['nan_avg_rating'] = nan_avg_rating_mask.astype(float)
	else:
		raise ValueError
	# Add boolean columns
	df['nan_n_images'] = df['n_images']==-1
	# Address extreme outliers
	df['n_images'] = np.log(2+df['n_images'])
	df['distance_to_center'] = np.log(1+df['distance_to_center'])
	df['n_reviews'] = np.log(1+df['n_reviews'])
	# Visualize finalized data
	sample_sz=10000
	visualize.bi_kdes(df, testing, sample_sz)
	visualize.bi_kdes(df[df['n_clicks']>0], testing, sample_sz, ' - n_clicks > 0')
	visualize.pairplots(df, testing, sample_sz)
	visualize.plot_finalized(df, testing, sample_sz)
	# Rescale features having known domain
	df['content_score'] = df['content_score']/100
	df['avg_rating'] = df['avg_rating']/100
	df['avg_saving_percent'] = df['avg_saving_percent']/99
	city_id_freq = df['city_id'].value_counts()
	df['city_id_freq'] = df['city_id'].apply(lambda x:city_id_freq[x])
	if not testing:
		return df, avg_rating_rounded_mean
	else:
		return df

# Iteration 8
def preprocess_1(df, testing=False, prior=None, imputation_strategy=0):
	# Convert all entries to float
	df = df.apply(pd.to_numeric, errors='coerce', downcast='float')
	if not testing:
		# Drop entries with NaNs in hotel_id and city_id
		df.dropna(subset=['hotel_id', 'city_id'], inplace=True)
		# Drop hotel_id and city_id columns
		df.drop(columns=['hotel_id', 'city_id'], inplace=True)
		# Drop entries with NaNs in relevant features and out-of-expected-domain entries
		df = df[
		(df['content_score']>=0) &
		(df['content_score']<=100) &
		(df['n_images']>=-1) &
		(df['distance_to_center']>=0) &
		((df['avg_rating'].isnull()) | (df['avg_rating']>=0)) &
		((df['avg_rating'].isnull()) | (df['avg_rating']<=100)) &
		(df['stars']>=0) &
		(df['stars']<=5) &
		(df['n_reviews']>=0) &
		(df['avg_rank']>=1) &
		(df['avg_price']>0) &
		(df['avg_saving_percent']>=0) &
		(df['avg_saving_percent']<100) &
		(df['n_clicks']>=0)
		]
	else:
		df.drop(columns=['city_id'], inplace=True)
		feat_list = ['hotel_id', 'content_score', 'n_images', 'distance_to_center', 'stars', 'n_reviews', 'avg_rank', 'avg_price', 'avg_saving_percent']
		df.loc[:, feat_list] = df.loc[:, feat_list].fillna(0)
	# Impute missing values in avg_rating and stars
	if imputation_strategy==0:
		if not testing:
			avg_rating_rounded_mean = round(df['avg_rating'][df['avg_rating'].notnull()].mean())
		else:
			avg_rating_rounded_mean = prior[0]
		nan_avg_rating_mask = df['avg_rating'].isnull()
		df['avg_rating'][nan_avg_rating_mask] = avg_rating_rounded_mean
		df['nan_avg_rating'] = nan_avg_rating_mask.astype(float)
	else:
		raise ValueError
	# Add boolean columns
	df['nan_n_images'] = df['n_images']==-1
	# Address extreme outliers
	df['n_images'] = np.log(2+df['n_images'])
	df['distance_to_center'] = np.log(1+df['distance_to_center'])
	df['n_reviews'] = np.log(1+df['n_reviews'])
	# Visualize finalized data
	sample_sz=10000
	visualize.bi_kdes(df, testing, sample_sz)
	visualize.bi_kdes(df[df['n_clicks']>0], testing, sample_sz)
	visualize.pairplots(df, testing, sample_sz)
	visualize.plot_finalized(df, testing, sample_sz)
	# Rescale features having known domain
	df['content_score'] = df['content_score']/100
	df['avg_rating'] = df['avg_rating']/100
	df['avg_saving_percent'] = df['avg_saving_percent']/99
	if not testing:
		return df, avg_rating_rounded_mean
	else:
		return df

# Iteration 1-7
def preprocess_0(df, testing=False, prior=None, imputation_strategy=0):
	# Convert all entries to float
	df = df.apply(pd.to_numeric, errors='coerce', downcast='float')
	# Drop entries with NaNs in hotel_id and city_id
	df.dropna(subset=['hotel_id', 'city_id'], inplace=True)
	# Drop hotel_id column
	if not testing:
		df.drop(columns='hotel_id', inplace=True)
	# Drop entries with NaNs in relevant features and out-of-expected-domain entries
	df = df[
	(df['content_score']>=0) &
	(df['content_score']<=100) &
	(df['n_images']>=-1) &
	(df['distance_to_center']>=0) &
	((df['avg_rating'].isnull()) | (df['avg_rating']>=0)) &
	((df['avg_rating'].isnull()) | (df['avg_rating']<=100)) &
	((df['stars'].isnull()) | (df['stars']>=0)) &
	((df['stars'].isnull()) | (df['stars']<=5)) &
	(df['n_reviews']>=0) &
	(df['avg_rank']>=1) &
	(df['avg_price']>0) &
	(df['avg_saving_percent']>=0) &
	(df['avg_saving_percent']<100)
	]
	if not testing:
		df = df[df['n_clicks']>=0]
	# Impute missing values in avg_rating and stars
	if imputation_strategy==0:
		if not testing:
			avg_rating_rounded_mean = round(df['avg_rating'][df['avg_rating'].notnull()].mean())
			stars_rounded_mean = round(df['stars'][(df['stars'].notnull()) & (df['stars']>0)].mean())
		else:
			avg_rating_rounded_mean = prior[0]
			stars_rounded_mean = prior[1]
		nan_avg_rating_mask = df['avg_rating'].isnull()
		df['avg_rating'][nan_avg_rating_mask] = avg_rating_rounded_mean
		df['nan_avg_rating'] = nan_avg_rating_mask.astype(float)
		nan_stars_mask = (df['stars'].isnull()) | (df['stars']==0)
		df['stars'][nan_stars_mask] = stars_rounded_mean
		df['nan_stars'] = nan_stars_mask.astype(float)
	else:
		raise ValueError
	# Address extreme outliers
	df['n_images'] = np.log(2+df['n_images'])
	df['distance_to_center'] = np.log(1+df['distance_to_center'])
	df['n_reviews'] = np.log(1+df['n_reviews'])
	# Visualize finalized data
	sample_sz=10000
	visualize.bi_kdes(df, testing, sample_sz)
	visualize.bi_kdes(df[df['n_clicks']>0], testing, sample_sz)
	visualize.pairplots(df, testing, sample_sz)
	visualize.plot_finalized(df, testing, sample_sz)
	# Rescale features having known domain
	df['content_score'] = df['content_score']/100
	df['avg_rating'] = df['avg_rating']/100
	df['stars'] = df['stars']/5
	df['avg_saving_percent'] = df['avg_saving_percent']/99
	if not testing:
		return df, avg_rating_rounded_mean, stars_rounded_mean
	else:
		return df
