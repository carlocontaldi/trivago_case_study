import numpy as np
import pandas as pd
from code import visualize, outlier_detection

def are_whole_numbers(series):
	return series.apply(lambda n:n.is_integer()).all()

def eda(train, test, sample):
	print('\nExploratory Data Analysis')
	print('# train entries:', len(train), ', # test entries:', len(test))
	print('train column names & types:', train.dtypes, sep='\n')
	print('test column names & types:', test.dtypes, sep='\n')
	print('sample column names & types:', sample.dtypes, sep='\n')
	train = train.apply(pd.to_numeric, errors='coerce', downcast='float')
	test = test.apply(pd.to_numeric, errors='coerce', downcast='float')
	sample = sample.apply(pd.to_numeric, errors='coerce', downcast='float')
	print('All column types have been converted to float. Non-numerical values have been replaced with NaNs.')

	print('\nMissing Data Handling')
	print('All entries in train, test and sample sets have non-null hotel_id:', ~train['hotel_id'].isnull().any() & ~test['hotel_id'].isnull().any() & ~sample['hotel_id'].isnull().any())
	print('# train entries with missing [city_id, content_score, n_images, distance_to_center, avg_rating, stars, n_reviews]:', len(train[train['city_id'].isnull() & train['content_score'].isnull() & train['n_images'].isnull() & train['distance_to_center'].isnull() & train['avg_rating'].isnull() & train['stars'].isnull() & train['n_reviews'].isnull()]))
	train = train[train['city_id'].notnull()]
	print('Above-mentioned entries with missing values have been dropped.')
	print('# train entries with missing values by feature:', train.isnull().sum(), sep='\n')
	train.dropna(subset=['n_images', 'distance_to_center', 'avg_price'], inplace=True)
	print('Above-mentioned entries with missing values have been dropped, excepting for avg_rating and stars.')

	print('\nFeature analysis: hotel_id')
	# print('All entries in train, test and sample sets have non-null hotel_id:', ~train['hotel_id'].isnull().any() & ~test['hotel_id'].isnull().any() & ~sample['hotel_id'].isnull().any())
	print('All hotel_ids are whole numbers:', are_whole_numbers(train['hotel_id']) and are_whole_numbers(test['hotel_id']))
	print('All entries have unique hotel_id in each set:', len(train['hotel_id'])==train['hotel_id'].nunique() and len(test['hotel_id'])==test['hotel_id'].nunique())
	trainIdSet, testIdSet, sampleIdSet = set(train['hotel_id']), set(test['hotel_id']), set(sample['hotel_id'])
	print('train and test sets are disjoint:', len(trainIdSet|testIdSet) == len(trainIdSet) + len(testIdSet))
	print('test and sample sets involve the same entries:', len(testIdSet)==len(sampleIdSet) & len(testIdSet|sampleIdSet) == len(testIdSet))
	train.drop(columns='hotel_id', inplace=True)
	print('The hotel_id column has been dropped.')

	print('\nFeature analysis: city_id')
	print('# train entries with missing city_id:', train['city_id'].isnull().sum())
	## print('# train entries with missing [city_id, content_score, n_images, distance_to_center, avg_rating, n_reviews]:', len(train[train.loc[:, ['city_id', 'content_score', 'n_images', 'distance_to_center', 'avg_rating', 'n_reviews']].isnull().all(axis=1)]))	# implementation on the next line is faster
	# print('# train entries with missing [city_id, content_score, n_images, distance_to_center, avg_rating, stars, n_reviews]:', len(train[train['city_id'].isnull() & train['content_score'].isnull() & train['n_images'].isnull() & train['distance_to_center'].isnull() & train['avg_rating'].isnull() & train['stars'].isnull() & train['n_reviews'].isnull()]))
	# train = train[train['city_id'].notnull()]
	# print('Above-mentioned entries have been dropped.')
	print('All city_ids in train set are whole numbers:', are_whole_numbers(train['city_id']))
	print('city_id value counts summary:', train['city_id'].value_counts().describe(), sep='\n')
	visualize.twentiles(train['city_id'].value_counts(), 'city_id Value Counts')
	visualize.sorted_cumsum_by_city_id(train)
	
	print('\nFeature analysis: content_score')
	# print('# train entries with missing content_score:', train['content_score'].isnull().sum())
	print('All content_scores in train set are whole numbers:', are_whole_numbers(train['content_score']))
	print('content_score summary:', train['content_score'].describe(), sep='\n')
	visualize.kde(train['content_score'], -10, 110, 'content_score')
	print('Proportion of train entries having content_score in the range [40, 65]:', round(len(train[(train['content_score']>=40) & (train['content_score']<=65)])/len(train), 3))
	print('Proportion of train entries having content_score in the range [16, 24]:', round(len(train[(train['content_score']>=16) & (train['content_score']<=24)])/len(train), 3))

	print('\nFeature analysis: n_images')
	# print('# train entries with missing n_images:', train['n_images'].isnull().sum())
	# train = train[train['n_images'].notnull()]
	# print('Above-mentioned entries have been dropped.')
	print('All n_images in train set are whole numbers:', are_whole_numbers(train['n_images']))
	print('n_images summary:', train['n_images'].describe(), sep='\n')
	print('n_images top value counts:', train['n_images'].value_counts().iloc[:16], sep='\n')
	visualize.twentiles(train['n_images'], 'n_images')
	visualize.kde(train['n_images'], -2, 20, 'n_images', 0.99)	# series has been clipped from extreme outliers at 0.99 quantile to improve graphic result
	visualize.box(train['n_images'], 'n_images', -25, 200)
	q1, q3 = train['n_images'].quantile([.25, .75])
	iqr_thresh = q3+1.5*(q3-q1)
	print('The proportion of samples in ({}, 75] is {}.'.format(round(iqr_thresh), round(outlier_detection.get_proportion(train['n_images'], iqr_thresh, 76), 5)))
	# outlier_detection.outlier_detection(train['n_images'])

	print('\nFeature analysis: distance_to_center')
	# print('train entries with missing distance_to_center:', train[train['distance_to_center'].isnull()], sep='\n')
	# train = train[train['distance_to_center'].notnull()]
	# print('Above-mentioned entries have been dropped.')
	print('All distance_to_center values in train set are whole numbers:', are_whole_numbers(train['distance_to_center']))
	print('distance_to_center summary:', train['distance_to_center'].describe(), sep='\n')
	visualize.twentiles(train['distance_to_center'], 'distance_to_center', 2)
	visualize.kde(train['distance_to_center'], -1e4, 5e4, 'distance_to_center', 0.98)	# series has been filtered from extreme outliers to improve graphic result
	visualize.box(train['distance_to_center'], 'distance_to_center', -1e4, 2e5)
	# outlier_detection.outlier_detection(train['distance_to_center'])

	print('\nFeature analysis: avg_rating')
	print('# train entries with missing avg_rating:', len(train[train['avg_rating'].isnull()]))
	missing = train[train['avg_rating'].isnull()]
	non_missing = train[train['avg_rating'].notnull()]
	print('All avg_rating values in train set are whole numbers:', are_whole_numbers(non_missing['avg_rating']))
	print('avg_rating (non-missing) summary:', non_missing.describe(), sep='\n')
	print('avg_rating (missing) summary:', missing.describe(), sep='\n')
	visualize.compare_twentiles(non_missing['n_images'], missing['n_images'], 'n_images', 'avg_rating non_missing', 'avg_rating missing')
	visualize.compare_twentiles(non_missing['stars'], missing['stars'], 'stars', 'avg_rating non_missing', 'avg_rating missing')
	visualize.twentiles(non_missing['avg_rating'], 'avg_rating (non-missing)')
	visualize.kde(non_missing['avg_rating'], 40, 100, 'avg_rating (non-missing)')
	finalized_avg_rating = train['avg_rating'].copy()
	finalized_avg_rating[finalized_avg_rating.isnull()] = round(non_missing['avg_rating'].mean())
	print('Missing values have been imputed with a rounded mean imputation strategy.')
	visualize.kde(finalized_avg_rating, 40, 100, 'avg_rating (after rounded mean imputation)')

	print('\nFeature analysis: stars')
	# print('# train entries with missing stars:', len(train[train['stars'].isnull()]))
	print('stars summary:', train['stars'].describe(), sep='\n')
	print('All non-missing stars values in train set are whole numbers:', are_whole_numbers(train['stars'][train['stars'].notnull()]))
	print('Sizes of data grouped by stars:', train.groupby('stars')['stars'].count(), sep='\n')
	visualize.stars_compare_violins(train)
	visualize.kde(train['stars'][train['stars']>0], 0, 6, 'stars ([1, 5] only)')
	finalized_stars = train['stars'].copy()
	finalized_stars[finalized_stars.isnull()] = 0
	finalized_stars[finalized_stars==0] = round(finalized_stars[finalized_stars>0].mean())
	visualize.kde(finalized_stars, 0, 6, 'stars (after rounded mean imputation)')
	print('Missing and 0 values in stars have been replaced with a rounded mean imputation strategy.')

	print('\nFeature analysis: n_reviews')
	# print('# train entries with missing n_reviews:', len(train[train['n_reviews'].isnull()]))
	print('All n_reviews values in train set are whole numbers:', are_whole_numbers(train['avg_rating']))
	print('n_reviews summary:', train['n_reviews'].describe(), sep='\n')
	visualize.twentiles(train['n_reviews'], 'n_reviews')
	visualize.kde(train['n_reviews'], -1e3, 2e4, 'n_reviews', 0.99)
	visualize.box(train['n_reviews'], 'n_reviews', -1e3, 1e4)
	# outlier_detection.outlier_detection(train['n_reviews'])

	print('\nFeature analysis: avg_rank')
	# print('# train entries with missing avg_rank:', len(train[train['avg_rank'].isnull()]))
	print('All avg_rank values in train set are whole numbers:', are_whole_numbers(train['avg_rank']))
	print('avg_rank summary:', train['avg_rank'].describe(), sep='\n')
	visualize.kde(train['avg_rank'], -5, 105, 'avg_rank')
	visualize.last_quantiles(train['avg_rank'], 'avg_rank')
	visualize.box(train['avg_rank'], 'avg_rank', 73, 105)
	print('Number of data points with avg_rank equal to each integer in the range [77, 100]: {}, for a total of {} data points.'.format([len(train[train['avg_rank']==i]) for i in range(77, 101)], len(train[train['avg_rank']>=77])))
	# outlier_detection.outlier_detection(train['avg_rank'])

	print('\nFeature analysis: avg_price')
	# print('# train entries with missing avg_price:', len(train[train['avg_price'].isnull()]))
	# train = train[train['avg_price'].notnull()]
	# print('Above-mentioned entries have been dropped.')
	print('avg_price summary:', train['avg_price'].describe(), sep='\n')
	visualize.twentiles(train['avg_price'], 'avg_price')
	visualize.kde(train['avg_price'], 0, 1000, 'avg_price', 0.999)
	visualize.box(train['avg_price'], 'avg_price', 0, 8000)
	# outlier_detection.outlier_detection(train['avg_price'])

	print('\nFeature analysis: avg_saving_percent')
	# print('# train entries with missing avg_saving_percent:', len(train[train['avg_saving_percent'].isnull()]))
	print('All avg_saving_percent values in train set are whole numbers:', are_whole_numbers(train['avg_saving_percent']))
	print('avg_saving_percent summary:', train['avg_saving_percent'].describe(), sep='\n')
	visualize.twentiles(train['avg_saving_percent'], 'avg_saving_percent')
	visualize.kde(train['avg_saving_percent'], -5, 100, 'avg_saving_percent')
	visualize.kde(train['avg_saving_percent'][train['avg_saving_percent']>0], -5, 100, 'avg_saving_percent > 0')
	visualize.compare_twentiles(train['stars'][train['avg_saving_percent']==0], train['stars'][train['avg_saving_percent']>0], 'stars', 'avg_saving_percent=0', 'avg_saving_percent>0')

	print('\nFeature analysis: n_clicks')
	# print('# train entries with missing n_clicks:', len(train[train['n_clicks'].isnull()]))
	print('All n_clicks values in train set are whole numbers:', are_whole_numbers(train['n_clicks']))
	print('n_clicks summary:', train['n_clicks'].describe(), sep='\n')
	visualize.twentiles(train['n_clicks'], 'n_clicks')
	visualize.last_quantiles(train['n_clicks'], 'n_clicks')
	visualize.kde(train['n_clicks'], -50, 1000, 'n_clicks', 0.99)
	visualize.box(train['n_clicks'], 'n_clicks', -5, 25)
	print('All n_clicks values in train set are even numbers:', train['n_clicks'].apply(lambda n:n%2==0).all())
	print('Proportion of training entries having n_clicks=0:', round(len(train[train['n_clicks']==0])/len(train), 3))
