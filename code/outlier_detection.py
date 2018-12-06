import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

_z_score_thresh = 3.5

def z_score(series, thresh=_z_score_thresh):
	z_score = (series-np.mean(series))/np.std(series)
	return np.abs(z_score)>thresh

def modified_z_score(series, thresh=_z_score_thresh):
	mod_z_score = 0.6745*(series-np.median(series))/series.mad()
	return np.abs(mod_z_score)>thresh

def elliptic_envelope(series, contamination=0.1):
	clf = EllipticEnvelope(contamination=contamination, random_state=0)
	series = series.values.reshape(-1, 1)
	clf.fit(series)
	return clf.predict(series)

def isolation_forest(series, contamination=0.1):
	clf = IsolationForest(contamination=contamination, random_state=0)
	series = series.values.reshape(-1, 1)
	clf.fit(series)
	return clf.predict(series)

def isolation_forest_new(series):
	clf = IsolationForest(behaviour='new', contamination='auto', random_state=0)
	series = series.values.reshape(-1, 1)
	clf.fit(series)
	return clf.predict(series)

def dbscan(series):
	series = series.values.reshape(-1, 1)
	scaler = MinMaxScaler()
	series = scaler.fit_transform(series)
	clf = DBSCAN(algorithm='kd_tree', random_state=0)		# default leaf_size reduced from 30 to 5 for complexity reasons
	return clf.fit_predict(series)

def get_proportion(series, inf, sup):
	return len(series[(series>inf) & (series<sup)]) / len(series)

def outlier_print(title, inliers, series):
	print('Outlier detection: {} identified outliers outside of the range [{}, {}]. Resulting outlier proportion: {}.'.format(title, round(min(inliers)), round(max(inliers)), round(get_proportion(series, max(inliers), max(series)+1), 5)))

def outlier_detection(series):
	inliers_0 = series[~z_score(series)]
	outlier_print('Z-Score', inliers_0, series)
	inliers_1 = series[~modified_z_score(series)]
	outlier_print('Modified Z-Score', inliers_1, series)
	inliers_2 = series[elliptic_envelope(series) == 1]
	outlier_print('Elliptic Envelope', inliers_2, series)
	inliers_3 = series[isolation_forest(series) == 1]
	outlier_print('Isolation Forest', inliers_3, series)
	inliers_4 = series[isolation_forest_new(series) == 1]
	outlier_print('Isolation Forest (new version)', inliers_4, series)
