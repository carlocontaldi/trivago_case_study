import numpy as np
import pandas as pd
from scipy.sparse import issparse
from code import evaluate
from sklearn.base import clone, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics.scorer import make_scorer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures, MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, HuberRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

class DummyTransformer(TransformerMixin):
	def __init__(self):
		pass
	def transform(self, X, *_):
		return X
	def fit(self, X, *_):
		return self

def wmse_log_xgb(y_pred, d_xgb_test):
	y_true = d_xgb_test.get_label()
	return 'WMSE', float(evaluate.wmse_log(y_true, y_pred))

def wmse_xgb(y_pred, d_xgb_test):
	y_true = d_xgb_test.get_label()
	return 'WMSE', float(evaluate.wmse(y_true, y_pred))

def get_categories(X, inf=5):
	data = pd.DataFrame(X)
	vcs = data.iloc[:, 0].value_counts()
	vcs = vcs[vcs>=5]
	categories = [
		np.array(sorted(vcs.index)),
		np.arange(6)
	]
	return categories

# Iteration 12-13
def train_10(X_train_dev, y_train_dev, dev_size=0.1, n_folds=10):
	print('Model 10 - (MaxAbsScaler) + (OneHotEncoder), XGBRegressor on n_clicks with GridSearchCV')
	num_transformer = Pipeline([
		('normalizer', MaxAbsScaler())
	])
	cat_transformer = Pipeline([
		('dummy', DummyTransformer())
	])
	preprocessor = ColumnTransformer([
		('cat', cat_transformer, [0, 5]),
		('num', num_transformer, [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12])
	])
	model = xgb.XGBRegressor()
	wmse_scorer = make_scorer(evaluate.wmse, greater_is_better=False)
	param_grid = {
		'n_estimators': [500],
		'learning_rate': [.1, .2, .5],
		'max_depth': [3, 5, 7],
		'n_jobs': [-1],
		'random_state': [0]
	}
	cv = KFold(n_splits=n_folds, shuffle=True, random_state=0)
	grid = GridSearchCV(model, param_grid, cv=cv, scoring=wmse_scorer, iid=False, return_train_score=True, error_score=np.nan, n_jobs=-1, verbose=3)
	X_xgb_train_dev, X_xgb_test, y_xgb_train_dev, y_xgb_test = train_test_split(X_train_dev, y_train_dev, test_size=dev_size, shuffle=True, random_state=0)
	X_xgb_train_dev = preprocessor.fit_transform(X_xgb_train_dev)
	X_xgb_test = preprocessor.transform(X_xgb_test)
	fit_params = {
		'early_stopping_rounds' : 10,
		'sample_weight' : 1+y_xgb_train_dev,
		'sample_weight_eval_set' : [1+y_xgb_test],
		'eval_metric' : wmse_xgb,
		'eval_set' : [[X_xgb_test, y_xgb_test]]
	}
	grid.fit(X_xgb_train_dev, y_xgb_train_dev, **fit_params)
	print('Best Parameters:', grid.best_params_)
	print('Validation WMSE:', round(-grid.best_score_, 5))
	preprocessor, best_model = clone(preprocessor), clone(grid.best_estimator_)
	X_train_dev = preprocessor.fit_transform(X_train_dev)
	best_model.fit(X_train_dev, y_train_dev)
	return (preprocessor, best_model)

# Iteration 11
def train_9(X_train_dev, y_train_dev, dev_size=0.1, n_folds=10):
	print('Model 9 - (MinMaxScaler) + (OneHotEncoder), XGBRegressor on log1p(n_clicks) with GridSearchCV')
	cat_transformer = Pipeline([
		('encoder', OneHotEncoder(categories='auto', handle_unknown='ignore'))
	])
	num_transformer = Pipeline([
		('normalizer', MinMaxScaler(feature_range=(0, 1)))
	])
	no_transformer = Pipeline([
		('transformer', DummyTransformer())
	])
	preprocessor = ColumnTransformer([
		('cat', cat_transformer, [4]),
		('num', num_transformer, [1, 2, 5, 6, 7]),
		('no', no_transformer, [0, 3, 8, 9, 10])
	])
	regressor = xgb.XGBRegressor()
	model = Pipeline([
		('preprocessor', preprocessor),
		('regressor', regressor)
	])
	wmse_scorer = make_scorer(evaluate.wmse_log, greater_is_better=False)
	y_train_dev = np.log1p(y_train_dev)
	param_grid = {
		'regressor__n_estimators': [500],
		'regressor__learning_rate': [.005, 0.01, .03, .05],
		'regressor__max_depth': [5, 7, 9, 11],
		'regressor__n_jobs': [-1],
		'regressor__random_state': [0]
	}
	cv = KFold(n_splits=n_folds, shuffle=True, random_state=0)
	grid = GridSearchCV(model, param_grid, cv=cv, scoring=wmse_scorer, iid=False, return_train_score=True, error_score=np.nan, n_jobs=-1, verbose=3)
	X_xgb_train_dev, X_xgb_test, y_xgb_train_dev, y_xgb_test = train_test_split(X_train_dev, y_train_dev, test_size=dev_size, shuffle=True, random_state=0)
	fit_params = {
		'regressor__early_stopping_rounds' : 10,
		'regressor__sample_weight' : 1+y_xgb_train_dev,
		'regressor__sample_weight_eval_set' : [1+y_xgb_test],
		'regressor__eval_metric' : wmse_log_xgb,
		'regressor__eval_set' : [[X_xgb_test, y_xgb_test]]
	}
	grid.fit(X_xgb_train_dev, y_xgb_train_dev, **fit_params)
	print('Best Parameters:', grid.best_params_)
	print('Validation WMSE:', round(-grid.best_score_, 5))
	best_model = clone(grid.best_estimator_)
	best_model.fit(X_train_dev, y_train_dev)
	return best_model

# Iteration 9-10
def train_8(X_train_dev, y_train_dev, dev_size=0.1, n_folds=10):
	print('Model 8 - (MinMaxScaler) + (OneHotEncoder), RandomForestRegressor on log1p(n_clicks) with RandomizedSearchCV')
	num_transformer = Pipeline([
		('normalizer', MinMaxScaler(feature_range=(0, 1)))
	])
	cat_transformer = Pipeline([
		('encoder', OneHotEncoder(categories='auto', handle_unknown='ignore'))
	])
	no_transformer = Pipeline([
		('transformer', DummyTransformer())
	])
	preprocessor = ColumnTransformer([
		('cat', cat_transformer, [4]),
		('num', num_transformer, [1, 2, 5, 6, 7]),
		('no', no_transformer, [0, 3, 8, 9, 10])
	])
	regressor = RandomForestRegressor()
	model = Pipeline([
		('preprocessor', preprocessor),
		('regressor', regressor)
	])
	wmse_scorer = make_scorer(evaluate.wmse_log, greater_is_better=False)
	y_train_dev = np.log1p(y_train_dev)
	max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
	max_depth.append(None)
	param_dist = {
		'regressor__n_estimators': [int(x) for x in np.linspace(start = 50, stop = 500, num = 10)],
		'regressor__max_features': ['auto', 'sqrt'],
		'regressor__max_depth': max_depth,
		'regressor__min_samples_split': [2, 5, 10],
		'regressor__min_samples_leaf': [1, 2, 4],
		'regressor__bootstrap': [True, False],
		'regressor__random_state': [0]
	}
	cv = KFold(n_splits=n_folds, shuffle=True, random_state=0)
	rnd_search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=cv, scoring=wmse_scorer, iid=False, return_train_score=True, error_score=np.nan, n_jobs=-1, random_state=0, verbose=4)
	rnd_search.fit(X_train_dev, y_train_dev)
	print('Best Parameters:', rnd_search.best_params_)
	print('Validation WMSE:', round(-rnd_search.best_score_, 5))
	best_model = clone(rnd_search.best_estimator_)
	best_model.fit(X_train_dev, y_train_dev)
	return best_model

# Iteration 8
def train_7(X_train_dev, y_train_dev, dev_size=0.1, n_folds=10):
	print('Model 7 - (MinMaxScaler) + RandomForestRegressor on log1p(n_clicks) with GridSearchCV')
	num_transformer = Pipeline([
		('normalizer', MinMaxScaler(feature_range=(0, 1)))
	])
	regressor = RandomForestRegressor(random_state=0)
	model = Pipeline([
		('preprocessor', num_transformer),
		('regressor', regressor)
	])
	wmse_scorer = make_scorer(evaluate.wmse_log, greater_is_better=False)
	y_train_dev = np.log1p(y_train_dev)
	param_grid = {
		'regressor__n_estimators': [50, 100, 200],
		'regressor__max_features': [None, 1/3]
	}
	grid = GridSearchCV(model, param_grid, cv=10, scoring=wmse_scorer, iid=False, return_train_score=True, error_score=np.nan, n_jobs=-1, verbose=5)
	grid.fit(X_train_dev, y_train_dev)
	print('Best Parameters:', grid.best_params_)
	print('Validation WMSE:', round(-grid.best_score_, 5))
	best_model = clone(grid.best_estimator_)
	best_model.fit(X_train_dev, y_train_dev)
	return best_model

# Iteration 7
def train_6(X_train_dev, y_train_dev, dev_size=0.1, n_folds=10):
	print('Model 6 - (MinMaxScaler) + (OneHotEncoder, OneHotReducer), KernelizedSVR on log1p(n_clicks/2) with GridSearchCV')
	num_transformer = Pipeline([
		('normalizer', MinMaxScaler())
	])
	cat_transformer = Pipeline([
		('encoder', OneHotEncoder(categories='auto', handle_unknown='ignore')),
		('reducer', OneHotReducer())
	])
	preprocessor = ColumnTransformer([
		('cat', cat_transformer, [0, 5]),
		('num', num_transformer, [2, 3, 6, 7, 8])
	])
	regressor = SVR()
	model = Pipeline([
		('preprocessor', preprocessor),
		('regressor', regressor)
	])
	param_grid = {
		'regressor__kernel': ['poly', 'rbf'],
		'regressor__gamma': ['scale']
	}
	wmse_scorer = make_scorer(evaluate.wmse_log_2, greater_is_better=False)
	grid = GridSearchCV(model, param_grid, cv=10, scoring=wmse_scorer, iid=False, return_train_score=True, error_score=np.nan, n_jobs=-1, verbose=10)
	y_train_dev = np.log1p(y_train_dev/2)
	grid.fit(X_train_dev, y_train_dev)
	print('Best Parameters:', grid.best_params_)
	print('Validation WMSE:', round(-grid.best_score_, 5))
	best_model = clone(grid.best_estimator_)
	best_model.fit(X_train_dev, y_train_dev)
	return best_model

# Iteration 6
def train_5(X_train_dev, y_train_dev, dev_size=0.1, n_folds=10):
	print('Model 5 - (MinMaxScaler) + (OneHotEncoder, OneHotReducer), LinearSVR on log1p(n_clicks/2) with GridSearchCV')
	num_transformer = Pipeline([
		('normalizer', MinMaxScaler())
	])
	cat_transformer = Pipeline([
		('encoder', OneHotEncoder(categories='auto', handle_unknown='ignore')),
		('reducer', OneHotReducer())
	])
	no_transformer = Pipeline([
		('transformer', DummyTransformer())
	])
	preprocessor = ColumnTransformer([
		('cat', cat_transformer, [0, 5]),
		('num', num_transformer, [2, 3, 6, 7, 8]),
		('no', no_transformer, [1, 4, 9, 10, 11, 12])
	])
	regressor = LinearSVR()
	model = Pipeline([
		('preprocessor', preprocessor),
		('regressor', regressor)
	])
	param_grid = {
		'regressor__epsilon': [0, 0.01, 0.1, 1],
		'regressor__C': [0.01, 0.2, 0.5, 0.8, 1],
		'regressor__dual': [True, False],
		'regressor__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
		'regressor__random_state': [0]
	}
	wmse_scorer = make_scorer(evaluate.wmse_log_2, greater_is_better=False)
	grid = GridSearchCV(model, param_grid, cv=10, scoring=wmse_scorer, iid=False, return_train_score=True, error_score=np.nan, n_jobs=-1, verbose=10)
	y_train_dev = np.log1p(y_train_dev/2)
	grid.fit(X_train_dev, y_train_dev)
	print('Best Parameters:', grid.best_params_)
	print('Validation WMSE:', round(-grid.best_score_, 5))
	best_model = clone(grid.best_estimator_)
	best_model.fit(X_train_dev, y_train_dev)
	return best_model

# Iteration 5
def train_4(X_train_dev, y_train_dev, dev_size=0.1, n_folds=10):
	print('Model 4 - (MinMaxScaler, PolynomialFeatures) + (OneHotEncoder, OneHotReducer), ElasticNet on log1p(n_clicks/2) with GridSearchCV')
	num_transformer = Pipeline([
		('normalizer', MinMaxScaler()),
		('poly', PolynomialFeatures(3))
	])
	cat_transformer = Pipeline([
		('encoder', OneHotEncoder(categories='auto', handle_unknown='ignore')),
		('reducer', OneHotReducer())
	])
	no_transformer = Pipeline([
		('transformer', DummyTransformer())
	])
	preprocessor = ColumnTransformer([
		('cat', cat_transformer, [0, 5]),
		('num', num_transformer, [2, 3, 6, 7, 8]),
		('no', no_transformer, [1, 4, 9, 10, 11, 12])
	])
	regressor = ElasticNet()
	model = Pipeline([
		('preprocessor', preprocessor),
		('regressor', regressor)
	])
	param_grid = {
		'regressor__alpha': [10 ** x for x in range(-2, 1)],
		'regressor__l1_ratio': [0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 1],
		'regressor__selection': ['random'],	# enabled to speed up computation
		'regressor__random_state': [0]
	}
	wmse_scorer = make_scorer(evaluate.wmse_log_2, greater_is_better=False)
	grid = GridSearchCV(model, param_grid, cv=10, scoring=wmse_scorer, iid=False, return_train_score=True, error_score=np.nan, n_jobs=-1, verbose=10)
	y_train_dev = np.log1p(y_train_dev/2)
	grid.fit(X_train_dev, y_train_dev)
	print('Best Parameters:', grid.best_params_)
	print('Validation WMSE:', round(-grid.best_score_,5))
	best_model = clone(grid.best_estimator_)
	best_model.fit(X_train_dev, y_train_dev)
	return best_model

# Iteration 4
def train_3(X_train_dev, y_train_dev, dev_size=0.1, n_folds=10):
	print('Model 3 - (MinMaxScaler) + (OneHotEncoder, OneHotReducer), ElasticNet on log1p(n_clicks/2) with GridSearchCV')
	num_transformer = Pipeline([
		('normalizer', MinMaxScaler())
	])
	cat_transformer = Pipeline([
		('encoder', OneHotEncoder(categories='auto', handle_unknown='ignore')),
		('reducer', OneHotReducer())
	])
	no_transformer = Pipeline([
		('transformer', DummyTransformer())
	])
	preprocessor = ColumnTransformer([
		('cat', cat_transformer, [0]),
		('num', num_transformer, [2, 3, 6, 7, 8]),
		('no', no_transformer, [1, 4, 5, 9, 10, 11, 12])
	])
	regressor = ElasticNet()
	model = Pipeline([
		('preprocessor', preprocessor),
		('regressor', regressor)
	])
	param_grid = {
		'regressor__alpha': [10 ** x for x in range(-2, 2)],
		'regressor__l1_ratio': [0.01, 0.05, 0.2, 0.8, 0.95, 1],
		'regressor__selection': ['random'],	# enabled to speed up computation
		'regressor__random_state': [0]
	}
	wmse_scorer = make_scorer(evaluate.wmse_log_2, greater_is_better=False)
	grid = GridSearchCV(model, param_grid, cv=10, scoring=wmse_scorer, iid=False, return_train_score=True, error_score=np.nan, n_jobs=-1, verbose=10)
	y_train_dev = np.log1p(y_train_dev/2)
	grid.fit(X_train_dev, y_train_dev)
	print('Best Parameters:', grid.best_params_)
	print('Validation WMSE:', round(-grid.best_score_,5))
	best_model = clone(grid.best_estimator_)
	best_model.fit(X_train_dev, y_train_dev)
	return best_model

# Iteration 3
def train_2(X_train_dev, y_train_dev, dev_size=0.1, n_folds=10):
	print('Model 2 - (MinMaxScaler) + (OneHotEncoder, OneHotReducer), LinearRegression() on log1p(n_clicks/2)')
	num_transformer = Pipeline([
		('normalizer', MinMaxScaler())
	])
	cat_transformer = Pipeline([
		('encoder', OneHotEncoder(categories='auto', handle_unknown='ignore')),
		('reducer', OneHotReducer())
	])
	no_transformer = Pipeline([
		('transformer', DummyTransformer())
	])
	preprocessor = ColumnTransformer([
		('cat', cat_transformer, [0]),
		('num', num_transformer, [2, 3, 6, 7, 8]),
		('no', no_transformer, [1, 4, 5, 9, 10, 11, 12])
	])
	regressor = LinearRegression()
	model = Pipeline([
		('preprocessor', preprocessor),
		('regressor', regressor)
	])
	kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
	kf.get_n_splits(X_train_dev)
	y_train_dev = np.log1p(y_train_dev/2)
	y_pred_dev = y_train_dev.copy()
	for train_index, test_index in kf.split(X_train_dev):
		X_t, X_v = X_train_dev[train_index], X_train_dev[test_index]
		y_t = y_train_dev[train_index]
		k_model = clone(model)
		k_model.fit(X_t, y_t)
		y_pred_dev[test_index] = k_model.predict(X_v)
	y_pred_dev = np.round(np.expm1(y_pred_dev)*2)
	print('Validation WMSE:', round(evaluate.wmse(y_train_dev, y_pred_dev), 5))
	model.fit(X_train_dev, y_train_dev)
	return model

# Iteration 2
def train_1(X_train_dev, y_train_dev, dev_size=0.1, n_folds=10):
	print('Model 1 - (MinMaxScaler) + (OneHotEncoder, TruncatedSVD), LinearRegression on log1p(n_clicks/2)')
	num_transformer = Pipeline([
		('normalizer', MinMaxScaler())
	])
	cat_transformer = Pipeline([
		('encoder', OneHotEncoder(categories='auto', handle_unknown='ignore')),
		('dim_reducer', TruncatedSVD(300, random_state=0))
	])
	no_transformer = Pipeline([
		('transformer', DummyTransformer())
	])
	preprocessor = ColumnTransformer([
		('cat', cat_transformer, [0]),
		('num', num_transformer, [2, 3, 6, 7, 8]),
		('no', no_transformer, [1, 4, 5, 9, 10, 11, 12])
	])
	regressor = LinearRegression()
	model = Pipeline([
		('preprocessor', preprocessor),
		('regressor', regressor)
	])
	kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
	kf.get_n_splits(X_train_dev)
	y_train_dev = np.log1p(y_train_dev/2)
	y_pred_dev = y_train_dev.copy()
	for train_index, test_index in kf.split(X_train_dev):
		X_t, X_v = X_train_dev[train_index], X_train_dev[test_index]
		y_t = y_train_dev[train_index]
		k_model = clone(model)
		k_model.fit(X_t, y_t)
		y_pred_dev[test_index] = k_model.predict(X_v)
	y_pred_dev = np.round(np.expm1(y_pred_dev)*2)
	print('Validation WMSE:', round(evaluate.wmse(y_train_dev, y_pred_dev), 5))
	model.fit(X_train_dev, y_train_dev)
	return model

# Iteration 1
def train_0(X_train_dev, y_train_dev, dev_size=0.1, n_folds=10):
	print('Model 0 - (MinMaxScaler) + (OneHotEncoder, TruncatedSVD), LinearRegression')
	num_transformer = Pipeline([
		('normalizer', MinMaxScaler())
	])
	cat_transformer = Pipeline([
		('encoder', OneHotEncoder(categories='auto', handle_unknown='ignore')),
		('dim_reducer', TruncatedSVD(300, random_state=0))
	])
	no_transformer = Pipeline([
		('transformer', DummyTransformer())
	])
	preprocessor = ColumnTransformer([
		('cat', cat_transformer, [0, 5]),
		('num', num_transformer, [2, 3, 6, 7, 8]),
		('no', no_transformer, [1, 4, 5, 9, 10, 11, 12])
	])
	regressor = LinearRegression()
	model = Pipeline([
		('preprocessor', preprocessor),
		('regressor', regressor)
	])
	kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
	kf.get_n_splits(X_train_dev)
	y_pred_dev = y_train_dev.copy()
	for train_index, test_index in kf.split(X_train_dev):
		X_t, X_v = X_train_dev[train_index], X_train_dev[test_index]
		y_t = y_train_dev[train_index]
		k_model = clone(model)
		k_model.fit(X_t, y_t)
		y_pred_dev[test_index] = k_model.predict(X_v)
	y_pred_dev = np.round(np.maximum(y_pred_dev, 0))
	print('Validation WMSE:', round(evaluate.wmse(y_train_dev, y_pred_dev), 5))
	model.fit(X_train_dev, y_train_dev)
	return model
