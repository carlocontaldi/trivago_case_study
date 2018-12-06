import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random
plt.plasma()
sns.set()
sns.set_style('whitegrid')
mpl.rcParams['figure.figsize'] = (10.24, 7.68)
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def twentiles(series, title='', step=1, ax=None):
	"""Represent twentiles of input series. Implementation optimized for values distributing over diverse orders of magnitude"""
	bins = [round(e, 2) for e in np.linspace(0, 1, 21)]
	if ax==None:
		fig, ax = plt.subplots()
	quantiles = series.quantile(bins)
	(quantiles-min(quantiles)+1).apply(np.log).plot('bar', ax=ax)
	for patch, label in list(zip(ax.patches, quantiles.astype(int)))[::step]:
		height = patch.get_height()
		ax.text(patch.get_x() + patch.get_width()/2, height, label, ha='center', va='bottom')
	ax.set_title(''.join([title, ' - Twentiles']))
	ax.set_xlabel('Twentile')
	ax.set_ylabel('log(rescaledInput)')
	if ax is None:
		plt.show()

def last_quantiles(series, title='', step=1, ax=None, inf=0.95):
	"""Represent last quantiles of input series. Implementation optimized for values distributing over diverse orders of magnitude"""
	bins = [round(e, 4) for e in np.linspace(inf, 1, 21)]
	if ax is None:
		fig, ax = plt.subplots()
	quantiles = series.quantile(bins)
	(quantiles-min(quantiles)+1).apply(np.log).plot('bar', ax=ax)
	for patch, label in list(zip(ax.patches, quantiles.astype(int)))[::step]:
		height = patch.get_height()
		ax.text(patch.get_x() + patch.get_width()/2, height, label, ha='center', va='bottom')
	ax.set_title(''.join([title, ' - Last Quantiles']))
	ax.set_xlabel('Quantile')
	ax.set_ylabel('log(rescaledInput)')
	if ax is None:
		plt.show()

def compare_last_quantiles(series_a, series_b, sup_title, title_a, title_b, inf=0.95):
	fig, ax = plt.subplots(nrows=2, ncols=1)
	last_quantiles(series_a, title_a, ax=ax[0], inf=inf)
	ax[0].axes.get_xaxis().set_visible(False)
	last_quantiles(series_b, title_b, ax=ax[1], inf=inf)
	fig.suptitle(''.join([sup_title, ' - Last Quantile Comparison']))
	plt.show()

def kde(series, inf, sup, title, clip_quantile=None):
	"""Plot input series KDE."""
	fig, ax = plt.subplots()
	if clip_quantile is not None:
		sns.kdeplot(series, ax=ax, clip=(min(series), series.quantile(clip_quantile)))
	else:
		sns.kdeplot(series, ax=ax)
	ax.set_xlim(inf, sup)
	ax.set_title(''.join([title, ' - Kernel Density Estimation']))
	plt.show()

def box(series, title, inf, sup):
	"""Plot input series box plots."""
	fig, ax = plt.subplots(nrows=1, ncols=2)
	series.plot('box', ax=ax[0])
	series.plot('box', ax=ax[1])
	ax[1].set_ylim(inf, sup)
	fig.suptitle(''.join([title, ' - Box Plots']))
	plt.show()

def compare_twentiles(series_a, series_b, sup_title, title_a, title_b):
	fig, ax = plt.subplots(nrows=2, ncols=1)
	twentiles(series_a, title_a, ax=ax[0])
	ax[0].axes.get_xaxis().set_visible(False)
	twentiles(series_b, title_b, ax=ax[1])
	fig.suptitle(''.join([sup_title, ' - Twentile Comparison']))
	plt.show()

def sorted_cumsum_by_city_id(train_data):
	(train_data['city_id'].value_counts().cumsum()/len(train_data)).reset_index().drop(columns='index').plot(legend=False, title='Normalized Cumulative Sum of Hotels by Cities Ordered by Decreasing Hotel Frequency')
	vcs = train_data['city_id'].value_counts()
	sorted_unique_vcs = sorted(vcs.unique())
	fig, ax = plt.subplots()
	plt.plot(sorted_unique_vcs, np.cumsum([len(vcs[vcs==c]) for c in sorted_unique_vcs])/len(vcs))
	ax.set_xlim((0, 20))
	ax.set_title('Normalized Cumulative Sum of City Counts Ordered by Number of Hotels in them')
	plt.show()

def violin(x, y, df, inf=None, sup=None, ax=None):
	"""Plot input series violin plot."""
	if ax is None:
		fig, ax = plt.subplots()
	if inf is not None and sup is not None:
		ax.set_ylim(inf, sup)
		sns.violinplot(x=x, y=y, data=df[df[y]<=df[y].quantile([.99]).values[0]], ax=ax)
	else:
		sns.violinplot(x=x, y=y, data=df, ax=ax)
	ax.set_title(y)

def stars_compare_violins(df):
	mpl.rcParams['figure.figsize'] = (19.2, 10.80)
	fig, ax = plt.subplots(nrows=3, ncols=3)
	violin('stars', 'content_score', df, ax=ax[0, 0])
	violin('stars', 'n_images', df, -2, 10, ax=ax[0, 1])
	violin('stars', 'distance_to_center', df, -5e3, 1.5e4, ax=ax[0, 2])
	violin('stars', 'avg_rating', df, 50, 100, ax=ax[1, 0])
	violin('stars', 'n_reviews', df, -500, 3000, ax=ax[1, 1])
	violin('stars', 'avg_rank', df, -5, 30, ax=ax[1, 2])
	violin('stars', 'avg_price', df, -25, 400, ax=ax[2, 0])
	violin('stars', 'avg_saving_percent', df, -5, 45, ax=ax[2, 1])
	violin('stars', 'n_clicks', df, -10, 20, ax=ax[2, 2])
	for i in range(2):
		for j in range(3):
			ax[i, j].axes.get_xaxis().set_visible(False)
	mpl.rcParams['figure.figsize'] = (10.24, 7.68)
	plt.show()

def pairplots(df, testing=False, sz=10000):
	"""Plot input DataFrame pairplots."""
	vars=['content_score', 'n_images', 'distance_to_center', 'avg_rating', 'stars',  'n_reviews', 'avg_rank', 'avg_price', 'avg_saving_percent']
	df_tmp = df.sample(sz, random_state=0)
	sns.set(font_scale=0.7)
	if not testing:
		df_tmp['n_clicks'] = df_tmp['n_clicks'].apply(lambda x: np.ceil(np.log1p(x)) if x!=2 else 1)
		df_tmp['n_clicks'] = np.clip(df_tmp['n_clicks'], 0, 5)
		print('Assessing groups balance:', df_tmp['n_clicks'].groupby(df_tmp['n_clicks']).count(), sep='\n')
		pp = sns.pairplot(df_tmp, vars=vars, hue='n_clicks', palette='plasma', diag_kind='kde', plot_kws={'s': 6, 'linewidth': 0.3})
		handles = pp._legend_data.values()
		pp.fig.legend(handles=handles, labels=['0 clicks', '1-2 clicks', '3-7 clicks', '8-19 clicks', '20-53 clicks', '54+ clicks'], loc='lower right')
	else:
		pp = sns.pairplot(df_tmp, vars=['content_score', 'n_images', 'distance_to_center', 'avg_rating', 'stars', 'n_reviews', 'avg_rank', 'avg_price', 'avg_saving_percent'], diag_kind='kde', plot_kws={'s': 6, 'linewidth': 0.3})
	plt.show()
	sns.set(font_scale=1)

def bi_kde(series, y, title, ax=None, clip_quantile=None):
	"""Plot input series Bivariate KDE plot."""
	if ax is None:
		fig, ax = plt.subplots()
	if clip_quantile is not None:
		sns.kdeplot(series, y, ax=ax, clip=((min(series), series.quantile(clip_quantile)), (min(y), y.quantile(clip_quantile))), cmap='plasma')
	else:
		sns.kdeplot(series, y, ax=ax, cmap='plasma')
	if ax is None:
		ax.set_title(''.join([title, ' - Bivariate Kernel Density Estimation with log(1+n_clicks)']))
		plt.show()

def bi_kdes(df, testing=False, sz=10000, title='', stars_series=None):
	"""Plot input DataFrame Bivariate KDE plots."""
	if not testing:
		df_vis = df.sample(min(len(df), sz), random_state=0)
		y = np.log1p(df_vis['n_clicks'])
		fig, ax = plt.subplots(nrows=3, ncols=3)
		for i, col in enumerate(['content_score', 'n_images', 'distance_to_center', 'avg_rating', 'stars', 'n_reviews', 'avg_rank', 'avg_price', 'avg_saving_percent']):
			bi_kde(df_vis[col], y, col, ax[i//3, i%3])
		fig.suptitle(''.join(['Bivariate Kernel Density Estimation', title]))
		plt.subplots_adjust(hspace=0.4, wspace=0.3)
		plt.show()

def plot_finalized(df, test=False, sz=10000, cmap='plasma', clip_n_clicks=5):
	"""Apply dimensionality reduction to finalized data and plot it in a 3D space."""
	df = df.sample(sz, random_state=0)
	X = df.loc[:, ['content_score', 'n_images', 'distance_to_center', 'avg_rating', 'stars', 'n_reviews', 'avg_rank', 'avg_price', 'avg_saving_percent']]
	pca = PCA(n_components=3, random_state=0)
	pca_out = pca.fit_transform(X)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	if not test:
		y = df.loc[:, 'n_clicks'].values
		y = np.log(1 + df.loc[:, 'n_clicks'].values)
		ax.scatter(pca_out[:, 0], pca_out[:, 1], pca_out[:, 2], c=np.clip(y, 0, clip_n_clicks), cmap=cmap)
		normalize = mpl.colors.Normalize(vmin=0, vmax=clip_n_clicks)
		cax, _ = mpl.colorbar.make_axes(ax)
		cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
		cbar.set_label('n_clicks [clip(log(1+n_clicks), 0, 5)]')
		ax.set_title('Finalized Training Data - 3D Visualization')
	else:
		ax.scatter(pca_out[:, 0], pca_out[:, 1], pca_out[:, 2])
		ax.set_title('Finalized Test Data - 2D Visualization')
	ax.set_xlim((-200, 400))
	plt.show()

def eval_regression(y_true, y_pred, n_points=1000, cmap='plasma'):
	"""Plot regression results."""
	random.seed(0)
	np.random.seed(0)
	x = np.arange(0, n_points)
	idx_sample = random.sample(range(len(y_true)), n_points)
	s_y_true, s_y_pred = y_true[idx_sample], y_pred[idx_sample]
	idx_sorted = np.argsort(s_y_true)
	s_y_true, s_y_pred = s_y_true[idx_sorted], s_y_pred[idx_sorted]
	s_y_true, s_y_pred = np.log1p(s_y_true), np.log1p(s_y_pred)
	mse = (s_y_pred-s_y_true)**2
	ax_pred = sns.scatterplot(x=x, y=s_y_pred, hue=mse, palette=cmap, s=16, linewidth=0)
	ax_true = sns.scatterplot(x=x, y=s_y_true, s=12, color='black', linewidth=0)
	ax_pred.set_xlabel('sample')
	ax_pred.set_ylabel('log(1+n_clicks)')
	pred_leg = mpl.lines.Line2D([0], [0], marker='o', color='white', label='predicted n_clicks', markeredgecolor='black', markersize=8)
	true_leg = mpl.lines.Line2D([0], [0], marker='o', color='white', label='true n_clicks', markerfacecolor='black', markersize=8)
	plt.legend(handles=[true_leg, pred_leg])
	ylim = ax_pred.get_ylim()
	ax2=ax_pred.twinx()
	ax2.set_ylabel('WMSE sample weight')
	ax2.set_ylim(ylim[0]+1, ylim[1]+1)
	normalize = mpl.colors.Normalize(vmin=min(mse), vmax=max(mse))
	cax, _ = mpl.colorbar.make_axes(ax_pred)
	cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
	cbar.set_label('MSE(true_n_clicks, pred_n_clicks)')
	ax_pred.set_title('Regression Results')
	plt.show()
