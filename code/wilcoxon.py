from scipy.stats import mannwhitneyu

def wilcoxon(a, b):
	"""Get p-value describing the statistical significance of the comparison between input series based on the Mann-Whitney (Wilcoxon) test."""
	u, prob = mannwhitneyu(a, b)
	print(prob)
	return prob

def compute_wilcoxon(a_path, b_path):
	import pickle
	with open(a_path, 'rb') as f:
		a=pickle.load(f)
	with open(b_path, 'rb') as f:
		b=pickle.load(f)
	wilcoxon(a, b)
