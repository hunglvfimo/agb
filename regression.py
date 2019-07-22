import os

import numpy as np
import pandas as pd
import argparse

import matplotlib.pyplot as plt 

from scipy import stats, sqrt 
from sklearn.metrics import mean_squared_error

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 

from sklearn.svm import SVR 
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.model_selection import  GridSearchCV

import xgboost
from xgboost import plot_importance

import statsmodels.api as sm
import statsmodels.formula.api as smf

from collections import OrderedDict

def fit_pca(data, ratio=0.9):
	# finding minimum n_components parameter to explain >= ratio% variance
	pca 	= PCA()
	pca.fit(data)
	
	variance = 0.0
	for i in range(len(pca.explained_variance_ratio_)):
		variance += pca.explained_variance_ratio_[i]
		if variance >= ratio:
			break
	print("Number of PCA components: %d" % i)
	pca 	= PCA(n_components=i)
	return pca
	
def fit_svr(X_train, y_train):
	params 	=  [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
				{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
	svr 	= SVR(gamma='scale')
	clf 	= GridSearchCV(svr, params, cv=10, refit=True, scoring='neg_mean_squared_error')
	clf.fit(X_train, y_train)
	print("Best parameters set found on development set:")
	print()
	print(clf.best_params_)

	return clf

def fit_linear(X_train, y_train):
	clf = LinearRegression().fit(X_train, y_train)
	return clf

def find_important_features(X_train, y_train):
	clf = xgboost.XGBRegressor(colsample_bytree=0.4,
								gamma=0,
								learning_rate=0.07,
								max_depth=3,
								min_child_weight=1.5,
								n_estimators=1000,
								reg_alpha=0.75,
								reg_lambda=0.45,
								subsample=0.6,
								seed=28)
	clf.fit(X_train, y_train)
	important_ft_inds = np.argsort(clf.feature_importances_)
	
	return important_ft_inds

def fit_xgb(X_train, y_train):
	parameters = {
		'colsample_bytree': [0.4, 0.6, 0.8],
		'gamma': 			[0, 0.03, 0.1, 0.3],
		'min_child_weight': [1.5, 6, 10],
		'learning_rate': 	[0.1, 0.07],
		'max_depth':		[3, 5],
		'n_estimators': 	[10, 50, 100],
		'reg_alpha': 		[1e-5, 1e-2, 0.75],
		'reg_lambda': 		[1e-5, 1e-2, 0.45],
		'subsample': 		[0.6, 0.95]  
	}

	# best params
	# parameters = {
	# 		'colsample_bytree'	: [0.4], 
	# 		'gamma'				: [0], 
	# 		'learning_rate'		: [0.07], 
	# 		'max_depth'			: [3], 
	# 		'min_child_weight'	: [10], 
	# 		'n_estimators'		: [100], 
	# 		'reg_alpha'			: [1e-05], 
	# 		'reg_lambda'		: [0.01], 
	# 		'subsample'			: [0.6]
	# }

	xgb = xgboost.XGBRegressor(objective='reg:squarederror', seed=28)
	clf = GridSearchCV(estimator=xgb, param_grid=parameters, refit=True, cv=5, scoring='neg_mean_squared_error')
	clf.fit(X_train, y_train)
	print('best params')
	print(clf.best_params_)

	return clf

def fit_mixedlm(X_train, y_train):
	data = sm.datasets.get_rdataset("dietox", "geepack").data
	print(data.columns)

def predict(X_train, y_train, X_test, y_test, model_name='linear', train_set="train", plot=False, parser=None):
	# norm
	scaler 	= StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test 	= scaler.transform(X_test)

	pca 	= fit_pca(X_train)
	X_train = pca.fit_transform(X_train)
	X_test 	= pca.transform(X_test)

	# training
	if model_name == 'svr':
		clf = fit_svr(X_train, y_train)
	elif model_name == 'xgb':
		clf = fit_xgb(X_train, y_train)
	elif model_name == "mixed":
		clf = fit_mixedlm(X_train, y_train)
	else:
		clf = fit_linear(X_train, y_train)

	# train performance
	train_rmse 		= sqrt(mean_squared_error(y_train, clf.predict(X_train)))
	print('Train RMSE:', train_rmse)

	# test performance
	y_pred 			= clf.predict(X_test)
	pred_rmse 		= sqrt(mean_squared_error(y_test, y_pred))
	print('Prediction RMSE:', 	pred_rmse)

	if plot:
		samples = np.arange(len(y_test)) 
		plt.plot(samples, y_test, linewidth=1, ls='-', color='#c92508', label='y_true')
		plt.plot(samples, y_pred, linewidth=1, ls='--', color='#2348ff', label='y_pred')
		plt.suptitle(model_name)
		plt.title("Train RMSE: %.3f, Test RMSE: %.3f" % (train_rmse, pred_rmse))
		plt.legend(fontsize=12)
		plt.savefig(os.path.join(parser.log_dir, "%s_%s.png" % (model_name, train_set)))

if __name__ == '__main__':
	parser     = argparse.ArgumentParser()
	parser.add_argument('--model_name', default="linear", help='Regression model')
	parser.add_argument('--data_dir', default="data", help='')
	parser.add_argument('--log_dir', default="logs", help='')
	parser.add_argument('--train_csv', default="train.csv", help='')
	parser.add_argument('--test_csv', default="test.csv", help='')
	parser = parser.parse_args()

	# reading data
	train_df 	= pd.read_csv(os.path.join(parser.data_dir, parser.train_csv))
	test_df 	=  pd.read_csv(os.path.join(parser.data_dir, parser.test_csv))

	feature_cols 	= list(train_df.columns)[1:]
	# feature_cols 	= [name for name in list(train_df.columns) if name.endswith('5')]

	target_col 		= "AGB_Mean"
	print("Number of features:", len(feature_cols))

	X_train = train_df[feature_cols] 
	print("Number of samples:", X_train.shape[0])
	y_train = train_df[target_col] 

	X_test 	= test_df[feature_cols] 
	y_test 	= test_df[target_col] 

	predict(X_train, y_train, X_test, y_test, model_name=parser.model_name, train_set=parser.train_csv.split(".")[0], plot=True, parser=parser)