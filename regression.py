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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import  GridSearchCV

from params import *

def fit_svr(X_train, y_train):
	params 	=  [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
				{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
	svr 	= SVR(gamma='scale')
	clf 	= GridSearchCV(svr, params, cv=10, refit=True, scoring='r2')
	clf.fit(X_train, y_train)
	print("Best parameters set found on development set:")
	print()
	print(clf.best_params_)
	print()
	print("Grid scores on development set:")
	print()
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	for mean, std, params in zip(means, stds, clf.cv_results_['params']):
		print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

	print()
	print("Detailed report:")
	print()
	print("The model is trained on the full development set.")
	print("The scores are computed on the full evaluation set.")
	print()
	return clf

def fit_linear(X_train, y_train):
	clf = LinearRegression().fit(X_train, y_train)
	return clf

def predict(X_train, y_train, X_test, y_test, model_name='linear', train_set="train", plot=False):
	# norm
	scaler 	= StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test 	= scaler.transform(X_test)

	# feature selection
	pca 	= PCA(n_components=20)
	X_train = pca.fit_transform(X_train)
	X_test 	= pca.transform(X_test)

	# training
	if model_name == 'svr':
		clf = fit_svr(X_train, y_train)
	else:
		clf = fit_linear(X_train, y_train)
	
	# model performance
	train_r2 	= clf.score(X_train, y_train)
	print('Train R^2:', 		train_r2)
	pred_r2  	= clf.score(X_test, y_test)
	print('Prediction R^2:', 	pred_r2)
	# prediction
	y_true, y_pred = y_test, clf.predict(X_test)
	pred_rmse = sqrt(mean_squared_error(y_pred, y_true))
	print('Prediction RMSE:', 	pred_rmse)

	if plot:
		samples = np.arange(len(y_true))
		plt.plot(samples, y_true, linewidth=2, ls='--', color='#c92508', label='y_true')
		plt.plot(samples, y_pred, linewidth=2, ls='-', color='#2348ff', label='y_pred')
		plt.suptitle(model_name)
		plt.title("R^2: %.3f, RMSE: %.3f" % (pred_r2, pred_rmse))
		plt.legend(fontsize=12)
		plt.savefig(os.path.join(LOG_DIR, "%s_%s.png" % (model_name, train_set)))

if __name__ == '__main__':
	parser     = argparse.ArgumentParser()
	parser.add_argument('--model_name', default="linear", help='Regression model')
	parser.add_argument('--train_csv', default="train.csv", help='')
	parser.add_argument('--test_csv', default="test.csv", help='')
	parser = parser.parse_args()

	# reading data
	train_df 	= pd.read_csv(os.path.join(DATA_DIR, parser.train_csv))
	test_df 	=  pd.read_csv(os.path.join(DATA_DIR, parser.test_csv))

	feature_cols 	= list(train_df.columns)[1:]
	target_col 		= "AGB_Mean"
	print("Number of features:", len(feature_cols))

	X_train = train_df[feature_cols] 
	y_train = train_df[target_col] 

	X_test 	= test_df[feature_cols] 
	y_test 	= test_df[target_col] 

	predict(X_train, y_train, X_test, y_test, model_name=parser.model_name, train_set=parser.train_csv.split(".")[0], plot=True)