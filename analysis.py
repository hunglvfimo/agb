import os

import numpy as np
import pandas as pd
import argparse

import matplotlib.pyplot as plt 

from params import *

def show_histogram(data):
	samples = np.arange(len(data))
	plt.hist(data, color='#c92508')
	plt.xlabel('Values', fontsize=16)
	plt.ylabel('#samples', fontsize=16, color='black')
	plt.legend(fontsize=12)
	plt.show()

if __name__ == '__main__':
	parser     = argparse.ArgumentParser()
	parser.add_argument('--dataset', default="data.csv", help='Regression model')
	parser = parser.parse_args()

	# reading data
	df = pd.read_csv(os.path.join(DATA_DIR, parser.dataset))

	feature_cols 	= list(df.columns)[:-1]
	target_col 		= "AGB_Mean"

	X = df[feature_cols] 
	y = df[target_col] 

	show_histogram(y)