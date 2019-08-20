import numpy as np
import os
import argparse
import glob
from shutil import copyfile

from sklearn.model_selection import train_test_split

def ensure_dir(path):
	if not os.path.isdir(path):
		os.mkdir(path)

if __name__ == '__main__':
	parser     	= argparse.ArgumentParser()
	parser.add_argument('--image_dir', default=".", help='Path to images dir')
	parser.add_argument('--save_dir', default=".", help='Path to images dir')
	parser.add_argument('--rng', type=int, default=28)
	parser 		= parser.parse_args()

	ensure_dir(os.path.join(parser.save_dir, "train_%d" % parser.rng))
	ensure_dir(os.path.join(parser.save_dir, "test_%d" % parser.rng))

	for label in os.listdir(parser.image_dir):
		ensure_dir(os.path.join(parser.save_dir, "train_%d" % parser.rng, label))
		ensure_dir(os.path.join(parser.save_dir, "test_%d" % parser.rng,  label))

		filepaths 		= glob.glob(os.path.join(parser.image_dir, label, "*.tif"))
		X_train, X_test = train_test_split(filepaths, test_size=0.33, random_state=parser.rng)

		for filepath in X_train:
			target_path = os.path.join(parser.save_dir, "train_%d" % parser.rng, label, os.path.basename(filepath))
			copyfile(filepath, target_path)
		
		for filepath in X_test:
			target_path = os.path.join(parser.save_dir, "test_%d" % parser.rng, label, os.path.basename(filepath))
			copyfile(filepath, target_path)
