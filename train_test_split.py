import numpy as np
import os
import argparse
import glob

import cv2
import tifffile as tiff
from sklearn.model_selection import train_test_split
import csv

def ensure_dir(path):
	if not os.path.isdir(path):
		os.mkdir(path)

if __name__ == '__main__':
	parser     	= argparse.ArgumentParser()
	parser.add_argument('--ground_truth', default="data.tif", help='Path to ground truth file')
	parser.add_argument('--rng', type=int, default=28)
	parser 		= parser.parse_args()

	base_dir 	= os.path.dirname(parser.ground_truth)

	cnt_dict 	= dict()

	gt_image 	= tiff.imread(parser.ground_truth)

	binary_image 				= np.zeros(gt_image.shape, dtype=np.uint8)
	binary_image[gt_image > 0] 	= 255
	_, contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		# compute the center of the contour
		cx, cy 	= cnt[0, 0]

		# label of the contour
		label 	= gt_image[cy, cx]
		if label not in cnt_dict.keys():
			cnt_dict[label] = []
		cnt_dict[label].append(cnt)

	train_image = np.zeros(gt_image.shape)
	test_image 	= np.zeros(gt_image.shape)
	for k in cnt_dict.keys():
		cnt_train, cnt_test = train_test_split(cnt_dict[k], test_size=0.3, random_state=parser.rng)

		cv2.drawContours(train_image, 	cnt_train, 	-1, k, 1)
		cv2.drawContours(test_image, 	cnt_test, 	-1, k, 1)

	tiff.imsave(os.path.join(base_dir, "train.tif"), train_image, planarconfig='contig')
	tiff.imsave(os.path.join(base_dir, "test.tif"), test_image,  planarconfig='contig')