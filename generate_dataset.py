import os
import glob

import numpy as np
import pandas as pd
import argparse

from scipy.stats import kurtosis, skew
from skimage.feature import greycomatrix, greycoprops

import geoio
import tifffile as tiff
import fiona

from params import *
from tqdm import tqdm

PATCH_SIZES = [4]
GLCM_PROPS 	= ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']

def ensure_dir(path):
	if not os.path.isdir(path):
		os.mkdir(path)

def exband_histgram(src_matrix):
	# replace inf and nan with 0
	src_matrix = np.vectorize(lambda x: x if (np.isfinite(x) and not np.isnan(x)) else 0)(src_matrix)

	min_value = src_matrix.min()
	max_value = src_matrix.max()
	if min_value == max_value:
		return np.zeros(src_matrix.shape, dtype=np.uint8)

	min_result = 0
	max_result = 255

	# convert from min_result to max value
	grad = (max_result - min_result) / (max_value - min_value)
	intercept = min_result - min_value * grad
	src_matrix = src_matrix * grad + intercept

	# convert standard deviation
	src_matrix = (src_matrix - src_matrix.mean()) / src_matrix.std() * 50 + src_matrix.mean()

	src_matrix[src_matrix < min_result] = min_result
	src_matrix[src_matrix > max_result] = max_result
	np.vectorize(lambda x: x if (np.isfinite(x) and not np.isnan(x)) else 0)(src_matrix)

	return src_matrix.astype(np.uint8)

def generate_patches(gts, labels, train_test, filepath, save_dir, patch_size=4):
	for label in labels:
		ensure_dir(os.path.join(save_dir, "train", str(label)))
		ensure_dir(os.path.join(save_dir, "test", str(label)))

	image 		= geoio.GeoImage(filepath)
	image_data 	= image.get_data()

	for loc, label, istrain in tqdm(zip(gts, labels, train_test)):
		loc_y, loc_x 	= loc
		x, y 			= image.proj_to_raster(loc_x, loc_y)
		x, y 			= int(x), int(y)

		if x >= patch_size and y >= patch_size and x < image_data.shape[2] - patch_size and y < image_data.shape[1] - patch_size:
			left_x 		= x - patch_size
			right_x 	= x + patch_size + 1

			bot_y 		= y - patch_size
			top_y 		= y + patch_size + 1
			
			patch 		= image_data[:, bot_y: top_y, left_x: right_x]
			patch 		= np.swapaxes(patch, 0, 2)

			if istrain:
				tiff.imsave(os.path.join(save_dir, "train", str(label), "%d_%d.tif" % (y, x)), patch, planarconfig='contig')
			else:
				tiff.imsave(os.path.join(save_dir, "test", str(label), "%d_%d.tif" % (y, x)), patch, planarconfig='contig')

def generate_features(gts, targets, filepath):
	ft_table 	= []
	targets 	= []

	patch_size 	= 1
	image 		= geoio.GeoImage(filepath)
	image_data 	= image.get_data()
	for loc, label in tqdm(zip(gts, labels)):
		x, y 		= image.proj_to_raster(loc[0], loc[1])
		x, y 		= int(x), int(y)

		if x >= patch_size and y >= patch_size and x < image_data.shape[2] - patch_size and y < image_data.shape[1] - patch_size:
			data_point 	= []

			left_x 		= x - patch_size
			right_x 	= x + patch_size + 1

			bot_y 		= y - patch_size
			top_y 		= y + patch_size + 1

			patch 		= image_data[:, bot_y: top_y, left_x: right_x]
			patch 		= np.swapaxes(patch, 0, 2)
					
			data_point.append(np.mean(patch))
			data_point.append(np.std(patch))
			data_point.append(skew(patch.reshape(-1)))
			data_point.append(kurtosis(patch.reshape(-1)))
					
			# calculate second-order stats
			glcm = greycomatrix(exband_histgram(patch), [1], [i * np.pi / 8 for i in range(8)])
			for prop in GLCM_PROPS:	
				data_point.append(greycoprops(glcm, prop)[0, 0])

			ft_table.append(data_point)
			targets.append(label)

	targets 	= np.array(targets)
	ft_table 	= np.array(ft_table)
	data_table 	= np.hstack((targets, ft_table))
	headers 	= ["Target"] + ['b_%d' % i for i in range(ft_table.shape[1])]

	df = pd.DataFrame(data=data_table, index=None, columns=headers)
	return df

def raster_to_gt_points(gt_path, train_mask_path=None):
	gt_img 		= geoio.GeoImage(gt_path)
	gt_data 	= gt_img.get_data()
	gt_data 	= gt_data[0, ...]

	if train_mask_path is not None:
		train_mask 	= tiff.imread(train_mask_path)

	print(gt_data.shape, train_mask.shape)
	
	ys, xs 		= np.where(gt_data > 0)

	gts 		= []
	targets 	= []
	isTrain 	= []
	for x, y in zip(xs, ys):
		loc_x, loc_y = gt_img.raster_to_proj(x, y)

		gts.append((loc_y, loc_x))
		targets.append(int(gt_data[y, x]))
		
		if train_mask_path is not None:
			isTrain.append(train_mask[y, x] > 0)
		else:
			# every points are test points
			isTrain.append(False)
	
	return gts, targets, isTrain

def shp_to_gt_points(shape_file, train_mask_path=None):
	shape 	= fiona.open(shape_file)

	if train_mask_path is not None:
		train_mask 	= tiff.imread(train_mask_path)

	gts 	= []
	targets = []
	isTrain = []
	for point in shape:
		properties 	= point["properties"]
		geometry 	= point["geometry"]

		val 		= properties['Type_EN']
		val 		= val.split(" ")[-1]
		val 		= val.split("_")
		val.sort()
		val 		= "".join(val)

		loc 		= geometry['coordinates']

		gts.append(loc)
		targets.append(val)

		if train_mask_path is not None:
			pass
		else:
			# every points are test points
			isTrain.append(False)

	return gts, targets, isTrain

if __name__ == '__main__':
	parser     	= argparse.ArgumentParser()
	parser.add_argument('--patch-size',     default=4, 	type=int,	help='batch size')
	parser.add_argument('--ground-truth', 	default="data.tif", 	help='Path to ground truth file')
	parser.add_argument('--train-mask', 	default=None, 	help='Path to train mask file')
	parser.add_argument('--image-path', 	default=".", 			help='Path to images dir')
	parser.add_argument('--save-dir', 		default=".", 			help='Path to images dir')

	
	parser 		= parser.parse_args()
	
	ensure_dir(parser.save_dir)
	ensure_dir(os.path.join(parser.save_dir, "train"))
	ensure_dir(os.path.join(parser.save_dir, "test"))

	ext 	= os.path.basename(parser.ground_truth).split(".")[-1]
	func 	= None
	if ext in ["shp", "SHP"]:
		func = shp_to_gt_points
	elif ext in ["tif", "TIF", "tiff", "TIFF"]:
		func = raster_to_gt_points
	else:
		RaiseValueError("Ground truth file is not supported!")

	gts, targets, train_test 	= func(parser.ground_truth, parser.train_mask)
	print(len(gts), len(targets))

	generate_patches(gts, targets, train_test, parser.image_path, parser.save_dir, patch_size=parser.patch_size)