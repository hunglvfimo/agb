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

PATCH_SIZES = [1, 2, 3]
GLCM_PROPS 	= ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']

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

if __name__ == '__main__':
	parser     	= argparse.ArgumentParser()
	parser.add_argument('--shape_file', default="AGB_ground_truth.shp", help='Path to ground truth shapefile')
	parser.add_argument('--image_dir', default=".", help='Path to images dir')
	parser 		= parser.parse_args()

	shape 		= fiona.open(parser.shape_file)
	ft_table 	= None
	targets 	= []
	headers 	= []

	first_file 	= True
	for filepath in glob.glob(os.path.join(parser.image_dir, "*.tif")):
		basename 	= os.path.basename(filepath).split(".")[0]
		print(basename)

		image 		= geoio.GeoImage(filepath)
		image_data 	= image.get_data()
		image_data 	= image_data[0, ...]

		data_file	= None
		first_point = True
		for point in shape:
			properties 	= point["properties"]
			geometry 	= point["geometry"]

			agb 		= properties['AGB_Mean']
			loc 		= geometry['coordinates']
			
			x, y 		= image.proj_to_raster(loc[0], loc[1])
			x, y 		= int(x), int(y)
			if x >= 0 and y >= 0 and x < image_data.shape[1] and y < image_data.shape[0]:
				if first_file:
					targets.append(agb)

				data_point 	= []
				# get value of this point
				if first_point:
					headers.append(basename)

				data_point.append(image_data[y, x])
				# get feature around this point
				for patch_size in PATCH_SIZES:
					left_x 		= x - patch_size if x - patch_size >= 0 else 0
					right_x 	= x + patch_size + 1 if x + patch_size + 1 < image_data.shape[1] else image_data.shape[1]

					bot_y 		= y - patch_size if y - patch_size >= 0 else 0
					top_y 		= y + patch_size + 1 if y + patch_size + 1 < image_data.shape[0] else image_data.shape[0]

					patch 		= image_data[bot_y: top_y, left_x: right_x]
					# calculate first-order stats
					if first_point:
						headers.append("%s_mean_%d" % (basename, patch_size))
						headers.append("%s_std_%d" % (basename, patch_size))
						headers.append("%s_skew_%d" % (basename, patch_size))
						headers.append("%s_kurtosis_%d" % (basename, patch_size))
					
					data_point.append(np.mean(patch))
					data_point.append(np.std(patch))
					data_point.append(skew(patch.reshape(-1)))
					data_point.append(kurtosis(patch.reshape(-1)))
					
					# calculate second-order stats
					glcm = greycomatrix(exband_histgram(patch), [1], [i * np.pi / 8 for i in range(8)])
					for prop in GLCM_PROPS:
						if first_point:
							headers.append("%s_%s_%d" % (basename, prop, patch_size))
						
						data_point.append(greycoprops(glcm, prop)[0, 0])
				data_point = np.array(data_point)

				if data_file is None:
					data_file = data_point
				else:
					data_file = np.vstack((data_file, data_point))

				first_point = False
		
		if ft_table is None:
			ft_table = data_file
		else:
			ft_table = np.hstack((ft_table, data_file))

		first_file = False

	targets 	= np.array(targets)
	targets 	= np.expand_dims(targets, axis=0)
	targets 	= np.transpose(targets)

	data_table 	= np.hstack((targets, ft_table))
	headers 	= ["AGB_Mean"] + headers

	df = pd.DataFrame(data=data_table, index=None, columns=headers)
	df.to_csv(os.path.join(DATA_DIR, "data.csv"), index=False)
				

