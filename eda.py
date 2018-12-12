import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
import os
import glob

from scipy.misc import imresize
from keras.utils import np_utils
from random import shuffle
from sklearn.utils import resample
import shutil
from numpy.random import choice

def train_val_holdout(root_path, final_path, df, train_ratio = 0.7, validation_ratio = 0.2, holdout_ratio = 0.1, resize_size = (299,299)):
	train_folder = final_path + '/train'
	validation_folder = final_path + '/validation'
	holdout_folder = final_path + '/holdout'

	for root, dirs, files in os.walk(root_path, topdown=False):
		i = 0

		for name in files:
			ext = name.split('.')[-1]
			galaxy_id = name.split('.')[0]
			if ext in ['jpg','png']:
				# current_path = os.path.join(root, name)
                # root_dir, category = os.path.split(root)
				current_path = os.path.join(root,name)
				import pdb; pdb.set_trace()
				category = df.query('GalaxyID == @galaxy_id')['labels'].values[0]
				pdb.set_trace()
				val_split_dir = choice([train_folder, validation_folder, holdout_folder],
				1, p =[train_ratio, validation_ratio, holdout_ratio])[0]


				new_dir = os.path.join(val_split_dir, category)
				if not os.path.exists(new_dir):
					os.makedirs(new_dir)

				new_path = os.path.join(new_dir, name)
				o_img = cv2.imread(current_path)
				new_img = cv2.resize(o_img, resize_size)
				cv2.imwrite(new_path,new_img)
				print(new_path,i)
				i += 1


def read_images(paths):

	lst = glob.glob(paths + '*.jpg')
	images = [cv2.imread(file) for file in lst]
	names = [os.path.split(file)[1] for file in lst]
	id_num = [int(file.split('.')[0]) for file in names]

	return images, id_num

def targets(df, p = 0.5):

	'''
	input: df: dataframe
			p: percentage of votes for question; certainty
	output: dataframe with 'label' column

	'''

	conditions = [
		(df['Class1.2'] >= p) & (df['Class3.2'] >= p),
		(df['Class2.1'] >= p) & (df['Class6.2'] >= p),
		(df['Class1.2'] >= p) & (df['Class3.1'] >= p),
		(df['Class1.1'] >= p) & (df['Class6.2'] >= p),
		(df['Class6.1'] >= p) & (df['Class8.4'] >= p),
		(df['Class1.3'] >= p)]
	categories = ['spiral', 'edge_view_spiral', 'barred_spiral', 'elliptical', 'irregular', 'star']
	df['labels'] = np.select(conditions, categories, default='other')

	return df

def remove_dir(paths):
	for i in range(len(paths)):
		try:
		    shutil.rmtree(paths[i])
		except OSError as e:  ## if failed, report it back to the user ##
		    print ("Error: %s - %s." % (e.filename, e.strerror))

def process_images(images, size = 60):
    """
    Import image at 'paths', center and crop to size
    """

    count = len(images)
    arr = np.zeros(shape=(count,size,size,3))
    for i in range(count):
        img = images[i]
        img = img.T[:,106:106*3,106:106*3] #crop 424x424x3 to 212x212x3
        img = imresize(img,size=(size,size,3),interp="cubic") # shrink size to make easier to compute
        arr[i] = img

    return arr.astype(int)
