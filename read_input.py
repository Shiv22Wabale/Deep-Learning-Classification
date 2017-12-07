import pandas as pd
import numpy as np
import tensorflow as tf
import random
from config import cfg

class Read_input(object):
	def __init__(self, path, image_dir, batch_size, num_epochs, shuffle = True, test_size = 10, distortion_range = 1000) :
		self.distortion_range = distortion_range
		self.image_batch, self.label_batch, self.image_test, self.label_test = self.read_image_labels(path, image_dir, batch_size, num_epochs, shuffle, test_size)
		#self.images, self.labels = self.create_features_list(path, image_dir)

	def dataParser( self, path ) :
		data = pd.read_csv(path, sep = '\s*,\s*', header = None, engine='python')
		x = [data[0][i].split() for i in range(len(data) -1)]
		y = np.asarray(x[2:len(x)-1])
		index = 16
		images = y[:, 0]
		a = np.array([images])
		labels = y[:, index]
		b = np.array([labels])
		c = np.concatenate((a.T, b.T), axis = 1)

		return c

	def create_features_list(self, path, image_dir):
		c = self.dataParser(path)
		labels_list = [[1, 0] if l == '1' else [0, 1] for l in c[:,1]]
		images_list = [image_dir + x for x in c[:,0]]
		return images_list, labels_list

	def create_features_list_train_test(self, path, image_dir):
		images_list, labels_list = self.create_features_list(path, image_dir);

		r = random.random()
		random.shuffle(images_list, lambda : r)
		random.shuffle(labels_list, lambda : r)
		length = len(images_list) // 90
		return images_list[:length], labels_list[:length], images_list[length:], labels_list[length:]
	
	def generate_augmentated_image(self, example):
		if random.randrange(self.distortion_range) != 0 :
			return example
		else :
			distorted_image = tf.random_crop(example, [28, 28, 3])

			distorted_image = tf.image.random_flip_left_right(distorted_image)

			distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
			distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
			return distorted_image

	def read_images_from_disk(self, images_list, labels_list, num_epochs, shuffle, train):
		from tensorflow.python.framework import ops
		from tensorflow.python.framework import dtypes
	
		images = ops.convert_to_tensor(images_list, dtype=dtypes.string)
		labels = ops.convert_to_tensor(labels_list, dtype=dtypes.uint8)

		input_queue = tf.train.slice_input_producer([images, labels], num_epochs=num_epochs, shuffle=shuffle)
		label_ = input_queue[1]

		file_contents = tf.read_file(input_queue[0])
		example = tf.image.decode_jpeg(file_contents, channels=3)
		if tf.shape(example)[2] == 1 & train:
			example = tf.image.rgb_to_grayscale(example)

		if  cfg.augmentation:
			example = self.generate_augmentated_image(example)

		images_ = tf.image.per_image_standardization(tf.image.resize_images(example, [224, 224]))
		

		return images_, label_

	def read_image_labels(self, path, image_dir, batch_size, num_epochs, shuffle, test_size):

		images_list, labels_list, test_images_list, test_labels_list = self.create_features_list_train_test(path, image_dir);

		train_image, train_label = self.read_images_from_disk(images_list, labels_list, num_epochs, shuffle, True)

		image_batch, label_batch = tf.train.batch([train_image, train_label], batch_size=batch_size)

		test_image, test_label = self.read_images_from_disk(test_images_list, test_labels_list, 100000000, shuffle, False)

		image_test, label_test = tf.train.batch([test_image, test_label], batch_size=test_size)
 

		return image_batch, label_batch, image_test, label_test
