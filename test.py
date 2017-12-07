from config import cfg
from read_input import Read_input

import tensorflow as tf
import numpy as np

def main(_) :
	read_object = Read_input(cfg.label_file, cfg.image_directory, cfg.batch_size, cfg.num_epochs, cfg.shuffle, cfg.test_size, cfg.distortion_range)
	#print(read_object.images.shape)
	#print(read_object.labels.shape)
	#print([ i if read_object.output[i, 1] == 1 else 0 for i in range(read_object.output.shape[0])])
	#for i in range(400):
	#	if read_object.labels[i] == 1:
	#		print(read_object.images[i])
	#print(read_object.output[143:146, 1])
	print(read_object.image_batch)
	print(read_object.label_batch)
	print(read_object.image_test)
	print(read_object.label_test)

if __name__ == "__main__":
    tf.app.run()
