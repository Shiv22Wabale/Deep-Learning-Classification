from config import cfg
#from read_input import Read_input

from shallow_cnn_celeba import CNN_CelebA # comment/uncomment to run shallow cnn
#from cnn_celeba_arch3 import CNN_CelebA # comment/uncomment to run costom cnn


import tensorflow as tf
import numpy as np

def main(_) :
	cnn_object = CNN_CelebA(cfg.path_check_point, cfg.summaries_dir)
	if cfg.train_model == True:
		read_object = cnn_object.train_model(cfg.label_file, cfg.image_directory, cfg.batch_size, cfg.num_epochs, cfg.shuffle, cfg.test_size, cfg.distortion_range)
		print("Training Completed ......... ! ")
	else :
		print("Prediction started .....")
		read_object = cnn_object.test_model(cfg.label_file, cfg.image_directory, cfg.batch_size, cfg.num_epochs, cfg.shuffle, cfg.test_size, cfg.distortion_range)

if __name__ == "__main__":
    tf.app.run()
