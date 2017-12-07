import tensorflow as tf

flags = tf.app.flags


flags.DEFINE_string('label_file', './list_attr_celeba.txt', 'The path to label file')
flags.DEFINE_string('image_directory', '/tmp/ShivrajML/images/img_align_celeba/', 'The path to the base directory of the image folder')
flags.DEFINE_string('path_check_point', './check_point/model.ckpt', 'The file path with file name to save the check points')
flags.DEFINE_string('summaries_dir', '/home/ubuntu/ML3/log_arch', 'For tensorflow dashboard')

flags.DEFINE_integer('batch_size', 180, 'Batch size')
flags.DEFINE_integer('num_epochs', 200, 'Number of epochs')
flags.DEFINE_boolean('shuffle', True, 'Shuffle the input data')
flags.DEFINE_boolean('train_model', False, 'To train the model')
flags.DEFINE_integer('test_size', 1024, 'To test the model')

flags.DEFINE_integer('distortion_range', 2, 'Range of images to be distorted for augmentation')
flags.DEFINE_boolean('augmentation', False, 'True for image augmentation, False without image augmentation training')


cfg = tf.app.flags.FLAGS
