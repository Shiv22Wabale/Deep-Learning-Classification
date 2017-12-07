from read_input import Read_input

import tensorflow as tf

class CNN_CelebA(object) :

	def __init__(self, path_check_point, summaries_dir) :
		self.path_check_point = path_check_point
		self.summaries_dir = summaries_dir

	def weight_variable(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(self, shape):
		initial = tf.constant(0.2, shape=shape)
		return tf.Variable(initial)

	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	def define_cnn(self, x) :
		#self.saver = tf.train.Saver() # to save the model
		
		# Reshape the image
		x_image = tf.reshape(x, [-1, 64, 64, 3])

		#--------1
		W_conv1 = self.weight_variable([5, 5, 3, 32])
		b_conv1 = self.bias_variable([32])

		h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1) # 32 * 32 * 32
		#--------1

		#---------2
		W_conv2 = self.weight_variable([5, 5, 32, 64])
		b_conv2 = self.bias_variable([64])

		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = self.max_pool_2x2(h_conv2) # 16 * 16 * 64 
		#---------2

		#---------3
		W_conv3 = self.weight_variable([7, 7, 64, 128])
		b_conv3 = self.bias_variable([128])

		h_conv3 = tf.nn.relu(self.conv2d(h_pool2, W_conv3) + b_conv3)
		h_pool3 = self.max_pool_2x2(h_conv3) # 8 * 8 * 128 --- 
		#---------3

		#---------4 fc
		W_fc1 = self.weight_variable([8 * 8 * 128, 1024])
		b_fc1 = self.bias_variable([1024])

		h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 128])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
		#--------4 fc

		#--------5 fc last
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		W_fc2 = self.weight_variable([1024, 2])
		b_fc2 = self.bias_variable([2])

		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
		#-------5 fc last

		self.saver = tf.train.Saver() # to save the model

		return y_conv, keep_prob

	def loss(self) :

		x = tf.placeholder(tf.float32, [None, 64, 64, 3])
		y_ = tf.placeholder(tf.float32, [None, 2])

		y_conv, keep_prob = self.define_cnn(x)

		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_conv, labels = y_))
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
		tf.summary.scalar('cross_entropy', cross_entropy)

		correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)

		merged = tf.summary.merge_all()

		return x, y_, keep_prob, y_conv, train_step, accuracy, merged

	def save_model(self, sess):
		save_path = self.saver.save(sess, self.path_check_point)
		print("Model saved in file: %s" % save_path)

	#def get_image_labels():
		

	def train_model(self, path, image_dir, batch_size, num_epochs, shuffle, test_size, distortion_range):
		read_object = Read_input(path, image_dir, batch_size, num_epochs, shuffle, test_size, distortion_range)
		image_batch = read_object.image_batch
		label_batch = read_object.label_batch
		image_test = read_object.image_test
		label_test = read_object.label_test

		with tf.Session() as sess:
			train_writer = tf.summary.FileWriter(self.summaries_dir + '/train', sess.graph)
			test_writer = tf.summary.FileWriter(self.summaries_dir + '/test')

			x, y_, keep_prob, y_conv, train_step, accuracy, merged = self.loss()
			sess.run(tf.global_variables_initializer())
			sess.run([
				tf.local_variables_initializer(),
				tf.global_variables_initializer(),
				])

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(sess=sess, coord=coord)

			#try:
			# in most cases coord.should_stop() will return True
			# when there are no more samples to read
			# if num_epochs=0 then it will run for ever
			i = 0
			try :
				while not coord.should_stop():
					# will start reading, working data from input queue
					# and "fetch" the results of the computation graph
					# into raw_images and raw_labels
					print(i)
					raw_images, raw_labels = sess.run([image_batch, label_batch])

					#y_conv = def_cnn()
					#x, y_, keep_prob, y_conv, train_step, accuracy = self.loss()

					#train_step.run(feed_dict={x: raw_images, y_: raw_labels, keep_prob: 0.5})
					
					summary, _ = sess.run([merged, train_step], feed_dict={x: raw_images, y_: raw_labels, keep_prob: 0.5})
					train_writer.add_summary(summary, i)

					if i % 10 == 0:
						raw_images_test, raw_labels_test = sess.run([image_test, label_test])
						#pred = y_conv.eval(feed_dict={x:raw_images_test, keep_prob: 1.0})
						#index = pred.argmax(1)#max(enumerate(pred), key=operator.itemgetter(1))
						#index_y = raw_labels_test.argmax(1)#max(enumerate(raw_labels), key=operator.itemgetter(1))
						#acc = sum(index == index_y)/(float(len(index_y)))
						#print(acc, y_conv.eval(feed_dict={x: raw_images_test, keep_prob : 1.0}))
						#print(raw_labels)

						summary, acc = sess.run([merged, accuracy], feed_dict={x: raw_images_test, y_: raw_labels_test, keep_prob: 1.0})
						print("The tensorflow accuracy {}".format(acc))
						#summary = sess.run(merged, feed_dict=feed_dict(False))
						test_writer.add_summary(summary, i)
					#else :
						#summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
						#train_writer.add_summary(summary, i)

					if i % 100 == 0:
						# Save the variables to disk.
						self.save_model(sess)
						
					i = i + 1

					#print(cross_entropy)
					#print raw_labels
					#finally:
				coord.request_stop()
				coord.join(threads)
			except Exception as e:
				print(e)
				print("Done Training")
			finally:
				train_writer.close()
				test_writer.close()
