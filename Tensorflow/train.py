import sys
sys.path.append('../MNIST')
import os.path

import tensorflow as tf
import numpy as np
import loadData
import util
import bpnn

def train_mnist(max_iter, learning_rate, f):
	resume=os.path.exists('./session/bpnn.meta')

	#load data
	x_train=tf.cast(loadData.load_train_images().reshape(-1,784),dtype='float32')
	y_train_raw = loadData.load_train_labels()
	y_train=tf.one_hot(y_train_raw, 10)

	y=bpnn.classifier(x_train)
	global_step = tf.Variable(0, trainable=False)
	lr = tf.placeholder(tf.float32)

	#Loss function
	loss=tf.reduce_mean(tf.reduce_sum(tf.square(y - y_train),1))
	train =  tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
	accuracy= util.accuracy(y,y_train)

	config=tf.ConfigProto(allow_soft_placement= True,log_device_placement= True)
	saver = tf.train.Saver()
	sess=tf.Session(config=config)
	if resume:
		saver.restore(sess,'./session/bpnn')
	else:
		sess.run(tf.global_variables_initializer())

    #training
	step=0
	while step<max_iter:
		_, l, acc, gs = sess.run([train, loss, accuracy,global_step],feed_dict={lr:learning_rate})
		step=gs
		if(step%f==0):print ('Globle step: %d, Loss: %f Acc: %f'%(step, l, acc))

    #Save model for continuous learning
	saver.save(sess, './session/bpnn')

	sess.close()