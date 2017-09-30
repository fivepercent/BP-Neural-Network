import util
import tensorflow as tf

def classifier(x):
	#Initialize parameters in nn
	F1_weights=util.weights([784, 784],0.01,'F1_weights')
	F1_bias=util.bias([784],'F1_bias')

	F2_weights=util.weights([784, 10],0.01,'F2_weights')
	F2_bias=util.bias([10],'F2_bias')

	F1=tf.matmul(x, F1_weights)+F1_bias
	ReLU1=tf.nn.relu(F1)
	F2=tf.matmul(ReLU1, F2_weights)+F2_bias
	y=tf.nn.softmax(F2)

	return y