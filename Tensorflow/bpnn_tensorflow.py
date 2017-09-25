import tensorflow as tf
import numpy as np

MNIST_TRAIN = "train.csv"
MNIST_TEST = "test.csv"

#Load data
training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
    filename=MNIST_TRAIN,
    target_column=0,
    target_dtype=np.int,
    features_dtype=np.float32)

#Split data set for training and testing
num_samples=len(training_set.data)
split = int(num_samples * 2/3)
train_data=training_set.data/255

x_train = train_data[:split]
x_test = train_data[split:]

train_target=training_set.target

y_train_raw = train_target[:split]
y_test_raw = train_target[split:]

m=y_train_raw.shape[0]
y_train=np.zeros((m,10))
for i in range(m):
    y_train[i][y_train_raw[i]]=1.0

#Initialize parameters in nn
num_input=training_set.data.shape[1]
v=tf.Variable(tf.random_uniform([num_input, num_input], 0, 0.01),name='v')
w=tf.Variable(tf.random_uniform([num_input, 10], 0, 0.005),name='w')
threshold_h=tf.Variable(tf.random_uniform([1, num_input], 0, 1),name='threshold_h')
threshold_y=tf.Variable(tf.random_uniform([1, 10], 0, 0.5),name='threshold_y')

#Fully connected BP network
pre_b=tf.matmul(x_train, v)-threshold_h
b=tf.sigmoid(pre_b)
pre_y=tf.matmul(b,w)-threshold_y
y=tf.sigmoid(pre_y)

#Loss function
loss=tf.reduce_mean(tf.reduce_sum(tf.square(y - y_train),1))
train =  tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

#Training
for step in range(1, 4001):
    sess.run(train)
    if(step%20==0):
        print (step, sess.run(loss))


#Save session for continuous learning
saver=tf.train.Saver()
saver.save(sess, 'bpnn', global_step=4000)

sess.close()