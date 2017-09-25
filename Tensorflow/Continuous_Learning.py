
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np

#Load saved Model
saver=tf.train.import_meta_graph('bpnn-4000.meta')
graph = tf.get_default_graph()
sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess,'bpnn-4000') # <---------
v=graph.get_tensor_by_name('v_5:0')
w=graph.get_tensor_by_name('w_5:0')
threshold_h=graph.get_tensor_by_name('threshold_h_5:0')
threshold_y=graph.get_tensor_by_name('threshold_y_5:0')


# In[ ]:


#Loss function
loss=graph.get_tensor_by_name('Mean_9:0')
train =  tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# In[4]:


#Training
for step in range(4501, 5001):
    sess.run(train)
    if step%20==0:
        print (step, sess.run(loss))


# In[ ]:


l=sess.run(loss)
step=5000
while l>0.1:
    sess.run(train)
    step+=1
    l=sess.run(loss)
    if step%20==0:
        print (step, l)


# In[10]:


saver.save(sess, 'bpnn', global_step=8032)


# In[4]:


sess.close()

