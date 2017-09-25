
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import csv
import pandas as pd
from tensorflow.python.platform import gfile


# In[2]:


#Load test data
with gfile.Open('test.csv') as csv_file:
    data_file = csv.reader(csv_file)
    data = []
    for row in data_file:
        data.append(np.asarray(row, dtype=np.float32))
x_test = np.array(data)


# In[3]:


#Normalization
x_test=x_test/255


# In[5]:


#Load saved model
saver=tf.train.import_meta_graph('bpnn-8032.meta')
graph = tf.get_default_graph()
sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver.restore(sess,'bpnn-8032') # <---------
v=graph.get_tensor_by_name('v_5:0')
w=graph.get_tensor_by_name('w_5:0')
threshold_h=graph.get_tensor_by_name('threshold_h_5:0')
threshold_y=graph.get_tensor_by_name('threshold_y_5:0')


# In[7]:


#BP network
pre_b=tf.matmul(x_test, v)-threshold_h
b=tf.sigmoid(pre_b)
pre_y=tf.matmul(b,w)-threshold_y
y=tf.sigmoid(pre_y)
y_out=tf.argmax(y,1)


# In[10]:


out=sess.run(y_out)


# In[12]:


pd.DataFrame({"ImageId": range(1, len(out)+1), "Label": out}).to_csv('out.csv', index=False, header=True)


# In[ ]:




