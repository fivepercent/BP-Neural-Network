# Imports
import pandas as pd # load csv's (pd.read_csv)
import numpy as np # math (lin. algebra)

import sklearn as skl # machine learning
from sklearn.ensemble import RandomForestClassifier

# Visualisation
import matplotlib.pyplot as plt # plot the data
import seaborn as sns # data visualisation
sns.set(color_codes=True)
#% matplotlib inline

# load data as Pandas.DataFrame
train_df = pd.read_csv('train.csv')
train_data = train_df.values

test_df = pd.read_csv('test.csv')
test_data = test_df.values

# Holdout ( 2/3 to 1/3 )
num_samples = train_data.shape[0] # number of features
print("Number of all samples: \t\t", num_samples)
split = int(num_samples * 2/3)

train = train_data[:split]
test = train_data[split:]

print("Number of samples used for training: \t", len(train), 
      "\nNumber of samples used for testing: \t", len(test))

clf=BPneuralNetwork_one(10,784,0.1,10)
x_train,y_train=clf.loadData(train_data)
model=clf.fit(x_train,y_train)

score=model.score(test)

print(score)