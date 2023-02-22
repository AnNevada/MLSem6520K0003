#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", header=None)

data[0] = pd.Categorical(data[0]).codes

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X = est.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[ ]:




