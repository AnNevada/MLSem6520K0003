#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[5]:


wine_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', delimiter=';')
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']


# In[6]:


corr = X.corrwith(y)


# In[7]:


corr = corr.abs().sort_values(ascending=False)
print(corr)


# In[ ]:




