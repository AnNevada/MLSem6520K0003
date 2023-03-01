#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('abalone.data', names=['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings'])

df['sex'] = pd.factorize(df['sex'])[0]

print(df.head())


# In[7]:


X = df.drop('rings', axis=1)
y = df['rings']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)


# In[9]:


y_pred = model.predict(X_test)

print(y_pred)


# In[ ]:




