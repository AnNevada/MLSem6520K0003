#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector

data = load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = LogisticRegression(solver='liblinear', multi_class='auto')

sfs = SequentialFeatureSelector(model, n_features_to_select=3, direction='forward')

sfs.fit(X_train, y_train)

selected_features = sfs.get_support()

X_train_sfs = X_train[:, selected_features]
X_test_sfs = X_test[:, selected_features]

model.fit(X_train_sfs, y_train)

accuracy = model.score(X_test_sfs, y_test)
print(f"Accuracy: {accuracy}")


# In[ ]:




