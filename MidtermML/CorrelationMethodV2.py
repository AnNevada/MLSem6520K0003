#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score

# Load the wine dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df = pd.read_csv(url, header=None)

# Set the column names
column_names = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',
                'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

df.columns = column_names

# Split the dataset into training and testing sets
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate the correlation between each feature and the target variable
corr_matrix = X_train.corr()
corr_with_target = corr_matrix['Proline'].sort_values(ascending=False)

# Select the most important features
important_features = corr_with_target.index[:5]

print(important_features)

# Create the decision tree classifier
dt = DecisionTreeClassifier(random_state=42)

# Define the RFE estimator
rfe = RFE(estimator=dt, n_features_to_select=5, step=1)

# Fit the RFE estimator to the training data
rfe.fit(X_train, y_train)

# Print the ranking of the features
print("Feature ranking:", rfe.ranking_)

# Train a model using the selected features
X_train_selected = X_train[important_features]
X_test_selected = X_test[important_features]

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_selected, y_train)
y_pred = dt.predict(X_test_selected)

print('Accuracy:', accuracy_score(y_test, y_pred))


# In[ ]:




