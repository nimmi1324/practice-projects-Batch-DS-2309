#!/usr/bin/env python
# coding: utf-8

# # Import Neccesary Libraries

# In[52]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from warnings import filterwarnings
filterwarnings(action='ignore')
   


# # The dataset is related to red and white variants of the Portuguese "Vinho Verde" wine. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).
# 
# This dataset can be viewed as classification task. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.
# 

# Importing Dataset

# In[47]:


wine= pd.read_csv('C:/Users/HP/Desktop/Learning-2023/ML-Datasets/Red Wine/winequality-red.csv')
print("Successfully Imported Data!")
wine.head()


# In[49]:


print(wine.shape)
wine.describe(include='all')


# The task here is to predict the quality of red wine on a scale of 0–10 given a set of features as inputs.when we need to predict several scale of classification.

# # fixed acidity

# In[51]:


wine.groupby('quality').mean()


# In[53]:


print(wine.shape)


# Finding Null Values

# In[55]:


print(wine.isna().sum())


# In[56]:


wine.corr()


# In[57]:


wine.groupby('quality').mean()


# Data Analysis
# 
# 
# Countplot-

# In[58]:


sns.countplot(wine['quality'])
plt.show()


# In[59]:


sns.countplot(wine['pH'])
plt.show()


# In[60]:


sns.countplot(wine['alcohol'])
plt.show()


# In[61]:


sns.countplot(wine['fixed acidity'])
plt.show()


# In[62]:


sns.countplot(wine['volatile acidity'])
plt.show()


# In[63]:


sns.countplot(wine['citric acid'])
plt.show()


# In[64]:


sns.countplot(wine['density'])
plt.show()


# KDE plot-

# In[65]:


sns.kdeplot(wine.query('quality > 2').quality)


# Distplot-

# In[66]:


sns.distplot(wine['alcohol'])


# In[67]:


wine.plot(kind ='box',subplots = True, layout =(4,4),sharex = False)


# In[68]:


wine.plot(kind ='density',subplots = True, layout =(4,4),sharex = False)


# Histogram

# In[69]:


wine.hist(figsize=(10,10),bins=50)
plt.show()


# In[70]:


corr = wine.corr()
sns.heatmap(corr,annot=True)


# pair plot-

# In[71]:


sns.pairplot(wine)


# In[72]:


sns.violinplot(x='quality', y='alcohol', data=wine)


# In[74]:


# Create Classification version of target variable
wine['goodquality'] = [1 if x >= 7 else 0 for x in wine['quality']]# Separate feature variables and target variable
X = wine.drop(['quality','goodquality'], axis = 1)
Y = wine['goodquality']


# See proportion of good vs bad wines

# In[75]:


wine['goodquality'].value_counts()


# Feature Importance

# In[84]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.ensemble import ExtraTreesClassifier
classifiern = ExtraTreesClassifier()
classifiern.fit(X,Y)
score = classifiern.feature_importances_
print(score)


# Splitting Dataset

# In[87]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=7)


# LogisticRegression-

# In[88]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy Score:",accuracy_score(Y_test,Y_pred))


# In[89]:


confusion_mat = confusion_matrix(Y_test,Y_pred)
print(confusion_mat)


# Using KNN-

# In[91]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))


# Using SVC-

# In[92]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train,Y_train)
pred_y = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,pred_y))


# Using Decision Tree-

# In[93]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',random_state=7)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))


# Using Random Forest

# In[101]:


from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, Y_train)
y_pred2 = model2.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred2))


# Using Xgboost

# In[102]:


import xgboost as xgb
 model5 = xgb.XGBClassifier(random_state=1)
  model5.fit(X_train, Y_train)
y_pred5 = model5.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred5))


# In[105]:


from sklearn.svm import SVC
cl = SVC(kernel="rbf")
cl.fit(X_train,Y_train)

cm = confusion_matrix(Y_test,cl.predict(X_test))
print(cm)


# In[ ]:




