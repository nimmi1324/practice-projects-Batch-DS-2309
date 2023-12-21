#!/usr/bin/env python
# coding: utf-8

# # Medical Cost Personal Insurance Project
# 
# 
# Batch -DS2309
# 

# # Import Neccesary Libraries

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn import metrics


# Project Description

# Health insurance is a type of insurance that covers medical expenses that arise due to an illness. These expenses could be related to hospitalisation costs, cost of medicines or doctor consultation fees. The main purpose of medical insurance is to receive the best medical care without any strain on your finances. Health insurance plans offer protection against high medical costs. It covers hospitalization expenses, day care procedures, domiciliary expenses, and ambulance charges, besides many others. Based on certain input features such as age , bmi,,no of dependents ,smoker ,region  medical insurance is calculated 

# Columns                                            
# •	age: age of primary beneficiary
# •	sex: insurance contractor gender, female, male
# •	bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9.
# •	children: Number of children covered by health insurance / Number of dependents
# •	smoker: Smoking
# •	region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
# •	charges: Individual medical costs billed by health insurance
# 

# In[4]:


df=pd.read_csv('C:/Users/HP/Desktop/Learning-2023/ML-Datasets/Medical Cost Insurance/medical_cost_insurance.csv')
df


# Now, we can observe the data and its shape(rows x columns)
# 
# This dataset contains 1338 data points with 6 independent features and 1 target feature(charges).

# In[11]:


df.info()


# we can see that the dataset contains 2 columns with float values 3 with categorical values and the rest contains integer values.

# In[14]:


df.head()


# The dataset contains 6 indepandance feature and 1 target feature(charges),it contains starting  5 features.

# In[15]:


df.tail()


# it contains last 5 target features.

# In[16]:


df.shape


# In[17]:


print(f"The rows and columns in the dataset:{df.shape}")
print(f"\n The column headers in the dataset:{df.columns}")


# This line are shows there arde 1338 rows and 7 column and another one is coloumn header index /dtype.

# In[18]:


df.describe()


# We can look at the descriptive statistical measures of the continuous data available in the dataset.

# In[19]:


df.dtypes


# It gives brief about the dataset which includes indexing type, int,dtype,float.

# In[20]:


df.isnull().sum()


# here we can conclude that there are no null values in the dataset given.

# # Let's visualize it using heatmap

# In[21]:


sns.heatmap(df.isnull())


# There is no null values in this  chart.

# In[23]:


features = ['sex', 'smoker', 'region']
 
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(1, 3, i + 1)
 
    x = df[col].value_counts()
    plt.pie(x.values,
            labels=x.index,
            autopct='%1.1f%%')
 
plt.show()


# The data provided to us is equally distributed among the sex and the region columns but in the smoker column, we can observe a ratio of 80:20.

# In[24]:


df['region'].value_counts().sort_values()


# This is  provide us four diffrent data(dtype-int64) in two regions.

# In[25]:


df['children'].value_counts().sort_values()


# In[26]:


#converting categorical to numerical
clean_df = {'sex': {'male' : 0 , 'female' : 1} ,
                 'smoker': {'no': 0 , 'yes' : 1},
                   'region' : {'northwest':0, 'northeast':1,'southeast':2,'southwest':3}
               }
df_copy = df.copy()
df_copy.replace(clean_df, inplace=True)


# In[27]:


df_copy.info()


# it provide dtypes,int and hpw much memory usage-73.3kb.

# In[28]:


df_copy.describe()


# In[29]:


df_copy.nunique().to_frame("No. of unique values")


# it shows no of unique value are present in this dataset.

# In[31]:


#checking the value counts of each column
for i in df_copy.columns:
    print(df_copy[i].value_counts())
print("\n")


# In[32]:


df_copy.duplicated().sum()


# In[36]:


sns.heatmap(df_copy.corr(),linewidth=0.1,fmt="0.1g",linecolor='red',annot=True,cmap="Blues_r")


# In[37]:


plt.figure(figsize=(12,10))
plt.title('Age VS Charges')
sns.barplot(x='age',y='charges',data=df_copy,palette='BuPu')
plt.show()


# its provide age vs charges ratio.

# In[38]:


features = ['sex', 'children', 'smoker', 'region']
 
plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
    plt.subplot(2, 2, i + 1)
    df.groupby(col).mean()['charges'].plot.bar()
plt.show()


# Charges are on the higher side for males as compared to females but the difference is not that much.
# Premium charged from the smoker is around thrice that which is charged from non-smokers.
# Charges are approximately the same in the given four regions.
# features = ['age', 'bmi']

# In[41]:


df.drop_duplicates(inplace=True)
sns.boxplot(df['age'])


# we can see that there are no outliers present in age column

# In[42]:


sns.boxplot(df['bmi'])


# Due to the presence of outliers present in bmi column we need to treat the outliers by replacing the values with mean as the bmi column consists of continuous data.

# In[51]:


from scipy.stats import zscore


# In[53]:


out_features=df_copy[['charges','age','sex','bmi','children','smoker','region']]
df=np.abs(zscore(out_features))
df


# In[ ]:




