#!/usr/bin/env python
# coding: utf-8

# ## Importing necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


pip install xgboost


# In[3]:


pip install -U scikit-learn


# In[4]:


from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier ,RandomForestClassifier ,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge,Lasso
from sklearn.metrics import roc_auc_score ,mean_squared_error,accuracy_score,classification_report,roc_curve,confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from scipy.stats.mstats import winsorize
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns',None)
import six
import sys
sys.modules['sklearn.externals.six'] = six


# ## Data Loading and Cleaning

# ### Load and Prepare dataset

# In[5]:


# accessing to the folder where the file is stored
path = 'C:/Users/Abdullah/anaconda3/term-deposit-marketing-2020.csv'

# Load the dataframe
dataframe = pd.read_csv(path)

print('Shape of the data is: ',dataframe.shape)

dataframe.head()


# ## Check Numeric and Categorical Features

# In[6]:


# IDENTIFYING NUMERICAL FEATURES

numeric_data = dataframe.select_dtypes(include=np.number)

# select_dtypes selects data with numeric features

numeric_col = numeric_data.columns

# we will store the numeric features in a variable

print("Numeric Features:")
print(numeric_data.head())
print("===="*20)


# In[7]:


# IDENTIFYING CATEGORICAL FEATURES
categorical_data = dataframe.select_dtypes(exclude=np.number) # we will exclude data with numeric features
categorical_col = categorical_data.columns

# we will store the categorical features in a variable


print("Categorical Features:")
print(categorical_data.head())
print("===="*20)


# In[8]:


# CHECK THE DATATYPES OF ALL COLUMNS:

print(dataframe.dtypes)


# ### Check Missing Data

# In[9]:


# To identify the number of missing values in every feature

# Finding the total missing values and arranging them in ascending order
total = dataframe.isnull().sum()

# Converting the missing values in percentage
percent = (dataframe.isnull().sum()/dataframe.isnull().count())
print(percent)
dataframe.head()


# ### Dropping missing values

# In[10]:


# dropping features having missing values more than 60%
dataframe = dataframe.drop((percent[percent > 0.6]).index,axis= 1)

# checking null values
print(dataframe.isnull().sum())


# ### Fill null values in continuous features

# In[11]:


# imputing missing values with mean

for column in numeric_col:
    mean = dataframe[column].mean()
    dataframe[column].fillna(mean,inplace = True)
    
#     imputing with median
#     for column in numeric_col:
#     mean = dataframe[column].median()
#     dataframe[column].fillna(mean,inpalce = True)


# ## Check for Class Imbalance

# In[12]:


# we are finding the percentage of each class in the feature 'y'
class_values = (dataframe['y'].value_counts()/dataframe['y'].value_counts().sum())*100
print(class_values)


# ## EDA & Data Visualizations

# ### Univariate analysis of Categorical columns

# In[13]:


# Selecting the categorical columns
categorical_col = dataframe.select_dtypes(include=['object']).columns
plt.style.use('ggplot')
# Plotting a bar chart for each of the cateorical variable
for column in categorical_col:
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    dataframe[column].value_counts().plot(kind='bar')
    plt.title(column)


# ### Imputing unknown values of categorical columns

# In[14]:


# Impute mising values of categorical data with mode
for column in categorical_col:
    mode = dataframe[column].mode()[0]
    dataframe[column] = dataframe[column].replace('unknown',mode)


# ### Univariate analysis of Continuous columns

# In[15]:


for column in numeric_col:
    plt.figure(figsize=(20,5))
    plt.subplot(121)
    sns.distplot(dataframe[column])
    plt.title(column)


# In[16]:


for column in numeric_col:
    plt.figure(figsize=(20,5))
    plt.subplot(122)
    sns.boxplot(dataframe[column])
    plt.title(column)


# ### Bivariate Analysis - Categorical Columns

# In[17]:


for column in categorical_col:
    plt.figure(figsize=(20,4))
    plt.subplot(121)
    sns.countplot(x=dataframe[column],hue=dataframe['y'],data=dataframe)
    plt.title(column)    
    plt.xticks(rotation=90)


# In[18]:


f,ax=plt.subplots(figsize=(10,10))
sns.heatmap(dataframe.corr(),annot=True,linewidths=0.5,linecolor="black",fmt=".1f",ax=ax)
plt.show()


# #### Treating outliers in the continuous columns

# In[19]:


numeric_col = dataframe.select_dtypes(include=np.number).columns

for col in numeric_col:    
    dataframe[col] = winsorize(dataframe[col], limits=[0.05, 0.1],inclusive=(True, True))

# Now run the code snippet to check outliers again


# #### Label Encode Categorical variables

# In[20]:


dataframe.to_csv(r'preprocessed_data.csv',index=False)


# In[ ]:




