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


# ### Treating outliers in the continuous columns

# In[19]:


numeric_col = dataframe.select_dtypes(include=np.number).columns

for col in numeric_col:    
    dataframe[col] = winsorize(dataframe[col], limits=[0.05, 0.1],inclusive=(True, True))

# Now run the code snippet to check outliers again


# ### Label Encode Categorical variables

# In[20]:


# Initializing Label Encoder
le = LabelEncoder()

# Iterating through each of the categorical columns and label encoding them
for feature in categorical_col:
    try:
        dataframe[feature] = le.fit_transform(dataframe[feature])
    except:
        print('Error encoding '+feature)


# In[21]:


dataframe.to_csv(r'preprocessed_data.csv',index=False)


# ## Model Selection

# #### Training And Test Datasets

# In[22]:


X,y = dataframe.drop(["y"],1).values, dataframe["y"].values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)


# In[23]:


# check the recorded instances of Train and test data sets for X nd y

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[24]:


from sklearn.metrics import fbeta_score, make_scorer
ftwo_scorer = make_scorer(fbeta_score, beta=2)
ftwo_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]},
                    scoring=ftwo_scorer)


# In[25]:


#using logisitic regression unoptimised version to find f score and accuracy

parameters = {"C":[0.001, 0.01, 0.1, 1, 10, 100, 1000]}

log_model_before_smote = LogisticRegression(random_state = 42, penalty = "l2")

fbeta_scorer = make_scorer(fbeta_score, beta = 0.5) #fbeta score

grid_item = GridSearchCV(log_model_before_smote, param_grid = parameters, scoring = fbeta_scorer) #grid search on the classsifier using 'scorer' as the scoring method

grid_fit = grid_item.fit(X_train, y_train) #fitting grid search to training data and find optimal parameters using fit()

best_estimators = grid_fit.best_estimator_ #get the estimator

best_predictions = best_estimators.predict(X_test) #predictions on unoptimised model

print("\nUnoptimised Model\n----")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test,best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test,best_predictions, beta = 0.5)))
print(best_estimators)


# In[26]:


# using parameters obtained above to make an optimised logistitc model

parameters = {"C":[0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6 ]}
log_model_before_smote = LogisticRegression(random_state = 42, penalty = "l2")

fbeta_scorer = make_scorer(fbeta_score, beta = 0.5) #fbeta score

grid_item = GridSearchCV(log_model_before_smote, param_grid = parameters, scoring = fbeta_scorer) #grid search on the classsifier using 'scorer' as the scoring method

grid_fit = grid_item.fit(X_train, y_train) #fitting grid search to training data and find optimal parameters using fit()

best_estimators = grid_fit.best_estimator_ #get the estimator

best_predictions = best_estimators.predict(X_test) #predictions on unoptimised model

print("\nOptimised Model\n----")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test,best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test,best_predictions, beta = 0.5)))
print(best_estimators)


# In[27]:


# Now, using the model with optimised parameters to predict and get the confusion matrix

model_log = LogisticRegression(C=1, random_state=42,penalty = "l2")
model_log.fit(X_train, y_train)
y_pred_log = model_log.predict(X_test)

conf_matrix = confusion_matrix(y_pred_log,y_test)
print(conf_matrix)
print(classification_report(y_test, y_pred_log))


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)


# In[29]:


xyz=@@


# In[32]:


get_ipython().system('pip3 install Imblearn import imblearn')


# In[33]:


from imblearn.over_sampling import SMOTE


# In[30]:


pip install -U imbalanced-learn


# In[31]:


conda install -c glemaitre imbalanced-learn


# In[ ]:


get_ipython().system('pip install imblearn')


# In[34]:


import numpy as np
import pandas as pd
##from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from collections import Counter

# Generate a synthetic imbalanced dataset (you can replace this with your dataset)
##X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],n_informative=3, n_redundant=1, flip_y=0, n_features=20,n_clusters_per_class=1, n_samples=1000, random_state=42)

# Check the class distribution before SMOTE
##print("Class distribution before SMOTE:", Counter(y))

# Instantiate the SMOTE resampler
smote = SMOTE(sampling_strategy='auto', random_state=42)

# Apply SMOTE to the dataset
X_resampled, y_resampled = smote.fit_resample(X, y)

# Check the class distribution after SMOTE
print("Class distribution after SMOTE:", Counter(y_resampled))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[45]:


# Use the predict_proba method to get class probabilities for the testing data
y_probabilities = model.predict_proba(X_test)[:, 1]  # Probability for class 1

# Define a threshold (e.g., 0.5) to convert probabilities to binary predictions
threshold = 0.5

# Create binary predictions based on the threshold
y_predictions = (y_probabilities > threshold).astype(int)

# Compute the confusion matrix
confusion = confusion_matrix(y_test, y_predictions)

print("Confusion Matrix:")
print(confusion)


# In[44]:


y_probabilities = model.predict_proba(X_test)
print("Class 1 Probabilities:")
print(y_probabilities[:5, 1])


# In[42]:


from imblearn.combine import SMOTETomek


# In[41]:





# In[40]:


# since earlier we had found that our response variable classes are imbalanced so we decided to use Smote and Tomek method to perform mix of oversmapling and undersampling to balance both outcomes across our response variable.
X_columns = dataframe.drop(["y"],1).columns
smote_tomek  = SMOTETomek(sampling_strategy = 'auto')
X_smt, y_smt = smote_tomek.fit_sample(X_train, y_train)

df_X_smt = pd.DataFrame(data = X_smt, columns = X_columns)
df_y_smt = pd.DataFrame(data = y_smt, columns = ['y'])


#print statements to check 

print("length of oversampled data is ",len(df_y_smt))
print("percentage of subscription ", (len(df_y_smt[df_y_smt['y']== 1])/len(df_y_smt))*100)
print("percentage of no subscription ",(len(df_y_smt[df_y_smt['y']== 0])/len(df_y_smt))*100)


# ### Model Selection

# In[ ]:


#Now, performing logistic regression with sampled features 

model_lr = LogisticRegression(random_state = 42, penalty = "l2")
model_rfc = RandomForestClassifier(n_estimators= 200, max_features= 'auto', max_depth= 20 , criterion= 'gini')

list_model = [model_lr, model_rfc]
for model in list_model:
    rfe = RFE(model,n_features_to_select = 20)
    X_smt_rfe = rfe.fit_transform(X_smt, y_smt)
    X_test_rfe = rfe.transform(X_test)
    # model  = model.fit(X_smt_rfe,y_smt)
    no_stratified_folds = StratifiedKFold(n_splits = 5, random_state= 1 )
    crossval_score_model = cross_val_score(model,X_smt_rfe ,y_smt, scoring = 'accuracy', cv = no_stratified_folds,n_jobs= 1, error_score='raise'  )
    print("Accuracy for model {} is : {}".format(model,np.mean(crossval_score_model)))
    print("Standard deviation for model {} is : {}".format(model,np.std(crossval_score_model)))


# In[ ]:


#So we are, performing Random forest classification with sampled features 

model_rfc = RandomForestClassifier(n_estimators= 200, max_features= 'auto', max_depth= 20 , criterion= 'gini')

rfe = RFE(model_rfc,n_features_to_select = 5)
X_smt_rfe = rfe.fit_transform(X_smt, y_smt)
X_test_rfe = rfe.transform(X_test)
model_rfc.fit(X_smt_rfe,y_smt)
y_pred = model_rfc.predict(X_test_rfe)
conf_matrix_rfe = confusion_matrix(y_test,y_pred)
print(conf_matrix_rfe)
print(classification_report(y_test, y_pred))


# #### Confusion Matrix, Precision, Recall, F1-Score

# In[ ]:


def metrics_model(y_test, y_pred):
    conf_matrix_rfe = confusion_matrix(y_test,y_pred)
    TP = conf_matrix_rfe[1,1]
    FN = conf_matrix_rfe[1,0]
    FP = conf_matrix_rfe[0,1]
    TN = conf_matrix_rfe[0,0]
    
    #printing confusion matrix
    
    print("confusion matrix:\n",conf_matrix_rfe)
    
    #print the accuracy score
    print("Accuracy:", round(accuracy_score(y_test, y_pred),2))
    
    #print the sensitivity/recall/true positive rate
    print("Sensitivity:", round(recall_score(y_test, y_pred),2))
    
    #precision/positive predictive value
    print("Precision:", round(precision_score(y_test, y_pred),2))


# In[ ]:


print(metrics_model(y_test, y_pred))


# In[ ]:


# let's see if increaing threshold could reduce false positives

y_pred_prob = model_rfc.predict_proba(X_test_rfe)[:,1]
y_pred_threshold = np.where(y_pred_prob< 0.45, 0 , 1)


# In[ ]:




