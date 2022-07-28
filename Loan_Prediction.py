#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[68]:


df_train = pd.read_csv("Ruturaj Patil - train_data.csv")
df_test = pd.read_csv('Ruturaj Patil - test_data.csv')


# In[69]:


df_test.shape


# In[70]:


df_train.shape


# In[71]:


df_train.shape


# In[72]:


df_train.describe()


# In[73]:


df_train.info()


# In[74]:


df_train.isnull().sum()


# In[75]:


df_train = df_train.dropna()


# In[76]:


df_train.shape


# In[77]:


df_train.head()


# In[78]:


df_train.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)


# In[79]:


df_train.head()


# In[80]:


df_train['Dependents'].value_counts()


# In[81]:


df_train.head()


# In[82]:


df_train['Loan_Amount_Term'].value_counts()


# In[83]:


df_train= df_train.replace(to_replace='3+', value=4)


# In[84]:


df_train['Dependents'].value_counts()


# In[85]:


sns.countplot(x='Education',hue='Loan_Status',data=df_train)


# In[86]:


sns.countplot(x='Married',hue='Loan_Status',data=df_train)


# In[87]:


df_train.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)


# In[88]:


df_train.head()


# In[89]:


X = df_train.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y = df_train['Loan_Status']


# In[90]:


from sklearn.model_selection import train_test_split


# In[91]:


X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=2)


# In[92]:


print(X.shape, X_train.shape, X_test.shape)


# In[93]:


from sklearn import svm
from sklearn.metrics import accuracy_score


# In[94]:


classifier = svm.SVC(kernel='linear')


# In[95]:


classifier.fit(X_train,y_train)


# In[96]:


X_train_prediction = classifier.predict(X_train)
training_data_accuray = accuracy_score(X_train_prediction,y_train)


# In[97]:


training_data_accuray


# In[98]:


X_test_prediction = classifier.predict(X_test)
test_data_accuray = accuracy_score(X_test_prediction,y_test)


# In[99]:


test_data_accuray


# In[100]:


import pickle

pickle.dump(classifier , open('model.pkl','wb'))
loan_prediction = pickle.load(open('model.pkl','rb'))


# In[ ]:




