#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train= pd.read_csv('titanic_train.csv')


# In[3]:


train['Age'].head()


# In[5]:


sns.boxplot(x='Pclass',y='Age',data=train)


# In[6]:


#adding mean to null values
def add_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[7]:


train['Age']=train[['Age','Pclass']].apply(add_age,axis=1)


# In[8]:


train['Age'] 


# In[9]:


train.drop('Cabin',axis=1,inplace=True)


# In[10]:


train.columns


# In[11]:


train.dropna(inplace=True)


# In[12]:


train.isnull()


# In[13]:


train.head()


# In[14]:


sex=pd.get_dummies(train['Sex'],drop_first=True)


# In[15]:


embarked=pd.get_dummies(train['Embarked'],drop_first=True)


# In[16]:


embarked


# In[17]:


train = pd.concat([train,sex,embarked],axis=1)


# In[18]:


train.head()


# In[19]:


train.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace=True)


# In[20]:


train.head()


# In[21]:


train.drop(['PassengerId'],axis=1,inplace=True)


# In[22]:


train.head()


# In[23]:


train.columns


# In[24]:


x=train[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'male', 'Q',
       'S']]
y=train['Survived']


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=101)


# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


lr=LogisticRegression()


# In[29]:


lr.fit(x_train,y_train)


# In[30]:


prediction=lr.predict(x_test)


# In[31]:


prediction


# In[32]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[33]:


print(classification_report(y_test,prediction))


# In[34]:


confusion_matrix(y_test,prediction)


# In[ ]:




