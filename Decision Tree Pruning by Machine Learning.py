#!/usr/bin/env python
# coding: utf-8

# # Decision Trees with Pruning

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets import load_iris


# In[2]:


iris = load_iris()


# In[3]:


iris


# In[4]:


iris.data


# In[5]:


iris.target


# In[6]:


import seaborn as sns


# In[8]:


df = sns.load_dataset('iris')


# In[9]:


df.head()


# In[10]:


#independent and Dependent Features
X = df.iloc[:,:-1]
y = iris.target


# In[11]:


X,y


# In[13]:


#Train test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[14]:


X_train


# In[19]:


from sklearn.tree import DecisionTreeClassifier


# ## Post Pruning Decision Tree

# In[20]:


treemodel = DecisionTreeClassifier(max_depth=2)


# In[21]:


treemodel.fit(X_train, y_train)


# In[30]:


from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treemodel, filled=True)


# In[31]:


#Prediction


# In[32]:


y_pred = treemodel.predict(X_test)


# In[33]:


y_pred


# In[34]:


from sklearn.metrics import classification_report, accuracy_score


# In[35]:


score = accuracy_score(y_pred, y_test)


# In[36]:


score


# In[59]:


print(classification_report(y_pred, y_test))


# # Pre Pruning Decision Tree

# In[60]:


parameter = {
    "criterion":["gini","entropy","log_loss"],
    "splitter":["best","random"],
    "max_depth":[1,2,3,4,5],
    "max_features": ['auto','sqrt','log2']
}


# In[61]:


from sklearn.model_selection import GridSearchCV


# In[62]:


treemodel=DecisionTreeClassifier()
cv=GridSearchCV(treemodel, param_grid=parameter, cv=5, scoring='accuracy')


# In[63]:


cv.fit(X_train, y_train)


# In[64]:


cv.best_params_


# In[65]:


y_test


# In[66]:


y_pred


# In[67]:


y_pred = cv.predict(X_test)


# In[68]:


y_pred


# In[69]:


from sklearn.metrics import classification_report, accuracy_score


# In[70]:


score = accuracy_score(y_pred, y_test)


# In[71]:


score


# In[72]:


print(classification_report(y_pred,y_test))

