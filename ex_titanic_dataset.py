#!/usr/bin/env python
# coding: utf-8

# In[132]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import path
import re


# In[133]:


titanic = pd.read_csv("titanic_data.csv")


# In[134]:


print(titanic.shape)


# In[135]:


print(titanic.head())


# In[136]:


titanic.head()


# In[137]:


titanic.describe()


# In[138]:


titanic.info()


# In[139]:


print(np.unique(titanic["PassengerId"].values).size)


# In[140]:


np.unique(titanic["PassengerId"]).size


# In[141]:


titanic.set_index(["PassengerId"], inplace = True)


# In[142]:


titanic.head()


# In[143]:


titanic["SexCode"] = np.where(titanic["Sex"] == "female", 1, 0)


# In[144]:


titanic.head()


# In[145]:


np.unique(titanic["PClass"].values).size


# In[146]:


class_mapping = {"1st" : 1, "2nd" : 2, "3rd" : 3}


# In[147]:


titanic["PClass"] = titanic["PClass"].map(class_mapping)


# In[148]:


titanic.head()


# In[149]:


titanic.isnull().sum()


# In[150]:


avgAge = titanic["Age"].mean()


# In[151]:


titanic["Age"].fillna(avgAge, inplace = True)


# In[152]:


titanic["Age"].isnull().sum()


# In[153]:


titanic["Sex"].groupby(titanic["Sex"]).size()


# In[154]:


titanic.groupby(titanic["Sex"]).size()


# In[155]:


titanic.groupby(("Sex")).mean()


# In[156]:


patt = re.compile(r"\,\s(\S+\s)")


# In[157]:


titles = []


# In[158]:


for index, row in titanic.iterrows():
    m = re.search(patt, row["Name"])
    if m is None:
        title = "Mrs" if row["SexCode"] == 1 else "Mr"
    else:
        title = m.group(0)
        title = re.sub(r",", "", title).strip()
        if title[0] != "M":
            title = "Mrs" if row["SexCode"] == 1 else "Mr"
        else:
            if title[0] == "M" and title[1] == "a":
                print('Working...')
                title = "Mrs" if row["SexCode"] == 1 else "Mr"
    titles.append(title)


# In[159]:


titles


# In[160]:


titanic.head()


# In[161]:


titanic["Title"] = titles


# In[162]:


titanic.head()


# In[163]:


print(np.unique(titles).shape[0], np.unique(titles))


# In[164]:


titanic["Title"] = titanic["Title"].replace("Mlle", "Miss")
titanic["Title"] = titanic["Title"].replace("Ms", "Miss")


# In[165]:


titanic["Title"].unique()


# In[166]:


titanic[["Title", "Survived"]].groupby(["Title"]).mean()


# In[167]:


titanic["Died"] = np.where(titanic["Survived"] == 0, 1, 0)


# In[168]:


titanic.head()


# In[169]:


titanic["Age"].plot(kind = "hist", bins = 15)
df = titanic[titanic.Survived == 0]
df["Age"].plot(kind = "hist", bins = 15)
df = titanic[titanic.Survived == 1]
df["Age"].plot(kind = "hist", bins = 15)


# In[170]:


fig, axes = plt.subplots(nrows = 1, ncols = 2)
df = titanic[["Survived", "Died"]].groupby(titanic["Title"]).sum()
df.plot(kind = "bar", ax = axes[0])
df = titanic[["Survived", "Died"]].groupby(titanic["Title"]).mean()
df.plot(kind = "bar", ax = axes[1])


# In[171]:


fig, axes = plt.subplots(nrows = 1, ncols = 2)
df = titanic[["Survived", "Died"]].groupby(titanic["Sex"]).sum()
df.plot(kind = "bar", ax = axes[0])
df = titanic[["Survived", "Died"]].groupby(titanic["Sex"]).mean()
df.plot(kind = "bar", ax = axes[1])


# In[172]:


df = titanic[["Survived", "Died"]].groupby(titanic["PClass"]).sum()
df.plot(kind = "bar")
df = titanic[["Survived", "Died"]].groupby(titanic["PClass"]).mean()
df.plot(kind = "bar")


# In[182]:


titanic.head()


# In[ ]:




