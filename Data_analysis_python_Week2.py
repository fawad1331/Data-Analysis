#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


path="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"


# In[3]:


df=pd.read_csv(path,header=None)


# In[4]:


df.head()


# In[5]:


df.tail(10)


# In[6]:


headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]


# In[7]:


print(headers)


# In[8]:


df.columns=headers


# In[9]:


df.head()


# In[11]:


df.dropna(subset=["price"],axis=0)


# In[12]:


df.columns


# In[13]:


df.to_csv("automobile.csv",index=False)


# In[34]:


df['symboling']=df['symboling']+1


# In[36]:


df.describe()


# In[18]:


df.describe(include='all')


# In[23]:


df[['wheel-base', 'compression-ratio']]


# In[24]:


df[['wheel-base', 'compression-ratio']].describe()


# In[37]:


import numpy as np


# In[38]:


df.replace("?", np.nan, inplace = True)
df.head(5)


# In[39]:


missing_data=df.isnull()


# In[40]:


missing_data.head(5)


# In[ ]:




