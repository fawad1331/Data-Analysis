#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[5]:


path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df=pd.read_csv(path)
df.head()


# In[6]:


get_ipython().system(' pip install seaborn')


# In[13]:


import seaborn as sns
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


print(df.dtypes)


# In[24]:


df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()


# In[29]:


sns.regplot(x='engine-size',y='price',data=df)


# In[30]:


df[['engine-size','price']].corr()


# In[31]:


sns.regplot(x='highway-mpg',y='price',data=df)


# In[32]:


df[['highway-mpg','price']].corr()


# In[33]:


sns.regplot('peak-rpm','price',data=df)


# In[34]:


df[['peak-rpm','price']].corr()


# In[35]:


sns.regplot(x='price',y='stroke',data=df)


# In[36]:


sns.boxplot(x='body-style',y='price',data=df)


# In[37]:


sns.boxplot(x='engine-location',y='price',data=df)


# In[38]:


sns.boxplot(x='drive-wheels',y='price',data=df)


# In[40]:


df.describe()


# In[51]:


drive_wheels_counts=df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts


# In[52]:


drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts


# In[54]:


engine_loc_counts=df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name='engine-location'


# In[56]:


engine_loc_counts


# In[57]:


df_group_one = df[['drive-wheels','body-style','price']]


# In[59]:


df_group_one=df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df_group_one.head()


# In[60]:


df_gptest2 = df[['body-style','price']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'],as_index= False).mean()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline grouped_test_bodystyle')


# In[64]:


from scipy import stats


# In[65]:


pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# In[67]:


pearson_coef, p_value = stats.pearsonr(df['horsepower'],df['price'])


# In[68]:


print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# In[69]:


pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 


# In[70]:


grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)


# In[ ]:




