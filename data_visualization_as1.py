#!/usr/bin/env python
# coding: utf-8

# In[55]:


from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[2]:


print('xlrd installed')


# In[43]:


df_sur=pd.read_csv("https://cocl.us/datascience_survey_data")


# In[44]:


df_sur.head()


# In[45]:


df_sur.set_index('Unnamed: 0',inplace=True)


# In[19]:


df_sur.head()


# In[46]:


df_sur.sort_values("Very interested",axis=0,ascending=False,inplace=True)


# In[47]:


new_df_sur=df_sur


# In[48]:


new_df_sur.head()


# In[51]:


new_df_sur['Very interested%']=(new_df_sur['Very interested']/2233*100).round(2).astype(str)
new_df_sur['Somewhat interested%']=(new_df_sur['Somewhat interested']/2233*100).round(2).astype(str)
new_df_sur['Not interested%']=(new_df_sur['Not interested']/2233*100).round(2).astype(str)


# In[52]:


new_df_sur.head()


# In[62]:


ax = new_df_sur.plot(kind='bar', 
                figsize=(20, 8),
                rot=90,color = ['#5cb85c','#5bc0de','#d9534f'],
                width=.8,
                fontsize=14)
plt.title('Percentage of Respondents Interest in Data Science Areas',fontsize=16)
ax.set_facecolor('white')
ax.legend(fontsize=14,facecolor = 'white') 
ax.get_yaxis().set_visible(False)
for p in ax.patches:
    ax.annotate(np.round(p.get_height(),decimals=2), 
                (p.get_x()+p.get_width()/2., p.get_height()), 
                ha='center', 
                va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize = 14
               )


# In[ ]:




