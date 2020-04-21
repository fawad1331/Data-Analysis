#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()


# In[5]:


from sklearn.linear_model import LinearRegression


# In[6]:


lm=LinearRegression()
lm


# In[8]:


X=df[['highway-mpg']]
Y=df[['price']]


# In[9]:


lm.fit(X,Y)


# In[11]:


Yhat=lm.predict(X)
Yhat[0:5]


# In[12]:


lm.intercept_


# In[14]:


lm.coef_


# In[16]:


lm1=LinearRegression()
lm1


# In[19]:


lm1.fit(df[['highway-mpg']],df[['price']])


# In[20]:


lm1.coef_


# In[21]:


lm1.intercept_


# In[22]:


Z=df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]


# In[23]:


lm.fit(Z,df[['price']])


# In[24]:


lm.intercept_


# In[25]:


lm.coef_


# In[27]:


lm2=LinearRegression()
lm2


# In[28]:


lm2.fit(df[['normalized-losses','highway-mpg']],df[['price']])


# In[29]:


lm2.intercept_


# In[30]:


lm2.coef_


# In[31]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[33]:


width=12
height=10
plt.figure(figsize=(width,height))
sns.regplot(x='highway-mpg',y='price',data=df)


# In[34]:


sns.regplot(x='peak-rpm',y='price',data=df)


# In[37]:


df[['peak-rpm','highway-mpg','price']].corr()


# In[39]:


Y_hat = lm.predict(Z)
Y_hat


# In[40]:


plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


# In[41]:


def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()


# In[44]:


x=df['highway-mpg']
y=df['price']


# In[45]:


f=np.polyfit(x,y,3)
p = np.poly1d(f)
print(p)


# In[46]:


PlotPolly(p,x,y,'highway-mpg')


# In[47]:


np.polyfit(x, y, 3)


# In[51]:


f1 = np.polyfit(x, y, 5)
p1 = np.poly1d(f1)
print(p)
PlotPolly(p1,x,y, 'Highway MPG')


# In[52]:


from sklearn.preprocessing import PolynomialFeatures


# In[53]:


pr=PolynomialFeatures(degree=2)
pr


# In[54]:


Z_pr=pr.fit_transform(Z)


# In[55]:


Z.shape
Z_pr.shape


# In[56]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[57]:


Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]


# In[58]:


pipe=Pipeline(Input)
pipe


# In[59]:


pipe.fit(Z,y)


# In[61]:


ypipe=pipe.predict(Z)
ypipe[0:4]


# In[64]:


lm.fit(X,Y)
print('The R-square is: ', lm.score(X, Y))


# In[65]:


Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])


# In[66]:


from sklearn.metrics import mean_squared_error


# In[67]:


mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)


# In[68]:


lm.fit(Z, df['price'])
print('The R-square is: ', lm.score(Z, df['price']))


# In[69]:


Y_predict_multifit = lm.predict(Z)


# In[70]:


print('The mean square error of price and predicted value using multifit is: ',       mean_squared_error(df['price'], Y_predict_multifit))


# In[75]:


new_input=np.arange(1, 100, 1).reshape(-1, 1)


# In[76]:


lm.fit(X, Y)
lm


# In[77]:


yhat=lm.predict(new_input)
yhat[0:5]


# In[78]:


plt.plot(new_input, yhat)
plt.show()


# In[ ]:




