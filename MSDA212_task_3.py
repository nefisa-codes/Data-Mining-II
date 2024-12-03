#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install mlxtend


# In[31]:


import pandas as pd 
from pandas import DataFrame 
import numpy as np
from mlxtend.preprocessing  import TransactionEncoder
import warnings
warnings.filterwarnings('ignore')


# In[32]:


data = pd.read_csv('medical_market_basket.csv')
data.head()


# In[33]:


data.shape


# In[34]:


data = data[data['Presc01'].notna()]


# In[35]:


data.shape


# In[36]:


data.head()


# In[37]:


# converting dataframe to list of lists
rows = [] 
for i in range (0,7501):
        rows.append([str(data.values[i,j])
for j in range (0,20)])


# In[38]:


#list fed to TransactionEncoder


# In[39]:


DE =TransactionEncoder()
array =DE.fit(rows).transform(rows)

transcation=pd.DataFrame(array,columns =DE.columns_)


# In[11]:


transcation


# In[40]:


for col in transcation.columns: print (col)


# In[41]:


#remove empty columns

cleaned_df=transcation.drop(['nan'],axis =1)
cleaned_df.head(7505)


# In[42]:


cleaned_df.to_csv('df_clean1.csv',index = False)


# In[43]:


cleaned_df.columns


# In[44]:


from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


data =pd.read_csv('df_clean1.csv')


# In[46]:


data.head()


# In[47]:


data.shape


# In[20]:


# the most comment medications 


# In[48]:


count =data.loc[:,:].sum()
pop_item=count.sort_values(ascending=False).head(5)
pop_item=pop_item.to_frame()
pop_item=pop_item.reset_index()
pop_item=pop_item.rename(columns = {'index':'medications',0:'count'})
print(pop_item)


# In[49]:


#data visualization  for the most prescribed meds


# In[23]:


plt.rcParams['figure.figsize']=(10,6)
ax=pop_item.plot.barh(x='medications', y ='count')
plt.title('Popular medications')
plt.gca().invert_yaxis()


# In[50]:


#creating apriori object called rules
rules=apriori(data,min_support = 0.02,use_colnames =True)
rules.head(5)


# In[52]:


rul_table = association_rules(rules,metric='lift', min_threshold = 1)
rul_table.head(20)


# In[53]:


top_three_rules = rul_table.sort_values('confidence',ascending =False).head(3)
top_three_rules


# In[54]:


top_three_rules = rul_table.sort_values('lift',ascending =False).head(3)
top_three_rules


# In[55]:


top_three_rules = rul_table.sort_values('support',ascending =False).head(3)
top_three_rules


# In[56]:


filtered_rules = rul_table[rul_table['lift'] > 1]


# In[57]:


sorted_rules = filtered_rules.sort_values(by='lift', ascending=False)
sorted_rules


# In[ ]:




