#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
from sklearn.decomposition import PCA


# In[2]:


med_data= pd.read_csv('medical_clean.csv')


# In[3]:


med_data.head()


# In[4]:


med_data.shape


# In[5]:


med_data.describe()


# In[6]:


med_data.columns #looking at the columns


# In[7]:


med_data.isnull().sum() # checking for missing  data


# In[8]:


med_data.duplicated().any() # checking  for duplicates


# In[9]:


df_numeric = med_data.select_dtypes(exclude=['object', 'category'])


# In[10]:


cont_df = df_numeric.drop(['CaseOrder', 'Zip','Children','Doc_visits','Full_meals_eaten','vitD_supp','Age','Population','Item1', 'Item2',
       'Item3', 'Item4', 'Item5', 'Item6', 'Item7', 'Item8'],axis =1)

cont_df.columns


# In[11]:


cont_df.head()# all columns are continoues 


# In[12]:


cont_df.shape


# In[13]:


from scipy import stats


# In[14]:


cont_df.std() # checking for outliers 


# In[15]:


from scipy.stats import zscore


# In[16]:


# identifying and treating outliers

# Calculate Z-scores
z_scores = cont_df.apply(zscore)

# Filter out rows where any column's Z-score is greater than 3 or less than -3
clean_data =cont_df [(z_scores.abs() < 3).all(axis=1)]

print(clean_data)


# In[17]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[18]:


#normalizing data


# In[19]:


scaler = StandardScaler()
norm_df = scaler.fit_transform(clean_data)


# In[20]:


scaled_df = pd.DataFrame(norm_df, columns = clean_data.columns)


# In[21]:


scaled_df.head()


# In[22]:


scaled_df.shape


# In[23]:


scaled_df.to_csv ('scaled_df.csv')


# In[24]:


#analyze data


# In[25]:


pca= PCA()


# In[26]:


norm_data = pca.fit_transform(scaled_df)


# In[27]:


leading_matrix = pd.DataFrame(pca.components_,columns= scaled_df.columns, index = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7'])
leading_matrix 


# In[28]:


#determine the components with thighest variance using the kaizer plot 


# In[29]:


pccomp = np.arange(pca.n_components_)+1
pccomp


# In[30]:


#kaiser rule 


# In[31]:


#extracting variance for each PC 
var = pca.explained_variance_
var


# In[32]:


plt.figure(figsize=(13, 6))
plt.plot(pccomp, var, 'b-')  
plt.title('Scree Plot', fontsize=16)  
plt.xlabel('Number of Components', fontsize=16)
plt.ylabel('Explained Variance Ratio', fontsize=16)
plt.axhline( y=1,color = 'r',linestyle = 'dashdot')
plt.grid(True)
plt.show()


# In[33]:


print (dict(zip(['PC1','PC2','PC3','PC4'],pccomp))) # based on the plot above we only keep the first 4 PCs


# In[34]:


print('Variance of the first  four  principal components:')
print (pca.explained_variance_[:4])


# In[35]:


# the percentage of explained variance for the significant principal component captured after the kaier plot

explained_var = pca.explained_variance_ratio_[:4] * 100
explained_var


# In[36]:


#Total variance
total_var_captured= np.sum(explained_var)
total_var_captured


# In[ ]:




