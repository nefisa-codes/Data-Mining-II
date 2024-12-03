#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats


# In[2]:


med_data = pd.read_csv('medical_clean.csv')


# In[3]:


med_data.columns #looking at the columns


# In[4]:


med_data.describe() #looking at data stats for each variables 


# In[5]:


med_data.dtypes # looking at datatypes for each variables.


# In[6]:


med_data.isnull().sum() # checking for missing  data


# In[7]:


med_data.isnull().sum() # checking for missing  data



# In[8]:


# seperating the variables that will be used for clustering 
new_data = med_data[['Lat','Lng','Additional_charges']].copy() 


# In[ ]:





# In[9]:


new_data # these are the columns i will be working  with


# In[10]:


new_data.describe()


# In[11]:


from scipy.stats import zscore


# In[12]:


# identifying and treating outliers

# Calculate Z-scores
z_scores = new_data.apply(zscore)

# Filter out rows where any column's Z-score is greater than 3 or less than -3
clean_data =new_data [(z_scores.abs() < 3).all(axis=1)]

print(clean_data)


# In[13]:


# Scaling the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clean_data)
print (scaled_data)


# In[ ]:


clean_data.to_csv('MSDA212_PA_cleanData.cvs')


# In[ ]:


# looking that the data before clustering
import seaborn as sns 

sns.scatterplot(data=clean_data ,x='Lng', y ='Lat', hue='Additional_charges')


# In[ ]:


from sklearn.preprocessing import StandardScaler
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt


# In[ ]:


# Scaling the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clean_data)
print (scaled_data)


# In[ ]:


#  K-means clustering
centroids, _ = kmeans(scaled_data, 3)
cluster_labels, _ = vq(scaled_data, centroids)

scaled_data = clean_data.copy()

# Adding cluster labels to the original DataFrame
scaled_data['cluster_labels'] = cluster_labels



# Now, I will calculate the distortion for cluster ranges between 2 and 11. Distortion measures how well the clusters fit the data; it decreases as the number of clusters increases. As shown below, there is a significant drop in distortion from 2 to 3 clusters, indicating that the optimal number of clusters might be 3. I will plot the elbow method for better visualization.

# In[ ]:


#Calculatting distortions for different numbers of clusters

distortions = []
num_clusters = range(2, 11)

for i in num_clusters:
    centroids, distortion = kmeans(scaled_data, i)
    distortions.append(distortion)
    cluster_labels, _ = vq(scaled_data, centroids)
   
    print(f"Number of clusters: {i}, Distortion: {distortion}")
    


# In[ ]:


# the elbow plot

plt.figure(figsize=(8, 4))
plt.plot(num_clusters, distortions, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.show()


# Looking at the plot, there is a significant drop in distortion from 2 to 3 clusters, and the graph continues to decrease. The elbow plot clearly shows that 3 is the optimal number of clusters for analyzing additional charges paid across the country.

# 
# The scatter plot below illustrates three distinct groups, each representing different amounts paid for additional charges. To enhance the clarity of this visualization, I will compute a cluster summary.
# 
# As shown in the cluster summary code below, the average additional charge per patient is $10,683.01 in Cluster 0. In Clusters 1 and 2, the average additional charges per patient are $9,916.44 and $23,714.35, respectively.

# In[ ]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='Lng', y='Lat', hue='cluster_labels', data=scaled_data)
plt.title('K-means Clustering Results')
plt.xlabel('Lng')
plt.ylabel('Lat')
plt.legend(title='Cluster Labels')
plt.show()


# In[ ]:


cluster_summary = scaled_data.groupby('cluster_labels')['Additional_charges'].describe()

cluster_summary


# In[ ]:





# In[ ]:




