
#***********Role of Dendrograms for Hierarchical Clustering***********

# Press CTRL + K + U to remove comments

# Import python packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
customer_data = pd.read_csv('shopping_data.csv')

# Write out some info on dataset
print(customer_data.shape)  # 200records 5 attributes
print(customer_data.head())

# Remove the first 3 columns (ID, genre and age) bc unnecessary
data = customer_data.iloc[:, 3:5].values


# Example datapoints
X = np.array([[5,3],
[10,15],
[15,12],
[24,10],
[30,30],
[85,70],
[71,80],
[60,78],
[70,55],
[80,91] ])

# Plot data points
labels = range(1, 201)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(data[:,0],data[:,1], label='True Position')
for label, x, y in zip(labels, data[:, 0], data[:, 1]):
 plt.annotate(label,xy=(x, y),xytext=(-3, 3),textcoords='offset points', ha='right',va='bottom')
plt.show()

# Draw dendrograms for hierarchcal clustering
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
linked = linkage(data, 'single')
labelList = range(1, 201)
plt.figure(figsize=(10, 7))
dendrogram(linked,
 orientation='top',
 labels=labelList,
 distance_sort='descending',
 show_leaf_counts=True)
plt.show()



#********Hierarchical Clustering via Scikit-Learn********

# Test data points
X = np.array([[5,3],
[10,15],
[15,12],
[24,10],
[30,30],
[85,70],
[71,80],
[60,78],
[70,55],
[80,91] ])

# Import liberary for predicting cluster
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')   # 2 clusters using euclidian distance
cluster.fit_predict(data)  # returns names of cluster each point belong to

# Plot result
plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow')
plt.show() 
