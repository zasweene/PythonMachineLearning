import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

customer_data = pd.read_csv('customers.csv')

customer_data.head()
customer_data.describe()
customer_data.info()
customer_data.isnull().sum()

cust_shape = customer_data.shape
print(cust_shape)
customer_data.isnull().sum()

encoded_data = pd.get_dummies(customer_data)
matrix = encoded_data.corr()

sns.heatmap(matrix, annot = True)
plt.show()

col_param_1 = 'Annual Income (k$)'
col_param_2 = 'Spending Score (1-100)'

matrix_max = 0.98

customer_data = customer_data[['Annual Income (k$)', 'Spending Score (1-100)']]

num_clust = 15

wcss_list = []

for i in range(1, num_clust + 1):
    print(f'k={i}')
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(customer_data)
    wcss_list.append(kmeans.inertia_)

plt.figure(figsize=(20, 10))
plt.plot(range(1, num_clust + 1), wcss_list, marker = 'o', color = 'red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS Values')
plt.show()

chosen_cluster = 6

kmeans = KMeans(n_clusters=chosen_cluster, init='k-means++', n_init=10, random_state=42)
kmeans.fit(customer_data)

Y = kmeans.fit_predict(customer_data)

print(kmeans.cluster_centers_)
print(Y)

max_centre = kmeans.cluster_centers_.max()

X = customer_data.iloc[:, :].values

plt.figure(figsize = (20, 10))
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], c = 'lime', label = 'Cluster 1')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], c = 'maroon', label = 'Cluster 2')
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], c = 'gold', label = 'Cluster 3')
plt.scatter(X[Y == 3, 0], X[Y == 3, 1], c = 'violet', label = 'Cluster 4')
plt.scatter(X[Y == 4, 0], X[Y == 4, 1], c = 'blue', label = 'Cluster 5')
plt.scatter(X[Y == 5, 0], X[Y == 5, 1], c = 'black', label = 'Cluster 6')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'cyan', label = 'Centroids')
plt.legend(loc = "upper right")
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()