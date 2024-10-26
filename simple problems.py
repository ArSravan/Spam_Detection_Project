import pandas as pd
import numpy as np

dataset = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\ow\Social_Network_Ads.csv")

x = dataset.iloc[:, [3,4]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(2,11):
    kmeans = KMeans(n_clusters = i, random_state= 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(2, 11), wcss)

kmeans = KMeans(n_clusters= 5, init= 'k-means++', random_state= 42)
y_means = kmeans.fit_predict(x)

print(plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = "red", label = "cluster1"))
print(plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = "blue", label = "cluster2"))
print(plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = "green", label = "cluster3"))
print(plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = "orange", label = "cluster4"))
print(plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = "cyan", label = "cluster5"))