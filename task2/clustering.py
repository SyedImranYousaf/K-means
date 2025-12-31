import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


df = pd.read_csv("C:\\Users\\Hp\\OneDrive\\Desktop\\AI_5th_semester\\lab13kmeans\\K-means\\task2\\Wholesale_customers_data.csv")


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)


kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

print("K-Means Silhouette Score:",
      silhouette_score(X_scaled, kmeans_labels))


with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)


dbscan = DBSCAN(eps=1.2, min_samples=6)
dbscan_labels = dbscan.fit_predict(X_scaled)

labels_unique = set(dbscan_labels)
if len(labels_unique) > 1 and -1 in labels_unique:
    mask = dbscan_labels != -1
    print("DBSCAN Silhouette Score:",
          silhouette_score(X_scaled[mask], dbscan_labels[mask]))
else:
    print("DBSCAN silhouette not applicable")


with open("dbscan_model.pkl", "wb") as f:
    pickle.dump(dbscan, f)


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=kmeans_labels)
plt.title("K-Means Clustering")

plt.subplot(1,2,2)
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=dbscan_labels)
plt.title("DBSCAN Clustering")

plt.show()
