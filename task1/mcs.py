
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


def load_data(filepath):
    df = pd.read_csv(filepath)
    print("Dataset Loaded Successfully\n")
    print(df.head())
    return df


def preprocess_data(df):
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled


def elbow_method(X_scaled):
    inertia = []
    K = range(1, 11)

    for k in K:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(K, inertia, marker='o')
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("WCSS")
    plt.grid(True)
    plt.show()



def apply_kmeans(X_scaled, k=5):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print("K-Means Silhouette Score:", score)
    return labels


def plot_kmeans(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=df['Annual Income (k$)'],
        y=df['Spending Score (1-100)'],
        hue=df['KMeans_Cluster'],
        palette='tab10'
    )
    plt.title("K-Means Clustering")
    plt.show()


def apply_dbscan(X_scaled, eps=0.9, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("DBSCAN Number of Clusters:", n_clusters)

    if n_clusters > 1:
        score = silhouette_score(X_scaled, labels)
        print("DBSCAN Silhouette Score:", score)

    return labels


def plot_dbscan(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=df['Annual Income (k$)'],
        y=df['Spending Score (1-100)'],
        hue=df['DBSCAN_Cluster'],
        palette='tab10'
    )
    plt.title("DBSCAN Clustering (-1 = Noise)")
    plt.show()


def save_results(df, filename):
    df.to_csv(filename, index=False)
    print("Results saved to", filename)



def main():
    df = load_data("C:\\Users\\Hp\\OneDrive\\Desktop\\AI_5th_semester\\lab13kmeans\\K-means\\task1\\Mall_Customers.csv")

    X, X_scaled = preprocess_data(df)

    elbow_method(X_scaled)

    df['KMeans_Cluster'] = apply_kmeans(X_scaled, k=5)
    plot_kmeans(df)

    df['DBSCAN_Cluster'] = apply_dbscan(X_scaled)
    plot_dbscan(df)

    save_results(df, "C:\\Users\\Hp\\OneDrive\\Desktop\\AI_5th_semester\\lab13kmeans\\K-means\\task1\\Mall_Customers_Clustered.csv")


if __name__ == "__main__":
    main()
