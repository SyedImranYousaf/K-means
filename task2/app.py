import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

st.title("Customer Clustering App")

data = pd.read_csv("Wholesale_customers_data.csv")

# data = pd.read_csv("C:\\Users\\Hp\\OneDrive\\Desktop\\AI_5th_semester\\lab13kmeans\\K-means\\task2\\Wholesale_customers_data.csv")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

algo = st.selectbox("Select Algorithm", ["K-Means", "DBSCAN"])

if algo == "K-Means":
    model = pickle.load(open("kmeans_model.pkl", "rb"))
    # model = pickle.load(open("C:\\Users\\Hp\\OneDrive\\Desktop\\AI_5th_semester\\lab13kmeans\\K-means\\task2\\kmeans_model.pkl", "rb"))
    labels = model.predict(X_scaled)
else:
    model = pickle.load(open("dbscan_model.pkl", "rb"))
        # model = pickle.load(open("C:\\Users\\Hp\\OneDrive\\Desktop\\AI_5th_semester\\lab13kmeans\\K-means\\task2\\dbscan_model.pkl", "rb"))
    labels = model.fit_predict(X_scaled)

data["Cluster"] = labels
st.dataframe(data)

st.scatter_chart(data, x="Fresh", y="Grocery", color="Cluster")
