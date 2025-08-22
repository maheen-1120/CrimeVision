import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
df = pd.read_csv("Crime_dataset.csv")

le = LabelEncoder()
encoded_df = df.copy()
for col in ['City','Crime_Type','Weapon','Victim_Gender','Suspect_Gender','Arrest_Made']:
    encoded_df[col] = le.fit_transform(encoded_df[col])

st.title("Crime Data Dashboard")

city = st.selectbox("Select City", ["All"] + sorted(df["City"].unique().tolist()))
crime_type = st.selectbox("Select Crime Type", ["All"] + sorted(df["Crime_Type"].unique().tolist()))

filtered_df = df.copy()
if city != "All":
    filtered_df = filtered_df[filtered_df["City"] == city]
if crime_type != "All":
    filtered_df = filtered_df[filtered_df["Crime_Type"] == crime_type]

st.subheader("Filtered Crime Data")
st.dataframe(filtered_df)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Crimes Per City")
    city_counts = df["City"].value_counts().reset_index()
    city_counts.columns = ["City","Count"]
    fig, ax = plt.subplots(figsize=(3,3))
    sns.barplot(x="City", y="Count", data=city_counts, palette="pastel", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=False)

with col2:
    st.subheader("Crime Type Distribution")
    fig, ax = plt.subplots(figsize=(3,3))
    sns.countplot(x="Crime_Type", data=filtered_df, palette="pastel", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=False)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Victim Gender Distribution")
    fig, ax = plt.subplots(figsize=(3,3))
    sns.countplot(x="Victim_Gender", data=filtered_df, palette="pastel", ax=ax)
    st.pyplot(fig, use_container_width=False)

with col4:
    st.subheader("Clustering of Crimes")
    X = encoded_df[['City','Crime_Type','Weapon','Victim_Gender','Suspect_Gender','Arrest_Made']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    encoded_df['Cluster'] = kmeans.fit_predict(X)
    clustered_df = df.copy()
    clustered_df['Cluster'] = encoded_df['Cluster']
    st.dataframe(clustered_df[['Incident_ID','City','Crime_Type','Cluster']])
    fig, ax = plt.subplots(figsize=(3,3))
    sns.scatterplot(data=clustered_df, x="City", y="Crime_Type", hue="Cluster", palette="pastel", ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=False)
