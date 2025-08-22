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

st.subheader("Clustering of Crimes")
X = encoded_df[['City','Crime_Type','Weapon','Victim_Gender','Suspect_Gender','Arrest_Made']]
kmeans = KMeans(n_clusters=3, random_state=42)
encoded_df['Cluster'] = kmeans.fit_predict(X)
clustered_df = df.copy()
clustered_df['Cluster'] = encoded_df['Cluster']
st.dataframe(clustered_df[['Incident_ID','City','Crime_Type','Cluster']])

st.subheader("Crime Data Visualizations")

fig, ax = plt.subplots(figsize=(6,3))
city_counts = df['City'].value_counts()
ax.plot(city_counts.index, city_counts.values, marker='o')
ax.set_title("Crimes Per City")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(6,3))
crime_counts = filtered_df['Crime_Type'].value_counts()
ax.plot(crime_counts.index, crime_counts.values, marker='o')
ax.set_title("Crime Type Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(6,3))
gender_counts = filtered_df['Victim_Gender'].value_counts()
ax.plot(gender_counts.index, gender_counts.values, marker='o')
ax.set_title("Victim Gender Distribution")
plt.tight_layout()
st.pyplot(fig)
