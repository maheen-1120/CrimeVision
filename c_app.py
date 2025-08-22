import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")
st.title("Crime Data Dashboard")

df = pd.read_csv("Crime_dataset.csv")

st.sidebar.header("Filters")
cities = st.sidebar.multiselect("Select Cities", df["City"].unique(), default=df["City"].unique())
crime_types = st.sidebar.multiselect("Select Crime Types", df["Crime_Type"].unique(), default=df["Crime_Type"].unique())
df_filtered = df[(df["City"].isin(cities)) & (df["Crime_Type"].isin(crime_types))]

st.subheader("Crimes per City")
fig, ax = plt.subplots(figsize=(4,3))
sns.countplot(data=df_filtered, x="City", ax=ax, palette="pastel")
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Crimes per Type")
fig, ax = plt.subplots(figsize=(4,3))
sns.countplot(data=df_filtered, x="Crime_Type", ax=ax, palette="pastel")
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Weapons Used")
fig, ax = plt.subplots(figsize=(4,3))
sns.countplot(data=df_filtered, x="Weapon", ax=ax, palette="pastel")
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("Victim Gender Distribution")
fig, ax = plt.subplots(figsize=(4,3))
sns.countplot(data=df_filtered, x="Victim_Gender", ax=ax, palette="pastel")
st.pyplot(fig)

st.subheader("Suspect Gender Distribution")
fig, ax = plt.subplots(figsize=(4,3))
sns.countplot(data=df_filtered, x="Suspect_Gender", ax=ax, palette="pastel")
st.pyplot(fig)

st.subheader("Arrests Made")
fig, ax = plt.subplots(figsize=(4,3))
sns.countplot(data=df_filtered, x="Arrest_Made", ax=ax, palette="pastel")
st.pyplot(fig)

st.subheader("Clustering Analysis")
df_encoded = df_filtered.copy()
le = LabelEncoder()
for col in ["City","Crime_Type","Weapon","Victim_Gender","Suspect_Gender","Arrest_Made"]:
    df_encoded[col] = le.fit_transform(df_encoded[col])
if len(df_encoded) > 1:
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    df_encoded["Cluster"] = kmeans.fit_predict(df_encoded[["City","Crime_Type","Weapon"]])
    fig, ax = plt.subplots(figsize=(4,3))
    sns.scatterplot(data=df_encoded, x="City", y="Crime_Type", hue="Cluster", palette="pastel", ax=ax)
    st.pyplot(fig)
else:
    st.write("Not enough data for clustering.")
