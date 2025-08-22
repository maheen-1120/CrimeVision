import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

sns.set_style("whitegrid")

st.set_page_config(layout="wide")
df = pd.read_csv("Crime_dataset.csv")

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
df['Time'] = pd.to_datetime(df['Time'], format='%I.%M %p', errors='coerce').dt.time
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

le = LabelEncoder()
encoded_df = df.copy()
for col in ['City','Crime_Type','Weapon','Victim_Gender','Suspect_Gender','Arrest_Made']:
    encoded_df[col] = le.fit_transform(encoded_df[col])

st.title("Crime Data Dashboard")

city = st.selectbox("Select City", ["All"] + sorted(df["City"].unique()))
crime_type = st.selectbox("Select Crime Type", ["All"] + sorted(df["Crime_Type"].unique()))
year_filter = st.selectbox("Select Year", ["All"] + sorted(df["Year"].dropna().unique().astype(int)))
month_filter = st.selectbox("Select Month", ["All"] + list(range(1,13)))

filtered_df = df.copy()
if city != "All":
    filtered_df = filtered_df[filtered_df["City"] == city]
if crime_type != "All":
    filtered_df = filtered_df[filtered_df["Crime_Type"] == crime_type]
if year_filter != "All":
    filtered_df = filtered_df[filtered_df["Year"] == year_filter]
if month_filter != "All":
    filtered_df = filtered_df[filtered_df["Month"] == month_filter]

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

fig, ax = plt.subplots(figsize=(4,2.5))
sns.lineplot(x=city_counts := df['City'].value_counts().index,
             y=city_counts_values := df['City'].value_counts().values,
             marker='o', ax=ax, color='teal')
ax.set_title("Crimes Per City", fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(4,2.5))
sns.lineplot(x=crime_counts.index, y=crime_counts.values := filtered_df['Crime_Type'].value_counts().values,
             marker='o', ax=ax, color='orange')
ax.set_title("Crime Type Distribution", fontsize=10)
plt.xticks(rotation=45, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(4,2.5))
victim_counts = filtered_df['Victim_Gender'].value_counts()
suspect_counts = filtered_df['Suspect_Gender'].value_counts()
df_gender = pd.DataFrame({'Victim': victim_counts, 'Suspect': suspect_counts}).fillna(0)
df_gender.plot(kind='bar', ax=ax, color=['#FF6F61', '#6B5B95'], width=0.6)
ax.set_title("Victim vs Suspect Gender Distribution", fontsize=10)
plt.xticks(rotation=0, fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
st.pyplot(fig)

st.subheader("Crime Counts Over Time")
if not filtered_df.empty:
    df_time_series = filtered_df.groupby('Date').size()
    fig, ax = plt.subplots(figsize=(4,2.5))
    sns.lineplot(x=df_time_series.index, y=df_time_series.values, marker='o', ax=ax, color='purple')
    ax.set_title("Crime Counts Over Time", fontsize=10)
    ax.set_xlabel("Date", fontsize=8)
    ax.set_ylabel("Number of Crimes", fontsize=8)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

st.subheader("Date and Time Information")
st.write("Date Range:", filtered_df['Date'].min(), "to", filtered_df['Date'].max())
st.write(filtered_df[['Date','Time']])
