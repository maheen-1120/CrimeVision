import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

sns.set_style("white")
st.set_page_config(layout="wide")
df = pd.read_csv("Crime_dataset.csv")

le = LabelEncoder()
encoded_df = df.copy()
for col in ['City','Crime_Type','Weapon','Victim_Gender','Suspect_Gender','Arrest_Made']:
    encoded_df[col] = le.fit_transform(encoded_df[col])

st.title("Crime Data Dashboard")
city = st.selectbox("Select City", ["All"] + sorted(df["City"].unique()))
crime_type = st.selectbox("Select Crime Type", ["All"] + sorted(df["Crime_Type"].unique()))

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

city_counts = df['City'].value_counts()
fig, ax = plt.subplots(figsize=(2.5,2))
ax.plot(city_counts.index, city_counts.values, marker='o', markersize=3, color='teal', linewidth=1.5)
ax.set_title("Crimes Per City", fontsize=7)
plt.xticks(rotation=45, fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
st.pyplot(fig)

crime_counts = filtered_df['Crime_Type'].value_counts()
fig, ax = plt.subplots(figsize=(2.5,2))
ax.plot(crime_counts.index, crime_counts.values, marker='o', markersize=3, color='orange', linewidth=1.5)
ax.set_title("Crime Type Distribution", fontsize=7)
plt.xticks(rotation=45, fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
st.pyplot(fig)

victim_counts = filtered_df['Victim_Gender'].value_counts()
suspect_counts = filtered_df['Suspect_Gender'].value_counts()
df_gender = pd.DataFrame({'Victim': victim_counts, 'Suspect': suspect_counts}).fillna(0)
fig, ax = plt.subplots(figsize=(2.5,2))
df_gender.plot(kind='bar', ax=ax, color=['#FF6F61', '#6B5B95'], width=0.5, edgecolor='black')
ax.set_title("Victim vs Suspect Gender", fontsize=7)
plt.xticks(rotation=0, fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()
st.pyplot(fig)

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df_time_series = df.groupby('Date').size()
    fig, ax = plt.subplots(figsize=(2.5,2))
    ax.plot(df_time_series.index, df_time_series.values, marker='o', markersize=2, color='purple', linewidth=1.5)
    ax.fill_between(df_time_series.index, df_time_series.values, alpha=0.15, color='purple')
    ax.set_title("Crime Counts Over Time", fontsize=7)
    ax.set_xlabel("Date", fontsize=5)
    ax.set_ylabel("Number of Crimes", fontsize=5)
    plt.xticks(rotation=45, fontsize=5)
    plt.yticks(fontsize=5)
    plt.tight_layout()
    st.pyplot(fig)
