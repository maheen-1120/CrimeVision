import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

pastel_colors = ['#FFB6C1', '#ADD8E6', '#D8BFD8']

def plot_in_middle(fig):
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

city_counts = df['City'].value_counts()
fig, ax = plt.subplots(figsize=(3,2.5), dpi=150)
ax.plot(city_counts.index, city_counts.values, marker='o', markersize=5, color=pastel_colors[0], linewidth=2)
ax.set_title("Crimes Per City", fontsize=9)
plt.xticks(rotation=45, fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plot_in_middle(fig)

crime_counts = filtered_df['Crime_Type'].value_counts()
fig, ax = plt.subplots(figsize=(3,2.5), dpi=150)
ax.plot(crime_counts.index, crime_counts.values, marker='o', markersize=5, color=pastel_colors[1], linewidth=2)
ax.set_title("Crime Type Distribution", fontsize=9)
plt.xticks(rotation=45, fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
plot_in_middle(fig)

victim_counts = filtered_df['Victim_Gender'].value_counts()
suspect_counts = filtered_df['Suspect_Gender'].value_counts()
df_gender = pd.DataFrame({'Victim': victim_counts, 'Suspect': suspect_counts}).fillna(0)
fig, ax = plt.subplots(figsize=(3,2), dpi=150)
df_gender.plot(kind='bar', ax=ax, color=pastel_colors, width=0.5, edgecolor='black')
ax.set_title("Victim vs Suspect Gender", fontsize=9)
plt.xticks(rotation=0, fontsize=6)
plt.yticks(fontsize=6)
plt.ylim(0, max(df_gender.max())*1.1)
plt.tight_layout()
plot_in_middle(fig)

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    heat_data = df.groupby(['Month', 'Year']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(4,3), dpi=150)
    sns.heatmap(heat_data, annot=True, fmt="d", cmap="Pastel1", cbar_kws={'label': 'Crime Count'}, ax=ax)
    ax.set_title("Crime Counts Heatmap (Month vs Year)", fontsize=9)
    ax.set_xlabel("Year", fontsize=8)
    ax.set_ylabel("Month", fontsize=8)
    plt.tight_layout()
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)
