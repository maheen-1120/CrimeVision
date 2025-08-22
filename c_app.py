import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap

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

pink_shades = ['#FF99AA', '#FFB3C6', '#FFCCE5']
blue_shades = ['#99CCFF', '#66B2FF', '#3399FF']
purple_shades = ['#CFA3FF', '#B399FF', '#A366FF']

def plot_in_middle(fig):
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

city_counts = df['City'].value_counts()
fig, ax = plt.subplots(figsize=(5.5,4), dpi=160)
ax.plot(city_counts.index, city_counts.values, marker='o', markersize=6, color=pink_shades[0], linewidth=2)
ax.set_title("Crimes Per City", fontsize=11)
plt.xticks(rotation=45, fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plot_in_middle(fig)
st.markdown("<br>", unsafe_allow_html=True)

crime_counts = filtered_df['Crime_Type'].value_counts()
fig, ax = plt.subplots(figsize=(5.5,4), dpi=160)
ax.plot(crime_counts.index, crime_counts.values, marker='o', markersize=6, color=blue_shades[0], linewidth=2)
ax.set_title("Crime Type Distribution", fontsize=11)
plt.xticks(rotation=45, fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plot_in_middle(fig)
st.markdown("<br>", unsafe_allow_html=True)

victim_counts = filtered_df['Victim_Gender'].value_counts()
suspect_counts = filtered_df['Suspect_Gender'].value_counts()
df_gender = pd.DataFrame({'Victim': victim_counts, 'Suspect': suspect_counts}).fillna(0)
fig, ax = plt.subplots(figsize=(5.5,3.5), dpi=160)
df_gender.plot(kind='bar', ax=ax, color=['#FF99AA', '#CFA3FF'], width=0.5, edgecolor='black')
ax.set_title("Victim vs Suspect Gender", fontsize=11)
plt.xticks(rotation=0, fontsize=8)
plt.yticks(fontsize=8)
plt.ylim(0, max(df_gender.max())*1.1)
plt.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plot_in_middle(fig)
st.markdown("<br>", unsafe_allow_html=True)

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    heat_data = df.groupby(['Month', 'Year']).size().unstack(fill_value=0)
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig, ax = plt.subplots(figsize=(6,5), dpi=160)
    cmap = LinearSegmentedColormap.from_list("custom_pastel", ["#FFFFFF", "#FF99AA", "#FF4D6D"])  
    sns.heatmap(heat_data, annot=True, fmt="d", cmap=cmap,
                cbar_kws={'label': 'Crime Count', 'shrink':0.7}, ax=ax)
    ax.set_title("Crime Counts Heatmap (Month vs Year)", fontsize=12)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("Month", fontsize=10)
    ax.set_yticks(heat_data.index + 0.5)
    ax.set_yticklabels([month_labels[m-1] for m in heat_data.index], rotation=0, fontsize=9)
    plt.tight_layout()
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)
