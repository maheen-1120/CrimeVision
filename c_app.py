import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rcParams

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

pink_shades = ['#FF6699', '#FF99AA', '#FFB3C6']
blue_shades = ['#3399FF', '#66B2FF', '#99CCFF']
purple_shades = ['#9966FF', '#B399FF', '#CFA3FF']
line_colors = ['#FF6699', '#3399FF', '#9966FF', '#FFB366', '#66FFB3', '#B366FF']

rcParams['font.family'] = 'Comic Sans MS'

def plot_in_middle(fig):
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

city_counts = filtered_df['City'].value_counts()
fig, ax = plt.subplots(figsize=(12,7), dpi=300)
for i, city_name in enumerate(city_counts.index):
    ax.scatter(city_name, city_counts[city_name], color=line_colors[i % len(line_colors)], s=200, label=city_name)
ax.plot(city_counts.index, city_counts.values, color='grey', linewidth=3, alpha=0.7)
ax.set_title("Crimes Per City", fontsize=22, fontweight='bold')
ax.set_xlabel("City", fontsize=18, fontweight='bold')
ax.set_ylabel("Count", fontsize=18, fontweight='bold')
plt.xticks(rotation=45, fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
ax.legend(fontsize=12, prop={'family':'Comic Sans MS','weight':'bold'})
plt.tight_layout()
plot_in_middle(fig)
st.markdown("<br>", unsafe_allow_html=True)

crime_counts = filtered_df['Crime_Type'].value_counts()
fig, ax = plt.subplots(figsize=(12,7), dpi=300)
for i, crime_name in enumerate(crime_counts.index):
    ax.scatter(crime_name, crime_counts[crime_name], color=line_colors[i % len(line_colors)], s=200, label=crime_name)
ax.plot(crime_counts.index, crime_counts.values, color='grey', linewidth=3, alpha=0.7)
ax.set_title("Crime Type Distribution", fontsize=22, fontweight='bold')
ax.set_xlabel("Crime Type", fontsize=18, fontweight='bold')
ax.set_ylabel("Count", fontsize=18, fontweight='bold')
plt.xticks(rotation=45, fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
ax.legend(fontsize=12, prop={'family':'Comic Sans MS','weight':'bold'})
plt.tight_layout()
plot_in_middle(fig)
st.markdown("<br>", unsafe_allow_html=True)

victim_counts = filtered_df['Victim_Gender'].value_counts()
suspect_counts = filtered_df['Suspect_Gender'].value_counts()
df_gender = pd.DataFrame({'Victim': victim_counts, 'Suspect': suspect_counts}).fillna(0)
fig, ax = plt.subplots(figsize=(12,6), dpi=300)
df_gender.plot(kind='bar', ax=ax, color=['#FF6699', '#9966FF'], width=0.5, edgecolor='black')
ax.set_title("Victim vs Suspect Gender", fontsize=22, fontweight='bold')
ax.set_xlabel("Gender", fontsize=18, fontweight='bold')
ax.set_ylabel("Count", fontsize=18, fontweight='bold')
plt.xticks(rotation=0, fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.ylim(0, max(df_gender.max())*1.1)
plt.legend(fontsize=12, prop={'family':'Comic Sans MS','weight':'bold'}, loc='upper right')
plt.tight_layout()
plot_in_middle(fig)
st.markdown("<br>", unsafe_allow_html=True)

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    heat_data = filtered_df.groupby(['Month', 'Year']).size().unstack(fill_value=0)
    month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig, ax = plt.subplots(figsize=(12,8), dpi=300)
    cmap = LinearSegmentedColormap.from_list("custom_heat", ["#FFF0F5", "#FFB6C1", "#FF69B4"])  
    sns.heatmap(heat_data, annot=True, fmt="d", cmap=cmap,
                annot_kws={'weight':'bold', 'fontsize':13, 'fontfamily':'Comic Sans MS'},
                cbar_kws={'label': 'Crime Count', 'shrink':0.7}, ax=ax)
    ax.set_title("Crime Counts Heatmap (Month vs Year)", fontsize=22, fontweight='bold')
    ax.set_xlabel("Year", fontsize=18, fontweight='bold')
    ax.set_ylabel("Month", fontsize=18, fontweight='bold')
    ax.set_yticks(heat_data.index + 0.5)
    ax.set_yticklabels([month_labels[m-1] for m in heat_data.index], rotation=0, fontsize=14, fontweight='bold')
    plt.tight_layout()
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)
