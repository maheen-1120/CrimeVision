import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(layout="wide")
sns.set_theme(style="whitegrid")

uploaded_file = st.file_uploader("Upload your crime dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month

    crime_types = df['Crime Type'].unique().tolist() if 'Crime Type' in df.columns else []
    selected_crime = st.sidebar.selectbox("Select Crime Type", ["All"] + crime_types)
    if selected_crime != "All":
        filtered_df = df[df['Crime Type'] == selected_crime].copy()
    else:
        filtered_df = df.copy()

    if 'Victim Gender' in df.columns and 'Suspect Gender' in df.columns:
        plt.rcParams.update({'font.family':'Comic Sans MS'})
        fig, ax = plt.subplots(figsize=(9,6), dpi=300)
        gender_counts = pd.crosstab(filtered_df['Victim Gender'], filtered_df['Suspect Gender'])
        gender_counts.plot(kind="bar", ax=ax, width=0.7, color=["#FF69B4","#87CEFA","#9370DB"])
        ax.set_title("Victim vs Suspect Gender Distribution", fontsize=20, fontweight="bold")
        ax.set_xlabel("Victim Gender", fontsize=16, fontweight="bold")
        ax.set_ylabel("Count", fontsize=16, fontweight="bold")
        ax.legend(title="Suspect Gender", fontsize=12, title_fontsize=14, loc="upper right")
        col1, col2, col3 = st.columns([1,2,1])
        with col2: st.pyplot(fig, use_container_width=False)

    if 'City' in df.columns:
        plt.rcParams.update({'font.family':'Comic Sans MS'})
        fig, ax = plt.subplots(figsize=(9,6), dpi=300)
        city_counts = filtered_df['City'].value_counts()
        sns.barplot(x=city_counts.index, y=city_counts.values, ax=ax, palette=["#FF69B4","#87CEFA","#9370DB"])
        ax.set_title("Crimes per City", fontsize=20, fontweight="bold")
        ax.set_xlabel("City", fontsize=16, fontweight="bold")
        ax.set_ylabel("Count", fontsize=16, fontweight="bold")
        ax.tick_params(axis='x', rotation=45)
        col1, col2, col3 = st.columns([1,2,1])
        with col2: st.pyplot(fig, use_container_width=False)

        fig, ax = plt.subplots(figsize=(9,6), dpi=300)
        sns.lineplot(x=city_counts.index, y=city_counts.values, marker="o", linewidth=2.5, ax=ax, color="#FF69B4")
        ax.set_title("Crimes per City (Line Plot)", fontsize=20, fontweight="bold")
        ax.set_xlabel("City", fontsize=16, fontweight="bold")
        ax.set_ylabel("Count", fontsize=16, fontweight="bold")
        ax.tick_params(axis='x', rotation=45)
        col1, col2, col3 = st.columns([1,2,1])
        with col2: st.pyplot(fig, use_container_width=False)

    if 'Crime Type' in df.columns:
        plt.rcParams.update({'font.family':'Comic Sans MS'})
        fig, ax = plt.subplots(figsize=(9,6), dpi=300)
        crime_counts = filtered_df['Crime Type'].value_counts()
        sns.barplot(x=crime_counts.index, y=crime_counts.values, ax=ax, palette=["#9370DB","#87CEFA","#FF69B4"])
        ax.set_title("Crime Type Distribution", fontsize=20, fontweight="bold")
        ax.set_xlabel("Crime Type", fontsize=16, fontweight="bold")
        ax.set_ylabel("Count", fontsize=16, fontweight="bold")
        ax.tick_params(axis='x', rotation=45)
        col1, col2, col3 = st.columns([1,2,1])
        with col2: st.pyplot(fig, use_container_width=False)

        fig, ax = plt.subplots(figsize=(9,6), dpi=300)
        sns.lineplot(x=crime_counts.index, y=crime_counts.values, marker="o", linewidth=2.5, ax=ax, color="#9370DB")
        ax.set_title("Crime Type Distribution (Line Plot)", fontsize=20, fontweight="bold")
        ax.set_xlabel("Crime Type", fontsize=16, fontweight="bold")
        ax.set_ylabel("Count", fontsize=16, fontweight="bold")
        ax.tick_params(axis='x', rotation=45)
        col1, col2, col3 = st.columns([1,2,1])
        with col2: st.pyplot(fig, use_container_width=False)

    if 'Date' in df.columns:
        plt.rcParams.update({'font.family':'Comic Sans MS'})
        filtered_df['Year'] = filtered_df['Date'].dt.year
        filtered_df['Month'] = filtered_df['Date'].dt.month
        heat_data = filtered_df.groupby(['Month','Year']).size().unstack(fill_value=0)
        month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        fig, ax = plt.subplots(figsize=(12,8), dpi=300)
        cmap = LinearSegmentedColormap.from_list("custom_heat", ["#E6E6FA","#FFB6C1","#9370DB"])
        sns.heatmap(heat_data, annot=True, fmt="d", cmap=cmap,
                    annot_kws={'weight':'bold','fontsize':14,'fontfamily':'Comic Sans MS'},
                    cbar_kws={'label':'Crime Count','shrink':0.7}, ax=ax)
        ax.set_title("Crime Counts Heatmap (Month vs Year)", fontsize=22, fontweight='bold')
        ax.set_xlabel("Year", fontsize=18, fontweight='bold')
        ax.set_ylabel("Month", fontsize=18, fontweight='bold')
        ax.set_yticks(heat_data.index + 0.5)
        ax.set_yticklabels([month_labels[m-1] for m in heat_data.index], rotation=0, fontsize=14, fontweight='bold')
        plt.tight_layout()
        col1, col2, col3 = st.columns([1,2,1])
        with col2: st.pyplot(fig, use_container_width=False)
