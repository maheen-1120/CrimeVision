import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans

st.title("Crime Data Analysis App")

uploaded_file = st.file_uploader("Upload Crime Dataset (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.sidebar.header("Filters")
    city_filter = st.sidebar.multiselect("Select City", options=df["City"].unique(), default=df["City"].unique())
    crime_filter = st.sidebar.multiselect("Select Crime Type", options=df["Crime_Type"].unique(), default=df["Crime_Type"].unique())
    weapon_filter = st.sidebar.multiselect("Select Weapon", options=df["Weapon"].unique(), default=df["Weapon"].unique())

    df = df[df["City"].isin(city_filter) & df["Crime_Type"].isin(crime_filter) & df["Weapon"].isin(weapon_filter)]

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Crime Count by City")
    if not df.empty:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="City", palette="pastel")
        st.pyplot(plt.gcf())

    st.subheader("Crime Type Distribution")
    if not df.empty:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x="Crime_Type", palette="pastel")
        st.pyplot(plt.gcf())

    st.subheader("Arrest Made Distribution")
    if not df.empty:
        plt.figure(figsize=(5, 4))
        sns.countplot(data=df, x="Arrest_Made", palette="pastel")
        st.pyplot(plt.gcf())

    st.subheader("Weapon Distribution")
    if not df.empty:
        plt.figure(figsize=(7, 5))
        sns.countplot(data=df, x="Weapon", palette="pastel")
        st.pyplot(plt.gcf())

    st.subheader("Predict Arrest Made")
    if "Arrest_Made" in df.columns and not df.empty:
        df["Arrest_Made"] = df["Arrest_Made"].map({"Yes": 1, "No": 0})
        df = df.dropna(subset=["Arrest_Made"])
        if not df.empty:
            features = df.drop(columns=["Incident_ID", "Date", "Time", "Arrest_Made"], errors="ignore")
            features = features.fillna("Unknown")
            le = LabelEncoder()
            for col in features.columns:
                if features[col].dtype == "object":
                    features[col] = le.fit_transform(features[col])
            X = features
            y = df["Arrest_Made"]
            if len(y.unique()) > 1:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                st.write(f"Model Accuracy: {acc:.2f}")
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Pastel1", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
                st.pyplot(plt.gcf())
            else:
                st.write("Not enough class variety for training.")

    st.subheader("Clustering Crime Data")
    features = df.drop(columns=["Incident_ID", "Date", "Time", "Arrest_Made"], errors="ignore")
    features = features.fillna("Unknown")
    le = LabelEncoder()
    for col in features.columns:
        if features[col].dtype == "object":
            features[col] = le.fit_transform(features[col])
    if len(features) > 1:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(features)
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x="Cluster", y=features.columns[0], hue="Cluster", palette="pastel")
        st.pyplot(plt.gcf())
        st.write(df[["Incident_ID", "Cluster"]].head())
