import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="PatrolIQ", layout="wide")

st.title("PatrolIQ - Smart Safety Analytics")

file = st.file_uploader("Upload Crime Dataset", type=["csv"])

if file is not None:

    df = pd.read_csv(file)
    df = df.sample(20000)

    df = df.dropna(subset=['Latitude', 'Longitude'])
    df['Date'] = pd.to_datetime(df['Date'])

    df['Hour'] = df['Date'].dt.hour
    df['Month'] = df['Date'].dt.month

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Crimes", len(df))
    col2.metric("Total Arrests", int(df['Arrest'].sum()))
    col3.metric("Crime Types", df['Primary Type'].nunique())

    X = df[['Latitude', 'Longitude']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=5, random_state=42)
    df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

    df_small = df.sample(10000)

    dbscan = DBSCAN(eps=0.01, min_samples=10)
    df_small['dbscan_cluster'] = dbscan.fit_predict(df_small[['Latitude','Longitude']])

    st.subheader("DBSCAN Clusters")

    counts = df_small['dbscan_cluster'].value_counts().reset_index()
    counts.columns = ['Cluster', 'Count']

    fig_db = px.bar(counts, x='Cluster', y='Count')
    st.plotly_chart(fig_db, use_container_width=True)
    hier = AgglomerativeClustering(n_clusters=5)
    df['hier_cluster'] = hier.fit_predict(X_scaled)

    st.subheader("KMeans Clusters")

    fig_kmeans = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color="kmeans_cluster",
        zoom=10,
        height=500
    )
    fig_kmeans.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(fig_kmeans, use_container_width=True)

    st.subheader("DBSCAN Clusters")
    st.subheader("Hierarchical Clusters")

    fig_hier = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color="hier_cluster",
        zoom=10,
        height=500
    )
    fig_hier.update_layout(mapbox_style="carto-positron")
    st.plotly_chart(fig_hier, use_container_width=True)

    st.subheader("Crime Distribution by Hour")
    fig_hour = px.histogram(df, x="Hour", nbins=24)
    st.plotly_chart(fig_hour, use_container_width=True)

    st.subheader("Crime Trend by Month")
    fig_month = px.histogram(df, x="Month")
    st.plotly_chart(fig_month, use_container_width=True)

    st.subheader("Top Crime Types")
    top_crime = df['Primary Type'].value_counts().head(10)
    fig_bar = px.bar(x=top_crime.values, y=top_crime.index, orientation='h')
    st.plotly_chart(fig_bar, use_container_width=True)

    features = df[['Latitude', 'Longitude', 'Hour', 'Month']]
    scaled_features = StandardScaler().fit_transform(features)

    pca = PCA(n_components=2)
    comp = pca.fit_transform(scaled_features)

    df['pca1'] = comp[:, 0]
    df['pca2'] = comp[:, 1]

    st.subheader("PCA Visualization")
    fig_pca = px.scatter(df, x='pca1', y='pca2', color='kmeans_cluster')
    st.plotly_chart(fig_pca, use_container_width=True)

    st.subheader("Raw Data")
    st.dataframe(df.head(100))

else:
    st.write("Upload dataset to continue")