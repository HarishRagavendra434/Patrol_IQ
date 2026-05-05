from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

def run_kmeans(df):
    X = df[['Latitude', 'Longitude']]
    X = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=5, random_state=42)
    df['kmeans_cluster'] = model.fit_predict(X)
    return df, model

def run_dbscan(df):
    X = df[['Latitude', 'Longitude']]
    model = DBSCAN(eps=0.01, min_samples=10)
    df['dbscan_cluster'] = model.fit_predict(X)
    return df

def run_hierarchical(df):
    X = df[['Latitude', 'Longitude']]
    model = AgglomerativeClustering(n_clusters=5)
    df['hier_cluster'] = model.fit_predict(X)
    return df
