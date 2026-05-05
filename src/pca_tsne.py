from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

def apply_pca(df):
    X = df[['Latitude', 'Longitude', 'Hour', 'Month']]
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    df['pca1'] = components[:, 0]
    df['pca2'] = components[:, 1]
    return df, pca

def apply_tsne(df):
    X = df[['Latitude', 'Longitude', 'Hour', 'Month']]
    tsne = TSNE(n_components=2, random_state=42)
    comp = tsne.fit_transform(X)
    df['tsne1'] = comp[:, 0]
    df['tsne2'] = comp[:, 1]
    return df
