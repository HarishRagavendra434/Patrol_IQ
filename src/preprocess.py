import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    df = df.dropna(subset=['Latitude', 'Longitude'])
    df['Date'] = pd.to_datetime(df['Date'])

    df['Hour'] = df['Date'].dt.hour
    df['Day'] = df['Date'].dt.day_name()
    df['Month'] = df['Date'].dt.month
    df['Weekend'] = df['Day'].isin(['Saturday', 'Sunday']).astype(int)

    return df
