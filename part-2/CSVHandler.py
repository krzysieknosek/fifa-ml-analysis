import pandas as pd


def csvToMartixes(filename):
    df = pd.read_csv(filename, low_memory=False)
    df = df.dropna(subset=['value_eur', 'overall'])
    df = df[df['value_eur'] > 0]

    x = df['value_eur'].values.astype(float)
    y = df['overall'].values.astype(float)

    # Normalizacja X
    x_mean = x.mean()
    x_std = x.std()
    x_norm = (x - x_mean) / x_std

    return x, y, x_norm, x_std
