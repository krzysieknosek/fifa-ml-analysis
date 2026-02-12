import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


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


def csvToSampledMartixes(filename, over_under_sampled=None):
    df = pd.read_csv(filename, low_memory=False)
    df = df.dropna(subset=['value_eur', 'overall', 'club_position'])
    df = df[df['value_eur'] > 0]

    scaler = StandardScaler()
    df['value_eur_scaled'] = scaler.fit_transform(df[['value_eur']])

    le = LabelEncoder()
    df['position_encoded'] = le.fit_transform(df['club_position'])

    X = df[['value_eur_scaled']]
    y_target = df['position_encoded']

    if over_under_sampled == 'oversampled':
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y_target)

    elif over_under_sampled == 'undersampled':
        undersampler = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = undersampler.fit_resample(X, y_target)

    else:
        X_resampled, y_resampled = X, y_target

    mean_overall_by_position = df.groupby('position_encoded')['overall'].mean()
    y_overall = np.array([mean_overall_by_position[pos] for pos in y_resampled])

    x_original = X_resampled.to_numpy().flatten()
    x_norm = x_original
    x_std = 1.0

    return x_original, y_overall, x_norm, x_std