import pandas as pd

# Pipeline & processing imports
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Evaluation imports
from sklearn.metrics import accuracy_score, f1_score


# Pipeline
num_columns = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_columns),
])

# Import and split data from CSV file
df = pd.read_csv(r"C:\Users\krzys\Desktop\MSID\CZĘŚĆ III\players_22.csv", low_memory=False)

necessary_columns = num_columns + ['club_position']
df = df[necessary_columns].dropna(subset=['club_position'])

X = df.drop(columns='club_position')
y = df['club_position']


# Fit preprocessing on training data
X_prepared = preprocessor.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Balance data

over_under_sample = 'undersample'

if over_under_sample == 'oversample':
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_prepared, y_encoded)
elif over_under_sample == 'undersample':
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_prepared, y_encoded)
else:
    X_resampled, y_resampled = X_prepared, y_encoded

# Split into sets

X_trainval, X_eval, y_trainval, y_eval = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)


model = LogisticRegression(max_iter=1000)

# Training
model.fit(X_train, y_train)

# Evaluation
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
eval_pred = model.predict(X_eval)

train_acc = accuracy_score(y_train, train_pred)
val_acc = accuracy_score(y_val, val_pred)
eval_acc = accuracy_score(y_eval, eval_pred)

train_f1 = f1_score(y_train, train_pred, average='macro')
val_f1 = f1_score(y_val, val_pred, average='macro')
eval_f1 = f1_score(y_eval, eval_pred, average='macro')

print(f"Training Accuracy: {train_acc:.2f} | F1 Score: {train_f1:.2f}")
print(f"Validation Accuracy: {val_acc:.2f} | F1 Score: {val_f1:.2f}")
print(f"Evaluation Accuracy: {eval_acc:.2f} | F1 Score: {eval_f1:.2f}")