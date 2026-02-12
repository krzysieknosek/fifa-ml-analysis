from pathlib import Path

import pandas as pd

# Pipeline & processing imports
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Evaluation imports
from sklearn.metrics import mean_squared_error

# Pipeline
num_columns = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
cat_columns = ['club_position']


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_columns),
    ('cat', cat_pipeline, cat_columns)
])

# Importing and splitting data from CSV file
DATA_PATH = Path(__file__).resolve().parent.parent / "players_22.csv"
df = pd.read_csv(DATA_PATH, low_memory=False)

necessary_columns = num_columns + cat_columns + ['overall']
df = df[necessary_columns].dropna(subset=['overall'])

X = df.drop(columns='overall')
y = df['overall']

# Split into sets
X_trainval, X_eval, y_trainval, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)

# Fit preprocessing on training data
X_train_prepared = preprocessor.fit_transform(X_train)
X_val_prepared = preprocessor.transform(X_val)
X_eval_prepared = preprocessor.transform(X_eval)

# Machine learning models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "SVR": SVR()
}

for model_name, model in models.items():
    print(f"\n{model_name}")

    # Training
    model.fit(X_train_prepared, y_train)

    # Evaluation on training dataset
    train_pred = model.predict(X_train_prepared)
    train_mse = mean_squared_error(y_train, train_pred)
    print(f"Training MSE: {train_mse:.2f}")

    # Evaluation on validating dataset
    val_pred = model.predict(X_val_prepared)
    val_mse = mean_squared_error(y_val, val_pred)
    print(f"Validation MSE: {val_mse:.2f}")

    # Evaluation on evaluating dataset
    eval_pred = model.predict(X_eval_prepared)
    eval_mse = mean_squared_error(y_eval, eval_pred)
    print(f"Evaluation MSE: {eval_mse:.2f}")
