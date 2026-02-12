from pathlib import Path

from CSVHandler import csvToMartixes
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# ML model
def model(params, x):
    a, b = params
    return a * x + b


current_dir = Path(__file__).resolve().parent

root_dir = current_dir.parent

DATA_PATH = root_dir / "players_22.csv"

x, y, x_norm, x_std = csvToMartixes(str(DATA_PATH))

# Split into sets
x_trainval, x_eval, y_trainval, y_eval = train_test_split(x_norm, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.25, random_state=42)


# Prepare design matrix
def prepareDesignMatrix(x_subset):      # (macierz projektowa)
    ones = np.ones_like(x_subset)
    return np.vstack([x_subset, ones])


X_train = prepareDesignMatrix(x_train)
Y_train = y_train[np.newaxis, :]

# Train the model (calculate parameters a, b)
XtX = X_train @ X_train.T
if np.linalg.det(XtX) == 0:
    raise ValueError("Matrix XtX is singular and cannot be inverted.")

T = np.linalg.inv(XtX) @ X_train @ Y_train.T
a, b = T.ravel()
print(f"Trained parameters: a = {a:.4f}, b = {b:.4f}")


# Evaluate MSE on datasets
for name, x_subset, y_subset in [
    ("Training", x_train, y_train),
    ("Validation", x_val, y_val),
    ("Evaluation", x_eval, y_eval)
]:
    y_pred = model((a, b), x_subset)
    mse = mean_squared_error(y_subset, y_pred)
    print(f"MSE on {name} set: {mse:.2f}")
