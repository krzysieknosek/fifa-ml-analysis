from CSVHandler import csvToSampledMartixes
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def model(params, x):
    a, b = params
    return a * x + b


def gradientDescent(x, y, lr=0.01, epochs=1000, l1=0.0, l2=0.0):
    a, b = 0.0, 0.0
    for epoch in range(epochs):
        y_pred = model((a, b), x)
        error = y - y_pred

        grad_a = -2 * np.mean(x * error)
        grad_b = -2 * np.mean(error)

        grad_a += l1 * np.sign(a)
        grad_a += 2 * l2 * a

        a -= lr * grad_a
        b -= lr * grad_b

    return np.array([a, b])


# Load data
x, y, x_norm, x_std = csvToSampledMartixes(r"C:\Users\krzys\Desktop\MSID\CZĘŚĆ III\players_22.csv",
                                           'oversampled')


# Split into sets
x_train, x_eval, y_train, y_eval = train_test_split(x_norm, y, test_size=0.2, random_state=42)

# K-Fold Cross-Validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)
fold = 1
mse_val_scores, mse_training_scores = [], []

for i, (train_index, test_index) in enumerate(kf.split(x_train)):
    print(f"\nFold {fold}")
    x_fold_train, x_fold_val = x_train[train_index], x_train[test_index]
    y_fold_train, y_fold_val = y_train[train_index], y_train[test_index]

    params = gradientDescent(x_fold_train, y_fold_train, lr=0.01, epochs=1000, l1=0.01, l2=0.01)

    y_val_pred = model(params, x_fold_val)
    y_training_pred = model(params, x_fold_train)
    training_mse = mean_squared_error(y_fold_train, y_training_pred)
    val_mse = mean_squared_error(y_fold_val, y_val_pred)
    mse_val_scores.append(val_mse)
    mse_training_scores.append(training_mse)

    print(f"Params: a = {params[0]:.4f}, b = {params[1]:.4f}")
    print(f"Training MSE: {training_mse:.2f}")
    print(f"Validation MSE: {val_mse:.2f}")
    fold += 1

print(f"\nAverage training MSE across folds: {np.mean(mse_training_scores):.2f}")
print(f"\nAverage validation MSE across folds: {np.mean(mse_val_scores):.2f}")


y_pred = model(params, x_eval)
mse = mean_squared_error(y_eval, y_pred)
print(f"\nMSE on Evaluation set: {mse:.2f}")