# plik linear_regression_convergence_monitor rozszerza model regresji liniowej z wykorzystaniem gradient descent z II części projektu
# o liczenie błędu MSE co każdą epokę, a następnie rysowanie wykresu błedu na przestrzeni epok

# plik polynomial_regression_convergence rozszerza plik linear_regression_convergence_monitor o wykorzystanie PolynomialFeatures w celu rozbudowania modelu

from CSVHandler import csvToMartixes
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def model(params, x):
    w, b = params
    return x @ w + b


def gradientDescent(x_train, y_train, x_val, y_val, lr=0.01, epochs=1000):
    n_samples, n_features = x_train.shape
    w = np.zeros(n_features)
    b = 0.0
    train_errors = []
    val_errors = []

    for epoch in range(epochs):
        y_pred_train = x_train @ w + b
        error = y_train - y_pred_train

        grad_w = -2 * x_train.T @ error / n_samples
        grad_b = -2 * np.mean(error)

        w -= lr * grad_w
        b -= lr * grad_b

        train_mse = mean_squared_error(y_train, y_pred_train)
        y_pred_val = x_val @ w + b
        val_mse = mean_squared_error(y_val, y_pred_val)

        train_errors.append(train_mse)
        val_errors.append(val_mse)

    return (w, b), train_errors, val_errors


# Load data
x, y, x_norm, x_std = csvToMartixes(r"C:\Users\krzys\Desktop\MSID\CZĘŚĆ III\players_22.csv")

# Data split
x_trainval, x_eval, y_trainval, y_eval = train_test_split(x_norm, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.25, random_state=42)

pf = PolynomialFeatures(degree=2, include_bias=False)
x_train_poly = pf.fit_transform(x_train.reshape(-1, 1))
x_val_poly = pf.transform(x_val.reshape(-1, 1))
x_eval_poly = pf.transform(x_eval.reshape(-1, 1))

# Train the model
start = time.time()
params, train_errors, val_errors = gradientDescent(x_train_poly, y_train, x_val_poly, y_val, lr=0.01, epochs=1000)

# A graph
graph_from_range = 200
plt.plot(train_errors[graph_from_range:], label="Train MSE")
plt.plot(val_errors[graph_from_range:], label="Validation MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Convergence Plot")
plt.legend()
plt.grid(True)
plt.show()

training_time = time.time() - start

print(f"Training time: {training_time:.4f} seconds")

for i, train_error in enumerate(train_errors):
    print(f"epoch {i}: {train_error:.4f}")
