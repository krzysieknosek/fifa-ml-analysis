# linear_regression_convergence_monitor 2_raw rozszerza model regresji liniowej z wykorzystaniem gradient descent z II części projektu
# o liczenie błędu MSE co każdą epokę, a następnie rysowanie wykresu błedu na przestrzeni epok

# plik polynomial_regression_convergence rozszerza plik linear_regression_convergence_monitor o wykorzystanie PolynomialFeatures w celu rozbudowania modelu

from CSVHandler import csvToMartixes
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def model(params, x):
    a, b = params
    return a * x + b


def gradientDescent(x_train, y_train, x_val, y_val, lr=0.01, epochs=1000):
    a, b = 0.0, 0.0
    train_errors = []
    val_errors = []

    for epoch in range(epochs):
        y_pred_train = model((a, b), x_train)
        error = y_train - y_pred_train

        grad_a = -2 * np.mean(x_train * error)
        grad_b = -2 * np.mean(error)

        a -= lr * grad_a
        b -= lr * grad_b

        train_mse = mean_squared_error(y_train, y_pred_train)
        y_pred_val = model((a, b), x_val)
        val_mse = mean_squared_error(y_val, y_pred_val)

        train_errors.append(train_mse)
        val_errors.append(val_mse)

    return np.array([a, b]), train_errors, val_errors


# Load data
x, y, x_norm, x_std = csvToMartixes(r"C:\Users\krzys\Desktop\MSID\CZĘŚĆ III\players_22.csv")

# Data split
x_trainval, x_eval, y_trainval, y_eval = train_test_split(x_norm, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.25, random_state=42)


# Train the model
start = time.time()
params, train_errors, val_errors = gradientDescent(x_train, y_train, x_val, y_val, lr=0.01, epochs=1000)

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

print(f"Trained parameters: a = {params[0]:.4f}, b = {params[1]:.4f}")
print(f"Training time: {training_time:.4f} seconds")

for i, train_error in enumerate(train_errors):
    print(f"epoch {i}: {train_error:.4f}")
