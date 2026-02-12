from CSVHandler import csvToMartixes
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def model(params, x):
    a, b = params
    return a * x + b


def gradientDescent(x, y, lr=0.01, epochs=1000):
    a, b = 0.0, 0.0  # initial weights
    n = len(x)
    for epoch in range(epochs):
        y_pred = model((a, b), x)
        error = y - y_pred

        grad_a = -2 * np.mean(x * error)
        grad_b = -2 * np.mean(error)

        a -= lr * grad_a
        b -= lr * grad_b

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: MSE = {mean_squared_error(y, y_pred):.4f}")

    return np.array([a, b])


# Load data
x, y, x_norm, x_std = csvToMartixes(r"C:\Users\krzys\Desktop\MSID\players_22.csv")

# Data split
x_trainval, x_eval, y_trainval, y_eval = train_test_split(x_norm, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.25, random_state=42)


# Train the model
start = time.time()
params = gradientDescent(x_train, y_train, lr=0.01, epochs=1000)
training_time = time.time() - start

print(f"Trained parameters: a = {params[0]:.4f}, b = {params[1]:.4f}")
print(f"Training time: {training_time:.4f} seconds")

# Evaluation
for name, x_subset, y_subset in [
    ("Training", x_train, y_train),
    ("Validation", x_val, y_val),
    ("Evaluation", x_eval, y_eval)
]:
    y_pred = model(params, x_subset)
    subset_mse = mean_squared_error(y_subset, y_pred)
    print(f"MSE on {name} set: {subset_mse:.2f}")
