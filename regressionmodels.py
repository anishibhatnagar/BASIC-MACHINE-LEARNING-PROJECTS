import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load Iris dataset
iris = load_iris()
X = iris.data[:, 3].reshape(-1, 1)  # Petal Width
y = iris.data[:, 0].reshape(-1, 1)  # Sepal Length

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Gradient Descent Implementation
def gradient_descent(X, y, lr=0.1, n_iter=1000):
    m = len(y)
    X_b = np.c_[np.ones((m, 1)), X]
    theta = np.random.randn(2, 1)
    for _ in range(n_iter):
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta -= lr * gradients
    return theta, X_b

theta_gd, X_train_b = gradient_descent(X_train, y_train)
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]
y_pred_gd_train, y_pred_gd_test = X_train_b.dot(theta_gd), X_test_b.dot(theta_gd)

# Other regression models
models = {
    "Least Squares": LinearRegression(),
    "Polynomial": LinearRegression(),
    "LASSO": Lasso(alpha=0.1),
    "Ridge": Ridge(alpha=1.0)
}

# Polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Store predictions
predictions = {"Gradient Descent": (y_pred_gd_train, y_pred_gd_test)}
for name, model in models.items():
    if name == "Polynomial":
        model.fit(X_train_poly, y_train)
        y_pred_train = model.predict(X_train_poly)
        y_pred_test = model.predict(X_test_poly)
    else:
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
    predictions[name] = (y_pred_train, y_pred_test)

# Plotting predictions for all models
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="black", label="Actual Data", alpha=0.6)

# Generate sorted X values for smooth line plotting
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_line_b = np.c_[np.ones((len(X_line), 1)), X_line]
X_line_poly = poly_features.transform(X_line)

# Predict and plot each model
for name, model in models.items():
    if name == "Polynomial":
        y_line = model.predict(X_line_poly)
    else:
        y_line = model.predict(X_line)
    plt.plot(X_line, y_line, label=name)

# Gradient Descent Line
y_line_gd = X_line_b.dot(theta_gd)
plt.plot(X_line, y_line_gd, label="Gradient Descent", linestyle='--')

plt.xlabel("Petal Width (cm)")
plt.ylabel("Sepal Length (cm)")
plt.title("Regression Model Comparisons on Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()
