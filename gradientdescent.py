import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load inbuilt Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# We will predict 'sepal length (cm)' using 'petal width (cm)'
X_data = tf.constant(df['petal width (cm)'].values.astype(np.float32))
y_data = tf.constant(df['sepal length (cm)'].values.astype(np.float32))

# Initialize weights and bias
w = tf.Variable(0.0)
b = tf.Variable(0.0)

# Hyperparameters
learning_rate = 0.1
epochs = 1000

# Gradient Descent Training Loop
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = w * X_data + b
        loss = tf.reduce_mean(tf.square(y_data - y_pred))  # MSE

    gradients = tape.gradient(loss, [w, b])
    w.assign_sub(learning_rate * gradients[0])
    b.assign_sub(learning_rate * gradients[1])

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d}: Loss = {loss:.4f}, w = {w.numpy():.4f}, b = {b.numpy():.4f}")

# Plot the data and fitted line
plt.figure(figsize=(8, 5))
plt.scatter(X_data, y_data, label="Actual Data", color="blue")
plt.plot(X_data, w.numpy() * X_data + b.numpy(), color="red", label="Fitted Line")
plt.xlabel("Petal Width (cm)")
plt.ylabel("Sepal Length (cm)")
plt.title("Gradient Descent Linear Regression (Iris Dataset)")
plt.legend()
plt.grid(True)
plt.show()

# Final model
print(f"\nFinal Model: Sepal Length = {w.numpy():.4f} × Petal Width + {b.numpy():.4f}")
