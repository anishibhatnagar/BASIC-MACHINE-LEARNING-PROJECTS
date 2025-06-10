# logistic
import pandas as pd
import numpy as np
from google.colab import files

# Upload CSV file
uploaded = files.upload()

# Get the file name from the uploaded file
file_name = list(uploaded.keys())[0]

# Try reading the CSV file
try:
    data = pd.read_csv(file_name)
    print("Data loaded successfully!")
    print(data.head())  # Show the first few rows to confirm the data loaded
except Exception as e:
    print(f"Error loading the data: {e}")


X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Confusion Matrix & Accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)
print("Accuracy of the Model: {0}%".format(accuracy_score(y_test, y_pred) * 100))

age = int(input("Enter New Customer Age: "))
sal = int(input("Enter New Customer Salary: "))
newCust = [[age, sal]]
result = model.predict(sc.transform(newCust))

print(result)
if result == 1:
    print("Customer will Buy")
else:
    print("Customer won't Buy")
