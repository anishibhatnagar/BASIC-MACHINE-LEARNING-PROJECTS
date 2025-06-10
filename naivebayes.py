import pandas as pd
from google.colab import files  

uploaded = files.upload()

file_path = list(uploaded.keys())[0]
data = pd.read_csv(file_path)

# Separate data by class label
yes_data = data[data['Status'] == 1]
no_data = data[data['Status'] == 0]

# Prior probabilities
P_yes = len(yes_data) / len(data)
P_no = len(no_data) / len(data)

# Input features for prediction
X = {'Age': 45, 'Salary': 22000}

# Likelihood function
def calculate_likelihood(attribute, value, class_data):
    return len(class_data[class_data[attribute] == value]) / len(class_data)

# Likelihoods given 'yes' class
P_X_given_yes = (
    calculate_likelihood('Age', X['Age'], yes_data) *
    calculate_likelihood('Salary', X['Salary'], yes_data)
)

# Likelihoods given 'no' class
P_X_given_no = (
    calculate_likelihood('Age', X['Age'], no_data) *
    calculate_likelihood('Salary', X['Salary'], no_data)
)

# Posterior probabilities
P_yes_given_X = P_X_given_yes * P_yes
P_no_given_X = P_X_given_no * P_no

# Final prediction
prediction = 'yes' if P_yes_given_X > P_no_given_X else 'no'

# Output
print(f"P(Yes|X) = {P_yes_given_X:.4f}")
print(f"P(No|X) = {P_no_given_X:.4f}")
print(f"Prediction: The class label is '{prediction}' (customer {'buys' if prediction == 'yes' else 'does not buy'}).")
