#find-s
from google.colab import files
import pandas as pd
import csv
uploaded = files.upload()
df=pd.read_csv("finds.csv")
print(df)
hypothesis = ['%', '%', '%', '%', '%', '%']
with open('finds.csv', newline='') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')
    print("The given training examples are:")
    training_data = []
    for row in reader:
        print(row)
        if row[-1].strip().lower() == 'yes':
            training_data.append(row)
if not training_data:
    print("No positive examples found!")
    exit()
print("\nThe positive examples are:")
for example in training_data:
    print(example)
print("\n")
num_attributes = len(training_data[0]) - 1
hypothesis = training_data[0][:num_attributes]
print("The initial hypothesis are :\n", hypothesis)
for i in range(1, len(training_data)):
    print(f"\nComparing with training example {i+1}: {training_data[i][:num_attributes]}")
    for j in range(num_attributes):
        if hypothesis[j] != training_data[i][j]:
            hypothesis[j] = '?'
    print("Updated hypothesis:", hypothesis)
print("\nThe maximally specific Find-S hypothesis for the given training examples is:")
print(hypothesis)
