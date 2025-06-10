#random forest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
# Preprocess Data
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'C': 0, 'Q': 1, 'S': 2}}, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Survived']), data['Survived'], test_size=0.2, random_state=42)
# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=4, criterion='entropy', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), n_estimators=50, random_state=42),
    "Bagging": BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=4), n_estimators=100, random_state=42),
    "Soft Voting": VotingClassifier(estimators=[
        ('DT', DecisionTreeClassifier(max_depth=5, random_state=42)),
        ('RF', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('SVM', SVC(kernel='linear', probability=True, random_state=42))
    ], voting='soft')
}
# Train & Evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
