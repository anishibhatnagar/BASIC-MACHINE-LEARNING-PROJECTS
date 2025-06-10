#decision tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
data.fillna({'Age': data['Age'].median(), 'Embarked': data['Embarked'].mode()[0]}, inplace=True)
data.replace({'Sex': {'male': 0, 'female': 1}, 'Embarked': {'C': 0, 'Q': 1, 'S': 2}}, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Survived']), data['Survived'], test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=42)
model.fit(X_train, y_train)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}")
plt.figure(figsize=(16, 10))
plot_tree(model, filled=True, feature_names=X_train.columns, class_names=['Died', 'Survived'], fontsize=10)
plt.show()
