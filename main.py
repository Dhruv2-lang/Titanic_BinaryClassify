import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

dataset=pd.read_csv("gender_submission.csv")
dataset=pd.read_csv("train.csv")
dataset=pd.read_csv("train.csv")

print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

print(dataset.isnull().sum())
print(dataset["Age"].describe())
print(dataset["Embarked"].value_counts())

dataset=dataset.drop("Cabin",axis=1)
print(dataset.columns)

print(dataset["Age"].median())
print(dataset["Embarked"].mode())
print(dataset.isnull().sum())

dataset["Age"] = dataset["Age"].fillna(dataset["Age"].median())
dataset["Embarked"] = dataset["Embarked"].fillna(dataset["Embarked"].mode()[0])
print(dataset.isnull().sum())

dataset = dataset.drop(["PassengerId", "Name", "Ticket"], axis=1)
print(dataset.columns)
print(dataset.head())

dataset = pd.get_dummies(dataset, columns=["Sex", "Embarked"], drop_first=True)
print(dataset.head())
print(dataset.columns)

X = dataset.drop("Survived", axis=1)
Y = dataset["Survived"]
print(X.head())
print()
print(Y.head())

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=1000)
model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)

print("Confusion Matrix:")
print(cm)
accuracy = accuracy_score(Y_test,Y_pred)

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print("Precision:", precision_score(Y_test, Y_pred))
print("Recall:", recall_score(Y_test, Y_pred))
print("F1 Score:", f1_score(Y_test, Y_pred))
