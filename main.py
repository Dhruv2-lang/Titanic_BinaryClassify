import pandas as pd

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