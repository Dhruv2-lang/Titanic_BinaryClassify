import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

dataset = pd.read_csv("train.csv")


original_dataset = dataset.copy()

print("First 5 rows:")
print(dataset.head())

print("\nDataset Shape:")
print(dataset.shape)

print("\nDataset Information:")
print(dataset.info())

print("\nDataset Description:")
print(dataset.describe())


print("\nMissing Values:")
print(dataset.isnull().sum())

print("\nAge Statistics:")
print(dataset["Age"].describe())

print("\nEmbarked Value Counts:")
print(dataset["Embarked"].value_counts())



dataset = dataset.drop("Cabin", axis=1)


dataset["Age"] = dataset["Age"].fillna(
    dataset["Age"].median()
)


dataset["Embarked"] = dataset["Embarked"].fillna(
    dataset["Embarked"].mode()[0]
)

print("\nMissing Values After Cleaning:")
print(dataset.isnull().sum())



dataset["Title"] = dataset["Name"].str.extract(
    r",\s*([^.]*)\."
)


rare_titles = [
    "Lady",
    "Countess",
    "Capt",
    "Col",
    "Don",
    "Dr",
    "Major",
    "Rev",
    "Sir",
    "Jonkheer",
    "Dona"
]

dataset["Title"] = dataset["Title"].replace(
    rare_titles,
    "Rare"
)


dataset["Title"] = dataset["Title"].replace({
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs"
})

print("\nTitle Counts:")
print(dataset["Title"].value_counts())



dataset["FamilySize"] = (
    dataset["SibSp"] +
    dataset["Parch"] +
    1
)


dataset["IsAlone"] = (
    dataset["FamilySize"] == 1
).astype(int)

print("\nFamily Features:")
print(
    dataset[
        ["SibSp", "Parch", "FamilySize", "IsAlone"]
    ].head(10)
)




dataset = dataset.drop(
    ["PassengerId", "Name", "Ticket"],
    axis=1
)




dataset = pd.get_dummies(
    dataset,
    columns=["Sex", "Embarked", "Title"],
    drop_first=True
)

print("\nFinal Columns:")
print(dataset.columns)

print("\nFinal Dataset:")
print(dataset.head())



X = dataset.drop("Survived", axis=1)

Y = dataset["Survived"]

print("\nX:")
print(X.head())

print("\nY:")
print(Y.head())



X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)




model = RandomForestClassifier(
    n_estimators=500,
    min_samples_split=5,
    random_state=42
)


cv_scores = cross_val_score(
    model,
    X_train,
    Y_train,
    cv=5,
    scoring="accuracy"
)

print("\nCross-Validation Scores:")
print(cv_scores)

print(
    "Mean CV Accuracy:",
    cv_scores.mean()
)




model.fit(
    X_train,
    Y_train
)



feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
)

feature_importance = feature_importance.sort_values(
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)




plt.figure(figsize=(10, 6))

feature_importance.plot(
    kind="bar"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "feature_importance.png"
)

plt.show()


Y_pred = model.predict(
    X_test
)



cm = confusion_matrix(
    Y_test,
    Y_pred
)

print("\nConfusion Matrix:")
print(cm)

print(
    "\nAccuracy:",
    accuracy_score(Y_test, Y_pred)
)

print(
    "Precision:",
    precision_score(Y_test, Y_pred)
)

print(
    "Recall:",
    recall_score(Y_test, Y_pred)
)

print(
    "F1 Score:",
    f1_score(Y_test, Y_pred)
)




gender_survival = (
    original_dataset
    .groupby("Sex")["Survived"]
    .mean()
)

print("\nSurvival Rate by Gender:")
print(gender_survival)

plt.figure(figsize=(6, 4))

gender_survival.plot(
    kind="bar"
)

plt.title(
    "Survival Rate by Gender"
)

plt.xlabel("Gender")
plt.ylabel("Survival Rate")

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "survival_by_gender.png"
)

plt.show()




class_survival = (
    original_dataset
    .groupby("Pclass")["Survived"]
    .mean()
)

print("\nSurvival Rate by Passenger Class:")
print(class_survival)

plt.figure(figsize=(6, 4))

class_survival.plot(
    kind="bar"
)

plt.title(
    "Survival Rate by Passenger Class"
)

plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "survival_by_class.png"
)

plt.show()



plt.figure(figsize=(8, 5))

plt.hist(
    original_dataset[
        original_dataset["Survived"] == 0
    ]["Age"].dropna(),
    bins=20,
    alpha=0.6,
    label="Did Not Survive"
)

plt.hist(
    original_dataset[
        original_dataset["Survived"] == 1
    ]["Age"].dropna(),
    bins=20,
    alpha=0.6,
    label="Survived"
)

plt.title(
    "Age Distribution by Survival"
)

plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.legend()

plt.tight_layout()

plt.savefig(
    "age_distribution.png"
)

plt.show()