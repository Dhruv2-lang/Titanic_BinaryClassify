# 🚢 Titanic Binary Classification

A machine learning project that predicts whether a passenger survived the Titanic disaster using passenger information such as age, gender, passenger class, fare, family size, and passenger title.

This project demonstrates a complete machine learning workflow, including data exploration, data cleaning, feature engineering, categorical encoding, model training, hyperparameter tuning, cross-validation, model evaluation, feature importance analysis, and data visualization.

---

## 📌 Project Overview

The Titanic Binary Classification project is a **supervised machine learning** project.

The goal is to predict whether a passenger survived the Titanic disaster based on information about that passenger.

The target variable is:

- `0` → Did not survive
- `1` → Survived

Since there are only two possible outcomes, this is a:

> **Binary Classification Problem**

The project focuses not only on achieving a good prediction accuracy, but also on understanding the complete machine learning process and the factors that influenced passenger survival.

---

## 🎯 Problem Statement

Given information about a passenger, such as:

- Age
- Sex
- Passenger class
- Fare
- Number of siblings/spouses
- Number of parents/children
- Port of embarkation
- Passenger title
- Family size

the objective is to predict whether the passenger survived the Titanic disaster.

The project also investigates which passenger characteristics had the strongest relationship with survival.

---

## 🤔 Why the Titanic Dataset?

The Titanic dataset is a useful machine learning learning project because it contains many of the challenges that occur in real-world datasets.

It includes:

- Missing values
- Numerical features
- Categorical features
- Irrelevant features
- Feature engineering opportunities
- Classification problems
- Relationships between multiple variables

It also contains meaningful real-world patterns.

For example, the historical concept of **"women and children first"** can be observed in the relationship between gender, age, and survival.

This makes the dataset useful for understanding not only how a machine learning model works, but also why the model makes certain predictions.

---

# 📊 Dataset

The project uses the Titanic dataset obtained from Kaggle.

The training dataset contains:

- **891 passengers**
- **12 original columns**

The original columns are:

| Column | Description |
|---|---|
| PassengerId | Unique passenger identification number |
| Survived | Target variable |
| Pclass | Passenger class |
| Name | Passenger name |
| Sex | Passenger gender |
| Age | Passenger age |
| SibSp | Number of siblings/spouses aboard |
| Parch | Number of parents/children aboard |
| Ticket | Ticket number |
| Fare | Passenger fare |
| Cabin | Cabin number |
| Embarked | Port of embarkation |

---

# 🧠 Machine Learning Concepts

This project demonstrates several important machine learning concepts.

## Supervised Learning

Supervised learning is a type of machine learning where the model learns from data that already contains the correct answers.

In this project:

```text
Passenger Information
        ↓
Machine Learning Model
        ↓
Survival Prediction

