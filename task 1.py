import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== Titanic Survival Prediction =====================
df = pd.read_csv('titanic.csv')
df.head()

# DATA INFO CHECK
df.info()
df.isnull().sum()


# DATA CLEANING
# Fill missing agewith mean
df['Age']= df['Age'].fillna(df['Age'].mean())

# Fill missing fare with mean
df['Fare']= df['Fare'].fillna(df['Fare'].mean())

# Fill missing embarked with mode
df['Embarked']= df['Embarked'].fillna(df['Embarked'].mode()[0])

# Convert embarked to numeric
df['Embarked']= df['Embarked'].map({
    'S':0,
    'C':1,
    'Q':2,
})

# Convert sex to numeric
df['Sex']= df['Sex'].map({'male':0, 'female':1})

# drop unnecessary columns
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
print(df.isnull().sum())
print(df.head())

# Features & target
# (features and target selection, where independent variablle (x) and dependent variable (y) are separate for model training.)
X = df.drop("Survived", axis=1)
Y = df['Survived']

# train test split
# this step does not  product a visible output. it split the dataset into training and testing sets stored in variables.
from sklearn.model_selection import train_test_split

X_train, X_train, Y_train, Y_train= train_test_split(X, Y, test_size=0.2, random_state=42)

# model(logistic regression)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# prediction
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# model evalution
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)


# Actual vs prediction survival heatmap

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(Y_test, Y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")


plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()




