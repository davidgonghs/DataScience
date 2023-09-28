import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

diabetes = pd.read_csv("dataset/diabetes.csv")

#preview data
diabetes.head()

#preview info
diabetes.info()


#preview description
diabetes.describe()

print(diabetes.isnull().sum())

# Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome

# Pregnancies
#percent of missing "Pregnancies"
print("Pregnancies : ")
print(diabetes['Pregnancies'].value_counts())
sns.countplot(x='Pregnancies', data=diabetes, palette='Set2')

# Glucose
#percent of missing "Glucose"
ax = diabetes["Glucose"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='Glucose')
plt.show()

# BloodPressure
#percent of missing "BloodPressure"
ax = diabetes["BloodPressure"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='BloodPressure')
plt.show()

# SkinThickness
#percent of missing "SkinThickness"
ax = diabetes["SkinThickness"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='SkinThickness')
plt.show()

# Insulin
#percent of missing "Insulin"
ax = diabetes["Insulin"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='Insulin')
plt.show()

# BMI
#percent of missing "BMI"
ax = diabetes["BMI"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='BMI')
plt.show()

# DiabetesPedigreeFunction
ax = diabetes["DiabetesPedigreeFunction"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='DiabetesPedigreeFunction')
plt.show()

# Age
#percent of missing "Age"
ax = diabetes["Age"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='Age')
plt.show()

# Outcome
#percent of missing "Outcome"
print("Outcome : ")
print(diabetes['Outcome'].value_counts())
sns.countplot(x='Outcome', data=diabetes, palette='Set2')


# Create a heatmap to visualize the correlation matrix
sns.heatmap(diabetes.corr(), annot=True, cmap='RdYlGn')
plt.show()

