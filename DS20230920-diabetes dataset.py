import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\GHS\Downloads\diabetes.csv")

#preview data
print(data.head())

#preview info
data.info()

#preview description
data.describe()

#check missing values
print(data.isnull().sum())

# Pregnancies
#percent of missing "Pregnancies"
print("Pregnancies : ")
print(data['Pregnancies'].value_counts())
sns.countplot(x='Pregnancies', data=data, palette='Set2')

# Glucose
#percent of missing "Glucose"
ax = data["Glucose"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='Glucose')
plt.show()


# BloodPressure
#percent of missing "BloodPressure"
ax = data["BloodPressure"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='BloodPressure')
plt.show()

# SkinThickness
#percent of missing "SkinThickness"
ax = data["SkinThickness"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='SkinThickness')
plt.show()

# Insulin
#percent of missing "Insulin"
ax = data["Insulin"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='Insulin')
plt.show()

# BMI
#percent of missing "BMI"
ax = data["BMI"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='BMI')
plt.show()

# DiabetesPedigreeFunction
ax = data["DiabetesPedigreeFunction"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='DiabetesPedigreeFunction')
plt.show()

# Age
ax = data["Age"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='Age')
plt.show()

# Outcome
print("Outcome : ")
print(data['Outcome'].value_counts())
sns.countplot(x='Outcome', data=data, palette='Set1')

