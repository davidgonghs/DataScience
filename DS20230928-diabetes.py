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

# get total number of rows
print("Total number of rows : {0}".format(len(diabetes[diabetes['SkinThickness'] == 0])))


# check Glucose value is 0
diabetes[diabetes['Glucose'] == 0]
# check BloodPressure value is 0
diabetes[diabetes['BloodPressure'] == 0]
# check SkinThickness value is 0
diabetes[diabetes['SkinThickness'] == 0]
# check Insulin value is 0
diabetes[diabetes['Insulin'] == 0]
# check BMI value is 0
diabetes[diabetes['BMI'] == 0]

# remove which value is 0 that row
diabetes['Glucose'] = diabetes['Glucose'].replace(0, np.NaN)
diabetes['BloodPressure'] = diabetes['BloodPressure'].replace(0, np.NaN)
diabetes['SkinThickness'] = diabetes['SkinThickness'].replace(0, np.NaN)
diabetes['Insulin'] = diabetes['Insulin'].replace(0, np.NaN)
diabetes['BMI'] = diabetes['BMI'].replace(0, np.NaN)

# delete all the row which value is NaN
diabetes.dropna(inplace=True)


# check number of row again
print("Total number of rows : {0}".format(len(diabetes)))




# Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome

# Pregnancies
diabetes['Pregnancies'].unique()

print("Pregnancies : ")
print(diabetes['Pregnancies'].value_counts())
sns.countplot(x='Pregnancies', data=diabetes, palette='Set2')


# Glucose
diabetes['Glucose'].unique()

# find the mean value for "Glucose"
print("Glucose : ")
ax = diabetes["Glucose"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='Glucose')
plt.show()

# classify "Glucose" base on 140, if greater than 140, then it is 1, otherwise 0, if smaller than 70
# during 70-140 is normal
# during <70 is hypoglycemia
# during >140 is hyperglycemia
diabetes.loc[diabetes['Glucose'] <= 70, 'Glucose'] = 0
diabetes.loc[(diabetes['Glucose'] > 70) & (diabetes['Glucose'] < 140), 'Glucose'] = 1
diabetes.loc[diabetes['Glucose'] >= 140, 'Glucose'] = 2

print("Glucose : ")
print(diabetes['Glucose'].value_counts())
# set sns title
sns.countplot(x='Glucose', data=diabetes, palette='Set2')








# BloodPressure
print("BloodPressure : ")
ax = diabetes["BloodPressure"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='BloodPressure')
plt.show()

# classify "BloodPressure" base on <90, 90-130,140-159

# SkinThickness
ax = diabetes["SkinThickness"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='SkinThickness')
plt.show()

# Insulin
ax = diabetes["Insulin"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='Insulin')
plt.show()

# BMI
ax = diabetes["BMI"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='BMI')
plt.show()

# DiabetesPedigreeFunction
ax = diabetes["DiabetesPedigreeFunction"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='DiabetesPedigreeFunction')
plt.show()

# Age
ax = diabetes["Age"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='Age')
plt.show()

# Outcome
print("Outcome : ")
print(diabetes['Outcome'].value_counts())
sns.countplot(x='Outcome', data=diabetes, palette='Set2')


# Create a heatmap to visualize the correlation matrix
sns.heatmap(diabetes.corr(), annot=True, cmap='RdYlGn')
plt.show()

