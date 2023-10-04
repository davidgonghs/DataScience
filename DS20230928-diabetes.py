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
totalRows = len(diabetes)
print("Total number of rows : {0}".format(len(diabetes)))

# check how many 0 value in each column
clucoseMissingNumber = len(diabetes[diabetes['Glucose'] == 0])
print("Glucose Missing number: {0}".format(clucoseMissingNumber))
print("Glucose Missing percentage: {0}".format(clucoseMissingNumber/totalRows))

# check how many 0 value in BloodPressure
BloodPressureMissingNumber = len(diabetes[diabetes['BloodPressure'] == 0])
print("BloodPressure Missing number: {0}".format(BloodPressureMissingNumber))
print("BloodPressure Missing percentage: {0}".format(BloodPressureMissingNumber/totalRows))

# check how many 0 value in SkinThickness
SkinThicknessMissingNumber = len(diabetes[diabetes['SkinThickness'] == 0])
print("SkinThickness Missing number: {0}".format(SkinThicknessMissingNumber))
print("SkinThickness Missing percentage: {0}".format(SkinThicknessMissingNumber/totalRows))

# check how many 0 value in Insulin
InsulinMissingNumber = len(diabetes[diabetes['Insulin'] == 0])
print("Insulin Missing number: {0}".format(InsulinMissingNumber))
print("Insulin Missing percentage: {0}".format(InsulinMissingNumber/totalRows))

# check how many 0 value in BMI
BMIMissingNumber = len(diabetes[diabetes['BMI'] == 0])
print("BMI Missing number: {0}".format(BMIMissingNumber))
print("BMI Missing percentage: {0}".format(BMIMissingNumber/totalRows))


print("SkinThickness Missing number: {0}".format(len(diabetes[diabetes['SkinThickness'] == 0])))
print("Insulin Missing number: {0}".format(len(diabetes[diabetes['Insulin'] == 0])))
print("BMI Missing number: {0}".format(len(diabetes[diabetes['BMI'] == 0])))

# get total number of rows
print("DiabetesPedigreeFunction Missing number: {0}".format(len(diabetes[diabetes['DiabetesPedigreeFunction'] == 0])))



# delete all the row which value is NaN
diabetes.dropna(inplace=True)

# get percentage of missing value
print("Glucose Missing number: {0}".format(len(diabetes[diabetes['Glucose'] == 0])))
print("BloodPressure Missing number: {0}".format(len(diabetes[diabetes['BloodPressure'] == 0])))
print("SkinThickness Missing number: {0}".format(len(diabetes[diabetes['SkinThickness'] == 0])))
print("Insulin Missing number: {0}".format(len(diabetes[diabetes['Insulin'] == 0])))
print("BMI Missing number: {0}".format(len(diabetes[diabetes['BMI'] == 0])))

# remove column "Insulin" and "SkinThickness"
diabetes.drop(['Insulin', 'SkinThickness'], axis=1, inplace=True)


# remove all the row which value is 0
diabetes = diabetes[(diabetes[['Glucose','BloodPressure','BMI']] != 0).all(axis=1)]




# Handling Outliers
plt.figure(figsize=(12, 5))
# check outliers
# Glucose
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
sns.boxplot(y=diabetes['Pregnancies'])
plt.title('Pregnancies Boxplot')

plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
sns.countplot(x='Pregnancies', data=diabetes, palette='Set2')
plt.title('Pregnancies Countplot')

# clean data for pregnancies remove over than 13
diabetes = diabetes[diabetes['Pregnancies'] < 13]



# BloodPressure
sns.boxplot(x=diabetes['BloodPressure'])
# get outliers number of BloodPressure
Q1 = diabetes['BloodPressure'].quantile(0.25)
Q3 = diabetes['BloodPressure'].quantile(0.75)
IQR = Q3 - Q1
print("BloodPressure outliers number : {0}".format(len(diabetes[(diabetes['BloodPressure'] < (Q1 - 1.5 * IQR)) | (diabetes['BloodPressure'] > (Q3 + 1.5 * IQR))])))

# BMI
sns.boxplot(x=diabetes['BMI'])
# get outliers number of BMI
Q1 = diabetes['BMI'].quantile(0.25)
Q3 = diabetes['BMI'].quantile(0.75)
IQR = Q3 - Q1
print("BMI outliers number : {0}".format(len(diabetes[(diabetes['BMI'] < (Q1 - 1.5 * IQR)) | (diabetes['BMI'] > (Q3 + 1.5 * IQR))])))

# get quantile 0.75
diabetes['BMI'].quantile(0.75)

# remove BMI over than quantile 0.75
diabetes = diabetes[diabetes['BMI'] < 36.5]


# age
sns.boxplot(x=diabetes['Age'])
# get outliers number of age
Q1 = diabetes['Age'].quantile(0.25)
Q3 = diabetes['Age'].quantile(0.75)
IQR = Q3 - Q1
print("Age outliers number : {0}".format(len(diabetes[(diabetes['Age'] < (Q1 - 1.5 * IQR)) | (diabetes['Age'] > (Q3 + 1.5 * IQR))])))


# remove age over than 41
diabetes = diabetes[diabetes['Age'] < 41]

# Pregnancies
sns.boxplot(x=diabetes['Pregnancies'])





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


# create a data frame for the labels
data = diabetes.Dataframe({'X':diabetes['Pregnancies','Glucose','BloodPressure','BMI','DiabetesPedigreeFunction','Age','Outcome'],'Y':{diabetes['Outcome','Age','DiabetesPedigreeFunction','BMI','BloodPressure','Glucose','Pregnancies']}})


# create a line plot for the labels
plt.plot(data['X'], data['Y'], kind='line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Line Plot')
plt.show()
