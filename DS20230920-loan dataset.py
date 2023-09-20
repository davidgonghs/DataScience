import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\GHS\Downloads\Loan.csv")

#preview data
print(data.head())


#preview info
data.info()


#preview description
data.describe()

#check missing values
print(data.isnull().sum())

# Gender
#percent of missing "Gender"
print('Percent of missing "Gender" records is %.2f%%' %((data['Gender'].isnull().sum()/data.shape[0])*100))
print("Number of people who take a loan group by gender : ")
print(data['Gender'].value_counts())
sns.countplot(x='Gender', data=data, palette='Set2')


# married
#percent of missing "married"
print('Percent of missing "Married" records is %.2f%%' %((data['Married'].isnull().sum()/data.shape[0])*100))
print("Number of people who take a loan group by married : ")
print(data['Married'].value_counts())
sns.countplot(x='Married', data=data, palette='Set1')

# dependent
#percent of missing "Dependents"
print('Percent of missing "Dependents" records is %.2f%%' %((data['Dependents'].isnull().sum()/data.shape[0])*100))
print("Number of people who take a loan group by dependents : ")
print(data['Dependents'].value_counts())
sns.countplot(x='Dependents', data=data, palette='Set3')

# self_employed
#percent of missing "Self_Employed"
print('Percent of missing "Self_Employed" records is %.2f%%' %((data['Self_Employed'].isnull().sum()/data.shape[0])*100))
print("Number of people who take a loan group by self_employed : ")
print(data['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed', data=data, palette='Set2')

#LoanAmount
#percent of missing "LoanAmount"
print('Percent of missing "LoanAmount" records is %.2f%%' %((data['LoanAmount'].isnull().sum()/data.shape[0])*100))
print("Number of people who take a loan group by LoanAmount : ")
print(data['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount', data=data, palette='Set1')

# use hist
ax = data["LoanAmount"].hist(density=True, stacked=True, color='teal', alpha=0.6)
ax.set(xlabel='Loan Amount')
plt.show()

# group loan amount
bins = [0, 100, 200, 700]
group = ['Low', 'Average', 'High']
data['LoanAmount_bin'] = pd.cut(data['LoanAmount'], bins, labels=group)
print(data['LoanAmount_bin'].value_counts())
sns.countplot(x='LoanAmount_bin', data=data, palette='Set3')

#loan_amount_term
#percent of missing "Loan_Amount_Term"
print('Percent of missing "Loan_Amount_Term" records is %.2f%%' %((data['Loan_Amount_Term'].isnull().sum()/data.shape[0])*100))
print("Number of people who take a loan group by Loan_Amount_Term : ")
print(data['Loan_Amount_Term'].value_counts())
sns.countplot(x='Loan_Amount_Term', data=data, palette='Set2')

# credit_history
#percent of missing "Credit_History"
print('Percent of missing "Credit_History" records is %.2f%%' %((data['Credit_History'].isnull().sum()/data.shape[0])*100))
print("Number of people who take a loan group by Credit_History : ")
print(data['Credit_History'].value_counts())
sns.countplot(x='Credit_History', data=data, palette='Set1')


# Assuming you have a DataFrame 'data' with numerical columns
numerical_columns = data.select_dtypes(include=['int64', 'float64'])

# Calculate the correlation matrix
correlation_matrix = numerical_columns.corr()


#gender change to 0,1
# Assuming you have a DataFrame named 'data'
data['Gender'].replace({'Male': 1, 'Female': 0}, inplace=True)

#married change to 0,1
# Assuming you have a DataFrame named 'data'
data['Married'].replace({'Yes': 1, 'No': 0}, inplace=True)

#dependent change to 0,1,2,3
# Assuming you have a DataFrame named 'data'
data['Dependents'].replace({'0': 0, '1': 1, '2': 2, '3+': 3}, inplace=True)

# education change to 0,1
# Assuming you have a DataFrame named 'data'
data['Education'].replace({'Graduate': 1, 'Not Graduate': 0}, inplace=True)

# self_employed change to 0,1
# Assuming you have a DataFrame named 'data'
data['Self_Employed'].replace({'Yes': 1, 'No': 0}, inplace=True)

# property_area change to 0,1,2
# Assuming you have a DataFrame named 'data'
data['Property_Area'].replace({'Urban': 0, 'Rural': 1, 'Semiurban': 2}, inplace=True)

# loan_status change to 0,1
# Assuming you have a DataFrame named 'data'
data['Loan_Status'].replace({'Y': 1, 'N': 0}, inplace=True)


# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title("Correlation Matrix Heatmap")
plt.show()


