import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

loan = pd.read_csv(r"C:\Users\GHS\Downloads\Loan.csv")
#preview data
loan.head()

#preview info
loan.info()

#preview description
loan.describe()

#check missing values
loan.isnull().sum()

print(loan.columns.tolist())

# ['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']

#Gender fillna
loan['Gender'].mode()
loan['Gender'] = loan['Gender'].fillna(loan['Gender'].mode()[0])
loan.nunique()

loan['Married'].mode()
loan['Married'] = loan['Married'].fillna(loan['Married'].mode()[0])
loan.nunique()

# Dependents
loan['Dependents'].mode()
loan['Dependents'] = loan['Dependents'].fillna(loan['Dependents'].mode()[0])
loan.nunique()



loan['Self_Employed'].mode()
loan['Self_Employed'] = loan['Self_Employed'].fillna(loan['Self_Employed'].mode()[0])
loan['Self_Employed'].unique()


# LoanAmount
loan['LoanAmount'].mode()
loan['LoanAmount'] = loan['LoanAmount'].fillna(loan['LoanAmount'].mode()[0])
loan.nunique()

# Loan_Amount_Term
loan['Loan_Amount_Term'].mode()
loan['Loan_Amount_Term'] = loan['Loan_Amount_Term'].fillna(loan['Loan_Amount_Term'].mode()[0])
loan.nunique()

# Credit_History
loan['Credit_History'].mode()
loan['Credit_History'] = loan['Credit_History'].fillna(loan['Credit_History'].mode()[0])
loan.nunique()

#check missing values
loan.isnull().sum()

#gender change to 0,1
# Assuming you have a DataFrame named 'data'
loan['Gender'].replace({'Male': 1, 'Female': 0}, inplace=True)

#married change to 0,1
# Assuming you have a DataFrame named 'data'
loan['Married'].replace({'Yes': 1, 'No': 0}, inplace=True)

#dependent change to 0,1,2,3
# Assuming you have a DataFrame named 'data'
loan['Dependents'].replace({'0': 0, '1': 1, '2': 2, '3+': 3}, inplace=True)

# education change to 0,1
# Assuming you have a DataFrame named 'data'
loan['Education'].replace({'Graduate': 1, 'Not Graduate': 0}, inplace=True)

# self_employed change to 0,1
# Assuming you have a DataFrame named 'data'
loan['Self_Employed'].replace({'Yes': 1, 'No': 0}, inplace=True)

# property_area change to 0,1,2
# Assuming you have a DataFrame named 'data'
loan['Property_Area'].replace({'Urban': 0, 'Rural': 1, 'Semiurban': 2}, inplace=True)

# loan_status change to 0,1
# Assuming you have a DataFrame named 'data'
loan['Loan_Status'].replace({'Y': 1, 'N': 0}, inplace=True)


# Assuming you have a DataFrame 'data' with numerical columns
numerical_columns = loan.select_dtypes(include=['int64', 'float64'])

# Calculate the correlation matrix
correlation_matrix = numerical_columns.corr()


cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Loan_ID', 'Dependents']
loan = loan.drop(columns=cols, axis=1)
loan.head()

# Create a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title("Correlation Matrix Heatmap")
plt.show()


from sklearn.preprocessing import LabelEncoder
cols = ['Gender','Married','Education','Self_Employed','Credit_History','Property_Area']
le = LabelEncoder()
for col in cols:
    loan[col] = le.fit_transform(loan[col])