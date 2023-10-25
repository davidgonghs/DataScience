import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,mean_squared_error,r2_score
data = pd.read_csv("dataset/insurance.csv")
#preview data
data.head()

data.info()

data.isnull().sum()

# check how many unique value in each column
data.nunique()

### Columns Details:
### age: age of primary beneficiary
### sex: insurance contractor gender, female, male
### bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
### children: Number of children covered by health insurance / Number of dependents
### smoker: Smoking
### region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
### charges: Individual medical costs billed by health insurance

# change smoker to 0 or 1
data['smoker'] = data['smoker'].apply(lambda x: 1 if x == 'yes' else 0)

# change sex to 0 or 1
data['sex']= data['sex'].apply(lambda x: 1 if x == 'male' else 0)

# change region to 0,1,2,3
data['region'] = data['region'].apply(lambda x: 0 if x == 'northeast' else (1 if x == 'southeast' else (2 if x == 'southwest' else 3)))

# first round find X = age, y= charges
X = data[['age']]
y = data['charges']

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# use linear regression to predict charges based on age
# use Logistic Regression
# use Polynomial Regression



