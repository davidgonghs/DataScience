import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("dataset/advertising.csv")
#preview data
data.head()

data.info()

# do Exploratory data analysis
sns.pairplot(data, hue='Clicked on Ad', palette='bwr')

# check msising value
data.isnull().sum()

# check how many unique value in each column
data.nunique()

# get all rows number
totalRows = len(data)


# use linear regression to predict Sales based on TV, Radio, Newspaper
# split data into train and test
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# do 散点图 and 残差图
# 散点图 by each column , compare sales and each column
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=7, aspect=0.7, kind='reg')

# do feature scaling
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# after scaling do 散点图 and 残差图
# 散点图 by each column , compare sales and each column
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=7, aspect=0.7, kind='reg')

# do linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

# print intercept
print(lm.intercept_)

# print coefficient
print(lm.coef_)

# print coefficient dataframe
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# get mse
from sklearn import metrics
y_pred = lm.predict(X_test)
print('MSE:', metrics.mean_squared_error(y_test, y_pred))


plt.scatter(y_test, y_pred)

# 残差图
sns.distplot((y_test-y_pred), bins=50)


# figure out the residual
sns.distplot((y_test-y_pred), bins=50)

# mean_squared_error
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))

# r_2 score
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

X = data.drop(['Sales','Radio','Newspaper'], axis=1)
y = data.drop(['TV','Radio','Newspaper'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# create a linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# get prediction
y_pred = model.predict(X_test)

# get mse
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
# get r_2 score
print('r_2 score:', r2_score(y_test, y_pred))

# get Scatter plot have line of best fit
sns.regplot(x='TV', y='Sales', data=data)


plt.scatter(y_pred, y_test,c='blue', marker='o', s=25)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], c='black', lw=2)
plt.xlabel('Predicted Data',c='green')
plt.ylabel('Actual Data',c='green')
plt.title('Predicted Data VS Actual Data',c='green')
plt.show()

# do stander scaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# do linear regression
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, y_train)

# core relation matrix
data.corr()

