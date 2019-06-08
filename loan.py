import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
import sklearn.metrics as SM

dataset = pd.read_csv('loan_pred.txt')
dataset = dataset.drop('Loan_ID', axis = 1)

labelencoder_X = LabelEncoder()
dataset['Gender'] = labelencoder_X.fit_transform(dataset['Gender'].astype(str))
dataset['Married'] = labelencoder_X.fit_transform(dataset['Married'].astype(str))
dataset['Education'] = labelencoder_X.fit_transform(dataset['Education'].astype(str))
dataset['Self_Employed'] = labelencoder_X.fit_transform(dataset['Self_Employed'].astype(str))
dataset['Property_Area'] = labelencoder_X.fit_transform(dataset['Property_Area'].astype(str))

labelencoder_Y = LabelEncoder()
dataset['Loan_Status'] = labelencoder_Y.fit_transform(dataset['Loan_Status'].astype(str))

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11].values

imputer = Imputer(missing_values = 2, strategy = 'median', axis = 0)
X[:, [0,1,3,4]] = imputer.fit_transform(X[:, [0,1,3,4]])

imputer = Imputer(missing_values = 'NaN', strategy = 'median', axis = 0)
X[:, [7,8,9]] = imputer.fit_transform(X[:, [7,8,9]])

df_X = pd.DataFrame(X)
df_X[2] = df_X[2].str.replace('3+','4')
df_X[2] = df_X[2].str.replace('+','')

X1 = df_X.iloc[:,:].values
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
X1[:, [2]] = imputer.fit_transform(X1[:, [2]])

df_X = pd.DataFrame(X1, columns = ['Gender','Married','Dependents','Education','Self_Employed','Applicant_Income','Co-Applicant_Income','Loan_Amount','Loan_Amount_Term','Credit_History','Property_Area'])

df_X1 = pd.get_dummies(df_X, columns=['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area'], drop_first=True)

X1 = df_X1.iloc[:,:]

scaler = MinMaxScaler()
X1 = scaler.fit_transform(X1)

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.3, random_state = 42)

log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)

gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)

log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)

rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)

print(SM.accuracy_score(y_test,y_pred))
print(SM.recall_score(y_test,y_pred))
print(SM.precision_score(y_test,y_pred))
print(SM.f1_score(y_test,y_pred))