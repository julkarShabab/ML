import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
#print(x)
y = dataset.iloc[:,-1].values
#print(y)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean') # missing values are locatedd and their values get replaced by the preferred strategy
imputer.fit(x[:,1:3]) #from column 1 to column 2 since python excludes the upper value
x[:,1:3] = imputer.transform(x[:,1:3])
#print(x)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
#threee things added in transformer :
#what kind of encoding,what kind pf encoding,coolumn index
#remainder = passthrough means other columns will remain intact
x = np.array(ct.fit_transform(x)) #converts the matrix into numpy array
# print(x)

from sklearn.preprocessing import LabelEncoder #label encoder one at a time -->> binary outcome
le = LabelEncoder() #convert categorical labels into integer values
y = le.fit_transform(y)
# print(y)

 
#splitting data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.2, random_state=1)
print("x train",X_train)
print("x test",X_test)
print(Y_train)
print(Y_test)
print(">>>>>>")

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:]) #here we're taking the age and the salary column only
X_test[:,3:] = sc.transform(X_test[:,3:])
print(X_train)
print(">>>")
print(X_test)

#| Data      | What you use       | Why                     |
#| --------- | ------------------ | ----------------------- |
#| `X_train` | `fit_transform()`  | Learn + scale the data  |
#| `X_test`  | `transform()` only | Just scale, no learning |
