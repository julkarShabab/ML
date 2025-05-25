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
print(x)