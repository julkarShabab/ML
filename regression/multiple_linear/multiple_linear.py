import numpy as np
import matplotlib.pyplot as pd
import pandas as pd

dataset = pd.read_csv('regression/multiple_linear/50_Startups.csv')
x = dataset.iloc[:,:-1].values
print(x)
y = dataset.iloc[:,-1].values
print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.2, random_state=1)