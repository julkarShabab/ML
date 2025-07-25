import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('regression/simple_linear/Salary_Data.csv')
x = dataset.iloc[:,:-1].values
#print(x)
y = dataset.iloc[:,-1].values
#print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size = 0.2, random_state=1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #creating object
regressor.fit(X_train,Y_train)

y_pred = regressor.predict(X_test) #y_pred here contains the predicted salaries
#we're using X_test here because we need to predict on year not on salaries

#train set
plt.scatter(X_train,Y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('salary vs exp')
plt.xlabel('year of exp')
plt.ylabel('salary')
plt.show()

#test set

plt.scatter(X_test,Y_test,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue') # this line remains same because we're keeping the blue line same
plt.title('salary vs exp(test set)')
plt.xlabel('year of exp')
plt.ylabel('salary')
plt.show()