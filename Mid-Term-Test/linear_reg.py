import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
from pandas.core.common import random_state


df_sal = pd.read_csv('Salary_Data.csv')
print(df_sal.head())

print(df_sal.describe())
 
plt.title('Salary Distribution Plot')
sns.distplot(df_sal['Salary'])

plt.show()
 
plt.scatter(df_sal['YearsExperience'], df_sal['Salary'], color = 'red')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.box(False)
plt.show()

 
X = df_sal.iloc[:, :1]  
y = df_sal.iloc[:, 1:]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred_test = regressor.predict(X_test)     
y_pred_train = regressor.predict(X_train)   

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, y_pred_train, color = 'firebrick')
plt.title('Salary vs Experience - Training Set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['X_train/Pred(y_test)', 'X_train/y_train'], title = 'Sal/Exp', loc='best', facecolor='white')
plt.box(False)
plt.show()