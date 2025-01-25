import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#to ignore warnings
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("used_cars.csv")

# Cleaning Data

data.head()
data.tail()
data.info()
data.nunique()
data.isnull().sum()
(data.isnull().sum()/(len(data))) * 100


# Removing the S.No column
data = data.drop(['S.No.'], axis=1)
data.info()


from datetime import date

date.today().year
data['Car_Age'] = date.today().year - data['Year']
data.head()

data['Brand'] = data.Name.str.split().str.get(0)
data['Model'] = data.Name.str.split().str.get(1) + data.Name.str.split().str.get(2)
data[['Name', 'Brand', 'Model']]


data.Brand.unique()
data.Brand.nunique()


searchfor = ['Isuzu', 'ISUZU', 'Mini', 'Land']
data[data.Brand.str.contains('|'.join(searchfor))].head(5)
data["Brand"].replace({"ISUZU": "Isuzu", "Mini": "Mini Cooper", "Land": "Land Rover"}, inplace=True)



# EDA

data.describe().T
data.describe(include='all').T


cat_cols = data.select_dtypes(include=['object']).columns
num_cols = data.select_dtypes(include=np.number).columns.tolist()
print("Categorical Variables")
print(cat_cols)
print("Numerical Variable")
print(num_cols)