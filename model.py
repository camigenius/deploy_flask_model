from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
import pickle
import pandas as pd
import numpy as np



df = pd.read_csv('./data/housing.data.txt',header = None,sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print(df.head())


X = df.drop('MEDV',axis=1).values
y = df['MEDV'].values



X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
slr = LinearRegression()
slr.fit(X_train,y_train)

pickle.dump(slr,open("model.pkl","wb"))






