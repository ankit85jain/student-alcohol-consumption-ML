# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 14:23:54 2020

@author: Ankit jan
not expected result
"""

import pandas as pd
import numpy as np

data=pd.read_csv('School_train_data.csv')
test=pd.read_csv('School_test_user.csv')
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

Y=data.Result.values

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(Y)


data_X=data.drop(['Result'],axis=1)
print(data_X.dtypes)
col=['school','sex','address', 'famsize','Pstatus', 'Mjob','Fjob', 'reason','guardian','schoolsup','famsup', 'paid','activities','nursery','higher','internet','romantic']
for name in col:
    labelEncoder=LabelEncoder()
    data_X[name] = labelEncoder.fit_transform(data_X[name])
    test[name] = labelEncoder.transform(test[name])

  
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 30, random_state =1)
regressor.fit(data_X, y)
# Predicting a new result
y_pred = regressor.predict(test)
result=['PASS' if y>=0.5 else 'FAIL' for y in y_pred]
df=pd.DataFrame(result, columns=["result"])
df.to_csv('./result.csv') 



