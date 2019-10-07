# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

train = pd.read_csv("C:/Users/u1195599/Downloads/train.csv")
test = pd.read_csv("C:/Users/u1195599/Downloads/test_nn.csv")

train['Type'] = 'Train'
test['Type'] = 'Test'

fullData = pd.concat([train,test], axis=0)


ID_col = ['User_ID', 'Product_ID']
flag_col = ['Type']
target_col = ['Purchase']
cat_cols= ['Gender','Age','City_Category','Stay_In_Current_City_Years', 'Type']
num_cols= list(set(list(fullData.columns))-set(cat_cols)-set(ID_col)-set(target_col)-set(flag_col))


num_cat_cols = num_cols+cat_cols

for var in num_cat_cols:
    if fullData[var].isnull().any()==True:
        fullData[var+'_NA']=fullData[var].isnull()*1
        
fullData[num_cols] = fullData[num_cols].fillna(fullData[num_cols].mean())

fullData[cat_cols] = fullData[cat_cols].fillna(value = -9999)


from sklearn.preprocessing import LabelEncoder
for var in cat_cols:
    number = LabelEncoder()
    fullData[var] = number.fit_transform(fullData[var].astype('str'))
    
features = list(set(list(fullData.columns))-set(ID_col)-set(target_col))

fullData[features] = fullData[features]/fullData[features].max()

train=fullData[fullData['Type']==1]
test=fullData[fullData['Type']==0]
features=list(set(list(fullData.columns))-set(ID_col)-set(target_col)-set(flag_col))

X = train[features].values
y = train[target_col].values

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.30, random_state=42)



#Model
conda install keras
from keras import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(100, input_dim=11, activation= "relu"))
model.add(Dense(50, activation= "relu"))
model.add(Dense(1))
model.summary() #Print model Summary


model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])


model.fit(X_train, y_train, epochs=10)

from sklearn.metrics import mean_squared_error
pred= model.predict(X_valid)
score = np.sqrt(mean_squared_error(y_valid,pred))
print (score)


X_test = test[features].values

y_test_nn = model.predict(X_test)

