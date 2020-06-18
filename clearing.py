# Функция для очистки данных(без удаления nan)
import time
import pandas as pd
import re
import ast
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
def clearing(data):
    data = data.drop(['MlsId', 'mls-id', 'private pool', 'PrivatePool', 'propertyType', 'status', 'fireplace', 'state'],axis = 1)

    def func_stor(x):
        if type(x) != int:
            x = re.findall(r'\d+',x)
            if (len(x) > 1):
                x = x[0]
            elif len(x) == 1:
                x = x[0]
            if not x:
                return 1
        return x
    
    def func_area(x):
        if type(x) != int:
            x = re.findall(r'\d+',x)
            if len(x) > 1:
                x = x[0] + x[1]
            elif len(x) == 1:
                x = x[0]
            if not x:
                return 4940
        return int(x)

    def func_lot(x):
        if type(x) != int:
            x = re.findall(r'\d+',x)
            if len(x) > 1:
                x = x[0] + x[1]
            elif len(x) == 1:
                x = x[0]
            if not x:
                return 5200
        return int(x)
    
    def func_built(x):
        x = re.findall(r'\d+',x)
        if not x:
            return 1976
        return int(x[0])
        
    def data_func1(x):
        a = ast.literal_eval(x)
        return a['atAGlanceFacts'][0].get('factValue')

    def data_func6(x):
        a = ast.literal_eval(x)
        return a['atAGlanceFacts'][5].get('factValue')
    
    def data_func8(x):
        m = []
        a = ast.literal_eval(x)
        a = dict(a[0])
        k = a['data'].get('Distance')
        for i in k:
            h = re.findall(r'\d+',i)
            if len(h) == 0:
                continue
            if len(h) == 1:
                m.append(float(h[0]))
                continue
            d = h[0] + '.' + h[1]
            m.append(float(d))
        if len(m) == 0:
            return 1.1
        else:
            return min(m)
    data['baths'] = data['baths'].astype(str)
    data['beds'] = data['beds'].astype(str)
    data['stories'] = data['stories'].astype(str)
    data['baths'] = data['baths'].fillna(1).apply(func_stor)
    data['stories'] = data['stories'].fillna(1).apply(func_stor)
    data['beds'] = data['beds'].fillna(1).apply(func_stor)
    data['built'] = data['homeFacts'].apply(data_func1)
    data['lotsize'] = data['homeFacts'].apply(data_func6)
    data['distance'] = data['schools'].apply(data_func8)
    data = data.drop(['homeFacts', 'schools'], axis = 1)
    data['lotsize'] = data['lotsize'].fillna(5200).apply(func_lot)
    data['built'] = data['built'].fillna(1976)
    data['built'] = data['built'].astype(str)
    data['built'] = data['built'].apply(func_built)
    data['built'] = data['built'].astype(int)
    data['sqft'] = data['sqft'].fillna(4940).apply(func_area)
    data['baths'] = data['baths'].astype(int)
    data['beds'] = data['beds'].astype(int)
    data['stories'] = data['stories'].astype(int)
    data = data.drop(data.loc[data['city'].isnull()].index)
    data = data.drop(data.loc[data['street'].isnull()].index)
    data = data.drop(data.loc[data['zipcode'].isnull()].index)
    totrain = pd.read_csv('obr.csv')
    full = pd.concat([data, totrain], ignore_index=True)
    categorical_columns = ['city', 'street', 'zipcode']
    for col in categorical_columns:
        le = LabelEncoder()
        le.fit(list(full[col].astype(str).values))
        full[col] = le.transform(list(full[col].astype(str).values))
    data = full[:len(data)]
    del full
    return data

# Загрузка модели
with open('mymodel3.pkl', 'rb') as pkl_file:
    regressor = pickle.load(pkl_file)
