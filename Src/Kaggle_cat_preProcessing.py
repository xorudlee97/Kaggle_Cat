import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

data_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Kaggle_Cat/Data/")
train_data = pd.read_csv(data_dir+"/train.csv",sep=',', header=[0]) 
test_data = pd.read_csv(data_dir+"/test.csv",sep=',', header=[0])


train_dataframe = pd.DataFrame(train_data)
test_dataframe = pd.DataFrame(test_data)

# id 값 추출
# train_dataframe = train_dataframe.drop('id', axis=1)
# test_dataframe = test_dataframe.drop('id', axis=1)

# 원본 데이터 Numpy로 저장
ori_numpy_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Kaggle_Cat/Numpy_Ori/")
ori_numpy_train_data = np.array(train_dataframe)
ori_numpy_test_data  = np.array(test_dataframe)
np.save(ori_numpy_dir+"/ori_train.npy", ori_numpy_train_data)
np.save(ori_numpy_dir+"/ori_test.npy", ori_numpy_test_data)

# 데이터 전처리
def watch_train_data(train_data):
    for data in train_data:
        print("=================================")
        print(train_dataframe[data].value_counts())
        print("=================================")

def change_YesNo_Data(input_train_data, input_test_data):
    Yes_OR_No = {'T':1, 'F':0, 'Y':1, 'N':0}
    input_train_data['bin_3'] = input_train_data['bin_3'].map(Yes_OR_No)
    input_train_data['bin_4'] = input_train_data['bin_4'].map(Yes_OR_No)
    input_test_data['bin_3'] = input_test_data['bin_3'].map(Yes_OR_No)
    input_test_data['bin_4'] = input_test_data['bin_4'].map(Yes_OR_No)
    return input_train_data, input_test_data

def change_Categorical_nom_Data(input_train_data, input_test_data):
    mapper_nom0 = {'Green':0, 'Blue':1, 'Red':2}
    mapper_nom1 = {'Trapezoid':0, 'Square':1, 'Star':2, 'Circle':3, 'Polygon':4, 'Triangle':5}
    mapper_num2 = {'Lion':0, 'Cat':1, 'Snake':2, 'Dog':3, 'Axolotl':4, 'Hamster':5}
    mapper_num3 = {'Russia':0, 'Canada':1,'Finland':2, 'China':3, 'Costa Rica':4, 'India':5}
    mapper_num4 = {'Oboe':0, 'Piano':1, 'Bassoon':2, 'Theremin':3}
    for col, mapper in zip(['nom_0', 'nom_1','nom_2','nom_3','nom_4'], [mapper_nom0, mapper_nom1, mapper_num2, mapper_num3, mapper_num4]):
        input_train_data[col] = input_train_data[col].replace(mapper)
        input_test_data[col] = input_test_data[col].replace(mapper)
    return input_train_data, input_test_data

from sklearn.preprocessing import MinMaxScaler

def chage_Standard_nom_Data(input_train_data, input_test_data):
    Standard_nom_list = ['nom_5','nom_6','nom_7','nom_8','nom_9']
    
    for col in Standard_nom_list:
        # print(input_train_data[col].apply( lambda x: hash(str(x)) % 30000))
        input_train_data[col] = input_train_data[col].apply( lambda x: hash(str(x))% 1000000000)
        input_test_data[col]  = input_test_data[col].apply( lambda x: hash(str(x))% 1000000000)
        scaler = MinMaxScaler()
        scaler.fit(input_train_data[col].values.reshape(-1, 1))
        input_train_data[col] = scaler.transform(input_train_data[col].values.reshape(-1, 1))
        input_test_data[col] = scaler.transform(input_test_data[col].values.reshape(-1, 1))
    return input_train_data, input_test_data

def change_Categorical_ord_Data(input_train_data, input_test_data):
    mapper_ord_1 = {'Novice': 1, 'Contributor': 2, 'Expert': 3, 'Master': 4, 'Grandmaster': 5}
    mapper_ord_2 = {'Freezing': 1, 'Cold': 2, 'Warm': 3, 'Hot': 4,'Boiling Hot': 5, 'Lava Hot': 6}
    mapper_ord_3 = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 
                    'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15}
    mapper_ord_4 = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 
                    'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15,
                    'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 
                    'W': 23, 'X': 24, 'Y': 25, 'Z': 26}
    for col, mapper in zip(['ord_1', 'ord_2', 'ord_3', 'ord_4'], [mapper_ord_1, mapper_ord_2, mapper_ord_3, mapper_ord_4]):
        input_train_data[col] = input_train_data[col].replace(mapper)
        input_test_data[col] = input_test_data[col].replace(mapper)
    return input_train_data, input_test_data

from sklearn.preprocessing import OrdinalEncoder

def change_Categorical_ord_5_Data(input_train_data, input_test_data):
    encoder = OrdinalEncoder(categories='auto')
    encoder.fit(input_train_data.ord_5.values.reshape(-1, 1))
    input_train_data.ord_5 = encoder.transform(input_train_data.ord_5.values.reshape(-1, 1))
    input_test_data.ord_5 = encoder.transform(input_test_data.ord_5.values.reshape(-1, 1))
    return input_train_data, input_test_data


# watch_train_data(train_dataframe)
train_data, test_data = change_YesNo_Data(train_dataframe, test_dataframe)
train_data, test_data = change_Categorical_ord_Data(train_data, test_data)
train_data, test_data = change_Categorical_ord_5_Data(train_data, test_data)
train_data, test_data = change_Categorical_nom_Data(train_data, test_data)
train_data, test_data = chage_Standard_nom_Data(train_data, test_data)
# watch_train_data(train_data)
# watch_train_data(test_data)

# 원본 데이터 Numpy로 저장
pro_numpy_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Kaggle_Cat/Numpy_Pro/")
pro_numpy_train_data = np.array(train_data)
pro_numpy_test_data  = np.array(test_data)
np.save(pro_numpy_dir+"/pro_train.npy", pro_numpy_train_data)
np.save(pro_numpy_dir+"/pro_test.npy", pro_numpy_test_data)

