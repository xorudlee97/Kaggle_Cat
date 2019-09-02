from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import os
from keras import backend as K

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


pro_numpy_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Kaggle_Cat/Numpy_Pro/")
list_data = np.load(pro_numpy_dir+"/pro_train.npy")
pre_data = np.load(pro_numpy_dir+"/pro_test.npy")

# print(list_data.shape)
# print(pre_data.shape)

X_data = list_data[:, 1:-1]
Y_data = list_data[:, [-1]]
X_pred = pre_data[:, 1:]
# print(X_data.shape)
# print(Y_data.shape)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_data, Y_data,random_state=77,test_size=0.3, shuffle=True
)
# LSTM 모델
# X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# X_pred = np.reshape(X_pred, (X_pred.shape[0], X_pred.shape[1], 1))
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()

# Dense 모델
model.add(Dense(512, input_shape=(23,), activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))

model.add(Dense(700, activation='relu'))
model.add(Dense(700, activation='relu'))
model.add(Dense(700, activation='relu'))

model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# LSTM 모델
# model.add(LSTM(128, input_shape=(23, 1)))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[auc])

model.fit(X_train, Y_train, batch_size=5000, epochs=1500)

loss, auc = model.evaluate(X_test, Y_test, batch_size=500)
print("auc: ", auc)

y_predict = model.predict(X_pred)
print(y_predict)

# for predict in y_predict:
#     if predict > 0.3 and predict < 0.7:
#         predict = predict;
#     else:
#         predict = np.round(predict)
# print(y_predict)

# y_predict = np.round(y_predict)
# print(y_predict)

import pandas as pd

# 결과 값을 CSV로 저장
# sub['target'] = y_preds
submit_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Kaggle_Cat/Data/")
result_dir = os.path.dirname("D:/LTK_AI/LTK_AI_Study/AI_Study/Kaggle_Cat/Result/")

submit_fie = pd.read_csv(submit_dir+"/sample_submission.csv", index_col='id')
submit_fie['target'] = y_predict
submit_fie.to_csv(result_dir+'/Predict_New_Result_5.csv')
