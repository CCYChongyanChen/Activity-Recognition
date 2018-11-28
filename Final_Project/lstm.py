#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 21:52:26 2018

@author: chongyan
"""
import os
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.cross_validation import KFold as KF
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation

"""
def load_data(file_name, sequence_length=10, split=0.8):
    df = pd.read_csv(file_name, header=0)
    df.columns = ['nose_x', 'nose_y', 'neck_x', 'neck_y', 'rshoulder_x', 'rshoulder_y', 
         'relbow_x', 'relbow_y', 'rwrist_x', 'rwrist_y', 'lshoulder_x', 'lshoulder_y', 'lelbow_x', 'lelbow_y', 
        'lwrist_x', 'lwrist_y', 'reye_x', 'reye_y', 'leye_x', 'leye_y', 'label']
    data_all = np.array(df).astype(float)
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)
    data = []
    for i in range(len(data_all) - sequence_length - 1):
        data.append(data_all[i: i + sequence_length + 1])
    reshaped_data = np.array(data).astype('float64')
    print(reshaped_data)
    np.random.shuffle(reshaped_data)
    x = reshaped_data[:, :-1]

    y = reshaped_data[:, -1]
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x = x[: split_boundary]
    test_x = x[-100:]

    train_y = y[: split_boundary]
    test_y = y[-100:]

    return train_x, train_y, test_x, test_y, scaler
"""

def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(55,input_shape=((11,20)), return_sequences=True))
#    print(model.layers)
    model.add(LSTM(110, dropout=0.2, recurrent_dropout=0.2,return_sequences=False))
#    model.add(LSTM(220, return_sequences=False))
    model.add(Dense(output_dim=1))
    model.add(Activation('sigmoid'))

    model.compile(loss='mse', optimizer='rmsprop')
    return model


def train_model(train_x, train_y, test_x, test_y):
    model = build_model()

    try:
        if os.path.exists('lstm')==True:
            model=load_model('lstm')
        model.fit(train_x, train_y, batch_size=1024, nb_epoch=100, validation_split=0.01)
        
        predict = model.predict(test_x)
        predict = np.reshape(predict, (predict.size, ))
        
        model.save('lstm')
    except KeyboardInterrupt:
        print(predict)
        print(test_y)
#    print(predict)
#    print(test_y)
#    try:
#        fig = plt.figure(1)
#        plt.plot(predict, 'r:')
#        plt.plot(test_y, 'g-')
#        plt.legend(['predict', 'true'])
#    except Exception as e:
#        print(e)
    return predict, test_y


if __name__ == '__main__':
    
    file_name='/home/chongyan/activity_recognition/Final_project/Tim.csv'
    sequence_length=10
    split=0.8
    df_= pd.read_csv(file_name, header=0)
    df_.columns = ['nose_x', 'nose_y', 'neck_x', 'neck_y', 'rshoulder_x', 'rshoulder_y', 
         'relbow_x', 'relbow_y', 'rwrist_x', 'rwrist_y', 'lshoulder_x', 'lshoulder_y', 'lelbow_x', 'lelbow_y', 
        'lwrist_x', 'lwrist_y', 'reye_x', 'reye_y', 'leye_x', 'leye_y', 'label']
    df_x=df_.iloc[:,:-1]
    df_y=df_.iloc[:,-1]
    
    df_y=df_y.replace('steering',0)
    df_y=df_y.replace('calling_left',1)
    df_y=df_y.replace('calling_right',1)
    df_y=df_y.replace('reading',2)
    df_y=df_y.replace('texting',3)
    df_y=df_y.replace('eating',4)
    
    data_all_ = np.array(df_x).astype(float)
    
    y_all_ =np.array([df_y]).T
    scaler = MinMaxScaler()
    data_all_ = scaler.fit_transform(data_all_)
    data_all=np.concatenate((data_all_,y_all_),axis=1)
    data = []
    for i in range(len(data_all) - sequence_length - 1):
        data.append(data_all[i: i + sequence_length + 1])        
    reshaped_data = np.array(data).astype('float64')
    print(reshaped_data)
#    np.random.shuffle(reshaped_data)
    
    
    x = reshaped_data[:,: ,:-1]
    y = reshaped_data[:,-1,-1]




    
    
    
    kf = KF(len(y),n_folds=2,random_state=None, shuffle=True)
    lstm_average_acc=[]
    for train_index, test_index in kf:
        X_train, X_test = x[train_index], x[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1], 20))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 20))
        predict_y, test_y = train_model(X_train,Y_train,X_test, Y_test)
        try:
            predict_y = scaler.inverse_transform([[i] for i in predict_y])#???????????????????????
            predict_y = scaler.inverse_transform(predict_y)       
        except:
            print('d')
#        test_y = scaler.inverse_transform(test_y)
#    fig2 = plt.figure(2)
        plt.plot(np.rint(predict_y), 'r-')
        plt.plot(test_y, 'g:')
        
        plt.legend(['predict', 'true'])
        plt.show()
#    print("LSTM cross-validation Accuracy:",np.mean(lstm_average_acc))