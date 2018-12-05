#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 21:52:26 2018

@author: chongyan
"""

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
from matplotlib import pyplot
from sklearn.model_selection import KFold, ShuffleSplit
from numpy import concatenate
from sklearn.cross_validation import KFold as KF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.optimizers import Adam,SGD,Nadam
from keras.layers import LSTM, Dense, Activation,Dropout
import keras

def validation(number):
    file_name='/home/chongyan/activity_recognition/Final_project/final_project_data2/test'+str(number)+'.csv'
    sequence_length=10
    df_= pd.read_csv(file_name, header=0)
    df_.columns = ['nose_x', 'nose_y', 'relbow_x', 'relbow_y', 'rwrist_x', 'rwrist_y', 'lelbow_x', 'lelbow_y', 'lwrist_x', 'lwrist_y','lr_wrist_dist', 'r_wrist_to_nose_dist', 'r_wrist_velocity_y','label']
    df_x=df_.iloc[:,:-1]
    df_y=df_.iloc[:,-1]
    
    df_y=df_y.replace('steering',0)
    df_y=df_y.replace('calling_left',1)
    df_y=df_y.replace('calling_right',1)
    df_y=df_y.replace('reading',2)
    df_y=df_y.replace('texting',3)
    df_y=df_y.replace('eating_to_mouth',4)
    
    df_y=df_y.replace('eating_to_lap',5)
    data_all_ = np.array(df_x).astype(float)
    
    y_all_ =np.array([df_y]).T
    scaler = MinMaxScaler()
    data_all_ = scaler.fit_transform(data_all_)
    data_all=np.concatenate((data_all_,y_all_),axis=1)
    data = []
    for i in range(len(data_all) - sequence_length - 1):
        data.append(data_all[i: i + sequence_length + 1])        
    reshaped_data = np.array(data).astype('float64')
    
    x = reshaped_data[:,: ,:-1]
    y = reshaped_data[:,-1,-1]
    y= keras.utils.to_categorical(y,6)
            
    labels=[0,1,2,3,4,5]
    model=load_model('lstm'+str(number))
    
    X_test= np.reshape(x, (x.shape[0],x.shape[1], 13))
    scores = model.evaluate(X_test, y, verbose=0)
    
    confmatrix = np.zeros((5,5))
    
    
    print(scores)
    predict_y = model.predict(X_test)
    
    predict_y= [labels[np.argmax(x)] for x in predict_y]
    
    y= [labels[np.argmax(x)] for x in y]
    confmatrix+= confusion_matrix(y, predict_y)      
    print(confmatrix)
    #test_y= [labels[np.argmax(x)] for x in test_y]
    #try:
    #    predict_y = scaler.inverse_transform([[i] for i in predict_y])#???????????????????????
    #    predict_y = scaler.inverse_transform(predict_y)       
    #except:
    #    print("d")
    #y = scaler.inverse_transform(y)
    plt.figure(figsize=(15,5))
    plt.plot(predict_y, 'r-')
    plt.plot(y, 'g:')
    
    plt.legend(['predict', 'true'])
    plt.show()

    return scores, confmatrix

def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    
    model.add(LSTM(128,dropout=0.2, recurrent_dropout=0.2,input_shape=((11,13)),return_sequences=True))
    model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2,return_sequences=False))
#    model.add(Dense(32, activation='linear'))
#    model.add(Dropout(0.2))
    model.add(Dense(6, activation='softmax'))
#    model.add(Activation('softmax'))#softmax, sigmoid,
#    adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#    sgd=SGD(lr=0.01, momentum=0., decay=0., nesterov=False)
    nadam=Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#    model.compile(loss='mae', optimizer=nadam)#rmsprop
    model.summary()

    model.compile(loss='categorical_crossentropy',
              optimizer=nadam,
              metrics=['accuracy'])
    return model

def train_model(train_x, train_y, test_x, test_y,number):
    model = build_model()
    labels=[0,1,2,3,4,5]
    try:
        if os.path.exists('lstm'+str(number))==True:
            model=load_model('lstm'+str(number))
        history=model.fit(train_x, train_y, batch_size=1024, nb_epoch=100)
        scores = model.evaluate(test_x, test_y, verbose=0)
        print(scores)
        pyplot.plot(history.history['loss'], label='train')
#        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()
        predict = model.predict(test_x)
        predict= [labels[np.argmax(x)] for x in predict]
        
        test_y= [labels[np.argmax(x)] for x in test_y]
#        predict = [labels[np.argmax(x)] for x in predict]
#        predict = np.reshape(predict, (predict.size, ))
        model.save('lstm'+str(number))
    except KeyboardInterrupt:
        print(predict)
        print(test_y)
    return predict, test_y,scores,history.history


if __name__ == '__main__':
#    for number in range(3,5):
        """
        file_name='/home/chongyan/activity_recognition/Final_project/final_project_data2/train'+str(number)+'.csv'
        sequence_length=10
        split=0.8
        df_= pd.read_csv(file_name, header=0)
        df_.columns = ['nose_x', 'nose_y', 'relbow_x', 'relbow_y', 'rwrist_x', 'rwrist_y', 'lelbow_x', 'lelbow_y', 'lwrist_x', 'lwrist_y','lr_wrist_dist', 'r_wrist_to_nose_dist', 'r_wrist_velocity_y','label']
        df_x=df_.iloc[:,:-1]
        df_y=df_.iloc[:,-1]
        
        df_y=df_y.replace('steering',0)
        df_y=df_y.replace('calling_left',1)
        df_y=df_y.replace('calling_right',1)
        df_y=df_y.replace('reading',2)
        df_y=df_y.replace('texting',3)
        df_y=df_y.replace('eating_to_mouth',4)
        df_y=df_y.replace('eating_to_lap',5)
        
        data_all_ = np.array(df_x).astype(float)
        y_all_ =np.array([df_y]).T
    #    data_all=np.concatenate((data_all_,y_all_),axis=1)
        
        scaler = MinMaxScaler()
        data_all_= scaler.fit_transform(data_all_)
        data_all=np.concatenate((data_all_,y_all_),axis=1)
        data = []
        for i in range(len(data_all) - sequence_length - 1):
            data.append(data_all[i: i + sequence_length + 1])        
        reshaped_data = np.array(data).astype('float64')
    #    np.random.shuffle(reshaped_data)
        
        
        x = reshaped_data[:,: ,:-1]
        y = reshaped_data[:,-1,-1]
    
        history2_all=[]
    
    
        
        
        kf = KF(len(y),n_folds=10,random_state=None, shuffle=True)
        lstm_average_acc=[]
        for train_index, test_index in kf:
            X_train, X_test = x[train_index], x[test_index]
            Y_train, Y_test = y[train_index], y[test_index]
            
            Y_train = keras.utils.to_categorical(Y_train, 6)
            Y_test = keras.utils.to_categorical(Y_test,6)
            
            X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1], 13))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 13))
            predict_y, test_y,scores,history2= train_model(X_train,Y_train,X_test, Y_test,number)
             #(412L,1L) 
            history2_all.append(history2)
            
            print(scores)
    
            plt.figure(figsize=(25,5))
            plt.plot(np.rint(predict_y), 'r-')
            plt.plot(test_y, 'g:')
            plt.title("self-driving car activity recognition")
            plt.legend(['predict', 'true'])
            plt.show()
         """
        accuracy_all=[]
        confumatrix_all=[]
        accuracy, confumatix=validation(4)
#        accuracy_all.append(accuracy)
#        confumatrix_all.append(confumatix)
    #    print("LSTM cross-validation Accuracy:",np.mean(lstm_average_acc))