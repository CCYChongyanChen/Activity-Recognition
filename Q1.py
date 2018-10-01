# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 22:26:17 2018

@author: 70508
"""
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy import signal

from sklearn.ensemble import RandomForestClassifier
from sklearn import cluster
from sklearn.cross_validation import KFold



def load_data(data):
    AX=data['accelerometerAccelerationX']
    AY=data['accelerometerAccelerationY']
    AZ=data['accelerometerAccelerationZ']
    #Time=data['loggingSample']
    #label=data['label']
    #fAX=signal.detrend(AX)
#    plt.plot(fAX)
#    plt.plot(AY)
#    plt.plot(AZ)
    DATA=pd.concat([AX, AY,AZ], axis=1)
    
    scaler=preprocessing.MinMaxScaler()
    DATA=scaler.fit_transform(DATA)#!!!!!!!!!!!!!!!!!!!!!!!!!!!
    

    return DATA


def load_data2(data_):
    AX=data_['accelerometerAccelerationX']
    AY=data_['accelerometerAccelerationY']
    AZ=data_['accelerometerAccelerationZ']
    #Time=data['loggingSample']
    #label=data['label']
    fAX=signal.detrend(AX)
    fAY=signal.detrend(AY)
    fAZ=signal.detrend(AZ)
    plt.figure(1,figsize=(30,15))
   
    
#    plt.plot(AX)
#    print(AX.shape)
#    plt.plot(fAX)
#    plt.plot(AY)
#    plt.plot(AZ)
#    print(len(fAX),lenf(fAY),len(fAZ))
    DATA=np.vstack((fAX,fAY,fAZ))
    DATA=DATA.T
#    scaler=preprocessing.MinMaxScaler()
#    DATA=scaler.fit_transform(DATA)#!!!!!!!!!!!!!!!!!!!!!!!!!!!


    return DATA


def sliding_win(DATA):
        
    frame_size=200
    step_size=100
    for counter in range(0,len(DATA),step_size):
        dataset_frame=DATA[counter:counter+frame_size,:]
        dataset_frame_mean=np.mean(dataset_frame,axis=0)
        dataset_frame_var=np.var(dataset_frame,axis=0)
        dataset_frame_min=np.min(dataset_frame,axis=0)
        dataset_frame_max=np.max(dataset_frame,axis=0)
#        dataset_frame_ptp=np.ptp(dataset_frame,axis=0)
#        dataset_frame_cv=np.mean(dataset_frame,axis=0)/np.std(dataset_frame,axis=0)

#        ,dataset_frame_min,dataset_frame_max,dataset_frame_ptp,dataset_frame_cv
        dataset_frame_features=np.hstack((dataset_frame_var,dataset_frame_mean))
        if counter==0:
            dataset_feature_vector=dataset_frame_features
        else:
            dataset_feature_vector=np.vstack((dataset_feature_vector, dataset_frame_features))
    return dataset_feature_vector




if __name__ == '__main__':

    dance_data= pd.read_csv('G:\\1st_semester\\activity_recognition\\a2\\Q1\\dance.csv')
    dance_data=dance_data.iloc[554:-400,:]
    
    walk_data= pd.read_csv('G:\\1st_semester\\activity_recognition\\a2\\Q1\\walk.csv')
    walk_data=walk_data.iloc[20072:-5077,:]
    car_data= pd.read_csv('G:\\1st_semester\\activity_recognition\\a2\\Q1\\car.csv')
    car_data=car_data.iloc[5070:-5077,:]
    
    run_data= pd.read_csv('G:\\1st_semester\\activity_recognition\\a2\\Q1\\run.csv')
    run_data=run_data.iloc[1479:-500,:]
    upstair1_data= pd.read_csv('G:\\1st_semester\\activity_recognition\\a2\\Q1\\upstair1.csv')
    upstair1_data=upstair1_data.iloc[1300:-586,:]
    upstair2_data= pd.read_csv('G:\\1st_semester\\activity_recognition\\a2\\Q1\\upstair2.csv')
    upstair2_data=upstair2_data.iloc[200:,:]
    upstair_data=pd.concat([upstair1_data,upstair2_data])
#    AX=run_data['accelerometerAccelerationX']#!!!
#    plt.figure()
#    plt.plot(AX)
    
    
    Dance=sliding_win(load_data2(dance_data))
    Walk=sliding_win(load_data2(walk_data))
    Car=sliding_win(load_data2(car_data))
    Run=sliding_win(load_data2(run_data))
    Upstair=sliding_win(load_data2(upstair_data))
    
    Dance=np.hstack([Dance,np.zeros((len(Dance),1))])
    Walk=np.hstack([Walk,np.ones((len(Walk),1))])
    Car=np.hstack([Car,np.ones((len(Car),1))*2])
    Run=np.hstack([Run,np.ones((len(Car),1))*3])
    Upstair=np.hstack([Upstair,np.ones((len(Car),1))*4])
    
    
    All_Data=np.vstack((Dance,Walk,Car,Run,Upstair))
    X=All_Data[:,:12]
    Y=All_Data[:,-1]
    
    
    
    
        
    # =============================================================================
    # Classification- 
    # Generative- Gaussian Naive Bayes
    # discriminative -SVM
    # =============================================================================
    #10-FOLD Accuracy
    kf = KFold(len(Y),n_folds=10,random_state=None, shuffle=False)#!!!!!!!!!!!!!!!!!!!!!!!
    SVM_average_acc=[]
    GNB_average_acc=[]
    RF_average_acc=[]
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        clf = SVC(kernel='rbf')
        clf.fit(X_train, Y_train)
        SVM_acc = clf.score(X_test,Y_test)
        SVM_average_acc.append(SVM_acc)
        
        gnb=GaussianNB()
        gnb.fit(X_train,Y_train)
        GNB_acc = gnb.score(X_test,Y_test)
        GNB_average_acc.append(GNB_acc)
        joblib.dump(gnb,'gnb.model')
        
        rf=RandomForestClassifier()
        rf.fit(X_train,Y_train)
        RF_acc = rf.score(X_test,Y_test)
        RF_average_acc.append(RF_acc)
    
    print("SVM cross-validation Accuracy:",np.mean(SVM_average_acc),'\n',SVM_average_acc)
    print("GNB cross-validation Accuracy:",np.mean(GNB_average_acc),'\n',GNB_average_acc)
    print("RF cross-validation Accuracy:",np.mean(RF_average_acc),'\n',RF_average_acc)
    
    
    
    x_train,x_test, y_train, y_test =train_test_split(X,Y,test_size=0.1, random_state=0)
    #SVM-Split train/test Accuracy
    clf = SVC(C=1,kernel='rbf')
    clf.fit(x_train, y_train)
    SVM_s_acc = clf.score(x_test,y_test)
    print("SVM Split train/test Accuracy:",SVM_s_acc)
    #GNB -Split train/test Accuracy
    gnb=GaussianNB()
    gnb.fit(x_train,y_train)
    GNB_s_acc = gnb.score(x_test,y_test)
    print("GNB -Split train/test Accuracy:",GNB_s_acc)
    
    rf=RandomForestClassifier(n_estimators=100)
    rf.fit(x_train,y_train)
    RF_s_acc = rf.score(x_test,y_test)    
    print("RF -Split train/test Accuracy:",RF_s_acc)
    
    
    
    # =============================================================================
    # Clustering
    # =============================================================================
    
    #K-MEAN
    k=5#!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    
    #K-MEAN Clustering 
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    label_=['dance','walk','car','run','upstair']
    color_=['blue','yellow','green','red','purple']
    plt.figure()
    for i in range(k):
        # select only data observations with cluster label == i
        #plt.figure(1,figsize=(15,15))
        ds = All_Data[np.where(labels==i)]
        # plot the data observations
#        color=color_[i],label=str(i)+label_[i]
        plt.plot(ds[:,0],ds[:,2],'o')
#        plt.legend(loc="lower right")  #set legend location
        # plot the centroids
        lines = plt.plot(centroids[i,0],centroids[i,2],'kx')#???
        
        # make the centroid x's bigger
        plt.setp(lines,ms=15.0)
        plt.setp(lines,mew=2.0)
