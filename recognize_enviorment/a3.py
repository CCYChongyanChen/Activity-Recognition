#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 18:39:04 2018

@author: chongyan
"""
from __future__ import print_function
import librosa
import librosa.display
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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import cluster
from sklearn.cross_validation import KFold
from numba import jit
@jit
def sliding_win(DATA,sr):
    
    frame_size=22050
    step_size=11025
    for counter in range(0,len(DATA),step_size):
        features=DATA[counter:counter+frame_size]
#        freature_mean=np.mean(freatures,axis=0)
#        freature_var=np.var(freatures,axis=0)
        feature_stft=librosa.feature.chroma_stft(features,sr=sr,n_fft=2048,hop_length=512,n_chroma=40)
        feature_stft=feature_stft.T
        feature_mfcc=librosa.feature.mfcc(features, sr=sr,n_mfcc=40)
        feature_mfcc=feature_mfcc.T
        feature_cqt=librosa.feature.chroma_cqt(features, sr=sr,n_chroma=40)
        feature_cqt=feature_cqt.T
        feature_melspec=  librosa.feature.melspectrogram(features,sr=sr,n_fft=2048,hop_length=512,n_mels=40)
        feature_logmelspec =librosa.power_to_db(feature_melspec)
        feature_logmelspec =feature_logmelspec.T
        dataset_frame_features=np.hstack((feature_stft,feature_mfcc,feature_cqt,feature_logmelspec))
        
        
        mean=np.mean(features)
        var=np.var(features)        
        stft=np.mean(librosa.feature.chroma_stft(features,sr=sr,n_fft=2048,hop_length=512,n_chroma=40).T,axis=0)
        mfccs = np.mean(librosa.feature.mfcc(y=features, sr=sr, n_mfcc=40).T,axis=0)#
        mel = np.mean(librosa.feature.melspectrogram(features, sr=sr).T,axis=0)#
        contrast = np.mean(librosa.feature.spectral_contrast(features, sr=sr).T,axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=features,sr=sr).T,axis=0)
     
        
        dataset_frame_features2=np.hstack((mfccs,stft,mel,contrast,tonnetz,mean,var))
        
    
          
        if counter==0:
            dataset_feature_vector=dataset_frame_features
            
            dataset_feature_vector2=dataset_frame_features2
        else:
            dataset_feature_vector=np.vstack((dataset_feature_vector, dataset_frame_features))
            
            dataset_feature_vector2=np.vstack((dataset_feature_vector2, dataset_frame_features2))
            
    scaler=preprocessing.MinMaxScaler()
    dataset_feature_vector=scaler.fit_transform(dataset_feature_vector)
    return dataset_feature_vector,dataset_feature_vector2

def features_(data):
    feature_stft=librosa.feature.chroma_stft(data,sr=sr,n_fft=2048,hop_length=512,n_chroma=40)
    feature_stft=feature_stft.T
    feature_mfcc=librosa.feature.mfcc(data, sr=sr,n_mfcc=40)
    feature_mfcc=feature_mfcc.T
    feature_cqt=librosa.feature.chroma_cqt(data, sr=sr,n_chroma=40)
    feature_cqt=feature_cqt.T
    feature_melspec=  librosa.feature.melspectrogram(data,sr=sr,n_fft=2048,hop_length=512,n_mels=40)
    feature_logmelspec =librosa.power_to_db(feature_melspec)
    feature_logmelspec =feature_logmelspec.T
    dataset_frame_features=np.hstack((feature_stft,feature_mfcc,feature_cqt,feature_logmelspec))
    scaler=preprocessing.MinMaxScaler()
    dataset_feature_vector=scaler.fit_transform(dataset_frame_features)
    return dataset_feature_vector

@jit
def four_classifications(X,Y):
    kf = KFold(len(Y),n_folds=10,random_state=None, shuffle=True)#!!!!!!!!!!!!!!!!!!!!!!!
    SVM_average_acc=[]
    GNB_average_acc=[]
    RF_average_acc=[]
    MLP_average_acc=[]
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
        
        rf=RandomForestClassifier(n_estimators=100)
        rf.fit(X_train,Y_train)
        RF_acc = rf.score(X_test,Y_test)
        RF_average_acc.append(RF_acc)
    

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(X_train, Y_train)
        MLP_acc = clf.score(X_test,Y_test)
        MLP_average_acc.append(MLP_acc)
        
    print("SVM cross-validation Accuracy:",np.mean(SVM_average_acc))
    print("GNB cross-validation Accuracy:",np.mean(GNB_average_acc))
    print("RF cross-validation Accuracy:",np.mean(RF_average_acc))
    print("MLP cross-validation Accuracy:",np.mean( MLP_average_acc))
 
    
    return True
@jit
def K_Mean_Clustering(X):
    k=3#!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    
    #K-MEAN Clustering 
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
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
        lines = plt.plot(centroids[i,0],centroids[i,1],'kx')#???
        
        # make the centroid x's bigger
        plt.setp(lines,ms=15.0)
        plt.setp(lines,mew=2.0)
    return True

if __name__ == '__main__':
    coffee, sr = librosa.load('/home/chongyan/activity_recogntion/coffee.wav')
    kitchen, sr = librosa.load('/home/chongyan/activity_recogntion/kitchen.wav')
    soccer, sr = librosa.load('/home/chongyan/activity_recogntion/soccer.wav')

    Coffee1,C2=sliding_win(coffee,sr)
    Kitchen1,K2=sliding_win(kitchen,sr)
    Soccer1,S2=sliding_win(soccer,sr)
    
    
    Coffee1=np.hstack([Coffee1,np.ones((len(Coffee1),1))*0])
    Kitchen1=np.hstack([Kitchen1,np.ones((len(Kitchen1),1))*1])
    Soccer1=np.hstack([Soccer1,np.ones((len(Soccer1),1))*2])
    C2=np.hstack([C2,np.ones((len(C2),1))*0])
    K2=np.hstack([K2,np.ones((len(K2),1))*1])
    S2=np.hstack([S2,np.ones((len(S2),1))*2])
    
    
    All_Data1=np.vstack((Coffee1,Kitchen1,Soccer1))
    All_Data2=np.vstack((C2,K2,S2))
    X=All_Data1[:,:159]
    Y=All_Data1[:,160]
    
    
    
    X2=All_Data2[:,:222]
    Y2=All_Data2[:,223]
    four_classifications(X,Y)
    
    four_classifications(X2,Y2)