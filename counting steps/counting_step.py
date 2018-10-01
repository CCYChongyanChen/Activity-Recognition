# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 05:06:13 2018

@author: 70508
"""

from sklearn.externals import joblib
import scipy
from sklearn.cross_validation import KFold
import numpy as np
from scipy import signal
from scipy.signal import filtfilt,lfilter
from numpy import mean
from scipy.signal import argrelmax, argrelmin
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from sklearn import preprocessing
from sklearn import cluster
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
def load_data(data):
    AX=data['acx']
    AY=data['acy']
    AZ=data['acz']
    GX=data['gyx']
    GY=data['gyy']
    GZ=data['gyz']
   
    
    DATA=pd.concat([AX, AY,AZ,GX,GY,GZ], axis=1)
    
    
    scaler=preprocessing.MinMaxScaler()
    DATA=scaler.fit_transform(DATA)#!!!!!!!!!!!!!!!!!!!!!!!!!!!
    

    return DATA
def load_data2(data):
    AX=data['acx']
    AY=data['acy']
    AZ=data['acz']
    #Time=data['loggingSample']
    #label=data['label']
    fAX=signal.detrend(AX)
    fAY=signal.detrend(AY)
    fAZ=signal.detrend(AZ)
    
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

def scale(n, data_):
    output = []
    list_len = len(data_)
    for i in range(list_len):
        base = data_[i] * n
        output.append(base) 
        if (i + 1 < list_len) and (data_[i] + 1 == data_[i + 1]) :
            for j in range(1, n):
                output.append(base + j)
    return output
        
def BUTTER_LPASS(data,amplingrate=30,label=False):
    
    fdata=signal.detrend(data)
    if label== True:
        WN=0.2
    if label==False:
        WN=0.3
    b, a = scipy.signal.butter(10, WN, 'low')
    output_signal = scipy.signal.filtfilt(b, a, fdata)
    return output_signal
def sliding_win(DATA,frame_size,step_size):
        
    for counter in range(0,len(DATA),step_size):
        dataset_frame=DATA[counter:counter+frame_size,:]
        dataset_frame_mean=np.mean(dataset_frame,axis=0)
        dataset_frame_var=np.var(dataset_frame,axis=0)
        dataset_frame_min=np.min(dataset_frame,axis=0)
        dataset_frame_max=np.max(dataset_frame,axis=0)
        dataset_frame_ptp=np.ptp(dataset_frame,axis=0)
        dataset_frame_cv=np.mean(dataset_frame,axis=0)/np.std(dataset_frame,axis=0)

        
        dataset_frame_features=np.hstack((dataset_frame_var,dataset_frame_mean,dataset_frame_min,dataset_frame_max,dataset_frame_ptp,dataset_frame_cv))
        if counter==0:
            dataset_feature_vector=dataset_frame_features
        else:
            dataset_feature_vector=np.vstack((dataset_feature_vector, dataset_frame_features))
    return dataset_feature_vector
def sliding_win2(DATA,frame_size,step_size):
        
    for counter in range(0,len(DATA),step_size):
        dataset_frame=DATA[counter:counter+frame_size,:]
        dataset_frame_min=np.min(dataset_frame,axis=0)
        
        dataset_frame_max=np.max(dataset_frame,axis=0)
        dataset_frame_features=np.hstack((dataset_frame_min,dataset_frame_max))
        if counter==0:
            dataset_feature_vector=dataset_frame_features
        else:
            dataset_feature_vector=np.vstack((dataset_feature_vector, dataset_frame_features))
    return dataset_feature_vector

if __name__ == '__main__':
    samplingrate=30
    walk_step_=[]
    run_step_=[]
    total_step_=[]
    names_=['acx','acy','acz','gyx','gyy','gyz','all_ac','all_gy']
    step_data= pd.read_csv('G:\\1st_semester\\activity_recognition\\a2\\Q2\\steps.csv',names=['t','acx','acy','acz','t2','gyx','gyy','gyz'])
    
#    step_data= pd.read_csv('G:\\1st_semester\\activity_recognition\\a2\\Q2\\my_step.csv',names=['t','acx','acy','acz','t2','gyx','gyy','gyz'])
    step_size=70
    frame_size=70
#K-MEAN Clustering 
    WnR=sliding_win2(load_data(step_data),frame_size,step_size)
    #K-MEAN
    k=2
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(WnR)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    
# Classification about which is running and which is walking
    ds0 =list(np.where(labels==0)[0])
    ds1 =list(np.where(labels==1)[0])
    ds000 = scale(70, ds0)
    ds111 = scale(70, ds1)
    new_step_data_0=step_data.iloc[ds000,:]
    new_step_data_1=step_data.iloc[ds111,:]
    gnb=joblib.load('gnb.model')
    
    if len(new_step_data_0)> len(new_step_data_1):
        test=new_step_data_0.iloc[:200,1:4]
        WnR_=sliding_win(load_data2(test),200,100)
        result=gnb.predict(WnR_)
        if mean(result)>=3:
            walk_is_0=True
            print('0longer,0 is walk',result)
        else:
            walk_is_0=False
            print('0longer;1 is walk',result)
    else:
        test=new_step_data_1.iloc[:200,1:4]
        WnR_=sliding_win(load_data2(test),200,100)
        result=gnb.predict(WnR_)
        if mean(result)>=3:
            walk_is_0=False
            print('1longer;1 is walk',result)
        else:
            walk_is_0=True
            print('1longer;0 is walk',result)
# apply different Filter on running and walking,respectively
            
       
    for i in range(5):
        
        if walk_is_0==True:
            fwalk=BUTTER_LPASS(new_step_data_0[names_[i]],samplingrate,True)
            frun=BUTTER_LPASS(new_step_data_1[names_[i]],samplingrate,False)
        else:
            fwalk=BUTTER_LPASS(new_step_data_1[names_[i]],True)
            frun=BUTTER_LPASS(new_step_data_0[names_[i]],False)
# Different Peak detection algorithms
#        print(names_[i]+"arg_walk steps:",len(argrelmax(fwalk)[0]))
#        print(names_[i]+"arg_run_steps:",len(argrelmax(frun)[0]))
        #Seansonal decomposition
        result=seasonal_decompose(np.array(fwalk),model='additive',freq=30)
        season_max=np.max(result.seasonal)
        season_walk_step=len(np.where(result.seasonal==season_max)[0])
        result=seasonal_decompose(np.array(frun),model='additive',freq=30)
        season_max=np.max(result.seasonal)
        season_run_step=len(np.where(result.seasonal==season_max)[0])
        print(names_[i]+"season_total_steps", season_run_step+season_walk_step,'  ',"season_walk steps:",season_walk_step,'  ',"season_run_steps:",season_run_step)
        total_step_.append(season_run_step+season_walk_step)
        walk_step_.append(season_walk_step)
        run_step_.append(season_run_step)
        # scipy
        test_=scipy.signal.find_peaks(fwalk, height=np.mean(fwalk)+np.var(fwalk),distance=0.25*samplingrate, rel_height=0.5)[0]
        walk_step=len(test_)
        test_run=scipy.signal.find_peaks(frun, height=np.mean(frun)+np.var(frun),  distance=0.2*samplingrate,rel_height=0.5)[0]
        run_step=len(test_run)
        print(names_[i]+"scipy_total steps", walk_step+run_step,'  ',"scipy_walk steps:",walk_step,'  ',"scipy_run_steps:",run_step)
        total_step_.append(walk_step+run_step)
        walk_step_.append(walk_step)
        run_step_.append(run_step)
        
#All-accelerometer
    AX=step_data['acx']
    AY=step_data['acy']
    AZ=step_data['acz']
    facx=signal.detrend(AX)
    facy=signal.detrend(AY)
    facz=signal.detrend(AY)
    all_ac = np.sqrt(np.square(facx) + np.square(facy) + np.square(facz))
    facc_all=BUTTER_LPASS(all_ac,True)
#    plt.figure(1,figsize=(15,5))
#    plt.plot(facc_all)
    print("AA_scipt_total steps:",len(scipy.signal.find_peaks(facc_all, height=np.mean(facc_all)+np.var(facc_all),distance=0.25*samplingrate, rel_height=0.5)[0]))
    
    GX=step_data['gyx']
    GY=step_data['gyy']
    GZ=step_data['gyz']
    FGX=signal.detrend(GX)
    FGY=signal.detrend(GY)
    FGZ=signal.detrend(GY)
    all_G = np.sqrt(np.square(FGX) + np.square(FGY) + np.square(FGZ))
    fG_all=BUTTER_LPASS(all_G,True)
    plt.figure(1,figsize=(15,5))
    plt.plot(frun)
    plt.scatter(test_run,np.zeros(len(test_run)))
    print("AG_scipt_total steps:",len(scipy.signal.find_peaks(fG_all, height=np.mean(fG_all)+np.var(fG_all),distance=0.25*samplingrate, rel_height=0.5)[0]))
    
    
    print('final_result:\n', 'total_step:',np.median(total_step_),'\n','walk_step:',np.median(walk_step_),'run_step:',np.median(run_step_))
 