from __future__ import print_function
import numpy as np
import pandas as pd
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

BODY_PARTS_TO_USE = ['nose_x', 'nose_y', 'relbow_x', 'relbow_y', 'rwrist_x', 'rwrist_y', 'lelbow_x', 'lelbow_y', 
        'lwrist_x', 'lwrist_y']

SECONDARY_FEATURES = ['lr_wrist_dist', 'r_wrist_to_nose_dist', 'r_wrist_velocity_y']

LABEL_COL = ['label']

def Classifications_CV(X,Y):
    print(len(Y))
    confmatrix = np.zeros((7,7))
    kf = KFold(n_splits=2,random_state=None, shuffle=True)#!!!!!!!!!!!!!!!!!!!!!!!
    GNB_average_acc=[]
    RF_average_acc=[]
    MLP_average_acc=[]
    for train_index, test_index in kf.split(Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        gnb=GaussianNB()
        gnb.fit(X_train,Y_train)
        GNB_acc = gnb.score(X_test,Y_test)
        GNB_average_acc.append(GNB_acc)
        joblib.dump(gnb,'gnb.model')
        
        rf=RandomForestClassifier(n_estimators=100)
        rf.fit(X_train,Y_train)
        RF_acc = rf.score(X_test,Y_test)
        RF_average_acc.append(RF_acc)
        joblib.dump(rf,'rf.model')
        print(rf.classes_)
        print(confusion_matrix(Y_test, rf.predict(X_test)))

        clf = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(20, 20), 
                        random_state=1, verbose=False)
        clf.fit(X_train, Y_train)
        predictions = clf.predict(X_test)
        
        #confmatrix+= confusion_matrix(Y_test, predictions) 
        joblib.dump(clf,'clf.model')     
    
        MLP_acc = clf.score(X_test,Y_test)
        MLP_average_acc.append(MLP_acc)
        
    print("GNB cross-validation Accuracy:",np.mean(GNB_average_acc))
    print("RF cross-validation Accuracy:",np.mean(RF_average_acc))
    print("MLP cross-validation Accuracy:",np.mean( MLP_average_acc))
    #print(confmatrix)

def Classifications_LOO(X_train,Y_train, X_test, Y_test):
    confmatrix = np.zeros((7,7))
    GNB_average_acc=[]
    RF_average_acc=[]
    MLP_average_acc=[]
    
    gnb=GaussianNB()
    gnb.fit(X_train,Y_train)
    GNB_acc = gnb.score(X_test,Y_test)
    GNB_average_acc.append(GNB_acc)
    joblib.dump(gnb,'gnb.model')
    
    rf=RandomForestClassifier(n_estimators=100)
    rf.fit(X_train,Y_train)
    RF_acc = rf.score(X_test,Y_test)
    RF_average_acc.append(RF_acc)
    joblib.dump(rf,'rf.model')
    print(rf.classes_)
    print(confusion_matrix(Y_test, rf.predict(X_test)))

    clf = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(100, 20), 
                    random_state=1, verbose=False)
    clf.fit(X_train, Y_train)
    predictions = clf.predict(X_test)
    print(clf.classes_)
    print(confusion_matrix(Y_test, predictions))
    
    #confmatrix+= confusion_matrix(Y_test, predictions) 
    joblib.dump(clf,'clf.model')     

    MLP_acc = clf.score(X_test,Y_test)
    MLP_average_acc.append(MLP_acc)
        
    print("GNB cross-validation Accuracy:",np.mean(GNB_average_acc))
    print("RF cross-validation Accuracy:",np.mean(RF_average_acc))
    print("MLP cross-validation Accuracy:",np.mean( MLP_average_acc))
    #print(confmatrix)

    return np.mean(RF_average_acc), np.mean( MLP_average_acc)

def Classifications_MLP(X_train,Y_train, X_test, Y_test):
    confmatrix = np.zeros((7,7))
    MLP_average_acc=[]

    for i in range(10,100):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(i, i), 
                        random_state=1, verbose=False)
        clf.fit(X_train, Y_train)
        predictions = clf.predict(X_test)
        #print(clf.classes_)
        #print(confusion_matrix(Y_test, predictions))
        
        #confmatrix+= confusion_matrix(Y_test, predictions) 
        #joblib.dump(clf,'clf.model')     

        MLP_acc = clf.score(X_test,Y_test)
        MLP_average_acc.append(MLP_acc)
            
        print("Size:", i, "MLP cross-validation Accuracy:",MLP_acc)

def extract_test_train(feature_csvs, test_idx):
    WINDOW_SIZE = 30 # MAX Window size in number of frames (most samples will be max window size)
    WINDOW_OVERLAP = 15 # Window overlap in number of frames
    FRAME_TIME = 1.0/15.0 # The time of one frame

    test = pd.DataFrame(columns=BODY_PARTS_TO_USE+SECONDARY_FEATURES+LABEL_COL)
    train = test

    for idx, feature_csv in enumerate(feature_csvs):
        raw_poses = pd.read_csv(feature_csv, header=0)
        raw_poses.columns = ['nose_x', 'nose_y', 'neck_x', 'neck_y', 'rshoulder_x', 'rshoulder_y', 
             'relbow_x', 'relbow_y', 'rwrist_x', 'rwrist_y', 'lshoulder_x', 'lshoulder_y', 'lelbow_x', 'lelbow_y', 
            'lwrist_x', 'lwrist_y', 'reye_x', 'reye_y', 'leye_x', 'leye_y', 'label']

        out_data = pd.DataFrame(columns=BODY_PARTS_TO_USE+SECONDARY_FEATURES+LABEL_COL)

        # Parts of raw data that we want to use
        raw_poses_cropped = raw_poses[BODY_PARTS_TO_USE]
        label_column = raw_poses[LABEL_COL]

        # Compute secondary features that are NOT time dependent
        # lr_wrist_dist
        lr_wrist_dist = pow(pow(raw_poses['lwrist_x'] - raw_poses['rwrist_x'],2) + 
            pow(raw_poses['lwrist_y'] - raw_poses['rwrist_y'],2), 0.5)

        # r_wrist_to_nose_dist
        r_wrist_to_nose_dist = pow(pow(raw_poses['rwrist_x'] - raw_poses['nose_x'],2) + 
            pow(raw_poses['rwrist_y'] - raw_poses['nose_y'],2), 0.5)
        # r_wrist_to_nose_dist = pow(pow(raw_poses['lwrist_x'] - raw_poses['rwrist_x'],2) + 
        #     pow(raw_poses['lwrist_y'] - raw_poses['rwrist_y'],2), 0.5)

        # Perform windowing and compute time dependent data like velocity
        current_idx = 0
        raw_data_length = raw_poses.shape[0]
        while current_idx + WINDOW_SIZE - 1 < raw_data_length:
            # The activity of the current window
            current_activity = raw_poses[current_idx:current_idx+1][LABEL_COL].values[0][0]

            # Compute the end idx
            activity_transition_points = raw_poses[current_idx:current_idx+WINDOW_SIZE][['label']]. \
                ne(raw_poses[current_idx:current_idx+WINDOW_SIZE][['label']].shift().bfill()).astype(int)==1
            if activity_transition_points.sum().values[0] == 0:
                end_idx = current_idx+WINDOW_SIZE
                next_idx = current_idx+WINDOW_OVERLAP
            else:
                end_idx = raw_poses[current_idx:current_idx+WINDOW_SIZE].index[activity_transition_points['label']][0]
                next_idx = end_idx

            # Extract the curent window
            curwnd_raw_poses_cropped = raw_poses_cropped[current_idx:end_idx]
            curwnd_lr_wrist_dist = lr_wrist_dist[current_idx:end_idx]
            curwnd_r_wrist_to_nose_dist = r_wrist_to_nose_dist[current_idx:end_idx]

            # Compute the time dependent features
            window_time = (end_idx-current_idx)*FRAME_TIME
            r_wrist_velocity_y = (curwnd_raw_poses_cropped['rwrist_y'] - curwnd_raw_poses_cropped['rwrist_y'].shift().bfill()).sum()/window_time
            r_wrist_velocity_x = (curwnd_raw_poses_cropped['rwrist_x'] - curwnd_raw_poses_cropped['rwrist_x'].shift().bfill()).sum()/window_time
            r_wrist_velocity = pow(pow(r_wrist_velocity_x, 2) + pow(r_wrist_velocity_y, 2), 0.5)

            # Add features to output
            out_idx = len(out_data)
            out_data.loc[out_idx, BODY_PARTS_TO_USE] = curwnd_raw_poses_cropped.tail(1).values[0]
            out_data.loc[out_idx, LABEL_COL] = current_activity
            out_data.loc[out_idx, SECONDARY_FEATURES[0]] = curwnd_lr_wrist_dist.max()
            out_data.loc[out_idx, SECONDARY_FEATURES[1]] = curwnd_r_wrist_to_nose_dist.tail(1).values[0]
            out_data.loc[out_idx, SECONDARY_FEATURES[2]] = r_wrist_velocity_y

            # Set the idx for next iteration
            current_idx = next_idx

        if idx == test_idx:
            test = test.append(out_data)    
        else:
            train = train.append(out_data)

        # print(out_data[out_data['label']=='eating_to_mouth']['r_wrist_to_nose_dist'].mean())
        # print(out_data[out_data['label']=='eating_to_lap']['r_wrist_to_nose_dist'].mean())
        # print(out_data[out_data['label']=='texting']['r_wrist_to_nose_dist'].mean())
        # print(out_data[out_data['label']=='reading']['r_wrist_to_nose_dist'].mean())
        # print(out_data[out_data['label']=='steering']['r_wrist_to_nose_dist'].mean())
        # print('\n')

    return test, train
def data_augment_eating(orig_data, count):
    new_data = pd.DataFrame(0, index=np.arange(count*2), columns=BODY_PARTS_TO_USE+SECONDARY_FEATURES+LABEL_COL)
    # Do the to_mouth feature first
    data_cropped = orig_data[orig_data['label']=='eating_to_mouth']
    for BODY_PART in BODY_PARTS_TO_USE:
        mean = data_cropped[BODY_PART].mean()
        std = data_cropped[BODY_PART].std()/2

        new_data.loc[0:count-1, BODY_PART] = std*np.random.randn(count,1) + mean

    to_mouth_mean = data_cropped['r_wrist_velocity_y'].mean()
    to_mouth_std = data_cropped['r_wrist_velocity_y'].std()/2
    new_data.loc[0:count-1, 'r_wrist_velocity_y'] = to_mouth_std*np.random.randn(count,1) + to_mouth_mean
    new_data.loc[0:count-1, 'label'] = 'eating_to_mouth'

    # Do the to_lap feature
    data_cropped = orig_data[orig_data['label']=='eating_to_lap']
    for BODY_PART in BODY_PARTS_TO_USE:
        mean = data_cropped[BODY_PART].mean()
        std = data_cropped[BODY_PART].std()/2

        new_data.loc[count:2*count-1, BODY_PART] = std*np.random.randn(count,1) + mean
    to_lap_mean = data_cropped['r_wrist_velocity_y'].mean()
    to_lap_std = data_cropped['r_wrist_velocity_y'].std()/2
    new_data.loc[count:2*count-1, 'r_wrist_velocity_y'] = to_lap_std*np.random.randn(count,1) + to_lap_mean
    new_data.loc[count:2*count-1, 'label'] = 'eating_to_lap'

    # Compute secondary features that are NOT time dependent
    # lr_wrist_dist
    new_data['lr_wrist_dist'] = pow(pow(new_data['lwrist_x'] - new_data['rwrist_x'],2) + 
        pow(new_data['lwrist_y'] - new_data['rwrist_y'],2), 0.5)

    # r_wrist_to_nose_dist
    new_data['r_wrist_to_nose_dist'] = pow(pow(new_data['rwrist_x'] - new_data['nose_x'],2) + 
        pow(new_data['rwrist_y'] - new_data['nose_y'],2), 0.5)
    # new_data['r_wrist_to_nose_dist'] = pow(pow(new_data['lwrist_x'] - new_data['rwrist_x'],2) + 
    #     pow(new_data['lwrist_y'] - new_data['rwrist_y'],2), 0.5)

    return orig_data.append(new_data)

if __name__ == '__main__':
    feature_csvs=['../../Work/videos/RealData/Chongyan_features.csv', \
                '../../Work/videos/RealData/Ershad_features.csv', \
                '../../Work/videos/RealData/Tim_features.csv', \
                '../../Work/videos/RealData/Subject_1_features.csv', \
                '../../Work/videos/RealData/Subject_2_features.csv', \
                '../../Work/videos/RealData/Subject_3_features.csv', \
                '../../Work/videos/RealData/Subject_5_features.csv', \
                '../../Work/videos/RealData/Subject_6_features.csv']

    overall_accuracy = pd.DataFrame(0, index=np.arange(6), columns=['RF', 'MLP'])

    for person_id in range(0,6):
        test, train = extract_test_train(feature_csvs, person_id)
        print(test.shape)
        print(train.shape)

        train = data_augment_eating(train, 140)
        test = data_augment_eating(test, 35)
        
        combined = train.append(test)
        combined.to_csv('feature.csv',index=False)

        # print('r_wrist_velocity_y')
        # print(max(abs(combined[combined['label'] == 'steering']['r_wrist_velocity_y'])))
        # print(max(abs(combined[combined['label'] == 'texting']['r_wrist_velocity_y'])))
        # print(max(abs(combined[combined['label'] == 'calling_left']['r_wrist_velocity_y'])))
        # print(max(abs(combined[combined['label'] == 'calling_right']['r_wrist_velocity_y'])))
        # print(max(abs(combined[combined['label'] == 'reading']['r_wrist_velocity_y'])))
        # print(max(abs(combined[combined['label'] == 'eating_to_mouth']['r_wrist_velocity_y'])))
        # print(max(abs(combined[combined['label'] == 'eating_to_lap']['r_wrist_velocity_y'])))

        # print('lr_wrist_dist')
        # print((abs(combined[combined['label'] == 'steering']['lr_wrist_dist']).mean()))
        # print((abs(combined[combined['label'] == 'texting']['lr_wrist_dist']).mean()))
        # print((abs(combined[combined['label'] == 'calling_left']['lr_wrist_dist']).mean()))
        # print((abs(combined[combined['label'] == 'calling_right']['lr_wrist_dist']).mean()))
        # print((abs(combined[combined['label'] == 'reading']['lr_wrist_dist']).mean()))
        # print((abs(combined[combined['label'] == 'eating_to_mouth']['lr_wrist_dist']).mean()))
        # print((abs(combined[combined['label'] == 'eating_to_lap']['lr_wrist_dist']).mean()))

        # train = train[(train['label'] == 'eating_to_mouth') | (train['label'] == 'eating_to_lap')]
        # test = test[(test['label'] == 'eating_to_mouth') | (test['label'] == 'eating_to_lap')]
        # train = train[['lr_wrist_dist', 'label']]
        # test = test[['lr_wrist_dist', 'label']]

        start_feature = test.columns[0]
        end_feature = test.columns[-2]
        # all_inputdata = np.array(combined.loc[:, start_feature:end_feature])
        # target = np.array(combined.loc[:, 'label'])
        # Classifications_CV(all_inputdata,target)
        rf_acc, mlp_acc = Classifications_LOO(train.loc[:, start_feature:end_feature],train.loc[:, 'label'],
            test.loc[:, start_feature:end_feature],test.loc[:, 'label'])

        overall_accuracy.loc[person_id] = [rf_acc, mlp_acc]

    print(overall_accuracy)