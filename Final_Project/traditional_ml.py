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

def Four_classifications(X,Y):
    print(len(Y))
    confmatrix = np.zeros((7,7))
    kf = KFold(n_splits=5,random_state=None, shuffle=True)#!!!!!!!!!!!!!!!!!!!!!!!
    SVM_average_acc=[]
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
    print(confmatrix)

def extract_test_train(feature_csvs, test_idx):
    BODY_PARTS_TO_USE = ['nose_x', 'nose_y', 'relbow_x', 'relbow_y', 'rwrist_x', 'rwrist_y', 'lelbow_x', 'lelbow_y', 
        'lwrist_x', 'lwrist_y']

    SECONDARY_FEATURES = ['lr_wrist_dist', 'r_wrist_to_nose_dist', 'r_wrist_velocity_y']

    LABEL_COL = ['label']

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
        r_wrist_to_nose_dist = pow(pow(raw_poses['lwrist_x'] - raw_poses['rwrist_x'],2) + 
            pow(raw_poses['lwrist_y'] - raw_poses['rwrist_y'],2), 0.5)

        # Perform windowing and compute time dependent data like velocity
        current_idx = 0
        raw_data_length = raw_poses.shape[0]
        while current_idx + WINDOW_SIZE - 1 < raw_data_length:
            # The activity of the current window
            current_activity = raw_poses[current_idx:current_idx+1][LABEL_COL].values[0][0]

            # Compute the end idx
            if raw_poses[current_idx+WINDOW_SIZE-1:current_idx+WINDOW_SIZE][LABEL_COL].values[0][0] == current_activity:
                end_idx = current_idx+WINDOW_SIZE
                next_idx = current_idx+WINDOW_OVERLAP
            else:
                activity_transition_points = raw_poses[current_idx:current_idx+WINDOW_SIZE][['label']]. \
                    ne(raw_poses[current_idx:current_idx+WINDOW_SIZE][['label']].shift().bfill()).astype(int)==1
                end_idx = raw_poses[current_idx:current_idx+WINDOW_SIZE].index[activity_transition_points['label']][0]
                next_idx = end_idx

            # Extract the curent window
            curwnd_raw_poses_cropped = raw_poses_cropped[current_idx:end_idx]
            curwnd_lr_wrist_dist = lr_wrist_dist[current_idx:end_idx]
            curwnd_r_wrist_to_nose_dist = r_wrist_to_nose_dist[current_idx:end_idx]

            # Compute the time dependent features
            window_time = (end_idx-current_idx)*FRAME_TIME
            r_wrist_velocity_y = (curwnd_raw_poses_cropped['rwrist_y'].tail(1).values[0] - curwnd_raw_poses_cropped['rwrist_y'].head(1).values[0])/window_time

            # Add features to output
            out_idx = len(out_data)
            out_data.loc[out_idx, BODY_PARTS_TO_USE] = curwnd_raw_poses_cropped.tail(1).values[0]
            out_data.loc[out_idx, LABEL_COL] = current_activity
            out_data.loc[out_idx, SECONDARY_FEATURES[0]] = curwnd_lr_wrist_dist.tail(1).values[0]
            out_data.loc[out_idx, SECONDARY_FEATURES[1]] = curwnd_r_wrist_to_nose_dist.tail(1).values[0]
            out_data.loc[out_idx, SECONDARY_FEATURES[2]] = r_wrist_velocity_y

            # Set the idx for next iteration
            current_idx = next_idx

        if idx == test_idx:
            test = test.append(out_data)    
        else:
            train = train.append(out_data)

    return test, train


if __name__ == '__main__':
    csv_paths=['../../../Work/videos/RealData/Chongyan.csv', '../../../Work/videos/RealData/Ershad.csv', '../../../Work/videos/RealData/Tim.csv']
    video_paths=['../../../Work/videos/RealData/Chongyan.MOV', '../../../Work/videos/RealData/Ershad.MOV', '../../../Work/videos/RealData/Tim.MOV']
    feature_csvs=['../../Work/videos/RealData/Chongyan_features.csv', '../../Work/videos/RealData/Ershad_features.csv', '../../Work/videos/RealData/Tim_features.csv']

    test, train = extract_test_train(feature_csvs, 2)
    print(test.shape)
    print(train.shape)
    
    combined = train.append(test)
    combined.to_csv('feature.csv',index=False)
    all_inputdata = np.array(combined.loc[:, 'nose_x':'r_wrist_velocity_y'])
    target = np.array(combined.loc[:, 'label'])
    Four_classifications(all_inputdata,target)