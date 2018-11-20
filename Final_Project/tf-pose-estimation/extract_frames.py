
from __future__ import print_function
import cv2
import numpy as np
import pandas as pd
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import tensorflow as tf
from matplotlib.pyplot import specgram
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tf_pose import common
from tf_pose.estimator import TfPoseEstimator, BodyPart
from tf_pose.networks import get_graph_path, model_wh

def Four_classifications(X,Y):
    print(len(Y))
    confmatrix = np.zeros((5,5))
    kf = KFold(n_splits=10,random_state=None, shuffle=True)#!!!!!!!!!!!!!!!!!!!!!!!
    SVM_average_acc=[]
    GNB_average_acc=[]
    RF_average_acc=[]
    MLP_average_acc=[]
    for train_index, test_index in kf.split(Y):
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
        joblib.dump(rf,'rf.model')

        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200, 250), 
                        random_state=1, verbose=False)
        clf.fit(X_train, Y_train)
        predictions = clf.predict(X_test)
#        print(confusion_matrix(Y_test, predictions))
        confmatrix+= confusion_matrix(Y_test, predictions) 
        joblib.dump(clf,'clf.model')     
    
        MLP_acc = clf.score(X_test,Y_test)
        MLP_average_acc.append(MLP_acc)
        
    print("SVM cross-validation Accuracy:",np.mean(SVM_average_acc))
    print("GNB cross-validation Accuracy:",np.mean(GNB_average_acc))
    print("RF cross-validation Accuracy:",np.mean(RF_average_acc))
    print("MLP cross-validation Accuracy:",np.mean( MLP_average_acc))
    print(confmatrix)
    """
    Split-test-split
    """
#    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state=0)
#    clf=SVC(C=1,kernel='rbf')
#    clf.fit(x_train,y_train)
#    SVM_s_acc=clf.score(x_test,y_test)
#    print("SVM split train/test Accuracy:",SVM_s_acc)
#    gnb=GaussianNB()
#    gnb.fit(x_train,y_train)
#    GNB_s_acc=gnb.score(x_test,y_test)
#    print("GBN split train/test Accuracy:",GNB_s_acc)
#    rf=RandomForestClassifier(n_estimators=100)
#    rf.fit(x_train,y_train)
#    RF_s_acc=rf.score(x_test,y_test)
#    print("RF Split train/test Accuracy:",RF_s_acc)
#    clf2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
#    clf2.fit(x_train, y_train)
#    MLP_s_acc = clf2.score(x_test,y_test)
#    
#    print("MLP Split train/test Accuracy:",MLP_s_acc)!
    
    return True
def extract_frames(csv_path,video_path,output_path):
    fps_time = 0
    
    # Read label time ranges
    rawdata = pd.read_csv(csv_path, header=0)
    rawdata.columns = ['start_time', 'end_time', 'label']
    
    cap = cv2.VideoCapture(video_path)
    #cap.set(cv2.CAP_PROP_POS_MSEC, 170*1000)
    #cap.set(cv2.CAP_PROP_POS_MSEC, 194*1000)
    
    # Setup model
    w = 243
    h = 432
    showBG = True
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))
    
    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

    BODY_PARTS_INTEREST = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "REye": 14,
                    "LEye": 15}
    
    # Save a pose to csv every n frames
    counter = 0
    ma_counter = 0
    n = 5
    ma_length = 5 # Length of the moving average
    part_weight = 1.0/ma_length # Weight of a part in the moving average
    last_label = ''
    
    # Moving average of all body part locations; Used for filtering out jitters of the posenet
    bodypart_array_new = [] # Current set of bodyparts
    bodypart_array_ma = [] # Moving average set of bodyparts
    ma_bodyparts_array = [] # Sets of bodyparts for computing the moving average
    bodypart_to_del_idx = ma_length - 1; # Index of the set of bodyparts to overwrite in ma_bodyparts_array
    # Initialize arrays
    for i in range(0,10):
        bodypart_array_new.insert(i, BodyPart(0,0,0.5,0.5,0))
        bodypart_array_ma.insert(i, BodyPart(0,0,0,0,0)) # Must be all zeros
    for i in range(0,ma_length):
        ma_bodyparts_array.insert(i, bodypart_array_ma)

    bodypart_locations = np.zeros(2*10)
    out_data = pd.DataFrame(columns=['nose_x', 'nose_y', 'neck_x', 'neck_y', 'rshoulder_x', 'rshoulder_y', 
         'relbow_x', 'relbow_y', 'rwrist_x', 'rwrist_y', 'lshoulder_x', 'lshoulder_y', 'lelbow_x', 'lelbow_y', 
        'lwrist_x', 'lwrist_y', 'reye_x', 'reye_y', 'leye_x', 'leye_y', 'label'])
    
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    
    while cap.isOpened():
        ret_val, image = cap.read()
    
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
        #if timestamp > 575:
        #    break
    
        label = rawdata['label'][(timestamp > rawdata['start_time']) & (timestamp < rawdata['end_time'])]
        try:
            humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=2)
        except:
            break
        if len(humans) > 0:
            humans = [humans[0]]

            # Extract body part of interest
            # If a part is not detected, use the part's last location
            for idx, key in enumerate(BODY_PARTS_INTEREST):
                try:
                    bodypart_array_new[idx] = humans[0].body_parts[BODY_PARTS[key]]
                except:
                    bodypart_array_new[idx] = bodypart_array_ma[idx]
    
            # Perform moving average
            if ma_counter == ma_length:
                bodypart_array_delete = ma_bodyparts_array[bodypart_to_del_idx]
                for idx, part in enumerate(bodypart_array_ma):
                    part.x = part.x + bodypart_array_new[idx].x*part_weight - \
                        bodypart_array_delete[idx].x*part_weight
                    part.y = part.y + bodypart_array_new[idx].y*part_weight - \
                        bodypart_array_delete[idx].y*part_weight
                    bodypart_array_ma[idx] = part
                ma_bodyparts_array[bodypart_to_del_idx] = copy.deepcopy(bodypart_array_new)
                bodypart_to_del_idx = (bodypart_to_del_idx - 1)%ma_length
                bodypart_array = bodypart_array_ma
            else:
                # We have not seen enough samples yet for the moving average
                # Just use the current sample as the output of the moving average filter
                bodypart_array = bodypart_array_new
                ma_counter = ma_counter + 1
                ma_bodyparts_array[ma_length - ma_counter] = copy.deepcopy(bodypart_array_new)
                for idx, part in enumerate(bodypart_array_ma):
                    part.x += bodypart_array_new[idx].x*part_weight
                    part.y += bodypart_array_new[idx].y*part_weight
                    bodypart_array_ma[idx] = part

            # Update the humans array with the moving average data
            humans[0].body_parts = {}
            for idx, (key, part) in enumerate(zip(BODY_PARTS_INTEREST, bodypart_array)):
                humans[0].body_parts[BODY_PARTS[key]] = part

            # Draw the skeleton (only the parts of interest)
            if not showBG:
                image = np.zeros(image.shape)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    
            # Generate feature vector
            for idx, part in enumerate(bodypart_array):
                bodypart_locations[idx*2] = part.x
                bodypart_locations[idx*2+1] = part.y

            # Add label and append to feature matrix
            if label.size != 0:
                # Check if the current label has persisted over the past n frames
                if last_label != label.values[0]:
                    counter = 0
                else:
                    if counter == n:
                        print(label.values[0])
                        out_data.loc[len(out_data), 'nose_x':'leye_y'] = bodypart_locations
                        out_data.loc[len(out_data)-1, 'label'] = label.values[0]
                        counter = 0
                    else:
                        counter += 1
                last_label = label.values[0]

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        try:
            cv2.putText(image,"Prediction:"+str(predic), (10, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except:
            cv2.putText(image,"Prediction:", (10, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
    
    out_data.to_csv(output_path,index=False)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    csv_path='../../../Work/videos/RealData/Chongyan.csv'
    video_path='../../../Work/videos/RealData/Chongyan.MOV'
    output_path='../../../Work/Project/Chongyan_poses.csv'
    #extract_frames(csv_path,video_path,output_path)

    raw_poses = pd.read_csv(output_path, header=0)
    raw_poses.columns = ['nose_x', 'nose_y', 'neck_x', 'neck_y', 'rshoulder_x', 'rshoulder_y', 
         'relbow_x', 'relbow_y', 'rwrist_x', 'rwrist_y', 'lshoulder_x', 'lshoulder_y', 'lelbow_x', 'lelbow_y', 
        'lwrist_x', 'lwrist_y', 'reye_x', 'reye_y', 'leye_x', 'leye_y', 'label']
    
    all_inputdata = np.array(raw_poses.loc[:, 'nose_x':'leye_y'])
    target = np.array(raw_poses.loc[:, 'label'])
    Four_classifications(all_inputdata,target)
   