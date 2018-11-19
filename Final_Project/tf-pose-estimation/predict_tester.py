
from __future__ import print_function
import cv2
import numpy as np
import pandas as pd
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import tensorflow as tf
from matplotlib.pyplot import specgram
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.model_selection import KFold as KF
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
def prediction(x):
    rf2=joblib.load('../Work/Project/rf.model')
    y=rf2.predict(x)
    return y

def Four_classifications(X,Y):
    print(len(Y))
    confmatrix = np.zeros((5,5))
    kf = KF(len(Y),n_folds=10,random_state=None, shuffle=True)#!!!!!!!!!!!!!!!!!!!!!!!
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
def extract_frames_(csv_path,video_path,output_path):
    fps_time = 0
    
    # Read label time ranges
    #rawdata = pd.read_csv(csv_path, header=0)
    #rawdata.columns = ['start_time', 'end_time', 'label']
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, 170*1000)
    #cap.set(cv2.CAP_PROP_POS_MSEC, 194*1000)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    videoout = cv2.VideoWriter('prediction.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    
    # Setup model
    w = 432
    h = 243
    showBG = True
    e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))
    
    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
    
    # Save a pose to csv every n frames
    counter = 0
    n = 5
    last_label = ''
    
    # Array of all body part locations; Used for filtering out jitters of the posenet
    bodypart_array = []
    # Initialize array
    for i in range(0,10):
        bodypart_array.insert(i, BodyPart(0,0,0.5,0.5,0))

    bodypart_locations = np.zeros(2*10)
    out_data = pd.DataFrame(columns=['nose_x', 'nose_y', 'neck_x', 'neck_y', 'rshoulder_x', 'rshoulder_y', 
         'relbow_x', 'relbow_y', 'rwrist_x', 'rwrist_y', 'lshoulder_x', 'lshoulder_y', 'lelbow_x', 'lelbow_y', 
        'lwrist_x', 'lwrist_y', 'reye_x', 'reye_y', 'leye_x', 'leye_y', 'label'])
    
    if cap.isOpened() is False:
        print("Error opening video stream or file")
    
    while cap.isOpened():
        ret_val, image = cap.read()
    
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
        if timestamp > 575:
            break
    
        #label = rawdata['label'][(timestamp > rawdata['start_time']) & (timestamp < rawdata['end_time'])]
    
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=2)
        if len(humans) > 0:
            humans = [humans[0]]
            if not showBG:
                image = np.zeros(image.shape)
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    
            # Extract individual body parts (total of 10 parts)
            try:
                nose = humans[0].body_parts[BODY_PARTS['Nose']]
            except:
                nose = bodypart_array[0]
            try:
                neck = humans[0].body_parts[BODY_PARTS['Neck']]
            except:
                neck = bodypart_array[1]
            try:
                rshoulder = humans[0].body_parts[BODY_PARTS['RShoulder']]
            except:
                rshoulder = bodypart_array[2]
            try:
                relbow = humans[0].body_parts[BODY_PARTS['RElbow']]
            except:
                relbow = bodypart_array[3]
            try:
                rwrist = humans[0].body_parts[BODY_PARTS['RWrist']]
            except:
                rwrist = bodypart_array[4]
            try:
                lshoulder = humans[0].body_parts[BODY_PARTS['LShoulder']]
            except:
                lshoulder = bodypart_array[5]
            try:
                lelbow = humans[0].body_parts[BODY_PARTS['LElbow']]
            except:
                lelbow = bodypart_array[6]
            try:
                lwrist = humans[0].body_parts[BODY_PARTS['LWrist']]
            except:
                lwrist = bodypart_array[7]
            try:        
                reye = humans[0].body_parts[BODY_PARTS['REye']]
            except:
                reye = bodypart_array[8]
            try:
                leye = humans[0].body_parts[BODY_PARTS['LEye']]
            except:
                leye = bodypart_array[9]
    
            bodypart_array = [nose, neck, rshoulder, relbow, rwrist, lshoulder, lelbow, lwrist, reye, leye]
    
            for idx, part in enumerate(bodypart_array):
                bodypart_locations[idx*2] += part.x
                bodypart_locations[idx*2+1] += part.y
    
            if counter == n:
#                        print(label.values[0])
                out_data.loc[len(out_data), 'nose_x':'leye_y'] = bodypart_locations/n
                predic=prediction(np.array([bodypart_locations/n]))
                #out_data.loc[len(out_data)-1, 'label'] = label.values[0]
                counter = 0
                bodypart_locations = np.zeros(2*10)
            else:
                counter += 1

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        try:
            cv2.putText(image,"Prediction:"+str(predic), (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except:
            cv2.putText(image,"Prediction:", (10, 70),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        videoout.write(image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
    
    #out_data.to_csv(output_path,index=False)
    cv2.destroyAllWindows()
    cap.release()
    videoout.release()

if __name__ == '__main__':
    csv_path='../Work/videos/IMG_2431.csv'
    video_path='../Work/videos/IMG_2431.MOV'
    output_path='poses2.csv'
    extract_frames_(csv_path,video_path,output_path)
    #raw_poses = pd.read_csv('poses.csv', header=0)
    #raw_poses.columns = ['nose_x', 'nose_y', 'neck_x', 'neck_y', 'rshoulder_x', 'rshoulder_y', 
    #     'relbow_x', 'relbow_y', 'rwrist_x', 'rwrist_y', 'lshoulder_x', 'lshoulder_y', 'lelbow_x', 'lelbow_y', 
    #    'lwrist_x', 'lwrist_y', 'reye_x', 'reye_y', 'leye_x', 'leye_y', 'label']
    
    #all_inputdata = np.array(raw_poses.loc[:, 'nose_x':'leye_y'])
    #target = np.array(raw_poses.loc[:, 'label'])
    #Four_classifications(all_inputdata,target)
   