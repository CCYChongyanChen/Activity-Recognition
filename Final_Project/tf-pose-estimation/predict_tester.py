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
    rf2=joblib.load('clf.model')
    y=rf2.predict(x)
    return y
    
def extract_frames(video_path, csv_path):
    fps_time = 0

    # Read label time ranges
    rawdata = pd.read_csv(csv_path, header=0)
    rawdata.columns = ['start_time', 'end_time', 'label']
    
    cap = cv2.VideoCapture(video_path)
    #cap.set(cv2.CAP_PROP_POS_MSEC, 90*1000)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    videoout = cv2.VideoWriter('prediction.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    
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

    conf_matrix_labels = {'steering': 0, 'texting': 1, 'calling_right': 2, 'calling_left': 3, 'reading': 4, 'eating': 5}
    conf_matrix = np.zeros((len(conf_matrix_labels),len(conf_matrix_labels)))
    
    # Save a pose to csv every n frames
    counter = 0
    ma_counter = 0
    ma_length = 5 # Length of the moving average
    part_weight = 1.0/ma_length # Weight of a part in the moving average
    
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
    
    if cap.isOpened() is False:
        print("Error opening video stream or file")

    cv2.namedWindow('output',cv2.WINDOW_NORMAL) 
    cv2.resizeWindow('output', 576,1024)
    
    while cap.isOpened():
        ret_val, image = cap.read()
    
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0

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
            neck = bodypart_array[BODY_PARTS['Neck']]
            for idx, part in enumerate(bodypart_array):
                bodypart_locations[idx*2] = part.x - neck.x
                bodypart_locations[idx*2+1] = part.y - neck.y

            # Perform prediction
            predic=prediction(np.array([bodypart_locations]))

            # Add to confusion matrix
            if label.size != 0:
                # Check if the current label has persisted over the past n frames
                conf_matrix[conf_matrix_labels[label.values[0]], conf_matrix_labels[predic[0]]] += 1
                print(conf_matrix)

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        try:
            cv2.putText(image,"Prediction:"+str(predic), (10, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except:
            cv2.putText(image,"Prediction:", (10, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('output', image)
        #videoout.write(image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
    
    cv2.destroyAllWindows()
    cap.release()
    videoout.release()

if __name__ == '__main__':
    video_path='../../../Work/videos/RealData/Tim.MOV'
    csv_path='../../../Work/videos/RealData/Tim.csv'
    extract_frames(video_path, csv_path)