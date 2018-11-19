import cv2
import numpy as np
import pandas as pd
import csv
import time

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

fps_time = 0

# Read label time ranges
rawdata = pd.read_csv('../Work/videos/IMG_2431.csv', header=0)
rawdata.columns = ['start_time', 'end_time', 'label']

cap = cv2.VideoCapture('../Work/videos/IMG_2431.MOV')
cap.set(cv2.CAP_PROP_POS_MSEC, 170*1000)
#cap.set(cv2.CAP_PROP_POS_MSEC, 194*1000)

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
n = 10
last_label = ''

# Array of all body part locations; Used for filtering out jitters of the posenet
bodypart_array = []
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

    label = rawdata['label'][(timestamp > rawdata['start_time']) & (timestamp < rawdata['end_time'])]

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

        if label.size != 0:
            # Check if the current label has persisted over the past n frames
            if last_label != label.values[0]:
                counter = 0
                bodypart_locations = np.zeros(2*10)
            else:
                if counter == n:
                    print(label.values[0])
                    out_data.loc[len(out_data), 'nose_x':'leye_y'] = bodypart_locations/n
                    out_data.loc[len(out_data)-1, 'label'] = label.values[0]
                    counter = 0
                    bodypart_locations = np.zeros(2*10)
                else:
                    counter += 1
            last_label = label.values[0]


    cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #cv2.imshow('tf-pose-estimation result', image)
    fps_time = time.time()
    if cv2.waitKey(1) == 27:
        break

out_data.to_csv('poses.csv')
cv2.destroyAllWindows()
