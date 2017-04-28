##################################
# CS 403 project
# python script to find emotion from an image
##################################

from utility_functions import *

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import joblib
import dlib

from tabulate import tabulate

# emotion classes
categories = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']

# Dlib face detector
detector = dlib.get_frontal_face_detector()

# Facial landmarks detector
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def get_landmarks(_image):
    """
    takes in an image, detects face in it,
    if face is detected then finds facial landmarks of that face,
    extracts some features as explained in code
    return : list of features corresponding to the detected face,
             if no face is detected, returns "error"
    """
    image = _image.copy()
    detections = detector(image, 1)
    
    # for all detected face instances individually
    for k, d in enumerate(detections):
        
        # get Facial Landmarks with the predictor class
        shape = predictor(image, d)
        
        xlist = []
        ylist = []
        # store X and Y coordinates in two lists
        for i in range(1, 68):
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
                
        xmean = xlist[29]
        ymean = ylist[29]
        
        # get distance between each point and the central point in both axes
        xcentral = [(x - xmean) for x in xlist]
        ycentral = [(y - ymean) for y in ylist]
        
        # point 29 - nose tip
        # point 26 - middle point b/w two eyes
        
        # get angle that nose line makes with vertical, this for correcting tilted faces.
        angle_nose = np.arctan2((ylist[26] - ymean), (xlist[26] - xmean)) * 180 / np.pi
        if angle_nose < 0:
            angle_nose += 90
        else:
            angle_nose -= 90
        
        # landmarks_vectorised is the feature vector corresponding to the face
        """
        landmarks_vectorised is of the form
        [p1_x_rel, p1_y_rel, dist(p, centre), angle()]
        """
        landmarks_vectorised = []
        for i in range(0, 67):
            """
            rx - x coordinate of relative position of a point w.r.t central point
            ry - y coordinate of relative position of a point w.r.t central point
            x - absolute x coordinate
            y - absolute y coordinate
            """
            
            rx = xcentral[i]
            ry = ycentral[i]
            x = xlist[i]
            y = ylist[i]
        
            landmarks_vectorised.append(rx)
            landmarks_vectorised.append(ry)

            # calculate length of point i from central point
            dist = np.linalg.norm(np.array([rx, ry]))
            landmarks_vectorised.append(dist)

            # get the angle the vector describes relative to the image, 
            # corrected for the offset that the nosebrigde has when the face is titled
            anglerelative = (np.arctan2((-ry), (-rx)) * 180 / np.pi) - angle_nose
            landmarks_vectorised.append(anglerelative)
        
    if len(detections) < 1: 
        landmarks_vectorised = "error"     
    
    return landmarks_vectorised

def test_external_image(image_path, model_path):
    image = cv.imread(image_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)

    # plt.imshow(clahe_image)
    # plt.show()

    features = get_landmarks(clahe_image)

    clf = joblib.load(model_path)
    predicted_labels = clf.predict([features])

    print(categories[predicted_labels[0]])
    df = pd.DataFrame(clf.predict_proba([features]), columns=categories)
    print(tabulate(df, headers='keys', tablefmt='psql'))

def print_usage():
	print('''[error] : run this script with 2 arguments :\n$ python face_expression.py <image_location> <trained_model_location>''')

if __name__ == '__main__':
	
	# usage --> python face_expression.py <image_location> <trained_model_location>

	# check number of args
	if (len(sys.argv) < 3):
		print_usage()
		sys.exit(0)

	image_loc = sys.argv[1]
	model_loc = sys.argv[2]

	if not os.path.exists(image_loc):
		print('[error] : image does not exist')
		sys.exit(0)

	if not os.path.exists(model_loc):
		print('[error] : models does not exist')
		sys.exit(0)

	test_external_image(image_loc, model_loc)














