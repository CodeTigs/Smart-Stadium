import cv2
import mediapipe as mp
#import time
#import numpy as np
from math import sqrt,atan2, degrees
#from mediapipe.framework.formats import landmark_pb2

class point:
    def __init__(self,x,y,z):
        self.x, self.y, self.z = x,y,z

    def calculoDistancia(self,point):
        return sqrt((self.x-point.x)**2 + (self.y-point.y)**2 + (self.z-point.z)**2)
    
    def calculoAngulo(self, point):
        dy = abs(point.y - self.y)
        dxz = sqrt((self.x - point.x) ** 2 + (self.z - point.z) ** 2)
        radian_angle = atan2(dy, dxz)
        return degrees(radian_angle)


mp_pose = mp.solutions.pose
mp_hand = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
hand = mp_hand.Hands()

video = cv2.VideoCapture(0)

while True:
    success, img = video.read()
    img = cv2.flip(img, 1)
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    
    try:
        #processando corpo inteiro
        results = pose.process(frameRGB)
        posePoints = results.pose_landmarks
        if posePoints is not None:
            mp_drawing.draw_landmarks(img, posePoints, mp_pose.POSE_CONNECTIONS)
            #mp_drawing.draw_landmarks(img,landmark_subset,mp_pose.POSE_CONNECTIONS)
    except:
        pass

    try:
        #processando maos
        results2 = hand.process(frameRGB)
        handPoints = results2.multi_hand_landmarks
        if handPoints is not None:
            for points in handPoints:
                mp_drawing.draw_landmarks(img, points,mp_hand.HAND_CONNECTIONS)
    except:
        pass

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()