import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

cap = cv.VideoCapture('test_countryroad.mp4')
orb = cv.ORB_create()

def get_orbs(img): 

    kp, des = orb.detectAndCompute(img, None) 
    return cv.drawKeypoints(img, kp, None, color=(0,255,0))

while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret:
        print("can not receive frame")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    color = cv.cvtColor(frame, cv.IMREAD_COLOR)
    # w_orbs = get_orbs(color)
    # cv.circle(color, (516 , 920), color=(0,255,0), radius=3)
    cv.imshow('frame', get_orbs(color))

    if cv.waitKey(1) == ord('q'): 
        break


