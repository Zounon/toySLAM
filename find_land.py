import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from display import Display2D
from extractor import Extractor
import g2o 

W = 1920//2
H = 1080//2
F = 747 
K = np.array([[F,0,W//2], [0,F,H//2], [0,0, 1]])

disp = Display2D(W,H)
fe = Extractor(K)

orb = cv.ORB_create()


def process_frame(img): 
    img = cv.resize(img, (W,H))
    # kp = feature_extractor.extract(img)
    matches, pose = fe.extract2(img)
    if pose is None: 
        return
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # kp = cv.goodFeaturesToTrack(gray, 1000, 0.01, 10)

    # for i in kp: 
    #     x,y = map(lambda x: int(x), i.ravel())
    #     # print(x,y)
    #     cv.circle(img, (x,y),   color=(0,255,0), radius=3)
    # kp, des = orb.detectAndCompute(img, None)
    
    for pt1, pt2 in matches: 
        u1, v1 = map(lambda x: int(round(x)), pt1)
        u2, v2 = map(lambda x: int(round(x)), pt2)

        # denomalize for display 
        u1,v1 = fe.denormalize(pt1)
        u2,v2 = fe.denormalize(pt2)
        # print(u1,v1,u2,v2)
        cv.circle(img, (u1,v1), color=(0,255,0), radius=3)
        cv.line(img, (u1,v1),(u2,v2), color=(255,0,0))

    disp.paint(img)
    
    # disp.point(img)
    # cv.imshow('win', cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0))

if __name__ == "__main__":
    cap = cv.VideoCapture('test_countryroad.mp4')
    while cap.isOpened(): 
        ret, frame = cap.read()
        if ret == True: 
            process_frame(frame)
        else: 
            break



    # cv.waitKey(0)
    # cv.destroyAllWindows()
# img = cv.imread('./countryroad_frames/frame1.jpg')

# # init orb detector
# orb = cv.ORB_create()

# # find the keypoints with ORB 
# kp = orb.detect(img, None)

# # compute the descriptors with ORB
# kp, des = orb.compute(img, kp) 

# # draw only kp location
# img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
# plt.imshow(img2), plt.show()
# cv.imshow('display window',img)

# cv.waitKey(0)
# cv.destroyAllWindows()

