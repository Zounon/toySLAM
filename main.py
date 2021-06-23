#!/usr/bin/env python3
import numpy as np
import os
import cv2 as cv
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform
from display import Display2D
from matplotlib import pyplot as plt

orb = cv.ORB_create()
# class Map(object):
#     def __init__(self):
#         self.frames = []
#         self.last_frame = None

#     def add_frame(frame_obj):
#         self.frames.append(frame_obj)
#         self.last_frame = frames[-1]

# class Frame(object):
#     def __init__(self):
#         self.K = np.array(K)
#         self.pose = None
#         self.h, self.w = None, None
#         map.add_frame(self)

# turn [[x,y]] -> [[x,y,1]]
class Map(object):
    def __init__(self) -> None:
        self.frames = []


class Camera(object):
    def __init__(self, W, H, K, F):
        self.W = W
        self.H = H
        self.K = K
        self.Kinv = np.linalg.inv(K)
        self.F = F


    def undistort_points(self, uvs): 
        uvs_contiguous = np.ascontiguousarray(uvs[:, :2]).reshape((uvs.shape[0], 1, 2))
        uvs_undistorted = cv.undistortPoints(uvs_contiguous, self.K, self.D, None, self.K)
        return uvs_undistorted.ravel().reshape(uvs_undistorted.shape[0], 2)

    def unproject_points(self, uvs):
        return np.dot(self.Kinv, add_ones(uvs).T).T[:, 0:2] 

class FeatureMatcher(object): 
    def __init__(self):
        self.matches = []
        self.good_matches = []
        # Using BruteForceMatcher with Hamming and No crosscheck
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, False) 

    def match_des(self, des1, des2): 
        self.matches = self.matcher.knnMatch(des1, des2, k=2)
        self.good_matches = self.getGoodMatches(self.matches, des1, des2)
        return self.good_matches

    # input: des1 are query discriptors, des2 are train descriptors 
    # output: idx1, idx2 are vectors of indcies for the good matches in des1, des2
    # the func checks to ensure that each trainIdx has exactly one queryIdx
    def getGoodMatches(self, matches, des1, des2): 
        len_des2 = len(des2)
        idx1, idx2 = [], []
        ratio = 0.7

        if matches is not None: 
            float_inf = float('inf')
            dist_match = defaultdict(lambda: float_inf)
            index_match = dict()

            for m,n in matches: 
                if m.distance > ratio * n.distance: 
                    continue 

                dist = dist_match[m.trainIdx]
                if dist == float_inf: 
                    # we know trainIdx has not been matched
                    dist_match[m.trainIdx] = m.distance
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    index_match[m.trainIdx] = len(idx2)-1
                else: 
                    if m.distance < dist: 
                        # we already have a match for trainIdx, 
                        # if the stored match is worse, replace it
                        index = index_match[m.trainIdx]
                        assert(idx2[index] == m.trainIdx)
                        idx1[index] = m.queryIdx
                        idx2[index] = m.trainIdx
        return idx1, idx2

class Frame(object):
    def __init__(self, img):
        self.img_rgb = img
        slam.frames.append(self)
        self.img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.id = len(slam.frames) 

        # self.kps, self.des, self.pts = self.compute_kp_des(self.img)
        # self.good_matches = None

        # if self.id < 5:
        #     self.annotated_img = cv.drawKeypoints(img, self.kps, img)
        # else:
        #     self.annotated_img = self.annotateImg(
        #                             self.img, slam.frames[-2].kps, slam.frames[-1].kps)

    def estimatePose(self, last_kps, curr_kps):
        self.adj_last_kps = self.cam.unproject_points( self.cam.undistort_points( last_kps) ) 
        self.adj_curr_kps = self.cam.unproject_points( self.cam.undistort_points( curr_kps) ) 
        
        E, self.mask = cv.findEssentialMat(
                                            self.adj_curr_kps, self.adj_last_kps, focal=1, 
                                            pp=(0., 0.), method=cv.RANSAC, prob=0.999, 
                                            kRansacThresholdNormalized=0.0003
                                            )

    def process_first_frame(self): 
        self.kps, self.des = orb.detectAndCompute(self.img_grey, mask=None)

        # self.kps is a list of keypoint objects, we want the numberical (float) values
        # so we convert list of keypoints to array of points
        self.kps = np.array([x.pt for x in self.kps], dtype=np.float32)
        
        # paint the keypoints and display the result
        self.drawFeatures()
    
    def process_next_frame(self, last_frame, curr_frame): 
        curr_frame.matches = []
        curr_frame.matches = cv.knnMatch(last_frame.des, curr_frame.des, k=2) # last des are query, curr des are train



        # estimate pose 
        R, t = self.estimatePose(last_frame.kps, curr_frame.kps)
        
        pass


    def drawFeatures(self): 
        annotated_img = self.img_rgb.copy()
        if slam.FIRST_IMAGE == True or True: 
            for point in self.kps:
                a,b = point.ravel()
                cv.circle(annotated_img, (int(a), int(b)), 5, (0,255,0), -1)
        disp2d.paint(annotated_img)



    def compute_kp_des(self, img):
        orb = cv.ORB_create()
        bf = cv.BFMatcher(cv.NORM_HAMMING)
        kps = orb.detect(img, None)
        kps, des = orb.compute(img, kps)
        pts = np.array([x.pt for x in kps], dtype=np.float32)
        return kps, des, pts

        # ret = []
        # idx1, idx2 = [], []
        # idx1s, idx2s = set(), set()
        # if len(slam.frames) > 2:
        #     matches = bf.knnMatch(des, slam.frames[-2].des, k=2)
        #     for m,n in matches:
        #         if m.distance < 0.75*n.distance:
        #             if m.queryIdx not in idx1s and m.trainIdx not in idx2s
        #                 idx1.append(m.queryIdx)
        #                 idx2.append(m.trainIdx)
        #                 idx1s.add(m.queryIdx)
        #                 idx2s.add(m.trainIdx)
        #                 pt1 = slam.frames[-1].kps[m.queryIdx].pt
        #                 pt2 = slam.frames[-2].kps[m.trainIdx].pt
        #                 ret.append((pt1, pt2))
        # else:
        #     return kps, des, np.array(pts)
        # idx1 = np.array(idx1)
        # ret = np.array(ret)
        # return kps, des, ret

class SLAM(object):
    def __init__(self, W, H, K):
        self.map = Map()
        self.frames = []
        self.W, self.H, self.K = W, H, K
        
        self.FIRST_IMAGE = True
        self.curr_frame, self.last_frame = None, None

    def process_frame(self, img):
        # -----------------------------------------
        # Step 1: Capture new frame as a Frame object
        if self.FIRST_IMAGE == True: 
            self.curr_frame = Frame(img)
            self.FIRST_IMAGE = False 
            self.curr_frame.process_first_frame()
            return 

        else: 
            self.curr_frame = Frame(img)


        # -----------------------------------------
        # Step 2: Extract matches between current frame and the one before
        self.last_frame = self.frames[-2]
        self.curr_frame.process_next_frame(self.last_frame, self.curr_frame)

        
        



        


if __name__ == "__main__":
    cap = cv.VideoCapture('test_countryroad.mp4')

    # camera parameters
    W = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    CNT = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    F = float(os.getenv("F", "525"))
    if W > 1024:
        downscale = 1024.0/W
        F *= downscale
        H = int(H * downscale)
        W = 1024
    print("using camera %dx%d with F %f" % (W, H, F))
    # camera intrinsics
    K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])
    Kinv = np.linalg.inv(K)

    cam = Camera(W, H, K, F)
    disp2d = Display2D(W, H)
    slam = SLAM(W, H, K)

    while cap.isOpened():
        ret, img = cap.read()
        img = cv.resize(img, (W, H))

        if ret == True:
            slam.process_frame(img)
        else:
            break

