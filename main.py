#!/usr/bin/env python3
import numpy as np
import os
import cv2 as cv
from collections import defaultdict
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform, FundamentalMatrixTransform
from display import Display2D
from matplotlib import pyplot as plt

from not_my_functions import extractRt, add_ones, normalize, denormalize

orb = cv.ORB_create()
class Map(object):
    def __init__(self):
        self.frames = []
        self.last_frame = None

    def add_frame(frame_obj):
        self.frames.append(frame_obj)
        self.last_frame = frames[-1]


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
        self.D = None

    def undistort_points(self, uvs): 
        # print('undistort_points:', uvs)
        # uvs_contiguous = np.ascontiguousarray(uvs[:, :2]).reshape((uvs.shape[0], 1, 2))
        # uvs_undistorted = cv.undistortPoints(uvs_contiguous, self.K, self.D, None, self.K)
        # ret = uvs_undistorted.ravel().reshape(uvs_undistorted.shape[0], 2)
        # print('post undistort: ', ret)

        # print('this is ret: ')
        # ret = np.dot(self.K, np.array([uvs[0], uvs[1], 1.0]))
        # print(ret)
        # # ret = int(round(ret[0])), int(round(ret[1]))
        # print(ret)

        ret = cv.undistortPoints(np.expand_dims(uvs, axis=1), self.K, self.D, None, self.K)
        ret = ret.ravel().reshape(ret.shape[0], 2)
        return ret

    def unproject_points(self, uvs):
        return np.dot(self.Kinv, self.add_ones(uvs).T).T[:, 0:2] 

    # turn [[x,y]] -> [[x,y,1]]
    def add_ones(self, x):
        if len(x.shape) == 1:
            #return np.concatenate([x,np.array([1.0])], axis=0)
            return np.array([x[0], x[1], 1])
            #return np.append(x, 1)
        else:
            return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

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

    def match_des_simple(self, des1, des2): 
        self.matches = self.matcher.knnMatch(des2, des1, k=2)
        for m,n in matches: 
            if m.distance < 0.75 * n.distance:
                kp1 = kps[m.queryIdx].pt
                kp2 = k
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

    def extract(self, img): 
        # to detect the features 
        feats = cv.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 500, qualityLevel=0.01, minDistance=3)


        # extract feats into keypoints
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        kps, des = orb.compute(img, kps)

        return kps, des


class Frame(object):
    def __init__(self, img):
        self.img_rgb = img
        slam.frames.append(self)
        self.img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.id = len(slam.frames) 
        print('new frame w/ id:', self.id)

class SLAM(object):
    def __init__(self, W, H, K):
        self.map = Map()
        self.frames = []
        self.W, self.H, self.K = W, H, K
        
        self.FIRST_IMAGE = True
        self.curr_frame, self.last_frame = None, None
        self.feature_matcher = FeatureMatcher()

    def process_frame(self, img):
        # -----------------------------------------
        # Step 1: Capture new frame as a Frame object
        if self.FIRST_IMAGE == True: 
            # Only the first frame exists
            self.curr_frame = Frame(img)
            self.last_frame = None

            # Process the only frame
            self.curr_frame.kps, self.curr_frame.des, self.curr_frame.pts = self.compute_kps_des_pts(self.curr_frame.img_rgb)
            # self.process_first_frame(self.curr_frame)
            disp2d.paint(img)
    
            # next frames will not be first
            self.FIRST_IMAGE = False 

            return 


        # -----------------------------------------
        # Step 2: Extract matches between current frame and the one before
        self.curr_frame = Frame(img)
        self.last_frame = self.frames[-2]

        # Get keypoints and features 
        self.curr_frame.kps, self.curr_frame.des, self.curr_frame.pts = self.compute_kps_des_pts(self.curr_frame.img_rgb)

        # Match keypoints of current and last frames
        self.curr_frame.matches = self.match_frames(self.last_frame, self.curr_frame)

        # Filter the match results
        idx1, idx2, Rt = self.filter_matches(self.curr_frame, self.last_frame)

        # annotate the frame with filtered match results 
        # self.annotate_frame(self.curr_frame)
        annotated_img = self.curr_frame.img_rgb.copy()
        print('\n')
        for pt1, pt2 in zip(self.curr_frame.pts[idx1], self.last_frame.pts[idx2]):
            # u1, v1 = normalize(Kinv, pt1)
            # u2, v2 = normalize(Kinv, pt2)
            u1, v1 = int(round(pt1[0])), int(round(pt1[1]))
            u2, v2 = int(round(pt2[0])), int(round(pt2[1]))
            cv.circle(annotated_img, (u1,v1), color=(0, 255, 0), radius=5)
            cv.line(annotated_img, (u1,v1), (u2, v2), (0, 255, 255), 3)

        print('num pts: ', len(pt1))
        disp2d.paint(annotated_img)
        # OLD: self.curr_frame.process_next_frame(self.last_frame, self.curr_frame)

        
    def compute_kps_des_pts(self, img):
        orb = cv.ORB_create()
        # bf = cv.BFMatcher(cv.NORM_HAMMING)

        # detection
        # turn img into 2D B/W and find features
        feats = cv.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 500, qualityLevel=0.01, minDistance=3)

        # from feats extract keypoints
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        kps, des = orb.compute(img, kps)

        # from keypoint objects extract (floating point) pts 
        pts = np.array([(kp.pt[0], kp.pt[1]) for kp in kps])

        # kps = orb.detect(img, None)
        # pts = np.array([x.pt for x in kps], dtype=np.float32)
        return kps, des, pts

    def match_frames(self, last_frame, curr_frame): 
        bf = cv.BFMatcher(cv.NORM_HAMMING)
        matches = bf.knnMatch(curr_frame.des, last_frame.des, k=2)
        return matches

    def filter_matches(self, curr_frame, last_frame):
        ret = []
        idx1, idx2 = [], []

        for m,n in curr_frame.matches: 
            if m.distance < 0.75 * n.distance:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)

                curr_point = curr_frame.pts[m.queryIdx]
                last_point = last_frame.pts[m.trainIdx]
                ret.append((curr_point, last_point))

        assert len(ret) >= 8
        ret = np.array(ret)
        idx1, idx2 = np.array(idx1), np.array(idx2)

        # normalize coords ???????????????
        ret[:, 0, :] = normalize(Kinv, ret[:, 0, :])
        ret[:, 1, :] = normalize(Kinv, ret[:, 1, :])


        # fit matrix ????????????????
        # print(ret[:, 0], ret[:, 1])
        # print(ret[:, 0].shape, ret[:, 1].shape)
        model, inliers = ransac((ret[:, 0], ret[:, 1]), 
                                FundamentalMatrixTransform,
                                # EssentialMatrixTransform, 
                                min_samples=8, 
                                residual_threshold=0.005, 
                                max_trials=200)

        # ignore outliers ????????????????
        Rt = extractRt(model.params)

        return idx1[inliers], idx2[inliers], Rt

    
    def process_first_frame(self, curr_frame): 
        curr_frame.kps, curr_frame.des = orb.detectAndCompute(curr_frame.img_grey, mask=None)

        # self.kps is a list of keypoint objects, we want the numberical (float) values
        # so we convert list of keypoints to array of points
        curr_frame.kps = np.array([x.pt for x in curr_frame.kps], dtype=np.float32)
        
        # paint the keypoints and display the result
        self.drawFeatures(None, curr_frame)


    def drawFeatures(self, last_frame, curr_frame): 
        annotated_img = curr_frame.img_rgb.copy()

        if slam.FIRST_IMAGE == True: 
            for point in curr_frame.kps:
                a,b = point.ravel()
                cv.circle(annotated_img, (int(a), int(b)), 5, (0,255,0), -1)

        else: 
            # TODO: implement draw features on next frames
            pass
        disp2d.paint(annotated_img)


        
       

    # def estimatePose(self, last_frame, curr_frame):
    #     self.adj_last_kps = cam.unproject_points( cam.undistort_points( last_frame.kps) ) 
    #     self.adj_curr_kps = cam.unproject_points( cam.undistort_points( curr_frame.kps) ) 

    #     # TODO Might have to set adjusted kps to the normal kps
    #     last_frame.kpn = self.adj_last_kps
    #     curr_frame.kpn = self.adj_curr_kps

    #     E, self.mask = cv.findEssentialMat(
    #                                         curr_frame.kpn, last_frame.kpn, focal=1, 
    #                                         pp=(0., 0.), method=cv.RANSAC, prob=0.999, 
    #                                         threshold=0.0003
    #                                         )
    #     _, R, t, mask = cv.recoverPose(E, curr_frame.kpn, last_frame.kpn, focal=1, pp=(0.,0.))
    #     
    #     # E, self.mask = cv.findEssentialMat(
    #     #                                     self.adj_curr_kps, self.adj_last_kps, focal=1, 
    #     #                                     pp=(0., 0.), method=cv.RANSAC, prob=0.999, 
    #     #                                     threshold=0.0003
    #     #                                     )
    #     # _, R, t, mask = cv.recoverPose(E, self.adj_curr_kps, self.adj_last_kps, focal=1, pp=(0.,0.))
    #     return R,t

    # 
    # def process_next_frame(self, last_frame, curr_frame): 
    #     # self.kps, self.des = orb.detectAndCompute(self.img_grey, mask=None)
    #     self.kps, self. des = slam.feature_matcher.extract(curr_frame.img_rgb)

    #     # self.kps is a list of keypoint objects, we want the numberical (float) values
    #     # so we convert list of keypoints to array of points
    #     # self.kps = np.array([x.pt for x in self.kps], dtype=np.float32)

    #     curr_frame.matches = []
    #     # curr_frame.matches = slam.feature_matcher.match_des(last_frame.des, curr_frame.des) # last des are query, curr des are train
    #     curr_frame.matches = slam.feature_matcher.match_des_simple(last_frame.des, curr_frame.des)
    #     print(len(curr_frame.matches))
    #     # estimate pose 
    #     #R, t = self.estimatePose(last_frame, curr_frame)
    #     self.drawFeatures(last_frame, curr_frame) 




        



        


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

