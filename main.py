#!/usr/bin/env python3
import numpy as np
import os
import cv2 as cv

import g2o
from multiprocessing import Process, Queue

from collections import defaultdict
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform, FundamentalMatrixTransform
from display import Display2D, Display3D
from matplotlib import pyplot as plt
from not_my_functions import extractRt, add_ones, normalize, denormalize, fundamentalToRt, poseRt

'''
TODO next time

cant triangulate points
I think pose estimation is the problem,
might need to implement g2o optimizer to get decent pose
'''
orb = cv.ORB_create()

import numpy
import g2o

def g2oOptimizer(frames, points, local_window, fix_points, verbose=False, round=50):
    if local_window is None: 
        local_frames = frames
    else:
        local_frames = frames[-local_window:]

    # create g2o optimizer
    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    opt.set_algorithm(solver)

    # add normalized camera
    cam = g2o.CameraParameters(1.0, (0.0, 0.0), 0)
    cam.set_id(0)
    opt.add_parameter(cam)

    robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
    graph_frames, graph_points = dict(), dict()

    # add frames to graph
    for frame in (local_frames if fix_points else frames): 
        pose = frame.pose
        se3 = g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3])
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_estimate(se3)

        v_se3.set_id(frame.id * 2)
        v_se3.set_fixed(frame.id <= 1 or frame not in local_frames)
        opt.add_vertex(v_se3)

        # confirm pose correctness
        est = v_se3.estimate()
        assert np.allclose(pose[0:3, 0:3], est.rotation().matrix())
        assert np.allclose(pose[0:3, 3], est.translation())

        graph_frames[frame] = v_se3

    # add points to frames
    for point in slam.points: 
        # check that new point is not already a part of the frame
        if not any([f in local_frames for f in point.frames]):
            continue

        graph_point = g20.VertexSBAPointXYZ()
        graph_point.set_id(point.id  * 2 + 1)
        graph_point.set_estimate(point.pt[0:3])
        graph_point.set_marginalized(True)
        graph_point.set_fixed(fix_points)
        opt.add_vertex(graph_point)
        graph_points[point] = graph_point

        # add edges
        for frame, idx in zip(point.frames, point.idxs):
            if frame not in graph_frames:
                continue
            edge = g2o.EdgeProjectP2MC()
            edge.set_vertex(0, 0)
            edge.set_vertex(1, graph_frames[frame]) 
            edge.set_measurement(frame.points[idx])
            edge.set_information(np.eye(2))
            edge.set_robust_kernel(robust_kernel)
            opt.add_edge(edge)
        
    if verbose:
        opt.set_verbose(True)
    opt.initialize_optimization()
    opt.optimize(30)

    # put frames back
    for gf in graph_frames: 
        est = graph_frames[gf].estimate()
        R = est.rotation().matrix()
        t = est.translation()
        gf.pose = poseRt(R,t)

    # put points back
    if not fix_points:
        for gp in graph_points:
            gp.pt = np.array(graph_points[gp].estimate())

    return opt.active_chi2()


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

    def filter_matches(self, last_frame, curr_frame):
        ret = []
        idx1, idx2 = [], []

        for m,n in curr_frame.matches: 
            if m.distance < 0.75 * n.distance:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)

                curr_point = curr_frame.pts[m.queryIdx]
                last_point = last_frame.pts[m.trainIdx]
                ret.append((last_point,curr_point))

        assert len(ret) >= 8
        ret = np.array(ret)
        idx2, idx1 = np.array(idx2), np.array(idx1)

        # normalize coords ???????????????
        ret[:, 0, :] = normalize(Kinv, ret[:, 0, :])
        ret[:, 1, :] = normalize(Kinv, ret[:, 1, :])

        # fit matrix ????????????????
        model, inliers = ransac((ret[:, 0], ret[:, 1]), 
                                # FundamentalMatrixTransform,
                                EssentialMatrixTransform, 
                                min_samples=8, 
                                residual_threshold=0.005, 
                                max_trials=200)

        # ignore outliers ????????????????
        Rt = extractRt(model.params)

        return idx1[inliers], idx2[inliers], fundamentalToRt(model.params)

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
        ratio = 0.75

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

    def estimatePose(self, last_frame, curr_frame):
        # undistort, unproject points?  
        kRansacThresholdNormalized = 0.0003
        kRansacProb = 0.999

        E, self.mask_match = cv.findEssentialMat(curr_frame.kps, last_frame.kps, focal=1, pp=(0., 0.), 
                                                mothod=cv.RANSAC, prob=kRansacProb, threshold=kRansacThresholdNormalized)
        _, R, t, mask = cv.recoverPose(E, curr_frame.kpn, last_frame.kpn, focal=1, pp=(0., 0.))
        return R, t

class Frame(object):
    def __init__(self, img):
        self.img_rgb = img
        slam.frames.append(self)
        self.img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # self.id = len(slam.frames) 
        self.id = slam.add_frame(self)
        IRt = np.eye(4)
        self.pose = np.array(IRt)
        # print('new frame w/ id:', self.id)

class Map(object):
    def __init__(self):
        self.frames = []
        self.last_frame = None

    def add_frame(frame_obj):
        self.frames.append(frame_obj)
        self.last_frame = frames[-1]

    # def optimize(self):
    #     optimizer = g2o.SparseOptimizer()
    #     solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    #     solver = g2o.OptimizationAlgorithmLevenberg(solver)
    #     optimizer.set_algorithm(solver)

    #     # super().initialize_optimization()
    #     # super().optimize(max_iterations)
    # 
    #     # ????
    #     robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))

    #     # add frames to graph
    #     for frame in self.frames: 
    #         sbacam = g2o.SBACam(g2o.SE3Quat(frame.pose[0:3, 0:3], frame.pose[0:3, 3]))
    #         sbacam.set_cam(frame.K[0][0],frame.K[1][1],frame.K[2][0],frame.K[2][1], 1.0)

    #         v_se3 = g2o.VertexSE3()
    #         v_se3.set_id(frame.id)
    #         v_se3.set_estimate(sbacam)
    #         v_se3.set_fixed(frame.id == 0)
    #         opt.add_vertex(v_se3)
    # 
    #     # add points to frames
    #     for point in self.points: 
    #         graph_point = g20.VertexSBAPointXYZ()
    #         graph_point.set_id(point.id + 0x100000)
    #         graph_point.set_estimate(point.pt[0:3])
    #         graph_point.set_marginalized(True)
    #         graph_point.set_fixed(False)
    #         opt.add_vertex(graph_point)
    #         for frame in point.frames:
    #               for f in p.frames:
    #         edge = g2o.EdgeProjectP2MC()
    #         edge.set_vertex(0, pt)
    #         edge.set_vertex(1, opt.vertex(f.id))
    #         uv = f.kps[f.pts.index(p)]
    #         edge.set_measurement(uv)
    #         edge.set_information(np.eye(2))
    #         edge.set_robust_kernel(robust_kernel)
    #         opt.add_edge(edge)
    #         
    #     opt.set_verbose(True)
    #     opt.initialize_optimization()
    #     opt.optimize(20)

class Point(object): 
    # Each point object represents a 3D point in the world
    # each point should be observed in multiple frames 

    def __init__(self, location, color): 
        self.location = location
        self.color = color
        # to keep track of each frame and frame index that the point appears in 
        self.keyframe_observation = dict() 
        self.frames = []
        self.idxs = []
        self.id = slam.add_point(self)

    def add_observation(self, frame, index):
        # self.past.append((frame, index))
        self.frames.append(frame) 
        self.idxs.append(index)


class SLAM(object):
    def __init__(self, W, H, K):
        self.map = Map()
        self.fm = FeatureMatcher()
        self.frames = []
        self.points = []
        self.max_point = 0
        self.max_frame = 0
        self.W, self.H, self.K = W, H, K
        
        self.FIRST_IMAGE = True
        self.curr_frame, self.last_frame = None, None
        self.feature_matcher = FeatureMatcher()

    def add_frame(self, frame): 
        i = self.max_frame
        self.max_frame += 1
        self.frames.append(frame)
        return i

    def add_point(self, point): 
        i = self.max_point
        self.max_point += 1
        self.points.append(point)
        return i
    
    def triangulate(self, pose1, pose2, pts1, pts2):
      ret = np.zeros((pts1.shape[0], 4))
      for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4,4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
      return ret

    def process_frame(self, img):
        # -----------------------------------------
        # Step 1: Capture new frame as a Frame object
        if self.FIRST_IMAGE == True: 
            # Only the first frame exists
            self.curr_frame = Frame(img)
            curr_frame = self.curr_frame
            self.last_frame = None
            # Process the only frame
            # Get keypoints and features 
            kps, des, pts = self.fm.compute_kps_des_pts(self.curr_frame.img_rgb)
            pts = normalize(Kinv, pts)
            curr_frame.kps, curr_frame.des, curr_frame.pts = kps, des, pts

            # Not Needed: self.curr_frame.kps, self.curr_frame.des, self.curr_frame.pts = self.compute_kps_des_pts(self.curr_frame.img_rgb)

            # self.process_first_frame(self.curr_frame)
            self.annotate_frame(None, self.curr_frame)
    
            # next frames will not be first
            self.FIRST_IMAGE = False 

            return 


        # -----------------------------------------
        # Step 2: Extract matches between current frame and the one before
        self.curr_frame = Frame(img)
        self.last_frame = self.frames[-2]

        curr_frame = self.curr_frame
        last_frame = self.last_frame

        # Get keypoints and features 
        kps, des, pts = self.fm.compute_kps_des_pts(self.curr_frame.img_rgb)
        curr_frame.kps, curr_frame.des, curr_frame.pts = kps, des, pts
        curr_frame.pts_array = [None] * len(curr_frame.pts)

        # Match keypoints of current and last frames
        curr_frame.matches = self.fm.match_frames(last_frame, curr_frame)

        # Filter the match results
        # estiamte the essential matrix
        # TODO? Make idx1, idx2, Rt an antribute of each Frame? 
        idx1, idx2, Rt = self.fm.filter_matches(last_frame, curr_frame)
        curr_frame.idx1, curr_frame.idx2, curr_frame.Rt = idx1, idx2, Rt
        # print(Rt)

        pose_opt = g2oOptimizer(slam.frames, slam.points, local_window=1, fix_points=True)
        print(pose_opt)
        curr_frame.pose = np.dot(Rt, last_frame.pose)

        # print(last_frame.pose)
        # print(curr_frame.pose)
        # print(len(last_frame.pts[idx2]))
        # print(len(curr_frame.pts[idx1]))

        # might need to swap last and curr frame position
        pts4d = self.triangulate(last_frame.pose, curr_frame.pose, curr_frame.pts[idx1], last_frame.pts[idx2])
        print('\n')
        print('len pts4d', len(pts4d))
        good_pts4d = np.array([curr_frame.pts_array[i] is None for i in idx1])
        good_pts4d &= np.abs(pts4d[:, 3]) != 0
        pts4d /= pts4d[:, 3:] # make the pts homogenous 
        
        new_pts_count = 0
        b1 = 0
        b2 = 0
        b3 = 0
        b4 = 0
        for i,point in enumerate(pts4d):
            if not good_pts4d[i]:
                b1+=1
                continue

            # check that points are infront of both cameras
            # is this measuring paralax? 
            curr_paralax = np.dot(curr_frame.pose, point)
            last_paralax = np.dot(last_frame.pose, point)
            print('\n')
            print(curr_frame.pose)
            print(last_frame.pose)
            print('\n')
            print(curr_paralax) 
            print(last_paralax)
            print('\n')
            if curr_paralax[2] < 0 or last_paralax[2] < 0:
                b2+=1
                continue

            # reproject 
            new_curr_point = np.dot(self.K, curr_paralax[:3])
            new_last_point = np.dot(self.K, last_paralax[:3])

            # check reprojection error
           #  cp_error = (new_curr_point[0:2] / new_curr_point[2]) - curr_frame.pts[curr_frame.idx1[i]]
           #  lp_error = (new_last_point[0:2] / new_last_point[2]) - last_frame.pts[last_frame.idx1[i]]
           #  cp_error = np.sum(cp_error ** 2)
           #  lp_error = np.sum(lp_error ** 2)
           #  if cp_error > 2 or lp_error > 2: 
           #      b3+=1
           #      continue 

            # add the point
            try: 
                color = curr_frame.rgb_img[int(round(curr_frame.pts[idx1[i], 1])), int(round(curr_frame.pts[idx1[i], 0]))] 
            except:
                b4+=1
                color = (255, 0, 0)

            pt = Point(point[0:3], color)
            pt.add_observation(curr_frame, idx1[i])
            pt.add_observation(last_frame, idx2[i])
            new_pts_count += 1

        print('added %d new points' % (new_pts_count))
        print('break poitns', b1, b2, b3, b4)

        # annotate the frame with filtered match results 
        curr_frame.img = self.annotate_frame(last_frame, curr_frame, idx1, idx2)

        # OLD: self.curr_frame.process_next_frame(self.last_frame, self.curr_frame)

    def annotate_frame(self, last_frame, curr_frame, idx1=None, idx2=None):
        annotated_img = self.curr_frame.img_rgb.copy()

        if last_frame == None:
            disp2d.paint(annotated_img)
            # annotated_img = cv.drawKeypoints(annotated_img, curr_frame.kps, annotated_img)
            for pt1 in curr_frame.pts:
                u1, v1 = int(round(pt1[0])), int(round(pt1[1]))
                cv.circle(annotated_img, (u1, v1), color=(255, 0, 0), radius=3)
            return 

        for pt2, pt1 in zip(last_frame.pts[idx2], curr_frame.pts[idx1]):
            #u1, v1 = denormalize(self.K, pt1)
            #u2, v2 = denormalize(self.K, pt2)
            u1, v1 = int(round(pt1[0])), int(round(pt1[1]))
            u2, v2 = int(round(pt2[0])), int(round(pt2[1]))
            # print(u1,v1)
            cv.circle(annotated_img, (u1,v1), color=(0, 255, 0), radius=3)
            cv.line(annotated_img, (u1,v1), (u2, v2), (0, 255, 255), 1)

        disp2d.paint(annotated_img)
        return annotated_img
        # print('num pts: ', len(curr_frame.pts))

    # def process_first_frame(self, curr_frame): 
    #     curr_frame.kps, curr_frame.des = orb.detectAndCompute(curr_frame.img_grey, mask=None)

    #     # self.kps is a list of keypoint objects, we want the numberical (float) values
    #     # so we convert list of keypoints to array of points
    #     curr_frame.kps = np.array([x.pt for x in curr_frame.kps], dtype=np.float32)
    #     
    #     # paint the keypoints and display the result
    #     self.drawFeatures(None, curr_frame)


    # def drawFeatures(self, last_frame, curr_frame): 
    #     annotated_img = curr_frame.img_rgb.copy()

    #     if slam.FIRST_IMAGE == True: 
    #         for point in curr_frame.kps:
    #             a,b = point.ravel()
    #             cv.circle(annotated_img, (int(a), int(b)), 5, (0,255,0), -1)

    #     else: 
    #         # TODO: implement draw features on next frames
    #         pass
    #     disp2d.paint(annotated_img)


        
       

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
    disp3d = Display3D()
    slam = SLAM(W, H, K)

    while cap.isOpened():
        ret, img = cap.read()

        if ret == True:
            img = cv.resize(img, (W, H))
            slam.process_frame(img)
        else:
            break

        print(len(slam.frames))
        disp3d.paint(slam.frames) 
