import numpy as np
import os
import cv2 as cv
from skimage.measure import ransac
# from skimage.transform import EssentialMatrixTransform
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


def add_ones(x):
    if len(x.shape) == 1:
        return np.concatenate([x, np.array([1.0])], axis=0)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

# from https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_geometric.py
class EssentialMatrixTransform(object):
    def __init__(self):
        self.params = np.eye(3)

    def __call__(self, coords):
        coords_homogeneous = np.column_stack([coords, np.ones(coords.shape[0])])
        return coords_homogeneous @ self.params.T

    def estimate(self, src, dst):
        assert src.shape == dst.shape
        assert src.shape[0] >= 8
        print(src.shape, dst.shape)
        # Setup homogeneous linear equation as dst' * F * src = 0.
        A = np.ones((src.shape[0], 9))
        A[:, :2] = src
        A[:, :3] *= dst[:, 0, np.newaxis]
        A[:, 3:5] = src
        A[:, 3:6] *= dst[:, 1, np.newaxis]
        A[:, 6:8] = src

        # Solve for the nullspace of the constraint matrix.
        _, _, V = np.linalg.svd(A)
        F = V[-1, :].reshape(3, 3)

        # Enforcing the internal constraint that two singular values must be
        # non-zero and one must be zero.
        U, S, V = np.linalg.svd(F)
        S[0] = S[1] = (S[0] + S[1]) / 2.0
        S[2] = 0
        self.params = U @ np.diag(S) @ V

        return True


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
        uvs_contiguous = np.ascontiguousarray(uvs[:, :2]).reshape((-1, 1, 2))
        uvs_undistorted = cv.undistortPoints(
            uvs_contiguous, self.K, None, self.K)
        return uvs_undistorted.ravel().reshape(uvs_undistorted.shape[0], 2)
        # arr = []
        # kps = np.array(kps)
        # for i in range(0, len(kps),2):
        # arr.append([kps[i], kps[i+1]])
#
        # kps = [[x,y] for x,y in kps]
        # print(kps.shape)
        # kps = kps[:, np.newaxis, :]
        # print(kps.shape)
        # print(kps.shape)
        print(kps)
        # return cv.undistortPoints(kps, self.K, None, None, self.K)

    def unproject_points(self, kps):
        return np.dot(self.Kinv, add_ones(kps).T).T[:, 0:2]


class Frame(object):
    def __init__(self, img, index_id):
        self.img = img
        self.id = index_id
        self.kps, self.des, self.pts = self.compute_kp_des(self.img)
        self.good_matches = None
        if self.id < 5:
            self.annotated_img = cv.drawKeypoints(img, self.kps, img)
        else:
            self.annotated_img = self.annotateImg(
                self.img, slam.frames[-2].kps, slam.frames[-1].kps)

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
        #             if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
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

    # def annotate_img(self, img, kps, des):
        # return cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
        # for i, pts in enumerate(zip(self))

    def annotateImg(self, img, kps_ref, kps_cur):
        for i, pts in enumerate(zip(kps_ref, kps_cur)):
            if self.good_matches[i]:
                p1, p2 = pts
                a, b = p1.ravel()
                c, d = p2.ravel()
                cv.line(img, (a, b), (c, d), (0, 255, 0), 1)
        return img


class SLAM(object):
    def __init__(self, W, H, K):
        self.map = Map()
        self.W, self.H, self.K = W, H, K
        self.frames = []

    def match_frames(self, f1, f2):
        bf = cv.BFMatcher(cv.NORM_HAMMING)
        matches = bf.knnMatch(f1.des, f2.des, k=2)

        # Lowe's ratio test
        ret = []
        idx1, idx2 = [], []
        idx1s, idx2s = set(), set()

        for m, n in matches:
            if m.distance < 0.75*n.distance:
                p1 = f1.kps[m.queryIdx]
                p2 = f2.kps[m.trainIdx]

                # be within orb distance 32
                if m.distance < 32:
                    # keep around indices
                    # TODO: refactor this to not be O(N^2)
                    if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
                        idx1.append(m.queryIdx)
                        idx2.append(m.trainIdx)
                        idx1s.add(m.queryIdx)
                        idx2s.add(m.trainIdx)
                        ret.append((p1.pt, p2.pt))

        # no duplicates
        assert(len(set(idx1)) == len(idx1))
        assert(len(set(idx2)) == len(idx2))

        assert len(ret) >= 8
        ret = np.array(ret)
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)

        # fit matrix            
        print((ret[:, 0].shape, ret[:, 1].shape))
        model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                EssentialMatrixTransform,
                                min_samples=8,
                                residual_threshold=0.02,
                                max_trials=100)

        print("Matches:  %d -> %d -> %d -> %d" %
              (len(f1.des), len(matches), len(inliers), sum(inliers)))
        return idx1[inliers], idx2[inliers]

    def estimatePose(self, kps_ref, kps_cur):
        kpn_ref = cam.unproject_points(cam.undistort_points(kps_ref))
        kpn_cur = cam.unproject_points(cam.undistort_points(kps_cur))
        print(type(kpn_ref))
        print(type(kpn_ref[0]))
        E, self.good_matches = cv.findEssentialMat(kpn_cur, kpn_ref, focal=1,
                                                   pp=(0., 0.), method=cv.RANSAC, prob=0.999,
                                                   threshold=0.0003)
        _, R, t, mask = cv.recoverPose(
            E, kpn_cur, kpn_ref, focal=1, pp=(0., 0.))
        return R, t  # Rrc, trc (with respect to 'ref' frame)

    def process_frame(self, img):
        new_frame_id = len(self.frames)
        frame_obj = Frame(img, new_frame_id)
        self.frames.append(frame_obj)
        if frame_obj.id < 2:
            return frame_obj.annotated_img
        print(self.frames[-2].pts.shape, self.frames[-1].pts.shape)
        R, t = self.estimatePose(self.frames[-2].pts, self.frames[-1].pts)
        idx1, idx2, slam.frames[-1].pts = slam.match_frames(slam.frames[-1], slam.frames[-2])
        return frame_obj.annotated_img


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
            disp2d.paint(slam.process_frame(img))
        else:
            break
