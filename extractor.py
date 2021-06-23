import cv2 as cv
import numpy as np
from numpy.core.defchararray import add
from numpy.core.numeric import normalize_axis_tuple
from skimage.measure import ransac 
from skimage.transform import FundamentalMatrixTransform 
from skimage.transform import EssentialMatrixTransform 

def add_ones(x): 
    # turn [[x,y]] -> [[x,y,1]]
    ret =  np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return ret

def extractRt(E): 
    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
    U,d,Vt = np.linalg.svd(E)
    assert np.linalg.det(U) > 0
    if np.linalg.det(Vt) < 0: 
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0: 
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2] 
    Rt = np.concatenate([R,t.reshape(3,1)], axis=1)
    return Rt


class Extractor(object): 
    Grid_X = 16//2
    Grid_Y = 12//2

    def __init__(self, K):
        self.orb = cv.ORB_create()
        self.bf = cv.BFMatcher(cv.NORM_HAMMING) 
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)


    def normalize(self, pts): 
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]

    def denormalize(self, pt): 
        ret = np.dot(self.K, [pt[0], pt[1], 1.0])
        # ret /= ret[2]
        return int(round(ret[0])), int(round(ret[1])) 
        # return int(round(pt[0] + self.w)), int(round(pt[1] + self.h))

    # def extract(self, img): 
    #     sy = img.shape[0]//self.Grid_Y
    #     sx = img.shape[1]//self.Grid_X
    #     kp_array = []
    #     for row_y in range(0, img.shape[0], sy):
    #         for row_x in range(0, img.shape[1], sx):
    #             img_chunk = img[row_y:row_y+sy, row_x:row_x+sx]
    #             kp = self.orb.detect(img_chunk, None)
    #             # print(img_chunk.shape)
    #             for p in kp:
    #                 p.pt = (p.pt[0] + row_x, p.pt[1] + row_y)
    #                 # print(p)
    #                 kp_array.append(p)
    #     return kp_array

    def extract2(self, img): 
        # detection
        feats = cv.goodFeaturesToTrack(
                                    np.mean(img, axis=2).astype(np.uint8),
                                     3000, 
                                     qualityLevel=0.01, 
                                     minDistance=3)

        # feats = cv.goodFeaturesToTrack(int())

        # extraction
        kps = [cv.KeyPoint(x=f[0][0], y=f[0][1], _size=3) for f in feats]
        kps, des = self.orb.compute(img, kps)

        # matching 
        ret = []
        if self.last is not None: 
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            for m,n in matches: 
                if m.distance < 0.75*n.distance: 
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))
            # matches = zip([kps[m.queryIdx] for m in matches], [kps[m.trainIdx] for m in matches]) 


        # filter 
        Rt = None
        if len(ret) > 0: 
            ret = np.array(ret)

            # subtract  to move to 0
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])


            # print(ret.shape)
            model, inliers = ransac((ret[:, 0], ret[:, 1]), 
                                    EssentialMatrixTransform, 
                                    min_samples=8,
                                    residual_threshold=0.005,
                                    max_trials=100)
            ret = ret[inliers]
            Rt = extractRt(model.params)

        # return 
        self.last = {'kps':kps, 'des':des}
        return ret, Rt 