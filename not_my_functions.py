import numpy as np
import os

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
  return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

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
  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t
  return ret

def poseRt(R, t):
  ret = np.eye(4)
  ret[:3, :3] = R
  ret[:3, 3] = t
  return ret

# pose
def fundamentalToRt(F):
  W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
  U,d,Vt = np.linalg.svd(F)
  if np.linalg.det(U) < 0:
    U *= -1.0
  if np.linalg.det(Vt) < 0:
    Vt *= -1.0
  R = np.dot(np.dot(U, W), Vt)
  if np.sum(R.diagonal()) < 0:
    R = np.dot(np.dot(U, W.T), Vt)
  t = U[:, 2]

  # TODO: Resolve ambiguities in better ways. This is wrong.
  if t[2] < 0:
    t *= -1
  
  # TODO: UGLY!
  if os.getenv("REVERSE") is not None:
    t *= -1
  return np.linalg.inv(poseRt(R, t))


def normalize(Kinv, pts):
  return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]


def denormalize(K, pt):
  ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
  # ret /= ret[2]
  return int(round(ret[0])), int(round(ret[1]))

# # turn [[x,y]] -> [[x,y,1]]
# def add_ones(x):
#   if len(x.shape) == 1:
#     return np.concatenate([x,np.array([1.0])], axis=0)
#   else:
#     return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
    if len(x.shape) == 1:
        ret = np.array([[x[0], x[1], 1]])
        print(ret)
        return ret
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


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
