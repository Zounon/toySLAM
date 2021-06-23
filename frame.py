#!/usr/bin/env python3
from pygame import init


class Frame(object): 
    def __init__(self, img, K) -> None:
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.pose = IRt 

        