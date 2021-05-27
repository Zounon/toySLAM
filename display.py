import cv2 as cv
import numpy as np
import pygame
# from pygame.locals import DOUBLEBUF


class Display2D(object): 
    def __init__(self, W, H): 
        pygame.display.init()
        self.screen = pygame.display.set_mode([W,H], pygame.DOUBLEBUF)
        self.surface = pygame.Surface(self.screen.get_size()).convert()

    def paint(self, img): 
        for event in pygame.event.get():
            pass

        pygame.surfarray.blit_array(self.surface, img.swapaxes(0,1)[:, :, [2,1,0]])
        self.screen.blit(self.surface, (0,0))

        pygame.display.flip()

# class Dispaly(object): 
#     def __init__(self, W, H): 
#         self.W, self.H = W, H
#         self.capture = cv.VideoCapture('test_countryroad.mp4')
# def get_orbs(video_frame): 
#     # img = cv.imread('./countryroad_frames/frame1.jpg')
#     img = video_frame

#     # init orb detector
#     orb = cv.ORB_create()

#     # find the keypoints with ORB 
#     kp = orb.detect(img, None)

#     # compute the descriptors with ORB
#     kp, des = orb.compute(img, kp) 

#     # draw only kp location
#     return cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
#     # plt.imshow(img2), plt.show()
#     # cv.imshow('display window',img)


# cap = cv.VideoCapture('test_countryroad.mp4')

# while cap.isOpened(): 
#     ret, frame = cap.read()
#     if not ret:
#         print("can not receive frame")
#         break
#     gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     color = cv.cvtColor(frame, cv.IMREAD_COLOR)
#     w_orbs = get_orbs(color)
#     # cv.circle(color, (516 , 920), color=(0,255,0), radius=3)
#     cv.imshow('frame', w_orbs)

#     if cv.waitKey(1) == ord('q'): 
#         break


# cap.release()
# cv.destroyAllWindows() 


