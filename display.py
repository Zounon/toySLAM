import cv2 as cv
import numpy as np
import pygame 
from pygame.locals import DOUBLEBUF

# class Display(object): 
#     def __init__(self, Width, Height): 
#         """
#         create
#         """
#         sdl2.ext.init()
#         self.W, self.H = Width, Height
#         self.window = sdl2.ext.Window("mySlam", size=(W,H))
#         self.window.show()


class Display2D(object):    
    ''' 
    Creates a view port for displaying frames, keypoints, etc. 

    function paint
        takes the frame img as input;
        returns nothing
        updates the frame 
    ''' 
    
    def __init__(self, W, H): 
        # self.W, self.H = W, H
        # self.window = sdl2.ext.Window("mySLAM", size=(W,H), position=(-500, -500))
        # self.window.show()

        pygame.init()
        self.screen = pygame.display.set_mode((W,H), DOUBLEBUF)
        self.surface = pygame.Surface(self.screen.get_size()).convert()


    def paint(self, img): 
        for event in pygame.event.get():
            pass

        # surfarray.blit_array: 
        #   Copy values from an array and place them directly onto the surface
        #
        # swapaxes(0,1) exchanges the z axis with y axis 
        # that means swaping the W,H for H,W 
        #
        # [:, :, [2,1,0]] takes the BGR colors and reorders them into RBG  
        pygame.surfarray.blit_array(self.surface, img.swapaxes(0,1)[:,:,[2,1,0]])
        
        # draws the surface onto the screen
        self.screen.blit(self.surface, (0,0))

        # takes advantage of the double buffer (DOUBLEBUF) to swap the surface and screen 
        pygame.display.flip()

