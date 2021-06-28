import cv2 as cv
import numpy as np
import pygame 
from pygame.locals import DOUBLEBUF

from multiprocessing import Process, Queue 
import pangolin
import OpenGL.GL as gl
import numpy as np

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

class Display3D(object): 
    def __init__(self): 
        self.state = None
        self.q = Queue()
        self.vp = Process(target=self.viewer_thread, args=(self.q,))
        self.vp.daemon = True
        self.vp.start()

    def viewer_thread(self, q): 
        self.viewer_init(1024, 768)
        while True:
            self.viewer_refresh(q)

    def viewer_init(self, w, h): 
        pangolin.CreateWindowAndBind('Map Viewer', w, h)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
        pangolin.ModelViewLookAt(0, -10, -8,
                                 0, 0, 0,
                                 0, -1, 0))
        self.handler = pangolin.Handler3D(self.scam)

        # Create Interactive View in window
        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, w/h)
        self.dcam.SetHandler(self.handler)
        # hack to avoid small Pangolin, no idea why it's *2
        # self.dcam.Resize(pangolin.Viewport(0,0,w*2,h*2))
        self.dcam.Activate()

    def viewer_refresh(self, q):
        # if there is something in the que, get it
        while not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        self.dcam.Activate(self.scam)

        if self.state is not None: 
            if self.state[0].shape[0] >= 2:
                # draw poses
                gl.glColor3f(0.0, 1.0, 0.0)
                pangolin.DrawCameras(self.state[0][:-1])
            
            if self.state[0].shape[0] >= 1:
                # draw current pose as yellow
                gl.glColor3f(1.0, 1.0, 0.0)
                pangolin.DrawCameras(self.state[0][-1:])

            #if self.state[1].shape[0] != 0:
            #    # draw keypoints
            #    gl.glPointSize(5)
            #    gl.glColor3f(1.0, 0.0, 0.0)
            #    pangolin.DrawPoints(self.state[1], (255,0,0)) 

        pangolin.FinishFrame()

    def paint(self, frames):
        if self.q is None:
            return

        poses, pts, colors = [], [], []
        for frame in frames:
            # invert pose for display only
            poses.append(np.linalg.inv(frame.pose))
            pts.append(frame.pts)
            # colors.append(p.color)
            colors.append((255, 0, 0))

        self.q.put((np.array(poses), np.array(pts), np.array(colors)/256.0))
