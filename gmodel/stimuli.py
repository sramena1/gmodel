# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 23:53:01 2016

@author: sramena1

Module creates the stimuli for testing grouping model. We
create the stimuli used in the experiments of the paper:

[1]. Zhou, Hong, Howard S. Friedman, and Rüdiger Von Der Heydt. "Coding of
border ownership in monkey visual cortex." The Journal of Neuroscience 20.17
(2000)

The stimuli were used for simulations in the model described in :

[2]. Craft, Edward, et al. "A neural model of figure–ground organization."
Journal of neurophysiology 97.6 (2007): 4310-4326.

"""
import numpy as np
import matplotlib as plt

def create_square_stim():
    """function to create and display a square stimulus"""
    
    stim0 = np.zeros(shape=(128,128),dtype='float',order='C')
    """ stim size of (128,128) pix is chosen. Can be any desired size"""
    E = np.zeros(shape=(128,128,2),dtype='float',order='C')
    """Edge responses, ie, the output from a Complex cell in V1 of cortex.
    Any segmentation, edge detection method can replace this. Gabor filter
    banks model Complex V1 cells best."""
    J = np.zeros_like(E)
    J = np.tile(J,[1,2])
    stim90 = np.zeros_like(stim0)
    ri1 = 63-18
    ri2 = 63+18
    ci1 = 63-18
    ci2 = 63+18
    stim90[ri1:ri2,ci1]= 1 #
    stim90[ri1-1:ri2+1,ci2]= 1
        
    stim0[ri1-1,ci1:ci2] = 1
    stim0[ci2, ci1:ci2] = 1
    
    E[:,:,0] = stim0
    E[:,:,1] = stim90
    
    if __name__ == "__main__" :
        f1 = plt.pyplot.figure()
        plt.pyplot.imshow(stim0)
        f1.show()
        
        f2 = plt.pyplot.figure()
        plt.pyplot.imshow(stim90)
        f2.show()
        
        f3 = plt.pyplot.figure()
        plt.pyplot.imshow(stim0+stim90)
        f3.show()
        
    return E, J
    
def create_overlap_squares():
    """Creates overlapping squares. T-Junctions encode overlapping at two
    locations where the squares overlap"""
    stim0 = np.zeros(shape=(128,128),dtype='float',order='C')
    """ stim size of (128,128) pix is chosen. Can be any desired size"""
    E = np.zeros(shape=(128,128,2),dtype='float',order='C')
    """Edge responses, ie, the output from a Complex cell in V1 of cortex.
    Any segmentation, edge detection method can replace this. Gabor filter
    banks model Complex V1 cells best.SOME VALUES ARE HARD CODED AS THESE ARE SIMPLE STIMULI CREATED TO VERIFY THE ANIMAL EXPERIMENTS"""
    J = np.zeros_like(E) 
    J = np.tile(J,[1,2])
    stim90 = np.zeros_like(stim0)
    
    stim0[48,51-8:50+8] = 1
    stim0[64,51-8:50+8] = 1
    stim0[57,59:57+9] = 1
    stim0[71,50:50+16] = 1

    stim90[57-8:57+7,43] = 1
    stim90[57-9:57+8,58] = 1
    stim90[65:66+5,50] = 1
    stim90[64-7:66+6,66]=1

    E[:,:,0] = stim0
    E[:,:,1] = stim90
    
    J[64,50-2:50+2,0] = 2
    J[57-2:57+2,58,1] = 2
    J[64,50-2:50+2,2] = 0
    J[57-2:57+2,58,3] = 0

    if __name__ == "__main__" :
        f1 = plt.pyplot.figure()
        plt.pyplot.imshow(stim0)
        f1.show()

        f2 = plt.pyplot.figure()
        plt.pyplot.imshow(stim90)
        f2.show()

        f3 = plt.pyplot.figure()
        plt.pyplot.imshow(stim0+stim90)
        f3.show()
        
        JJ = np.sum(J,axis=2)
        f4 = plt.pyplot.figure()
        plt.pyplot.imshow(stim0+stim90+JJ)
        f4.show()
    return E, J

def create_c_shape():

    stim0 = np.zeros(shape=(64,64),dtype='float',order='C')
    E = np.zeros(shape=(64,64,2),dtype='float',order='C')
    J = np.zeros_like(E)
    J = np.tile(J,[1,2])
    stim90 = np.zeros_like(stim0)
    
    stim0[16,32-8:32+8] = 1
    stim0[24,32:32+8] = 1
    stim0[40,31:32+8] = 1
    stim0[48,32-9:32+8] = 1

    stim90[32-16:32+16, 23] = 1
    stim90[24:24+16,31] = 1
    stim90[16:16+9,40] = 1
    stim90[32+8:49,40] = 1

    E[:,:,0] = stim0
    E[:,:,1] = stim90

    if __name__ == "__main__" :
        f1 = plt.pyplot.figure()
        plt.pyplot.imshow(stim0)
        f1.show()
    
        f2 = plt.pyplot.figure()
        plt.pyplot.imshow(stim90)
        f2.show()
    
        f3 = plt.pyplot.figure()
        plt.pyplot.imshow(stim0+stim90)
        f3.show()
    
    return E, J
    
def create_lines_and_squares9D():
    """ script to create lines and squares stimulus of Figure 9D in
    Craft etal paper"""
    stim0 = np.zeros(shape=(128,128),dtype='float',order='C')
    E = np.zeros(shape=(128,128,2),dtype='float',order='C') 
    J = np.zeros_like(E) 
    J = np.tile(J,[1,2])
    stim90 = np.zeros_like(stim0)
    
    stim0[56,24:39] = 1
    stim0[56,50:64] = 1
    stim0[56,73:88] = 1

    stim0[71,24:39] = 1
    stim0[71,50:64] = 1
    stim0[71,73:88] = 1

    stim90[64-7:64+8,16] = 1
    stim90[64-7:64+7,24] = 1
    stim90[64-8:64+8,39] = 1
    stim90[64-8:64+8,49] = 1
    
    stim90[64-8:64+8,64] = 1
    stim90[64-7:64+7,73] = 1
    stim90[64-8:64+8,88] = 1
    stim90[64-7:64+7,96] = 1
    
    E[:,:,0] = stim0
    E[:,:,1] = stim90

    if __name__ == "__main__" :
        plt.pyplot.close("all")
        f1 = plt.pyplot.figure()
        plt.pyplot.imshow(stim0)
        f1.show()

        f2 = plt.pyplot.figure()
        plt.pyplot.imshow(stim90)
        f2.show()

        f3 = plt.pyplot.figure()
        plt.pyplot.imshow(stim0+stim90)
        f3.show()

    return E, J
    
def create_lines_stim():
    stim0 = np.zeros(shape=(128,128),dtype='float',order='C')
    E = np.zeros(shape=(128,128,2),dtype='float',order='C') 
    J = np.zeros_like(E) 
    J = np.tile(J,[1,2])
    stim90 = np.zeros_like(stim0)
    
    stim90[64-8:64+7,16] = 1
    stim90[4-8:64+7,16+8] = 1
    stim90[64-8:64+7,16+8+16] = 1
    stim90[64-8:64+7,16+8+16+8] = 1
    
    stim90[64-8:64+7,16+8+16+8+16] = 1
    stim90[64-8:64+7,16+8+16+8+16+8] = 1
    stim90[64-8:64+7,16+8+16+8+16+8+16] = 1
    stim90[64-8:64+7,16+8+16+8+16+8+16+8] = 1
    stim90[64-7:64+7,96+16] = 1

    E[:,:,0] = stim0
    E[:,:,1] = stim90

    if __name__ == "__main__" :
        plt.pyplot.close("all")
        f1 = plt.pyplot.figure()
        plt.pyplot.imshow(stim90)
        f1.show()

        f2 = plt.pyplot.figure()
        plt.pyplot.imshow(stim0)
        f2.show()

        f3 = plt.pyplot.figure()
        plt.pyplot.imshow(stim0+stim90)
        f3.show()
    return E, J
    
def create_modified_c_stim():
    """Create modified C-shape with a square in the middle as in Figure 9A of Craft etal paper"""
    stim0 = np.zeros(shape=(64,64),dtype='float',order='C')
    E = np.zeros(shape=(64,64,2),dtype='float',order='C') 
    J = np.zeros_like(E) 
    J = np.tile(J,[1,2])
    stim90 = np.zeros_like(stim0)

    stim0[16,32-8:32+8] = 1
    stim0[24,32:32+8+8] = 1
    stim0[40,32:32+8+8] = 1
    stim0[48,32-8:32+8] = 1
    
    stim90[32-15:32+16,24] = 1
    stim90[25:24+16,32] = 1
    stim90[16:16+8,40] = 1
    stim90[32+8+1:49,40] = 1
    stim90[24:41,48] = 1
    
    E[:,:,0] = stim0
    E[:,:,1] = stim90

    J[24,40-2:40+2,0] = 0
    J[40,40-2:40+2,0] = 2

    J[24,40-2:40+2,2] = 2
    J[40,40-2:40+2,2] = 0

    if __name__ == "__main__" :
        plt.pyplot.close("all")
        f1 = plt.pyplot.figure()
        plt.pyplot.imshow(stim0)
        f1.show()

        f2 = plt.pyplot.figure()
        plt.pyplot.imshow(stim90)
        f2.show()

        f3 = plt.pyplot.figure()
        plt.pyplot.imshow(stim0+stim90)
        f3.show()
        
        JJ = np.sum(J,axis=2)
        f4 = plt.pyplot.figure()
        plt.pyplot.imshow(stim0+stim90+JJ)
        f4.show()
    return E, J