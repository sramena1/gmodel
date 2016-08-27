GROUPING MODEL 
Sudarshan Ramenahalli
The code is developed with Spyder desktop environment on Windows system.
This code replicates all the results in the paper:
[R1]. Craft, Edward, et al. "A neural model of figure–ground 
organization." Journal of neurophysiology 97.6 (2007): 4310-4326.
except:
1. results in figure 8(b) - lateral interactions
2. Figure A1 - parameter space
3. Time evolution of B and G cell responses (this is fast, non-ODE based version)

***************************************************************
IMPORTANT: WE ASSUME IMAGE SEGMENTATION/EDGE DETECTION IS ALREADY 
DONE AND EDGES ARE GIVEN TO US. THIS METHOD CAN BE INTERFACED WITH 
ANY STANDARD SEGMENTATION/EDGE DETECTION METHOD SUCH AS gPb, 
deepEdge, etc.
*****************************************************************

The package contains two main modules (in the inner 'gmodel' folder):
1. gmodel - contains all the functions realted to the 
computational model. This version implements fast border ownership 
assignment (convolution filters based, with 2-pass 
excitatory feedforward connections, no feedback) suitable for computer
vision application. 

2. stimuli - this module creates all the stimuli used in the 
paper above

=====================TESTING==============================
Tests for all stimuli are included in the 'tests' folder

How to run: Execute testAllStimuliFast.py to replicate all results. 
By uncommentiong within file different stimuli can be tested. 
Set the 'gmodel' directory (outer) as your working directory.

