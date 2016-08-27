# -*- coding: utf-8 -*-

"""
Created on Sun Aug 14 14:30:25 2016
All tests for all stimuli that were used in the neurophysiology experiments
is done here. This code should reproduce all the results in paper:

[R1]. [2]. Craft, Edward, et al. "A neural model of figure–ground organization." 
Journal of neurophysiology 97.6 (2007): 4310-4326.

@author: Sudarshan Ramenahalli, Johns Hopkins University
"""
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import gmodel.gmodel as gm
import gmodel.stimuli as stims
import numpy as np
#import matplotlib.pyplot as plt
# =====================SIMULATION PARAMETERS ========================
stimType = 'ModifiedC' # 'square', 'OverlapSquares','LinesSquares','Lines','ModifiedC'
all_radii = [2,3,5,8,12,18] # Radii of G cells
#R = [4,8,16,32] # Radii of G cells
weight_tj = 2 # weight that controls the influence of a T-Junction on 
                #G cells
# ================== STIMULI =========================
# By appropriately uncommenting, each stimulus can be tested.
if(stimType=='square'):
    edges,junctions = stims.create_square_stim()
    scale_fac = 2.0
elif(stimType=='C'):
    edges,junctions = stims.create_c_shape()
    scale_fac = 2.0
elif(stimType=='OverlapSquares'):
    edges,junctions = stims.create_overlap_squares()
    scale_fac = 10.0
elif(stimType=='LinesSquares'):
    edges,junctions = stims.create_lines_and_squares9D()
    scale_fac = 5.0
elif(stimType=='Lines'):
    edges,junctions = stims.create_lines_stim()
    scale_fac = 5.0
elif(stimType=='ModifiedC'):
    edges,junctions = stims.create_modified_c_stim()
    scale_fac = 5.0
#=====================================================
print "Executing fast grouping model, please wait"
nrows,ncols,noris = np.shape(edges)
nscales = len(all_radii)
edges_junctions = gm.combine_edges_junctions(edges,junctions,weight_tj)
mask_ej = np.copy(edges_junctions)
for ii in range(2*noris):
    edges_junctions_i = np.copy(edges_junctions[:,:,ii])
    edges_junctions_i[edges_junctions_i>=1] = 1
    mask_ej[:,:,ii] = np.copy(edges_junctions_i)

grouping_kers = gm.create_grouping_kernels(all_radii,noris)

bo_init_activity = edges_junctions # initially each pair of bo cells will have
# equal strength, net activity is zero at each location on border
gcell_activity = gm.feedback_to_grouping_cells(bo_init_activity, 
                                               grouping_kers,all_radii)
bocell_activity = gm.feedback_to_bocells(gcell_activity,grouping_kers,noris,all_radii)
bocell_activity_masked = bocell_activity*mask_ej
bocell_activity_final = gm.feedback_bocells(bocell_activity_masked)

# Due to large range of B cell responses, using matplotlib.quiver does not 
#visualize properly, arrows become too small or too large. Hence, only direction information is displayed.
gm.quiver_bo_response(mask_ej*bocell_activity_final,scale_fac)
gm.visualize_bo_activity(bocell_activity_final)
