# -*- coding: utf-8 -*-
"""
Core grouping module
contains:
1. Grouping model related definitions

1(a). combine_edges_junctions() - this function creates an integrated representation of edges an
d T-Junctions

1(b). feedback__bocells() - implements excitatory input between bo_cells cells
This is a recurrent feedback from bo cells of one BO direction preference to 
opposite BO preference cells at the same receptive field location

1(c). feedback_to_grouping_cells() - models the excitatory feedforward input
to G cells from bo_cells cells,whose BO preference direction points to the
 center of G cell RF

1(d). feedback_to_bocells() - same as toBode(), but excitatory connections (used ONLY for
"fast" version of code)

1(e). create_grouping_kernels() - to create the circular arcs of different
scales with decreasing connection weights with increasing radius for convolutions

2. Visualization functions:

2(a) quiver_bo_response() - visualize Border ownership direction

2(b) visualize_bo_activity() - visualized magnitude and direction of bo_cells cell responses

2(c). visualize_grouping_kernels() - to plot all grouping cell receptive fields (filter kernels)

3. Miscellaneous functions

Relevant research in:
[1]. Craft, Edward, et al. "A neural model of figure–ground organization."
Journal of neurophysiology 97.6 (2007): 4310-4326.

Details of neurophysiology experiments with macaques in:
[2]. Zhou, Hong, Howard S. Friedman, and Rüdiger Von Der Heydt. "Coding of
border ownership in monkey visual cortex." The Journal of Neuroscience 20.17
(2000)
"""
import numpy as np
import numpy.ma as ma
import matplotlib as plt
import scipy.ndimage as imlib
import scipy.signal as sig
from skimage.draw import circle_perimeter

# ========================MODEL PARAMETERS ===============================
DECAY = 100 # Decay time constant of a cell, assumed same for bo_cells and G cells
BETA = 0.5 # Model parameter controlling inhibitory feedback between bo_cells cells
GAMMA_0 = 4.5 #Model parameter controlling feedforward connection strength
              #from bo_cells to G cell
RHO_0 = 4.5 # Model parameter to control feedback strength from G to bo_cells cell
SD_RATIO = 2.5 # A Parameter to control the std deviation of the G cell
               #receptive field (a Gaussian kernel along the arc of a circle)
               # spread.

#========================== GROUPING MODEL FUNCTIONS=========================
def create_grouping_kernels(radii, numori):
    """creates the grouping cell kernels, arcs of a circle with decreasing
    connection strength as radius of the grouping cell increases. This ensures
    proximal edges are grouped with higher confidence"""
    grouping_cell_kernels = ()
    border_ownership_direcs, _ = compute_bo_direcs(numori)
    theta_half_step = border_ownership_direcs[1]/2
    numdirs = len(border_ownership_direcs)
    for radius in radii:
        ker2 = np.zeros((3*2*radius-1, 3*2*radius-1)) # for debug only

        for _, currdirec in enumerate(border_ownership_direcs):
            if currdirec >= 2*np.pi:
                currdirec = currdirec - 2*np.pi

            x_values = np.arange(-3*radius+1, 3*radius, 1)
            y_values = np.arange(-3*radius+1, 3*radius, 1)
            grid_x, grid_y = np.meshgrid(x_values, y_values, sparse='false', indexing='xy')
            (_, theta) = cart2pol(grid_x, grid_y)

            # the below 3 lines required to rotate kers properly and arrange
            #them in anti-clockwise direction, consisten with ori definitions of
            #bo_cells cells
            theta = np.rot90(theta, 1)
            theta = np.fliplr(theta)
            theta = theta + np.pi

            [circle_x, circle_y] = circle_perimeter(3*radius-1, 3*radius-1, radius)
            circle_perim = np.zeros_like(theta)
            circle_perim[circle_x, circle_y] = 1

# SPLIT CIRCLE PERIMETER INTO ARCS FOR MAKING CONVOLUTION KERNEL
            if currdirec == 0:
                th2 = theta
                theta = np.where(theta >= currdirec + theta_half_step, 0, theta)
                th2 = np.where(th2 <= 2*np.pi - theta_half_step, 0, th2)
                theta = theta+th2
            else:
                theta = np.where(theta >= currdirec + theta_half_step, 0, theta)
                theta = np.where(theta <= currdirec - theta_half_step, 0, theta)

            ker = np.multiply(theta, circle_perim)
            ker = np.where(ker != 0, 1, ker)

# CONVOLVE WITH GAUSSIAN TO ENLARGE RFs
            ker = imlib.gaussian_filter(ker, sigma=radius/SD_RATIO)
            ker = np.divide(ker, np.sum(ker, axis=None)) # normalize, sum to 1
            ker = np.divide(ker, numdirs) # total G cell activation should be 1,
            #so divide by number of BO directions
            grouping_cell_kernels = grouping_cell_kernels + (ker,)
            ker2 += ker

            if __name__ == "__main__":
                print np.sum(ker, axis=None)
                print ker.shape
                plt.pyplot.figure()
                plt.pyplot.imshow(ker)
                plt.pyplot.show()
    if __name__ == "__main__":
        plt.pyplot.figure()
        plt.pyplot.imshow(ker2)
        plt.pyplot.show()
    return grouping_cell_kernels

def combine_edges_junctions(complex_cells, tjunction_cells, weight_tjunctions):
    """reformats the E and T-junction cell responses for use with the ODE
    solver. T-junctions are local, strong cues for foreground objects,
    typically numerous in natural scenes and strongly influence object
    perception"""
    complex_cells = np.tile(complex_cells, [1, 2]) # making a copy
    complex_and_junction_cells = np.zeros_like(complex_cells)
    # COMBINE EDGES AND T-JUNCTIONS
    for bodir_idx in range(np.shape(complex_cells)[2]):
        complex_and_junction_cells[:, :, bodir_idx] = (complex_cells[
            :, :, bodir_idx] + weight_tjunctions*tjunction_cells[:, :, bodir_idx])
    return complex_and_junction_cells

def feedback_bocells(bo_cells):
    """implements the recurrent excitatory feedback between bo_cells cells of
    opposite BO preference, where a proportion of excitatory input is given to
    BO cell of opposite BO preference. This sets up a competition between
    bo_cells cells. The excitatory feedback is used ONLY for the Fast version of Bo
    computation, as in the "fast" version all connections are excitatory"""
    _, _, ndir = np.shape(bo_cells)
    copy_bocells = np.copy(bo_cells)
    nori = ndir/2
    for curr_ori in range(nori):
        bo_cells[:, :, curr_ori] = bo_cells[:, :, curr_ori] + BETA*copy_bocells[:, :, curr_ori+nori]
        bo_cells[:, :, curr_ori+nori] = (bo_cells[:, :, curr_ori+nori] +
                                         BETA*copy_bocells[:, :, curr_ori])
    return bo_cells

def feedback_to_bocells(grouping_cells, grouping_cell_kernels, nori, radii):
    """ feedback from G cells to bo_cells cells"""
    numrows, numcols, _ = np.shape(grouping_cells)
    ndir = 2*nori
    feedback_to_bo_cells = np.zeros([numrows, numcols, ndir], dtype='float', order='C')
    for currdirec in range(ndir):
        fb_gtob_allscales = np.zeros_like(grouping_cells[:, :, 0])

        for currscale, radius in enumerate(radii):

            ker = grouping_cell_kernels[currdirec + currscale*ndir]
            ker = np.rot90(ker, 2) # rotate kernel for opposite direction prefer
            gcell_resp = (sig.convolve2d(grouping_cells[:, :, currscale],
                                         ker, mode='same',
                                         boundary='fill'))
            fb_gtob_allscales += np.sqrt(RHO_0)*radius*gcell_resp
            #normalize, see Eqs (4-5) in Craft etal, 2006, J Neuro
        feedback_to_bo_cells[:, :, currdirec] = fb_gtob_allscales
    return feedback_to_bo_cells

def feedback_to_grouping_cells(bo_cells, grouping_cell_kernels, radii):
    """feedforward excitatory input from bo_cells cell to G cells. G cells integrate
    BO information from all bo_cells cells that point to the center of the G cell RF.
    This gives selectivity for convexity, proximity (based on connection
    strengths of Grouping kernels of different radii) and surroundedness"""
    numrows, numcols, ndir = np.shape(bo_cells) # num rows, columns, directions
    nsc = len(radii)
    grouping_cells = (np.zeros([numrows, numcols, nsc], dtype='float', order='C'))
    bo_resp_scale_direc = (np.zeros([numrows, numcols, nsc, ndir],
                                    dtype='float', order='C'))
    pairs, _ = nchoose2(range(ndir))
    #compute the input from a bo_cells cell to the G cell for each scale and
    #BO direction first. In the next block, we enforce selectivity for
    #co-circular objects by combining bo_cells cell resps for all possible BO direction
    #pairs
    for i, radius in enumerate(radii):
        for currdirec in range(ndir):
            ker = grouping_cell_kernels[currdirec + i*ndir]
            ker = np.rot90(ker, 2)
            bo_resp_ = (sig.convolve2d(bo_cells[:, :, currdirec], ker,
                                       mode='same',
                                       boundary='fill'))
            bo_resp_scale_direc[:, :, i, currdirec] = bo_resp_

    #below we amplify the G responses obtained in previous block for
#individual Border ownership directions only if they obey the principle of
#co-circularity. If edges are co-circular, the responses of corresponding G
#cells share the same center. Hence, point-wsie multiplication of G cells
#responses for all possible combinations of BO directions will amplify G
#response; if not they become zero. Hence we get selectivity for convexity
#or co-circularity which is a property of objects, not backgrounds. Then we
#apply a square-root non-linearity to G cell responses (can be any other
#like log-normal, logarithmic, etc) to prevent overshooting (seen in
#undamped non-linear system)
    for i, radius in enumerate(radii):
        co_circular_bo_resps = np.zeros((numrows, numcols), dtype='float', order='C')
        for pair in pairs:
            co_circular_bo_resps += (np.multiply(
                bo_resp_scale_direc[:, :, i, pair[0]],
                bo_resp_scale_direc[:, :, i, pair[1]]))
        sign_cocirc_resp = np.sign(co_circular_bo_resps)
        co_circular_bo_resps = np.fabs(co_circular_bo_resps)
        co_circular_bo_resps = (np.sqrt(GAMMA_0)*radius
                                *np.sqrt(co_circular_bo_resps))
        co_circular_bo_resps = co_circular_bo_resps*sign_cocirc_resp
        subsampled_resp = np.zeros_like(co_circular_bo_resps)
        subsampled_resp[::radius, ::radius] = co_circular_bo_resps[::radius, ::radius]
        grouping_cells[:, :, i] = subsampled_resp
    return grouping_cells

#=============== CUSTOM VISUALIZATION FUNCTIONS ==============================

def quiver_bo_response(bo_cells, scale_factor):
    """ to visualize border ownership direction computed by the model by
    drawing arrows. The direction of arrow points to the side of
    object/foreground """
    ndir = np.size(bo_cells, -1)
    nori = ndir/2
    border_ownership_direcs, _ = compute_bo_direcs(nori)
    border_ownership_direcs += np.pi/2 # adding pi/2 for drawing normals to edges
    #quiver_uu and quiver_vv are components of quiver vector for all
    #BO directions
    quiver_uu = np.zeros(np.shape(bo_cells)[0:2], dtype='float', order='C')
    quiver_vv = np.zeros_like(quiver_uu)
    for ori in range(nori):
        #ROTATE bo_cells CELL RESPONSES (along the edge) BASED ON BO DIR, FOR PLOTTING
        radii = [[-np.cos(border_ownership_direcs[ori])], [np.sin(border_ownership_direcs[ori])]]
        #u and v are quiver vector components for current BO direction
        quiver_uplus = np.ones_like(quiver_uu)*radii[0]
        quiver_vplus = np.ones_like(quiver_vv)*radii[1]
        bo_plus = bo_cells[:, :, ori] - bo_cells[:, :, ori+nori]
        bo_plus[bo_plus < 0] = 0
        #bo_plus[bo_plus>0]=1 # retain dir info only

        quiver_uu += quiver_uplus*bo_plus
        quiver_vv += quiver_vplus*bo_plus

# ROTATE bo_cells CELL RESPONSES BASED ON BO DIR, FOR PLOTTING
        radii = [[np.cos(border_ownership_direcs[ori])], [-np.sin(border_ownership_direcs[ori])]]

        quiver_uminus = np.ones_like(quiver_uu)*radii[0]
        quiver_vminus = np.ones_like(quiver_vv)*radii[1]

        bo_minus = bo_cells[:, :, ori+nori] - bo_cells[:, :, ori]
        bo_minus[bo_minus < 0] = 0
        #bo_minus[bo_minus>0]=1 # discard length/mag of bo_cells cell, retain dir only

        quiver_uu += quiver_uminus*bo_minus
        quiver_vv += quiver_vminus*bo_minus
    quiver_uu[quiver_uu == 0] = np.nan
    quiver_vv[quiver_vv == 0] = np.nan
    masked_quiver_uu = ma.masked_array(quiver_uu, np.isnan(quiver_uu))
    masked_quiver_vv = ma.masked_array(quiver_vv, np.isnan(quiver_vv))
    plt.pyplot.figure()
    plt.pyplot.quiver(masked_quiver_uu, masked_quiver_vv, scale=scale_factor)
    plt.pyplot.axis('square')
    plt.pyplot.axis('image')

def visualize_grouping_kernels(grouping_cell_kernels, radii, numori):
    """ to visualize the grouping cell kernels (receptive fields with
    corresponding weights) for all scales and directions. This visualization
    is important to make sure weights and RFs are proper."""
    numdirs = 2*numori
    for i, _ in enumerate(radii):
        for currdirec in range(numdirs):
            print currdirec + i*numdirs
            ker = grouping_cell_kernels[currdirec + i*numdirs]
            plt.pyplot.figure()
            plt.pyplot.imshow(ker)

def visualize_bo_activity(bo_cells):
    """ a different type of visualization of border ownership/grouping activity
    based on color coding, rather than arrows. Matplotlib quiver cannot
    properly plot the wide range of BO cell responses"""
    ndir = np.size(bo_cells, -1)
    nori = ndir/2
    border_ownership_direcs, _ = compute_bo_direcs(nori)
    border_ownership_direcs += np.pi/2 # adding pi/2 for drawing normals to edges
#uiver_uu and quiver_vv are components of quiver vector for all BO directions
    quiver_uu = np.zeros(np.shape(bo_cells)[0:2], dtype='float', order='C')
    quiver_vv = np.zeros_like(quiver_uu)
    for ori in range(nori):
        # ROTATE bo_cells CELL RESPONSES (along the edge) BASED ON BO DIR, FOR PLOTTING
        radii = [[-np.cos(border_ownership_direcs[ori])],
                 [np.sin(border_ownership_direcs[ori])]]
        #u and v are quiver vector components for current BO direction
        quiver_uplus = np.ones_like(quiver_uu)*radii[0]
        quiver_vplus = np.ones_like(quiver_vv)*radii[1]
        bo_plus = bo_cells[:, :, ori] - bo_cells[:, :, ori+nori]
        bo_plus[bo_plus < 0] = 0
        quiver_uu += quiver_uplus*bo_plus
        quiver_vv += quiver_vplus*bo_plus
# ROTATE bo_cells CELL RESPONSES BASED ON BO DIR, FOR PLOTTING
        radii = [[np.cos(border_ownership_direcs[ori])], [-np.sin(border_ownership_direcs[ori])]]
        quiver_uminus = np.ones_like(quiver_uu)*radii[0]
        quiver_vminus = np.ones_like(quiver_vv)*radii[1]
        bo_minus = bo_cells[:, :, ori+nori] - bo_cells[:, :, ori]
        bo_minus[bo_minus < 0] = 0
        quiver_uu += quiver_uminus*bo_minus
        quiver_vv += quiver_vminus*bo_minus
    quiver_uu[quiver_uu == 0] = np.nan
    quiver_vv[quiver_vv == 0] = np.nan
    masked_quiver_uu = ma.masked_array(quiver_uu, np.isnan(quiver_uu))
    masked_quiver_vv = ma.masked_array(quiver_vv, np.isnan(quiver_vv))
#====================Visualize BO direction info ==========================
    # From masked quiver_uu,quiver_vv matrices of BO dir components (on X- and Y- axes) get
    # and visualize the BO direction in degrees (range=[0,360])
    bo_direction = np.arctan2(np.ravel(masked_quiver_vv), np.ravel(masked_quiver_uu), order='C')
    bo_direction = np.rad2deg(bo_direction)
    bo_direction = bo_direction + 90 # to correct for [-pi,pi] range from atan2d
    bo_direction = from0to360d(bo_direction) # get continuous range [0,360] deg
    bo_direction = np.mod(bo_direction, 360)
    bo_direction = np.reshape(bo_direction, np.shape(masked_quiver_uu), order='C')
    plt.pyplot.figure()
    plt.pyplot.imshow(bo_direction)
    plt.pyplot.axis('square')
    plt.pyplot.axis('image')
    plt.pyplot.colorbar()
    plt.pyplot.title('Direction of border ownership response of bo_cells cells (degrees)')
#===============Visualize magnitude of BO response of the bo_cells cells============
    # The range of bo_cells cell is huge (3-8 orders of mag), hence compressing by
    # taking 4th root of BO resp strength, then normalize and plot as an image
    bo_resp_mag = np.sqrt(masked_quiver_uu*masked_quiver_uu + masked_quiver_vv*masked_quiver_vv)
    bo_resp_mag = np.divide(bo_resp_mag, np.min(bo_resp_mag, axis=None)) #all magnitudes >= 1
    bo_resp_mag = np.power(bo_resp_mag, 4) #range compress: better than sqrt, log,strict +ve
    plt.pyplot.figure()
    plt.pyplot.imshow(bo_resp_mag)
    plt.pyplot.axis('square')
    plt.pyplot.axis('image')
    plt.pyplot.colorbar()
    plt.pyplot.title('Magnitude of border ownership response of bo_cells cells (not to scale)')

# ===================== MISCELLANEOUS FUNCTIONS  ==========================

def cart2pol(x_value, y_value):
    """ given a 2D point in Cartesian coords, converts to polar coords """
    rho = np.sqrt(x_value**2 + y_value**2)
    theta = np.arctan2(y_value, x_value)
    return(rho, theta)

def pol2cart(rho, theta):
    """ converts a 2D point from polar coords to Cartesian coords """
    x_value = rho * np.cos(theta)
    y_value = rho * np.sin(theta)
    return(x_value, y_value)

def nchoose2(units):
    """function returns pairs of array elements. For example, if you want all
    possible pairs (without repitition) with N = [1,2,3,4] the
    function returns the pairs, (1,2), (1,3), (1,4), (2,3), (2,4,), (3,4) and
    the count of such pairs"""
    pairs = list()
    for i in units[:-1]:
        for j in units[units.index(i)+1:]:
            list.append(pairs, (i, j))
        num_pairs = len(pairs)
    return pairs, num_pairs

def compute_bo_direcs(numori):
    """get a list of BO directions in radians as well as degrees for use in
    other functions"""
    numdirs = 2*numori
    theta_incr = (2*np.pi)/numdirs
    border_ownership_direcs = 0.0
    currdirec = 0.0
    while currdirec < 2*np.pi:
        currdirec = currdirec + theta_incr
        border_ownership_direcs = np.append(border_ownership_direcs, currdirec)
    border_ownership_direcs = border_ownership_direcs[:-1]
    #border_ownership_direcs = border_ownership_direcs + np.pi/2
    bo_dirs_deg = np.rad2deg(border_ownership_direcs)
    return border_ownership_direcs, bo_dirs_deg


def from0to360d(theta):
    """ the function takes an angle computed by atan2d(), which ranges from
[0,pi] in anti-clockwise direction or [0,-pi] in clock-wise direction and
converts it into a single continuous range, [0,2*pi] in anti-clockwise
direction"""
    th0to360d = -np.ones_like(theta)
    for theta_idx, _ in enumerate(theta):
        if 90 < theta[theta_idx] <= 180:
            th0to360d[theta_idx] = theta[theta_idx]-90

        elif 0 < theta[theta_idx] <= 90:
            th0to360d[theta_idx] = theta[theta_idx]+270

        elif -90 < theta[theta_idx] <= 0:
            th0to360d[theta_idx] = 270+ theta[theta_idx]

        else:
            th0to360d[theta_idx] = theta[theta_idx]+270
    return th0to360d
