#%%


##### TO DO ###########
"""
1. A file somewhere, preferably on the desktop of the computer or in the GitHub folder alongside the code, 
with clear instructions regarding how we should name variables, functions, classes etc. (this can be though camelcase) 
How we should go about documenting any changes we make through comments in the code 
(obviously our own code will be run on our own branch but it is still important that we document the changes we make). 
And finally how we run the program in case it doesn’t start on system boot, including exact file we should run if there 
are any differences between them as there are at this moment in time.

"""

#####################   Importing functions ############################
import sys
sys.path.append("functions")

import serial
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from astropy.io import ascii
import time

import vriCalc
from vriCalc import observationManager
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
import pickle
import os
import pyautogui
import matplotlib.pylab as pylab
import functions2run
import matplotlib

import Alma_UI
#matplotlib.use('TkAgg')


#####################   Setting up parameters ############################

params = {'axes.titlesize': 'small'}
pylab.rcParams.update(params)
colormap = 'inferno'

LOOP_TIME = 0.1  # seconds
antenna_lim_min, antenna_lim_max = -3000, 3000

FREQ = 3e5  # MHz
DEC  = -40  # declination

Serial  = False  # True = Arduino attached
verbose = True
screenZoom = 0.7
ZOOM = 15
PLOT = True


print(os.getcwd())
LOOP_TIME = 0.1 #seconds
scale_array =40.0
antenna_lim_min ,antenna_lim_max= -3000,3000
hourangle_start = -2. # hourangle start of obs
hourangle_end = +2.  # hourangle end of obs
FREQ = 3e5 #MHz
DEC = -40 #declination
pixel_scale = 0.05 #arcseconds


if Serial:
    ser = functions2run.getserialinterface()
else:
    ser = None

starttime = time.time()
lasttime  = starttime

ui = Alma_UI.ALMA_UI()

########################## The main plotting starts here ##########################
x = np.linspace(-250, 250, 100)
y = np.linspace(-250, 250, 100)
x_grid, y_grid = np.meshgrid(x, y)
single_beam_50 = np.exp(-((x_grid)**2 + (y_grid)**2) / (2 * 50**2))
single_beam_80 = np.exp(-((x_grid)**2 + (y_grid)**2) / (2 * 80**2))

matplotlib.rcParams['toolbar'] = 'None'
plt.style.use('dark_background')
plt.ion()

fig, ax = plt.subplots()
#ax = ax_grid.flatten()  # ax[0..7]: left-to-right, top-to-bottom

scrsize = pyautogui.size()
mng = plt.get_current_fig_manager()

if Serial:
    mng.full_screen_toggle()

dummy = np.zeros((100, 100))

# Adam Wikström - 2026-04-21 - removed all figures but the ALMA View

# Adam Wikström - 2026-04-21 - Adjusted the Alma View to just show the graph and nothing else
im_p6 = ax.imshow(dummy, origin='lower', cmap=colormap)

# Remove
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_frame_on(False)

fig.tight_layout()
ax.set_visible(True)

# Adam Wikström - 2026-04-21 - removed "please add an antenna" overlay
# ── Clean exit: press 'q' or Escape in the plot window ────────────────────
exit_flag = [False]

def on_key(event):
    if event.key in ('q', 'escape'):
        exit_flag[0] = True

fig.canvas.mpl_connect('key_press_event', on_key)

##########################################################################
while not exit_flag[0]:
    print("------------------------------------------------------------------------------------------")
    if verbose:
        starttime = time.time()
        lasttime  = starttime
        thistime  = time.time()
        print("TIMING %f  %f" % (thistime - starttime, thistime - lasttime))
        lasttime = thistime

    try:
        if verbose:
            thistime = time.time()
            print("TIMING %f  %f" % (thistime - starttime, thistime - lasttime))
            lasttime = thistime
            computetimestart = time.time()
            thistime = time.time()
            print("TIMING %f  %f" % (thistime - starttime, thistime - lasttime))
            lasttime = thistime

        print('CHECK 1')
        bit_pos1, bit_pos2, buttons_config, buttons_image, ant_pos, \
            xx_antpos, yy_antpos, singledish, is_there_antenna = \
            functions2run.waitforserialchange(ser, IsThereArdruino=Serial, verbose=verbose)

        # ── No antenna: show overlay, skip all computation ────────────────
        if not is_there_antenna:
            # no_antenna_ax.set_visible(True) - Adam W - 2026-04-21 commented out
            # for a in ax:
            #     a.set_visible(False)
            fig.canvas.draw_idle()
            plt.pause(2)
            continue  # back to top of loop

        # ── Antenna present: hide overlay, show subplots ──────────────────
        # no_antenna_ax.set_visible(False) - Adam W - 2026-04-21 commented out
        # for a in ax:
        #     a.set_visible(True)

        print('CHECK 2')

        functions2run.write_alma_config_file(ant_pos)
        obsMan = observationManager(verbose=True, debug=True)

        if verbose:
            thistime = time.time()
            print("TIMING make observation manager (debugTrue) %f  %f" % (thistime - starttime, thistime - lasttime))
            lasttime = thistime

        obsMan.get_available_arrays()

        if verbose:
            thistime = time.time()
            print("TIMING make observation manager read arrays %f  %f" % (thistime - starttime, thistime - lasttime))
            lasttime = thistime

        imagefile, pixel_scale, integration_time, hourangle, \
            hourangle_start, hourangle_end = \
            functions2run.select_model_and_hourangle(bit_pos2, buttons_config, buttons_image)

        if verbose:
            print(imagefile, integration_time, hourangle_start, hourangle_end,
                  "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            print("TIMING preparations: %f" % (time.time() - computetimestart))
            computetimestart = time.time()
            thistime = time.time()
            print("TIMING %f  %f" % (thistime - starttime, thistime - lasttime))
            lasttime = thistime

        obsMan.select_array('ALMA_Custom-lego-alma',
                            haStart=hourangle_start, haEnd=hourangle_end, sampRate_s=300)
        obsMan.get_selected_arrays()

        if verbose:
            print("TIMING select arrays: %f" % (time.time() - computetimestart))
            computetimestart = time.time()
            thistime = time.time()
            print("TIMING %f  %f" % (thistime - starttime, thistime - lasttime))
            lasttime = thistime

        obsMan.set_obs_parms(FREQ, DEC)

        if verbose:
            print("TIMING set obs parms: %f" % (time.time() - computetimestart))
            computetimestart = time.time()

        obsMan.calc_uvcoverage()
        if verbose:
            print("TIMING uv coverage: %f" % (time.time() - computetimestart))
            computetimestart = time.time()
            thistime = time.time()
            print("TIMING %f  %f" % (thistime - starttime, thistime - lasttime))
            lasttime = thistime

        obsMan.load_model_image(imagefile)
        obsMan.set_pixscale(pixel_scale)

        if verbose:
            thistime = time.time()
            print("TIMING %f  %f" % (thistime - starttime, thistime - lasttime))
            lasttime = thistime

        if PLOT:
            try:
                # Calculate the FFT of the model image
                obsMan.invert_model()

                if verbose:
                    thistime = time.time()
                    print("TIMING invert model %f  %f" % (thistime - starttime, thistime - lasttime))
                    lasttime = thistime
                    computetimestart = time.time()

                obsMan.grid_uvcoverage()

                if verbose:
                    thistime = time.time()
                    print("TIMING %f  %f  grid uv_coverage" % (thistime - starttime, thistime - lasttime))
                    lasttime = thistime

                print("obsMan.calc_beam()")
                obsMan.calc_beam()

                if verbose:
                    thistime = time.time()
                    print("TIMING  %f  %f  calc beam" % (thistime - starttime, thistime - lasttime))
                    lasttime = thistime

                obsMan.invert_observation()

                if verbose:
                    thistime = time.time()
                    print("TIMING %f  %f  invert observations" % (thistime - starttime, thistime - lasttime))
                    lasttime = thistime

                ########## FIX 2: update existing artists, no clf() ##########

                    # Adam Wikström - 2026-04-21
                    # Removed updating of unused graphs
                    # ax[7]: ALMA observed image
                    data_p6 = np.real(obsMan.obsImgArr)
                    im_p6.set_data(data_p6)
                    im_p6.set_clim(vmin=data_p6.min(), vmax=data_p6.max())

            except Exception as e:
                print(e)

        else:
            print("Not plotting")

        if verbose:
            print(ant_pos)
            print("Time for one loop: %s" % str(time.time() - starttime))

        print("Pausing for   ", LOOP_TIME)
        print(plt.isinteractive())

        # FIX 2: draw_idle only redraws what changed — faster than plt.draw()
        fig.canvas.draw_idle()
        plt.pause(2)

    except KeyboardInterrupt:
        print("Keyboard interrupt — closing exhibit.")
        break

    except Exception as e:
        print(f"Unhandled error: {e}")
        time.sleep(1)
        continue

# ── Graceful shutdown ──────────────────────────────────────────name────────
plt.close('all')
# ser.close()

# %%