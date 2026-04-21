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
matplotlib.use('TkAgg')


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

########################## The main plotting starts here ##########################
x = np.linspace(-250, 250, 100)
y = np.linspace(-250, 250, 100)
x_grid, y_grid = np.meshgrid(x, y)
single_beam_50 = np.exp(-((x_grid)**2 + (y_grid)**2) / (2 * 50**2))
single_beam_80 = np.exp(-((x_grid)**2 + (y_grid)**2) / (2 * 80**2))

matplotlib.rcParams['toolbar'] = 'None'
plt.style.use('dark_background')
plt.ion()

fig, ax_grid = plt.subplots(2, 4)
ax = ax_grid.flatten()  # ax[0..7]: left-to-right, top-to-bottom

scrsize = pyautogui.size()
mng = plt.get_current_fig_manager()

if Serial:
    mng.full_screen_toggle()

dummy = np.zeros((100, 100))

# ── ax[0]: Single dish antenna position (scatter) ──────────────────────────
ax[0].set_title("Single Dish Antenna", fontsize=15)
ax[0].set_xlim(antenna_lim_min, antenna_lim_max)
ax[0].set_ylim(antenna_lim_min, antenna_lim_max)
ax[0].set_xlabel("x (m)")
ax[0].set_ylabel("y (m)")
ax[0].set_aspect('equal')
scatter_p0 = ax[0].scatter([], [])

# ── ax[1]: Model image ─────────────────────────────────────────────────────
im_p1 = ax[1].imshow(dummy, origin='lower', cmap=colormap)
ax[1].set_title("Picture of the source", fontsize=15)
ax[1].axes.get_xaxis().set_visible(False)
ax[1].axes.get_yaxis().set_visible(False)

# ── ax[2]: Single dish beam (STATIC — set once, never updated) ─────────────
im_p2 = ax[2].imshow(single_beam_50, origin='lower', cmap=colormap)
ax[2].text(-0.1, 0.5, r"$\otimes$", ha='center', va='center',
           fontsize=25, transform=ax[2].transAxes)
ax[2].set_title("Beam of single dish", fontsize=15)
ax[2].axes.get_xaxis().set_visible(False)
ax[2].axes.get_yaxis().set_visible(False)

# ── ax[3]: Single dish view ────────────────────────────────────────────────
im_p3 = ax[3].imshow(dummy, origin='lower', cmap=colormap)
ax[3].text(-0.1, 0.5, r"$=$", ha='center', va='center',
           fontsize=25, transform=ax[3].transAxes)
ax[3].set_title("Single Dish View", fontsize=15)
ax[3].axes.get_xaxis().set_visible(False)
ax[3].axes.get_yaxis().set_visible(False)

# ── ax[4]: ALMA perspective scatter ───────────────────────────────────────
ax[4].set_title("ALMA from source (perspective)", fontsize=15)
ax[4].text(4.5, -0.3, r"Powered by: Friendly VRI C.R. Purcell R. Truelove",
           ha='center', va='center', fontsize=8, transform=ax[4].transAxes)
ax[4].set_xlim(antenna_lim_min, antenna_lim_max)
ax[4].set_ylim(antenna_lim_min, antenna_lim_max)
ax[4].set_aspect('equal')
ax[4].set_xlabel("x (m)")
ax[4].set_ylabel("y (m)")
scatter_pp0 = ax[4].scatter([], [])

# ── ax[5]: Duplicate source image ─────────────────────────────────────────
im_p4 = ax[5].imshow(dummy, origin='lower', cmap=colormap)
ax[5].set_title("Picture of the source", fontsize=15)
ax[5].axes.get_xaxis().set_visible(False)
ax[5].axes.get_yaxis().set_visible(False)

# ── ax[6]: UV-coverage scatter / single ALMA beam (switches on singledish) ─
im_p5 = ax[6].imshow(dummy, origin='lower', cmap=colormap)
ax[6].text(-0.1, 0.5, r"$\otimes$", ha='center', va='center',
           fontsize=20, transform=ax[6].transAxes)
ax[6].set_title("Beam of the ALMA interferometer", fontsize=15)
ax[6].axes.get_xaxis().set_visible(False)
ax[6].axes.get_yaxis().set_visible(False)
scatter_p5        = ax[6].scatter([], [], s=1)
scatter_p5_mirror = ax[6].scatter([], [], s=1)

# ── ax[7]: ALMA / single-ALMA final image ─────────────────────────────────
im_p6 = ax[7].imshow(dummy, origin='lower', cmap=colormap)
ax[7].text(-0.1, 0.5, r"$=$", ha='center', va='center',
           fontsize=20, transform=ax[7].transAxes)
ax[7].set_title("ALMA view", fontsize=15)
ax[7].axes.get_xaxis().set_visible(False)
ax[7].axes.get_yaxis().set_visible(False)

fig.tight_layout()

# ── "Please add an antenna" overlay — hidden until needed ─────────────────
no_antenna_ax = fig.add_axes([0, 0, 1, 1])
no_antenna_ax.set_facecolor('black')
no_antenna_ax.text(0.5, 0.5, 'Please add an antenna',
                   color='white', fontsize=30,
                   ha='center', va='center',
                   transform=no_antenna_ax.transAxes)
no_antenna_ax.set_visible(False)
no_antenna_ax.set_zorder(10)

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
            no_antenna_ax.set_visible(True)
            for a in ax:
                a.set_visible(False)
            fig.canvas.draw_idle()
            plt.pause(2)
            continue  # back to top of loop

        # ── Antenna present: hide overlay, show subplots ──────────────────
        no_antenna_ax.set_visible(False)
        for a in ax:
            a.set_visible(True)

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

                # ax[0]: single dish — static dot at origin
                scatter_p0.set_offsets([[0, 0]])

                # ax[4]: ALMA perspective projection
                hrangle_rad  = np.radians(hourangle * 15)
                dec_rad      = np.radians(DEC)
                xx_earth_cen = -yy_antpos * np.sin(np.radians(-23.023))
                yy_earth_cen =  xx_antpos
                zz_earth_cen =  yy_antpos * np.cos(np.radians(-23.023))
                xx_antpos_proj = -(xx_earth_cen * np.sin(hrangle_rad) +
                                    yy_earth_cen * np.cos(hrangle_rad))
                yy_antpos_proj = (-xx_earth_cen * np.sin(dec_rad) * np.cos(hrangle_rad) +
                                   yy_earth_cen * np.sin(dec_rad) * np.sin(hrangle_rad) +
                                   zz_earth_cen * np.cos(dec_rad))

                if hourangle > 0:
                    scatter_pp0.set_offsets(np.c_[yy_antpos_proj,  xx_antpos_proj])
                elif hourangle < 0:
                    scatter_pp0.set_offsets(np.c_[-yy_antpos_proj, -xx_antpos_proj])
                else:
                    scatter_pp0.set_offsets(np.c_[xx_antpos_proj,  -yy_antpos_proj])

                # ax[1]: model image
                if imagefile == "/home/kiosk/alma_sid_07_05_2025/Alma_main/models/mistery_med.png":
                    qmarkimg = mpimg.imread(
                        '/home/kiosk/alma_sid_07_05_2025/Alma_main/models/mistery_qmark.png')
                    im_p1.set_data(qmarkimg)
                else:
                    data_p1 = np.real(obsMan.modelImgArr)
                    im_p1.set_data(data_p1)
                    im_p1.set_clim(vmin=data_p1.min(), vmax=data_p1.max())

                # ax[2]: single dish beam — STATIC, no update needed

                # ax[3]: single dish convolved view
                sgdish_image = gaussian_filter(obsMan.modelImgArr, 50, mode='constant')
                im_p3.set_data(sgdish_image)
                im_p3.set_clim(vmin=sgdish_image.min(), vmax=sgdish_image.max())

                # ax[5]: duplicate source image
                data_p4 = obsMan.modelImgArr
                im_p4.set_data(data_p4)
                im_p4.set_clim(vmin=data_p4.min(), vmax=data_p4.max())

            

                if not singledish:
                    ax[6].cla()
                    
                    u_data = obsMan.arrsSelected[0]['uArr_lam']
                    v_data = obsMan.arrsSelected[0]['vArr_lam']
                    
                    ax[6].scatter(u_data, v_data, s=1)
                    ax[6].scatter(-u_data, -v_data, s=1)
                    
                    # Force limits to be symmetric and fill the axes
                    uv_max = np.max(np.abs(np.concatenate([u_data, v_data]))) * 1.1
                    ax[6].set_xlim(-uv_max, uv_max)
                    ax[6].set_ylim(-uv_max, uv_max)
                    
                    ax[6].text(-0.1, 0.5, r"$\otimes$", ha='center', va='center',
                            fontsize=20, transform=ax[6].transAxes)
                    ax[6].set_title("Beam of the ALMA interferometer", fontsize=15)
                    ax[6].axes.get_xaxis().set_visible(False)
                    ax[6].axes.get_yaxis().set_visible(False)
                    ax[6].set_facecolor('black')
                    
                    if verbose:
                        thistime = time.time()
                        print("TIMING %f  %f uv coverage" % (thistime - starttime, thistime - lasttime))
                        lasttime = thistime

                    # ax[7]: ALMA observed image
                    data_p6 = np.real(obsMan.obsImgArr)
                    im_p6.set_data(data_p6)
                    im_p6.set_clim(vmin=data_p6.min(), vmax=data_p6.max())
                    ax[7].set_title("ALMA view", fontsize=15)

                else:
                    scatter_p5.set_visible(False)        # explicitly hide scatter
                    scatter_p5_mirror.set_visible(False) # explicitly hide scatter
                    im_p5.set_visible(True)
                    im_p5.set_data(single_beam_80)
                    im_p5.set_clim(vmin=single_beam_80.min(), vmax=single_beam_80.max())
                    ax[6].set_title("Beam of single ALMA antenna", fontsize=15)

                    sgdish_image_80 = gaussian_filter(obsMan.modelImgArr, 80, mode='constant')
                    im_p6.set_data(sgdish_image_80)
                    im_p6.set_clim(vmin=sgdish_image_80.min(), vmax=sgdish_image_80.max())
                    ax[7].set_title("Single ALMA View", fontsize=15)

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