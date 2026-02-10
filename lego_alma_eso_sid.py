#%%

"""
Matplot kinda crashing betweenn the inputs.
The plots can be created only when there is a change in serial input. Will save some computing power
"""
###import cv2

import serial
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from astropy.io import ascii
import time
from Imports.vriCalc import observationManager
from astropy.convolution import Gaussian2DKernel,convolve
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
import pickle
import os
import sys
import pyautogui

import matplotlib.pylab as pylab

import matplotlib#change here 10-2-2026
matplotlib.use('TkAgg')#change here 10-2-2026

import sys
# sys.path.append('/home/kiosk/alma_sid_07_05_2025/Alma_main/')

params = {'axes.titlesize':'small'}
pylab.rcParams.update(params)
colormap = 'inferno'

buttons_inp = "0100100"
# attenas_inp = "111111111111111111111111111111111111111111111"
attenas_inp = "000000000000000000000000000000000001000000000"

# https://medium.com/@CodyReichert/how-to-hide-title-bars-in-kde-plasma-5-348e0df4087f
# https://matplotlib.org/api/animation_api.html
# https://stackoverflow.com/questions/17212722/matplotlib-imshow-how-to-animate

# Runtime of a full loop: 1.3s
# scatter plots:      0.2 
# invert observation: 0.03 
# invert model:       0.03 
# calc beam:          0.04
# grid uv-converage:  0.15 
# uv-coverage:        0.004 
# select arrays:      0.000042
# set obs parms:      0.000003
# preparation:        0.11
# clf:                0.25 !!!!
# logo:               0.07 !!!
# 

#
# observation manager: if it is not created each time newly, the total time increases from loop to loop!
# obsMan = observationManager(verbose=False, debug=False) does save about 0.3 seconds!!!

# preparations
# all plots together: 0.8 !!
# => use animation so that only the data has to be redone, not the axis/scaling/limit setting etc
# 
#
#
#


#The Friendly Virtual Radio Interferometer (VRI) needs a config file whith the positions of the individual antennas.
#this function reads a list of antenna positions (XY). It then opens a template file from where it copies the
#header required by VRI and writes it to the VRI config file. It then writes the rest of the config file based 
#on the x,y positions of the antennas in ant_pos.

print(os.getcwd())
LOOP_TIME = 0.1 #seconds
scale_array =40.0
antenna_lim_min ,antenna_lim_max= -3000,3000
hourangle_start = -2. # hourangle start of obs
hourangle_end = +2.  # hourangle end of obs
FREQ = 3e5 #MHz
DEC = -40 #declination
pixel_scale = 0.05 #arcseconds
# imagefile1 = "/home/kiosk/alma_sid_07_05_2025/Alma_main/image/agb_star.jpg"
# imagefile2 = "/home/kiosk/alma_sid_07_05_2025/Alma_main/image/galaxy_gas.jpg"
# imagefile3 = "/home/kiosk/alma_sid_07_05_2025/Alma_main/image/hltau.jpg"
# imagefile4 = "/home/kiosk/alma_sid_07_05_2025/Alma_main/image/outflow.jpg"


imagefile1 = "./image/agb_star.jpg"
imagefile2 = "./image/galaxy_gas.jpg"
imagefile3 = "./image/hltau.jpg"
imagefile4 = "./image/outflow.jpg"

hourangle  = 0
integration_time = 2




bitdict = {'hourangle_m6':     1,
           'hourangle_0':      2,
           'hourangle_p6':     3,
           'agb_star':      6,
           'galaxy_gas':       5,
           'hltau':       4,
           'outflow':   7}



bitdict_config = {'hourangle_m6':     1,
           'hourangle_0':      2,
           'hourangle_p6':     3}


bitdict_image = {'agb_star': 3,
           'galaxy_gas': 1,
           'hltau': 2,
           'outflow': 4}

NoSerial = False

webcam = False


verbose = True

screenZoom = 0.7

# The zoom value seems to control the fouriertransform image. Original value 24
# -----------------------------------------------------------------------------
ZOOM = 15

PLOT = True
# antenna_filename = "/home/kiosk/alma_sid_07_05_2025/Alma_main/sid_ant_text.csv"
antenna_filename = "sid_ant_text_latest_test.csv"

# model_logo_filename = "/home/kiosk/alma_sid_07_05_2025/Alma_main/models/aifa-logo.png"
model_logo_filename = "./models/chalmers-uni.png"


############### Functions ###################

def get_antenna_dict(filename=antenna_filename):
    ant_database = pd.read_csv(filename, sep=";", header='infer')
    ant_bits = np.array(ant_database['bit']) +1
    ant_bits_posx = np.array(ant_database['posx'], dtype=float)
    ant_bits_posy = np.array(ant_database['posy'], dtype=float)

    position_array = (np.array([ant_bits_posx,ant_bits_posy]).T)*scale_array
    ant_dict = dict(zip(ant_bits,position_array))

    return ant_dict

ant_dict = get_antenna_dict()


def getserialinterface(deviceroot="/dev/ttyACM", maxdevice=3, boudrate=115200):
    """returns the serial interface"""       
    
    while True:
        count = 0
        while count < maxdevice:
            device = f"{deviceroot}{count}"
            try:
            # parameter timeout freezes it in the place
               ser = serial.Serial(device, baudrate=boudrate)
               print(f"Serial interface {device} found!")           
               return ser
            except:
               print(f"Serial interface {device} not found!. Trying the next one ...")           
               count += 1   
               # for testing, use this line, but uncomment for real runs
               return None

        print("No serial device found. Please plug it in. ...")
        time.sleep(2)
    

ser = getserialinterface()
print(ser)
def waitforserialchange(ser, bitdict, ant_dict, npadarray=45, verbose=False):
    """Waits for a change to the serial interface, i.e. any change of a contact.
       Pauses in case there is no contact on one of the hour angles
    Parameters:   
    device (str)
       device to which the serial interface is connected
       
    npadarray (int)
       split position of the 64 input array of the microcontroller into the two arrays   
       the first npadarray are for the antennas, the remaining for the buttons
       
    Return:
    list
       positions of antennas with closed contacts
       
    list   
       positions of buttons with closed contacts

    """   
    print('CHECK3')
    validhaselection = False
    
    while not validhaselection:
        if (NoSerial):
            # buttons = "0110000"
            buttons = buttons_inp
            # attenas = "111111111111111111111111111111111111111111111"
            attenas = attenas_inp

            serialinput = attenas  + buttons
            # serialinput= "11111111000111110001101000111000000000000000 0000000100010000100" 
            print('this is the length of serial input',len(serialinput))
        else:
            print('CHECK4')
            serialinput = str(ser.readline().decode("utf-8").strip())
            if ser.in_waiting > 0:
                print('CHECK4')
                # In case there were several triggers for redrawing recorded,
                # take the last one
                serialbuffer = ser.read(ser.in_waiting).decode("utf-8").strip().split('\n')  
                if verbose:
                    print(f"Inputs recorded in the last loop {len(serialbuffer)}. Taking the last one")
                serialinput = serialbuffer[-1]
        time.sleep(1)

        linein2  = serialinput[npadarray:]
        linein   = serialinput[:npadarray]

        print(len(linein))
        print(len(linein2))
        
        count = 0
        while int(linein) == 0:
            print("Waiting for a valid antenna selection ...")
            time.sleep(2)
            serialinput = str(ser.readline().decode("utf-8").strip())
            linein2  = serialinput[npadarray:]
            linein   = serialinput[:npadarray]
            print(linein)
#            print(linein)
#            print(linein2)
#            count += 1
#            if count > 1:
#                pass
#                buttons_inp_new = "0010100"
#                attenas_inp_new = "000000010000100010000100000010000000010001000"
#                serialinput = attenas_inp_new  + buttons_inp_new

        
        # The array from the controller should have 45 positions for 45 antennas. (npadarray)
        # Antenna list will have a list of antenna numbers and they should start as 1.
        # This is why we add a "0" to the array.
        # ------------------------------------------------------------------------------------
        bit_pos1  = np.where(np.array([bit == '1' for bit in "0"+linein]) == True)[0]
        bit_pos2 = np.where(np.array([bit == '1' for bit in "0"+linein2]) == True)[0]
        buttons_config = np.where(np.array([bit == '1' for bit in "0"+linein2[:2]]) == True)[0]
        buttons_image = np.where(np.array([bit == '1' for bit in "0"+linein2[3:]]) == True)[0]
        if verbose:
            print(serialinput)        
            print((len(serialinput)))
        
            print(f"antennas:   {linein}")
            print(f"buttons:    {linein2}")
        
            print(f"antenna list:    {bit_pos1}")
            print(f"button  list:    {bit_pos2}      ({' '.join([key for key in bitdict if bitdict[key] in bit_pos2])})")

            print(f"buttons_config:    {buttons_config}")
            print(f"buttons_image:    {buttons_image}")




        #import ipdb; ipdb.set_trace()
        ant_pos =  np.array([np.array(ant_dict[bb]) for bb in bit_pos1]) #multiply with a factor, default 13 to scale up the array baselines

        print(ant_pos)
        if len(ant_pos)>0 and (bitdict['hourangle_m6'] in bit_pos2) or (bitdict['hourangle_0'] in bit_pos2) or (bitdict['hourangle_p6'] in bit_pos2):

            xx_antpos, yy_antpos = ant_pos.T
            if len(ant_pos) > 1: 
                create_config_file(ant_pos)
                singledish = False
            elif len(ant_pos)==1: #one antenna show a singledish image
                singledish = True    

            validhaselection = True
        else:
            print('Waiting for valid hourangle selection and at least one antenna ...')

    # TODO: convert everything into the usage of bit_pos1 and bit_pos2. Rename bit_pos1 to bit_pos11
    return bit_pos1, bit_pos2, buttons_config,buttons_image,ant_pos, xx_antpos, yy_antpos, singledish



#bitdict_config = {'hourangle_m6':     1,
#           'hourangle_0':      2,
#           'hourangle_p6':     3}


#bitdict_image = {'agb_star': 1,
#           'galaxy_gas': 2,
#           'hltau': 3,
#           'outflow': 4}

# bit_pos1, bit_pos2, buttons_config,buttons_image,ant_pos, xx_antpos, yy_antpos, singledish = waitforserialchange(ser, bitdict, ant_dict, verbose=verbose) 


def select_model_and_hourangle(bitdict, bit_pos2,bitdict_config,bitdict_image,buttons_config,buttons_image, verbose=False):
    """Select the images and return the corresponding pixel scale and integration time and hour angle   
    """

    webcam = False

    pixel_scale = 0.0055


# bitdict = {'hourangle_m6':     1,
#            'hourangle_0':      2,
#            'hourangle_p6':     3,
#            'agb_star':      4,
#            'galaxy_gas':       5,
#            'hltau':       6,
#            'outflow':   7}



    #load the imagefiles based on the bit values from the box
    if len(buttons_image) == 0:
        imagefile = imagefile1
    else:
        if  bitdict_image['agb_star'] == buttons_image[0]:
            imagefile = imagefile1
        elif bitdict_image['galaxy_gas']  == buttons_image[0]:
            imagefile = imagefile2
        elif bitdict_image['hltau']  == buttons_image[0]:
            imagefile = imagefile3
            # pixel_scale = 0.1
            # webcam = True
        elif bitdict_image['outflow']  == buttons_image[0]:
            imagefile = imagefile4
        else:
            imagefile = imagefile1
        # pixel_scale = 0.1
        # webcam = True
    # print(imagefile)
    # if not bitdict['agb_star'] in bit_pos2:
    #     imagefile = imagefile1
    # elif not bitdict['galaxy_gas'] in bit_pos2:
    #     imagefile = imagefile2
    # elif not bitdict['hltau'] in bit_pos2:
    #     imagefile = imagefile3
    #     pixel_scale = 0.1
    #     webcam = True
    # elif not bitdict['outflow'] in bit_pos2:
    #     imagefile = imagefile4
    #     pixel_scale = 0.1
    #     webcam = True

    # if (not bitdict['sgal'] in bit_pos2) and not (bitdict['cam'] in bit_pos2) and (not bitdict['m51'] in bit_pos2):
    #     imagefile = "models/marilyn-einstein.png"
    #     pixel_scale = 0.1
    
    # if bitdict['sgal'] in bit_pos2 and bitdict['m51'] in bit_pos2 and bitdict['cam'] in bit_pos2:
    #     imagefile= "models/mistery_med.png"
        
    # which hourangle are we observing, is it a full track of hourangle or only dawn/dusk/meridian        
    # if bitdict['fulltrk'] in bit_pos2:
    integration_time = 3
    if bitdict['hourangle_0'] in bit_pos2:
        hourangle = 0 
        hourangle_start = hourangle - integration_time * 0.5
        hourangle_end   = hourangle + integration_time * 0.5
    elif bitdict['hourangle_m6'] in bit_pos2:
        hourangle = -5 
        hourangle_start = hourangle
        hourangle_end   = hourangle_start + integration_time
    elif bitdict['hourangle_p6'] in bit_pos2:
        hourangle = 5
        hourangle_end   = hourangle
        hourangle_start = hourangle_end -integration_time

    # else:
    #     integration_time = 12
    #     hourangle = 0

#    hourangle_start = hourangle - integration_time * 0.5
#    hourangle_end   = hourangle + integration_time * 0.5
    
    if verbose:
        print(imagefile, integration_time, hourangle_start, hourangle_end)

    return webcam, imagefile, pixel_scale, integration_time, hourangle, hourangle_start, hourangle_end
    # return None

# select_model_and_hourangle(bitdict, bit_pos2,bitdict_config,bitdict_image,buttons_config,buttons_image)
  
def create_config_file(ant_pos):
    try:
        with open("./arrays_old/template_file.txt","r") as hfile,open("./arrays_old/lego_alma.config","w") as fullfile:
            header = hfile.readlines()
            [fullfile.write(h) for h in header]
            for p in ant_pos:
                fullfile.write(str(p[0])+","+str(p[1])+"\n")
        return True
    except Exception as e:
        print("Error in creating a config file!", e)
        return False


def write_alma_config_file(filename, antenna_coords):
    filename = './arrays/' +filename
    with open(filename, 'w') as f:
        f.write("#-----------------------------------------------------------------------------#\n")
        f.write("#                                                                             #\n")
        f.write("# Array definition file for ALMA, Cycle 6, Config 5, 12-m antennas.          #\n")
        f.write("#                                                                             #\n")
        f.write("#-----------------------------------------------------------------------------#\n")
        f.write("# Baseline Range: 15m-1.4km\n\n")
        f.write("# Name of the telescope\n")
        f.write("telescope = ALMA\n\n")
        f.write("# Name of the configuration\n")
        f.write("config = Custom-lego-alma\n\n")
        f.write("# Latitude of the array centre\n")
        f.write("latitude_deg = -23.0229\n\n")
        f.write("# Antenna diameter\n")
        f.write("diameter_m = 12.0\n\n")
        f.write("# Antenna coordinates (offset E, offset N)\n")
        for e, n in antenna_coords:
            f.write(f"{e}, {n}\n")


#change dir to where the script is
#os.chdir("/home/dpg-physik/Downloads/friendlyVRI-master")

# here we read the file which tells the program the antenna number, the x,y positions, and the corresponding bit
# from the electronics. This file is generated once when setting up the lego alma.



# Make sure we only get a serial interface if we have requested that one is used.
# -------------------------------------------------------------------------------
if (not NoSerial):
    ser = getserialinterface()
else:
    ser = None

matplotlib.rcParams['toolbar'] = 'None'
plt.style.use('dark_background')

plt.figure()#change here 10-2-2026
plt.ion()#change here 10-2-2026
# all images should have the same pixel size. Depending on the screen size. 
# For example 400x400. This should speed up the processing.


starttime = time.time()
lasttime  = starttime

imglogo=mpimg.imread(model_logo_filename)                

todo = """
TODO:
 - implement a verbose setting and set it to False for the production run. The printing to the console also takes time unnecessarily.
   In production, the script should be completely SILENT! before a new input comes, the timing of the last loop (without the input waiting
   time can be plotted. That does not cost.
 - can a larger sampling rate (600) provide more or less the same result for the visitors but be faster?
 - marylin Einstein??? Replace 
 - if the image has not changed, then the image does not need to be read and not need to be inverted

 - the initial plotting seems fine.
 - we will have to do the xlim setting for the fft etc
 - can all images be transformed to the same pixel size? Should be possible!
 - once that's done, remove the plotting and subplotting from the stuff below
 - add  im.set_array(f(x, y)) instead
 - or   im.set_data, depending ...
 - make the while loop a function called 'update'
 - place 
ani = animation.FuncAnimation(fig, update, blit=True)

at the end
"""



print((plt.get_backend()))

# This resizes the figure to the size of the screen
# Comment out lines below to remove.
# The size of the window is controlled by the screenZoom variable.
# screenZoom = 1 is fullscreen. 0< screenZoom <=1
# -----------------------------------------------------------------
scrsize = pyautogui.size()
mng = plt.get_current_fig_manager()
# mng.resize(int(scrsize[0]*screenZoom), int(scrsize[1]*screenZoom))
#mng.resize(*mng.window.showMaximized())
#mng.window.showMaximized()
mng.full_screen_toggle()
# KDE: title-bar, right click, set 'apply initially' to remove title, to maximise etc

#--------------------------------------------------------------
#----------------------------------------------------------------



########################## Start of the main program ##########################
#start of main prgoram loop
#serialinput="0"
#serialinputlast=""
imglogo = mpimg.imread(model_logo_filename)                

Flag = True
while True:
    
#    Flag = False
    
    print("------------------------------------------------------------------------------------------")
    if verbose:
        starttime = time.time()
        
        lasttime = starttime
        thistime = time.time(); print(("TIMING %f  %f" %(thistime-starttime,thistime-lasttime))); lasttime= thistime
    #we fiddled with some constants here
    #    hourangle_start = hourangle - 0.5
    #    hourangle_end = hourangle + 0.5
    #    hourangle+=1
    #if hourangle > 6: 
    #        hourangle = -6
    #        hourangle_start = -6
    #        hourangle_end = 6
    #FREQ += 1000
    #if FREQ >= 3e4: FREQ = 5e3
    try:
        if verbose:
            thistime = time.time(); print(("TIMING %f  %f" %(thistime-starttime,thistime-lasttime))); lasttime= thistime
            computetimestart=time.time()    
        #start the VRI with the latest antenna position config file, which we generate on every loop baased on the
        #bit strings from the boxes.
        
        ### linein = ser.readline()
        
        # moved above the serial input. Does not cost time anymore.       
            thistime = time.time(); print(("TIMING %f  %f" %(thistime-starttime,thistime-lasttime))); lasttime= thistime
        print('CHECK 1')    
        bit_pos1, bit_pos2, buttons_config,buttons_image,ant_pos, xx_antpos, yy_antpos, singledish = waitforserialchange(ser, bitdict, ant_dict, verbose=verbose) 
        print('CHECK 2')   
        # bit_pos1, bit_pos2, ant_pos, xx_antpos, yy_antpos, singledish = waitforserialchange(ser, bitdict, ant_dict, verbose=verbose) 
        write_alma_config_file("lego_alma.config", ant_pos)
        obsMan = observationManager(verbose=True, debug=True)

        if verbose:
            thistime = time.time(); print(("TIMING make observation manager (debugTrue) %f  %f" %(thistime-starttime,thistime-lasttime))); lasttime= thistime
        obsMan.get_available_arrays()
        
        if verbose:
            thistime = time.time(); print(("TIMING make observation manager read arrays %f  %f" %(thistime-starttime,thistime-lasttime))); lasttime= thistime
        # bit_pos1, bit_pos2, buttons_config,buttons_image,ant_pos, xx_antpos, yy_antpos, singledish = waitforserialchange(ser, bitdict, ant_dict, verbose=verbose) 
        # # bit_pos1, bit_pos2, ant_pos, xx_antpos, yy_antpos, singledish = waitforserialchange(ser, bitdict, ant_dict, verbose=verbose) 
        # write_alma_config_file("lego_alma.config", ant_pos)

        for i in range(1,9):
            pp = plt.subplot(2,4,i)
            xmin, xmax = pp.get_xlim()
            ymin, ymax = pp.get_ylim()            
            rectangle = matplotlib.patches.Rectangle((xmin,xmin), xmax-xmin, ymax-ymin, color='black', alpha=0.4)
            pp.add_patch(rectangle)
        # webcam, imagefile, pixel_scale, integration_time, hourangle, hourangle_start, hourangle_end = select_model_and_hourangle(bitdict, bit_pos2)
        webcam, imagefile, pixel_scale, integration_time, hourangle, hourangle_start, hourangle_end = select_model_and_hourangle(bitdict, bit_pos2,bitdict_config,bitdict_image,buttons_config,buttons_image)
        
        if verbose:
            print(imagefile, integration_time,hourangle_start,hourangle_end,"<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            print(("TIMING preparations: %f" %(time.time()-computetimestart)))
            computetimestart=time.time()  
            thistime = time.time(); print(("TIMING %f  %f" %(thistime-starttime,thistime-lasttime))); lasttime= thistime
            
        # set up the VRI for this specific obs
        # Select array configurations and hour-angle ranges.
        obsMan.select_array('ALMA_Custom-lego-alma',haStart = hourangle_start,haEnd = hourangle_end,sampRate_s=300)
        #obsMan.select_array('ALMA_Cycle6-C43-5',haStart = hourangle_start,haEnd = hourangle_end,sampRate_s=300)
        #obsMan.select_array('ALMA_Cycle6-C43-1',haStart = hourangle_start,haEnd = hourangle_end,sampRate_s=300)
        obsMan.get_selected_arrays()
        
        if verbose:
            print(("TIMING select arrays: %f" %(time.time()-computetimestart)))
            computetimestart=time.time()  
            thistime = time.time(); print(("TIMING %f  %f" %(thistime-starttime,thistime-lasttime))); lasttime= thistime      
            
        # Set the observing frequency (MHz) and source declination (deg).
        obsMan.set_obs_parms(FREQ, DEC)
        
        if verbose:
            print(("TIMING set obs parmss: %f" %(time.time()-computetimestart)))            
            computetimestart=time.time() 
         
        # Calculate the uv-coverage
        obsMan.calc_uvcoverage()
        if verbose:
            print(("TIMING uv coverage: %f" %(time.time()-computetimestart)))
            computetimestart=time.time() 
            thistime = time.time(); print(("TIMING %f  %f" %(thistime-starttime,thistime-lasttime))); lasttime= thistime
        
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxxx
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        
        # if webcam is active do this

        obsMan.load_model_image(imagefile)
        obsMan.set_pixscale(pixel_scale)
        
        if verbose:    
            thistime = time.time(); print(("TIMING %f  %f" %(thistime-starttime,thistime-lasttime))); lasttime= thistime
        if PLOT:
            try:               
                # Calculate the FFT of the model image
                obsMan.invert_model()

                if verbose: 
                    thistime = time.time(); print(("TIMING invert model %f  %f" %(thistime-starttime,thistime-lasttime))); lasttime= thistime
                    computetimestart=time.time()                    
                # if not singledish:    
                #     # Grid the uv-coverage onto the same pixels as the FFT as the model image
                #     obsMan.grid_uvcoverage()
                    
                #     if verbose:
                #         thistime = time.time(); print(("TIMING %f  %f  grid uv_coverage" %(thistime-starttime,thistime-lasttime))); lasttime= thistime
                #     # Create the beam image
                #     print("obsMan.calc_beam()")
                #     obsMan.calc_beam()
                    
                #     if verbose:
                #         thistime = time.time(); print(("TIMING  %f  %f  calc beam" %(thistime-starttime,thistime-lasttime))); lasttime= thistime
                #     # Apply the uv-coverage and create observed image
                #     obsMan.invert_observation()

  
                    # Grid the uv-coverage onto the same pixels as the FFT as the model image
                obsMan.grid_uvcoverage()
                
                if verbose:
                    thistime = time.time(); print(("TIMING %f  %f  grid uv_coverage" %(thistime-starttime,thistime-lasttime))); lasttime= thistime
                # Create the beam image
                print("obsMan.calc_beam()")
                obsMan.calc_beam()
                
                if verbose:
                    thistime = time.time(); print(("TIMING  %f  %f  calc beam" %(thistime-starttime,thistime-lasttime))); lasttime= thistime
                # Apply the uv-coverage and create observed image
                obsMan.invert_observation()



                if verbose:  
                    thistime = time.time(); print(("TIMING %f  %f  invert observations " %(thistime-starttime,thistime-lasttime))); lasttime= thistime
                plt.clf()
                
                if verbose:
                    thistime = time.time(); print(("TIMING %f %f  clf" %(thistime-starttime,thistime-lasttime))); lasttime= thistime
                plogo = plt.subplot(111)
                plogo.imshow(imglogo)
                
                plogo.set_position([0.0,0.0,0.1,0.1],which='both')
                plogo.axes.get_xaxis().set_visible(False)
                plogo.axes.get_yaxis().set_visible(False)

                #plto antenna position
                p0 = plt.subplot(2,4,1)
                p0.set_title("Single Dish Antenna",fontsize = 15)
                p0.scatter(0,0)
                p0.set_xlim(antenna_lim_min ,antenna_lim_max)
                p0.set_ylim(antenna_lim_min ,antenna_lim_max)
                p0.set_xlabel("x (m)")
                p0.set_ylabel("y (m)")
                p0.set_aspect('equal')                
                
                pp0 = plt.subplot(2,4,5)
                pp0.set_title("ALMA from source (perspective)",fontsize = 15)
                pp0.text(4.5,-0.3,r"Powered by: Friendly VRI C.R. Purcell R. Truelove",ha='center', va='center',fontsize =8, transform=pp0.transAxes)
                
                hrangle_rad = np.radians(hourangle*15)
                dec_rad = np.radians(DEC)
                xx_earth_cen = -yy_antpos*np.sin(np.radians(-23.023))
                yy_earth_cen = xx_antpos
                zz_earth_cen = yy_antpos*np.cos(np.radians(-23.023))
            
                xx_antpos_proj = -(xx_earth_cen*np.sin(hrangle_rad)+yy_earth_cen*np.cos(hrangle_rad))
                yy_antpos_proj = -xx_earth_cen*np.sin(dec_rad)*np.cos(hrangle_rad) + yy_earth_cen*np.sin(dec_rad)*np.sin(hrangle_rad)+zz_earth_cen*np.cos(dec_rad)

                if hourangle > 0:
                    pp0.scatter(yy_antpos_proj,xx_antpos_proj)
                elif hourangle < 0:
                    pp0.scatter(-yy_antpos_proj,-xx_antpos_proj)
                elif hourangle == 0:
                    pp0.scatter(xx_antpos_proj,-yy_antpos_proj)
                                    
                pp0.set_xlim(antenna_lim_min ,antenna_lim_max)
                pp0.set_ylim(antenna_lim_min ,antenna_lim_max)
                pp0.set_aspect('equal')
                pp0.set_xlabel("x (m)")
                pp0.set_ylabel("y (m)")

                #plot original image
                p1 = plt.subplot(2,4,2)
                if imagefile == "/home/kiosk/alma_sid_07_05_2025/Alma_main/models/mistery_med.png":
                    qmarkimg=mpimg.imread('/home/kiosk/alma_sid_07_05_2025/Alma_main/models/mistery_qmark.png')                
                    p1.imshow(qmarkimg,cmap = colormap)
                else:
                    p1.imshow(np.real(obsMan.modelImgArr),origin = 'lower',cmap = colormap)
                p1.axes.get_xaxis().set_visible(False)
                p1.axes.get_yaxis().set_visible(False)
                p1.set_title("Picture of the source", fontsize = 15)


                # print("SINGLEDISH")
                p2 = plt.subplot(2,4,3)
                sigma = 50
                x = np.linspace(-250, 250, 100)
                y = np.linspace(-250, 250, 100)
                x_grid, y_grid = np.meshgrid(x, y)
                single_beam = np.exp(-((x_grid)**2 + (y_grid)**2)/(2*sigma**2))
                p2.imshow(single_beam,origin = 'lower',cmap = colormap)
                p2.text(-0.1,0.5,r"$\otimes$",ha='center', va='center',fontsize = 25, transform=p2.transAxes)
                p2.axes.get_xaxis().set_visible(False)
                p2.axes.get_yaxis().set_visible(False)
                p2.set_title("Beam of single dish",fontsize = 15)
            

                #plot different final image, no beam
                # with open(imagefile.split(".")[0]+"_SDOUT.pickle",'rb') as fin:
                #     simage_sd = pickle.load(fin) 
                sgdish_image = gaussian_filter(obsMan.modelImgArr,sigma,mode = 'constant')
                p3 = plt.subplot(2,4,4)
                p3.imshow(sgdish_image,origin = 'lower',cmap = colormap)
                p3.text(-0.1,0.5,r"$=$",ha='center', va='center',fontsize = 25, transform=p3.transAxes)
                p3.axes.get_xaxis().set_visible(False)
                p3.axes.get_yaxis().set_visible(False)
                p3.set_title("Single Dish View",fontsize = 15)
                


                #fft of original image
                
                p4 = plt.subplot(2,4,6)
                # mm,ll = np.shape(obsMan.modelFFTarr)
                #p4.imshow(np.log10(abs(obsMan.modelFFTarr)),cmap = 'gist_heat',interpolation = 'bicubic')
                # p4.imshow(np.log10(abs(obsMan.modelFFTarr)),cmap = 'gist_heat',interpolation = 'bicubic')
                p4.imshow(obsMan.modelImgArr,cmap = colormap,origin = 'lower')
                # p4.set_xlim(ll/2-ll/ZOOM,ll/2+ll/ZOOM)
                # p4.set_ylim(mm/2-mm/ZOOM,mm/2+mm/ZOOM)
                p4.axes.get_xaxis().set_visible(False)
                p4.axes.get_yaxis().set_visible(False)
                p4.set_title("Picture of the source",fontsize = 15)                
                

                if not singledish:    
                    computetimestart = time.time()
                    #plot uvcoverage
                    p5 = plt.subplot(2,4,7)
                    p5.scatter(obsMan.arrsSelected[0]['uArr_lam'],obsMan.arrsSelected[0]['vArr_lam'],s=1)
                    p5.scatter(-1.*obsMan.arrsSelected[0]['uArr_lam'],-1*obsMan.arrsSelected[0]['vArr_lam'],s=1)
                    thistime = time.time(); print(("TIMING %f  %f uv coverage" %(thistime-starttime,thistime-lasttime))); lasttime= thistime
                    # p5.set_xlim(obsMan.pixScaleFFTX_lam*(-ll/ZOOM),(obsMan.pixScaleFFTX_lam*(+ll/ZOOM)))
                    # p5.set_ylim(obsMan.pixScaleFFTY_lam*(-mm/ZOOM),(obsMan.pixScaleFFTY_lam*(+mm/ZOOM)))
                    p5.set_aspect(p0.get_aspect())
                    p5.text(-0.1,0.5,r"$\otimes$",ha='center', va='center',fontsize = 20, transform=p5.transAxes)
                    p5.axes.get_xaxis().set_visible(False)
                    p5.axes.get_yaxis().set_visible(False)
                    p5.set_title("Beam of the ALMA interferometer",fontsize = 15)                    


                    #fft of final image
                    p6 = plt.subplot(2,4,8)
                    # p6.imshow(np.log10(abs(obsMan.obsFFTarr)+1e3)-3,cmap = colormap,interpolation = 'bicubic',origin = 'lower')
                    p6.imshow(np.real(obsMan.obsImgArr),origin='lower',cmap=colormap)
                    # p6.set_xlim(ll/2-ll/ZOOM,ll/2+ll/ZOOM)
                    # p6.set_ylim(mm/2-mm/ZOOM,mm/2+mm/ZOOM)
                    p6.text(-0.1,0.5,r"$=$",ha='center', va='center',fontsize = 20, transform=p6.transAxes)
                    p6.set_aspect('equal')
                    p6.axes.get_xaxis().set_visible(False)
                    p6.axes.get_yaxis().set_visible(False)
                    p6.set_title("ALMA view",fontsize = 15)                    
                    computetimestart=time.time()
                else:
                    p5 = plt.subplot(2,4,7)
                    sigma = 80
                    x = np.linspace(-250, 250, 100)
                    y = np.linspace(-250, 250, 100)
                    x_grid, y_grid = np.meshgrid(x, y)
                    single_beam = np.exp(-((x_grid)**2 + (y_grid)**2)/(2*sigma**2))
                    p5.imshow(single_beam,origin = 'lower',cmap = colormap)
                    p5.text(-0.1,0.5,r"$\otimes$",ha='center', va='center',fontsize = 25, transform=p2.transAxes)
                    p5.axes.get_xaxis().set_visible(False)
                    p5.axes.get_yaxis().set_visible(False)
                    p5.set_title("Beam of single ALMA antenna",fontsize = 15)
                

                    #plot different final image, no beam
                    # with open(imagefile.split(".")[0]+"_SDOUT.pickle",'rb') as fin:
                    #     simage_sd = pickle.load(fin) 
                    sgdish_image = gaussian_filter(obsMan.modelImgArr,sigma,mode = 'constant')
                    p6 = plt.subplot(2,4,8)
                    p6.imshow(sgdish_image,origin = 'lower',cmap = colormap)
                    p6.text(-0.1,0.5,r"$=$",ha='center', va='center',fontsize = 25, transform=p3.transAxes)
                    p6.axes.get_xaxis().set_visible(False)
                    p6.axes.get_yaxis().set_visible(False)
                    p6.set_title("Single ALMA View",fontsize = 15)
                    

            except Exception as e:
                print(e)
                pass
    
        else:
            print("Not plotting")
            pass 

        if verbose:
            print(ant_pos)   
            print(("Time for one loop: %s" % str((time.time()-starttime))))
        print("Pausing for   " ,LOOP_TIME)
        print(plt.isinteractive())
        plt.draw()  #change here 10-2-2026
        plt.pause(2) #change here 10-2-2026
        
        #break
        # r=input()       





    except KeyboardInterrupt:
        plt.clf()
        pl = plt.subplot(111)
        pl.imshow(np.real(obsMan.modelImgArr),origin='lower',cmap='gist_heat')
        pl.axes.get_xaxis().set_visible(False)
        pl.axes.get_yaxis().set_visible(False)
        if False:
            # This is for getting screenshots for debugging. However in production
            # we do not save any images for reasons of privacy in case the images
            # were done with the webcam.
            plt.savefig(imagefile.split(".")[0]+"_almaview_saved"+"."+ imagefile.split(".")[1])

        break
# cam.release()
plt.close()
# ser.close()

# %%
