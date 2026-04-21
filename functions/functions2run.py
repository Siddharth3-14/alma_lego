#%%
import serial
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from astropy.io import ascii
import time
from vriCalc import observationManager
from astropy.convolution import Gaussian2DKernel,convolve
from scipy.ndimage import gaussian_filter
import matplotlib.image as mpimg
import pickle
import os
import sys
import pyautogui

import matplotlib.pylab as pylab





print(os.getcwd())
LOOP_TIME = 0.1 #seconds
scale_array =40.0
antenna_lim_min ,antenna_lim_max= -3000,3000
hourangle_start = -2. # hourangle start of obs
hourangle_end = +2.  # hourangle end of obs
FREQ = 3e5 #MHz
DEC = -40 #declination
pixel_scale = 0.05 #arcseconds

buttons_inp = "0100100"
attenas_inp = "0000000000000000000000001000000000001000000000"

imagefile1 = "./image/agb_star.jpg"
imagefile2 = "./image/galaxy_gas.jpg"
imagefile3 = "./image/hltau.jpg"
imagefile4 = "./image/outflow.jpg"



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


############### Functions ###################

antenna_filename = 'sid_ant_latest.csv'

def get_antenna_dict():
    ant_database = pd.read_csv(antenna_filename, sep=";", header='infer')
    ant_bits = np.array(ant_database['bit']) +1
    ant_bits_posx = np.array(ant_database['posx'], dtype=float)
    ant_bits_posy = np.array(ant_database['posy'], dtype=float)

    position_array = (np.array([ant_bits_posx,ant_bits_posy]).T)*scale_array
    ant_dict = dict(zip(ant_bits,position_array))

    return ant_dict

ant_dict = get_antenna_dict()
#%%

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


# def waitforserialchange(ser, ant_dict=ant_dict,IsThereArdruino =True, npadarray=45, verbose=False):
#     """Waits for a change to the serial interface, i.e. any change of a contact.
#        Pauses in case there is no contact on one of the hour angles
#     Parameters:   
#     device (str)
#        device to which the serial interface is connected
       
#     npadarray (int)
#        split position of the 64 input array of the microcontroller into the two arrays   
#        the first npadarray are for the antennas, the remaining for the buttons
       
#     Return:
#     list
#        positions of antennas with closed contacts
       
#     list   
#        positions of buttons with closed contacts

#     """   
#     print('CHECK3')

#     Serial = IsThereArdruino
#     validhaselection = False
#     is_there_antenna = True

#     while not validhaselection:
#         if (~Serial):
#             # buttons = "0110000"
#             buttons = buttons_inp
#             attenas = attenas_inp

#             serialinput = attenas  + buttons
#             print('this is the length of serial input',len(serialinput))
#         else:
#             print('CHECK4')
#             serialinput = str(ser.readline().decode("utf-8").strip())
#             if ser.in_waiting > 0:
#                 print('CHECK4')
#                 # In case there were several triggers for redrawing recorded,
#                 # take the last one
#                 serialbuffer = ser.read(ser.in_waiting).decode("utf-8").strip().split('\n')  
#                 if verbose:
#                     print(f"Inputs recorded in the last loop {len(serialbuffer)}. Taking the last one")
#                 serialinput = serialbuffer[-1]
#         time.sleep(1)

#         linein2  = serialinput[npadarray:]
#         linein   = serialinput[:npadarray]

#         print(len(linein))
#         print(len(linein2))
        
#         count = 0
#         count = 0
#         if int(linein) == 0:
#             print("No antennas")
#             is_there_antenna = False
#             return np.nan, np.nan, np.nan,np.nan,np.nan, np.nan, np.nan, np.nan,is_there_antenna
        
#         bit_pos1  = np.where(np.array([bit == '1' for bit in "0"+linein]) == True)[0]
#         bit_pos2 = np.where(np.array([bit == '1' for bit in "0"+linein2]) == True)[0]
#         buttons_config = np.where(np.array([bit == '1' for bit in "0"+linein2[:2]]) == True)[0]
#         buttons_image = np.where(np.array([bit == '1' for bit in "0"+linein2[3:]]) == True)[0]
#         if verbose:
#             print(serialinput)        
#             print((len(serialinput)))
        
#             print(f"antennas:   {linein}")
#             print(f"buttons:    {linein2}")
        
#             print(f"antenna list:    {bit_pos1}")
#             print(f"button  list:    {bit_pos2}      ({' '.join([key for key in bitdict if bitdict[key] in bit_pos2])})")

#             print(f"buttons_config:    {buttons_config}")
#             print(f"buttons_image:    {buttons_image}")




#         #import ipdb; ipdb.set_trace()
#         ant_pos =  np.array([np.array(ant_dict[bb]) for bb in bit_pos1]) #multiply with a factor, default 13 to scale up the array baselines

#         print(ant_pos)
#         if len(ant_pos)>0 and (bitdict['hourangle_m6'] in bit_pos2) or (bitdict['hourangle_0'] in bit_pos2) or (bitdict['hourangle_p6'] in bit_pos2):

#             xx_antpos, yy_antpos = ant_pos.T
#             if len(ant_pos) > 1: 
#                 # create_config_file(ant_pos)
#                 singledish = False
#             elif len(ant_pos)==1: #one antenna show a singledish image
#                 singledish = True    

#             validhaselection = True
#         else:
#             print('Waiting for valid hourangle selection and at least one antenna ...')

#     # TODO: convert everything into the usage of bit_pos1 and bit_pos2. Rename bit_pos1 to bit_pos11
#     return bit_pos1, bit_pos2, buttons_config,buttons_image,ant_pos, xx_antpos, yy_antpos, singledish,is_there_antenna


def waitforserialchange(ser, ant_dict=ant_dict, IsThereArdruino=True, npadarray=45, verbose=False):
    """
    Reads serial input once and returns current state of antennas and buttons.
    
    Parameters:
        ser: serial interface object (or None if no Arduino)
        ant_dict: dict mapping bit positions to antenna coordinates
        IsThereArdruino: bool, True if Arduino is connected
        npadarray: int, split position between antenna and button bits
        verbose: bool, print debug info
    
    Returns:
        bit_pos1, bit_pos2, buttons_config, buttons_image,
        ant_pos, xx_antpos, yy_antpos, singledish, is_there_antenna
    """
    print('CHECK3')
    is_there_antenna = True

    # ── Read serial input once ────────────────────────────────────────────
    if not IsThereArdruino:
        serialinput = attenas_inp + buttons_inp
    else:
        print('CHECK4')
        serialinput = str(ser.readline().decode("utf-8").strip())
        if ser.in_waiting > 0:
            # Take the latest input if multiple triggers recorded
            serialbuffer = ser.read(ser.in_waiting).decode("utf-8").strip().split('\n')
            if verbose:
                print(f"Inputs recorded in last loop: {len(serialbuffer)}. Taking the last one.")
            serialinput = serialbuffer[-1]

    # ── Split into antenna and button sections ────────────────────────────
    linein  = serialinput[:npadarray]   # antenna bits
    linein2 = serialinput[npadarray:]   # button bits

    if verbose:
        print(f"Serial input: {serialinput}  (len={len(serialinput)})")
        print(f"Antennas: {linein}")
        print(f"Buttons:  {linein2}")

    # ── No antennas: return early ─────────────────────────────────────────
    if int(linein) == 0:
        print("No antennas")
        is_there_antenna = False
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, is_there_antenna

    # ── Parse bit positions ───────────────────────────────────────────────
    bit_pos1       = np.where(np.array([b == '1' for b in "0" + linein]))[0]
    bit_pos2       = np.where(np.array([b == '1' for b in "0" + linein2]))[0]
    buttons_config = np.where(np.array([b == '1' for b in "0" + linein2[:2]]))[0]
    buttons_image  = np.where(np.array([b == '1' for b in "0" + linein2[3:]]))[0]

    if verbose:
        print(f"Antenna list:    {bit_pos1}")
        print(f"Button list:     {bit_pos2}")
        print(f"Buttons config:  {buttons_config}")
        print(f"Buttons image:   {buttons_image}")

    # ── Build antenna positions ───────────────────────────────────────────
    ant_pos = np.array([np.array(ant_dict[bb]) for bb in bit_pos1])
    print(ant_pos)

    # ── Single dish vs interferometer ─────────────────────────────────────
    singledish = len(ant_pos) == 1

    # ── Unpack x/y positions ──────────────────────────────────────────────
    if len(ant_pos) > 0:
        xx_antpos, yy_antpos = ant_pos.T
    else:
        xx_antpos, yy_antpos = np.nan, np.nan

    # ── Warn if no valid hour angle selected but continue anyway ──────────
    has_valid_ha = any([
        bitdict['hourangle_m6'] in bit_pos2,
        bitdict['hourangle_0']  in bit_pos2,
        bitdict['hourangle_p6'] in bit_pos2
    ])
    if not has_valid_ha:
        print("Warning: no valid hour angle selected — using defaults.")

    return bit_pos1, bit_pos2, buttons_config, buttons_image, \
           ant_pos, xx_antpos, yy_antpos, singledish, is_there_antenna

#bitdict_config = {'hourangle_m6':     1,
#           'hourangle_0':      2,
#           'hourangle_p6':     3}


#bitdict_image = {'agb_star': 1,
#           'galaxy_gas': 2,
#           'hltau': 3,
#           'outflow': 4}

# bit_pos1, bit_pos2, buttons_config,buttons_image,ant_pos, xx_antpos, yy_antpos, singledish = waitforserialchange(ser, bitdict, ant_dict, verbose=verbose) 


def select_model_and_hourangle(bit_pos2,buttons_config,buttons_image, verbose=False):
    """Select the images and return the corresponding pixel scale and integration time and hour angle   
    """

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

    if verbose:
        print(imagefile, integration_time, hourangle_start, hourangle_end)

    return imagefile, pixel_scale, integration_time, hourangle, hourangle_start, hourangle_end




def write_alma_config_file(antenna_coords):
    filename = 'arrays/lego_alma.config'
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

