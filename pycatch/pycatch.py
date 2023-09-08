"""
pycatch

main file

@author: S.G. Heinemann
"""
import os,sys
import numpy as np
import pathlib


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolor
from matplotlib.backend_bases import MouseButton

import astropy.units as u

import sunpy
import sunpy.map
import sunpy.util.net
from sunpy.net import Fido, attrs as a

from utils import calibration as cal
 

class pycatch:
    
    def __init__(self, **kwargs):
        
        self.dir                = kwargs['dir'] if 'magnetogram_file' in kwargs else pathlib.Path.home()
        self.save_dir           = kwargs['save_dir'] if 'magnetogram_file' in kwargs else pathlib.Path.home()
        self.map_file           = kwargs['map_file'] if 'magnetogram_file' in kwargs else None
        self.magnetogram_file   = kwargs['magnetogram_file'] if 'magnetogram_file' in kwargs else None
        
        self.map                = None
        self.magnetogram        = None
        self.point              = None
        self.curves             = None
        self.threshold          = None
        self.type               = None
        
        
    # Download data using sunpy FIDO
    def download(self, time, email = 'test@gmail.com',instr='AIA', wave=193,jsoc =True, **kwargs): #Fido.search **kwargs
        
        t=sunpy.time.parse_time(time)
        if instr == 'AIA' and jsoc:
            if email == 'test@gmail.com':
                print('> pycatch ## Warning  ##')
                print('> pycatch ## You must have an email address registered with JSOC before you are allowed to make a request. ##')
                return
            else:
                res = Fido.search(a.Time(t-10*u.min,t+10*u.min, near=t),a.jsoc.Series('aia.lev1_euv_12s'),a.Wavelength(wave*u.angstrom),a.jsoc.Notify(email), **kwargs)
                downloaded_files = Fido.fetch(res, path=self.dir) 
                self.map_file = downloaded_files[0]
                self.type = 'SDO' 

        else:
            res = Fido.search(a.Time(t-10*u.min,t+10*u.min, near=t),a.Instrument(instr),a.Wavelength(wave*u.angstrom), **kwargs)
            downloaded_files = Fido.fetch(res, path=self.dir) 
            self.map_file = downloaded_files[0]
            
            if instr == 'AIA':
                self.type = 'SDO' 
            elif instr == 'SECCHI':
                self.type = 'STEREO'
            elif instr == 'EUVI':
                self.type = 'SOHO'
            else:
                pass
        
        return 
    
    
    def download_magnetogram(self,email, cadence=720, **kwargs): #Fido.search **kwargs
        if self.type == 'SDO' and self.map is not None:
            t=sunpy.time.parse_time(self.map.meta['DATE-OBS'])
            if email == 'test@gmail.com':
                print('> pycatch ## WARNING ##')
                print('> pycatch ## You must have an email address registered with JSOC before you are allowed to make a request. ##')
                return
            else:
                res = Fido.search(a.Time(t-30*u.min,t+30*u.min, near=t),a.jsoc.Series(f'hmi.m_{cadence}s'), **kwargs)
                downloaded_files = Fido.fetch(res, path=self.dir) 
                self.magnetogram_file = downloaded_files[0]
        elif self.type == 'SOHO' and self.map is not None:
            print('> pycatch ## SOHO MAGNETOGRAMS NOT YET IMPLEMENTED ##')
        else:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        return
        
    
    # Load data using sunpy FIDO
    def load(self, mag=False):
        if mag:
            self.magnetogram=sunpy.map.Map(self.magnetogram_file)
        else:
            self.map=sunpy.map.Map(self.map_file)
        return
        
        
    # calibrate EUV data
    def calibration(self,**kwargs):
        if self.map is not None:
            if self.type == 'SDO':
                #**kwargs: register= True, normalize = True,deconvolve = False, alc = True, degradation = True, cut_limb = True, wave = 193
                self.map  = cal.calibrate_aia(self.map, **kwargs)
            elif self.type == 'STEREO':
                #**kwargs: register= True, normalize = True,deconvolve = False, alc = True, cut_limb = True
                self.map  = cal.calibrate_stereo(self.map, **kwargs)
            else:
                print(f'> pycatch ## CALIBRATION FOR {self.type} NOT YET IMPLEMENTED ##') 
        else:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')           
        return
        
        
    # select seed point from EUV data
    def select(self,fsize=(10,10)):
        if self.map is not None:            
            fig = plt.figure(figsize=fsize)
            ax = fig.add_subplot(projection=self.map)
            self.map.plot(axes=ax)
            plt.show()
            self.point=plt.ginput(n=1, timeout=120, show_clicks=True, mouse_add=MouseButton.LEFT, mouse_pop=MouseButton.RIGHT, mouse_stop=MouseButton.MIDDLE )
            plt.close()
        else:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##') 
        return
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

