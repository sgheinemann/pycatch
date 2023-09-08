"""
pycatch

main file

@author: S.G. Heinemann
"""
import os,sys
import numpy as np
import pathlib
import copy

import astropy.units as u

import sunpy
import sunpy.map
import sunpy.util.net
from sunpy.net import Fido, attrs as a

import pycatch.utils.calibration as cal
import pycatch.utils.extensions as ext
import pycatch.utils.plot as poptions
import pycatch.utils.ch_mapping as mapping

class pycatch:
    
    def __init__(self, **kwargs):
        
        self.dir                = kwargs['dir'] if 'dir' in kwargs else str(pathlib.Path.home())
        self.save_dir           = kwargs['save_dir'] if 'save_dir' in kwargs else pathlib.Path.home()
        self.map_file           = kwargs['map_file'] if 'map_file' in kwargs else None
        self.magnetogram_file   = kwargs['magnetogram_file'] if 'magnetogram_file' in kwargs else None
        
        self.map                = None
        self.original_map       = None
        self.magnetogram        = None
        self.point              = None
        self.curves             = None
        self.threshold          = None
        self.type               = None
        self.rebin_status       = None
        self.cutout_status      = None
        self.kernel             = None
        self.binmaps            = None
        self.properties         = {}
        
    # Download data using sunpy FIDO
    def download(self, time,instr='AIA', wave=193,jsoc =True, **kwargs): #Fido.search **kwargs
        ''' Download function for EUV data '''
        t=sunpy.time.parse_time(time)
        # jsoc not working !!
        # email = 'test@gmail.com',
        # if instr == 'AIA' and jsoc:
        #     if email == 'test@gmail.com':
        #         print('> pycatch ## WARNING ##')
        #         print('> pycatch ## You must have an email address registered with JSOC before you are allowed to make a request. ##')
        #         return
        #     else:
        #         res = Fido.search(a.Time(t-10*u.min,t+10*u.min),a.jsoc.Series('aia.lev1_euv_12s'),a.Wavelength(wave*u.angstrom),a.jsoc.Notify(email), **kwargs)
        #         tv=sunpy.time.parse_time(np.array([res.show('T_REC')[0][i][0] for i in range(res.file_num)]))
        #         indx=(np.abs(tv-t)).argmin()
        #         downloaded_files = Fido.fetch(res[:,indx], path=self.dir + '/{instrument}/{file}') 
        #         self.map_file = downloaded_files[0]
        #         self.type = 'SDO' 

        # else:
        res = Fido.search(a.Time(t-10*u.min,t+10*u.min, near=t),a.Instrument(instr),a.Wavelength(wave*u.angstrom), **kwargs)
        downloaded_files = Fido.fetch(res, path=self.dir + '/{instrument}/{file}' ) 
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
    
    
    def download_magnetogram(self, cadence=45, **kwargs): #Fido.search **kwargs
        if self.type == 'SDO' and self.map is not None:
            t=sunpy.time.parse_time(self.map.meta['DATE-OBS'])
             
             
            # jsoc not working !!
            # if email == 'test@gmail.com':
            #     print('> pycatch ## WARNING ##')
            #     print('> pycatch ## You must have an email address registered with JSOC before you are allowed to make a request. ##')
            #     return
            # else:
            #     res = Fido.search(a.Time(t-30*u.min,t+30*u.min, near=t),a.jsoc.Series(f'hmi.m_{cadence}s'), **kwargs)
            #     tv=sunpy.time.parse_time(np.array([res.show('T_REC')[0][i][0] for i in range(res.file_num)]))
            #     indx=(np.abs(tv-t)).argmin()
            #     downloaded_files = Fido.fetch(res[:,indx], path=self.dir + '/{instrument}/{file}') 
            #     self.magnetogram_file = downloaded_files[0]
            print('> pycatch ## WARNING ##')
            print('> pycatch ## Download of HMI 45s magnetograms only ##')            
            print('> pycatch ## Manually download 720s magnetograms from JSOC ##')   
            if cadence == 45:
                res = Fido.search(a.Time(t-10*u.min,t+10*u.min, near=t),a.Instrument('HMI'),a.Physobs("LOS_magnetic_field"), **kwargs)
                downloaded_files = Fido.fetch(res, path=self.dir + '/{instrument}/{file}') 
                self.magnetogram_file = downloaded_files[0]
            
        elif self.type == 'SOHO' and self.map is not None:
            print('> pycatch ## DOWNLOAD OF SOHO MAGNETOGRAMS NOT YET IMPLEMENTED ##')
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
            self.original_map = copy.deepcopy(self.map)
        return
        
        
    # calibrate EUV data
    def calibration(self,**kwargs):
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return

        if self.type == 'SDO':
            #**kwargs: register= True, normalize = True,deconvolve = False, alc = True, degradation = True, cut_limb = True, wave = 193
            self.map  = cal.calibrate_aia(self.map, **kwargs)
        elif self.type == 'STEREO':
            #**kwargs: register= True, normalize = True,deconvolve = False, alc = True, cut_limb = True
            self.map  = cal.calibrate_stereo(self.map, **kwargs)
        else:
            print(f'> pycatch ## CALIBRATION FOR {self.type} NOT YET IMPLEMENTED ##')      
        return

    # calibrate EUV data
    def calibration_mag(self,**kwargs):
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        
        self.magnetogram = cal.calibrate_hmi(self.magnetogram,self.map, **kwargs)   
        return
    
    # make submap
    def cutout(self,top=[1100,1100], bot=[-1100,-1100]):
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
          
        self.map=ext.cutout(self.map,top,bot)
        self.point, self.curves,  self.threshold = None, None, None
        self.cutout_status      = True
        
        if self.magnetogram is not None:
            self.magnetogram=ext.cutout(self.magnetogram,top,bot)
        return

    # rebin map
    def rebin(self,ndim=[1024,1024]):
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
          
        new_dimensions = ndim * u.pixel
        self.map = self.map.resample(new_dimensions)
        #self.map.data[:]=ext.congrid(self.map.data,(ndim[0],ndim[1]))  #### TEST CONGRID VS RESAMPLE
        self.rebin_status = True
        self.point, self.curves,  self.threshold = None, None, None
        
        if self.magnetogram is not None:
            self.magnetogram = self.magnetogram.resample(new_dimensions)
            #self.magnetogram.data[:]=ext.congrid(self.magnetogram.data,(ndim[0],ndim[1]))  #### TEST CONGRID VS RESAMPLE
        return        
        
    # select seed point from EUV data
    def select(self,fsize=(10,10)):
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        
        self.point=poptions.get_point_from_map(self.map, fsize)
        
        return
            
    # set threshold
    def set_threshold(self,threshold, median = True, no_percentage = False):
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        
        if median:
            if threshold > 2 and no_percentage:
                print('> pycatch ## WARNING ##')
                print(f'> pycatch ## Threshold set to {threshold} * median solar disk intensity ##')
                print(f'> pycatch ## Assuming input to be {threshold} % of the median solar disk intensity instead ##')
                threshold /= 100.
            threshold = ext.median_disk(self.map) * threshold
            
        self.threshold=threshold
        return
                        
            
    # pick threshold from histogram
    def threshold_from_hist(self,fsize=(10,10)):
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
    
        self.threshold=poptions.get_thr_from_hist(self.map,fsize)
        return            
            
    # pick threshold from area curves
    def threshold_from_curves(self,fsize=(10,10)):
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        
        if self.curves is None:
            print('> pycatch ## NO AREA CURVES CALCULATED ##')
            return
        
        self.threshold=poptions.get_thr_from_curves(self.map,self.curves,fsize)
        return                 

    # calculate area curves
    def calculate_curves(self,fsize=(10,10),verbose=True):
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
            
        if verbose:
            print('> pycatch ## WARNING ##')
            print('> pycatch ## Operation may take a bit ! ##')
            print('> pycatch ## You may disable this message by using the keyword verbose = False ##')
        
        xloc, area, uncertainty =mapping.get_curves(self.map,self.seed,kernel=self.kernel)
        self.curves = [xloc, area, uncertainty]
        return              
            
    # calculate binmap
    def extract_ch(self):
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        if self.threshold is None:
            print('> pycatch ## NO INTENSITY THRESHOLD SET ##')
            return            

        self.binmaps = [mapping.extract_ch(self.map, self.threshold, self.kernel,self.seed) for i in np.arange(5)-2]
        return               
            
    # calculate binmap
    def calculate_properties(self):
        if self.binmapsmap is None:
            print('> pycatch ## NO COORNAL HOLES EXTRACTED ##')
            return
        
        self.binmaps = [mapping.extract_ch(self.map, self.threshold, self.kernel,self.seed) for i in np.arange(5)-2]
        binmap, a, da, com, dcom, ex, dex = mapping.catch_calc(self.binmaps, binary =True)     
        imean,dimean,imed,dimed = mapping.get_intensity(self.binmaps, binary =True)  
        
        dict1={'A':a,'dA':da,'Imean':imean,'dImean':dimean,'Imed':imed,'dImed':dimed,'CoM':com,'dCoM':dcom,'ex':ex,'dex':dex}
        self.properties.update(dict1)
        return                 
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

