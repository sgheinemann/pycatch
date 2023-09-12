"""
pycatch

main file

@author: S.G. Heinemann
"""
import os,sys
import numpy as np
import pathlib
import copy
import pickle

import astropy.units as u

import sunpy
import sunpy.map
import sunpy.util.net
from sunpy.net import Fido, attrs as a

import pycatch.utils.calibration as cal
import pycatch.utils.extensions as ext
import pycatch.utils.plot as poptions
import pycatch.utils.ch_mapping as mapping
import pycatch as catch
import pycatch
from pycatch._version import __version__

class pycatch:
    __version__ = __version__
    
    def __init__(self, load=None, **kwargs):
        """
        Initialize pycatch object.
        --------
        Parameters
        ----------
        **kwargs : 
            dir: directory for storing and loading data (Default is the home directory)
            save_dir: directory for storing data (Default is the home directory)
            map_file: filepath to EUV/Intensity map, needs to be loadable with sunpy.map.map() (Default: None)
            magnetogram_file: filepath to magnetogram, needs to be loadable with sunpy.map.map() (Default: None)
            load: loads previously saved pycatch object from path, overrides any other keywords (Default: None)
            
        Returns 
        -------
        None

        """
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
        self.binmap             = None
        self.properties         =  {'A':None,'dA':None,'Imean':None,'dImean':None,'Imed':None,'dImed':None,'CoM':None,'dCoM':None,'ex':None,'dex':None,
                                    'Bs':None,'dBs':None,'Bus':None,'dBus':None,'Fs':None,'dFs':None,'Fus':None,'dFus':None,'FB':None,'dFB':None }
        self.names              =  ext.init_props()
        
        if load is not None:
            
            try:
                with open(load, "rb") as f:
                    data=pickle.load(f)
                
               # save_dict=
                for key,value in data.items():
                    setattr(self,key, value)
                
                print('> pycatch ## OBJECT SUCESSFULLY LOADED  ##')
                
            except Exception as ex:
                print("> pycatch ## Error during unpickling object (Possibly unsupported):", ex)
                print("> pycatch ## NO DATA LOADAD ##")
                return
 
        
    # save pycatch in pickle file
    def save(self, file=None, overwrite = False, no_original=True):
        """
        Save pycatch object in a pickle file.
        --------
        Parameters
        ----------
        **kwargs : 
            file: filepath to save, default is pycatch.dir (Default: False)
            overwrite: overwrites file (Default: False)
            no_original: does not save the original map to save disk space (Default: True)
            
        Returns 
        -------
        None

        """
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            print('> pycatch ## OBJECT NOT SAVED ##')
            return
        try:
            if no_original:
                dummy=copy.deepcopy(self.original_map)
                self.original_map=None
            if file is not None:
                fpath = file
            else:
                datestr=sunpy.time.parse_time(self.map.meta['DATE-OBS']).strftime('%Y%m%dT%H%M%S')
                typestr=self.map.meta['telescop'].replace('/','_')
                nr=0
                fpath=self.dir+'pyCATCH_'+typestr+'_'+datestr+f'_{nr}'+'.pkl'
                if not overwrite:
                    while os.path.isfile(fpath):
                        nr+=1
                        fpath=self.dir+'pyCATCH_'+typestr+'_'+datestr+f'_{nr}'+'.pkl'

                save_dict={}
                for key,value in self.__dict__.items():
                    save_dict.update({key:value}) 
                    
            with open(fpath, "wb") as f:
                pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'> pycatch ## OBJECT SAVED: {fpath}  ##')
            
            if no_original:
                self.original_map=dummy
                
        except Exception as ex:
            print("> pycatch ## Error during pickling object (Possibly unsupported):", ex)
        return
        
    # Download data using sunpy FIDO
    def download(self, time,instr='AIA', wave=193, **kwargs): #Fido.search **kwargs
        """
        Download EUV map using VSO.
        --------
        Parameters
        ----------
        **kwargs : 
            wave: set instrument (Default: 'AIA')
            wave: set wavelength of EUV image (Default: 193)
            takes all sunpy.Fido.search **kwargs (see sunpy documentation for more informations)
            
        Returns 
        -------
        None

        """
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
        
        return 
    
    
    def download_magnetogram(self, cadence=45, **kwargs): #Fido.search **kwargs
        """
        Download magnetogram matching the EUV date using VSO.
        --------
        Parameters
        ----------
        **kwargs : 
            cadence: download LOS magnetogram with _<cadence>s (Default: 45)
            takes all sunpy.Fido.search **kwargs (see sunpy documentation for more informations)
            
        Returns 
        -------
        None

        """
        
        if 'SDO' in self.type and self.map is not None:
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
            
        elif 'SOHO' in self.type and self.map is not None:
            print('> pycatch ## DOWNLOAD OF SOHO MAGNETOGRAMS NOT YET IMPLEMENTED ##')
        else:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        return
        
    
    # Load data using sunpy FIDO
    def load(self, mag=False, file = None):
        """
        Load maps.
        --------
        Parameters
        ----------
        **kwargs : 
            mag: load magnetogram (Default: False)
            file: filepath to load, if not set loads pycatch.map_file or pycatch.magnetogram_file (Default: False)
            
        Returns 
        -------
        None

        """
        if mag:
            if file is not None:
                self.magnetogram_file = file
            self.magnetogram=sunpy.map.Map(self.magnetogram_file)
        else:
            if file is not None:
                self.map_file = file
            self.map=sunpy.map.Map(self.map_file)
            self.original_map = copy.deepcopy(self.map)
            self.type=self.map.meta['telescop']
        return
        
        
    # calibrate EUV data
    def calibration(self,**kwargs):
        """
        Calibrate the intensity image.
        --------
        Parameters
        ----------
        **kwargs : 
            SDO/AIA:
            deconvolve: use PSF deconvolution (Default: False, takes custom PSF as input, Default: standard AIA-PSF)   
            register: co-registering the map (Default: True)
            normalize: Normalize intensity to 1s (Default: True)
            degradation: correct instrument degredation (Default: True)
            alc: Annulus Limb Correction, Python implementation from Verbeek et al. (2014): (Default: True)
            cut_limb: Set off-limb pixel to nan (Default: True)
            
            STEREO/SECCHI:
            deconvolve: NOT YET IMPLEMENTED FOR STEREO    
            register: co-registering the map (Default: True)
            normalize: Normalize intensity to 1s (Default: True)
            alc: Annulus Limb Correction, Python implementation from Verbeek et al. (2014): (Default: True)
            cut_limb: Set off-limb pixel to nan (Default: True)    
            
            SOHO/EUVI:
            NOT YET IMPLEMENTED
            
        Returns 
        -------
        None

        """
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return

        if 'SDO' in self.type:
            #**kwargs: register= True, normalize = True,deconvolve = False, alc = True, degradation = True, cut_limb = True, wave = 193
            self.map  = cal.calibrate_aia(self.map, **kwargs)
        elif 'STEREO' in self.type:
            #**kwargs: register= True, normalize = True,deconvolve = False, alc = True, cut_limb = True
            self.map  = cal.calibrate_stereo(self.map, **kwargs)
        else:
            print(f'> pycatch ## CALIBRATION FOR {self.type} NOT YET IMPLEMENTED ##')      
        return

    # calibrate EUV data
    def calibration_mag(self,**kwargs):
        """
        Calibrate the magnetogram.
        --------
        Parameters
        ----------
        **kwargs : 
            SDO/AIA:
            rotate: rotate map to North == up (Default: True)
            align: align with aia map (Default: True)
            cut_limb: Set off-limb pixel to nan (Default: True)  
            
            SOHO/EUVI:
            NOT YET IMPLEMENTED
            
        Returns 
        -------
        None

        """
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        
        self.magnetogram = cal.calibrate_hmi(self.magnetogram,self.map, **kwargs)   
        return
    
    # make submap
    def cutout(self,top=[1100,1100], bot=[-1100,-1100]):
        """
        Cut a subfield of the map (if a magnetogram is loaded it will also be cut.)
        --------
        Parameters
        ----------
        **kwargs : 
            top: coordinates of top right corner (Default: [1100,1100])
            bot: coordinates of bottom left corner (Default: [-1100,-1100])
            
        Returns 
        -------
        None

        """
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
    def rebin(self,ndim=[1024,1024],**kwargs):
        """
        Rebin maps to new resolution (if a magnetogram is loaded it will also be resampled.)
        --------
        Parameters
        ----------
        **kwargs : 
            ndim: new dimensions of map (Default: [1024,1024])
            takes all sunpy.map.Map.resample **kwargs (see sunpy documentation for more informations)
            
        Returns 
        -------
        None

        """
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
          
        new_dimensions = ndim * u.pixel
        self.map = self.map.resample(new_dimensions, **kwargs)
        #self.map.data[:]=ext.congrid(self.map.data,(ndim[0],ndim[1]))  #### TEST CONGRID VS RESAMPLE
        self.rebin_status = True
        self.point, self.curves,  self.threshold = None, None, None
        
        if self.magnetogram is not None:
            self.magnetogram = self.magnetogram.resample(new_dimensions, **kwargs)
            #self.magnetogram.data[:]=ext.congrid(self.magnetogram.data,(ndim[0],ndim[1]))  #### TEST CONGRID VS RESAMPLE
        return        
        
    # select seed point from EUV data
    def select(self,fsize=(10,10)):
        """
        Select seed point from intensity map.
        --------
        Parameters
        ----------
        **kwargs : 
            fsize: set figure size (Default: (10,10))
            
        Returns 
        -------
        None

        """
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        
        self.point=poptions.get_point_from_map(self.map, fsize)
        
        return
            
    # set threshold
    def set_threshold(self,threshold, median = True, no_percentage = False):
        """
        Set coronal hole extraction threshold
        --------
        Parameters
        ----------
        **kwargs : 
            median: input is assumed to be as fraction of the median solar disk intensity (default: True)
            no_percentage: input is given percent of the median solar disk intensity (default: False), only works in conjunction with median = True
            
        Returns 
        -------
        None

        """
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
    def threshold_from_hist(self,fsize=(10,5)):
        """
        Select coronal hole extraction threshold from solar disk intensity histogram.
        --------
        Parameters
        ----------
        **kwargs : 
            fsize: set figure size (Default: (10,10))
            
        Returns 
        -------
        None

        """
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
    
        self.threshold=poptions.get_thr_from_hist(self.map,fsize)
        return            
            
    # pick threshold from area curves
    def threshold_from_curves(self,fsize=(10,5)):
        """
        Select coronal hole extraction threshold from calculated area and uncertainty curves as function of intensity.
        Curves need to be calculated first using pycatch.calculate_curves()
        --------
        Parameters
        ----------
        **kwargs : 
            fsize: set figure size (Default: (10,10))
            
        Returns 
        -------
        None

        """
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        
        if self.curves is None:
            print('> pycatch ## NO AREA CURVES CALCULATED ##')
            return
        
        self.threshold=poptions.get_thr_from_curves(self.map,self.curves,fsize)
        return                 

    # calculate area curves
    def calculate_curves(self,verbose=True):
        """
        Calculate area and uncertainty curves as function of intensity.
        --------
        Parameters
        ----------
        **kwargs : 
            verbose: display warnings (Default: True)
            
        Returns 
        -------
        None

        """
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
            
        if verbose:
            if self.map.meta['cdelt1'] < 2:
                print('> pycatch ## WARNING ##')
                print('> pycatch ## Operation may take a bit ! ##')
                print('> pycatch ## You may disable this message by using the keyword verbose = False ##')
            
        xloc, area, uncertainty =mapping.get_curves(self.map,self.point,kernel=self.kernel)
        self.curves = [xloc, area, uncertainty/area]
        return              
            
    # calculate binmap
    def extract_ch(self, kernel=None):
        """
        Extract the coronal hole from the intensity map using the selected threshold and seed point.
        Outputs list of five binary maps to pycatch.binmaps that are used for calculating the uncertainties.
        --------
        Parameters
        ----------
        **kwargs : 
            kernel: size of circular kernel for morphological operations (Default: None == depending on resolution)        
            
        Returns 
        -------
        None

        """
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        if self.threshold is None:
            print('> pycatch ## NO INTENSITY THRESHOLD SET ##')
            return            
        if self.point is None:
            print('> pycatch ## NO SEED POINT SELECTED ##')
            return            
        
        if kernel is not None:
            self.kernel = kernel
        
        binmaps=[mapping.extract_ch(self.map, self.threshold+i, self.kernel,self.point) for i in np.arange(5)-2]
        
        self.binmap =mapping.to_5binmap(binmaps)
        return               
            
    # calculate morphological properties
    def calculate_properties(self, mag=False, align=False):
        """
        Calculate the morphological coronal hole properties from the extracted binary maps.
        --------
        Parameters
        ----------
            **kwargs : 
                mag: calculate magnetic properties INSTEAD (Default: False)    
                align: calles calibration_mag to align with binary map (Default: False)
                
        Returns 
        -------
        None

        """
        if mag:
            if self.binmap is None:
                print('> pycatch ## NO CORNAL HOLES EXTRACTED ##')
                return
            
            if self.magnetogram is None:
                print('> pycatch ## NO MAGNETOGRAM LOADED ##')
                return
            
            if align:
                self.magnetogram = cal.calibrate_hmi(self.magnetogram,self.binmap)  
                
            if self.binmap.shape != self.magnetogram.shape:
                print('> pycatch ## BINMAP AND MAGNETOGRAM ARE NOT MATCHING ##')
                return
            
            binmaps=mapping.from_5binmap(self.binmap)
            
            bs,dbs,bus,dbus,fs,dfs,fs,dfs,fb,dfb = mapping.catch_mag(binmaps, self.magnetogram)  
            
            dict1={'Bs':bs,'dBs':dbs,'Bus':bus,'dBus':dbus,'Fs':fs,'dFs':dfs,'Fus':fs,'dFus':dfs,'FB':fb,'dFB':dfb}
            self.properties.update(dict1)           

        else:
            if self.binmap is None:
                print('> pycatch ## NO CORNAL HOLES EXTRACTED ##')
                return
            
            binmaps=mapping.from_5binmap(self.binmap)
            binmap, a, da, com, dcom, ex, dex = mapping.catch_calc(binmaps)     
            imean,dimean,imed,dimed = mapping.get_intensity(binmaps, self.map)  
            
            dict1={'A':a,'dA':da,'Imean':imean,'dImean':dimean,'Imed':imed,'dImed':dimed,'CoM':com,'dCoM':dcom,'ex':ex,'dex':dex}
            self.properties.update(dict1)
        return                 
            
           
            
    # save properties to txt file
    def print_properties(self,file=None, overwrite=False):
        """
        Save properties to txt file.
        --------
        Parameters
        ----------
        **kwargs : 
            file: filepath to save, default is pycatch.dir (Default: False)
            overwrite: overwrites file (Default: False)
            
        Returns 
        -------
        None

        """
        if self.properties['A'] is None:
            print('> pycatch ## WARNING ##')
            print('> pycatch ## NO MORPHOLOGICAL PROPERTIES CALCULATED ##')
            print('> pycatch ## PROPERTIES NOT SAVED ##')
            return
        
        if self.properties['Bs'] is None:
            print('> pycatch ## WARNING ##')
            print('> pycatch ## NO MAGNETIC PROPERTIES CALCULATED ##')
        
        
        try:
            if file is not None:
                fpath = file
            
                datestr=sunpy.time.parse_time(self.map.meta['DATE-OBS']).strftime('%Y%m%dT%H%M%S')
                typestr=self.map.meta['telescop'].replace('/','_')
                nr=0
                fpath=self.dir+'pyCATCH_properties_'+typestr+'_'+datestr+f'_{nr}'+'.txt'
                if not overwrite:
                    while os.path.isfile(fpath):
                        nr+=1
                        fpath=self.dir+'pyCATCH_properties_'+typestr+'_'+datestr+f'_{nr}'+'.pkl'
            
                ext.printtxt(fpath, self.properties,self.names, pycatch.__version__)
                    
                print(f'> pycatch ## PROPERTIES SAVED: {fpath}  ##')
                

        except Exception as ex:
                print("> pycatch ## Error during saving file:", ex)
        return           
                
    
    # display coronal hole
    def plot_map(self,boundary=True,uncertainty=True,original=False, cutout=None, mag=False, fsize=(10,10),save=False,sfile=None,overwrite=True,**kwargs):
        """
        Display coronal hole plot.
        --------
        Parameters
        ----------
        **kwargs : 
            boundary: overplot coronal hole boundary (Default: True)
            uncertainty: show uncertainty of coronal hole boundary (Default: True)
            original: show original image (Default: False)
            cutout: display cutout around the extracted coronal hole (Default: None; Format: [[xbot,ybot],[xtop,ytop])
            mag: show magnetogram instead (Default: False)
            fsize: set figure size (Default: (10,10))
            save: save and close figure (Default: False)
            sfile: filepath to save image (otherwise use standard path), only in conjunction with save=True (Default: None)
            overwrite: overwrites plot (Default: True)
            takes all sunpy.map.Map.plot() **kwargs (see sunpy documentation for more informations)
            
        Returns 
        -------
        None

        """
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        if self.magnetogram is None and mag == True:
            print('> pycatch ## NO MAGNETOGRAM LOADED ##')
            return
        if (boundary == True or uncertainty == True) and self.binmap is None:
            print('> pycatch ## NO CORONAL HOLE BOUNDARY EXTRACTED ##')
            return

        if original == True and self.original_map is None:
            print('> pycatch ## NO ORIGINAL MAP LOADED ##')
            return
        
        addstr=''
        if original:
            pmap=self.original_map
            addstr+='_original'
        elif mag:
            pmap=self.magnetogram
            addstr+='_mag'
        else:
            pmap=self.map
            
        if cutout is not None:
            pmap=ext.cutout(pmap,cutout[1],cutout[0])
            pbinmap=ext.cutout(self.binmap,cutout[1],cutout[0])
            addstr+='_cut'
        else:
            pbinmap=self.binmap
            
        if boundary:
            addstr+='_boundary'
        if uncertainty:
            addstr+='_uncertainty'

        datestr=sunpy.time.parse_time(self.map.meta['DATE-OBS']).strftime('%Y%m%dT%H%M%S')
        typestr=self.map.meta['telescop'].replace('/','_')
        nr=0
        fpath=self.dir+'pyCATCH_plot_'+typestr+'_'+datestr+addstr+f'_{nr}'+'.pdf'
        if not overwrite:
            while os.path.isfile(fpath):
                nr+=1
                fpath=self.dir+'pyCATCH_properties_'+typestr+'_'+datestr+f'_{nr}'+'.pdf'
                                
                                
        poptions.plot_map(pmap,pbinmap,boundary,uncertainty, fsize, save, fpath,**kwargs)
            
        
        return                     
            
            
            
            
            
            
            
            
            

