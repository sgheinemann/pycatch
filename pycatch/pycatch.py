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
from _version import __version__

class pycatch:
    """
    A Python library for extracting and analyzing coronal holes from solar EUV images and magnetograms.

    Attributes
    ----------
        dir : str
            The directory path. Defaults to the user's home directory if not provided.
        save_dir : str
            The save directory path. Defaults to the user's home directory if not provided.
        map_file : str
            The map file path.
        magnetogram_file : str
            The magnetogram file path.
        map : sunpy.map.Map
            The loaded and configured EUV map.
            This map contains both the 2D data array and metadata associated with the EUV observation.
        original_map : sunpy.map.Map
            The original EUV map.
            This map contains the initially loaded EUV observation before any operations.
        magnetogram : sunpy.map.Map
            The loaded and configured magnetogram.
            This map contains both the 2D data array and metadata associated with the magnetic field data.
        point : list of float
            Seed point for coronal hole extraction.
        curves : tuple
            - The threshold range.
            - The calculated area curves for the coronal hole.
            - The uncertainty in the area curves.
        threshold : float
            Coronal hole extraction threshold.
        type : str
            Placeholder for type information.
        rebin_status : bool
            Whether the map was rebinned.
        cutout_status : bool
            Whether the map was cutout.
        kernel : int
            Size of the circular kernel for morphological operations. 
        binmap : sunpy.map.Map
            Single 5-level binary map with the coronal hole extraction, where each level represents a different threshold value.
        properties : dict
            A dictionary containing the calculate coronal hole properties.
        __version__ : str
            Version number of pyCATCH
        
    Parameters
    ----------
    dir : str, optional
        Directory for storing and loading data (default is the home directory).
    save_dir : str, optional
        Directory for storing data (default is the home directory).
    map_file : str, optional
        Filepath to EUV/Intensity map, needs to be loadable with sunpy.map.Map() (default: None).
    magnetogram_file : str, optional
        Filepath to magnetogram, needs to be loadable with sunpy.map.Map() (default: None).
    load : str, optional
        Loads a previously saved pyCATCH object from the specified path, which overrides any other keywords (default: None).
    
    Returns 
    -------
    None
    """
    __version__ = __version__
    
    def __init__(self, load=None, **kwargs):
        """
        Initialize a pyCATCH object.
        
        Parameters
        ----------
            dir : str, optional
                Directory for storing and loading data (default is the home directory).
            save_dir : str, optional
                Directory for storing data (default is the home directory).
            map_file : str, optional
                Filepath to EUV/Intensity map, needs to be loadable with sunpy.map.Map() (default: None).
            magnetogram_file : str, optional
                Filepath to magnetogram, needs to be loadable with sunpy.map.Map() (default: None).
            load : str, optional
                Loads a previously saved pyCATCH object from the specified path, which overrides any other keywords (default: None).
        
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
        Save a pyCATCH object to a pickle file.
        
        Parameters
        ----------
            file : str, optional
                Filepath to save the object, default is pycatch.dir (default: False).
            overwrite : bool, optional
                Flag to overwrite the file if it already exists (default: False).
            no_original : bool, optional
                Flag to exclude saving the original map to save disk space (default: True).
        
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
        Download an EUV map using VSO (Virtual Solar Observatory).
        
        Parameters
        ----------
            instrument : str, optional
                The instrument of the EUV image to download (default: 'AIA').
            wavelength : int, optional
                The wavelength of the EUV image to download (default: 193).
           **kwargs** : 
                Additional keyword arguments passed to sunpy.Fido.search (see sunpy documentation for more information).
        
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
        Download a magnetogram matching the EUV image date using VSO (Virtual Solar Observatory).
        
        Parameters
        ----------
            cadence : int, optional
                Download Line-of-Sight (LOS) magnetogram with the specified cadence in seconds (default: 45).
           **kwargs**: 
                Additional keyword arguments passed to sunpy.Fido.search (see sunpy documentation for more information).
        
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
        
        Parameters
        ----------
            mag : bool, optional
                Flag to load a magnetogram (default: False).
            file : str, optional
                Filepath to load a specific map. If not set, it loads pycatch.map_file or pycatch.magnetogram_file (default: None).
        
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
        
        Parameters
        ----------
           **kwargs**(SDO/AIA): 
                deconvolve : bool or numpy.ndarray, optional
                    Use PSF deconvolution (default: None, takes a custom PSF array as input, if True uses aiapy.psf.deconvolve).
                register : bool, optional
                    Co-register the map (default: True).
                normalize : bool, optional
                    Normalize intensity to 1s (default: True).
                degradation : bool, optional
                    Correct instrument degradation (default: True).
                alc : bool, optional
                    Apply Annulus Limb Correction, Python implementation from Verbeek et al. (2014) (default: True).
                cut_limb : bool, optional
                    Set off-limb pixel values to NaN (default: True).
            
           **kwargs**(STEREO/SECCHI):
                deconvolve : bool, optional
                    NOT YET IMPLEMENTED FOR STEREO.
                register : bool, optional
                    Co-register the map (default: True).
                normalize : bool, optional
                    Normalize intensity to 1s (default: True).
                alc : bool, optional
                    Apply Annulus Limb Correction, Python implementation from Verbeek et al. (2014) (default: True).
                cut_limb : bool, optional
                    Set off-limb pixel values to NaN (default: True).
        
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
        
        Parameters
        ----------
           **kwargs**(SDO/HMI): 
                rotate : bool, optional
                    Rotate the map so that North is up (default: True).
                align : bool, optional
                    Align with an AIA map (default: True).
                cut_limb : bool, optional
                    Set off-limb pixel values to NaN (default: True).

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
    def cutout(self,top=(1100,1100), bot=(-1100,-1100)):
        """
        Cut a subfield of the map (if a magnetogram is loaded, it will also be cut).
        
        Parameters
        ----------
            top : tuple, optional
                Coordinates of the top-right corner (default: (1100, 1100)).
            bot : tuple, optional
                Coordinates of the bottom-left corner (default: (-1100, -1100)).
        
        Returns 
        -------
        None
        """


        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
          
        self.map=mapping.cutout(self.map,top,bot)
        self.point, self.curves,  self.threshold = None, None, None
        self.cutout_status      = True
        
        if self.magnetogram is not None:
            self.magnetogram=ext.cutout(self.magnetogram,top,bot)
        return

    # rebin map
    def rebin(self,ndim=(1024,1024),**kwargs):
        """
        Rebin maps to a new resolution (if a magnetogram is loaded, it will also be resampled).
        
        Parameters
        ----------
            ndim : tuple, optional
                New dimensions of the map (default: (1024, 1024)).
           **kwargs**: 
                Additional keyword arguments passed to sunpy.map.Map.resample (see sunpy documentation for more information).
        
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
        Select a seed point from the intensity map.
        
        Parameters
        ----------
            fsize : tuple, optional
                Set the figure size (default: (10, 10)).
        
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
        Set the coronal hole extraction threshold.
        
        Parameters
        ----------
            median : bool, optional
                If True, the input is assumed to be a fraction of the median solar disk intensity (default: True).
            no_percentage : bool, optional
                If True, the input is given as a percentage of the median solar disk intensity (default: False). 
                This only works in conjunction with median=True.
        
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
        Select a threshold for coronal hole extraction from the solar disk intensity histogram.
        
        Parameters
        ----------
            fsize : tuple, optional
                Set the figure size. Default is (10, 5).
        
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
        Select a threshold for coronal hole extraction from calculated area and uncertainty curves as a function of intensity.
        
        Before using this function, you need to calculate the curves using `pycatch.calculate_curves()`.
        
        Parameters
        ----------
            fsize : tuple, optional
                Set the figure size. Default is (10, 5).
        
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
        Calculate area and uncertainty curves as a function of intensity.
        
        Parameters
        ----------
            verbose : bool, optional
                Display warnings. Default is True.
        
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
        This function outputs a binary map to pycatch.binmap.

        Parameters
        ----------
            kernel : int or None, optional
                Size of the circular kernel for morphological operations. Default is None. If None, a kernel size depending on resolution will be used.
        
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
        Calculate the morphological coronal hole properties from the extracted binary map.
        
        Parameters
        ----------
            mag : bool, optional
                Calculate magnetic properties instead. Default is False.
            align : bool, optional
                Call pycatch.calibration_mag() to align with binary map. Default is False.
        
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
        Save properties to a text file.
        
        Parameters
        ----------
            file : str, optional
                Filepath to save the data. Default is pycatch.dir.
            overwrite : bool, optional
                Flag to overwrite the file if it already exists. Default is False.
        
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
        Display a coronal hole plot.
        
        Parameters
        ----------
            boundary : bool, optional
                Overplot the coronal hole boundary. Default is True.
            uncertainty : bool, optional
                Show the uncertainty of the coronal hole boundary. Default is True.
            original : bool, optional
                Show the original image. Default is False.
            cutout : list of tuple, optional
                Display a cutout around the extracted coronal hole. Format: [(xbot, ybot), (xtop, ytop)]. Default is None.
            mag : bool, optional
                Show a magnetogram instead of the coronal hole plot. Default is False.
            fsize : tuple, optional
                Set the figure size. Default is (10, 10).
            save : bool, optional
                Save and close the figure. Default is False.
            sfile : str, optional
                Filepath to save the image (use only in conjunction with save=True). Default is None.
            overwrite : bool, optional
                Overwrite the plot if it already exists. Default is True.
           **kwargs**: keyword arguments
                Additional keyword arguments for sunpy.map.Map.plot(). See the sunpy documentation for more information.
        
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
            
            
            
            
            
            
            
            
            

