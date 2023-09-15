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
        Directory for storing and loading data. Default is the home directory.
    save_dir : str, optional
        Directory for storing data. Default is the home directory.
    map_file : str, optional
        Filepath to EUV/Intensity map, needs to be loadable with sunpy.map.Map(). Default is None.
    magnetogram_file : str, optional
        Filepath to magnetogram, needs to be loadable with sunpy.map.Map(). Default is None.
    load : str, optional
        Loads a previously saved pyCATCH object from the specified path, which overrides any other keywords. Default is None.
    
    Returns 
    -------
    None
    """
    
    __version__ = __version__
    
#############################################################################################################################################    

    def __str__(self):
        if self.map is not None:
            datestr=self.map.meta['DATE-OBS']
        else:
            datestr=None
        return f'pyCATCH v{self.__version__} ({self.type},{datestr})'
    
 #############################################################################################################################################      
 
    def __init__(self, restore=None, **kwargs):
        """
        Initialize a pyCATCH object.
        
        Parameters
        ----------
            dir : str, optional
                Directory for storing and loading data. Default is the home directory.
            save_dir : str, optional
                Directory for storing data. Default is the home directory.
            map_file : str, optional
                Filepath to EUV/Intensity map, needs to be loadable with sunpy.map.Map(). Default is None.
            magnetogram_file : str, optional
                Filepath to magnetogram, needs to be loadable with sunpy.map.Map(). Default is None.
            restore : str, optional
                Loads a previously saved pyCATCH object from the specified path, which overrides any other keywords. Default is None.
        
        Returns 
        -------
        None
        """
        
        # Check the type of the 'dir' argument
        dir_path = kwargs.get('dir', str(pathlib.Path.home()))
        if not isinstance(dir_path, str):
            raise TypeError("> pycatch ##  'dir' argument must be type str")

        # Check the type of the 'save_dir' argument
        save_dir_path = kwargs.get('save_dir', str(pathlib.Path.home()))
        if not isinstance(save_dir_path, str):
            raise TypeError("> pycatch ##  'save_dir' argument must be type str")

        # Check the type of the 'map_file' argument
        map_file_path = kwargs.get('map_file', None)
        if map_file_path is not None and not isinstance(map_file_path, str):
            raise TypeError("> pycatch ##  'map_file' argument must be type str or None")

        # Check the type of the 'magnetogram_file' argument
        magnetogram_file_path = kwargs.get('magnetogram_file', None)
        if magnetogram_file_path is not None and not isinstance(magnetogram_file_path, str):
            raise TypeError("> pycatch ##  'magnetogram_file' argument must be type str or None")

        # Check the type of the 'restore' argument
        if restore is not None and not isinstance(restore, str):
            raise TypeError("> pycatch ##  'restore' argument must be type str or None")
            
        self.dir                = dir_path
        self.save_dir           = save_dir_path
        self.map_file           = map_file_path
        self.magnetogram_file   = magnetogram_file_path
        
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
        
        if restore is not None:
            
            try:
                with open(restore, "rb") as f:
                    data=pickle.load(f)
                
               # save_dict=
                for key,value in data.items():
                    setattr(self,key, value)
                
                print('> pycatch ## OBJECT SUCESSFULLY LOADED  ##')
                
            except Exception as ex:
                print("> pycatch ## Error during unpickling object (Possibly unsupported):", ex)
                print("> pycatch ## NO DATA LOADAD ##")
                return
 
#############################################################################################################################################   
        
    # save pycatch in pickle file
    def save(self, file=None, overwrite = False, no_original=True):
        """
        Save a pyCATCH object to a pickle file.
        
        Parameters
        ----------
            file : str, optional
                Filepath to save the object, default is pycatch.dir. Default is None.
            overwrite : bool, optional
                Flag to overwrite the file if it already exists. Default is False.
            no_original : bool, optional
                Flag to exclude saving the original map to save disk space. Default is True.
        
        Returns 
        -------
        None
        """
        # Check the type of the 'file' argument
        if file is not None and not isinstance(file, str):
            print("> pycatch ## 'file' argument must be type str")
            return 
        
        # Check the type of the 'overwrite' argument
        if not isinstance(overwrite, bool):
            print("> pycatch ## 'overwrite' argument must be type bool")
            return 
        
        # Check the type of the 'no_original' argument
        if not isinstance(no_original, bool):
            print("> pycatch ## 'no_original' argument must be type bool")
            return 

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

#############################################################################################################################################   
        
    # Download data using sunpy FIDO
    def download(self, time,instr='AIA', wave=193,source='SDO', **kwargs): #Fido.search **kwargs
        """
        Download an EUV map using VSO (Virtual Solar Observatory). It downloads the closest image within +/- 1 hour of the time provided.
        
        Parameters
        ----------
            time : tuple, list, str, pandas.Timestamp, pandas.Series, pandas.DatetimeIndex, datetime.datetime, datetime.date, numpy.datetime64, numpy.ndarray, astropy.time.Time
                The time of the EUV image to download. Input needs be parsed by sunpy.time.parse_time()
            instrument : str, optional
                The instrument of the EUV image to download. Default is 'AIA'.
            source : str, optional
                The source of the EUV image to download. Default is 'SDO'.
            wavelength : int, optional
                The wavelength of the EUV image to download (in Angstrom). Default is 193.
            ** kwargs: 
                Additional keyword arguments passed to sunpy.Fido.search (see sunpy documentation for more information).
        
        Returns 
        -------
        None
        
        Notes 
        -------
        For EIT data: instr='EIT', wave=195, source='SOHO'
        For STEREO-A data: instr='SECCHI', wave=195, source='STEREO_A'
        For STEREO-B data: instr='SECCHI', wave=195, source='STEREO_B'
        """

        if not isinstance(instr, str):
            print("> pycatch ##  'instr' argument must be type str")
            return

        if not isinstance(source, str):
            print("> pycatch ##  'instr' argument must be type str")
            return
        
        # Check the type of the 'wave' argument
        if not isinstance(wave, int):
            print("> pycatch ## 'wave' argument must be type int")
            return 
        
        try:
            t=sunpy.time.parse_time(time)
        except:
            print("> pycatch ## 'time' argument must be valid input for sunpy.time.parse_time()")
            return 
        # jsoc not working !!
        try:
            res = Fido.search(a.Time(t-60*u.min,t+60*u.min, near=t),a.Instrument(instr),a.Wavelength(wave*u.angstrom), a.Source(source), **kwargs)
            if len(res) == 0:
                print(f"> pycatch ## No data found: {source} {instr} {wave}A, {t.value}")
                return
            downloaded_files = Fido.fetch(res, path=self.dir + '/{instrument}/{file}' ) 
            self.map_file = downloaded_files[0]
        except:
            print(f"> pycatch ## DOWNLOAD OF {source} {instr} {wave}A, {t.value} FAILED")
        return 
    
#############################################################################################################################################   
    
    def download_magnetogram(self, cadence=45,time=None, **kwargs): #Fido.search **kwargs
        """
        Download a magnetogram matching the EUV image date using VSO (Virtual Solar Observatory). It downloads the closest image within +/- 1 hour ot the time of the EUV image.
        
        Parameters
        ----------
            cadence : int, optional
                Download Line-of-Sight (LOS) magnetogram with the specified cadence in seconds. Default is 45.
            time : optional, tuple, None or list, str, pandas.Timestamp, pandas.Series, pandas.DatetimeIndex, datetime.datetime, datetime.date, numpy.datetime64, numpy.ndarray, astropy.time.Time
                The time of the magnetogram to download. Input needs be parsed by sunpy.time.parse_time()
                Overrides the date of the EUV map and the filepath of the downloaded data is not stored in self.magnetogram_file.
            ** kwargs : 
                Additional keyword arguments passed to sunpy.Fido.search (see sunpy documentation for more information).
        
        Returns 
        -------
        None
        """
        
        # Check the type of the 'cadence' argument
        if not isinstance(cadence, int):
            print("> pycatch ## 'cadence' argument must be type int")
            return 
                    
        # input date    
        if time is not None:
            print('> pycatch ## WARNING ##')
            print('> pycatch ## Download of HMI 45s magnetograms only ##')            
            print('> pycatch ## Manually download 720s magnetograms from JSOC ##')
            try:
                t=sunpy.time.parse_time(time)
            except:
                print("> pycatch ## 'time' argument must be valid input for sunpy.time.parse_time()")
                return 
                    
            try:
                if cadence == 45:
                    res = Fido.search(a.Time(t-60*u.min,t+60*u.min, near=t),a.Instrument('HMI'),a.Physobs("LOS_magnetic_field"), **kwargs)
                    if len(res) == 0:
                        print(f"> pycatch ## No magnetogram found: {t.value}")
                        return
                    downloaded_files = Fido.fetch(res, path=self.dir + '/{instrument}/{file}') 
                    return
            except:
                print(f"> pycatch ## DOWNLOAD OF MAGNETOGRAM {t.value} FAILED")           
                return
                
                
        else:
            if self.type is not None and self.map is not None:
            
                if 'SDO' in self.type: 
                    t=sunpy.time.parse_time(self.map.meta['DATE-OBS'])
        
                    print('> pycatch ## WARNING ##')
                    print('> pycatch ## Download of HMI 45s magnetograms only ##')            
                    print('> pycatch ## Manually download 720s magnetograms from JSOC ##')
                    try:
                        if cadence == 45:
                            res = Fido.search(a.Time(t-60*u.min,t+60*u.min, near=t),a.Instrument('HMI'),a.Physobs("LOS_magnetic_field"), **kwargs)
                            if len(res) == 0:
                                print(f"> pycatch ## No magnetogram found: {t.value}")
                                return
                            downloaded_files = Fido.fetch(res, path=self.dir + '/{instrument}/{file}') 
                            self.magnetogram_file = downloaded_files[0]
                            return
                    except:
                        print(f"> pycatch ## DOWNLOAD OF MAGNETOGRAM {t.value} FAILED")
                        return
                elif 'SOHO' in self.type:
                    print('> pycatch ## DOWNLOAD OF SOHO MAGNETOGRAMS NOT YET IMPLEMENTED ##')
            else:
                print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
                return
        return
        
 #############################################################################################################################################   
   
    # Load data using sunpy FIDO
    def load(self, mag=False, file = None):
        """
        Load maps.
        
        Parameters
        ----------
            mag : bool, optional
                Flag to load a magnetogram. Default is False.
            file : str, optional
                Filepath to load a specific map. If not set, it loads pycatch.map_file or pycatch.magnetogram_file. Default is None.
        
        Returns 
        -------
        None
        """

        # Check the type of the 'file' argument
        if file is not None and not isinstance(file, str):
            print("> pycatch ## 'file' argument must be type str")
            return 
        
        # Check the type of the 'mag' argument
        if not isinstance(mag, bool):
            print("> pycatch ## 'mag' argument must be type bool")
            return 
                
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
        
#############################################################################################################################################   
        
    # calibrate EUV data
    def calibration(self,**kwargs):
        """
        Calibrate the intensity image.
        
        Parameters
        ----------
            ** kwargs (SDO/AIA) :
                deconvolve : bool or numpy.ndarray, optional
                    Use PSF deconvolution. Default is None. It takes a custom PSF array as input, if True uses aiapy.psf.deconvolve.
                    WARNING: can take about 10 minutes.
                register : bool, optional
                    Co-register the map. Default is True.
                normalize : bool, optional
                    Normalize intensity to 1s. Default is True.
                degradation : bool, optional
                    Correct instrument degradation. Default is True.
                alc : bool, optional
                    Apply Annulus Limb Correction, Python implementation from Verbeek et al. (2014). Default is True.
                cut_limb : bool, optional
                    Set off-limb pixel values to NaN. Default is True.
            
            ** kwargs (STEREO/SECCHI) :
                deconvolve : bool, optional
                    NOT YET IMPLEMENTED FOR STEREO.
                register : bool, optional
                    Co-register the map. Default is True.
                normalize : bool, optional
                    Normalize intensity to 1s. Default is True.
                alc : bool, optional
                    Apply Annulus Limb Correction, Python implementation from Verbeek et al. (2014). Default is True.
                cut_limb : bool, optional
                    Set off-limb pixel values to NaN. Default is True.
        
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

#############################################################################################################################################   

    # calibrate EUV data
    def calibration_mag(self,**kwargs):
        """
        Calibrate the magnetogram.
        
        Parameters
        ----------
            ** kwargs (SDO/HMI) : 
                rotate : bool, optional
                    Rotate the map so that North is up. Default is True.
                align : bool, optional
                    Align with an AIA map. Default is True.
                cut_limb : bool, optional
                    Set off-limb pixel values to NaN. Default is True.

        Returns 
        -------
        None
        """

        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        
        self.magnetogram = cal.calibrate_hmi(self.magnetogram,self.map, **kwargs)   
        return

#############################################################################################################################################   
    
    # make submap
    def cutout(self,top=(1100,1100), bot=(-1100,-1100)):
        """
        Cut a subfield of the map (if a magnetogram is loaded, it will also be cut).
        
        Parameters
        ----------
            top : tuple, optional
                Coordinates of the top-right corner. Default is (1100, 1100).
            bot : tuple, optional
                Coordinates of the bottom-left corner. Default is (-1100, -1100)).
        
        Returns 
        -------
        None
        """
        # Check the type of the 'top' argument
        if not isinstance(top, tuple) or len(top) != 2 or not all(isinstance(val, (int,float)) for val in top):
            print("> pycatch ## 'top' argument must be a tuple of two float")
            return

        # Check the type of the 'bot' argument
        if not isinstance(bot, tuple) or len(bot) != 2 or not all(isinstance(val, (int,float)) for val in bot):
            print("> pycatch ## 'bot' argument must be a tuple of two float")
            return


        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
          
        self.map=mapping.cutout(self.map,top,bot)
        self.point, self.curves,  self.threshold = None, None, None
        self.cutout_status      = True
        
        if self.magnetogram is not None:
            self.magnetogram=mapping.cutout(self.magnetogram,top,bot)
        return

#############################################################################################################################################   

    # rebin map
    def rebin(self,ndim=(1024,1024),**kwargs):
        """
        Rebin maps to a new resolution (if a magnetogram is loaded, it will also be resampled).
        
        Parameters
        ----------
            ndim : tuple, optional
                New dimensions of the map. Default is (1024, 1024)).
            ** kwargs : 
                Additional keyword arguments passed to sunpy.map.Map.resample (see sunpy documentation for more information).
        
        Returns 
        -------
        None
        """
        if not isinstance(ndim, tuple) or len(ndim) != 2 or not all(isinstance(val, int) for val in ndim):
            print("> pycatch ## 'ndim' argument must be a tuple of two integers")
            return 
                    
            
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
          
        new_dimensions = ndim * u.pixel
        self.map = self.map.resample(new_dimensions,**kwargs)
        #self.map.data[:]=ext.congrid(self.map.data,(ndim[0],ndim[1]))  #### TEST CONGRID VS RESAMPLE
        self.rebin_status = True
        self.point, self.curves = None, None
        
        if self.magnetogram is not None:
            self.magnetogram = self.magnetogram.resample(new_dimensions, **kwargs)
            #self.magnetogram.data[:]=ext.congrid(self.magnetogram.data,(ndim[0],ndim[1]))  #### TEST CONGRID VS RESAMPLE
        return        
 
#############################################################################################################################################   
       
    # select seed point from EUV data
    def select(self,hint=False,fsize=(10,10)):
        """
        Select a seed point from the intensity map.
        
        Parameters
        ----------
            hint : bool, optional
                If True, highlights possible coronal holes. Default is False. 
            fsize : tuple, optional
                Set the figure size (in inch). Default is (10, 10).
        
        Returns 
        -------
        None
        """
        if not isinstance(fsize, tuple) or len(fsize) != 2 or not all(isinstance(val, (int, float)) for val in fsize):
            print("> pycatch ## 'fsize' argument must be a tuple of two numbers")
            return
        
        # Check the type of the 'hint' argument
        if not isinstance(hint, bool):
            print("> pycatch ## 'hint' argument must be type bool")
            return 

        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        
        self.point=poptions.get_point_from_map(self.map,hint, fsize)
        
        return

#############################################################################################################################################   
            
    # set threshold
    def set_threshold(self,threshold, median = True, no_percentage = False):
        """
        Set the coronal hole extraction threshold.
        
        Parameters
        ----------
            threshold : float
                The threshold value.
            median : bool, optional
                If True, the input is assumed to be a fraction of the median solar disk intensity. Default is True.
            no_percentage : bool, optional
                If True, the input is given as a percentage of the median solar disk intensity. Default is False.
                This only works in conjunction with median=True.
        
        Returns 
        -------
        None
        """
        
        # Check the type of the 'threshold' argument
        if not isinstance(threshold, (int,float)):
            print("> pycatch ## 'threshold' argument must be type int or float:")
            return
        
        # Check the type of the 'median' argument
        if not isinstance(median, bool):
            print("> pycatch ## 'median' argument must be type bool")
            return
        
        # Check the type of the 'no_percentage' argument
        if not isinstance(no_percentage, bool):
            print("> pycatch ## 'no_percentage' argument must be type bool")
            return
        
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
                        
#############################################################################################################################################   
    
    # suggest threshold based on CATCH statistics (Heinemann et al. 2019)
    # TH = 0.29 × Im + 11.53 [DN] 
    def suggest_threshold(self):
        """
        Suggest a coronal hole extraction threshold based on the CATCH statistics (Heinemann et al. 2019).
        
        This function calculates a threshold value using the formula:
        TH = 0.29 * Im + 11.53 [DN]
        
        where Im is the median solar disk intensity in Data Numbers (DN).
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return        
        self.threshold=ext.median_disk(self.map) * 0.29 + 11.53
        print(f'> pycatch ## THE SUGGESTED THREHSOLD IS {self.threshold} ##')
        return

#############################################################################################################################################   
    
    # pick threshold from histogram
    def threshold_from_hist(self,fsize=(10,5)):
        """
        Select a threshold for coronal hole extraction from the solar disk intensity histogram.
        
        Parameters
        ----------
            fsize : tuple, optional
                Set the figure size (in inch). Default is (10, 5).
        
        Returns
        -------
        None
        """
        if not isinstance(fsize, tuple) or len(fsize) != 2 or not all(isinstance(val, (int, float)) for val in fsize):
            print("> pycatch ## 'fsize' argument must be a tuple of two numbers")
            return
        
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
    
        self.threshold=poptions.get_thr_from_hist(self.map,fsize)
        return            

#############################################################################################################################################   
            
    # pick threshold from area curves
    def threshold_from_curves(self,fsize=(10,5)):
        """
        Select a threshold for coronal hole extraction from calculated area and uncertainty curves as a function of intensity.
        
        Before using this function, you need to calculate the curves using `pycatch.calculate_curves()`.
        
        Parameters
        ----------
            fsize : tuple, optional
                Set the figure size (in inch). Default is (10, 5).
        
        Returns
        -------
        None
        """
        if not isinstance(fsize, tuple) or len(fsize) != 2 or not all(isinstance(val, (int, float)) for val in fsize):
            print("> pycatch ## 'fsize' argument must be a tuple of two numbers")
            return
        
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        
        if self.curves is None:
            print('> pycatch ## NO AREA CURVES CALCULATED ##')
            return
        
        self.threshold=poptions.get_thr_from_curves(self.map,self.curves,fsize)
        return                 

#############################################################################################################################################   

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
        
        # Check the type of the 'verbose' argument
        if not isinstance(verbose, bool):
            print("> pycatch ##  'verbose' argument must be type bool")
            return
    
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

#############################################################################################################################################   
            
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
        
        # Check the type of the 'kernel' argument
        if kernel is not None and not isinstance(kernel, int):
            print("> pycatch ## 'kernel' argument must be type int or None")
            return

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
        
        if np.nansum(binmaps[4].data) == 0:
            print('> pycatch ## WARNING ##')
            print('> pycatch ## NO CORONAL HOLE EXTRACTED WITH THE CURRENT CONFIGURATION ##')
            return
        
        self.binmap =mapping.to_5binmap(binmaps)
        return               

#############################################################################################################################################   
            
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
        # Check the type of the 'mag' argument
        if not isinstance(mag, bool):
            print("> pycatch ## 'mag' argument must be type bool")
            return
        
        # Check the type of the 'align' argument
        if not isinstance(align, bool):
            print("> pycatch ## 'align' argument must be type bool")
            return
    

        if mag:
            if self.binmap is None:
                print('> pycatch ## NO CORNAL HOLES EXTRACTED ##')
                return
            
            if self.magnetogram is None:
                print('> pycatch ## NO MAGNETOGRAM LOADED ##')
                return
            
            if align:
                self.magnetogram = cal.calibrate_hmi(self.magnetogram,self.binmap)  
                
            if self.binmap.data.shape != self.magnetogram.data.shape:
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
            
#############################################################################################################################################              
            
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
        # Check the type of the 'file' argument
        if file is not None and not isinstance(file, str):
            print("> pycatch ## 'file' argument must be type str or None")
            return

        # Check the type of the 'overwrite' argument
        if not isinstance(overwrite, bool):
            print("> pycatch ## 'overwrite' argument must be type bool")
            return
            
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
            else:
                datestr=sunpy.time.parse_time(self.map.meta['DATE-OBS']).strftime('%Y%m%dT%H%M%S')
                typestr=self.map.meta['telescop'].replace('/','_')
                nr=0
                fpath=self.dir+'pyCATCH_properties_'+typestr+'_'+datestr+f'_{nr}'+'.txt'
                if not overwrite:
                    while os.path.isfile(fpath):
                        nr+=1
                        fpath=self.dir+'pyCATCH_properties_'+typestr+'_'+datestr+f'_{nr}'+'.txt'
            
                ext.printtxt(fpath, self.properties,self.names, pycatch.__version__)
                    
                print(f'> pycatch ## PROPERTIES SAVED: {fpath}  ##')
                

        except Exception as ex:
                print("> pycatch ## Error during saving file:", ex)
        return           

#############################################################################################################################################                   
    
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
                Set the figure size (in inch). Default is (10, 10).
            save : bool, optional
                Save and close the figure. Default is False.
            sfile : str, optional
                Filepath to save the image as pdf. If not provided, a default filename will be generated based on observation metadata.
                Use only in conjunction with save=True. Default is None.
            overwrite : bool, optional
                Overwrite the plot if it already exists. Default is True.
            ** kwargs : keyword arguments
                Additional keyword arguments for sunpy.map.Map.plot(). See the sunpy documentation for more information.
        
        Returns
        -------
        None
        """
        # Check the types of various arguments
        if not isinstance(boundary, bool):
            print("> pycatch ## 'boundary' argument must be type bool")
            return

        if not isinstance(uncertainty, bool):
            print("> pycatch ## 'uncertainty' argument must be type bool")
            return

        if not isinstance(original, bool):
            print("> pycatch ## 'original' argument must be type bool")
            return

        if cutout is not None and (not isinstance(cutout, list) or any(not isinstance(coord, tuple) or len(coord) != 2 for coord in cutout)):
            print("> pycatch ## 'cutout' argument must be a list of tuples with format [(xbot, ybot), (xtop, ytop)]")
            return

        if not isinstance(mag, bool):
            print("> pycatch ## 'mag' argument must be type bool")
            return

        if not isinstance(fsize, tuple) or len(fsize) != 2 or not all(isinstance(val, (int, float)) for val in fsize):
            print("> pycatch ## 'fsize' argument must be a tuple of two numbers")
            return

        if not isinstance(save, bool):
            print("> pycatch ## 'save' argument must be type bool")
            return

        if sfile is not None and not isinstance(sfile, str):
            print("> pycatch ## 'sfile' argument must be type str or None")
            return

        if not isinstance(overwrite, bool):
            print("> pycatch ## 'overwrite' argument must be type bool")
            return
            
            
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

        if sfile is not None:
            fpath = sfile
        else:
            datestr=sunpy.time.parse_time(self.map.meta['DATE-OBS']).strftime('%Y%m%dT%H%M%S')
            typestr=self.map.meta['telescop'].replace('/','_')
            nr=0
            fpath=self.dir+'pyCATCH_plot_'+typestr+'_'+datestr+addstr+f'_{nr}'+'.pdf'
            if not overwrite:
                while os.path.isfile(fpath):
                    nr+=1
                    fpath=self.dir+'pyCATCH_plot_'+typestr+'_'+datestr+f'_{nr}'+'.pdf'
                                
                                
        poptions.plot_map(pmap,pbinmap,boundary,uncertainty, fsize, save, fpath,**kwargs)
        return                     
                
#############################################################################################################################################   
            
    # save binary map to fits file
    def bin2fits(self,file=None, small=False, overwrite=False):          
        """
        Save the coronal hole binary map to a FITS file.
    
        Parameters
        ----------
        file : str, optional
            Filepath to save the FITS file. If not provided, a default filename will be generated based on observation metadata. Default is None.
        small : bool, optional
            Save a smaller region around the coronal hole. Default is False.
        overwrite : bool, optional
            Flag to overwrite the file if it already exists. Default is False.
    
        Returns
        -------
        None
        """            

        # Check the type of the 'file' argument
        if file is not None and not isinstance(file, str):
            print("> pycatch ## 'file' argument must be type str or None")

        # Check the type of the 'small' argument
        if not isinstance(small, bool):
            print("> pycatch ## 'small' argument must be type bool")

        # Check the type of the 'overwrite' argument
        if not isinstance(overwrite, bool):
            print("> pycatch ## 'overwrite' argument must be type bool")
            
        if self.binmap is None:
            print('> pycatch ## WARNING ##')
            print('> pycatch ## NO CORONAL HOLE EXTRACTED ##')
            print('> pycatch ## NO FITS FILE SAVED ##')
            return
        
        #try:  
        
        if small:
            addstr='_small'
        else:
            addstr=''
            
        if file is not None:
            fpath = file
        else:
            datestr=sunpy.time.parse_time(self.map.meta['DATE-OBS']).strftime('%Y%m%dT%H%M%S')
            typestr=self.map.meta['telescop'].replace('/','_')
            nr=0
            fpath=self.dir+'pyCATCH_binmap_'+typestr+'_'+datestr+addstr+f'_{nr}'+'.fits'
            
            if not overwrite:
                while os.path.isfile(fpath):
                    nr+=1
                    fpath=self.dir+'pyCATCH_binmap_'+typestr+'_'+datestr+addstr+f'_{nr}'+'.fits'
            
        if overwrite:
            if os.path.isfile(fpath):
                os.remove(fpath)
        
        meta_update={'pyCATCH':self.__version__,'THR':self.threshold,'SEED':self.point}
            
        if small:
            bot,top=ext.get_extent(self.binmap)
            print(top,bot) 
            pbinmap=mapping.cutout(self.binmap,(top[0]+50,top[1]+50),(bot[0]-50,bot[1]-50))
            pbinmap.meta.update(meta_update)
            pbinmap.save(fpath)
            pass
        else:
            self.binmap.meta.update(meta_update)
            self.binmap.save(fpath)
            
        print(f'> pycatch ## CORONAL HOLE EXTRACTION SAVED: {fpath}  ##')
                
       # except Exception as ex:
       #         print("> pycatch ## Error during saving file:", ex)
        return                       
            
#############################################################################################################################################               

