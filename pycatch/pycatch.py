import os,sys
import numpy as np
import pathlib
import copy
import pickle
import itertools

import astropy.units as u

import sunpy
import sunpy.map
import sunpy.util.net
from sunpy.net import Fido, attrs as a

import matplotlib.colors as colors
import matplotlib as mpl

import pycatch.utils.calibration as cal
import pycatch.utils.extensions as ext
import pycatch.utils.plot as poptions
import pycatch.utils.ch_mapping as mapping
from pycatch._version import __version__

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
    
    
    Methods
    -------
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
        dir_path = kwargs.get('dir', str(pathlib.Path.home())+'/')
        if not isinstance(dir_path, str):
            raise TypeError("> pycatch ##  'dir' argument must be type str")

        # Check the type of the 'save_dir' argument
        save_dir_path = kwargs.get('save_dir', str(pathlib.Path.home())+'/')
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
        self.properties = {
            'A': [], 'dA': [], 'Imean': [], 'dImean': [], 'Imed': [], 'dImed': [], 
            'CoM': [], 'dCoM': [], 'ex': [], 'dex': [], 'Bs': [], 'dBs': [], 
            'Bus': [], 'dBus': [], 'Fs': [], 'dFs': [], 'Fus': [], 'dFus': [], 
            'FB': [], 'dFB': []
        }
        self._keys = [
            'A', 'dA', 'Imean', 'dImean', 'Imed', 'dImed', 'CoM', 'dCoM', 'ex', 'dex',
            'Bs', 'dBs', 'Bus', 'dBus', 'Fs', 'dFs', 'Fus', 'dFus', 'FB', 'dFB'
        ]
        #self.properties         =  {'A':None,'dA':None,'Imean':None,'dImean':None,'Imed':None,'dImed':None,'CoM':None,'dCoM':None,'ex':None,'dex':None,
                                    #'Bs':None,'dBs':None,'Bus':None,'dBus':None,'Fs':None,'dFs':None,'Fus':None,'dFus':None,'FB':None,'dFB':None }
        self.names              =  ext.init_props()
        self._is_notebook        = self._check_notebook()
        
        # Configure backend dynamically before plotting if in script mode
        self._setup_backend()
        
        
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

 
    def _check_notebook(self):
        """Check if execution environment is specifically a Jupyter Notebook or JupyterLab.
    
        Excludes Spyder, standard terminal IPython, and plain scripts.
        """
        try:
            from IPython import get_ipython
    
            ipython = get_ipython()
            if ipython is None:
                return False
    
            # Spyder explicitly sets these modules/attributes
            if "spyder" in sys.modules or hasattr(ipython, "spyder_kernel"):
                return False
    
            shell = ipython.__class__.__name__
    
            # Jupyter Notebook/Lab or Google Colab
            if shell in ["ZMQInteractiveShell", "Shell"]:
                return True
    
            return False
        except (NameError, ImportError):
            return False
    
    
    def _setup_backend(self):
        """Safely switch to an interactive window backend without killing the IPython/Spyder kernel."""
        if not self._is_notebook:
            current_backend = mpl.get_backend().lower()
    
            # If backend is non-interactive (agg) or inline (Spyder default)
            if "agg" in current_backend or "inline" in current_backend:
                try:
                    from IPython import get_ipython
    
                    ipython = get_ipython()
    
                    if ipython is not None:
                        # SAFEST WAY IN SPYDER/IPYTHON: Use magic command.
                        # This safely detaches the inline hook and starts the Qt event loop.
                        ipython.run_line_magic("matplotlib", "qt")
                    else:
                        # Fallback for non-IPython plain Python scripts
                        mpl.use("QtAgg", force=True)
    
                except Exception:
                    # If Qt fails, try fallback GUI magic
                    try:
                        from IPython import get_ipython
    
                        get_ipython().run_line_magic("matplotlib", "tk")
                    except Exception:
                        pass
                    
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
    def download(self, time,instr='AIA', wave=193,source='SDO', email = None, mode='JSOC', series='aia.lev1_euv_12s',  **kwargs):
        r"""
        Download an EUV map using either VSO or JSOC via SunPy FIDO.

        Downloads the closest image within +/- 15 minutes of the provided time.

        Parameters
        ----------
        time : tuple, list, str, pandas.Timestamp, pandas.Series, pandas.DatetimeIndex, datetime.datetime, datetime.date, numpy.datetime64, numpy.ndarray, astropy.time.Time
            The timestamp of the EUV image to download. Must be parseable by `sunpy.time.parse_time()`.
        instr : str, optional
            The instrument name. Default is 'AIA'.
        wave : int, optional
            The wavelength in Angstroms. Default is 193.
        source : str, optional
            The observational source/spacecraft. Default is 'SDO'.
        email : str, optional
            User email required for JSOC queries when `mode='JSOC'`. Default is None.
        mode : str, optional
            Download client mode, either 'VSO' or 'JSOC'. Default is 'JSOC'.
        series : str, optional
            JSOC series name to query when `mode='JSOC'`. Default is 'aia.lev1_euv_12s'.
        \*\*kwargs : dict, optional
            Additional keyword arguments passed to `sunpy.net.Fido.search`.

        Returns
        -------
        None

        Notes
        -----
        Common provider parameters for non-SDO observations:
        * EIT: instr='EIT', wave=195, source='SOHO'
        * STEREO-A: instr='SECCHI', wave=195, source='STEREO_A'
        * STEREO-B: instr='SECCHI', wave=195, source='STEREO_B'
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
        except Exception:
            print("> pycatch ## ERROR -- time argument must be valid input for sunpy.time.parse_time()")
            return 
        
        if mode == 'JSOC':
            if email is None:
                print("> pycatch ##  ERROR -- NO EMAIL FOR JSOC DOWNLOAD PROVIDED")
                return

            try:
                res = Fido.search(
                    a.Time(t - 15 * u.min, t + 15 * u.min),
                    a.jsoc.Series(series),
                    a.jsoc.Notify(email),
                    a.Wavelength(wave * u.angstrom),
                    **kwargs
                )
                
                if len(res) == 0 or len(res['jsoc']) == 0:
                    print(f"> pycatch ## No JSOC data found for {series} at {t.value}")
                    return
                try:
                    obs_times = sunpy.time.parse_time(res['jsoc']['T_OBS'])
                except Exception:
                    obs_times = sunpy.time.parse_time(res['jsoc']['T_REC'])
                    
                closest_idx = np.abs(obs_times - t).argmin()
                closest_time = obs_times[closest_idx]
                
                # 3. Query JSOC again for a narrow 1-second window around that exact timestamp
                exact_res = Fido.search(
                    a.Time(closest_time - 0.5 * u.s, closest_time + 0.5 * u.s),
                    a.jsoc.Series(series),
                    a.jsoc.Notify(email),
                    a.Wavelength(wave * u.angstrom),
                    **kwargs
                )
                downloaded_file = Fido.fetch(exact_res, path=self.dir + f'/{series}/' +'{file}' ) 
                self.map_file = downloaded_file[0]
                for filepath in downloaded_file:
                    print("> pycatch ## Downloaded to:", filepath)

            except Exception:
                    print(f"> pycatch ## DOWNLOAD OF {series} {wave} {t.value} FAILED")
                    
        elif mode == 'VSO':
            try:
                res = Fido.search(
                    a.Time(t - 15 * u.min, t + 15 * u.min, near=t),
                    a.Instrument(instr),
                    a.Wavelength(wave * u.angstrom),
                    a.Source(source),
                    **kwargs
                )
                if len(res) == 0 or len(res['VSO']) == 0:
                    print(f"> pycatch ## No VSO data found: {source} {instr} {wave}, {t.value}")
                    return

                downloaded_file = Fido.fetch(res, path=self.dir + f'/{instr}/' +'{file}' ) 
                self.map_file = downloaded_file[0]
                for filepath in downloaded_file:
                    print("> pycatch ## Downloaded to:", filepath)
            except Exception:
                print(f"> pycatch ## DOWNLOAD OF {source} {instr} {wave} {t.value} FAILED")
        else:
            print("> pycatch ## ERROR -- no mode of download provided. Please use 'VSO' or 'JSOC'.")
        return 
    
#############################################################################################################################################   
    
    def download_magnetogram(self, cadence=45,time=None, email = None, mode='JSOC', series='hmi.M_720s', **kwargs):
        r"""
        Download a magnetogram closest in time to a given date or an existing EUV image map.
    
        Parameters
        ----------
        cadence : int, optional
            Temporal resolution / cadence expectation in seconds. Default is 45.
        time : tuple, str, pandas.Timestamp, datetime.datetime, astropy.time.Time, or None, optional
            Target time of the magnetogram to download, parsed via `sunpy.time.parse_time()`.
            If None, attempts to extract the date from `self.map.meta['DATE-OBS']`.
        email : str, optional
            Email address required for JSOC exports. Required if `mode='JSOC'`.
        mode : {'JSOC', 'VSO'}, optional
            Data service interface to query. Default is 'JSOC'.
        series : str, optional
            JSOC series name to query. Default is 'hmi.M_720s'.
        \*\*kwargs :
            Additional keyword arguments passed directly to `sunpy.net.Fido.search`.
    
        Returns
        -------
        None
            Sets `self.magnetogram_file` to the path of the downloaded file.
        """
        
        # Check the type of the 'cadence' argument
        if not isinstance(cadence, int):
            print("> pycatch ## 'cadence' argument must be type int")
            return 
                    
        # input date    
        if time is not None:
            if (cadence != 45) & mode == 'VSO':
                print('> pycatch ## WARNING ##')
                print('> pycatch ## VSO can only download SDO/HMI 45s magnetograms and not 720s ##')            
                print('> pycatch ## To download 720s magnetograms use mode = JSOC ##')
            try:
                t=sunpy.time.parse_time(time)
            except Exception:
                print("> pycatch ## 'time' argument must be valid input for sunpy.time.parse_time()")
                return 

        elif getattr(self, 'type', None) is not None and getattr(self, 'map', None) is not None:
            if 'SDO' in self.type: 
                t=sunpy.time.parse_time(self.map.meta['DATE-OBS'])
        
        else:
            print('> pycatch ## ERROR -- NO INTENSITY IMAGE LOADED AND NO TIME GIVEN##')
            return
                            
        if mode == 'JSOC':
            if email is None:
                print("> pycatch ##  ERROR -- NO EMAIL FOR JSOC DOWNLOAD PROVIDED")
                return
            
            try:
                res = Fido.search(
                    a.Time(t - 15 * u.min, t + 15 * u.min),
                    a.jsoc.Series(series),
                    a.jsoc.Notify(email),
                    **kwargs
                )
                
                if len(res) == 0 or len(res['jsoc']) == 0:
                    print(f"> pycatch ## No JSOC data found for {series} at {t.value}")
                    return
                
                try:
                    obs_times = sunpy.time.parse_time(res['jsoc']['T_OBS'])
                except Exception:
                    obs_times = sunpy.time.parse_time(res['jsoc']['T_REC'])
                    
                closest_idx = np.abs(obs_times - t).argmin()
                closest_time = obs_times[closest_idx]
                
                exact_res = Fido.search(
                    a.Time(closest_time - 0.5 * u.s, closest_time + 0.5 * u.s),
                    a.jsoc.Series(series),
                    a.jsoc.Notify(email),
                    **kwargs
                )
                downloaded_file = Fido.fetch(exact_res, path=self.dir + f'/{series}/' +'{file}' ) 
                self.magnetogram_file = downloaded_file[0]
                for filepath in downloaded_file:
                    print("> pycatch ## Downloaded to:", filepath)
        
            except Exception:
                print(f"> pycatch ## DOWNLOAD OF {series} {t.value} FAILED")
                    
        elif mode == 'VSO': 
            try:
                if cadence == 45:
                    res = Fido.search(
                    a.Time(t - 60 * u.min, t + 60 * u.min, near=t),
                    a.Instrument('HMI'),
                    a.Physobs("LOS_magnetic_field"),
                    **kwargs
                    )
                    
                if len(res) == 0 or len(res['VSO']) == 0:
                    print(f"> pycatch ## No VSO data found: {t.value}")
                    return
                    
                    downloaded_file = Fido.fetch(res, path=self.dir + '/HMI/' +'{file}') 
                    self.magnetogram_file = downloaded_file[0]
                    for filepath in downloaded_file:
                        print("> pycatch ## Downloaded to:", filepath)
                    
            except Exception:
                print(f"> pycatch ## DOWNLOAD OF MAGNETOGRAM {t.value} FAILED")           
                return
        return
        
 #############################################################################################################################################   
   
    # Load data using sunpy FIDO
    def load(self,file = None, mag=False):
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
            self.magnetogram.plot_settings['norm'] = colors.Normalize(vmin=-50, vmax=50)
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
        
        if self.threshold is None:
            print('> pycatch ## WARNING: NO THRESHOLD SELECTED ##')
            print('> pycatch ## SETTING SUGGESTED THRESHOLD VALUE ##')
            self.threshold = self.suggest_threshold()
            
        self.point=poptions.get_point_from_map(self.map,hint, fsize, is_notebook=self._is_notebook)
        
        return

#############################################################################################################################################   
            
    # set threshold
    def set_threshold(self,threshold, median = True, no_percentage = False):
        """
        Set the coronal hole extraction threshold.
        
        Parameters
        ----------
            threshold : float, int, list, or np.ndarray
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
        # Convert input to a NumPy array for unified checking
        threshold_arr = np.atleast_1d(np.asarray(threshold))
        
        # Check if the underlying elements are int or float
        if not np.issubdtype(threshold_arr.dtype, np.number):
            print("> pycatch ## 'threshold' argument must contain only ints or floats:")
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
        
        if self.point is not None and len(threshold_arr) > 1:
            if len(self.point) != len(threshold_arr):
                print(f'> pycatch ## NUMBER OF SELECTED CORONAL HOLES ({len(self.point)}) DOES NOT MATCH NUMBER OF THRESHOLDS ({len(threshold_arr)})##')
                return
        
        if median:
            # Check if any element > 2 when no_percentage is True
            if np.any(threshold_arr > 2.0) and no_percentage:
                print('> pycatch ## WARNING ##')
                print(f'> pycatch ## Threshold set to {threshold_arr} * median solar disk intensity ##')
                print(f'> pycatch ## Assuming input to be {threshold_arr} % of the median solar disk intensity instead ##')
                threshold_arr = threshold_arr / 100.0
            
            threshold_arr = ext.median_disk(self.map) * threshold_arr

        if len(threshold_arr) == 1:
            if self.point is None:
                self.threshold=threshold_arr[0]
            elif len(self.point) == 1:
                self.threshold=threshold_arr[0]
            else:
                self.threshold = np.full(len(self.point), threshold_arr[0])
        else:
            self.threshold=threshold_arr
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
        threshold=ext.median_disk(self.map) * 0.29 + 11.53
        print(f'> pycatch ## THE SUGGESTED THRESHOLD IS {threshold} ##')
        if self.point is None:
            self.threshold = threshold
        else:
            npoints = len(self.point)
            if npoints <= 1:
                self.threshold = threshold
            else:
                self.threshold = np.full(npoints, threshold)
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
    
        threshold=poptions.get_thr_from_hist(self.map,fsize, is_notebook=self._is_notebook)
        print(f'> pycatch ## THE SELECTED THRESHOLD IS {threshold} ##')
        if self.point is None:
            self.threshold = threshold
        else:
            npoints = len(self.point)
            if npoints <= 1:
                self.threshold = np.atleast_1d(np.asarray(threshold))
            else:
                self.threshold = np.full(npoints, threshold)
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
        
        thresholds=[]
        for curve in self.curves:
            threshold=poptions.get_thr_from_curves(self.map,curve,fsize, is_notebook=self._is_notebook)
            thresholds.append(threshold)
        self.threshold=thresholds
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
            
        if self.point is None:
            print('> pycatch ## NO CORONAL HOLE SELECTED ##')
            return
        
        if len(self.point) == 1:
            xloc, area, uncertainty =mapping.get_curves(self.map,self.point,kernel=self.kernel)
            self.curves = [(xloc, area, uncertainty/area)]
        else:
            curves=[]
            for point in self.point:
                xloc, area, uncertainty =mapping.get_curves(self.map,self.point,kernel=self.kernel)
                curves.append((xloc, area, uncertainty/area))
            self.curves=curves
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
        
        self.wipe_properties()   # wipe properties in case they were calculated earlier
        
        if len(self.point) == 1:
            binmaps=[mapping.extract_ch(self.map, self.threshold+i, self.kernel,self.point) for i in np.arange(5)-2]
            
            if np.nansum(binmaps[4].data) == 0:
                print('> pycatch ## WARNING ##')
                print('> pycatch ## NO CORONAL HOLE EXTRACTED WITH THE CURRENT CONFIGURATION ##')
                return
            
            self.binmap =[mapping.to_5binmap(binmaps)]
        else:
            binmaps=[]
            for point,threshold in zip(self.point,self.threshold):
                
                bmaps=[mapping.extract_ch(self.map, threshold+i, self.kernel,[point]) for i in np.arange(5)-2]
                bmap=mapping.to_5binmap(bmaps)
                binmaps.append(bmap)
                
                if np.nansum(bmaps[4].data) == 0:
                    print('> pycatch ## WARNING ##')
                    print(f'> pycatch ## NO CORONAL HOLE FOUND WITH THE CURRENT CONFIGURATION FOR LOCATION {point}##')
                
            self.binmap = binmaps
                
        return               

#############################################################################################################################################   
    def wipe_properties(self):
        """Resets all property lists to empty lists."""
        self.properties = {key: [] for key in self._keys}
        
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
                self.magnetogram = cal.calibrate_hmi(self.magnetogram,self.map)  
                
            if self.map.data.shape != self.magnetogram.data.shape:
                print('> pycatch ## BINMAP AND MAGNETOGRAM ARE NOT MATCHING ##')
                return
            
            for binmap in self.binmap:
    
                binmaps=mapping.from_5binmap(binmap)
            
                bs,dbs,bus,dbus,fs,dfs,fus,dfus,fb,dfb = mapping.catch_mag(binmaps, self.magnetogram)  
                
                dict_stage2 = {'Bs': bs, 'dBs': dbs, 'Bus': bus, 'dBus': dbus, 'Fs': fs, 
               'dFs': dfs, 'Fus': fus, 'dFus': dfus, 'FB': fb, 'dFB': dfb}

                for key, val in dict_stage2.items():
                    self.properties[key].append(val)
    
            
            #dict1={'Bs':bs,'dBs':dbs,'Bus':bus,'dBus':dbus,'Fs':fs,'dFs':dfs,'Fus':fs,'dFus':dfs,'FB':fb,'dFB':dfb}
            #self.properties.update(dict1)           

        else:
            if self.binmap is None:
                print('> pycatch ## NO CORNAL HOLES EXTRACTED ##')
                return
            
            for binmap in self.binmap:
                binmaps=mapping.from_5binmap(binmap)
                binmap, a, da, com, dcom, ex, dex = mapping.catch_calc(binmaps)     
                imean,dimean,imed,dimed = mapping.get_intensity(binmaps, self.map)  
                
                dict_stage1 = {'A': a, 'dA': da, 'Imean': imean, 'dImean': dimean, 'Imed': imed, 
               'dImed': dimed, 'CoM': com, 'dCoM': dcom, 'ex': ex, 'dex': dex}

                for key, val in dict_stage1.items():
                    self.properties[key].append(val)
            
            #dict1={'A':a,'dA':da,'Imean':imean,'dImean':dimean,'Imed':imed,'dImed':dimed,'CoM':com,'dCoM':dcom,'ex':ex,'dex':dex}
            #self.properties.update(dict1)
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
        
        
        #try:
        if True:
            if file is None:
                datestr = sunpy.time.parse_time(self.map.meta['DATE-OBS']).strftime('%Y%m%dT%H%M%S')
                typestr = self.map.meta['telescop'].replace('/', '_')
            
            nr = 0
            
            for nr, instance_values in enumerate(itertools.zip_longest(*self.properties.values(), fillvalue=None)):
                single_instance = dict(zip(self._keys, instance_values))
                
                if file is not None:
                    base, ext_str = os.path.splitext(file)
                    fpath = f"{base}_{nr}{ext_str}"
                else:
                    fpath = os.path.join(self.dir, f'pyCATCH_properties_{typestr}_{datestr}_{nr}.txt')
                    
                    if not overwrite:
                        while os.path.isfile(fpath):
                            nr += 1
                            fpath = os.path.join(self.dir, f'pyCATCH_properties_{typestr}_{datestr}_{nr}.txt')
            
                ext.printtxt(fpath, single_instance, self.names, self.__version__)
                print(f'> pycatch ## PROPERTIES SAVED: {fpath}  ##')
    

        #except Exception as ex:
        #        print("> pycatch ## Error during saving file:", ex)
        return           

#############################################################################################################################################                   
    
    # display coronal hole
    def plot_map(self, boundary=True, uncertainty=True, original=False, combined=True, 
                 small=False, cutout=None, grid=False, mag=False, fsize=(10, 10), 
                 save=False, sfile=None, overwrite=False, **kwargs):
        """Display a coronal hole plot."""
    
        # Argument validation
        if not isinstance(boundary, bool):
            print("> pycatch ## 'boundary' argument must be type bool")
            return
        if not isinstance(uncertainty, bool):
            print("> pycatch ## 'uncertainty' argument must be type bool")
            return
        if not isinstance(original, bool):
            print("> pycatch ## 'original' argument must be type bool")
            return
        if not isinstance(small, bool):
            print("> pycatch ## 'small' argument must be type bool")
            return
        if cutout is not None and (not isinstance(cutout, list) or any(not isinstance(coord, tuple) or len(coord) != 2 for coord in cutout)):
            print("> pycatch ## 'cutout' argument must be a list of tuples with format [(xbot, ybot), (xtop, ytop)]")
            return
        if not isinstance(grid, bool):
            print("> pycatch ## 'grid' argument must be type bool")
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
    
        # Data availability checks
        if self.map is None:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        if self.magnetogram is None and mag:
            print('> pycatch ## NO MAGNETOGRAM LOADED ##')
            return
        if (boundary or uncertainty) and self.binmap is None:
            print('> pycatch ## NO CORONAL HOLE BOUNDARY EXTRACTED ##')
            return
        if original and self.original_map is None:
            print('> pycatch ## NO ORIGINAL MAP LOADED ##')
            return
    
                    
        # Base map assignment
        addstr = ''
        if original:
            base_pmap = self.original_map
            addstr += '_original'
        elif mag:
            base_pmap = self.magnetogram
            addstr += '_mag'
        else:
            base_pmap = self.map
    
        if combined:
            addstr += '_combined'
            cbmap = mapping.combine_binmaps(self.binmap)
    
            if original:
                new_dimensions = self.original_map.data.shape * u.pixel
                cbmap = cbmap.resample(new_dimensions)
    
            if small:
                pad=100
                bot, top = ext.get_extent(cbmap)
                pbinmap = mapping.cutout(cbmap, (top[0] + pad, top[1] + pad), (bot[0] - pad, bot[1] - pad))
                pmap = mapping.cutout(base_pmap, (top[0] + pad, top[1] + pad), (bot[0] - pad, bot[1] - pad))
            elif cutout is not None:
                pmap = mapping.cutout(base_pmap, cutout[1], cutout[0])
                pbinmap = mapping.cutout(cbmap, cutout[1], cutout[0])
                addstr += '_cut'
            else:
                pmap = base_pmap
                pbinmap = cbmap
    
            if boundary:
                addstr += '_boundary'
            if uncertainty:
                addstr += '_uncertainty'
    
            fpath = self._generate_fpath(sfile, addstr, overwrite)
            poptions.plot_map(pmap, pbinmap, boundary, uncertainty, fsize, save, fpath, grid, **kwargs)
    
        else:
            for idx, cbmap in enumerate(self.binmap):
                curr_addstr = addstr + f'_idx{idx}'
                
                if original:
                    base_pmap = self.original_map
                    new_dimensions = self.original_map.data.shape * u.pixel
                    cbmap = cbmap.resample(new_dimensions)

    
                if small:
                    pad=100
                    bot, top = ext.get_extent(cbmap)
                    pbinmap = mapping.cutout(cbmap, (top[0] + pad, top[1] + pad), (bot[0] - pad, bot[1] - pad))
                    pmap = mapping.cutout(base_pmap, (top[0] + pad, top[1] + pad), (bot[0] - pad, bot[1] - pad))
                elif cutout is not None:
                    pmap = mapping.cutout(base_pmap, cutout[1], cutout[0])
                    pbinmap = mapping.cutout(cbmap, cutout[1], cutout[0])
                    curr_addstr += '_cut'
                else:
                    pmap = base_pmap
                    pbinmap = cbmap
    
                if boundary:
                    curr_addstr += '_boundary'
                if uncertainty:
                    curr_addstr += '_uncertainty'
    
                # Index output files if sfile is supplied to avoid overwriting each loop step
                loop_sfile = f"{os.path.splitext(sfile)[0]}_{idx}{os.path.splitext(sfile)[1]}" if sfile else None
                fpath = self._generate_fpath(loop_sfile, curr_addstr, overwrite)
                
                    
                poptions.plot_map(pmap, pbinmap, boundary, uncertainty, fsize, save, fpath, grid, **kwargs)
    
        return
    
    def _generate_fpath(self, sfile, addstr, overwrite):
        """Helper method to derive target file paths reliably."""
        if sfile is not None:
            return sfile
    
        datestr = sunpy.time.parse_time(self.map.meta['DATE-OBS']).strftime('%Y%m%dT%H%M%S')
        typestr = self.map.meta['telescop'].replace('/', '_')
        nr = 0
        filename = f'pyCATCH_plot_{typestr}_{datestr}{addstr}_{nr}.pdf'
        fpath = os.path.join(self.dir, filename)
    
        if not overwrite:
            while os.path.isfile(fpath):
                nr += 1
                filename = f'pyCATCH_plot_{typestr}_{datestr}{addstr}_{nr}.pdf'
                fpath = os.path.join(self.dir, filename)
    
        return fpath

#############################################################################################################################################                   
    
    # display loaded maps
    def show(self,original=False,binmap=False ,cutout=None, grid=False, mag=False, fsize=(10,10),**kwargs):
        r"""
        Display a coronal hole plot, EUV map, binary detection map, or magnetogram.

        Parameters
        ----------
        original : bool, optional
            Show the original uncropped/unprocessed image. Default is False.
            Overrides `binmap`, `mag`, and default map selection.
        binmap : bool, optional
            Show the combined binary coronal hole extraction map. Default is False.
            Overrides `mag` and default map selection, but is overridden by `original`.
        cutout : list of tuple, optional
            Display a bounding-box cutout around the target region. 
            Format: [(xbot, ybot), (xtop, ytop)]. Default is None.
        grid : bool, optional
            Display heliographic grid lines on the map. Default is False.
        mag : bool, optional
            Show the associated magnetogram map instead of the EUV map or coronal hole plot.
            Default is False. Overridden by `original` and `binmap`.
        fsize : tuple of (int or float), optional
            Figure dimensions (width, height) in inches. Default is (10, 10).
        \*\*kwargs :
            Additional keyword arguments passed to `sunpy.map.Map.plot()`.

        Returns
        -------
        None
        """
        # Check the types of various arguments
        if not isinstance(original, bool):
            print("> pycatch ## 'original' argument must be type bool")
            return

        if cutout is not None and (not isinstance(cutout, list) or len(cutout) != 2 or any(not isinstance(coord, tuple) or len(coord) != 2 for coord in cutout)):
            print("> pycatch ## 'cutout' argument must be a list of two tuples: [(xbot, ybot), (xtop, ytop)]")
            return
        

        if not isinstance(grid, bool):
            print("> pycatch ## 'grid' argument must be type bool")
            return


        if not isinstance(mag, bool):
            print("> pycatch ## 'mag' argument must be type bool")
            return

        if not isinstance(fsize, tuple) or len(fsize) != 2 or not all(isinstance(val, (int, float)) for val in fsize):
            print("> pycatch ## 'fsize' argument must be a tuple of two numbers")
            return
            
            
        if self.map is None and mag == False:
            print('> pycatch ## NO INTENSITY IMAGE LOADED ##')
            return
        if mag and self.magnetogram is None:
            print('> pycatch ## NO MAGNETOGRAM LOADED ##')
            return

        if original and self.original_map is None:
            print('> pycatch ## NO ORIGINAL MAP LOADED ##')
            return
        
        if binmap and self.binmap is None:
            print('> pycatch ## NO CORONAL HOLE EXTRACTION AVAILABLE ##')
            return
        
        
        if original:
            pmap=self.original_map
        elif binmap:
            pmap=mapping.combine_binmaps(self.binmap)
        elif mag:
            pmap=self.magnetogram
        else:
            pmap=self.map
        
        if cutout is not None:
            pmap=mapping.cutout(pmap,cutout[1],cutout[0])
                                
        poptions.show_map(pmap,grid, fsize,**kwargs)
        return            
             
#############################################################################################################################################   
            
                      
    
    def save_map(self, binary=True, original=False, intensity=False, mag=False, file=None, small=False, overwrite=False):
        """
        Save maps to FITS files.

        Parameters
        ----------
        binary : bool, optional
            Save the binary coronal hole map (`self.binmap`). Default is True.
        original : bool, optional
            Save the unprocessed original EUV map (`self.original_map`). Default is False.
        intensity : bool, optional
            Save the preprocessed EUV intensity map (`self.map`). Default is False.
        mag : bool, optional
            Save the magnetogram map (`self.magnetogram`). Default is False.
        file : str, optional
            Custom file path/name to save the FITS file. If None, default file names
            are generated based on observation metadata. Default is None.
        small : bool, optional
            Save a cropped region around the coronal hole. Only applicable when
            `binary=True`. Default is False.
        overwrite : bool, optional
            Flag to overwrite existing files. Default is False.

        Returns
        -------
        None
        """
        # --- Parameter Type Checking ---
        if file is not None and not isinstance(file, str):
            print("> pycatch ## 'file' argument must be type str or None")
            return
        if not isinstance(small, bool):
            print("> pycatch ## 'small' argument must be type bool")
            return
        if not isinstance(overwrite, bool):
            print("> pycatch ## 'overwrite' argument must be type bool")
            return

        # Restrict 'small' parameter to binary mode only
        if small and not binary:
            print("> pycatch ## WARNING: 'small=True' is only supported when 'binary=True'. Ignoring 'small'.")
            small = False

        # --- Define Map Selection Strategy ---
        # Maps configuration: (flag, attribute_name, default_prefix)
        save_targets = [
            (binary, 'binmap', 'pyCATCH_binmap'),
            (original, 'original_map', 'pyCATCH_origmap'),
            (intensity, 'map', 'pyCATCH_map'),
            (mag, 'magnetogram', 'pyCATCH_magmap')
        ]

        # Filter out disabled targets
        active_targets = [target for target in save_targets if target[0]]

        if not active_targets:
            print("> pycatch ## WARNING ## No map selected to save.")
            return

        # Warn if custom file name is passed when saving multiple maps
        if file is not None and len(active_targets) > 1:
            print("> pycatch ## WARNING: Custom 'file' path provided while saving multiple maps. Files will overwrite each other unless 'file' is unique per map.")

        # --- Process Each Requested Map ---
        for _, attr, prefix in active_targets:
            map_obj = getattr(self, attr, None)

            if map_obj is None:
                print(f'> pycatch ## WARNING ## {attr.upper()} IS NONE. SKIPPING SAVE ##')
                continue
            
            if attr == 'binmap':
                map_obj=mapping.combine_binmaps(map_obj)
                
            # Target-specific metadata
            meta_update = {'pyCATCH': getattr(self, '__version__', 'unknown')}
            if attr == 'binmap':
                meta_update['THR'] = getattr(self, 'threshold', None)
                meta_update['SEED'] = getattr(self, 'point', None)

            addstr = '_small' if (small and attr == 'binmap') else ''

            # Construct filepath
            if file is not None:
                fpath = file
            else:
                datestr = sunpy.time.parse_time(map_obj.meta['DATE-OBS']).strftime('%Y%m%dT%H%M%S')
                typestr = map_obj.meta.get('telescop', 'UNKNOWN').replace('/', '_')
                nr = 0
                fpath = os.path.join(self.dir, f'{prefix}_{typestr}_{datestr}{addstr}_{nr}.fits')

                if not overwrite:
                    while os.path.isfile(fpath):
                        nr += 1
                        fpath = os.path.join(self.dir, f'{prefix}_{typestr}_{datestr}{addstr}_{nr}.fits')

            # Handle existing file overwrite
            if overwrite and os.path.isfile(fpath):
                os.remove(fpath)

            # Apply cutout or save standard map
            if small and attr == 'binmap':
                bot, top = ext.get_extent(map_obj)
                pbinmap = mapping.cutout(map_obj, (top[0] + 50, top[1] + 50), (bot[0] - 50, bot[1] - 50))
                pbinmap.meta.update(meta_update)
                pbinmap.save(fpath)
            else:
                map_obj.meta.update(meta_update)
                map_obj.save(fpath)

            print(f'> pycatch ## MAP SAVED [{attr}]: {fpath} ##')

        return
            
#############################################################################################################################################               

