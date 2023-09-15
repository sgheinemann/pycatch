import os, sys

import numpy as np

import astropy.units as u

import sunpy
import sunpy.map
import sunpy.util.net

import astropy.time
from sunpy.map.maputils import all_coordinates_from_map,coordinate_is_on_solar_disk
import copy

import aiapy.psf
import aiapy.calibrate as cal

#--------------------------------------------------------------------------------------------------
#prep aia image
def calibrate_aia(map, register= True, normalize = True,deconvolve = None, alc = True, degradation = True, cut_limb = True):
    """
    Calibrate and preprocess an AIA (Atmospheric Imaging Assembly) map.
    
    This function performs various calibration and preprocessing steps on an AIA map to prepare it for further analysis.
    
    Parameters
    ----------
    map : sunpy.map.Map
        The input AIA map to be calibrated and preprocessed.
    register : bool, optional
        Whether to perform image registration. Default is True.
    normalize : bool, optional
        Whether to normalize the exposure. Default is True.
    deconvolve : bool or None or numpy.ndarray, optional
        Whether to perform PSF deconvolution. If set to True, deconvolution with aiapy.psf.deconvolve is applied.
        If custom PSF array is given, it uses this array instead. If set to False, it is not applied.
    alc : bool, optional
        Whether to perform annulus limb correction. Default is True.
    degradation : bool, optional
        Whether to correct for instrument degradation. Default is True.
    cut_limb : bool, optional
        Whether to cut the limb of the solar disk. Default is True.
    
    Returns
    -------
    sunpy.map.Map
        A calibrated and preprocessed AIA map ready for analysis.
    """

    calmap=copy.deepcopy(map)
    #deconvolve image
    if deconvolve is not None:
        if deconvolve == True:
            psf = aiapy.psf.psf(calmap.wavelength)
            calmap = aiapy.psf.deconvolve(calmap,psf=psf)
        elif deconvolve == False:
            pass
        else:
            print('> pycatch ## CUSTOM PSF NOT YET IMPLEMENTED ##')    
        
    #register map
    if register:
        calmap=cal.register(calmap)
    
    #normalize map
    if normalize:
        calmap=cal.normalize_exposure(calmap)
    
    #correct for instrument degratation
    if degradation:
        deg= cal.degradation(map.meta['wavelnth']*u.angstrom,astropy.time.Time(calmap.meta['date-obs'],scale='utc') )
        deg_data=calmap.data / deg
        deg_data[deg_data < 1] = 1
        calmap=sunpy.map.Map(((deg_data),calmap.meta))

    
    # annulus limb correction
    if alc:
        calmap=annulus_limb_correction(calmap)

        
    #cut limb
    if cut_limb:
        hpc_coords=all_coordinates_from_map(calmap)
        mask=coordinate_is_on_solar_disk(hpc_coords)
        data=np.where(mask == True, calmap.data, np.nan)
        calmap=sunpy.map.Map((data,calmap.meta))
    
    return calmap

#--------------------------------------------------------------------------------------------------
#prep stereo image
def calibrate_stereo(map, register= True, normalize = True,deconvolve = None, alc = True,  cut_limb = True):
    """
    Calibrate and preprocess a STEREO EUV (Solar TErrestrial RElations Observatory) map.
    
    This function performs various calibration and preprocessing steps on a STEREO map to prepare it for further analysis.
    
    Parameters
    ----------
    map : sunpy.map.Map
        The input STEREO map to be calibrated and preprocessed.
    register : bool, optional
        Whether to perform map rotation to register the map. Default is True.
    normalize : bool, optional
        Whether to normalize the exposure. Default is True.
    deconvolve : bool or None or numpy.ndarray, optional
        Whether to perform PSF deconvolution. If set to True, deconvolution with aiapy.psf.deconvolve is applied.
        If custom PSF array is given, it uses this array instead. If set to False, it is not applied.
    alc : bool, optional
        Whether to perform annulus limb correction. Default is True.
    cut_limb : bool, optional
        Set off-limb pixel values to NaN. Default is True.
    
    Returns
    -------
    sunpy.map.Map
        A calibrated and preprocessed STEREO map ready for analysis.
    """

    calmap=copy.deepcopy(map)
    
    #deconvolve image
    if deconvolve:
        print('> pycatch ## DECONVOLUTION FOR STEREO NOT YET IMPLEMENTED ##')
        
    #register map
    if register:
        calmap=calmap.rotate(angle=calmap.meta['crota']*u.deg)
    
    #normalize map
    if normalize:
        data=(calmap.data.astype(float)-calmap.meta['biasmean'])/calmap.meta['exptime']
        calmap=sunpy.map.Map((data,calmap.meta))
        calmap.meta['exptime']=1
        
    
    # annulus limb correction
    if alc:
        calmap=annulus_limb_correction(calmap)

        
    #cut limb
    if cut_limb:
        hpc_coords=all_coordinates_from_map(calmap)
        mask=coordinate_is_on_solar_disk(hpc_coords)
        data=np.where(mask == True, calmap.data, np.nan)
        calmap=sunpy.map.Map((data,calmap.meta))
        
    return calmap
#--------------------------------------------------------------------------------------------------
# annulus limb correction for extraction
def annulus_limb_correction(map):
    """
    Apply annulus limb correction to a solar map.

    This function performs annulus limb correction on a solar map following the method described in Verbeek et al. (2014).
    Transferred from IDL to Python 3 by S.G. Heinemann, June 2022
    
    Parameters
    ----------
    map : sunpy.map.Map
        The input solar map to which the limb correction will be applied.

    Returns
    -------
    sunpy.map.Map
        A solar map with annulus limb correction applied.

    """
    coords = all_coordinates_from_map(map)
    if map.meta['telescop'] == 'STEREO':
        rsun=map.meta['rsun']
    else:
        rsun=map.meta['r_sun']*map.meta['cdelt1']
    dist = (np.sqrt( coords.Tx**2 + coords.Ty**2)/rsun).value
    alc = [0.70, 0.95, 1.08, 1.12]
    data= copy.deepcopy(map.data)
    
    #calc median of inner shell
    median_inner = np.nanmedian(data[dist < alc[0]])
    
    # correct middle-inner shell
    rng=np.arange(alc[0],alc[1],0.01)
    for i in rng:
        ind=np.where(np.logical_and(dist >= i , dist < i+0.01))
        median_shell = np.nanmedian(data[ind])
        alc1= 0.5 *np.sin(np.pi/(alc[1]-alc[0]) * (i - (alc[1] + alc[0])/2) ) +0.5
        corr = (1 - alc1) *data[ind] +alc1 * median_inner *data[ind] /median_shell
        data[ind] = corr
        
    # correct middle-outer shell
    rng=np.arange(alc[1],alc[2],0.01)
    for i in rng:
        ind=np.where(np.logical_and(dist >= i , dist < i+0.01))
        median_shell = np.nanmedian(data[ind])
        corr = median_inner * data[ind] / median_shell
        data[ind] = corr
        
    # correct outer shell
    rng=np.arange(alc[2],alc[3],0.01)
    for i in rng:
        ind=np.where(np.logical_and(dist >= i , dist < i+0.01))
        median_shell = np.nanmedian(data[ind])
        alc2= 0.5 *np.sin(np.pi/(alc[3]-alc[2]) * (i - (alc[3] + alc[2])/2) ) +0.5
        corr = (1 - alc2) *data[ind] +alc2 * median_inner *data[ind] /median_shell
        data[ind] = corr
        
    return sunpy.map.Map((data,map.meta))

#--------------------------------------------------------------------------------------------------
#prep hmi image
def calibrate_hmi(map,intensity_map, rotate= True, align = True, cut_limb = True):
    """
    Calibrate and preprocess an HMI (Helioseismic and Magnetic Imager) map.

    This function performs various calibration and preprocessing steps on an HMI map to prepare it for further analysis.

    Parameters
    ----------
    map : sunpy.map.Map
        The input HMI map to be calibrated and preprocessed.
    intensity_map : sunpy.map.Map
        An intensity map to align the HMI map with.
    rotate : bool, optional
        Whether to rotate the HMI map to have north up. Default is True.
    align : bool, optional
        Whether to align the HMI map with an intensity map. Default is True.
    cut_limb : bool, optional
        Set off-limb pixel values to NaN. Default is True.

    Returns
    -------
    sunpy.map.Map
        A calibrated and preprocessed HMI map ready for analysis.
    """
    
    calmap=copy.deepcopy(map)
    
    #rotate north up
    if rotate:
        calmap = calmap.rotate(order=3)
    
    #align with aia map
    if align:
        calmap= calmap.reproject_to(intensity_map.wcs)
        
    #cut limb
    if cut_limb:
        hpc_coords=all_coordinates_from_map(calmap)
        mask=coordinate_is_on_solar_disk(hpc_coords)
        data=np.where(mask == True, calmap.data, np.nan)
        calmap=sunpy.map.Map((data,calmap.meta))
        
    return calmap