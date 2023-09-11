import os, sys

import numpy as np

import astropy.units as u

import sunpy
import sunpy.map
import sunpy.util.net

import astropy.time
from sunpy.map.maputils import all_coordinates_from_map,coordinate_is_on_solar_disk
import copy

import aiapy
import aiapy.calibrate as cal

#--------------------------------------------------------------------------------------------------
#prep aia image
def calibrate_aia(map, register= True, normalize = True,deconvolve = False, alc = True, degradation = True, cut_limb = True):

    calmap=copy.deepcopy(map)
    #deconvolve image
    if deconvolve:
        calmap = aiapy.psf.deconvolve(calmap)
        
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
def calibrate_stereo(map, register= True, normalize = True,deconvolve = False, alc = True,  cut_limb = True):
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
    # Verbeek et al. (2014): The SPoCA-suite
    # Transferred from IDL to Python 3 by S.G. Heinemann, June 2022
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