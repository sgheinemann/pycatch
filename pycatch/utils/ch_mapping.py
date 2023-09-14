import os, sys

import numpy as np
import astropy.units as u
import scipy.ndimage as ndi
import copy
from joblib import Parallel, delayed
import numexpr as ne
import time

from astropy.coordinates import SkyCoord

import sunpy
import sunpy.map
import sunpy.util.net
from sunpy.coordinates import frames
from sunpy.map.maputils import all_coordinates_from_map,coordinate_is_on_solar_disk


from cv2 import morphologyEx
from cv2 import MORPH_OPEN,MORPH_CLOSE
import cv2

import pycatch.utils.extensions as ext

#--------------------------------------------------------------------------------------------------
# find minimum intensity around seed to speed up calculation of curves
def min_picker(seed, data):
    """
    Find the minimum value in a local neighborhood around a seed point in a data array.
    
    This function identifies the minimum value within a 5x5 local neighborhood centered around a seed point (xs, ys)
    in the input data array.
    
    Parameters
    ----------
    seed : tuple
        The seed point coordinates (xs, ys) as a tuple.
    data : array-like
        The input data array in which to search for the minimum value.
    
    Returns
    -------
    float
        The minimum value within the local neighborhood.
    """
    xs=int(seed[0][0])
    ys=int(seed[0][1])
    ival=data[xs,ys]
    for xj in [-2,-1,0,1,2]:
        for yj in [-2,-1,0,1,2]:
            val=data[xj+xs,yj+ys]
            if val < ival:
                ival = val
    return ival

#--------------------------------------------------------------------------------------------------
# make map cutout
def cutout(map,top,bot):
    """
    Create a cutout region from a solar map based on top and bottom coordinates.
    
    This function generates a cutout region from the input solar map (`map`) using specified top and bottom coordinates.
    
    Parameters
    ----------
    map : sunpy.map.Map
        The input solar map from which the cutout region will be created.
    top : tuple
        A tuple containing the top-right corner coordinates (x, y) of the cutout region in arcseconds.
    bot : tuple
        A tuple containing the bottom-left corner coordinates (x, y) of the cutout region in arcseconds.
    
    Returns
    -------
    sunpy.map.Map
        A cutout region of the input solar map based on the specified coordinates.
    """

    cmap=copy.deepcopy(map)
    topc=SkyCoord(top[0]*u.arcsec,top[1]*u.arcsec, frame=cmap.coordinate_frame)
    botc=SkyCoord(bot[0]*u.arcsec,bot[1]*u.arcsec, frame=cmap.coordinate_frame)
    return cmap.submap(botc,top_right=topc)

#--------------------------------------------------------------------------------------------------
# calculate area of binmaps for curve
def calc_area_curves(map, th,kernel, seed,minval, coreg):
    """
    Wrapper to calculate area curves for a coronal hole in a solar map.
    
    Parameters
    ----------
    map : sunpy.map.Map
        The input solar map containing coronal hole data.
    th : float
        The threshold value for coronal hole extraction.
    kernel : int or None
        The size of the circular kernel for morphological operations.
    seed : tuple
        The seed point coordinates (lon, lat) for coronal hole extraction.
    minval : float
        The minimum threshold value to consider for area calculation.
    coreg : sunpy.map.Map
        The coregistered solar map for area calculation.
    
    Returns
    -------
    numpy.ndarray
        The area curves for the coronal hole as function of intensity.
    """

    if th < minval:
        return np.nan
    binmap=extract_ch(map,th, kernel, seed)
    return ch_area(binmap,coreg,binary=True)

#--------------------------------------------------------------------------------------------------
#calculate area for all thrs
def get_curves(map,seed,kernel=None, upper_lim=False, cores = 8):
    """
    Top-level wrapper to calculate coronal hole area curves for a range of threshold values.
    
    This function calculates coronal hole area curves for a range of threshold values, optionally considering an upper limit. It also computes uncertainty in the area curves. The function uses parallel processing to speed up computation.
    
    Parameters
    ----------
    map : sunpy.map.Map
        The input solar map containing coronal hole data.
    seed : tuple
        The seed point coordinates (x, y) for coronal hole extraction.
    kernel : int or None, optional
        The size of the circular kernel for morphological operations. Default is None.
    upper_lim : int or False, optional
        The upper limit for the threshold range. If False, the limit is calculated based on the median value of the solar map data. Default is False.
    cores : int, optional
        The number of CPU cores to use for parallel processing. Default is 8.
    
    Returns
    -------
    tuple
        A tuple containing three arrays:
        - The threshold range.
        - The calculated area curves for the coronal hole.
        - The uncertainty in the area curves.
    """

    if upper_lim:
        rng=np.arange(0,int(upper_lim))
    else:
        hpc_coords=all_coordinates_from_map(map)
        mask=coordinate_is_on_solar_disk(hpc_coords)
        data=np.where(mask == True, map.data, np.nan)
        data_median = np.nanmedian(data)
        rng=np.arange(0,1.*int(data_median))
    
    minval=min_picker(seed, map.data) 
    coreg=curve_corr(map)
    output=Parallel(n_jobs=cores)(delayed(calc_area_curves)(map, th,kernel, seed,minval,coreg) for th in rng)
    
    uncertainty=[]
    for xpos in range(len(rng)-4):
        uncertainty.append(catch_uncertainty(output[xpos+2:xpos+7]))
    uc=np.array(uncertainty)
    return rng,np.array(output), np.pad(uc,(2,2),'constant', constant_values=(np.nan,np.nan))
        
#--------------------------------------------------------------------------------------------------
# extract coronal hole from EUV image based on intensity threhold and SEED point
def extract_ch(map,thr, kernel, seed):
    """
    Extract a coronal hole from a solar map based on a threshold and seed point.

    Parameters
    ----------
    map : sunpy.map.Map
        The input solar map containing coronal hole data.
    thr : float
        The threshold value for coronal hole extraction.
    kernel : int or None
        The size of the circular kernel for morphological operations. If None, the kernel size is automatically determined based on map resolution.
    seed : tuple
        The seed point coordinates (x, y) for coronal hole extraction.

    Returns
    -------
    sunpy.map.Map
        A "binary" map containing the extracted coronal hole region.
    """    
    if kernel is None:
        indx= ext.find_nearest((np.arange(5)+1)*0.6, map.meta['cdelt1'])
        kernels=2**np.arange(5)[::-1]
        kernel=kernels[indx]
        
    data=np.where(map.data < thr, 1, 0)
    data=data.astype(np.float32)
    kern=make_circle(kernel)
    data=morphologyEx(data,MORPH_CLOSE,kern)
    data=morphologyEx(data,MORPH_OPEN,kern)
    cont,hier=cv2.findContours(image=data.astype('uint8'), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    
    bindata=np.zeros((data.shape))
    for n,ct in enumerate(cont):
         if cv2.pointPolygonTest(ct,(seed[0][0],seed[0][1]),False) ==1:
             while hier[0][n][3] != -1:
                 ct=cont[hier[0][n][3]]
                 n=hier[0][n][3]
            
             bindata=cv2.fillPoly(bindata,pts=[ct],color=1)
             
             if hier[0][n][0] == -1:
                 num=len(cont)-n -1
             else:
                num=hier[0][n][0] -n -1

             for nchild in range(num):
                 child=cont[nchild+n+1]
                 bindata=cv2.fillPoly(bindata,pts=[child],color=0)
                 bindata=cv2.fillPoly(bindata,pts=child,color=1)
             break
    
    return sunpy.map.Map((bindata,map.meta))


#--------------------------------------------------------------------------------------------------
# calculate correction for curvature of the sun
def curve_corr(map):
    """
    Compute a correction factor for curvature in a solar map.

    Parameters
    ----------
    map : sunpy.map.Map
        The input solar map for which the correction factor is computed.
    
    Returns
    -------
    numpy.ndarray
        A 2D array representing the correction factor for curvature in the solar map.
    """
    coords = all_coordinates_from_map(map)
    if map.meta['telescop'] == 'STEREO':
        rsun=map.meta['rsun'] ## CHECK THIS !!
    else:
        rsun=map.meta['rsun_obs']
    xc=coords.Tx.value 
    yc=coords.Ty.value 
    return ne.evaluate('1/sqrt(1 - (xc**2 + yc**2)/rsun**2)')

#--------------------------------------------------------------------------------------------------
# calculate area of binary map in 10^10 km^2
def ch_area(map, coreg=None, binary = False):
    """
    Calculate the area of a coronal hole in a solar map.

    Parameters
    ----------
    map : sunpy.map.Map
        The input solar map containing coronal hole data.
    coreg : numpy.ndarray, optional
        An optional curvature correction factor for the map. If not provided, it will be computed internally.
    binary : bool, optional
        If True, the map is treated as binary data, where coronal hole pixels have a value of 1. If False, the map is thresholded to binary data. Default is False.

    Returns
    -------
    float
        The calculated area of the coronal hole in 10^10 km^2.
    """
    if np.nansum(map.data) == 0:
        return 0.
    if map.meta['telescop'] == 'STEREO':
        rsun=map.meta['rsun']
    else:
        rsun=map.meta['rsun_obs']
    arcsecTOkm = 1/rsun*696342
    if binary:
        data=map.data
    else:
        data = np.where(map.data == map.data, 1, 0)
    if coreg is None:
        vcoreg = curve_corr(map)
    else: 
        vcoreg=coreg
    tmparea = data * map.meta['cdelt1'] * map.meta['cdelt2'] * vcoreg * arcsecTOkm**2
    return np.nansum(tmparea)/1e10




#--------------------------------------------------------------------------------------------------

# make single binmap from 5 and return

def to_5binmap(binmaps):
    """
    Combine multiple binary maps into a single 5-level binary map.

    Parameters
    ----------
    binmaps : list of sunpy.map.Map
        A list of binary maps to be combined.

    Returns
    -------
    sunpy.map.Map
        A single 5-level binary map where each level represents a different threshold value.
    """
    for i,binmap in enumerate(binmaps):
        if i== 0: 
            help_array=binmap.data
        else:
            help_array+=binmap.data
    
    nbinmap=binmaps[0]
    nbinmap.data[:]=help_array[:]
    
    return nbinmap

def from_5binmap(binmap):
    """
    Split a 5-level binary map into multiple binary maps.

    Parameters
    ----------
    binmap : sunpy.map.Map
        A 5-level binary map containing multiple threshold levels.

    Returns
    -------
    list of sunpy.map.Map
        A list of binary maps, one for each threshold level extracted from the input 5-level binary map.
    """
    binmaps=[]
    nbinmap=copy.deepcopy(binmap)
    for i in range(5,0,-1):
        bmapdata=np.where(binmap.data >= i,1,0)
        nbinmap.data[:]=bmapdata
        binmaps.append(copy.deepcopy(nbinmap))
    return binmaps

#--------------------------------------------------------------------------------------------------

# uncertainty definition for catch properties

def catch_uncertainty(x):
    """
    Calculate the uncertainty of a given data array.
    
    This function computes the uncertainty of a given data array using the maximum absolute deviation from the mean.
    
    Parameters
    ----------
    x : numpy.ndarray
        The input data array for which uncertainty is calculated.
    
    Returns
    -------
    float
        The uncertainty value calculated as the maximum absolute deviation from the mean.
    """
    return np.nanmax( np.sqrt( (x-np.nanmean(x))**2 ) )

#--------------------------------------------------------------------------------------------------

# make circular kernel
def make_circle(s):
    """
    Generate a binary circular mask.

    This function generates a binary circular mask with a specified size, where the circle is centered within the mask.

    Parameters
    ----------
    s : int
        The size of the circular mask (side length).

    Returns
    -------
    numpy.ndarray
        A binary circular mask with the specified size.
    """
    center = ((s-1)/2, (s-1)/2)
    r=s/2

    Y, X = np.ogrid[:s, :s]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= r
    return (1*mask).astype(dtype='uint8')

#--------------------------------------------------------------------------------------------------
# extract coronal hole from EUV image based on intensity threhold and SEED point and calculate the CATCH area and uncertainty and properties.
# output as  A, dA, (lon,lat), (dlon,dlat)

def catch_calc(binmaps, binary =True):
    """
    Calculate coronal hole properties from a list of binary maps.
    
    This function calculates various properties of coronal holes, such as area, center of mass, extent, and their uncertainties, from a list of binary maps representing different levels of coronal hole segmentation.
    
    Parameters
    ----------
    binmaps : list of sunpy.map.Map
        A list of binary maps representing different levels of coronal hole segmentation.
    binary : bool, optional
        Indicates whether the input binary maps are binary (Default is True).
    
    Returns
    -------
    sunpy.map.Map
        A binary map representing the combined coronal hole regions.
    float
        The mean area of coronal holes.
    float
        The uncertainty of the mean area of coronal holes.
    numpy.ndarray
        An array containing the mean center of mass (lon, lat) of coronal holes.
    numpy.ndarray
        An array containing the uncertainty of the mean center of mass (lon, lat) of coronal holes.
    numpy.ndarray
        An array containing the mean extent (lon1, lon2, lat1, lat2) of coronal holes.
    numpy.ndarray
        An array containing the uncertainty of the mean extent (lon1, lon2, lat1, lat2) of coronal holes.
    """

    if np.sum(binmaps[0].data) == 0:
        return 0,0,0,0,0,0,0
    areas=np.zeros((5))
    com_pix=np.zeros((2,5))
    ext_pix=np.zeros((4,5))
    bdata=np.zeros((binmaps[0].data.shape[0],binmaps[0].data.shape[1]))
    coreg=curve_corr(binmaps[0])
    for i,binmap in enumerate(binmaps):
        print(np.sum(binmap.data))
        areas[i]=ch_area(binmap, binary =True, coreg=coreg)
        com=ndi.center_of_mass(binmap.data)
        com_pix[:,i]=com[1],com[0]
        
        cont,hier=cv2.findContours(image=binmap.data.astype('uint8'), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        x_c=(cont[0][:,0,0]+1-binmap.meta['crpix1'])*binmap.meta['cdelt1']
        y_c=(cont[0][:,0,1]+1-binmap.meta['crpix2'])*binmap.meta['cdelt2']
        eds= SkyCoord(x_c*u.arcsec,y_c*u.arcsec, frame = binmap.coordinate_frame)
        eds_lonlat=eds.transform_to(frames.HeliographicStonyhurst) 
        ext_pix[:,i]=np.nanmin(eds_lonlat.lon).value,np.nanmax(eds_lonlat.lon).value,np.nanmin(eds_lonlat.lat).value,np.nanmax(eds_lonlat.lat).value
        
        bdata += binmap.data
        

    x_coord=(com_pix[0,:]+1-binmap.meta['crpix1'])*binmap.meta['cdelt1']
    y_coord=(com_pix[1,:]+1-binmap.meta['crpix2'])*binmap.meta['cdelt2']
    com= SkyCoord(x_coord*u.arcsec,y_coord*u.arcsec, frame = binmap.coordinate_frame)
    com_lonlat=com.transform_to(frames.HeliographicStonyhurst)
    return sunpy.map.Map((bdata,binmap.meta)), np.nanmean(areas), catch_uncertainty(areas),  np.array([np.nanmean(com_lonlat.lon).value,np.nanmean(com_lonlat.lat).value]),np.array([catch_uncertainty(com_lonlat.lon.value),catch_uncertainty(com_lonlat.lat.value)]), (np.nanmean(ext_pix, axis=-1)),np.array([catch_uncertainty(ext_pix[0,:]), catch_uncertainty(ext_pix[1,:]), catch_uncertainty(ext_pix[2,:]), catch_uncertainty(ext_pix[3,:])])


#--------------------------------------------------------------------------------------------------
#get mean and median coronal hole intensity

def get_intensity(binmaps,map):
    """
    Calculate intensity properties of coronal holes from a list of binary maps.
    
    This function calculates the mean and median intensity properties of coronal holes from a list of binary maps representing different levels of coronal hole segmentation.
    
    Parameters
    ----------
    binmaps : list of sunpy.map.Map
        A list of binary maps representing different levels of coronal hole segmentation.
    map : sunpy.map.Map
        The original intensity map.
    
    Returns
    -------
    float
        The mean intensity of coronal holes.
    float
        The uncertainty of the mean intensity of coronal holes.
    float
        The median intensity of coronal holes.
    float
        The uncertainty of the median intensity of coronal holes.
    """

    if np.sum(binmaps[0].data) == 0:
        return 0,0,0,0
    
    imean=np.zeros((5))
    imed=np.zeros((5))
    
    for i,binmap in enumerate(binmaps):
        array=np.where(binmap.data == 1,binmap.data*map.data, np.nan)
        imean[i]=np.nanmean(array)
        imed[i]=np.nanmedian(array)

    return np.nanmean(imean), catch_uncertainty(imean),np.nanmean(imed), catch_uncertainty(imed)




#--------------------------------------------------------------------------------------------------
# calculate magnetic properties of  coronal hole 
def catch_mag(binmaps, magmap):
    """
    Wrapper to calculate magnetic properties of coronal holes from a list of binary maps and a magnetic field map.

    This function calculates various magnetic properties of coronal holes, including signed and unsigned mean magnetic flux density, signed and unsigned magnetic flux, and flux balance, from a list of binary maps representing different levels of coronal hole segmentation and a magnetic field map.

    Parameters
    ----------
    binmaps : list of sunpy.map.Map
        A list of binary maps representing different levels of coronal hole segmentation.
    magmap : sunpy.map.Map
        The magnetic field map.

    Returns
    -------
    float
        The mean signed magnetic flux density of coronal holes.
    float
        The uncertainty of the mean signed magnetic flux density of coronal holes.
    float
        The mean unsigned magnetic flux density of coronal holes.
    float
        The uncertainty of the mean unsigned magnetic flux density of coronal holes.
    float
        The mean signed magnetic flux of coronal holes.
    float
        The uncertainty of the mean signed magnetic flux of coronal holes.
    float
        The mean unsigned magnetic flux of coronal holes.
    float
        The uncertainty of the mean unsigned magnetic flux of coronal holes.
    float
        The mean flux balance of coronal holes.
    float
        The uncertainty of the mean flux balance of coronal holes.
    """
    
    if np.sum(binmaps[0].data) == 0:
        return 0,0,0,0,0,0,0,0,0,0
    
    bs=np.zeros((5))
    bus=np.zeros((5))
    fs=np.zeros((5))
    fus=np.zeros((5))
    fb=np.zeros((5))

    coreg=curve_corr(binmaps[0])
    
    for i,binmap in enumerate(binmaps):
        bs[i],bus[i],fs[i],fus[i]=ch_flux(binmap,magmap, coreg=coreg)
        
    fb=fs/fus
    
    return np.nanmean(bs), catch_uncertainty(bs),  np.nanmean(bus), catch_uncertainty(bus), np.nanmean(fs), catch_uncertainty(fs),  np.nanmean(fus), catch_uncertainty(fus), np.nanmean(fb), catch_uncertainty(fb)     

#--------------------------------------------------------------------------------------------------
# calculate the signed flux (Mx), area (10^10 km^2) and signed B (G) for binary map
def ch_flux(binmap, magmap, coreg=[0]):
    """
    Calculate magnetic properties of a coronal hole region from a binary map and a magnetic field map.

    This function calculates various magnetic properties of a coronal hole region, including signed mean magnetic flux density, unsigned mean magnetic flux density, signed magnetic flux, and unsigned magnetic flux, based on a binary map representing the coronal hole region and a magnetic field map.

    Parameters
    ----------
    binmap : sunpy.map.Map
        A binary map representing the coronal hole region.
    magmap : sunpy.map.Map
        The magnetic field map.
    coreg : list of float, optional
        An optional curvature correction factor for the map. If not provided, it will be computed internally.

    Returns
    -------
    float
        The mean signed magnetic flux density of the coronal hole region.
    float
        The mean unsigned magnetic flux density of the coronal hole region.
    float
        The signed magnetic flux of the coronal hole region.
    float
        The unsigned magnetic flux of the coronal hole region.
    """    
    rsun=map.meta['rsun_obs']
    arcsecTOkm = 1/rsun*696342
    data = np.where(binmap.data == binmap.data, 1, 0)
    
    if len(coreg) == 1:
        vcoreg = curve_corr(binmap)
    else: 
        vcoreg=coreg
        
    tmparea = data * binmap.meta['cdelt1'] * binmap.meta['cdelt2'] * vcoreg * arcsecTOkm**2
    fluxcoreg = data * binmap.meta['cdelt1'] * binmap.meta['cdelt2'] * vcoreg * arcsecTOkm**2 * 1e10 
    tmpflux = data * magmap.data * fluxcoreg * tmparea
    tmpflux_abs = data * np.abs(magmap.data) * fluxcoreg * tmparea
    
    return np.nansum(tmpflux/tmparea),np.nansum(tmpflux_abs/tmparea),np.nansum(tmpflux)/1e20,np.nansum(tmpflux_abs)/1e20 

