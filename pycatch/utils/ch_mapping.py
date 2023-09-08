import os, sys

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolor

import astropy.units as u

import scipy.ndimage as ndi

import sunpy
import sunpy.map
import sunpy.util.net
from sunpy.coordinates import frames

from astropy.coordinates import SkyCoord
import astropy.constants as const
from astropy.io import fits
import astropy.time
from sunpy.map.maputils import all_coordinates_from_map,coordinate_is_on_solar_disk
import copy


from cv2 import morphologyEx as morph
from cv2 import MORPH_OPEN,MORPH_CLOSE
import cv2

from joblib import Parallel, delayed
import numexpr as ne
import time

import pycatch.utils.extensions as ext

#--------------------------------------------------------------------------------------------------
# find minimum intensity around seed to speed up calculation of curves
def min_picker(seed, data):
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
# calculate area of binmaps for curve
def calc_area_curves(map, th,kernel, seed,minval, coreg):
    if th < minval:
        return np.nan
    binmap=extract_ch(map,th, kernel, seed)
    return ch_area(binmap,coreg)

#--------------------------------------------------------------------------------------------------
#calculate area for all thrs
def get_curves(map,seed,kernel=None, upper_lim=False, cores = 8):
    if upper_lim:
        rng=np.arange(0,int(upper_lim))
    else:
        hpc_coords=all_coordinates_from_map(map)
        mask=coordinate_is_on_solar_disk(hpc_coords)
        data=np.where(mask == True, map.data, np.nan)
        data_median = np.nanmedian(data)
        rng=np.arange(0,1.5*int(data_median))
    
    if kernel is None:
        indx= ext.find_nearest((np.arange()+1)*0.6, map.meta['cdelt1'])
        kernels=2**np.arange(5)[::-1]
        kernel=kernels[indx]
            
    minval=min_picker(seed, map.data) 
    coreg=curve_corr(map)
    output=Parallel(n_jobs=cores)(delayed(calc_area_curves)(map, th,kernel, seed,minval,coreg) for th in rng)
    
    uncertainty=[]
    for xpos in range(len(rng)-4):
        uncertainty.append(catch_uncertainty(output[xpos+2:xpos+7]))
        
    return rng,np.array(output),np.array(uncertainty)
        
#--------------------------------------------------------------------------------------------------
# extract coronal hole from EUV image based on intensity threhold and SEED point
def extract_ch(map,thr, kernel, seed):
    
    data=np.where(map.data < thr, 1, 0)
    data=data.astype(np.float32)
    kern=np.ones((kernel,kernel), np.uint8)
    data=morph(data,MORPH_CLOSE,kern)
    data=morph(data,MORPH_OPEN,kern)
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
    coords = all_coordinates_from_map(map)
    if map.meta['telescop'] == 'STEREO':
        rsun=map.meta['rsun']
    else:
        rsun=map.meta['r_sun']*map.meta['cdelt1']
    xc=coords.Tx.value 
    yc=coords.Ty.value 
    return ne.evaluate('1/sqrt(1 - (xc**2 + yc**2)/rsun**2)')

#--------------------------------------------------------------------------------------------------
# calculate area of binary map in 10^10 km^2
def ch_area(map, coreg=[0], binary = False):
    if np.nansum(map.data) == 0:
        return 0.
    if map.meta['telescop'] == 'STEREO':
        rsun=map.meta['rsun']
    else:
        rsun=map.meta['r_sun']*map.meta['cdelt1']
    arcsecTOkm = 1/rsun*696342
    if binary:
        data=map.data
    else:
        data = np.where(map.data == map.data, 1, 0)
    if len(coreg) == 1:
        vcoreg = curve_corr(map)
    else: 
        vcoreg=coreg
    tmparea = data * map.meta['cdelt1'] * map.meta['cdelt2'] * vcoreg * arcsecTOkm**2
    return np.nansum(tmparea)/1e10

#--------------------------------------------------------------------------------------------------

# uncertainty definition for catch properties

def catch_uncertainty(x):
    return np.nanmax( np.sqrt( (x-np.nanmean(x))**2 ) )


#--------------------------------------------------------------------------------------------------
# extract coronal hole from EUV image based on intensity threhold and SEED point and calculate the CATCH area and uncertainty and properties.
# output as  A, dA, (lon,lat), (dlon,dlat)

def catch_calc(binmaps, binary =True):
    if np.sum(binmaps[0].data) == 0:
        return 0,0,0,0,0,0,0
    areas=np.zeros((5))
    com_pix=np.zeros((2,5))
    ext_pix=np.zeros((4,5))
    bdata=np.zeros((binmaps[0].data.shape[0],binmaps[0].data.shape[1]))
    coreg=curve_corr(binmaps[0])
    for i,binmap in enumerate(binmaps):
        areas[i]=ch_area(binmap, binary =True, coreg=coreg)
        com=ndi.center_of_mass(binmap.data)
        com_pix[:,i]=com[1],com[0]
        
        cont,hier=cv2.findContours(image=binmap.data.astype('uint8'), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        x_c=(cont[0][:,0,0]+1-map.meta['crpix1'])*map.meta['cdelt1']
        y_c=(cont[0][:,0,1]+1-map.meta['crpix2'])*map.meta['cdelt2']
        eds= SkyCoord(x_c*u.arcsec,y_c*u.arcsec, frame = binmap.coordinate_frame)
        eds_lonlat=eds.transform_to(frames.HeliographicStonyhurst) 
        ext_pix[:,i]=np.nanmin(eds_lonlat.lon).value,np.nanmax(eds_lonlat.lon).value,np.nanmin(eds_lonlat.lat).value,np.nanmax(eds_lonlat.lat).value
        
        bdata += binmap.data
        

    x_coord=(com_pix[0,:]+1-map.meta['crpix1'])*map.meta['cdelt1']
    y_coord=(com_pix[1,:]+1-map.meta['crpix2'])*map.meta['cdelt2']
    com= SkyCoord(x_coord*u.arcsec,y_coord*u.arcsec, frame = binmap.coordinate_frame)
    com_lonlat=com.transform_to(frames.HeliographicStonyhurst)
    
    return sunpy.map.Map((bdata,map.meta)), np.nanmean(areas), catch_uncertainty(areas),  np.array([np.nanmean(com_lonlat.lon).value,np.nanmean(com_lonlat.lat).value]),np.array([catch_uncertainty(com_lonlat.lon.value),catch_uncertainty(com_lonlat.lat.value)]), (np.nanmean(ext_pix, axis=-1)),np.array([catch_uncertainty(ext_pix[0,:]), catch_uncertainty(ext_pix[1,:]), catch_uncertainty(ext_pix[2,:]), catch_uncertainty(ext_pix[3,:])])


#--------------------------------------------------------------------------------------------------
#get mean and median coronal hole intensity

def get_intensity(binmaps,map):
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
# calculate the signed flux (Mx), area (10^10 km^2) and signed B (G) for binary map
def ch_flux(binmap, magmap, coreg=[0]):
    if binmap.meta['telescop'] == 'STEREO':
        rsun=binmap.meta['rsun']
    else:
        rsun=binmap.meta['r_sun']*binmap.meta['cdelt1']
    arcsecTOkm = 1/rsun*696342
    data = np.where(binmap.data == binmap.data, 1, 0)
    
    if len(coreg) == 1:
        vcoreg = curve_corr(binmap)
    else: 
        vcoreg=coreg
        
    tmparea = data * binmap.meta['cdelt1'] * binmap.meta['cdelt2'] * vcoreg * arcsecTOkm**2
    fluxcoreg = data * binmap.meta['cdelt1'] * binmap.meta['cdelt2'] * vcoreg * arcsecTOkm**2 * 1e10 
    tmpflux = data * magmap.data * fluxcoreg * tmparea
    return np.nansum(tmpflux),np.nansum(tmparea)/1e10, np.nansum(tmpflux/tmparea)


#--------------------------------------------------------------------------------------------------
# extract coronal hole from EUV image based on intensity threhold and SEED point and calculate the CATCH area and uncertainty and properties.
# output as  A, dA, (lon,lat), (dlon,dlat)

def catch_calc(map, thr, kernel, seed, binary =True):
    binmaps = [extract_ch(map, thr+i, kernel,seed) for i in np.arange(5)-2]
    if np.sum(binmaps[0].data) == 0:
        return 0,0,0,0,0,0,0
    areas=np.zeros((5))
    com_pix=np.zeros((2,5))
    ext_pix=np.zeros((4,5))
    bdata=np.zeros((binmaps[0].data.shape[0],binmaps[0].data.shape[1]))
    coreg=curve_corr(binmaps[0])
    for i,binmap in enumerate(binmaps):
        areas[i]=ch_area(binmap, binary =True, coreg=coreg)
        com=ndi.center_of_mass(binmap.data)
        com_pix[:,i]=com[1],com[0]
        
        cont,hier=cv2.findContours(image=binmap.data.astype('uint8'), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        
        #edges = binmap.data- erode(binmap.data,np.ones((5,5),np.uint8))
        #cords= np.where(edges == 1)
        # x_c=(cords[1]+1-map.meta['crpix1'])*map.meta['cdelt1']
        # y_c=(cords[0]+1-map.meta['crpix2'])*map.meta['cdelt2']
        x_c=(cont[0][:,0,0]+1-map.meta['crpix1'])*map.meta['cdelt1']
        y_c=(cont[0][:,0,1]+1-map.meta['crpix2'])*map.meta['cdelt2']
        eds= SkyCoord(x_c*u.arcsec,y_c*u.arcsec, frame = binmap.coordinate_frame)
        eds_lonlat=eds.transform_to(frames.HeliographicStonyhurst) 
        ext_pix[:,i]=np.nanmin(eds_lonlat.lon).value,np.nanmax(eds_lonlat.lon).value,np.nanmin(eds_lonlat.lat).value,np.nanmax(eds_lonlat.lat).value
        
        bdata += binmap.data
        

    x_coord=(com_pix[0,:]+1-map.meta['crpix1'])*map.meta['cdelt1']
    y_coord=(com_pix[1,:]+1-map.meta['crpix2'])*map.meta['cdelt2']
    com= SkyCoord(x_coord*u.arcsec,y_coord*u.arcsec, frame = binmap.coordinate_frame)
    com_lonlat=com.transform_to(frames.HeliographicStonyhurst)
    
    hpc_coords=all_coordinates_from_map(map)
    return sunpy.map.Map((bdata,map.meta)), np.nanmean(areas), catch_uncertainty(areas),  np.array([np.nanmean(com_lonlat.lon).value,np.nanmean(com_lonlat.lat).value]),  np.array([catch_uncertainty(com_lonlat.lon.value),catch_uncertainty(com_lonlat.lat.value)]),(np.nanmean(ext_pix, axis=-1)),np.array([catch_uncertainty(ext_pix[0,:]), catch_uncertainty(ext_pix[1,:]), catch_uncertainty(ext_pix[2,:]), catch_uncertainty(ext_pix[3,:])])