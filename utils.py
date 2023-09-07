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

import aiapy
import aiapy.calibrate as cal

from cv2 import morphologyEx as morph
from cv2 import MORPH_OPEN,MORPH_CLOSE, erode
import cv2

import numexpr as ne
import time

#--------------------------------------------------------------------------------------------------
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
    
    # t0=time.time()

    # coreg=ne.evaluate('1/cos(arcsin(sqrt( xc**2 + yc**2)/rsun))')
    # t1=time.time()
    # dist=ne.evaluate('sqrt( xc**2 + yc**2)')
    # dist[dist >=rsun] = np.nan
    # coreg2=ne.evaluate('1/cos(arcsin(dist/rsun))')
    # t2=time.time()
    # dist = np.sqrt( coords.Tx**2 + coords.Ty**2)
    # dist[dist.value  >= rsun] = np.nan
    # angle = np.arcsin(dist.value/rsun)
    # coreg3 = 1./np.cos(angle)
    # t3=time.time()
    
    # coreg4=ne.evaluate('1/sqrt(1 - (xc**2 + yc**2)/rsun**2)')
    
    # t4=time.time()
    # dist=ne.evaluate('sqrt( xc**2 + yc**2)')
    # dist[dist >= rsun] = np.nan
    # coreg5=ne.evaluate('1/sqrt(1-(dist**2/rsun**2))')
    # t5=time.time()
    # dist = np.sqrt( coords.Tx**2 + coords.Ty**2)
    # dist[dist.value  >= rsun] = np.nan
    # coreg6 = 1./np.sqrt(1- dist.value**2/rsun**2)
    # t6=time.time()
    
    # print(t1-t0,t2-t1,t3-t2,t4-t3,t5-t4,t6-t5)
    # print(coreg2-coreg,coreg3-coreg,coreg4-coreg,coreg5-coreg,coreg6-coreg)
    #return coreg

#--------------------------------------------------------------------------------------------------
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
    if binary == True:
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
#--------------------------------------------------------------------------------------------------
# region grow for CH extraction
class Point(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y
        
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y

def getGrayDiff(img, currentPoint, tmpPoint, absolute=True):
    if absolute == True:
        res = img[tmpPoint.x,tmpPoint.y]
    else:
        res=abs((img[currentPoint.x,currentPoint.y]) - (img[tmpPoint.x,tmpPoint.y]))
    return res
        
def connectivity(p):
    if p == 4:
        connects = [Point(0,-1),Point(-1,0),Point(1,0),Point(0,1)]
    elif p == 8:
        connects = [Point(-1,-1),Point(0,-1),Point(1,-1),Point(-1,0),Point(1,0),Point(-1,1),Point(0,1),Point(1,1)]
    else:
        print('Error: No %s connectivity implemented!')%p
    
    return connects

def region_grow(img, seeds, thresh, p=8, absolute=True):
    
    
    height, weight = img.shape
    seedMark=np.zeros(img.shape)
    seedList=[]
    
    for seedi in seeds:
        seed= Point(int(seedi[1]),int(seedi[0]))
        seedList.append(seed)
    
    label=1
    connects = connectivity(p)
    
    while(len(seedList)>0):
        currentPoint=seedList.pop(0)
        
        if len(thresh) == 2:
            if img[currentPoint.x,currentPoint.y] <= thresh[0] or img[currentPoint.x,currentPoint.y] >= thresh[1]:
                continue
        else:                    
            if img[currentPoint.x,currentPoint.y] <= thresh[0]:
                continue
            
        seedMark[currentPoint.x,currentPoint.y] = label
        
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img,currentPoint, Point(tmpX,tmpY), absolute)
            
            if len(thresh) == 2:
                if grayDiff > thresh[0] and grayDiff < thresh[1] and seedMark[tmpX,tmpY] == 0:
                    seedMark[tmpX,tmpY] = label
                    seedList.append(Point(tmpX,tmpY))            
            else:                    
                if grayDiff < thresh and seedMark[tmpX,tmpY] == 0:
                    seedMark[tmpX,tmpY] = label
                    seedList.append(Point(tmpX,tmpY))
    return seedMark

#--------------------------------------------------------------------------------------------------
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
            
             #bindata=cv2.fillPoly(bindata,pts=ct,color=1)
             bindata=cv2.fillPoly(bindata,pts=[ct],color=1)
             
             if hier[0][n][0] == -1:
                 num=len(cont)-n -1
             else:
                num=hier[0][n][0] -n -1
            
             #print(num)
                            
             for nchild in range(num):
                 child=cont[nchild+n+1]
                 bindata=cv2.fillPoly(bindata,pts=[child],color=0)
                 bindata=cv2.fillPoly(bindata,pts=child,color=1)
             break

        
    #bindata = region_grow(data, seed, [0.5,1.5])
    
    return sunpy.map.Map((bindata,map.meta))

#--------------------------------------------------------------------------------------------------
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

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# uncertainty definition for catch properties

def catch_uncertainty(x):
    return np.nanmax( np.sqrt( (x-np.nanmean(x))**2 ) )
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# load and prep aia image

def load_aia(file, deconvolve = False, alc = False, degr = True, cut_limb = False):
    
    #load image into map    
    inp=sunpy.map.Map(file)
    
    #deconvolve image
    if deconvolve == True:
        inp_dec = aiapy.psf.deconvolve(inp)
    else:
        inp_dec = inp
        
    #register map
    inp_reg=cal.register(inp_dec)
    
    #normalize map
    inp_norm=cal.normalize_exposure(inp_reg)
    
    #correct for instrument degratation
    if degr == True:
        deg= cal.degradation(193*u.angstrom,astropy.time.Time(inp_norm.meta['date-obs'],scale='utc') )
        deg_data=inp_norm.data / deg
        deg_data[deg_data < 1] = 1
        inp_corr=sunpy.map.Map(((deg_data),inp_norm.meta))
    else:
        inp_corr=inp_norm
    
    # annulus limb correction
    if alc == True:
        inp_alc=annulus_limb_correction(inp_corr)
    else:
        inp_alc=inp_corr
        
    #cut limb
    if cut_limb == True:
        hpc_coords=all_coordinates_from_map(inp_alc)
        mask=coordinate_is_on_solar_disk(hpc_coords)
        data=np.where(mask == True, inp_alc.data, np.nan)
        output=sunpy.map.Map((data,inp_alc.meta))
    else:
        output=inp_corr
    
    return output