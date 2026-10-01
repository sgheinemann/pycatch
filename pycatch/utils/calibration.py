import os, sys

import numpy as np

from importlib import resources
import urllib.request
from pathlib import Path
from tqdm import tqdm
import copy
import matplotlib.colors as colors


import astropy.units as u
from astropy.nddata import CCDData

import sunpy
import sunpy.map
import sunpy.util.net
from sunpy.map.maputils import all_coordinates_from_map,coordinate_is_on_solar_disk


import aiapy.psf
import aiapy.calibrate as cal
from aiapy.calibrate.utils import get_pointing_table, get_correction_table


#--------------------------------------------------------------------------------------------------
#prep aia image
def calibrate_aia(map, register= True, pointing = True, respike= True, normalize = True, deconvolve = False, psf_model = 'SJH', alc = True, degradation = True, cut_limb = True):
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
    header=copy.deepcopy(calmap.meta)
    
    #respike image
    if respike:
        calmap = cal.respike(calmap)
        header['respike'] = 'Yes'
    
    #pointing correction
    if pointing:
        pointing_table = get_pointing_table("JSOC", time_range=(calmap.date- 12 * u.h, calmap.date + 12 * u.h))
        calmap = cal.update_pointing(calmap, pointing_table=pointing_table)
        header['pointng'] = 'Yes'
    
        
    #deconvolve image
    if deconvolve == True:
        if psf_model == 'SJH':
            psf =load_psf(f'psf_aia_{int(calmap.wavelength.value)}.fits') 
            #return

            aia_dec = deconvolve_bid(calmap.data, psf, large_psf = True, iterations = 20)

            calmap.data[:]=aia_dec
            header['decon'] = 'SJH'
            
        elif psf_model == 'AIA':
            psf = aiapy.psf.psf(calmap.wavelength)
            calmap = aiapy.psf.deconvolve(calmap,psf=psf)
            header['decon'] = 'AIA'
        else:
            print('> pycatch ## NO PSF MODEL SELECTED (pycatch supports SJH and AIA) ##')    
        
    #correct for instrument degratation
    if degradation:
        deg= cal.degradation(map.meta['wavelnth']*u.angstrom,map.date, correction_table=get_correction_table("jsoc") )
        deg_data=calmap.data / deg
        deg_data[deg_data < 1] = 1
        calmap=sunpy.map.Map(((deg_data),calmap.meta))
        header['degcorr'] = 'Yes'
        
    #register map
    if register:
        calmap = cal.register(calmap)
        header['register'] = 'Yes'
        
    #normalize map
    if normalize:
        calmap.data[:] = calmap.data[:]/calmap.exposure_time
        header['exptime'] = 1

    
    # annulus limb correction
    if alc:
        calmap=annulus_limb_correction(calmap)
        header['alc'] = 'Yes'

        
    #cut limb
    if cut_limb:
        hpc_coords=all_coordinates_from_map(calmap)
        mask=coordinate_is_on_solar_disk(hpc_coords)
        data=np.where(mask == True, calmap.data, np.nan)
        calmap=sunpy.map.Map((data,calmap.meta))
        header['rem_limb'] = 'Yes'
    
    return sunpy.map.Map((calmap.data,header))

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
def calibrate_hmi(map,intensity_map, rotate= False, align = True, cut_limb = True):
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
        Whether to rotate the HMI map to have north up. Default is False. Overridden by align = True
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
    
    #align with aia map
    if align:
        calmap= calmap.reproject_to(intensity_map.wcs)
    elif rotate:
        calmap = calmap.rotate(order=3)
    
    # 2. Fix potential CUNIT/CDELT unit mismatch if WCS forced it to degrees
    if calmap.meta.get('cunit1', '').lower() == 'deg':
        calmap.meta['cunit1'] = 'arcsec'
        calmap.meta['cunit2'] = 'arcsec'
        calmap.meta['cdelt1'] *= 3600.0
        calmap.meta['cdelt2'] *= 3600.0
        
    #cut limb
    if cut_limb:
        hpc_coords=all_coordinates_from_map(calmap)
        mask=coordinate_is_on_solar_disk(hpc_coords)
        data=np.where(mask == True, calmap.data, np.nan)
        calmap=sunpy.map.Map((data,calmap.meta))
    
    calmap.plot_settings['norm']= colors.Normalize(vmin=-100, vmax=100)
    return calmap



def get_psf_cache_dir() -> Path:
    r"""
    Returns the OS-appropriate user cache directory for pycatch:
    - Linux: ~/.cache/pycatch/psf/
    - macOS: ~/Library/Caches/pycatch/psf/
    - Windows: C:\Users\<User>\AppData\Local\pycatch\Cache\psf\
    """
    if os.name == "nt":  # Windows
        base_cache = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif os.uname().sysname == "Darwin":  # macOS
        base_cache = Path.home() / "Library" / "Caches"
    else:  # Linux / Unix
        base_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))

    cache_dir = base_cache / "pycatch" / "psf"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_psf_filepath(filename: str) -> Path:
    """
    Locates or downloads a PSF file into the OS-specific user cache directory 
    using its direct Dataverse File ID.
    """
    DATAVERSE_SERVER = "https://dataverse.harvard.edu"
    PSF_FILE_IDS = {
        "psf_aia_193.fits": "10615234",
        "psf_aia_131.fits": "10615236",
        "psf_aia_171.fits": "10615233",
        "psf_aia_211.fits": "10615238",
        "psf_aia_304.fits": "10615237",
        "psf_aia_335.fits": "10615235",
        "psf_aia_94.fits": "10615239",
    }

    if filename not in PSF_FILE_IDS:
        raise ValueError(
            f"> pycatch ## Unknown filename for psf '{filename}'. "
            f"> pycatch ## Available files: {list(PSF_FILE_IDS.keys())}"
        )

    # OS-independent cache directory resolution
    cache_dir = get_psf_cache_dir()
    psf_path = cache_dir / filename
    tmp_path = cache_dir / f"{filename}.tmp"
    
    file_id = PSF_FILE_IDS[filename]
    download_url = f"{DATAVERSE_SERVER}/api/access/datafile/{file_id}"

    req = urllib.request.Request(
        download_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept-Encoding": "identity",
        }
    )

    try:
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("Content-Length", 0))

            # 1. Skip download if complete file already exists
            if psf_path.is_file():
                if total_size == 0 or psf_path.stat().st_size == total_size:
                    return psf_path
                print(f"> pycatch ## Incomplete file detected for {filename}. Re-downloading...")
                psf_path.unlink()

            print(f"> pycatch ## Downloading {filename} (ID: {file_id}) to {tmp_path}...")
            chunk_size = 1024 * 1024  # 1 MB buffer

            # 2. Write to temporary .tmp file to prevent corrupting local state if interrupted
            with open(tmp_path, "wb") as f, tqdm(
                desc=filename,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=True,
            ) as bar:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))

            # 3. Rename .tmp -> .fits only on 100% completion
            tmp_path.replace(psf_path)

    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"> pycatch ## Download failed for '{filename}': {e}") from e

    return psf_path


def load_psf(filename: str) -> CCDData:
    """
    Fetches (if missing) and loads a PSF file into a CCDData object.
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DYT4ZL

    Example:
        psf = load_psf("psf_aia_193.fits")
    """
    psf_file = get_psf_filepath(filename)
    return CCDData.read(psf_file)



###################################################################################### PSF code below

"""
Created on Tue Feb  7 12:06:28 2023

@author: sjh
"""

def rebin_psf(psf, dimension):
    """
    Rebin the PSF to smaller dimensions

    If one wants to deconvolve a rebinned image with the PSF, the PSF has to be rebinned accordingly. However, the definition of the center of the PSF is not in the center of the 2d PSF array, but shifted by definition by half a pixel. This function accounts for this shift for rebinning the PSF.

    Parameters
    ----------
    psf : `~numpy.ndarray`
        The point spread function. 
    dimensions : [xdim, ydim]
        The new size of the PSF. 
    
    Returns
    -------
    np.ndarray
        The the rebinned PSF

   """
   
    psf = np.copy(psf)
    psf_rebinned = np.zeros(dimension)
    
    shape_fac = (psf.shape / np.array(dimension)).astype(np.int)
    dx_l = shape_fac[0]//2
    dy_l = shape_fac[1]//2
    
    for dx in range(dx_l):
        psf = np.append(np.array([psf[:, -dx-1]]).transpose(), psf, axis = 1)  #add an extra column, so that the summation is eased. Adding the values of the last column is technically not absolute correct, but there is no exact solution to this process. Using the last column conserves the total PSF weight.
    for dy in range(dy_l):
        psf = np.append([psf[-dy-1, :]], psf, axis = 0)
        
    for dx in range(-dx_l+1, dx_l):
      for dy in range(-dy_l+1, dy_l):
          psf_rebinned += psf[dx_l+dx:-(dx_l-dx):shape_fac[0], dy_l+dy:-(dy_l-dy):shape_fac[1]]
    
    for dy in range(-dy_l+1, dy_l):
          psf_rebinned += 0.5*psf[0:-2*dx_l:shape_fac[0], dy_l+dy:-(dy_l-dy):shape_fac[1]]
          psf_rebinned += 0.5*psf[2*dx_l::shape_fac[0], dy_l+dy:-(dy_l-dy):shape_fac[1]]
    for dx in range(-dx_l+1, dx_l):
          psf_rebinned += 0.5*psf[dx_l+dx:-(dx_l-dx):shape_fac[0], 0:-2*dy_l:shape_fac[1]]
          psf_rebinned += 0.5*psf[dx_l+dx:-(dx_l-dx):shape_fac[0], 2*dy_l::shape_fac[1]]
     
    psf_rebinned += 0.25*psf[0:-2*dx_l:shape_fac[0], 0:-2*dy_l:shape_fac[1]]
    psf_rebinned += 0.25*psf[2*dx_l::shape_fac[0], 0:-2*dy_l:shape_fac[1]]
    psf_rebinned += 0.25*psf[0:-2*dx_l:shape_fac[0], 2*dy_l::shape_fac[1]]
    psf_rebinned += 0.25*psf[2*dx_l::shape_fac[0], 2*dy_l::shape_fac[1]]
    
    return psf_rebinned





"""© Stefan Hofmeister
Removed cupy dependency for use on macOS: Stephan Heinemann 22.05.2026
"""



def deconvolve_bid(img, psf, iterations = 25, tolerance = .1, mask = None, large_psf = False, pad = True, estimate_background = True, constrain_positive = True):
    r"""
    Deconvolve an image with the point spread function.
    
    Perform image deconvolution on an image with the instrument
    point spread function using the bid algorithm published in 
    Hofmeister et al. (2024), The Basic Iterative Deconvolution: A Fast Instrumental Point-Spread Function Deconvolution Method That Corrects for Light That Is Scattered Out of the Field of View of a Detector, Solar Physics, Volume 299, Issue 6, article id.77
    https://ui.adsabs.harvard.edu/abs/2023arXiv231211784H/abstract
    
    Parameters
    ----------
    img : `numpy.ndarray`
        A 2D array representing an image.
    psf : `numpy.ndarray`
        The point spread function.
    iterations : `int`
        Maximum number of iterations.
    tolerance : `float`
        The image deconvolution stops when the maximum change from all pixels between the simulated observed image and the observed image is less than TOLERANCE counts.
    mask : `numpy.ndarray`, optional
        Allows selecting an image subregion for which the convolution is done. By that, the algorithm can massively speed up. Can be either a 1D array containing the four elements [left, bottom, right, top], or a 2D array masking the pixels that shall be deconvolved. If the 2D mask is used, the boundaries of the 1D array will be calculated from it, i.e., all pixels in a corresponding rectangular box will be deconvolved.
        At the moment, the dimensions of mask have to be smaller than half of the dimensions of the image. If you need to deconvolve a larger region, deconvolve the entire image instead.
    estimate_background : `bool`, default=True
        If a subregion deconvolution is used, it determines if an incoming scattered light estimate from the surrounding region to the subregion should be applied. This increases the fidelity of the result, but costs some computation time. Generally, it is required for average image intensities and below, but is not required for deconvolving bright image regions.
    pad : `bool`, default=True
        If True, increase the size of both the PSF and the image by a factor of two, and pad the PSF and image accordingly with zeros. As this is a Fourier-based method, this breaks the symmetric boundary conditions involved in the Fourier transform.
    large_psf : `bool`, default=False
        Usually, the PSF has the same dimension as the image, restricting scattered light to half of the image size. If set to True, the PSF given to the deconvolution has to be double the image size (that allows scattering over the full image range). The image will be padded with zeros to match the size of the full PSF, and deconvolution is done over the full PSF.
    constrain_positive : `bool`, default=True
        Constrain the deconvolution to positive result intensities. If True, it mitigates small ringing artifacts. If False, allow negative intensities in the reconstructions. Negative intensities are informative, as they can indicate issues (image calibration artifacts, slightly inaccurate PSF, ringing, etc.).
    
    Returns
    -------
    `sunpy.map.Map`
        Deconvolved image map.
    """
    #At the moment, the mask option only works if the shape of the selected region is smaller than 0.5 * the shape of the image
        
    #if mask is provided as a list, convert it to a 1d-array
    if isinstance(mask, list) or isinstance(mask, tuple): mask = np.array(mask)
    
    #this factor determines the speed of convergence, and should be set between [0.1, 1.0]
    k = 1.
    
    #createa a copy of the image and psf
    img = np.copy(img)
    psf = np.copy(psf)
    
    #for a psf deconvolution, the length of the axis should be even. Thus, if the mask provieded is odd, make it even by adding one row and/or columng
    if isinstance(mask, np.ndarray):
        bd_box_img, bd_box_psf, bd_box_makeeven = get_boundary_boxes(mask, img, psf)
        if estimate_background == True:
            #derive the scattered light of the surrounding into the boundary box, and correct the image for it.
            background = estimate_scattered_light(img, psf, bd_box_img, large_psf = large_psf, pad = pad)        
            img = img - background 
        #cut the image and psf. Since they are cut, large_psf looses its meaning and thus is set to zero
        img = img[bd_box_img[0] : bd_box_img[1] + 1, bd_box_img[2] : bd_box_img[3] + 1]
        psf = psf[bd_box_psf[0] : bd_box_psf[1] + 1, bd_box_psf[2] : bd_box_psf[3] + 1]  
        large_psf = True #If we cut the image, we definitively want to include scattered light over the entire subimage. Thus, large_psf is set to True. The psf boundary box has already been cut before accordingly.

    

    #before the psf deconvolution, we have to pad the image and psf with zeros to break the periodic boundary conditions for the convolution in the fourier domain            
    if pad == True:
        img, psf = pad_img_psf(img, psf, large_psf = large_psf, constant_values = np.nan)
        pad_mask = np.isnan(img)  #the pad mask is required in the next loop - at each iteration, the padding has to be restored.
        img[pad_mask] = 0.
        psf[np.isnan(psf)] = 0.

               

    
    #derive the fourier transform of the psf
    psf = np.roll(np.roll(psf, psf.shape[0]//2, axis=0),
                      psf.shape[1]//2,
                      axis=1)
    psf = np.fft.rfft2(psf) 
        
    img_decon = np.copy(img)
    tolerance=[tolerance]
    tolerance = np.array([tolerance])
    pbar = tqdm(total=iterations, desc="Deconvolving AIA Image", unit="iter")
    try:
        for n_iter in range(iterations):
            img_decon_last = np.copy(img_decon)
            #derive the foureir transform of the approximated deconvolved image
            img_decon_con = np.fft.rfft2(img_decon)
            #convolve it with the psf
            img_decon_con = img_decon_con * psf
            #and transform it back to the spatial domain
            img_decon_con = np.fft.irfft2(img_decon_con)
            img_decon_con[pad_mask]  = 0.
            
            #derive how far we are off between the deconvolved imaged convolved with the psf and the observed image, i.e., how consistent we are
            deviations = img_decon_con - img
            #and adjust the approximated deconvolved image accordingly
            img_decon -=  k * deviations
            if constrain_positive == True: img_decon[img_decon < 0] = 0
            pbar.update(1)
            #if the deconvolved image has converged, end the iterations
            dev =  np.max(np.abs(img_decon - img_decon_last))
            if dev <= tolerance[0]: break
    finally:
        pbar.close()
    
    #undo the padding
    if pad == True:
        img_decon, psf = unpad_img_psf(img_decon, psf, large_psf = large_psf)
    #if we had to enlarge the FOV of the mask to get an even pixel length of the axis, shrink the image again                
    if isinstance(mask, list) or isinstance(mask, tuple):
        img_decon = img_decon[0 - bd_box_makeeven[0] : img_decon.shape[0] - bd_box_makeeven[1] + 1,
                              0 - bd_box_makeeven[2] : img_decon.shape[1] - bd_box_makeeven[3] + 1]
        
    #and we are done
    img_decon = img_decon.astype(img.dtype)
    return img_decon
           


def deconvolve_richardson_lucy(img, psf, iterations=25, pad = True, large_psf = False, psf_min = 0):
    """
    Deconvolve an image with the point spread function

    Perform image deconvolution on an image with the instrument
    point spread function using the Richardson-Lucy deconvolution
    algorithm

    Parameters
    ----------
    img : 'numpy 2d array'
        An image.
    psf : `~numpy.ndarray`
        The point spread function. 
    iterations: `int`
        Number of iterations in the Richardson-Lucy algorithm
    pad: True/False
        If true, increase the size of both the psf and the image by a factor of two, and pad the psf and image accordingly with zeros. As this is a fourier-based method, this breaks the symmetric boundary conditions involved in the fourier transform.
    large_psf: True/False
        Usually, the PSF has the same dimension as the image, restricting scattered light to half of the image size. If set to true, the PSF given to the deconvolution has to be double the image size (that allows scattering over the full image range). The image will be padded with zeros to match the size of the full psf, and deconvolution is done over the full psf.

    Returns
    -------
    `~sunpy.map.Map`
        Deconvolved image

    Comments:
        Based on the aiapy.deconvolve method, as described in Cheung, M., 2015, *GPU Technology Conference Silicon Valley*, `GPU-Accelerated Image Processing for NASA's Solar Dynamics Observatory <https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=s5209-gpu-accelerated+imaging+processing+for+nasa%27s+solar+dynamics+observatory>`_
    """
    img, psf = np.copy(img), np.copy(psf)
    im_size = img.shape[0]
    psf_size = psf.shape[0]
    padsize_pad, padsize_large_psf = int(0.25*im_size), int(0.5*im_size)
    
    if large_psf:
        img = np.pad(img, padsize_large_psf)
        img[img == 0] = np.finfo(img.dtype).tiny
        im_size = im_size +2*padsize_large_psf
                 
    #padding is only required if the PSF is not large_psf. Else, the padding of the image has already be done above in the large_psf block.
    if pad and not large_psf:  
        psf, img = np.pad(psf, padsize_pad), np.pad(img, padsize_pad)
        im_size = im_size +2*padsize_pad
        psf_size = psf_size +2*padsize_pad

        
    # Center PSF at pixel (0,0)
    psf = np.roll(np.roll(psf, psf.shape[0]//2, axis=0),
                  psf.shape[1]//2,
                  axis=1)
    
    # Convolution requires FFT of the PSF
    psf = np.fft.rfft2(psf)
    psf_conj = psf.conj()

    img_decon = np.copy(img)
    for _ in range(iterations):
        ratio = img/np.fft.irfft2(np.fft.rfft2(img_decon)*psf)
        img_decon = img_decon*np.fft.irfft2(np.fft.rfft2(ratio)*psf_conj)


    
    if large_psf:
        img_decon = img_decon[padsize_large_psf : im_size - padsize_large_psf, padsize_large_psf : im_size - padsize_large_psf]
    
    if pad and not large_psf:
        img_decon = img_decon[padsize_pad : im_size - padsize_pad, padsize_pad : im_size - padsize_pad]
                    
    img_decon = img_decon.astype(img.dtype)
    
    return img_decon


def convolve_image(img, psf, pad = True, large_psf = False):
    img = np.copy(img)
    psf = np.copy(psf)
    im_size = img.shape[0]
    psf_size = psf.shape[0]
    padsize_pad, padsize_large_psf = int(0.25*im_size), int(0.5*im_size)
    
    if large_psf:
        img = np.pad(img, padsize_large_psf)
        img[img == 0] = np.finfo(img.dtype).tiny
        im_size = im_size +2*padsize_large_psf
    
    if pad and not large_psf:  
        psf, img = np.pad(psf, padsize_pad), np.pad(img, padsize_pad)
        im_size = im_size +2*padsize_pad
        psf_size = psf_size +2*padsize_pad
    
        
    # Center PSF at pixel (0,0)
    psf = np.roll(np.roll(psf, psf.shape[0]//2, axis=0),
                  psf.shape[1]//2,
                  axis=1)
    # Convolution requires FFT of the PSF
    psf = np.fft.rfft2(psf)
    img_con = np.fft.rfft2(img)
    img_con = img_con * psf
    img_con = np.fft.irfft2(img_con)

        
    if large_psf:
        img_con = img_con[padsize_large_psf : im_size - padsize_large_psf, padsize_large_psf : im_size - padsize_large_psf]
        
    if pad and not large_psf:
        img_con = img_con[padsize_pad : im_size - padsize_pad, padsize_pad : im_size - padsize_pad]

    img_con = img_con.astype(img.dtype)
    return img_con

  

def pad_img_psf(img, psf, large_psf =  False, constant_values = 0.):
    im_size =  np.array(img.shape)
    padsize_pad, padsize_large_psf = (0.25*im_size).astype(int), (0.5*im_size).astype(int)
    if large_psf:
        img = np.pad(img, ((padsize_large_psf[0], padsize_large_psf[0]), (padsize_large_psf[1], padsize_large_psf[1])), constant_values = constant_values)
        img[img == 0] = 0. #np.finfo(img.dtype).tiny
    else:   
        psf, img = np.pad(psf, ((padsize_pad[0], padsize_pad[0]), (padsize_pad[1], padsize_pad[1])), constant_values = constant_values), np.pad(img, ((padsize_pad[0], padsize_pad[0]), (padsize_pad[1], padsize_pad[1])), constant_values = constant_values)

    return img, psf

def unpad_img_psf(img, psf, large_psf = False):
    im_size =  np.array(img.shape)
    unpadsize_pad, unpadsize_large_psf = (1/6. * im_size).astype(int), (1/4. * im_size).astype(int)
    if large_psf:
        img = img[unpadsize_large_psf[0] : im_size[0] - unpadsize_large_psf[0], unpadsize_large_psf[1] : im_size[1] - unpadsize_large_psf[1]]
    else:
        img = img[unpadsize_pad[0] : im_size[0] - unpadsize_pad[0], unpadsize_pad[1] : im_size[1] - unpadsize_pad[1]]
        psf = psf[unpadsize_pad[0] : im_size[0] - unpadsize_pad[0], unpadsize_pad[1] : im_size[1] - unpadsize_pad[1]]
    return img, psf

def get_boundary_boxes(mask, img, psf):
        if mask.ndim == 1:
            bd_box_img = mask
        if mask.ndim == 2:
            mask = np.where(mask != 0)
            bd_box_img = [ min(mask[0]), max(mask[0]), min(mask[1]), max(mask[1])]
        bd_box_makeeven = np.array([0, 0, 0, 0])
        if (bd_box_img[1] - bd_box_img[0]) %2 == 0:
            bd_box_makeeven[1] = 1
        if (bd_box_img[3] - bd_box_img[2]) %2 == 0:
            bd_box_makeeven[3] = 1 
        if bd_box_img[1] + bd_box_makeeven[1] == img.shape[0]:
            bd_box_makeeven[0] -= 1
            bd_box_makeeven[1] -= 1
        if bd_box_img[3] + bd_box_makeeven[3] == img.shape[1]:
            bd_box_makeeven[2] -= 1
            bd_box_makeeven[3] -= 1
        bd_box_img += bd_box_makeeven
            
        #convert the boundary box of the mask to a corresponding boundary box for the psf
        bd_box_psf = [psf.shape[0]//2 - (bd_box_img[1] - bd_box_img[0] +1), psf.shape[0]//2 + (bd_box_img[1] - bd_box_img[0] +1) -1,
                      psf.shape[1]//2 - (bd_box_img[3] - bd_box_img[2] +1), psf.shape[1]//2 + (bd_box_img[3] - bd_box_img[2] +1) -1]
        return bd_box_img, bd_box_psf, bd_box_makeeven

def estimate_scattered_light(img_in, psf_in, bd_box_img, large_psf = False, pad = True):
    #derive the scattered light into the boundary box region
    img = np.copy(img_in)
    psf = np.copy(psf_in)
    
    #as we only want to derive the scattered light from the surrounding into the boundary box, set the image intensity in the boundary box to zero.
    #as we only want to have the scattered light, we set the intrinsic intensity, i.e., the center of the psf, to zero.
    img[bd_box_img[0] : bd_box_img[1] + 1, bd_box_img[2] : bd_box_img[3] + 1] = 0.
    psf[psf.shape[0]//2, psf.shape[1]//2] = 0.

    #pad the image and psf to break the periodic boundary condition involved by the convolution in the fourier domain
    if pad == True:
       img, psf = pad_img_psf(img, psf, large_psf = large_psf)
   
    
    #derive the fourier transform of the psf
    psf = np.roll(np.roll(psf, psf.shape[0]//2, axis=0),
                      psf.shape[1]//2,
                      axis=1)
    psf = np.fft.rfft2(psf) 
    #derive the foureir transform of the image
    img = np.fft.rfft2(img)
    #convolve it with the psf
    img_background = img * psf
    #and transform it back to the spatial domain
    img_background = np.fft.irfft2(img_background)
    
    #undo the padding
    if pad == True:
        img_background, psf = unpad_img_psf(img_background, psf, large_psf = large_psf) 
        
    return img_background
    
