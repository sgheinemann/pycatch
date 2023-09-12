
import numpy as np
import copy

import astropy.units as u
from astropy.coordinates import SkyCoord

import sunpy
from sunpy.map.maputils import all_coordinates_from_map,coordinate_is_on_solar_disk

import scipy.interpolate
import scipy.ndimage

# initialize units and names

def init_props():
        idict             =  {'A':('Area','10^10 km^2'),                        'dA':('Area Uncertainty','10^10 km^2'),
                              'Imean':('Mean Intensity','ct/s'),                'dImean':('Mean Intensity Uncertainty','ct/s'),
                              'Imedian':('Median Intensity','ct/s'),            'dImedian':('Mean Intensity Uncertainty','ct/s'),
                              'CoM':('Center of Mass (lon,lat)','°,°'),         'dCoM':('Center of Mass Uncertainty (lon,lat)','°,°'),
                              'ex':('Extent (lon1,lon2,lat1,lat2) ','°,°,°,°'), 'dex':('Extent Uncertainty (lon1,lon2,lat1,lat2)','°,°,°,°'),
                              'Bs':('Signed Mean Magnetic Flux Density','G'),   'dBs':('Signed Mean Magnetic Flux Density Uncertainty','G'),
                              'Bus':('Unsigned Mean Magnetic Flux Density','G'),'dBus':('Unsigned Mean Magnetic Flux Density Uncertainty','G'),
                              'Fs':('Signed Magnetic Flux','10^20 Mx'),         'dFs':('Signed Magnetic Flux Uncertainty','10^20 Mx'), 
                              'Fus':('Unsigned Magnetic Flux','10^20 Mx'),      'dFus':('Unsigned Magnetic Flux Uncertainty','10^20 Mx'),                                                    
                              'FB':('Flux Balance','%'),                        'dFB':('Flux Balance Uncertainty','%') }                                                     
    
        return idict

#print properties to txt file
def printtxt(file, pdict,names, version):
        with open(file, 'w') as f:
            f.write(f'# pyCATCH v{version}\n')
            
            f.write('#') 
            for key,value in names.items():
                f.write(f'{value[0]} ; ')
            f.write('\n')
            
            f.write('#') 
            for key,value in names.items():
                f.write(f'{value[1]} ; ')
            f.write('\n')    
            
            for key,value in pdict.items():
                f.write(f'{value} ; ')
        
        return
    
# median from disk
def median_disk(map):
        hpc_coords=all_coordinates_from_map(map)
        mask=coordinate_is_on_solar_disk(hpc_coords)
        data=np.where(mask == True, map.data, np.nan)
        return np.nanmedian(data)

# find nearest index
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# make map cutout
def cutout(map,top,bot):
    cmap=copy.deepcopy(map)
    topc=SkyCoord(top[0]*u.arcsec,top[1]*u.arcsec, frame=cmap.coordinate_frame)
    botc=SkyCoord(bot[0]*u.arcsec,bot[1]*u.arcsec, frame=cmap.coordinate_frame)
    return cmap.submap(botc,top_right=topc)

#rebin function
def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)
    
    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print( "[congrid] dimensions error. " )
        return None
    newdims = np.asarray( newdims, dtype=float )    
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = np.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa
    
    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = np.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + range( ndims - 1 )
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = np.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs        

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print ("Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported.")
        return None 