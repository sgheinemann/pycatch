'''
===============================
Basic Example of pyCATCH usage
===============================

This example python file details the basic usage of pyCATCH.
It documents the process of coronal hole extraction from downloading the data, performing the analysis and saving the results.
'''

#------------------------------------------------------------------------------
# import the pycatch class
from pycatch.pycatch import pycatch

#------------------------------------------------------------------------------
# initialize pycatch 
ch = pycatch()

#------------------------------------------------------------------------------
# download EUV data 
# default is AIA/SDO 193 data
ch.download('2013-05-29T18:00') # example date

#------------------------------------------------------------------------------
# load EUV data 
# filepath of the downloaded file is stored in ch.map_file, it can also be given manually
ch.load()

#------------------------------------------------------------------------------
# download magnetogram 
# default is HMI/SDO 45s data
# EUV map must be loaded for the magnetogram download, it downloads the magnetogram date closest to the EUV map date
ch.download_magnetogram()

#------------------------------------------------------------------------------
# load magnetogram 
# filepath of the downloaded file is stored in ch.magnetogram_file, it can also be given manually
ch.load(mag=True)

#------------------------------------------------------------------------------
# calibrate data
# EUV: Default behaviour: register, normalize,  correct for instrument degradation, annulus limb correction, set off limb pixel to nan 
ch.calibration()

# magnetogram: Default behaviour: rotate to north up, align with EUV map, set off limb pixel to nan 
ch.calibration_mag()

#------------------------------------------------------------------------------
# reduce resolution
# Default: bin to 1024x1024
# rebins both the EUV map and the magnetogram if both are loaded
ch.rebin()

#------------------------------------------------------------------------------
# cutout a sub region of the sun
# Default: cuts the map from (-1100,-1100) arsec to (1100,1100) arcsec 
ch.cutout()

#------------------------------------------------------------------------------
# select point within coronal hole that should be extracted
# hint displays possible coronal holes
ch.select(hint=True)


#------------------------------------------------------------------------------
# find threshold for coronal hole extractions
# 4 options
option=2
#----------
# 1) manually
if option == 1:
    ch.set_threshold(40) # 40% of median solar disk intensity

# 2) use thr derived from CATCH statistics (Heinemann et al. 2019)
#    it is advised to use this option to get a starting suggestion and then adjust the threshold as needed
elif option == 2:
    ch.suggest_threshold() #
    
# 3) select from solar disk intensity distribution
elif option == 3:
    ch.threshold_from_hist()
    
# 4) calculate the coronal hole area and uncertainty as function of intensity
#    and select the boundary to be where the uncertainty is lowest
#    Warning: Can be slow with high resolution
elif option == 4:
    ch.calculate_curves() 
    ch.threshold_from_curves()

else:
    pass
#------------------------------------------------------------------------------
# extract the coronal hole
ch.extract_ch()

#------------------------------------------------------------------------------
# view coronal hole extraction
# EUV map
ch.plot_map()

# magnetogram
ch.plot_map(mag=True)

#------------------------------------------------------------------------------
# calculate coronal hole properties
# morphological properties
ch.calculate_properties()

# magnetic properties
ch.calculate_properties(mag=True)

#------------------------------------------------------------------------------
# save catch results
# save properties in text file
ch.print_properties()

# save pycatch object // save your progress
ch.save()

# save plots
# EUV map
ch.plot_map(save=True)
# magnetogram
ch.plot_map(mag=True,save=True)

# save binary coronal hole map to fits file
ch.bin2fits()

#------------------------------------------------------------------------------