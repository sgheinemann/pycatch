=========
Changelog
=========
1.0.0 (October 1, 2026)
-------------------------

* Major update for release

    - Updated compatibility to current versions of Python, SunPy, aiapy, ...
    - Included JSOC download capabilities (now standard)
    - Added a new .show command for quicklook images of maps
    - Added advanced AIA PSF convolution by S.J. Hofmeister (new code and new PSF that are automatically downloaded)
    - Changed .bin2fits to .save_map to allow saving of different maps as FITS files (i.e. prepped input data)
    - Fixed data prep with new Python packages
    - Added progress bars to some steps
    - Added compatibility with Jupyter Notebooks
    - Added the capability to extract and analyze multiple coronal holes at the same time for a given image (now arbitrarily capped at 10)
    
* Bug fixing
	- fixed various small bugs
	
0.2.1 (Novemeber 9, 2023)
-------------------------

* Bug fixing
	- fixed a bug where importing pycatch would not find the _version.py file
	- fixed some path issue in the initialization of the home path
	- fixed what version was output in the .print_properties() function
	- fixed some function descriptions 
	- fixed a problem where the .print_properties() function would not write a file if no magnetic properties were calculated.
	- fixed an issue with where possibly additional windows open when trying to select a seed point

* Minor changes
	- changed the keyword order in the  .load()  routine to make it more intuitive
	- added aiapy to the list of required packages
	- switched default for .plot_map() from small from True to False
	- added a pdf version of the documentation (User_Manual.pdf that can be found on the github page)

0.2.0 (September 2023)
----------------------

* Initial functional beta release (beta - testing)


0.1.0 (September 2023)
----------------------

* Initial alpha build

