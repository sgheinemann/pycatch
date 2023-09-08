"""
pycatch

init file

@author: S.G. Heinemann
"""


import os,sys
import numpy as np
import pathlib


import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolor
from matplotlib.backend_bases import MouseButton

import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.constants as const
from astropy.io import fits

import sunpy
import sunpy.map
import sunpy.util.net
from sunpy.coordinates import frames
from sunpy.net import Fido, attrs as a

import copy

import aiapy
import aiapy.calibrate


import cv2
import pickle

from utils import calibration as cal