import os, sys

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolor
from matplotlib.backend_bases import MouseButton

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

from scipy.ndimage import gaussian_filter1d
from cv2 import morphologyEx as morph
from cv2 import MORPH_OPEN,MORPH_CLOSE
import cv2

import numexpr as ne
import time

mpl.rcParams['axes.unicode_minus'] = True
mpl.rcParams['mathtext.fontset'] = 'stixsans'
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['ytick.major.size']=    6    # major tick size in points
mpl.rcParams['ytick.minor.size']=    4       # minor tick size in points
mpl.rcParams['ytick.major.width']=   2     # major tick width in points
mpl.rcParams['ytick.minor.width']=   2     # minor tick width in points

mpl.rcParams['xtick.major.size']=    6    # major tick size in points
mpl.rcParams['xtick.minor.size']=    4       # minor tick size in points
mpl.rcParams['xtick.major.width']=   2     # major tick width in points
mpl.rcParams['xtick.minor.width']=   2     # minor tick width in points

# cursor class to click into plot (snaps to datapoint)

class SnappingCursor:
    """
    A cross-hair cursor that snaps to the data point of a line, which is
    closest to the *x* position of the cursor.

    For simplicity, this assumes that *x* values of the data are sorted.
    """
    def __init__(self,fig, ax, line, line2=None, names=['y'],xe=None,ye=None):
        self.ax = ax
        self.fig=fig
        self.names=names
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        self.x, self.y = line.get_data()
        self.x2, self.y2 = line2.get_data() if line2 is not None else [None,None]
        self.xe = xe if xe is not None else None
        self.ye = ye if ye is not None else None
        self.location=None
        
        self._last_index = None
        self._last_plindex = None
        # text location in axes coords
        self.text = ax.text(0.98, 0.9, '',fontsize=14, transform=ax.transAxes,va='top', ha='right',bbox=(dict(facecolor='white',alpha=0.8,boxstyle='round')))
        self._creating_background = False
        ax.figure.canvas.mpl_connect('draw_event', self.on_draw)
        self.cid1=line.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.cid2=line.figure.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        
    def on_draw(self, event):
        self.create_new_background()
        
    def update_background(self):
        if self._creating_background:
            # discard calls triggered from within this function
            return
        self._creating_background = True
        self.set_cross_hair_visible(False)
        self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
        self.set_cross_hair_visible(True)
        self._creating_background = False
        
    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

        
    def on_mouse_move(self, event):
        if self.background is None:
            self.create_new_background()
        if not event.inaxes:
            self._last_index = None
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.restore_region(self.background)
                self.ax.figure.canvas.blit(self.ax.bbox)
        else:
            self.set_cross_hair_visible(True)
            x, y = event.xdata, event.ydata
            index = min(np.searchsorted(self.x, x), len(self.x) - 1)
            if index == self._last_index:
                return  # still on the same data point. Nothing to do.
            self._last_index = index
            x = self.x[index]
            y = self.y[index]
            # update the line positions
            self.horizontal_line.set_ydata([y])
            self.vertical_line.set_xdata([x])

            
            self.ax.figure.canvas.restore_region(self.background)
            self.ax.draw_artist(self.horizontal_line)
            self.ax.draw_artist(self.vertical_line)
            
            legendstr=f'{self.names[0]} :{x:.2f}\n'
            if self.xe is not None:
                xin = self.xe[index]
                legendstr += f'{self.names[0]} :{xin:.2f}\n'
            
            legendstr += f'\n{self.names[1]}={y:.2f}'
            if self.y2 is not None:
                y2 = self.y2[index]
                legendstr += f'\n{self.names[2]}={y2:.2f}'
            
            if self.ye is not None:
                y5 = self.ye[index]
                legendstr += f'\n{self.names[-1]}={y5:.2f}'

            self.text.set_text(legendstr)
            self.ax.draw_artist(self.text)                
            self.ax.figure.canvas.blit(self.ax.bbox)

            
    def on_mouse_click(self, event):
        if self.background is None:
            self.create_new_background()
        if not event.inaxes:
            self._last_plindex = None
        else:
            if event.button == 1:
                x, y = event.xdata, event.ydata
                index = min(np.searchsorted(self.x, x), len(self.x) - 1)
                if index == self._last_plindex:
                    return  # still on the same data point. Nothing to do.
                self._last_plindex = index
                x = self.x[index]
                self.location = x 
                
                self.ax.figure.canvas.restore_region(self.base_background)
                p=self.ax.axvline(self.location,color='r', lw=1)
                self.ax.draw_artist(p)
                    
                self.ax.figure.canvas.blit(self.ax.bbox)
                self.update_background()

            elif event.button == 3:
                
                self.ax.figure.canvas.restore_region(self.base_background)
                self.ax.figure.canvas.blit(self.ax.bbox)
                self.update_background()

            elif event.button == 2:
                
                event.canvas.mpl_disconnect(self.cid1)
                event.canvas.mpl_disconnect(self.cid2)     
                
                plt.close('all')
                self.fig.canvas.stop_event_loop()
                
                
                
                
# open window to click for coronal hole seed point
def get_point_from_map(map, fsize):
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(projection=map)
    map.plot(axes=ax)
    plt.show()
    point=plt.ginput(n=1, timeout=120, show_clicks=True, mouse_add=MouseButton.LEFT, mouse_pop=MouseButton.RIGHT, mouse_stop=MouseButton.MIDDLE )
    plt.close()
    return point


# open window to click in histogram for threshold / use snapping cursor
def get_thr_from_hist(map, fsize):
    
    # calc hist of solar disk
    hpc_coords=all_coordinates_from_map(map)
    mask=coordinate_is_on_solar_disk(hpc_coords)
    data=np.where(mask == True, map.data, np.nan)
    data_median = np.nanmedian(data)
    hist, bedges = np.histogram(data, bins=150, range=(0,1.5*data_median), density=True)
    y=gaussian_filter1d(hist, 5)
    x=bedges[:-1]+(bedges[1]-bedges[0])/2
    
    
    plt.ion()

    fig = plt.figure(figsize=fsize)
    ax=fig.add_subplot()

    ytiti=r'$N_{\mathrm{Normalized}$'
    ax.set_ylabel(ytiti)           
    xtiti='I [DN/s]'    
    ax.set_xlabel(xtiti)

    ax.set_xlim(0,1.5*data_median)
    ax.set_ylim(0,np.nanmax(y)*1.1)

    p,=ax.plot(x,y,linewidth=2, label=r'$N')

    snap_cursor = SnappingCursor(fig,ax,p,names=[r'$I$ [DN/s]',r'$I_{med}$ [$\%$]',r'$N$'], x2=100*x/data_median)
    fig.canvas.start_event_loop(timeout=-1)
    plt.show()
    
    return snap_cursor.location
    

# open window to click in area curves for threshold / use snapping cursor
def get_thr_from_curves(map, curves ,fsize):
    
    hpc_coords=all_coordinates_from_map(map)
    mask=coordinate_is_on_solar_disk(hpc_coords)
    data=np.where(mask == True, map.data, np.nan)
    data_median = np.nanmedian(data)
    
    x=curves[0]
    y1=curves[1]
    y2=curves[2]
    
    plt.ion()

    fig = plt.figure(figsize=fsize)
    ax=fig.add_subplot()
    ax2 = ax.twinx()

    ytiti=r'$Coronal Hole Area [$10^{10}$km$^2$]'
    ax.set_ylabel(ytiti, color='blue')           
    
    
    ytiti=r'Uncertainty'
    ax2.set_ylabel(ytiti, color='red')   

   
    xtiti='I [DN/s]'    
    ax.set_xlabel(xtiti)

    ax.set_xlim(0,x[-1])
    ax.set_ylim(0,np.nanmax(y1)*1.1)
    ax2.set_ylim(0,np.nanmax(y2)*1.1)

    p1,=ax.plot(x,y1,linewidth=2)
    p2,=ax.plot(x,y2,linewidth=2)

    snap_cursor = SnappingCursor(fig,ax,p1,names=[r'$I$ [DN/s]',r'$I_{med}$ [$\%$]',r'$A$ [$10^{10}$km$^2$]',r'$\epsilon$'], x2=100*x/data_median)
    fig.canvas.start_event_loop(timeout=-1)
    plt.show()
    
    return snap_cursor.location    
    