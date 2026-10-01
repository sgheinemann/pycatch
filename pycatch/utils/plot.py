import os, sys

import numpy as np
from scipy.ndimage import gaussian_filter1d
import astropy.units as u

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import matplotlib.patheffects as pe

import sunpy
import sunpy.map
import sunpy.util.net
from sunpy.map.maputils import all_coordinates_from_map,coordinate_is_on_solar_disk

import pycatch.utils.extensions as ext

mpl.rcParams['axes.unicode_minus'] = True
mpl.rcParams['mathtext.fontset'] = 'stixsans'
mpl.rcParams['font.size'] = 16
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
    closest to the x position of the cursor.

    For simplicity, this assumes that x values of the data are sorted.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The matplotlib figure to which the cursor is attached.
    ax : matplotlib.axes.Axes
        The matplotlib axes to which the cursor is attached.
    line : matplotlib.lines.Line2D
        The line for which the cursor will snap to data points.
    line2 : matplotlib.lines.Line2D, optional
        An optional second line for which the cursor can snap to data points.
    names : list of str, optional
        A list of names for the y-values displayed in the cursor's tooltip.
    xe : array-like, optional
        An array of x-values associated with the data points.
    ye : array-like, optional
        An array of y-values associated with the data points.

    Attributes
    ----------
    location : float
        The x-value of the currently snapped data point.
    
    Methods
    -------
    on_mouse_click(event)
        Handles mouse clicks to capture the cursor's location.
    on_mouse_move(event)
        Handles mouse movement to snap the cursor to the nearest data point.
    on_draw(event)
        Handles drawing events to update the cursor's background.
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
        self.text = ax.text(0.02, 0.9, '',fontsize=14, transform=ax.transAxes,va='top', ha='left',bbox=(dict(facecolor='white',alpha=0.8,boxstyle='round')))
        self._creating_background = False
        ax.figure.canvas.mpl_connect('draw_event', self.on_draw)
        self.cid1=line.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.cid2=line.figure.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        
    def on_draw(self, event):
        self.create_new_background()

    def create_new_background(self):
        if self._creating_background:
            # discard calls triggered from within this function
            return
        self._creating_background = True
        self.set_cross_hair_visible(False)
        self.ax.figure.canvas.draw()
        self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
        self.set_cross_hair_visible(True)
        self._creating_background = False
        
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
            
            legendstr=f'{self.names[0]} :{x:.2f}'
            if self.xe is not None:
                xin = self.xe[index]
                legendstr += f'\n{self.names[1]} :{xin:.2f}'
            
            legendstr += f'\n{self.names[2]}={y:.4f}'
            if self.y2 is not None:
                y2 = self.y2[index]
                legendstr += f'\n{self.names[3]}={y2:.4f}'
            
            if self.ye is not None:
                y5 = self.ye[index]
                legendstr += f'\n{self.names[-1]}={y5:.4f}'

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
                
                event.canvas.mpl_disconnect(self.cid1)
                event.canvas.mpl_disconnect(self.cid2)     
                
                plt.close('all')
                self.fig.canvas.stop_event_loop()
                
                
                
                
    
# open window to click for coronal hole seed point
def get_point_from_map(map, hint, fsize, is_notebook=False, max_points=10):
    """
    Display a solar map and allow the user to interactively select a point on the map.
    
    Parameters
    ----------
    map : sunpy.map.GenericMap
        The solar map to be displayed.
    fsize : tuple of int
        The size of the figure (width, height) in inches.
    is_notebook : bool
        Is pycatch being run in a jupyter notebook
    max_points : int
        Max number of coronal holes/points to be selected
    
    Returns
    -------
    point : list of float
        A list containing the coordinates (x, y) of the selected point on the solar map.
    """
        
    # Enable interactive mode
    plt.ion()
    
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(projection=map)
    map.plot(axes=ax)
    
    
    thr=ext.median_disk(map)*0.35
    hpc_coords=all_coordinates_from_map(map)
    mask=coordinate_is_on_solar_disk(hpc_coords)
    if hint:
        data=np.where(np.logical_and(mask == True,map.data <= thr),1, np.nan)
    else:
        data=np.full_like(map.data,np.nan)
  
    thrmap=sunpy.map.Map((data,map.meta))
    thrmap.plot_settings['cmap']=plt.get_cmap('Blues')
    thrmap.plot(axes=ax, alpha =0.5, vmin=0.9, vmax=1.1 )

    fig.canvas.draw()

    # --- Branch 1: CLI Scripts (Native ginput) ---
    if not is_notebook:
        plt.show(block=False)
        points = plt.ginput(
            n=max_points, 
            timeout=120, 
            show_clicks=True, 
            mouse_add=MouseButton.LEFT, 
            mouse_pop=MouseButton.RIGHT, 
            mouse_stop=MouseButton.MIDDLE
        )
        plt.close(fig)
        return points

    # --- Branch 2:  Jupyter Notebooks (mpl_connect) ---
    else:
        fig.selected_points = []
        fig.markers = []
    
        ax.set_title(
            f"Select up to {max_points} points | Click: Add | Right-click: Undo\n"
            f"Enter / Double-Click: Finish",
            pad=15,
        )
    
        def finish():
            fig.canvas.mpl_disconnect(cid_click)
            fig.canvas.mpl_disconnect(cid_key)
            ax.set_title(
                f"Finished!\nCaptured {len(fig.selected_points)}/{max_points} point(s).",
                pad=15,
            )
            fig.canvas.draw_idle()
            if fig.canvas.start_event_loop:
                fig.canvas.stop_event_loop()
    
        def onclick(event):
            if event.inaxes != ax:
                return
    
            # Finish manually via Double-Click or Middle-Click
            if event.dblclick or event.button == MouseButton.MIDDLE:
                finish()
                return
    
            # Left Click -> Add Point
            if event.button == MouseButton.LEFT:
                if len(fig.selected_points) < max_points:
                    # --- KEY FIX FOR COORDINATE EQUALITY ---
                    # event.x, event.y are raw display pixels on the canvas.
                    # Converting via ax.transData.inverted() gets the exact WCS/data pixel coordinates that ginput uses.
                    x_pixel, y_pixel = ax.transData.inverted().transform(
                        (event.x, event.y)
                    )
    
                    # Append as exact standard float tuple matching ginput
                    fig.selected_points.append((float(x_pixel), float(y_pixel)))
    
                    # Plot marker using the event's axis display data (xdata, ydata)
                    (marker,) = ax.plot(
                        event.xdata,
                        event.ydata,
                        "rx",
                        markersize=10,
                        markeredgewidth=2,
                    )
                    fig.markers.append(marker)
    
                    # Auto-finish if maximum points limit reached
                    if len(fig.selected_points) == max_points:
                        finish()
                    else:
                        ax.set_title(
                            f"Selected {len(fig.selected_points)}/{max_points} point(s) | Right-click: Undo\n"
                            f"Press Enter or Double-click to complete early",
                            pad=15,
                        )
                        fig.canvas.draw_idle()
    
            # Right Click -> Undo Last Point
            elif event.button == MouseButton.RIGHT:
                if fig.selected_points and fig.markers:
                    fig.selected_points.pop()
                    marker = fig.markers.pop()
                    marker.remove()
                    ax.set_title(
                        f"Selected {len(fig.selected_points)}/{max_points} point(s) | Right-click: Undo\n"
                        f"Press Enter or Double-click to complete early",
                        pad=15,
                    )
                    fig.canvas.draw_idle()
    
        def onkey(event):
            if event.key in ["enter", " ", "q", "escape"]:
                finish()
    
        cid_click = fig.canvas.mpl_connect("button_press_event", onclick)
        cid_key = fig.canvas.mpl_connect("key_press_event", onkey)
    
        plt.show()
    
        # Block execution until the user finishes clicking
        fig.canvas.start_event_loop(timeout=120)
    
        return fig.selected_points


# open window to click in histogram for threshold / use snapping cursor
def get_thr_from_hist(map, fsize, is_notebook=False):
    """
    Interacticely get a threshold value from a histogram of solar disk data.
    
    Parameters
    ----------
    map : sunpy.map.GenericMap
        The solar map for which the threshold will be determined.
    fsize : tuple of int
        The size of the figure (width, height) in inches.
    is_notebook : bool
        Is pycatch being run in a jupyter notebook

    Returns
    -------
    thr : float
        The threshold value determined interactively from the histogram.
    """
    # calc hist of solar disk
    hpc_coords=all_coordinates_from_map(map)
    mask=coordinate_is_on_solar_disk(hpc_coords)
    data=np.where(mask == True, map.data, np.nan)
    data_median = np.nanmedian(data)
    hist, bedges = np.histogram(data, bins=150, range=(0,1*data_median), density=True)
    y=gaussian_filter1d(hist, 5)
    x=bedges[:-1]+(bedges[1]-bedges[0])/2
    
    
    plt.ion()

    fig = plt.figure(figsize=fsize)
    ax=fig.add_subplot()

    ytiti=r'$N_{\mathrm{Normalized}}$'
    ax.set_ylabel(ytiti)           
    xtiti=r'$I$ [DN/s]'    
    ax.set_xlabel(xtiti)

    ax.set_xlim(0,1.*data_median)
    ax.set_ylim(0,np.nanmax(y)*1.1)

    p,=ax.plot(x,y,linewidth=2, label=r'N')

    # --- Branch 1: CLI Scripts (Native SnappingCursor / ginput) ---
    if not is_notebook:
        # Assumes SnappingCursor is defined in your module scope
        snap_cursor = SnappingCursor(fig, ax, p, names=[r'$I$ [DN/s]', r'$I_{med}$ [$\%]', r'$N$'], xe=100 * x / data_median)
        fig.canvas.start_event_loop(timeout=-1)
        plt.show()
        return snap_cursor.location

    # --- Branch 2: Jupyter Notebooks (Single Point Async Listener) ---
    else:
        fig.threshold_val = None
        fig.v_line = None

        ax.set_title(
            "Click: Set/Replace threshold line | Right-click: Clear\n"
            "Enter / Double-Click: Finish & Lock in selection",
            pad=15
        )

        def finish():
            fig.canvas.mpl_disconnect(cid_click)
            fig.canvas.mpl_disconnect(cid_key)
            if fig.threshold_val is not None:
                ax.set_title(
                    f"Threshold Locked In: {fig.threshold_val:.2f} DN/s\n"
                    f"({100 * fig.threshold_val / data_median:.1f}% of disk median)",
                    pad=15
                )
            else:
                ax.set_title("Finished! No threshold selected.", pad=15)
            
            fig.canvas.draw_idle()
            print(f"Locked-in Threshold: {fig.threshold_val}")

            # Stop the blocking event loop so the function can return the value
            if hasattr(fig.canvas, 'stop_event_loop'):
                fig.canvas.stop_event_loop()

        def onclick(event):
            if event.inaxes != ax:
                return

            # Double-click or Middle-click locks in and finishes
            if event.dblclick or event.button == MouseButton.MIDDLE:
                finish()
                return

            # Left-click: Set or replace the existing threshold line
            if event.button == MouseButton.LEFT:
                fig.threshold_val = float(event.xdata)
                
                # Remove previous vertical line if user is replacing a point
                if fig.v_line is not None:
                    fig.v_line.remove()

                # Draw new vertical indicator line
                fig.v_line = ax.axvline(
                    x=fig.threshold_val, 
                    color='r', 
                    linestyle='--', 
                    linewidth=2
                )
                
                pct_median = (fig.threshold_val / data_median) * 100
                ax.set_title(
                    f"Selected Threshold: {fig.threshold_val:.2f} DN/s ({pct_median:.1f}% median)\n"
                    f"Click again to replace | Enter / Double-Click to lock in",
                    pad=15
                )
                fig.canvas.draw_idle()

            # Right-click: Clear active point
            elif event.button == MouseButton.RIGHT:
                if fig.v_line is not None:
                    fig.v_line.remove()
                    fig.v_line = None
                    fig.threshold_val = None
                    ax.set_title(
                        "Threshold cleared.\n"
                        "Click to select threshold | Enter / Double-Click: Finish",
                        pad=15
                    )
                    fig.canvas.draw_idle()

        def onkey(event):
            if event.key in ['enter', ' ', 'q', 'escape']:
                finish()

        cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
        cid_key = fig.canvas.mpl_connect('key_press_event', onkey)

        plt.show()

        # Block code execution until finish() calls stop_event_loop()
        fig.canvas.start_event_loop(timeout=-1)

        return fig.threshold_val
    

# open window to click in area curves for threshold / use snapping cursor
def get_thr_from_curves(map, curves ,fsize, is_notebook=False):
    """
    Interactively get a threshold value from a plot of coronal hole area curves.

    Parameters
    ----------
    map : sunpy.map.GenericMap
        The solar map for which the threshold will be determined.
    curves : tuple
        A tuple containing the data for plotting the curves, where `curves[0]` is
        the x-axis data, `curves[1]` is the y-axis data for the first curve, and
        `curves[2]` is the y-axis data for the second curve.
    fsize : tuple of int
        The size of the figure (width, height) in inches.

    Returns
    -------
    thr : float
        The threshold value determined interactively from the plot.
    """    
    hpc_coords=all_coordinates_from_map(map)
    mask=coordinate_is_on_solar_disk(hpc_coords)
    data=np.where(mask == True, map.data, np.nan)
    data_median = np.nanmedian(data)
    
    x=curves[0]
    y1=np.where(np.isinf(curves[1]),np.nan,curves[1])
    y2=np.where(np.isinf(curves[2]),np.nan,curves[2])    
    plt.ion()

    fig = plt.figure(figsize=fsize)
    ax=fig.add_subplot()
    ax2 = ax.twinx()

    ytiti=r'Coronal Hole Area [$10^{10}$km$^2$]'
    ax.set_ylabel(ytiti, color='blue')           
    
    
    ytiti=r'Uncertainty $\epsilon$'
    ax2.set_ylabel(ytiti, color='red')   
    ax2.set_yscale("log")
   
    xtiti=r'$I$ [DN/s]'    
    ax.set_xlabel(xtiti)

    ax.set_xlim(0,x[-1])
    ax.set_ylim(0,np.nanmax(y1)*1.1)
    ax2.set_ylim(np.nanmin(y2)*0.9,np.nanmax(y2)*1.1)

    p1,=ax.plot(x,y1,linewidth=2,color='blue')
    p2,=ax2.plot(x,y2,linewidth=2, color='red')

    # --- Branch 1: CLI Scripts (Native SnappingCursor) ---
    if not is_notebook:
        snap_cursor = SnappingCursor(fig, ax, p1, line2=p2, names=[r'$I$[DN/s]', r'$I_{med}$[$\%]', r'$A$ [$10^{10}$km$^2$]', r'$\epsilon$'], xe=100 * x / data_median)
        fig.canvas.start_event_loop(timeout=-1)
        plt.show()
        return snap_cursor.location    

    # --- Branch 2: Jupyter Notebooks (Interactive Curve Selection & Lock-In) ---
    else:
        fig.threshold_val = None
        fig.v_line = None

        ax.set_title(
            "Click: Set/Replace threshold line | Right-click: Clear\n"
            "Enter / Double-Click: Finish & Lock in selection",
            pad=15
        )

        def finish():
            fig.canvas.mpl_disconnect(cid_click)
            fig.canvas.mpl_disconnect(cid_key)
            if fig.threshold_val is not None:
                ax.set_title(
                    f"Threshold Locked In: {fig.threshold_val:.2f} DN/s\n"
                    f"({100 * fig.threshold_val / data_median:.1f}% of disk median)",
                    pad=15
                )
            else:
                ax.set_title("Finished! No threshold selected.", pad=15)
            
            fig.canvas.draw_idle()
            print(f"Locked-in Threshold: {fig.threshold_val}")

            # Stop blocking event loop to allow function return
            if hasattr(fig.canvas, 'stop_event_loop'):
                fig.canvas.stop_event_loop()

        def onclick(event):
            # Allow clicks inside primary ax or twin ax2
            if event.inaxes not in [ax, ax2]:
                return

            # Double-click or Middle-click locks in and finishes
            if event.dblclick or event.button == MouseButton.MIDDLE:
                finish()
                return

            # Left-click: Set or replace threshold line
            if event.button == MouseButton.LEFT:
                fig.threshold_val = float(event.xdata)
                
                # Remove previous vertical line indicator
                if fig.v_line is not None:
                    fig.v_line.remove()

                # Draw vertical indicator line across full plot height
                fig.v_line = ax.axvline(
                    x=fig.threshold_val, 
                    color='black', 
                    linestyle='--', 
                    linewidth=2
                )
                
                pct_median = (fig.threshold_val / data_median) * 100
                ax.set_title(
                    f"Selected Threshold: {fig.threshold_val:.2f} DN/s ({pct_median:.1f}% median)\n"
                    f"Click again to replace | Enter / Double-Click to lock in",
                    pad=15
                )
                fig.canvas.draw_idle()

            # Right-click: Clear active threshold line
            elif event.button == MouseButton.RIGHT:
                if fig.v_line is not None:
                    fig.v_line.remove()
                    fig.v_line = None
                    fig.threshold_val = None
                    ax.set_title(
                        "Threshold cleared.\n"
                        "Click to select threshold | Enter / Double-Click: Finish",
                        pad=15
                    )
                    fig.canvas.draw_idle()

        def onkey(event):
            if event.key in ['enter', ' ', 'q', 'escape']:
                finish()

        cid_click = fig.canvas.mpl_connect('button_press_event', onclick)
        cid_key = fig.canvas.mpl_connect('key_press_event', onkey)

        plt.show()

        # Block notebook execution until finish() calls stop_event_loop()
        fig.canvas.start_event_loop(timeout=-1)

        return fig.threshold_val
    

# open window to display coronal hole
def plot_map(map,bmap,boundary,uncertainty, fsize, save ,spath,grid,**kwargs):
    r"""
    Plot a solar map with optional boundaries and uncertainty overlays.
    
    Parameters
    ----------
    map : sunpy.map.GenericMap
        The solar map to be plotted.
    bmap : sunpy.map.GenericMap or None
        A binary mask map for boundaries or uncertainty, or None if not used.
    boundary : bool
        Whether to plot boundaries on the map.
    uncertainty : bool
        Whether to overlay uncertainty information on the map.
    fsize : tuple of int
        The size of the figure (width, height) in inches.
    save : bool
        Whether to save the figure to a file.
    spath : str
        The path to save the figure if `save` is True.
    \*\*kwargs
        Additional keyword arguments to pass to the `sunpy.map.Map.plot` function.
    
    Returns
    -------
    None
    """
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(projection=map)
    map.plot(axes=ax,**kwargs)

    ax.coords[0].set_ticks_visible(False)
    ax.coords[1].set_ticks_visible(False)
        
    if bmap is not None:
        bmap.meta['bunit']=''
        
        if uncertainty:
            hdata=np.where(np.logical_and(bmap.data < 5, bmap.data >0), 1,np.nan)
            umap=sunpy.map.Map((hdata,bmap.meta))
            umap.plot_settings['cmap']=plt.get_cmap('winter_r')
            umap.plot(axes=ax, alpha =0.7, vmin=0.9, vmax=1., autoalign=True)
    
        if boundary:
            bmap.draw_contours(levels=[2.5] * u.dimensionless_unscaled, axes=ax, colors='red', linewidths=1.5)

    if grid:
        ax.grid(True)
    else:
        ax.grid(False)

    ax.text(0.99,0.01,'pyCATCH', ha='right', va='bottom',fontfamily='fantasy', color='white', fontsize=20, 
            transform = ax.transAxes, path_effects=[pe.withStroke(linewidth=4, foreground="black")] )

    plt.show()
    if save:
        fig.savefig(spath,bbox_inches='tight')
        plt.close(fig)
        print(f'> pycatch ## FIGURE SAVED TO: {spath}  ##')
        
    return 



# quicklook image for maps
def show_map(map, grid=False, fsize=(8, 8), **kwargs):
    r"""
    Plot a quicklook visualization of a solar map with optional magnetogram 
    contours and grid overlays.

    Parameters
    ----------
    map : sunpy.map.GenericMap
        The primary solar map to be plotted.
    grid : bool, optional
        Whether to draw heliographic grid lines on the plot. Default is False.
    fsize : tuple of int or float, optional
        The size of the figure (width, height) in inches. Default is (8, 8).
    \*\*kwargs
        Additional keyword arguments passed directly to `map.plot()`.

    Returns
    -------
    matplotlib.axes.Axes
        The WCS axes object containing the solar map plot.
    """
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(projection=map)
    
    # Plot primary solar map
    map.plot(axes=ax, **kwargs)

    # Grid overlay
    if grid:
        map.draw_grid(axes=ax)

    return ax
















