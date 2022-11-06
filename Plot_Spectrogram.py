# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 17:51:35 2022

@author: eliza
"""
root="C:/Users/eliza/Desktop/Python_Scripts/"

import numpy as np
from scipy.io import readsav
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits import axes_grid1
import matplotlib.colors as colors
from matplotlib.patches import Polygon
import time as t
import pandas as pd
from tfcat import TFCat
import shapely.geometry as sg
from os import path
import matplotlib as mpl
import plotting_func as plt_func

'''____This was adapted from original code for the space labelling tool!!!___'''
def get_polygons(polygon_fp,start, end):
    unix_start=t.mktime(start.utctimetuple())
    unix_end=t.mktime(end.utctimetuple())
    #array of polygons found within time interval specified.
    polygon_array=[]
    if path.exists(polygon_fp):
        catalogue = TFCat.from_file(polygon_fp)
        for i in range(len(catalogue)):
                time_points=np.array(catalogue._data.features[i]['geometry']['coordinates'][0])[:,0]
                if any(time_points <= unix_end) and any(time_points >= unix_start):
                    polygon_array.append(np.array(catalogue._data.features[i]['geometry']['coordinates'][0]))
    #polgyon array contains a list of the co-ordinates for each polygon within the time interval         
    return polygon_array

'''____This was adapted from original code for the space labelling tool!!!___'''
def extract_data(file_data, time_view_start, time_view_end, val):
    # read the save file and copy variables
    time_index = 't'
    freq_index = 'f'
    val_index = val
    file = readsav(file_data)

    t_doy = file[time_index].copy()
    doy_one = pd.Timestamp(str(1997)) - pd.Timedelta(1, 'D')
    t_timestamp = np.array([doy_one + pd.Timedelta(t * 1440, 'm') for t in t_doy],
        dtype=pd.Timestamp)
    t_isostring = np.array([datetime.strftime(i,'%Y-%m-%dT%H:%M:%S') for i in t_timestamp])
    time =t_isostring
    #print(time)
    #time = np.vectorize(fix_iso_format)(t_isostring)
    time = np.array(time, dtype=np.datetime64)
    time_view = time[(time >= time_view_start) & (time < time_view_end)]

    # copy the flux and frequency variable into temporary variable in
    # order to interpolate them in log scale
    s = file[val_index][:, (time >= time_view_start) & (time <= time_view_end)].copy()
    frequency_tmp = file[freq_index].copy()

    # frequency_tmp is in log scale from f[0]=3.9548001 to f[24] = 349.6542
    # and then in linear scale above so it's needed to transfrom the frequency
    # table in a full log table and einterpolate the flux table (s --> flux
    frequency = 10**(np.arange(np.log10(frequency_tmp[0]), np.log10(frequency_tmp[-1]), (np.log10(max(frequency_tmp))-np.log10(min(frequency_tmp)))/399, dtype=float))
    flux = np.zeros((frequency.size, len(time_view)), dtype=float)

    for i in range(len(time_view)):
        flux[:, i] = np.interp(frequency, frequency_tmp, s[:, i])

    return time_view, frequency, flux

def plot_mask(ax,time_view_start, time_view_end, val, file_data,polygon_fp):
    #polgyon array contains a list of the co-ordinates for each polygon within the time interval
    polygon_array=get_polygons(polygon_fp, time_view_start, time_view_end)
    #signal data and time frequency values within the time range specified.
    time_dt64, frequency, flux=extract_data(file_data, time_view_start, time_view_end, val)
    time_unix=[i.astype('uint64').astype('uint32') for i in time_dt64]
    #Meshgrid of time/frequency vals.
    times, freqs=np.meshgrid(time_unix, frequency)
    #Total length of 2D signal array.
    data_len = len(flux.flatten())
    #indices of each item in flattened 2D signal array.
    index = np.arange(data_len, dtype=int)
    #Co-ordinates of each item in 2D signal array.
    coords = [(t, f) for t,f in zip(times.flatten(), freqs.flatten())]
    data_points = sg.MultiPoint([sg.Point(x, y, z) for (x, y), z in zip(coords, index)])
    #Make mask array.
    mask = np.zeros((data_len,))
    #Find overlap between polygons and signal array.
    #Set points of overlap to 1.
    for i in polygon_array:
        fund_polygon = sg.Polygon(i)
        fund_points = fund_polygon.intersection(data_points)
        if len(fund_points.bounds)>0:
            mask[[int(geom.z) for geom in fund_points.geoms]] = 1
    mask = (mask == 0)
    #Set non-polygon values to zero in the signal array.
    #flux_ones = np.where(flux>0, 1, np.nan)
    v = np.ma.masked_array(flux, mask=mask).filled(np.nan)
    
    #colorbar limits
    vmin = np.quantile(flux[flux > 0.], 0.05)
    vmax = np.quantile(flux[flux > 0.], 0.95)
    scaleZ = colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.get_cmap('binary_r').copy()
    
    #Plot Figure
    fontsize=20
    fig = plt.figure()
    im=ax.pcolormesh(time_dt64,frequency, v,norm=scaleZ,cmap=cmap,shading='auto')
    
    #format axis 
    ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=fontsize-6)
    ax.set_ylabel('Frequency (kHz)', fontsize=fontsize)
    ax.set_xlabel('Time', fontsize=fontsize)
    #ax.set_title(f'{time_view_start} to {time_view_end}', fontsize=fontsize + 2)
    dateFmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M')
    #For more concise formatting (for short time durations)
    #ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.xaxis.set_major_formatter(dateFmt)
    
    # Formatting colourbar
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax)
    if val == 's':
        cb.set_label(r'Flux Density'+'\n (W/m$^2$/Hz)', fontsize=fontsize-2)
    elif val =='v':
        cb.set_label('Normalized'+'\n Degree of'+'\n Circular Polarization', fontsize=fontsize-2)
        
    cb.ax.tick_params(labelsize=fontsize-2)
    #cb.remove()
    
    plt.close(fig)
    
    return ax


def plot_flux(ax,time_view_start, time_view_end, file, colour_in=None, frequency_lines=None):
    
    #Load data from .sav file
    time, freq, flux = extract_data(file, time_view_start=time_view_start,\
                                    time_view_end=time_view_end,val='s')
    #Parameters for colorbar
    #This is the function that does flux normalisation based on s/c location
    #vmin, vmax=plt_func.flux_norm(time[0], time[-1])   #change from log10 to actual values.
    clrmap ='viridis'
    vmin = np.quantile(flux[flux > 0.], 0.05)
    vmax = np.quantile(flux[flux > 0.], 0.95)
    scaleZ = colors.LogNorm(vmin=vmin, vmax=vmax)
    
    #Make figure
    fontsize = 20
    fig = plt.figure()
    im=ax.pcolormesh(time, freq, flux, norm=scaleZ,cmap=clrmap,  shading='auto')
    ax.set_yscale('log')
    
    
    #format axis 
    ax.tick_params(axis='both', which='major', labelsize=fontsize-5)
    ax.set_ylabel('Frequency (kHz)', fontsize=fontsize)
    #ax.set_xlabel('Time', fontsize=fontsize)
    ax.set_title(f'{time_view_start} to {time_view_end}', fontsize=fontsize + 2)
    
    ######### X label formatting ###############
    
    #For more concise formatting (for short time durations)
    #ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    
    #normal
    dateFmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M')
    ax.xaxis.set_major_formatter(dateFmt)
    
    #For using trajectory data
    #ax.xaxis.set_major_formatter(plt.FuncFormatter(plt_func.ephemeris_fmt_hour_tick))
    #eph_str = '\n'.join(['DOY\n',
     #           r'$R_{sc}$ ($R_{S}$)',
      #          r'$\lambda_{sc}$ ($^{\circ}$)',
       #         r'LT$_{sc}$ (Hrs)'])
    #kwargs = {'xycoords': 'figure fraction',
     #   'fontsize': fontsize-6}
    #kwargs['xy'] = (0.03, 0.009)
    #ax.annotate(eph_str,**kwargs)
    
    # Formatting colourbar
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax)
    cb.set_label(r'Flux Density'+'\n (W/m$^2$/Hz)', fontsize=fontsize-2)
    cb.ax.tick_params(labelsize=fontsize-2)
    #cb.remove()
    
    #For adding horizontal lines at specific frequencies
    if frequency_lines is not None:
        for i in frequency_lines:
            ax.hlines(i, time[0], time[-1], colors = 'darkslategray',linewidth=1,linestyles='--', label='{}kHz'.format(i))
          
     #For plotting polygons onto spectrogram.
    if colour_in is not None:
        for shape in colour_in:
            shape_=shape.copy()
            shape_[:,0]=[mdates.date2num(datetime.fromtimestamp(i)) for i in shape_[:,0]]
            ax.add_patch(Polygon(shape_, color='black', linestyle='dashed',linewidth=4, alpha=1, fill=False))
        
    plt.close(fig)
    return ax

def plot_pol(ax,time_view_start, time_view_end, file,colour_in=None,frequency_lines=None):
    
    #Load data from .sav file
    time, freq, pol = extract_data(file, time_view_start=time_view_start, \
                                   time_view_end=time_view_end,val='v')
    #Parameters for colorbar
    vmin=-1
    vmax=1
    clrmap ='binary'
    scaleZ = colors.Normalize(vmin=vmin, vmax=vmax)
    
    #Make figure
    fontsize = 20
    fig = plt.figure()
    im=ax.pcolormesh(time, freq, pol, norm=scaleZ, cmap=clrmap, shading='auto')
    ax.set_yscale('log')
    
    
    #format axis 
    ax.tick_params(axis='both', which='major', labelsize=fontsize-5)
    ax.set_ylabel('Frequency (kHz)', fontsize=fontsize)
    ax.set_xlabel('Time', fontsize=fontsize)
    #Uncomment to set title
    #ax.set_title(f'{time_view_start} to {time_view_end}', fontsize=fontsize + 2)
    
    ######### X label formatting ###############
    #For more concise formatting (for short time durations)
    #ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    
    #normal
    #dateFmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M')
    #ax.xaxis.set_major_formatter(dateFmt)
    
    #For using trajectory data
    ax.xaxis.set_major_formatter(plt.FuncFormatter(plt_func.ephemeris_fmt_hour_tick))
    eph_str = '\n'.join(['DOY\n',
                r'$R_{sc}$ ($R_{S}$)',
                r'$\lambda_{sc}$ ($^{\circ}$)',
                r'LT$_{sc}$ (Hrs)'])
    kwargs = {'xycoords': 'figure fraction',
        'fontsize': fontsize-6}
    kwargs['xy'] = (0.03, 0.009)
    ax.annotate(eph_str,**kwargs)
    
    
    # Formatting colourbar
    divider = axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=0.15, pad=0.2)
    cb = fig.colorbar(im, extend='both', shrink=0.9, cax=cax, ax=ax)
    cb.set_label('Normalized'+'\n Degree of'+'\n Circular Polarization', fontsize=fontsize-2)
    cb.ax.tick_params(labelsize=fontsize-2)
    #cb.remove()
    
    #For adding horizontal lines at specific frequencies
    if frequency_lines is not None:
        for i in frequency_lines:
            ax.hlines(i, time[0], time[-1], colors = 'darkslategray',linewidth=1,linestyles='--', label='{}kHz'.format(i))
              
    #For plotting polygons onto spectrogram.
    if colour_in is not None:
        for shape in colour_in:
            shape_=shape.copy()
            shape_[:,0]=[mdates.date2num(datetime.fromtimestamp(i)) for i in shape_[:,0]]
            ax.add_patch(Polygon(shape_, color=(0.163625, 0.471133, 0.558148), linestyle='dashed',linewidth=4, alpha=1, fill=False))
        
    plt.close(fig)
    return ax


#dates you would like to plot spectrogram for 
data_start=pd.Timestamp('2006-01-01')
data_end=pd.Timestamp('2006-01-04')
year = datetime.strftime(data_start, '%Y')

if year == '2017':
    file = root+'input_data/SKR_2017_001-258_CJ.sav'
else: 
    file = root+'input_data/SKR_{}_CJ.sav'.format(year)
    
#Uncomment this for plotting polygons from .json file
#polygon_fp=root+"output_data/ML_lfes.json"
#saved_polys = get_polygons(polygon_fp,data_start, data_end)
#Uncomment this not to plot polygons
saved_polys=None


#Make figure with given number of panels
num_panels=2
plt.ioff()
fig,ax = plt.subplots(num_panels,1,figsize=(16,12))
fig.subplots_adjust(wspace=0.5,hspace=0.5)
#add in extra argument 'frequency_lines'= [100, 200...] to plot horizontal lines at given frequency
ax[0]=plot_flux(ax[0], data_start, data_end, file,colour_in=saved_polys)
ax[1]=plot_pol(ax[1], data_start, data_end, file,colour_in=saved_polys)
#ax[2]=plot_mask(ax[2],data_start, data_end, 's', file,polygon_fp)
#plt.show()
plt.savefig('test.png')


        

        
     
        