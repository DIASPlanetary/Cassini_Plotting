   # -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:50:14 2022

@author: eliza
"""
root="C:/Users/eliza/Desktop/Python_Scripts/"


from mpl_toolkits import axes_grid1
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import configparser

config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']


def plot_b_total(ax, mag_df, csize=12):

    ax.plot(mag_df['datetime_ut'],mag_df['btotal'], color='dimgrey')
    figure = ax.figure
    figure.subplots_adjust(bottom=0.35)
    
    #limits on axes
    ymax=mag_df['btotal'].max()
    x_start= mag_df['datetime_ut'].iloc[0]
    x_end= mag_df['datetime_ut'].iloc[-1]
    ax.set_ylim(0,ymax) #refine y limits
    ax.set_xlim(x_start, x_end)
    
    #Format and label axes 
    ax.tick_params(labelsize=csize-2)
    
    ax.set_xlabel('Time')
    ax.set_ylabel("$B_{TOTAL}$ $(nT)$", fontsize=csize)
    ax.set_title(f'{x_start} to {x_end}', fontsize=csize+2)
    
    dateFmt = mdates.DateFormatter('%Y-%m-%d\n%H:%M')
    ax.xaxis.set_major_formatter(dateFmt)
    
    
    
    # create empty space to fit colorbar in spectrogram panel
    # (effectively line up top/bottom panels)
    #Can use this when you want to add this as extra panel in spectrograms.
    #divider = axes_grid1.make_axes_locatable(ax)
    #cax = divider.append_axes("right", size=0.15, pad=0.2)
    #cax.set_facecolor('none')
    #for axis in ['top','bottom','left','right']:
     #   cax.spines[axis].set_linewidth(0)
    #cax.set_xticks([])
    #cax.set_yticks([])
    return ax


fig,ax = plt.subplots(1,1,figsize=(16,12))
data_start=pd.Timestamp('2006-01-01')
data_end=pd.Timestamp('2006-01-04')
year = datetime.strftime(data_start, '%Y')
fp = output_data_fp + '/trajectory{}.csv'.format(year)
mag_df=pd.read_csv(fp,parse_dates=['datetime_ut'])
ax=plot_b_total(ax, mag_df)
plt.show()