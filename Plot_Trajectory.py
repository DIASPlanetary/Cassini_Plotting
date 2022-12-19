# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 11:35:15 2022

@author: eliza
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import configparser

config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']


def make_full_trajectory_plot(date1, date2):
    year = datetime.datetime.strftime(date1, '%Y')
    #Load in trajectory data by year.
    ephem = pd.read_csv(output_data_fp + '/trajectory{}.csv'.format(year), parse_dates=(['datetime_ut']))
    ephem = ephem.loc[ephem['datetime_ut'].between(date1, date2),:]
    ephem = ephem.drop_duplicates(subset='datetime_ut', keep='first', inplace=False)
    # for full trajectory plot load in trajectory combined for all years.
    #ephem=pd.read_csv(output_data_fp + '/trajectorytotal.csv', parse_dates=(['datetime_ut']))
    ephem = ephem.sort_values(by=['datetime_ut']).reset_index(drop=True)
    #Convert from datetime to doyfrac
    dtimes = [(i.timetuple().tm_yday + i.timetuple().tm_hour/24 + i.timetuple().tm_min /
               1440 + i.timetuple().tm_sec/86400) for i in ephem['datetime_ut']]
    plt.ioff()
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(30, 15), sharex=True)
    fig.suptitle('Spacecraft Trajectory {}'.format(year), fontsize=20, y=0.92)
    
    ax[0].tick_params( labelsize=15)
    ax[1].tick_params( labelsize=15)
    ax[2].tick_params( labelsize=15)
    #Plot Range
    ax[2].plot(dtimes, ephem['range'])
    ax[2].set_yticks([0,20,40,60],fontsize=15)
    ax[2].set_ylabel('Range (R$_s$)', fontsize=15)
    ax[2].set_ylim(ephem['range'].min(),ephem['range'].max())
    ax[2].set_xlim(min(dtimes),max(dtimes))
    ax[2].set_xlabel('DOY ({})'.format(year), fontsize=15)
    
    #Plot Latitude
    ax[1].plot(dtimes, ephem['lat'])
    ax[1].set_ylabel('Latitude ($^{\circ}$)', fontsize=15)
    ax[1].set_ylim(min(ephem['lat']), -min(ephem['lat']))
    ax[1].set_xlim(min(dtimes),max(dtimes))
    #ax[1].fill_between(dtimes, min(ephem['Latitude']), -20,alpha=0.3, color='gray')
    #ax[1].fill_between(dtimes, 20, -min(ephem['Latitude']),alpha=0.3, color='gray')
    #ax[1].hlines(-20, min(dtimes), max(dtimes))
    
    #Plot local time
    ax[0].set_yticks([0, 6, 12, 18, 24])
    ax[0].scatter(dtimes, ephem['localtime'], s=0.7)
    ax[0].set_ylabel('Local Time (Hrs)', fontsize=15)
    ax[0].tick_params(axis='both', labelsize=15)
    ax[0].set_ylim(0, 24)
    ax[0].set_xlim(min(dtimes),max(dtimes))
    

    return ax

day1str = '20060101'
day1 = pd.Timestamp(day1str)
day2str = '20061231'
day2 = pd.Timestamp(day2str)
ax = make_full_trajectory_plot(day1, day2)
plt.show()


