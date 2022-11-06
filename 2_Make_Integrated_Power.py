# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:29:53 2022

@author: eliza
"""
root="C:/Users/eliza/Desktop/Python_Scripts/"


from scipy.io.idl import readsav
import pandas as pd
from datetime import date
import numpy as np


def calc_integrated_pwr(p, f, f1, f2):
    f_indices = [i for i, j in enumerate(f) if (j > f1) & (j< f2)]
    p_range = p[f_indices, :]
    summed_p = p_range.sum(axis=0)
    return summed_p


def make_df(year):
    
    #Load SKR data.
    if year == 2017:
        file_data = root+'input_data/SKR_2017_001-258_CJ.sav'
    else: 
        file_data = root+'input_data/SKR_{}_CJ.sav'.format(year)
    
    file = readsav(file_data)
    doy = file['t']
    f = file['f']
    p = file['p']
    
    #Convert DOY from 1997 to DOY of given year.
    f_date = date(1997, 1, 1)
    l_date = date(year, 1, 1)
    delta = l_date - f_date
    days=delta.days
    doy = np.array(doy.copy())
    doy_ofthisyear = doy - days
    
    
    #Calculate the integrated power for the two frequency ranges.
    #The result will be two lists, each of the same dimensions as the DOY list.
    int_p_40_100 = calc_integrated_pwr(p, f, 40, 100)
    int_p_100_600 = calc_integrated_pwr(p, f, 100, 600)
    int_p_5_40 = calc_integrated_pwr(p, f, 5, 40)
    
    #Load in the Full Trajectory data for the year. 
    traj_fp = root+'output_data/interpedtrajectory{}.csv'.format(year)
    traj_df = pd.read_csv(traj_fp)
    rng, lat, lt = traj_df['Range'], traj_df['LT'], traj_df['Latitude']
    
    #Make dataframe to be used with Joe Reeds LFE thresholding code.
    df = pd.DataFrame({'5-40':int_p_5_40, '40-100': int_p_40_100, '100-600': int_p_100_600, 'Latitude': lat, 'LT': lt, 'Range(Sr)':rng})
    df.set_index(doy,inplace=True)
    
    return df


list_intpwr_files = []   
for i in range(2004, 2018, 1):
    df = make_df(i)
    fp = root+'output_data/intpwr_withtraj{}.csv'.format(i)
    df.to_csv(fp, index=False)
    list_intpwr_files.append(df)
    print(i)
        
ttl_dfs = pd.concat(list_intpwr_files)
ttl_dfs.to_csv(root+'output_data/intpwr_withtraj_2004_2017.csv', index=False)
