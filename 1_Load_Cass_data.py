# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 19:43:32 2021

@author: eliza
"""

import configparser
import numpy as np
import pandas as pd
from scipy.io.idl import readsav
from datetime import datetime
import scipy.interpolate as interpolate

config = configparser.ConfigParser()
config.read('configurations.ini')
input_data_fp = config['filepaths']['input_data']
output_data_fp= config['filepaths']['output_data']

def load_skr(CJ_SKR_YR_FP):
    data = readsav(CJ_SKR_YR_FP , python_dict=True)
    t_doy = data['t']
    n_freqs = len(data['f'])
    freqs = np.array(np.tile(data['f'], t_doy.shape[0]), dtype=np.float64)
    doy_one = pd.Timestamp(str(1997)) - pd.Timedelta(1, 'D')
    t_timestamp = pd.Series([doy_one + pd.Timedelta(t * 1440, 'm') for t in t_doy])
    t_timestamp=t_timestamp.dt.to_pydatetime()
    #datetime_ut = np.repeat(t_timestamp, n_freqs)
    #array_conv is a function, uses lambda function method with arr as the variable.
    #Inputs an array and flattens it to type 'F'
    array_conv = lambda arr: np.array(arr, dtype=np.float64).flatten('F')
    #Flux density in W/m^2Hz
    flux = array_conv(data['s'])
    flux = np.where(np.isclose(flux, 0., atol=1e-31) | (flux < 0.), np.nan, flux)
    #Normalised degree of circular polarization
    pol = array_conv(data['v'])
    #Power in W/sr
    pwr = array_conv(data['p'])
    return t_timestamp, freqs, n_freqs, flux, pol, pwr

def make_skr_df(CJ_SKR_YR_FP):
    t_timestamp, freqs,n_freqs,flux, pol, pwr = load_skr(CJ_SKR_YR_FP)
    n_sweeps = len(flux)/n_freqs
    sweeps = np.repeat(np.arange(n_sweeps),n_freqs)
    t_repeat = np.repeat(t_timestamp, n_freqs)
    df = pd.DataFrame({'sweep':sweeps,'datetime_ut': t_repeat, 'freq': freqs, 'flux':flux, 'pol': pol, 'pwr':pwr})
    return df
def traject(year):
    
    fp_krtp = input_data_fp + "/{}_FGM_KRTP_1M.TAB".format(year)

    df=pd.read_csv(fp_krtp, header=None,delim_whitespace=True)
    df = df.rename(columns={0: 'Time_isot',1:'B_r(nT)',2:'B_theta(nT)',3:'B_phi(nT)',
                            4:'B_total(nT)',5:'Range(R_s)',6:'Latitude',7: "East_Longitude",
                            8:'Local_Time',9:'NPTS'})
    func_tstmp_todoy = lambda x: ((x -datetime(int(datetime.strftime(x,'%Y')),1,1)).total_seconds()/86400) +1
    time_dt = df['Time_isot'].apply(lambda x: datetime.fromisoformat(x))
    doy_frac = list(map(func_tstmp_todoy, time_dt))
    fp_ksm = input_data_fp + "/{}_FGM_KSM_1M.TAB".format(year)
    df_ksm =pd.read_csv(fp_ksm, header=None,delim_whitespace=True)
    df_ksm = df_ksm.rename(columns={0: 'Time_isot',1:'B_x(nT)',2:'B_y(nT)',3:'B_z(nT)',
                            4:'B_total(nT)',5:'X(R_s)',6:'Y(R_s)',7: "Z(R_s)",
                            8:'Local_Time',9:'NPTS'})

    df_final =pd.DataFrame({'datetime_ut': time_dt, 'bphi_krtp':df['B_phi(nT)'],
                                    'br_krtp':df['B_r(nT)'],
                                    'btheta_krtp':df['B_theta(nT)'],
                                    'btotal':df['B_total(nT)'],
                                    'doyfrac':doy_frac,
                                    'lat':df['Latitude'],
                                    'localtime':df['Local_Time'],
                                    'range':df['Range(R_s)'],
                                    'xpos_ksm':df_ksm['X(R_s)'],
                                    'ypos_ksm':df_ksm['Y(R_s)'],
                                    'zpos_ksm':df_ksm['Z(R_s)']})
    return df_final  

    
    

def make_skr_traj_df(year,CJ_SKR_YR_FP):
    t_timestamp, freqs, n_freqs, flux, pol, pwr = load_skr(CJ_SKR_YR_FP)
    n_sweeps = len(flux)/n_freqs
    sweeps = np.repeat(np.arange(n_sweeps),n_freqs)
    t_repeat = np.repeat(t_timestamp, n_freqs)
    traj_df = traject(year)
    skr_df = pd.DataFrame({'datetime_ut':t_timestamp})
    interped_traj_df = interpolate_skr_data(traj_df, skr_df)
    lats, lts, r = interped_traj_df['Latitude'], interped_traj_df['LT'], interped_traj_df['Range']
    lats=np.repeat(lats, n_freqs)
    lts=np.repeat(lts, n_freqs)
    r=np.repeat(r, n_freqs)
    df = pd.DataFrame({'sweep':sweeps,'datetime_ut': t_repeat, 'freq': freqs, 'flux':flux, 'pol': pol, 'pwr':pwr,'Latitude':lats, 'LT': lts, 'Range': r})
    return df 

def find_data_gaps(year):
    df=traject(year)
    #df['dt']=df['Time_isot'].apply(lambda x : pd.Timestamp(x))
    #difference between timesteps in traj. data in minutes.
    timesteps=[(i-j).total_seconds()/60 for i,j in zip(df['datetime_ut'][1:],df['datetime_ut'][:-1])]
    #find indices of value where the time diff. between itself and subsequent is > 30 minutes.
    big_diffs=[i for i, j in enumerate(timesteps) if j >30]
    #subsequent indices
    second_bigdiffs = [i+1 for i in big_diffs]
    
    start_gaps = df.loc[big_diffs, 'datetime_ut']
    end_gaps = df.loc[second_bigdiffs,'datetime_ut']
    
    return start_gaps, end_gaps
def interpolate_skr_data(position_df, akr_df,
    pos_dt_label='datetime_ut'):
    """
    Based on A.Fogg's code for adding ephemeris data. Have changed bit for selecting
    relevant portion of position data with extra for interpolation
    """
    akr_df['unix'] = akr_df['datetime_ut'].astype(np.int64) / 1e9
    position_df['unix'] = position_df['datetime_ut'].astype(np.int64) / 1e9
    # Interpolate Local Time, Latitude and Range so trajectory data corresponds to the radio data.
    #using Alexandras code for interpolating LT.
    unwrapped_lts=np.unwrap(position_df['localtime'],period=24.0)
    func=interpolate.interp1d(position_df['unix'],unwrapped_lts,fill_value="extrapolate")
    unwrapped_interp_lt=func(akr_df['unix'])
    lt= unwrapped_interp_lt %24
    #Latitude
    lat = np.interp(akr_df['unix'], position_df['unix'], position_df['lat'])
    #Range.
    r = np.interp(akr_df['unix'], position_df['unix'], position_df['range'])
    traj_df = pd.DataFrame({'datetime_ut': akr_df['datetime_ut'], 'unix': akr_df['unix'],'LT':lt, 'Latitude':lat,'Range':r})
    
    start_gaps, end_gaps = find_data_gaps(year)
    for i, j in zip(start_gaps, end_gaps):
        traj_df.loc[traj_df['datetime_ut'].between(i,j),['LT', 'Range','Latitude']] = np.nan
    return traj_df



total_traj=[]  
total_skr_traj_df =[]

for year in range(2004,2018, 1):
    if year == 2017:
        CJ_SKR_YR_FP = input_data_fp + '/SKR_2017_001-258_CJ.sav'
    else: 
        CJ_SKR_YR_FP = input_data_fp + '/SKR_{}_CJ.sav'.format(year)
    

    #Dataframe with trajectory information.
    #columns are: 'datetime_ut','bphi_krtp' 'br_krtp' 'btheta_krtp' 'btotal' 'doyfrac'
     # 'lat' 'localtime'  'range' 'xpos_ksm' 'ypos_ksm' 'zpos_ksm'
    traj_df = traject(year)  
    total_traj.append(traj_df)
    fp = output_data_fp + '/trajectory{}.csv'.format(year)
    traj_df.to_csv(fp, index=False)
    
    #Load Radio data.
    t_timestamp, freqs, n_freqs, flux, pol, pwr = load_skr(CJ_SKR_YR_FP)
    radio_timestamps = pd.DataFrame({'datetime_ut':t_timestamp})
    
    #dataframe of interpolated trajectory values.
    #Columns are 'datetime_ut', 'unix', 'LT', 'Latitude', 'Range'
    interped_traj_df = interpolate_skr_data(traj_df, radio_timestamps)
    fp = output_data_fp + '/interpedtrajectory{}.csv'.format(year)
    interped_traj_df.to_csv(fp, index=False)

    #dataframe with columns 'sweep','datetime_ut','freq','flux','pol','pwr'
    skr_df = make_skr_df(CJ_SKR_YR_FP)
    fp = output_data_fp + '/skr_df{}.csv'.format(year)
    skr_df.to_csv(fp, index=False)

    #Save pandas dataframe with radio data and corresponding traj info.
    #Columns are: 'sweep':sweeps,'datetime_ut', 'freq','flux','pol','pwr','Latitude','LT','Range' (in krtp!)
    skr_traj_df = make_skr_traj_df(year,CJ_SKR_YR_FP)
    total_skr_traj_df.append(skr_traj_df)
    fp = output_data_fp + '/skr_traj_df_{}.csv'.format(year)
    skr_traj_df.to_csv(fp, index=False)
    
    
    print(year)
    
total_traj=pd.concat(total_traj)
total_traj.to_csv(output_data_fp + '/trajectorytotal.csv',index=False)   

total_skr_traj_df=pd.concat(total_skr_traj_df)
total_skr_traj_df.to_csv(output_data_fp +'/skr_traj_df_allyears.csv',index=False)  


    
    
    
    
    
