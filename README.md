# Cassini_Plotting
Number of functions for plotting Cassini RPWS and MAG data.

## Save code to folder in the following format. <br> 
- Folder <br> 
	- Cassini_Plotting
		- 1_Load_Cass_Data.py
		- 2_Make_Integrated_Power.py
		- 3_Sliding_Window_Flux_Norm.py
		- Plot_B_total.py
		- Plot_Spectrogram.py
		- Plot_Trajectory.py
		- plotting_func.py
	- input_data
		- 2004_FGM_KRTP_1M.TAB
		...
		- 2017_FGM_KRTP_1M.TAB

		- 2004_FGM_KSM_1M.TAB
		...
		- 2004_FGM_KSM_1M.TAB

		- SKR_2004_CJ.sav
		..
		- SKR_2017_001-258_CJ.sav

	- output_data

## Description of files. 
Need to run the first three files '1_Load_Cass_Data.py', ' 2_Make_Integrated_Power.py' and '3_Sliding_Window_Flux_Norm.py' before other scripts can be used. 
Detailed description of columns of .csv files etc are available in the scripts themselves.
### 1_Load_Cass_Data.py
Script that does the following:
- Makes .csv file for the combined MAG and trajectory data in krtp coordinates and ksm for each year saved to 'output_data' folder as 
  'trajectoryYEAR.csv' (e.g 'trajectory2004.csv'). The combined data in the same format for all years is saved as 'trajectorytotal.csv' in 'output_data'.
- Makes .csv file for the RPWS data saved to 'output_data' folder for each year as 'skr_dfYEAR.csv' (e.g 'skr_df2004.csv').
- Makes dataframe of trajectory data interpolated to the same time values as the radio data, saved to 'output_data' as 'interpedtrajectoryYEAR.csv' (e.g 'interpedtrajectoy2004.csv') for each year.
- Makes dataframe of radio data for each year combined with trajectory data interpolated to the same time values as radio data saved to folder 'output_data'
   as 'skr_traj_df_YEAR.csv' (e.g 'skr_traj_df_2004.csv'). The combined data in the same format for all years is saved as 'skr_traj_df_allyears.csv' in 'output_data'.


### 2_Make_Integrated_Power.py 

Script that makes a file with integrated power in frequency ranges 5-40, 40-100 and 100-600kHz along with trajectory data for each year saved as 'intpwr_withtrajYEAR.csv' ('intpwr_withtraj2004.csv'). The script also makes a file combining the data in each file to one .csv file saved in the 'output_data' folder saved as 'intpwr_withtraj_2004_2017.csv'.

### 3_Sliding_Window_Flux_Norm.py 

This .py file allows for bespoke colorbar normalisation of the RPWS flux density data based on the spacecraft location. The code in this file separates the data into two latitude regions >|5| degrees and <|5|degrees. It then further separates the data into local time bins (user can define these), so that colorbar can be normalised based on the local times sampled by Cassini within the flux density spectrogram plotting window. Script saves the binned data into two files in the 'output_data' folder, 'lowlat_flux.npy' and 'highlat_flux.npy'. After running this script, calling the function `flux_norm` in plotting_func.py implements the sliding window flux normalisation for the 10th and 80th percentiles of data found at given latitude/local time.

### Plot_B_total.py 

Plots B total for date range specified. 

### Plot_Spectrogram.py

Plots spectrograms of flux density and polarization. Can draw polygons around features by providing .json file in TFCAT format. These features can also be masked using the `plot_mask` function.

### Plot Trajectory.py

Makes a three panel plot for given date ranges showing range, latitude and local time. 

### plotting_func.py 

Functions used for the Plot_Spectrogram.py script.
