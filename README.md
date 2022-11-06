# Cassini_Plotting
Number of functions for plotting Cassini RPWS and MAG data.
Instructions for download.

Save code to folder in the following format.
->Folder
	-> Cassini_Plotting
		-> 1_Load_Cass_Data.py
		-> 2_Make_Integrated_Power.py
		-> 3_Sliding_Window_Flux_Norm.py
		-> Plot_B_total.py
		-> Plot_Spectrogram.py
		-> Plot_Trajectory.py
		-> plotting_func.py
	-> input_data
		-> 2004_FGM_KRTP_1M.TAB
		...
		-> 2017_FGM_KRTP_1M.TAB

		-> 2004_FGM_KSM_1M.TAB
		...
		-> 2004_FGM_KSM_1M.TAB

		-> SKR_2004_CJ.sav
		..
		-> SKR_2017_001-258_CJ.sav

	-> output_data

Description of files.

1_Load_Cass_Data.py
Script that does the following:
- Makes .csv file for the combined MAG and trajectory data in krtp coordinates and ksm for each year saved to 'output_data' folder as 
  'trajectory2004' for 2004 for example. Also the combined data in the same format for every year is saved as 'trajectorytotal.csv' in 'output_data'.
- Makes .csv file for the RPWS data saved to 'output_data' folder for each year as 'skr_df2004.csv' for example.
- Makes dataframe of trajectory data interpolated to the same time values as the radio data, saved to 'output_data' as 'interpedtrajectory2004.csv' (example       	for 2004) for each year.
- Makes dataframe of radio data for each year combined with trajectory data interpolated to the same time values as radio data saved to folder 'output_data'
   as 'skr_traj_df_2004.csv' (for 2004..)


2_Make_Integrated_Power.py 

Script that makes a file with integrated power in frequency ranges  5-40, 40-100 and 100-600 for each year saved as 
