#########################################################################################################################
import os, sys
import random
import numpy as np
import subprocess
import psutil as ps
import shutil
import multiprocessing as mp
#-----------------------------------------------------------
from pathlib import Path
from copy import deepcopy as dcopy
#-----------------------------------------------------------
from Chi2_analysis_for_HideSeek_data import run_parallel 
#-----------------------------------------------------------
# Import the auxiliary paths config file which contains all the directories
# and their descriptions
from paths_config import * 
#-----------------------------------------------------------
# Import the auxiliary library made especially for this script
sys.path.insert(1,'/data/NSOARES/high_resolution_pipeline/codes')
import auxiliary_lib as auxf
##########################################################################################################################
""" 
 This script aims to evaluate how well the decomposition using
 the Starlet wavelet transform approximates the original data
 and to quantify this analysis through a chi-square test.
 ------------------------------------------------------------
 Documentation for auxiliary functions can be found in the 
 file "auxiliary_lib.py". 

""" 
##########################################################################################################################
# About Directories 
# OBS: All outputs will be in the "output_tod_iter" directory and 
# will be organized by seed number, where a random seed corresponds 
# to the first round of the iteration loop and with each new round 
# of the loop its value is increased by one. So that with each use 
# of this script the outputs will be organized by different seed 
# number sequences.
# OBS: Starlet Wavelet Transform abbreviation -> SWT 
##########################################################################################################################

# Change hide.ini parameters (to run HIDE)
# Select a random seed
section_0 = "General"
def_seed0 = "seed0"
seed_number = random.randint(100, 9999)
# seed_number = 9221

section_1 = "Output"
outpath = "output_path"

# Select input map and input frequency file
section_2 = "AstroSignal"
filename_key = "astro_signal_file_name"
filename = "FG_I_128_980mhz1260mhz_30bins_full_L0.fits"
freq_key = "astro_signal_freq_file_name"
freq_filename = f"freqs_bingo_{seed_number}.fits"

section_3 = "Hitmap"
output_map_path = "path"
hide_out_path = auxf.name_split(hide_outpath, seed_number)
os.makedirs(hide_out_path, exist_ok = True)
channels_key = "n_channels_to_plot"
n_channels = 1
idx_channels = 0 # To the case that n_channels is equal to 1

section_4 = "Beam"
nside_key = "beam_nside"
nside_input_map = 128
freq_min_key = "beam_frequency_min"
freq_max_key = "beam_frequency_max"
beam_n_channels_key = "beam_number_channels"
freq_min = 980
freq_max = 989.33

freq_vec, _ = auxf.make_freq_file(freq_min, freq_max, hide_inptpath, seed_number)

auxf.change_ini(ini_path, section_0, def_seed0, seed_number)
auxf.change_ini(ini_path, section_1, outpath, hide_out_path)
auxf.change_ini(ini_path, section_2, filename_key, filename)
auxf.change_ini(ini_path, section_2, freq_key, freq_filename)
auxf.change_ini(ini_path, section_3, output_map_path, hide_out_path)
auxf.change_ini(ini_path, section_3, channels_key, n_channels)
auxf.change_ini(ini_path, section_3, def_seed0, seed_number)
auxf.change_ini(ini_path, section_4, nside_key, nside_input_map)
auxf.change_ini(ini_path, section_4, freq_min_key, freq_min)
auxf.change_ini(ini_path, section_4, freq_max_key, freq_max)

##########################################################################################################################
# Run HIDE
subprocess.run(["python", "run_hide.py", "bingo.py", "bingo_horn", "0", "140"], check=True, cwd=hide_dir)
#------------------------------------------------------------
# Start iteration process
# Main iteration loop: Processes the original TODs into naive maps, 
# performs the decomposition with SWT, obtaining the WJ and CJ coefficients
# for all J scales in FITS format, processes the coefficients with HIDE, and 
# obtains the TODs from SWT. Performs a comparison between the SWT TODs and the
# original ones, starting with the coarsest scale, calculates the difference 
# between them, and compares it with the equivalent reconstruction made with the 
# SWT coefficients. Afterward, quantifies this comparison using a chi-square test 
# for each hour of each TOD horn  
#------------------------------------------------------------
idx_main_iteration = 0
continue_main_iter = True
chi2_global_values = []
while continue_main_iter: # parte do critério de comparação de chi² do final

	# Run hitmap.py (with the frequency n_channels in parallel)
	subprocess.run(["python", "hitmap_bin_parallel.py"], check=True, cwd=hide_dir)
	#------------------------------------------------------------
	# Changing the hitmap output format to a SWT input format 
	naivemap_fname = f"naivemap_signal+noise_SEED{seed_number}_nch{idx_channels}_1d.fits" 
	# freq_vec = np.array([ 980.0, 989.33,  998.67, 1008.0,   1017.33, 1026.67, 1036.0, 1045.33,
	# 1054.67, 1064.0,   1073.33, 1082.67, 1092.0,   1101.33, 1110.67, 1120.0,  1129.33, 1138.67,
	#  1148.0,   1157.33, 1166.67, 1176.0,   1185.33, 1194.67, 1204.0,   1213.33, 1222.67,
	#  1232.0,   1241.33, 1250.67, 1260.0])
	hitmap_out_path = auxf.name_split(hitmap_outpath, seed_number)

	naivemap = auxf.combine_fits(os.path.join(hitmap_out_path, naivemap_fbase), nch=n_channels,
					  seed0=seed_number, extra_array=freq_vec, output_filename=os.path.join(hitmap_out_path, naivemap_fname))

	##########################################################################################################################
	# Execute SWT for a hitmap output
	J = 9 # number of scales decomposition
	nside = nside_input_map
	generate_maps = True # Either plots the output maps and performs a 
						  # comparison with the input map (True), or does nothing (False).
	# 					  # If maps are generated, they will be in the directory "swt_maps_path"
	SWT_maps_path = auxf.name_split(swt_maps_path, seed_number)
	auxf.execute_parallel_swt(naivemap, J, nside, hide_inptpath, SWT_maps_path, freq_vec, generate_maps, n_channels)

	##########################################################################################################################
	# Converting naivemap decomposition in TOD format
	coefficients = ["W", "C"]
	convert = False # Either convert all smoothing coefficients (C{J}) to TOD (True), 
				   # or convert only the last one (False), needed for reconstruction.
	for coeff_type in coefficients:

		for scale in range(J):

			if coeff_type == "C" and convert:
				pass
			elif coeff_type == "C" and scale < J-1:
				continue

			naivemap_swt = nm_base_swt_path.format(coeff_type=coeff_type,
												   scale=scale)
			swt_tod_path = swt_base_path.format(seed_number=seed_number,
															coeff_type=coeff_type,
															scale=scale)
			os.makedirs(swt_tod_path, exist_ok=True)

			# Change hide.ini parameters (output path and input filename)
			auxf.change_ini(ini_path, section_1, outpath, swt_tod_path)
			auxf.change_ini(ini_path, section_2, filename_key, naivemap_swt)

			# Run HIDE to produce the TODs
			subprocess.run(["python", "run_hide.py", "bingo.py", "bingo_horn", "0", "140"], check=True, cwd=hide_dir)

	##########################################################################################################################  
	# To quantify the accuracy of the modeling with SWT at each scale, a comparison 
	# will be made (with Chi-squared test), initially, between the difference of 
	# the original TOD and the first coarsest scale (W{J}) and the reconstruction 
	# through the remaining scales, which in theory would be equivalent to this 
	# difference minus the computational error introduced by SWT decomposition.
	#------------------------------------------------------------ 
	chi2_per_scale = np.zeros(J)
	positive_scales = []
	aux = 0
	horn_number = 140
	hour_number = 24

	for scale in range(J-1,-1,-1):

		# Before computing the TOD difference for this scale, back up the difference
		# directory from the previous scale (scale+1) so it can be restored later
		# if the chi-square test shows that this scale did not improve the modeling.
		# The backup is skipped for the coarsest scale (J-1) because there is no
		# prior difference directory to preserve at that point.
		if scale < (J-1):
			auxf.backup_diff_directory(base_diff_path, seed_number, scale+1)

		# Execute for each (horn, hour) the TOD difference and the equivalent reconstruction TOD
		total_cores = mp.cpu_count()
		used_cores = np.sum(np.array(ps.cpu_percent(percpu=True, interval=0.1)) > 10)
		idle_cores = total_cores - used_cores

		args_list = [
			(range(hour_number), h, scale, J, hide_outpath, last_base_diff_path,
			 swt_base_path, base_diff_path, seed_number, recons_base_path)
			for h in range(horn_number)
		]

		with mp.Pool(processes=idle_cores) as pool:
			results = pool.starmap(auxf.process_horn, args_list)
		#--------------------------------------------------------------

		# Call and run the 'Chi2_analysis_for_H&S_data' library to execute the Chi² Analysis 
		# (between equivalent reconstrution and tod difference). The library will calculate 
		# the Chi2 (and optionally the RMSE), store the results in an hdf5 format file, and 
		# plot waterfalls with the results (optional).

		analysis_parameters = {
			"n_horns": horn_number,
			"n_hours": hour_number,
			"n_bins": n_channels,
			"obs_date": "20200301",
			"base_results_path": base_chi2_path.format(seed_number=seed_number,scale = scale),
			"base_obsTOD_path": recons_base_path.format(seed_number=seed_number,scale = scale),
			"base_expTOD_path": os.path.join(base_diff_path.format(seed_number=seed_number, scale=scale), "2020", "03", "01") + os.sep,
			"err_data": np.full(36000, 1).tolist(),
			"dof": None,
			"analysis_identifier": seed_number,
			"show_process_info": False,
			"rmse": True,
			"plot_waterfalls": True,
			"horns_to_plot": None,
		}

		run_parallel(analysis_parameters, num_nodes=1, total_num_process=60, slurm=False)

		# Checks if the chi2 value has not improved between two sequential scales
		#--------------------------------------------------------------

		# Opens an HDF5 file and returns the 'chi2' and 'rmse' (optionally) groups as dictionaries
		# containing h5py dataset objects (without loading data into RAM).  
		chi2_path = os.path.join(base_chi2_path.format(seed_number=seed_number,scale = scale), f"chi2_4_HS_analysis_{seed_number}") + os.sep
		results_file_hdf5, chi2_dict, rmse_dict, dof_dict = auxf.open_hdf5_chi2_results(Path(chi2_path) / "results/results.h5")

		# Calculating the reduced Chi² value for the current scale.
		chi2_result = auxf.calculate_global_chi2_per_bin(chi2_dict, dof_dict, idx_channels)

		# Storing the current result.
		chi2_per_scale[scale] = dcopy(chi2_result)

		# Close the HDF5 file.
		results_file_hdf5.close()

		# Comparing the chi-square value obtained for this scale with that obtained for the previous scale.
		if scale == (J-1): continue # But if the actual scale is the first in the scales loop, 
									# there are no two values to compare.

		if chi2_per_scale[scale] <= chi2_per_scale[scale+1]:

			print(f"The data modeling, in the {seed_number} analysis performed using the {scale} \
			(from the SWT decomposition) scale, improved the Chi-square test value.")

			# The current scale improved the chi-square: discard the backup of the
			# previous scale's difference directory, keeping the updated version.
			
			# auxf.restore_diff_directory(base_diff_path, seed_number, scale)

			backup_to_discard = base_diff_path.format(seed_number=seed_number, scale=scale+1).rstrip('/') + '_backup'
			if os.path.exists(backup_to_discard): shutil.rmtree(backup_to_discard)

			positive_scales.append(scale)

			# Additionally, if the current scale improves the chi-square, it stores the TOD data for this scale.
			# This will serve as our final data model.
			x, model_path = auxf.store_data_model(data_model_base_path, swt_base_path, positive_scales, seed_number, horn_number, hour_number, J)

			# Plot the naive maps of the current state of the model
			path_model_hitmap = os.path.join(model_path, "maps") + os.sep
			auxf.change_ini(ini_path, section_0, def_seed0, seed_number)
			auxf.change_ini(ini_path, section_1, outpath, path_model_hitmap)
			auxf.change_ini(ini_path, section_3, output_map_path, path_model_hitmap)

			subprocess.run(["python", "hitmap_bin_parallel.py"], check=True, cwd=hide_dir)

		else:

			print(f"The data modeling, in the {seed_number} analysis performed using the {scale} \
			(from the SWT decomposition) scale, did not improve the Chi-square test value.\n")

			print(f"The scale {scale} will not be used in the data modeling. "
				  f"Restoring the difference directory to its state before scale {scale}.")

			# The current scale did not improve the chi-square: restore the difference
			# directory to its state before this scale's subtraction was applied.
			auxf.restore_diff_directory(base_diff_path, seed_number, scale+1)

# ##########################################################################################################################
	# It prepares the parameters so that the naive maps corresponding to the resulting set of TODs are produced.
	seed_number += 1
	last_J = positive_scales[-1] if positive_scales else J - 1
	new_input_hitmap = base_diff_path.format(seed_number=seed_number,
						 scale=last_J)
	
	hide_out_path = auxf.name_split(hide_outpath, seed_number) 
	os.makedirs(hide_out_path, exist_ok=True)
	
	auxf.change_ini(ini_path, section_0, def_seed0, seed_number)
	# auxf.change_ini(ini_path, section_1, outpath, new_input_hitmap)
	auxf.change_ini(ini_path, section_1, outpath, hide_out_path)
	auxf.change_ini(ini_path, section_3, output_map_path, new_input_hitmap)

# ##########################################################################################################################

	if positive_scales:
		chi2_global_values.append(chi2_per_scale[min(positive_scales)])
	else:
		print("No scale improved chi². Appending the first Chi² value")
		chi2_global_values.append(chi2_per_scale[J-1])

	if idx_main_iteration > 0 and chi2_global_values[idx_main_iteration] <= chi2_global_values[idx_main_iteration - 1]:
		idx_main_iteration += 1
	else:
		continue_main_iter = False
# ##########################################################################################################################
#                           ---------------------------END--------------------------------
# ##########################################################################################################################