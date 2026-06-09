import os, sys
import tempfile
import h5py
import csv
import glob
import shutil
import numpy as np
import healpy as hp
#-------------------------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from astropy.io import fits
from configparser import ConfigParser
from scipy.stats import chisquare, chi2
from copy import deepcopy as dcopy
from functools import partial
#-------------------------------------------------------------------------------------------------------------------------------------
sys.path.insert(1,'/data/NSOARES/high_resolution_pipeline/swt_modified_scripts')
import gmca4im_lib2 as g4i
######################################################################################################################################
"""
Auxiliary library for the script "high_resolution_algorithm.py"
"""
######################################################################################################################################
def make_freq_file(freq_min, freq_max, hide_inptpath, seed_number):
	"""
	Creates a new FITS frequency file containing a subset of the standard BINGO
	frequency vector, selecting only the channels between freq_min and freq_max
	(inclusive). The file is saved in the HIDE input directory and named after
	the current seed number to avoid conflicts between pipeline iterations.

	Parameters:
	freq_min (float): Lower bound frequency in MHz. The channel nearest to this
					  value in the standard frequency vector will be used as the
					  first element of the new vector.
	freq_max (float): Upper bound frequency in MHz. The channel nearest to this
					  value in the standard frequency vector will be used as the
					  last element of the new vector.
	hide_inptpath (str): Path to the HIDE input maps directory where both the
						 standard frequency file and the new file will be located.
	seed_number (int): Seed identifier used to name the output file uniquely,
					   following the pattern freqs_bingo_{seed_number}.fits.

	Returns:
	tuple:
		- new_vector (numpy.ndarray): 1D array of selected frequencies in MHz.
		- new_freq_file (str): Full path to the generated FITS file.
	"""
	
	new_freq_file    = os.path.join(hide_inptpath, f"freqs_bingo_{seed_number}.fits")
	standart_freq_file = os.path.join(hide_inptpath, "freqs_bingo.fits")

	with fits.open(standart_freq_file) as hdul0:
		standart_freq_vec = np.array(hdul0[0].data)
		primary_header    = hdul0[0].header

	min_idx = int(np.argmin(np.abs(standart_freq_vec - freq_min)))
	max_idx = int(np.argmin(np.abs(standart_freq_vec - freq_max)))

	new_vector  = standart_freq_vec[min_idx : max_idx + 1]
	new_primary = fits.PrimaryHDU(data=new_vector, header=primary_header)
	fits.HDUList([new_primary]).writeto(new_freq_file, overwrite=True)

	return new_vector, new_freq_file
	
######################################################################################################################################
def change_ini(ini_path, section, key, value):
	"""
	Changes the value of a parameter (key) in the hide.ini file for the specified section.

	Parameters:
	ini_path (str): Path to the .ini file.
	section (str): Name of the section where the parameter is located.
	key (str): Name of the parameter to be changed.
	value (any): New value for the parameter.

	Returns:
	None
	"""
	config = ConfigParser()
	config.read(ini_path, encoding='utf-8')
	config[section][key] = str(value)
	with open(ini_path, 'w', encoding='utf-8') as f:
		config.write(f)

######################################################################################################################################
def combine_fits(input_pattern, nch, seed0, extra_array, output_filename):
	"""
	Combines several naivemap FITS files from different frequency channels
	into a single file containing all original maps in nch.

	Parameters:
	input_pattern (str): Naming pattern for input files, with placeholders for seed0 and ch.
	nch (int): Total number of frequency channels to compile.
	seed0 (int): Seed value to substitute in the input pattern.
	extra_array (array-like): Frequency vector to be saved in the ImageHDU.
	output_filename (str): Path and name for the output FITS file.

	Returns:
	str: Path to the generated FITS file.
	"""
	if nch == 0: nch += 1
		
	first_file = input_pattern.format(seed0=seed0, ch=0)
	with fits.open(first_file, memmap=True) as hdul:
		npix = hdul[0].data.shape[0]

	tmp = tempfile.NamedTemporaryFile(delete=False)
	tmp.close()
	mm = np.memmap(tmp.name, dtype='float64', mode='w+', shape=(nch, npix))

	for ch in range(nch):
		fn = input_pattern.format(seed0=seed0, ch=ch)
		with fits.open(fn, memmap=True) as hdul:
			data = hdul[0].data
			if data.shape != (npix,):
				raise ValueError(f"{fn}: expected shape ({npix},), found {data.shape}")
			mm[ch,:] = data

	hdu0 = fits.PrimaryHDU(data=extra_array)
	hdu1 = fits.ImageHDU(data=mm)
	fits.HDUList([hdu0, hdu1]).writeto(output_filename, overwrite=True)

	del mm
	os.remove(tmp.name)
	print(f"File saved '{output_filename}'")
	return output_filename

######################################################################################################################################
def execute_parallel_swt(input_map, J, nside, fits_dir, maps_dir, freq_vec, generate_maps, n_bins):
	"""
	Executes the Starlet Wavelet Transform (SWT) in parallel for each frequency band
	of the input FITS file, according to available CPU cores, and stores the resulting
	decomposition coefficients in FITS files. Optionally collects the output of the first
	frequency channel and generates images of each scale's coefficients, the reconstructed map,
	and the difference between the original and the reconstruction.

	Parameters:
	input_map (str): Path and name of the input FITS file.
	J (int): Number of desired scales.
	nside (int): nside parameter for HEALPix maps.
	fits_dir (str): Directory to save temporary FITS files and SWT decomposition files.
	maps_dir (str): Directory to save generated images.
	freq_vec (array-like): Frequency vector associated with the maps.
	generate_maps (bool): Flag indicating whether to generate images.
	n_bins (int): Number of frequency bins passed to extract_swt_ffits_memmap
              to define the memmap shape for the SWT output.

	Returns:
	None
	"""
	
	g4i.parallel_wavelet_transform(input_map, J, nside, fits_dir, freq_vec)
	
	if generate_maps:
		c, _ = g4i.extract_swt_ffits_memmap(fits_dir, family='C', nbins=n_bins)
		w, _ = g4i.extract_swt_ffits_memmap(fits_dir, family='W', nbins=n_bins)
		print(c.shape, w.shape)

		os.makedirs(maps_dir, exist_ok=True)
		
		# Plot the output map and performs a comparison with the original (for channel 0)
		plt.rcParams['axes.titlesize'] = 16
		plt.rcParams['font.size'] = 16
		plt.rcParams['figure.figsize'] = (16,8)
		
		with fits.open(input_map) as hdul:
			imap = hdul[1].data[0, :]
		
		for i in range(0, len(c[0])):
			
			plt.figure()
			
			hp.mollview(c[0, i, :],  
							norm='hist', 
							cmap='jet',  
							title=fr'coef. escala ($c_{{{i+1}}}$)', 
							unit='mK', format='%.2e')
				
			plt.savefig(os.path.join(maps_dir, f"output_map_0_c{i+1}.png"), bbox_inches='tight')
			plt.close()
		
		for j in range(0, len(w[0])):
		
			plt.figure()
			
			hp.mollview(w[0,j,:],
					   norm = 'hist',
					   cmap = 'jet',
					   title=fr'coef. starlet ($w_{{{j+1}}}$)',
					   unit='mK',  format='%.2e')

			plt.savefig(os.path.join(maps_dir, f"output_map_0_w{j+1}.png"), bbox_inches='tight')
			plt.close()
		
		hp.mollview(imap, norm = 'hist', cmap = 'jet', sub=231, title=r'original', unit='mK',  format='%.2e')
		
		reconstruction = np.sum(w[0, :(J-1), :], axis=0) + np.sum(c[0, (J-1):J, :], axis=0)
		hp.mollview(reconstruction, norm = 'hist', cmap = 'jet', sub=232, title=r'reconstructed' , unit='mK',  format='%.2e')
		
		hp.mollview( imap - reconstruction, norm = 'hist', cmap = 'jet', sub=233, title=r'difference', unit='mK',  format='%.2e')
		plt.savefig(os.path.join(maps_dir, f"output_map_0_diff"), bbox_inches='tight')
		plt.close()
######################################################################################################################################
def name_split(directory, seed0):
	"""
	Updates a directory path by replacing the seed number in its structure.

	Parameters:
	directory (str): Path containing the substring 'seed'.
	seed0 (int): New seed value to insert.

	Returns:
	str: Modified path with the new seed.
	"""

	left, right = directory.split("seed", 1)
	return (left + f"{seed0}" + right)

######################################################################################################################################
def execute_tod_diff(dir_path1, dir_path2, base_path_diff, corneta, hora, seed0, J, scale):
	"""
	Computes the iterative difference between a time-ordered data (TOD) generated by H&S from the original map
	and a second TOD generated by H&S from one of the SWT detail coefficients. Additionally,
	copies the pointing coordinate and parameter files from the base directory to the difference directory.

	Parameters:
	dir_path1 (str): Base directory of the first data set.
	dir_path2 (str): Directory pattern for the second set with placeholders.
	file_path_diff (str): Pattern for the output difference path.
	corneta (int): Horn identifier.
	hora (int): Hour when the TOD was recorded.
	seed0 (int): Seed value used.
	J (int): Total scales of the SWT decomposition
	scale (int): SWT scale.

	Returns:
	None
	"""
	# Formatting adjustments to directory names =>
	#--------------------------------------------------------------
	# date selected during processing in H&S
	date = "20200301"
	# name format of original TODs
	file_name = f"bingo_tod_horn_{corneta}_{date}_{hora:02d}0000.h5"

	if scale == (J-1):
		dir_path1 = os.path.join(name_split(dir_path1, seed0), "2020", "03", "01") + os.sep
		file_path1 = os.path.join(dir_path1, file_name)
	else:
		file_path1 = dir_path1.format(seed_number = seed0, scale = scale+1,
									 horn = corneta, date = date, hour = hora)
		
	dir_path2 = dir_path2.format(seed_number=seed0, coeff_type="W", scale=scale)
	dir_path2 = os.path.join(dir_path2, "2020", "03", "01") + os.sep
	file_path2 = os.path.join(dir_path2, file_name)
	
	file_path_diff = os.path.join(base_path_diff.format(seed_number=seed0, scale=scale), "2020", "03", "01") + os.sep
	os.makedirs(file_path_diff, exist_ok=True)
	
	# Copying configuration files
	#--------------------------------------------------------------
	for pat in [
		f"coord_bingo_{corneta}_{date}.txt",
		f"params_bingo_{corneta}_{date}.txt"
	]:
		for src in glob.glob(os.path.join(dir_path1, pat)):
			if os.path.isfile(src):
				shutil.copy2(src, file_path_diff)
				
	# Execute difference
	#--------------------------------------------------------------
	with h5py.File(file_path1, 'r') as file1, h5py.File(file_path2, 'r') as file2:
		F_1 = file1["FREQUENCY"][:]
		T_1 = file1["TIME"][:]
		P0_1 = file1['P']['Phase0'][:]
		P1_1 = file1['P']['Phase1'][:]
		F_2 = file2["FREQUENCY"][:]
		T_2 = file2["TIME"][:]
		P0_2 = file2['P']['Phase0'][:]
		P1_2 = file2['P']['Phase1'][:]
		if np.all(F_1 == F_2) and np.all(T_1 == T_2):
			P0_diff = P0_1 - P0_2
			P1_diff = P1_1 - P1_2
			with h5py.File(os.path.join(file_path_diff, file_name), 'w') as new_h5:
				new_h5.create_group("P")
				new_h5.create_dataset("FREQUENCY", data=F_1)
				new_h5.create_dataset("TIME", data=T_1)
				new_h5["P"].create_dataset("Phase0", data=P0_diff)
				new_h5["P"].create_dataset("Phase1", data=P1_diff)
		else:
			print(f"Inconsistency in file {file_name}")

	return file_path_diff
######################################################################################################################################
def equivalent_reconstruction_tod(swt_base_path, seed_number, horn, hour, recons_dir, J, scale):
	"""
	Generates reconstructions, equivalent to the calculated difference, from the SWT coefficients
	stored in FITS files for all frequency channels.

	Parameters:
	swt_base_path (str): Path pattern for SWT files with placeholders.
	seed_number (int): Seed used.
	horn (int): Horn identifier.
	hour (int): Hour identifier.
	recons_dir (str): Output directory for reconstructions.
	J (int): Number of decomposition scales.
	scale (int): Current scale for reconstruction.

	Returns:
	None
	"""

	date = "20200301"
	fname = f"bingo_tod_horn_{horn}_{date}_{hour:02d}0000.h5"

	dir_C = swt_base_path.format(
		seed_number=seed_number,
		coeff_type='C',
		scale=(J-1)
	)
	C_path = os.path.join(dir_C, "2020", "03", "01", fname)
	with h5py.File(C_path, 'r') as fC:
		F = fC["FREQUENCY"][()]
		T = fC["TIME"][()]
		C0 = fC["P"]["Phase0"][:]
		C1 = fC["P"]["Phase1"][:]

	sum_D1 = np.zeros_like(C1)
	for j in range(max(1, scale)): # modificação scale => scale- (27/10) - modificação => (max(1, scale)) (29/06)
		dir_D = swt_base_path.format(
			seed_number=seed_number,
			coeff_type='W',
			scale=j       
		)
		D_path = os.path.join(dir_D, "2020", "03", "01", fname)
		with h5py.File(D_path, 'r') as fD:
			sum_D1 += fD["P"]["Phase1"][:]

	recons_P0 = C0
	recons_P1 = C1 + sum_D1

	dir_recons = recons_dir.format(seed_number=seed_number,
								   scale = scale)
	os.makedirs(dir_recons, exist_ok=True)
	out_recons = os.path.join(dir_recons, fname)
	os.makedirs(os.path.dirname(out_recons), exist_ok=True)
	with h5py.File(out_recons, 'w') as frec:
		frec.create_group("P")
		frec.create_dataset("FREQUENCY", data=F)
		frec.create_dataset("TIME",      data=T)
		frec["P"].create_dataset("Phase0", data=recons_P0)
		frec["P"].create_dataset("Phase1", data=recons_P1, compression="gzip")

	return out_recons

######################################################################################################################################
def process_horn(hour_number, horn, scale, J, hide_outpath, last_base_diff_path, 
	swt_base_path, base_diff_path, seed_number, recons_base_path):
	"""
	Execute for each scale, each horn and each hour: TOD original - TOD_W{J} sequencially; 
	Where W{J} is the detail SWT coefficient for a scale J.
	After that, calculate the equivalent reconstruction TOD. 

	Parameters:
	hour_number (int): Total hours.
	horn (int): Horn indicator.
	scale (int): Actual scale of the SWT decomposition. 
	J (int): Total scales of SWT decompostition.
	hide_outpath (str): HIDE output directory.
	last_base_diff_path (str): Base path of last evaluate difference.
	swt_base_path (str): Base path of the TODs generate from SWT output.
	base_diff_path (str): Base path of the difference between original TODs and SWT TODs.
	seed_number (int): Seed identifier.
	recons_base_path (str): Base path of the equivalent reconstruction to the difference TOD.
	
	Returns:
	None
	"""
	
	for hour in hour_number:

		if scale == (J-1): 
			dir_path1 = dcopy(hide_outpath)
		else: 
			dir_path1 = dcopy(last_base_diff_path)	
 
		#-------------------------------------------------	
		diff_path = execute_tod_diff(dir_path1, swt_base_path, base_diff_path, horn, hour, seed_number, J, scale)
		#-------------------------------------------------	
		# Execute equivalent reconstruction 
		recons_path = equivalent_reconstruction_tod(swt_base_path, seed_number, horn, hour, recons_base_path, J, scale) 

##########################################################################################################################################
def open_hdf5_chi2_results(file_path):
	"""
	Opens an HDF5 file and returns the 'chi2' and 'rmse' groups as dictionaries
	containing h5py dataset objects (without loading data into RAM).

	Parameters:
	file_path (str or Path): Path to the .h5 file.

	Returns:
	tuple (h5_file, chi2_dict, rmse_dict, dof_dict):
	- h5_file: h5py.File object (must be closed by the caller);
	- chi2_dict: dictionary {dataset_name: Dataset object} from the 'chi2' group;
	- rmse_dict: dictionary {dataset_name: Dataset object} from the 'rmse' group;
	- dof_dict: dictionary {dataset_name: Dataset object} from the 'dof' group.

	"""
	hd5f = h5py.File(file_path, 'r')
	chi2_dict = {}
	rmse_dict = {}
	dof_dict = {}

	if 'chi2' in hd5f:
		chi2_group = hd5f['chi2']
		for name, dataset in chi2_group.items():
			chi2_dict[name] = dataset 
	else:
		print(" 'chi2' group not found.")

	if 'rmse' in hd5f:
		rmse_group = hd5f['rmse']
		for name, dataset in rmse_group.items():
			rmse_dict[name] = dataset
	else:
		print(" 'rmse' group not found.")

	if 'dof' in hd5f:
		dof_group = hd5f['dof']
		for name, dataset in dof_group.items():
			dof_dict[name] = dataset 
	else:
		print(" 'dof' group not found.")

	return hd5f, chi2_dict, rmse_dict, dof_dict
#############################################################################################################################################
def backup_diff_directory(base_diff_path, seed_number, scale):
	"""
	Creates a backup copy of the TOD difference directory for a given scale,
	preserving its state before the next scale's subtraction is applied.

	The backup is stored alongside the original directory with a '_backup' suffix.
	This allows the directory to be restored later if the chi-square test indicates
	that the current scale did not improve the data modeling.

	Parameters
	----------
	base_diff_path : str
		Base path pattern for the difference directory, with placeholders
		{seed_number} and {scale}.
	seed_number : int
		Seed identifier used to format the path.
	scale : int
		SWT scale whose difference directory will be backed up.

	Returns
	-------
	backup_path : str
		Path to the created backup directory. Returns None if the source
		directory does not exist.
	"""
	source_path = base_diff_path.format(seed_number=seed_number, scale=scale)
	backup_path = source_path.rstrip('/') + '_backup'

	if not os.path.exists(source_path):
		print(f"Backup skipped: source directory not found at '{source_path}'.")
		return None

	if os.path.exists(backup_path):
		shutil.rmtree(backup_path)

	shutil.copytree(source_path, backup_path)
	print(f"Backup created at '{backup_path}'.")
	return backup_path

#############################################################################################################################################
def restore_diff_directory(base_diff_path, seed_number, scale):
	"""
	Restores the TOD difference directory for a given scale from its backup,
	reverting any subtraction applied during the current scale's processing.

	The backup directory (created by backup_diff_directory) replaces the current
	difference directory. Should be called when the chi-square test indicates that
	the current scale did not improve the data modeling.

	After a successful restore, the backup directory is removed.

	Parameters
	----------
	base_diff_path : str
		Base path pattern for the difference directory, with placeholders
		{seed_number} and {scale}.
	seed_number : int
		Seed identifier used to format the path.
	scale : int
		SWT scale whose difference directory will be restored.

	Returns
	-------
	None
	"""
	source_path = base_diff_path.format(seed_number=seed_number, scale=scale)
	backup_path = source_path.rstrip('/') + '_backup'

	if not os.path.exists(backup_path):
		print(f"Restore skipped: backup directory not found at '{backup_path}'.")
		return

	if os.path.exists(source_path):
		shutil.rmtree(source_path)

	shutil.copytree(backup_path, source_path)
	shutil.rmtree(backup_path)
	print(f"Directory '{source_path}' restored from backup and backup removed.")

#############################################################################################################################################
def calculate_global_chi2_per_bin(chi2_dict, dof_dict, n_ch):
	"""
	Calculates the global reduced chi-square for a single frequency channel,
	summing the chi-square values and degrees of freedom across all horns.

	For a given frequency channel index (n_ch), iterates over all horns present
	in chi2_dict, accumulates the chi-square values and the corresponding degrees
	of freedom, and returns the reduced chi-square (chi² / dof) for that channel.

	Parameters
	----------
	chi2_dict : dict
		Dictionary mapping horn identifiers to 2D arrays of shape (n_hours, n_bins),
		where each entry contains the chi-square values for that horn across all
		hours and frequency channels.
	dof_dict : dict
		Dictionary mapping horn identifiers to 2D arrays of shape (n_hours, n_bins),
		containing the degrees of freedom associated with each chi-square value.
	n_ch : int
		Index of the frequency channel for which the global reduced chi-square
		will be calculated. Must be in the range [0, n_bins - 1].

	Returns
	-------
	reduced_chi2 : float
		Global reduced chi-square for channel n_ch, defined as the total
		chi-square divided by the total degrees of freedom, summed over all horns.
	"""
	global_chi2_array = 0 
	global_dof_array = 0
	
	for horn in chi2_dict.keys():
		
		chi2_data = chi2_dict[horn]
		chi2_array = chi2_data[:,n_ch]

		dof_data = dof_dict[horn]
		dof_array = dof_data[:,n_ch]

		global_chi2_array += np.sum(chi2_array)
		global_dof_array += np.sum(dof_array)
	
	reduced_chi2 = global_chi2_array/global_dof_array

	return reduced_chi2
#############################################################################################################################################
def store_data_model(data_model_base_path, swt_base_path, positive_scales, seed_number, horn_number, hour_number, J):

	"""
	Builds and stores the final data model TOD for a given horn and hour by
	combining the smoothing coefficient TOD (C_{J-1}) with the sum of detail
	coefficient TODs (W_j) for all scales that improved the chi-square test.

	The resulting TOD represents the best approximation of the original signal
	obtained from the SWT decomposition up to the current iteration.

	Parameters
	----------
	data_model_base_path : str
		Base path pattern for the output data model directory,
		with placeholder {seed_number}.
	swt_base_path : str
		Path pattern for SWT TOD files, with placeholders
		{seed_number}, {coeff_type}, and {scale}.
	positive_scales : list of int
		List of SWT scale indices whose inclusion improved the
		chi-square test value and should be included in the model.
	seed_number : int
		Seed identifier used to format the output path.
	horn_number : list 
		List with all horns identifier.
	hour_number : list
		List with all hours identifier for the TODs file.
	J : int
		Total number of SWT decomposition scales. Used to locate
		the smoothing coefficient C_{J-1}.

	Returns
	-------
	out_data_model : str
		Full path to the saved HDF5 file containing the data model TOD.
	"""

	for horn in range(horn_number):
		for hour in range(hour_number):

			date = "20200301"
			fname = f"bingo_tod_horn_{horn}_{date}_{hour:02d}0000.h5"
		
			dir_C = swt_base_path.format(
				seed_number=seed_number,
				coeff_type='C',
				scale=(J-1)
			)
			C_path = os.path.join(dir_C, "2020", "03", "01", fname)
			with h5py.File(C_path, 'r') as fC:
				F = fC["FREQUENCY"][()]
				T = fC["TIME"][()]
				C0 = fC["P"]["Phase0"][:]
				C1 = fC["P"]["Phase1"][:]
		
			sum_D1 = np.zeros_like(C1)
			for scale in positive_scales:
				dir_D = swt_base_path.format(
						seed_number=seed_number,
						coeff_type='W',
						scale=scale      
				)
				D_path = os.path.join(dir_D, "2020", "03", "01", fname)
				with h5py.File(D_path, 'r') as fD:
					sum_D1 += fD["P"]["Phase1"][:]
			
			tod_model = C1 + sum_D1
			
			data_model_dir = os.path.join(data_model_base_path.format(seed_number=seed_number), "TODs") + os.sep
			out_data_model = os.path.join(data_model_dir, fname)
			os.makedirs(os.path.dirname(out_data_model), exist_ok=True)
			with h5py.File(out_data_model, 'w') as fmodel:
				fmodel.create_group("P")
				fmodel.create_dataset("FREQUENCY", data=F)
				fmodel.create_dataset("TIME",      data=T)
				fmodel["P"].create_dataset("Phase0", data=C0)
				fmodel["P"].create_dataset("Phase1", data=tod_model, compression="gzip")
		
	return out_data_model, data_model_dir
#############################################################################################################################################

#############################################################################################################################################
# 
#############################################################################################################################################