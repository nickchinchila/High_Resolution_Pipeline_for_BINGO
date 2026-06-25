"""
This module provides the class 'Chi2_for_Hide_Seek_data', designed to perform
parallel statistical analysis (chi-square and optionally RMSE)
on Time-Ordered Data (TOD) from HIDE & SEEK. The analysis compares
observed data against expected (simulated) data for each horn and hour.


[10/03/2026] - Nicolli Soares
"""
# =================================================================================================
import os
import sys
import random
import glob
import numpy as np
import h5py
import copy
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LogNorm
from mpi4py import MPI
#==================================================================================================
#==================================================================================================
class Chi2_for_Hide_Seek_data:

	"""
	Main class to coordinate the statistical analysis of input TOD files.
	Handles MPI topology setup, task distribution, statistical calculations
	(Chi2 and RMSE), memory mapping, waterfalls plots, and final data consolidation into HDF5.
	"""
	
	def __init__(self, n_horns, n_hours, n_bins, obs_date, base_results_path, base_obsTOD_path, 
				 base_expTOD_path, err_data, dof = None, analysis_identifier = None,
				 show_process_info = False, rmse = False):
		"""
		Initializes the analysis environment, sets up MPI communicators, 
		defines data attributes, creates the output directory structure, 
		and prepares the memory-mapped files.
		
		Parameters ==>
		----------
		n_horns : int
			Number of horns to process.
		n_hours : int
			Number of hours of observation on each horn.
		n_bins : int
			Number of frequency bins in each TOD.
		obs_date : str
			Observation date used in filenames (format YYYYMMDD).
		base_results_path : str or Path
			Root directory where all outputs will be stored.
		base_obsTOD_path : str or Path
			Directory containing observed TOD files (input).
		base_expTOD_path : str or Path
			Directory containing expected TOD files (input).
		err_data : array type
			Array containing the error of the observed TOD data. 
		dof : int, optional
			Degrees of freedom for chi-square test. If None, uses (number_data_points - 1).
		analysis_identifier : int, optional
			Number identifier of the current analysis. If not provided, a random ID to create a unique output subdirectory.
		show_process_info : bool, optional (Not active)
			If True, prints debug info per process.
		rmse : bool,  optional
			If True, also compute RMSE alongside chi-square.
		"""
		
		# Initializing MPI
		self.comm = MPI.COMM_WORLD
		self.world_rank = self.comm.Get_rank()
		self.world_size = self.comm.Get_size()
		
		# Data attributes
		self.num_horns = n_horns
		self.num_hours = n_hours
		self.num_bins = n_bins 
		self.date = obs_date
		self.base_tod1_path = base_obsTOD_path
		self.base_tod2_path = base_expTOD_path
		self.err_data = np.array(err_data)
		self.dof = dof
		if analysis_identifier is None: analysis_identifier = random.randint(0, 1000)
		self.analysis_idf = analysis_identifier
		self.process_info = show_process_info
		self.calculate_rmse = rmse
		# self.min_valid_samples = min_valid_samples

		# Base name for the BINGO TOD files
		self.base_tod_fname = "bingo_tod_horn_{horn}_{date}_{hour:02d}0000.h5"
		
		# Base paths for all results
		self.results_path = Path(base_results_path) / f"chi2_4_HS_analysis_{self.analysis_idf}"
		self.base_memmap_path = Path(self.results_path) / "memmaps"
		self.base_waterfall_path = Path(self.results_path) / "waterfalls"
		self.base_hdf5_path = Path(self.results_path) / "results"

		# Creating base paths
		if self.world_rank == 0:
			os.makedirs(self.base_memmap_path, exist_ok=True)
			os.makedirs(self.base_waterfall_path, exist_ok=True)
			os.makedirs(self.base_hdf5_path, exist_ok=True)
		self.comm.Barrier()

		# Initializing empty numpy memmaps
		if self.world_rank == 0:
			self.create_memmaps()
		self.comm.Barrier()
		
		# Extract mpi topology and joint the processes per node
		self._setup_mpi_topology()

	def _setup_mpi_topology(self):

		"""
		MPI topology: This method identifies which ranks run on 
		the same physical node, storing the local ranks, 
		local rank index, and node size. This allows for an 
		optimized two-level parallelization strategy (horns and hours).
		"""
		
		# Extract node name
		self.hostname = MPI.Get_processor_name()
		all_hosts = self.comm.allgather(self.hostname)

		# Obtain a list of nodes with their respective processes
		nodes = {}
		for rank, host in enumerate(all_hosts):
			if host not in nodes:
				nodes[host] = []
			nodes[host].append(rank)

		# Joint the ranks per node
		self.actual_node = None
		self.local_ranks = []
		self.node_size = 0
		
		for node, ranks in nodes.items():
			if self.world_rank in ranks:
				self.actual_node = node
				self.local_ranks = ranks
				self.local_rank = ranks.index(self.world_rank)
				self.node_size = len(ranks)
				break
		
		self.num_nodes = len(nodes)
		self.nodes = nodes
		
	def tasks_coordinator(self):

		"""
		Distribute (horn, hour) pairs across all MPI processes.
		Two-level distribution:
		1 - Horns are evenly assigned to nodes (first level).
		2 - Within each node, hours are evenly assigned to local ranks (second level).
		After calling this method, 'self.actual_horns' and 'self.actual_hours'
		contain the list of (horn, hour) pairs that the current process must handle.
		"""
		
		# Horn index
		all_horns = list(range(self.num_horns))
		# Hour index
		all_hours = list(range(self.num_hours))

		# Distributes the horns index evenly per nodes (first parallelization level)
		if self.world_rank == 0:
			node_horn_assignments = {}
			for i, horn in enumerate(all_horns):
				node_idx = i % self.num_nodes
				node_name = list(self.nodes.keys())[node_idx]
				if node_name not in node_horn_assignments:
					node_horn_assignments[node_name] = []
				node_horn_assignments[node_name].append(horn)
		else:
			node_horn_assignments = None

		# Broadcast the nodes assignments from rank 0 to all processes
		node_horn_assignments = self.comm.bcast(node_horn_assignments, root=0)
		
		actual_node_horns = node_horn_assignments[self.actual_node]

		# Distributes the hours index evenly per local processes (second parallelization level)
		self.actual_horns = []
		self.actual_hours = []
		
		for horn_idx, horn in enumerate(actual_node_horns):
			for hour_idx, hour in enumerate(all_hours):
				if hour_idx % self.node_size == self.local_rank:
					self.actual_horns.append(horn)
					self.actual_hours.append(hour)
					
	def execute_analysis(self):

		"""
		Process all (horn, hour) pairs assigned to this rank.
		For each pair, it calls 'calculate_chi2_rmse_for_horn_hour' and saves
		the results to the corresponding memory-mapped file.
		Errors are caught and printed to avoid crashing the entire parallel job.
		"""
		
		for horn, hour in zip(self.actual_horns, self.actual_hours):
			# if (self.process_info):
			# 	print(f"[Process Info] [PID:{os.getpid():6d}] Global Rank: {self.world_rank:3d} \
			# 	Node: {self.actual_node:15s} Local Rank: {self.local_rank:2d} Processing horn={horn}, hour={hour}")
			try:
				# Execute chi squared test between two TODs of a corresponding (horn, hour)
				chi2_per_hour_bin, rmse_per_hour_bin, dof_per_hour_bin = self.calculate_chi2_rmse_for_horn_hour(horn, hour)
				
				# Save data to memmap files
				self.save_to_memmap(horn, hour, chi2_per_hour_bin, rmse_per_hour_bin, dof_per_hour_bin)
				
			except Exception as e:
				print(f"Global ranking process {self.world_rank}: error processing horn {horn}, hour {hour}: {e}")

	def calculate_chi2_rmse_for_horn_hour(self, horn, hour):

		"""
		Main method to calculate the Chi-squared test and Root Mean Square Error (RMSE).
		Reads the respective HDF5 TOD files, ## the definition of valid data requires a better analysis ##
		and computes the statistics per frequency bin.

		Returns ==>
		-------
		chi2_per_hour_bin : ndarray, shape (n_bins,)
		dof : int
		rmse_per_hour_bin : ndarray or None
		"""
		
		# Construct the filepath for the observated TOD data
		tod1_name = self.base_tod_fname.format(horn=horn, 
					date=self.date, hour=hour)
		obs_TOD1 = os.path.join(self.base_tod1_path, tod1_name)
		
		# Construct the filepath for the expected TOD data
		tod2_name = self.base_tod_fname.format(horn=horn, 
					date=self.date, hour=hour)
		exp_TOD2 = os.path.join(self.base_tod2_path, tod2_name) 
		
		# Extracting data of the hdf5 files (TODs)
		with h5py.File(obs_TOD1, 'r') as f1, h5py.File(exp_TOD2, 'r') as f2:
			
			# Assuming the same frequency array for both files
			frequencies = f1['FREQUENCY'][()]   
			data_TOD1 = f1['P']['Phase1'][:]  # Shape: (n_bins, time)
			data_TOD2 = f2['P']['Phase1'][:]  # Shape: (n_bins, time)

		chi2_per_hour_bin = np.zeros(self.num_bins)
		dof_per_hour_bin = np.zeros(self.num_bins)
		# p_values = np.zeros(self.num_bins)
		
		if self.calculate_rmse: 
			rmse_per_hour_bin = np.zeros(self.num_bins)
		else:
			rmse_per_hour_bin = None
		
		if self.dof is None:  
			dof = data_TOD1.shape[1] - 1#- 1 # keep this -1 or not?
		else:
			dof = self.dof
		
		# Calculate Chi² for each frequency bin
		for freq_idx in range(self.num_bins):

			obs_data = data_TOD1[freq_idx, :]
			exp_data = data_TOD2[freq_idx, :]

			n_samples = len(obs_data)
			err = self.err_data[:n_samples]

			residuals = obs_data - exp_data
			chi2 = np.sum((residuals/ err)**2)
			
			# reduced_chi2 = chi2 / dof
			
			chi2_per_hour_bin[freq_idx] = chi2
			dof_per_hour_bin[freq_idx] = dof

			if self.calculate_rmse:
				rmse = np.sqrt(np.mean((obs_data - exp_data)**2))
				rmse_per_hour_bin[freq_idx] = rmse

		return chi2_per_hour_bin, rmse_per_hour_bin, dof_per_hour_bin

	def create_memmaps(self):

		"""
		Create memory-mapped files for chi² and (optionally) RMSE results.
	
		Files are created with mode 'w+' (overwrite if exists) and stored under
		'self.base_memmap_path'. Each horn has its own file, with shape (n_hours, n_bins).
		"""
			
		for horn in range(self.num_horns):

			chi2_path = os.path.join(self.base_memmap_path, f"chi2_{horn}.dat")
			chi2_mmap = np.memmap(chi2_path,
								  dtype='float64',
								  mode='w+',
								  shape=(self.num_hours, self.num_bins))
			del chi2_mmap

			dof_path = os.path.join(self.base_memmap_path, f"dof_{horn}.dat")
			dof_mmap = np.memmap(dof_path,
								 dtype='float64',
								 mode='w+',
								 shape=(self.num_hours, self.num_bins))
			del dof_mmap

			if self.calculate_rmse:
				rmse_path = os.path.join(self.base_memmap_path, f"rmse_{horn}.dat")
				rmse_mmap = np.memmap(rmse_path, 
									  dtype='float64', 
									  mode='w+',
									  shape=(self.num_hours, self.num_bins))
				del rmse_mmap

	def save_to_memmap(self, horn, hour, chi2_per_hour_bin, rmse_per_hour_bin, dof_per_hour_bin):

		"""
		Write the results for a single (horn, hour) into the memory-mapped files.
		Opens the memmap file for the given horn in 'r+' mode, updates the row
		corresponding to `hour`, and flushes changes to disk. This allows multiple
		processes to write concurrently to different rows of the same file without
		conflicts (since each process writes to a distinct hour index).
		"""
		
		chi2_path = os.path.join(self.base_memmap_path, f"chi2_{horn}.dat") 
		
		chi2_mmap = np.memmap(
			chi2_path, 
			dtype='float64', 
			mode='r+', 
			shape=(self.num_hours, self.num_bins)  
		)

		chi2_mmap[hour, :] = chi2_per_hour_bin
		chi2_mmap.flush()
		del chi2_mmap

		dof_path = os.path.join(self.base_memmap_path, f"dof_{horn}.dat")

		dof_mmap = np.memmap(
			dof_path,
			dtype='float64',
			mode='r+',
			shape=(self.num_hours, self.num_bins)
		)

		dof_mmap[hour, :] = dof_per_hour_bin
		dof_mmap.flush()
		del dof_mmap

		if self.calculate_rmse and rmse_per_hour_bin is not None:
			rmse_path = os.path.join(self.base_memmap_path, f"rmse_{horn}.dat")
			
			rmse_mmap = np.memmap(
				rmse_path, 
				dtype='float64', 
				mode='r+', 
				shape=(self.num_hours, self.num_bins)  
			)
			
			rmse_mmap[hour, :] = rmse_per_hour_bin
			rmse_mmap.flush()
			del rmse_mmap

	def generate_waterfalls(self, horns_to_plot):

		"""
		Generate waterfall plots for specified horns. Only rank 0 does this.
	
		Parameters ==>
		----------
		horns_to_plot : list of int or None
			If None, plot all horns.
		"""
		
		# Only Rank 0 should plot to avoid filesystem collision
		if self.world_rank != 0:
			return
 
		# If no specific horns provided, plot all of them
		if horns_to_plot is None:
			horns_to_plot = list(range(self.num_horns))
		
		# Ensure horns_to_plot is a list (in case a single int was passed)
		if isinstance(horns_to_plot, int):
			horns_to_plot = [horns_to_plot]
 
		print(f"Plotting {len(horns_to_plot)} waterfalls...")
	
		for horn in horns_to_plot:
			self.one_wtll(horn)

	def one_wtll(self, horn):

		"""
		Create a waterfall plot for a single horn.
		Loads chi² and (if enabled) RMSE arrays from memmap files.
		Uses logarithmic scale for chi² and handles NaN values (shown as black).
		Saves the figure as PNG under 'self.base_waterfall_path'.
		"""

		chi2_path = os.path.join(self.base_memmap_path, f"chi2_{horn}.dat")
		
		try:
			chi2_data = np.memmap(chi2_path, dtype='float64', mode='r+', shape=(self.num_hours, self.num_bins))
			chi2_array = np.array(chi2_data)
			
		except FileNotFoundError as e:
			print(f"Error loading Chi² memory-mapped files: {e}")
			return
		
		if self.calculate_rmse:
			rmse_path = os.path.join(self.base_memmap_path, f"rmse_{horn}.dat")
			try:
				rmse_data = np.memmap(rmse_path, dtype='float64', mode='r+', shape=(self.num_hours, self.num_bins))
				rmse_array = np.array(rmse_data)
			except FileNotFoundError as e:
				print(f"Error loading RMSE memory-mapped files: {e}")
				return
			
			fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
		else:
			fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
		
		# Isolate valid values 
		valid_chi2_data = chi2_array[~np.isnan(chi2_array) & (chi2_array > 0)]
		
		if len(valid_chi2_data) > 0:
			vmin_chi2 = np.min(valid_chi2_data)
			vmax_chi2 = np.max(valid_chi2_data)
		else:
			vmin_chi2, vmax_chi2 = 0.1, 1.0

		# Copy colormap and set bad values (NaNs) to black
		cmap_chi2 = copy.copy(plt.cm.viridis)
		cmap_chi2.set_bad(color='black')
		
		# Plot Chi2 waterfall
		im1 = ax1.imshow(chi2_array, aspect='auto', cmap=cmap_chi2, 
						 extent=[0, self.num_bins, 0, self.num_hours], origin='lower',
						 norm=LogNorm(vmin=vmin_chi2, vmax=vmax_chi2))
		
		ax1.set_title(f'Chi² - Horn {horn}', fontsize=14, fontweight='bold')
		ax1.set_xlabel('Frequency Bins', fontsize=12)
		ax1.set_ylabel('Time (Hours)', fontsize=12)
		ax1.set_yticks(np.arange(0, self.num_hours + 1, 4))
		ax1.set_xticks(np.arange(0, self.num_bins + 1, 5))
		
		cbar1 = plt.colorbar(im1, ax=ax1)
		cbar1.set_label('Chi² Value', fontsize=12)

		if self.calculate_rmse:
		
			cmap_rmse = copy.copy(plt.cm.plasma)
			cmap_rmse.set_bad(color='black')
			
			# Plot RMSE waterfall
			im2 = ax2.imshow(rmse_array, aspect='auto', cmap=cmap_rmse, 
							 extent=[0, self.num_bins, 0, self.num_hours], origin='lower')
			
			ax2.set_title(f'RMSE - Horn {horn}', fontsize=14, fontweight='bold')
			ax2.set_xlabel('Frequency Bins', fontsize=12)
			ax2.set_ylabel('Time (Hours)', fontsize=12)
			ax2.set_yticks(np.arange(0, self.num_hours + 1, 4))
			ax2.set_xticks(np.arange(0, self.num_bins + 1, 5))
			
			cbar2 = plt.colorbar(im2, ax=ax2)
			cbar2.set_label('RMSE Value', fontsize=12)
			
		plt.tight_layout()

		output_dir = self.base_waterfall_path
		os.makedirs(output_dir, exist_ok=True)
		
		output_path = os.path.join(output_dir, f'waterfall_horn{horn}.png')
		plt.savefig(output_path, dpi=300, bbox_inches='tight')
		plt.close()

	def save_to_hdf5(self):
		
		"""
		Consolidate all memmap data into a single compressed HDF5 file.
		Only rank 0 performs this step after all processes have finished writing.
		Creates groups '/chi2' and '/rmse' (if rmse=True). Each dataset is named
		'horn_XXX' and compressed with gzip level 4. Metadata attributes document
		the handling of missing data (NaN).
		"""
		
		if self.world_rank != 0:
			return
		
		hdf5_path = Path(self.base_hdf5_path) / "results.h5"
		
		with h5py.File(hdf5_path, 'w') as h5f:
			chi2_group = h5f.create_group('chi2')
			dof_group  = h5f.create_group('dof')
			
			if self.calculate_rmse:
				rmse_group = h5f.create_group('rmse')
		
			for horn in range(self.num_horns):
				
				# Process and save Chi² data
				chi2_mmap_path = Path(self.base_memmap_path) / f"chi2_{horn}.dat"

				try:
					chi2_data = np.memmap(chi2_mmap_path, dtype='float64', mode='r+', 
										  shape=(self.num_hours, self.num_bins))
					chi2_array = np.array(chi2_data)
					
					# Create dataset with gzip compression (level 4)
					dset_chi2 = chi2_group.create_dataset(f'horn_{horn:03d}', 
													data=chi2_array,
													compression='gzip',
													compression_opts=4)
					
					del chi2_data
					
				except FileNotFoundError as e:
					print(f"Error loading Chi² memory-mapped file for HDF5 consolidation (Horn {horn}): {e}")

				# Process and save DoF data
				dof_mmap_path = Path(self.base_memmap_path) / f"dof_{horn}.dat"

				try:
					dof_data = np.memmap(dof_mmap_path, dtype='float64', mode='r+',
										 shape=(self.num_hours, self.num_bins))
					dof_array = np.array(dof_data)

					dset_dof = dof_group.create_dataset(f'horn_{horn:03d}',
													data=dof_array,
													compression='gzip',
													compression_opts=4)
					dset_dof.attrs['description'] = (
						'Degrees of freedom used in the reduced chi-square calculation '
						'for each (hour, frequency bin) pair.'
					)

					del dof_data

				except FileNotFoundError as e:
					print(f"Error loading DoF memory-mapped file for HDF5 consolidation (Horn {horn}): {e}")

				# Process and save RMSE data
				if self.calculate_rmse:
					rmse_mmap_path = Path(self.base_memmap_path) / f"rmse_{horn}.dat"
					
					if rmse_mmap_path.exists():
						try:
							rmse_data = np.memmap(rmse_mmap_path, dtype='float64', mode='r+',
												  shape=(self.num_hours, self.num_bins))
							rmse_array = np.array(rmse_data)
							
							# Create dataset with gzip compression (level 4)
							dset_rmse = rmse_group.create_dataset(f'horn_{horn:03d}',
															data=rmse_array,
															compression='gzip',
															compression_opts=4)
							
							# # Add metadata attributes for missing data documentation
							# dset_rmse.attrs['missing_data_flag'] = 'NaN'
							# dset_rmse.attrs['min_valid_samples_threshold'] = self.min_valid_samples
							
							del rmse_data
							
						except Exception as e:
							print(f"Error loading RMSE memory-mapped file for HDF5 consolidation (Horn {horn}): {e}")

	def cleanup_memmaps(self):

		"""
		Delete all temporary .dat files created in the memmaps directory.
		Called only by rank 0 after HDF5 consolidation.
		"""
		
		if self.world_rank == 0:
			# Searching for memmaps files
			dat_files = glob.glob(str(self.base_memmap_path / "*.dat"))
			# Remove memmaps files
			for dat_file in dat_files:
				os.remove(dat_file)    
			print(f"Memory-mapped files removed")

	def run(self):
		
		"""
		Main execution flow for all processes.
		1 - Coordinate tasks (assign horn/hour pairs).
		2 - Execute analysis (compute chi²/RMSE and save to memmaps).
		3 - Wait for all processes to finish (Barrier).
		"""
		
		self.tasks_coordinator()
		
		self.execute_analysis()

		self.comm.Barrier()

	def finish_analysis(self, horns_to_plot=None, plot_waterfalls=False):
		
		"""
		Finalize the analysis after all parallel work is done.
		This method should be called after 'run'. It performs post-processing
		tasks exclusively on rank 0:
			- (Optionally) generate waterfall plots.
			- Consolidate memmap data into HDF5.
			- Clean up temporary memmap files.
		A final barrier ensures that no worker exits before cleanup is complete.
	
		Parameters ==>
		----------
		horns_to_plot : list of int, optional
			Which horns to plot. If None, plot all.
		plot_waterfalls : bool
			Whether to generate waterfall plots.
		"""
		self.comm.Barrier()
 
		if self.world_rank == 0:
			print(f"--- Finalizing Analysis (PID: {os.getpid()}) ---")
 
			# Generate Plots (if requested)
			if plot_waterfalls:
				print("Generating waterfall plots...")
				self.generate_waterfalls(horns_to_plot)
 
			# Consolidate Data into HDF5
			print("Consolidating data to HDF5...")
			self.save_to_hdf5()
 
			# Cleanup Temporary Files
			print("Cleaning up memory maps...")
			self.cleanup_memmaps()
			
		# Final barrier to ensure no worker exits before Rank 0 finishes cleanup
		self.comm.Barrier()
		
#==================================================================================================
#==================================================================================================