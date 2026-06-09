# paths_config_II.py

"""
About Directories
# OBS: All outputs will be in the "output_high_reso_test" directory and 
# will be organized by seed number, where a random seed corresponds 
# to the first round of the iteration loop and with each new round 
# of the loop its value is increased by one. So that with each use 
# of this script the outputs will be organized by different seed 
# number sequences.
# OBS: Starlet Wavelet Transform abbreviation -> SWT 
"""

import os
from pathlib import Path

class _Paths:
	"""
	Internal class that centralizes path definitions.
	All methods return strings.
	"""
	def __init__(self):
		# Common base path
		self.base = Path(r"/data/NSOARES/hide_seek/")

	@property
	def hide_dir(self) -> str:
		"""Directory where HIDE is installed."""
		return str(self.base / r"hide")

	@property
	def ini_path(self) -> str:
		"""HIDE configuration file directory."""
		return str(self.base / r"hide/hide.ini")

	@property
	def hide_inptpath(self) -> str:
		"""HIDE input maps directory."""
		return str(self.base / r"hide/hide/data/sky/")

	@property
	def hide_outpath(self) -> str:
		"""HIDE output directory."""
		return str(self.base / r"output_high_reso_test/seed/")

	@property
	def hitmap_outpath(self) -> str:
		"""HIDE & SEEK output maps directory."""
		return str(self.base / r"output_high_reso_test/seed/maps/")

	@property
	def swt_maps_path(self) -> str:
		"""SWT output maps."""
		return str(self.base / r"output_high_reso_test/seed/swt_maps/")

	@property
	def swt_base_path(self) -> str:
		"""Base path of TODs generated from SWT output."""
		return str(self.base / r"output_high_reso_test/{seed_number}/NSWT/{coeff_type}{scale}") + "/"

	@property
	def base_diff_path(self) -> str:
		"""Base path of the difference between original TODs and SWT TODs."""
		
		return str(self.base / r"output_high_reso_test/seed{seed_number}/tod_difference/diff_J{scale}") + "/"

	@property
	def data_model_base_path(self) -> str:
		"""Base path of the final data model obtaneid """
		
		return str(self.base / r"output_high_reso_test/seed{seed_number}/data_model") + "/"
		
	@property
	def base_bingo_name(self) -> str:
		"""Base name pattern for BINGO TOD files."""
		return "bingo_tod_horn_{horn}_{date}_{hour:02d}0000.h5"

	@property
	def last_base_diff_path(self) -> str:
		"""Base path of the last difference (in the last scale)."""
		return os.path.join(self.base_diff_path, "2020", "03", "01", self.base_bingo_name)

	@property
	def recons_base_path(self) -> str:
		"""Base path of the equivalent reconstruction to the difference TOD."""
		return str(self.base / r"output_high_reso_test/seed{seed_number}/NSWT/Equivalent_recons_J{scale}") + "/"

	@property
	def base_chi2_path(self) -> str:
		"""Base path for all chi-square test results for each hour of each horn at each decomposition scale."""
		
		return str(self.base / r"output_high_reso_test/seed{seed_number}/chi2/J{scale}") + "/"

	@property
	def naivemap_fbase(self) -> str:
		"""HIDE & SEEK output map name pattern."""
		return r"naivemap_signal+noise_SEED{seed0}_ch{ch}_nch30_1d.fits"

	@property
	def nm_base_swt_path(self) -> str:
		"""Name pattern for HIDE & SEEK output maps generated from SWT output."""
		return r"NSWT_{coeff_type}{scale}_SKY_128_980mhz1260mhz_30bins_full_L0.fits"


# Single instance of the internal class
_paths = _Paths()

# Export all variables
base_path = str(_paths.base)
hide_dir = _paths.hide_dir
ini_path = _paths.ini_path
hide_inptpath = _paths.hide_inptpath
hide_outpath = _paths.hide_outpath
hitmap_outpath = _paths.hitmap_outpath
swt_maps_path = _paths.swt_maps_path
swt_base_path = _paths.swt_base_path
base_diff_path = _paths.base_diff_path
data_model_base_path = _paths.data_model_base_path
base_bingo_name = _paths.base_bingo_name
last_base_diff_path = _paths.last_base_diff_path
recons_base_path = _paths.recons_base_path
base_chi2_path = _paths.base_chi2_path
naivemap_fbase = _paths.naivemap_fbase
nm_base_swt_path = _paths.nm_base_swt_path