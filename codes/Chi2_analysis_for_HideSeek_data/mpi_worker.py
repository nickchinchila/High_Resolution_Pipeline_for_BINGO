"""
MPI worker script for chi-square/RMSE analysis of HIDE & SEEK TOD data.

This script is designed to be launched by an MPI parallel environment (via mpirun/mpiexec or SLURM srun).
It reads a JSON configuration file (provided as command-line argument), instantiates the
Chi2_for_Hide_Seek_data class, and executes the parallel analysis.

Workflow ==>
    1 - Load parameters from the JSON config file.
    2 - Create an instance of Chi2_for_Hide_Seek_data with the given parameters.
    3 - Call the 'run()' method to perform the distributed computation.
    4 - Call 'finish_analysis()' to finalize (plot, consolidate HDF5, clean up temporary files).

This script is not intended to be run directly by users; instead, it is called by
'submission_coordinator.py' or a SLURM job script.

[09/03/2026] - Nicolli Soares
"""
# =================================================================================================
import sys
import json
import os
from chi2_rmse_analysis import Chi2_for_Hide_Seek_data
# =================================================================================================

def main():
    # Expect exactly one argument: path to the JSON configuration file
    if len(sys.argv) < 2:
        # Exit silently if no config file provided (this is a worker, not meant to be run standalone)
        sys.exit(1)
    
    config_path = sys.argv[1]
 
    # Load parameters from JSON
    with open(config_path, 'r') as f:
        params = json.load(f)
    
    # Extract optional plotting parameters; these are used only by rank 0 during finalization
    plot_waterfalls = params.get('plot_waterfalls', False)
    horns_to_plot = params.get('horns_to_plot', None)   # None means plot all horns
 
    # Instantiate the analysis class with parameters from config
    worker = Chi2_for_Hide_Seek_data(
        n_horns=params['n_horns'],
        n_hours=params['n_hours'],
        n_bins=params['n_bins'],
        obs_date=params['obs_date'],
        base_results_path=params['base_results_path'],
        base_obsTOD_path=params['base_obsTOD_path'],
        base_expTOD_path=params['base_expTOD_path'],
		err_data=params['err_data'],
        dof=params.get('dof'),                           # optional degrees of freedom
        analysis_identifier=params.get('analysis_identifier'),  # optional job ID
        show_process_info=params.get('show_process_info', False),
        rmse=params.get('rmse', False),
    )
    
    # Execute the main parallel analysis (distributes work across MPI ranks)
    worker.run()
 
    # Finalize: generate plots (if requested), consolidate results into HDF5, clean up memmaps
    worker.finish_analysis(
        horns_to_plot=horns_to_plot,
        plot_waterfalls=plot_waterfalls
    )
 
if __name__ == "__main__":
    main()