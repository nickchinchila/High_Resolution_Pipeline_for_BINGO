"""
Coordinator script for submitting chi-square/RMSE analysis jobs to a cluster or local MPI.

This script handles the creation of a JSON configuration file and the submission of the analysis
to either a SLURM-managed cluster (default) or a local machine using mpiexec.

It generates a unique job ID (or reuses one provided in params), writes a configuration file
'config_job_<job_id>.json', and then submits the job via 'sbatch' (SLURM) or 'mpiexec' (local).

The actual parallel work is performed by 'mpi_worker.py', which is invoked on all allocated
MPI ranks.
 
[09/03/2026] - Nicolli Soares
"""
# =================================================================================================
import subprocess
import json
import os
import random
import sys
# =================================================================================================
def run_parallel(params, num_nodes=2, total_num_process=64, slurm=True):
    """
    Submit a parallel chi-square analysis job.

    Parameters
    ----------
    params : dict
        Dictionary containing all parameters for the analysis (see Chi2_for_Hide_Seek_data constructor).
        Additionally, may include 'analysis_identifier' (int), 'plot_waterfalls' (bool),
        and 'horns_to_plot' (list or None).
    num_nodes : int
        Number of nodes to request when using SLURM. Ignored if slurm=False.
    total_num_process : int
        Total number of MPI processes (ranks) to launch. For SLURM, this equals --ntasks.
    slurm : bool
        If True, submit via SLURM using sbatch; otherwise run locally with mpiexec.

    Returns
    -------
    str
        Path to the generated JSON configuration file (useful for tracking/logging).
    """
    
    # Determine the directory where this script resides (to locate mpi_worker.py)
    package_dir = os.path.dirname(os.path.abspath(__file__))
    worker_script = str(os.path.join(package_dir, 'mpi_worker.py'))

    # Verify that the worker script exists
    if not os.path.exists(worker_script):
        raise FileNotFoundError(f"Error: Could not find worker script at {worker_script}")
    
    # Generate a job ID if not provided in params
    job_id = params.get('analysis_identifier', random.randint(1000, 9999))
    config_filename = os.path.abspath(f"config_job_{job_id}.json")
 
    # Write the configuration file
    with open(config_filename, 'w') as f:
        json.dump(params, f, indent=4)
 
    print(f"--- Starting Job ID {job_id} ---")
    print(f"Parameters file saved in: {config_filename}")
 
    if slurm:
        # Create a temporary SLURM submission script
        slurm_script_name = f"submit_job_{job_id}.sh"
        
        with open(slurm_script_name, 'w') as f:
            f.write("#!/bin/bash\n")
            # f.write(f"#SBATCH --job-name=chi2_analysis_{job_id}\n")
            # f.write(f"#SBATCH --nodes={num_nodes}\n")
            # f.write(f"#SBATCH --ntasks={total_num_process}\n")
            # f.write(f"#SBATCH --output=slurm-%j.out\n")
            # f.write(f"#SBATCH --error=slurm-%j.err\n")
            f.write("\n")
            
            # (Optional) Add module loads and virtual environment activation here if needed
            # Uncomment and modify as appropriate for your cluster environment
            # f.write("# module load openmpi\n")
            # f.write("# source /path/to/venv/bin/activate\n")
            # f.write("\n")
            
            # Launch the parallel job using srun
            f.write(f"srun --mpi=pmix -n 4 --cpus-per-task=48 python {worker_script} {config_filename}\n")

        cmd = ['bash', slurm_script_name]
        print(f"Generated SLURM script: {slurm_script_name}")
        
    else:
        # Local execution using mpiexec (useful for testing on a single machine)
        print("Configuring local MPI execution...")
        cmd = [
            'mpiexec',
            '-n', str(total_num_process),
            'python', worker_script,
            config_filename
        ]

    print(f"Executing command: {' '.join(cmd)}")
    
    # Execute the submission command
    try:
        # Use subprocess.run with check=True to raise an error if the command fails.
        # For local mpiexec, we let the output stream to the terminal in real time.
        # For sbatch, capture output to show the assigned job ID.
        result = subprocess.run(cmd, check=True, text=True, capture_output=slurm)
        print("\n--- Analysis job successfully submitted ---")
        if slurm:
            # SLURM sbatch prints the job ID to stdout; show it
            print("SLURM output:", result.stdout)
        else:
            # For local runs, output is already printed; just confirm completion
            print("Local MPI job completed.")
        
    except subprocess.CalledProcessError as e:
        print("Error during job submission:")
        print(e.stderr)
 
    return config_filename