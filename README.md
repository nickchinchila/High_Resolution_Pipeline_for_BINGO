# High_Resolution_Pipeline_for_BINGO

Considering the limited (40 arcminute) angular resolution of the BINGO radiotelescope, this project aims to develop a system capable of recovering information from smaller angular-scale structures using an iterative algorithm. 

This algorithm utilizes the following tools (which had to be adapted and/or developed): 

* **HIDE & SEEK**
  * Adapted for integration times other than 1s 
  * Adapted to work in parallel using Python multiprocessing methods 
  * Repository: https://github.com/zxcorr/hide/tree/HIDE_parallel

* **Python Library - Chi2 analysis_for HideSeek data** 
  * Two-level parallelization using MPI
  * Adapted to run on clusters and standard servers 
  * Repository: https://github.com/nickchinchila/Chi2_analysis_for_HideSeek_data

* **GMCA4im scripts** 
  * Adapted for SWT decomposition at various scales 
  * Parallelized SWT using Python multiprocessing
  * Adapted for compatibility of output files with H&S
  * Repository: https://github.com/isab3lla/gmca4im 

---

## 3. Library Chi² analysis for HIDE & SEEK data structure 

![Library Chi² analysis for HIDE & SEEK data structure](chi2_library_flowchart.png)

## 4. High Resolution Model Construction Algorithm 

![High Resolution Model Construction Algorithm](high_reso_flowchart.png)
