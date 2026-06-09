"""
Chi2 Analysis for Hide & Seek Data Library
==========================================

This library provides a parallel framework for performing chi-square
and RMSE analysis between observed and expected Time-Ordered Data (TOD) from the 
HIDE & SEEK data. It is designed to handle large datasets efficiently using MPI 
for parallelism, numpy memmaps for temporary storage, and 
produces waterfall plots (2D colour maps) as visual output. Results are finally 
consolidated into a single compressed HDF5 file.

 
[10/03/2026] - Nicolli Soares
"""
 
# =================================================================================================

from .chi2_rmse_analysis import Chi2_for_Hide_Seek_data
from .submission_coordinator import run_parallel
 
__all__ = [
    'Chi2_for_Hide_Seek_data',
    'run_parallel',
]
 
__version__ = '1.1'
__author__ = 'Nicolli Soares'