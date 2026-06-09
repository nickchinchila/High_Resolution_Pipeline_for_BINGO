from numpy.lib import format as fmt
import numpy as np
import healpy as hp
import scipy.fftpack
import copy 
import h5py
import time
import os
import glob
import re
import psutil as ps
import multiprocessing as mp
import astropy.io.fits as fits

import pyMRS as pym
import gmca 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
" Modified by Nicolli Soares P. -- April, 2026" 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


## speed of light ##
c = 3.0e8  # m/s

#########################################################
############   to crop the sim (to speed-up)  ###########
#########################################################

def nu_ch_f(nu_ch_in,dnu_out):
	du_in = abs(nu_ch_in[-1]-nu_ch_in[-2])
	a1 = nu_ch_in[0] - du_in/2; a2 = nu_ch_in[-1] + du_in/2
	M = int((a2-a1)/dnu_out)
	if (dnu_out*M)!=(a2-a1):
		print('just dnu multiples!')
		sys.exit()
	nu_ch_out = np.linspace(a1+dnu_out/2,a2-dnu_out/2,M)	

	return nu_ch_out


def merging_maps(nu_ch_in,nu_ch_out,maps_in,dnu_out):
	
	deltanu_in = abs(nu_ch_in[-1]-nu_ch_in[-2])
	maps_out  = [0] * len(nu_ch_out)  

	deltanu_out = abs(nu_ch_out[-1]-nu_ch_out[-2])
	N = int(deltanu_out/deltanu_in)
	if (deltanu_in*N)!=deltanu_out:
		print('just dnu multiples!')
		sys.exit()		

	for i in range(len(nu_ch_out)):
		maps_out[i] = sum(maps_in[N*i:N*i+N]) / N
		
	return maps_out



#########################################################
##################   useful functions   #################
#########################################################

## from vector to matrix and viceversa
def alm2tab(alm,lmax):

	size = np.size(alm)
	tab  = np.zeros((lmax+1,lmax+1,2))

	for r in range(0,size):
		l,m = hp.sphtfunc.Alm.getlm(lmax,r)
		tab[l,m,0] = np.real(alm[r])
		tab[l,m,1] = np.imag(alm[r])

	return tab

def tab2alm(tab):

	lmax = np.int(np.shape(tab)[0])-1
	taille = np.int(lmax*(lmax+3)/2)+1
	alm = np.zeros((taille,),dtype=complex)

	for r in range(0,taille):
		l,m = hp.sphtfunc.Alm.getlm(lmax,r)
		alm[r] = np.complex(tab[l,m,0],tab[l,m,1])

	return alm


## getting the spherical harmonic coefficients
## from a map
def almtrans(map_in,lmax=None):

	if lmax==None:
		lmax = 3.*hp.get_nside(map_in)
		print("lmax = ",lmax)

	alm = hp.sphtfunc.map2alm(map_in,lmax=lmax)

	tab = alm2tab(alm,lmax)

	return tab


## convolution:
## multiplying the spherical harmonic coefficients
def alm_product(tab,beam_l):
	length=np.size(beam_l)
	lmax = np.shape(tab)[0]

	if lmax > length:
		print("Filter length is too small")

	for r in range(lmax):
		tab[r,:,:] = beam_l[r]*tab[r,:,:]

	return tab


## from a_lm back to map
def almrec(tab,nside):

	alm = tab2alm(tab)
	map_out = hp.alm2map(alm,nside,verbose=False)

	return map_out


def plot_cl(fmap,verbose=False):
	LMAX = 3*hp.get_nside(fmap)
	cl = hp.anafast(fmap, lmax=LMAX)
	ell = np.arange(len(cl))
	y = ell * (ell + 1) * cl/2.0/np.pi

	if verbose:
		print("l (l+1) C_l /(2pi) [mK^2] vs l")

	return ell, y

#########################################################
###################   gaussian beam   ###################
#########################################################

## angle in radians of the FWHM
def theta_FWHM(nu,dish_diam): # nu in MHz, dish_diam in m
	return c*1e-6/nu/float(dish_diam) # rad

## solid angle of beam in steradian 
def Omega_beam(nu,dish_diam): # nu in MHz, dish_diam in m 
	return np.pi/(4.*np.log(2))*theta_FWHM(nu,dish_diam)**2

## how many beams to cover my survey area (fraction of sky)
def N_beams(f_sky,nu,dish_diam): # nu in MHz, dish_diam in m 
	return 4*np.pi*f_sky/Omega_beam(nu,dish_diam)

## Fourier transform of the gaussian beam
def getBeam(theta_FWHM,lmax): # theta_FWHM in radians
	sigma_b = theta_FWHM/np.sqrt(8.*np.log(2.))

	l = np.linspace(0,lmax,lmax+1)
	ell = l*(l+1)

	return np.exp(-ell*sigma_b*sigma_b/2)

## convolving the map with the beam
## outputs the new map
def convolve(map_in,beam_l,lmax):

	alm = almtrans(map_in,lmax=lmax)
	tab = alm_product(alm,beam_l)
	m = almrec(tab,nside=hp.get_nside(map_in))

	return m


#########################################################
###################   thermal noise   ###################
#########################################################

def T_sky(nu): # K
	return 60.*(300./nu)**2.55  # K

def T_rcvr(nu,T_inst): # K
	temp_sky = T_sky(nu)
	return 0.1* temp_sky + T_inst

def T_sys(nu,T_inst): # K
	return T_rcvr(nu,T_inst) + T_sky(nu)

## final sigma in mK 
def sigma_N(nu,dnu,T_inst,f_sky,t_obs,Ndishes,dish_diam):
	t_obs = t_obs * 3600 # hrs to s
	dnu = dnu * 1.e6 # MHz to Hz

	temp_sys = T_sys(nu,T_inst)  # in K
	A = np.sqrt(N_beams(f_sky,nu,dish_diam)/dnu/t_obs/Ndishes)
	
	return temp_sys * A *1e3  # mK

def noise_map(sigma,nside=512):
	npixels = hp.nside2npix(nside)
	m = np.random.normal(0.0, sigma, npixels)
	return m

#########################################################
###################    GMCA running   ###################
#########################################################

# J is the number of WT scale
# spherical or 2D patch?

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
" Modifications by Nicolli Soares P. -- April, 2026 => " 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def wt_one_map(args):

	start_time = time.time()
	data, map_index, J, fits_dir, npix, n_maps = args
	LMAX = 3 * hp.npix2nside(npix)
	nscale = J + 1  
	
	WT, CT = pym.wttrans(data, nscale=nscale, lmax=LMAX)
	X_wt = WT[:, :J].T
	cX_wt = CT.T
	
	for scale in range(J):
		# Escrever coeficiente Cj
		cj_file = os.path.join(fits_dir, f"{n_maps}maps_C{scale}.dat")
		with open(cj_file, 'r+b') as f: 
			cj_mmap = np.memmap(f, dtype='float64', mode='r+', shape=(n_maps, npix))
			cj_mmap[map_index] = cX_wt[scale]
			cj_mmap.flush()
		
		# Escrever coeficiente Wj
		wj_file = os.path.join(fits_dir, f"{n_maps}maps_W{scale}.dat")
		with open(wj_file, 'r+b') as f:
			wj_mmap = np.memmap(f, dtype='float64', mode='r+', shape=(n_maps, npix))
			wj_mmap[map_index] = X_wt[scale]
			wj_mmap.flush()
	
	print(f"Finished one map in {((time.time() - start_time)/60):.2f} minutes")
	
	return map_index  
	
#----------------------------------------------------------------------------------------------------------

def parallel_wavelet_transform(input_map, J, nside, fits_dir, freq_vec):
	
	# Carrega mapa de input
	with fits.open(input_map) as h:
		hdul = copy.deepcopy(h)
		freqs = hdul[0].data
		maps = hdul[1].data
	
	X = copy.deepcopy(maps)
	npix = hp.nside2npix(nside)
	n_maps = len(X)
	# print("n_maps",n_maps)

	# Cria arquivos temporários vazios
	for scale in range(J):
		for prefix in ['C', 'W']:
			filename = os.path.join(fits_dir, f"{n_maps}maps_{prefix}{scale}.dat")
			if not os.path.exists(filename):
				arr = np.memmap(filename, dtype='float64', mode='w+', shape=(n_maps, npix))
				del arr  
	#***************************************************************************
	# Realiza wavelet transform para todos os mapas em paralelo
	start_time = time.time()
	# Define número de cores a serem utilizados
	total_cores = mp.cpu_count()
	used_cores = np.sum(np.array(ps.cpu_percent(percpu=True, interval=0.1)) > 10)  
	idle_cores = total_cores - used_cores
	
	with mp.Pool(processes=total_cores) as pool:
		args = [(X[i], i, J, fits_dir, npix, n_maps) for i in range(n_maps)]
		results = pool.map(wt_one_map, args)
	
		
	#***************************************************************************
	# Criando arquivos ".fits" em um formato compatível com o HIDE & SEEK 
	# Cada arquivo ".fits" corresponde a uma escala de decomposição (coef. Wj ou Cj) da SWT em 30 faixas de frequência.
	# PrimaryHDU com shape (J_npix,nbins)
	# ImageHDU com shape (31,)
	
	for scale in range(J):
		# Processar arquivo Cj
		cj_file = os.path.join(fits_dir, f"{n_maps}maps_C{scale}.dat")
		with open(cj_file, 'rb') as f:
			cj_data = np.memmap(f, dtype='float64', mode='r', shape=(n_maps, npix))
			fits_file = os.path.join(fits_dir, f"NSWT_C{scale}_SKY_{nside}_980mhz1260mhz_30bins_full_L0.fits")
			
			hdu_primary = fits.PrimaryHDU(cj_data)
			hdu_freq = fits.ImageHDU(freq_vec)
			hdul = fits.HDUList([hdu_primary, hdu_freq])
			hdul.writeto(fits_file, overwrite=True)
			os.remove(cj_file)  
		
		# Processar arquivo Wj
		wj_file = os.path.join(fits_dir, f"{n_maps}maps_W{scale}.dat")
		with open(wj_file, 'rb') as f:
			wj_data = np.memmap(f, dtype='float64', mode='r', shape=(n_maps, npix))
			fits_file = os.path.join(fits_dir, f"NSWT_W{scale}_SKY_{nside}_980mhz1260mhz_30bins_full_L0.fits")
			
			hdu_primary = fits.PrimaryHDU(wj_data)
			hdu_freq = fits.ImageHDU(freq_vec)
			hdul = fits.HDUList([hdu_primary, hdu_freq])
			hdul.writeto(fits_file, overwrite=True)
			os.remove(wj_file)  
	
	print(f"Decomposition successfully finish in {((time.time()-start_time)/60):.2f} minutes")

#-------------------------------------------------------------------------------------------------------
def make_mmap_fname(fits_dir, family, J, nbins):
	fname = f"SWT_{family}_maps_J{J}_{nbins}.mmap"
	return os.path.join(fits_dir, fname)
#-------------------------------------------------------------------------------------------------------
def extract_swt_ffits_memmap(fits_dir, family, nbins):
	
	pattern = rf"NSWT_{family}(\d+)_.*\.fits$"
	files = glob.glob(os.path.join(fits_dir, f"NSWT_{family}*_SKY_*.fits"))
	files = sorted(
		files,
		key=lambda fn: int(re.search(pattern, os.path.basename(fn)).group(1))
	)
	J = len(files)

	with fits.open(files[0], memmap=True) as hdul0:
		arr0 = hdul0[0].data         # shape (nbins, npix)
		# print("array.shape", arr0.shape)
		freq = hdul0[1].data         # shape (nbins+1,)
		# print("freq shape",freq.shape)
	nbins0, npix = arr0.shape
	# print("nbins0",nbins0)

	mmap_fname = make_mmap_fname(fits_dir, family, J, nbins0) 

	maps = np.memmap(
		filename=mmap_fname,
		dtype=arr0.dtype,
		mode='w+',
		shape=(nbins0, J, npix)
	)

	for j, fn in enumerate(files):
		with fits.open(fn, memmap=True) as hdul:
			data = hdul[0].data   # shape (nbins, npix)
		maps[:, j, :] = data      

	maps.flush()
	
	maps = np.memmap(
		filename=mmap_fname,
		dtype=arr0.dtype,
		mode='r',
		shape=(nbins0, J, npix)
	)

	return maps, freq
#-------------------------------------------------------------------------------------------------------------------------
def wavelet_transform(X, J):
	n_maps = len(X)
	npix = len(X[0])
	nscale = J + 1  
	
	LMAX = 3 * hp.npix2nside(npix)
	
	print("Starting Starlet transform ...")
	start_w = time.time()
	
	for r in range(n_maps):

		start_test = time.time()
		
		WT, CT = pym.wttrans(X[r], nscale=nscale, lmax=LMAX)
		
		X_wt = WT[:, 0:J].T
		
		cX_wt = CT.T

		end_test = time.time()
		t_test = end_test - start_test
		print(f"Finished map {r+1}/{n_maps} in {t_test/60:.2f} min")
		
		# print("shape x", X_wt.shape)
		# print("shape c", cX_wt.shape)
		
		yield X_wt, cX_wt
		
	end_w = time.time()
	tw = end_w - start_w
	print("Finished wavelet transform(s) in: {:.2f} min".format(tw/60))
#-----------------------------------------------------------------------------------------------------------------------

def process_map_HS_format(X_wt, cX_wt, map_index, freq_vector, npix, fits_dir, J):

	for scale in range(J): 
		
		# # Criando arquivos temporários para os coef. Cj para uma faixa de freq (map_index).
		Cj_name = os.path.join(fits_dir,f"30maps_C{scale}.dat") 
		mode_mmap = 'r+' if os.path.exists(Cj_name) else 'w+'
		data_cj = np.memmap(Cj_name, dtype = "float64", mode = mode_mmap, shape = (30,npix))
		data_cj[map_index] = copy.deepcopy(cX_wt[scale,:])
		data_cj.flush()
		del data_cj
		
		# Criando arquivos temporários para os coef. Wj para uma faixa de freq (map_index).
		Wj_name = os.path.join(fits_dir,f"30maps_W{scale}.dat") 
		md_mmap = 'r+' if os.path.exists(Wj_name) else 'w+'
		data_wj = np.memmap(Wj_name, dtype = "float64", mode = md_mmap, shape = (30,npix))
		data_wj[map_index] = copy.deepcopy(X_wt[scale,:])
		data_wj.flush()
		del data_wj

	# Criando arquivos ".fits" em um formato compatível com o HIDE & SEEK 
	# Cada arquivo ".fits" corresponde a uma escala de decomposição (coef. Wj ou Cj) da SWT em 30 faixas de frequência.
	# PrimaryHDU com shape (J_npix,nbins)
	# ImageHDU com shape (31,)
	
	if map_index == 29:

		for scale in range(J):
	
			Cj_name = os.path.join(fits_dir,f"30maps_C{scale}.dat")
			final_temp_Cj = np.memmap(Cj_name, dtype = "float64", mode = "r", shape = (30,npix))
	
			filenameC_fits = os.path.join(fits_dir,f"NSWT_C{scale}_SKY_128_980mhz1260mhz_30bins_full_L0.fits")
			dataaa = (final_temp_Cj)#.T
			primary_hdu = fits.PrimaryHDU(data=dataaa) 
			image_hdu = fits.ImageHDU(data=freq_vector)
			hdul = fits.HDUList([primary_hdu, image_hdu])
			hdul.writeto(filenameC_fits, overwrite=True)	
			os.remove(Cj_name)
			
			#----------------------------------------------------------------------------------------#
			
			Wj_name = os.path.join(fits_dir,f"30maps_W{scale}.dat")
			final_temp_Wj = np.memmap(Wj_name, dtype = "float64", mode = "r", shape = (30,npix))
	
			filenameW_fits = os.path.join(fits_dir,f"NSWT_W{scale}_SKY_128_980mhz1260mhz_30bins_full_L0.fits")
			DATAAA = (final_temp_Wj)#.T
			primary1_hdu = fits.PrimaryHDU(data=DATAAA) 
			image1_hdu = fits.ImageHDU(data=freq_vector)
			hdul = fits.HDUList([primary1_hdu, image1_hdu])
			hdul.writeto(filenameW_fits, overwrite=True)	
			os.remove(Wj_name)

		print("Files written successfully")
#-------------------------------------------------------------------------------------------------------------------
def process_map_fits(X_wt, cX_wt, map_index, fits_filename):
	
	hdu_wt = fits.ImageHDU(data=X_wt, name=f'WT_{map_index:03d}')
	hdu_ct = fits.ImageHDU(data=cX_wt, name=f'CT_{map_index:03d}')
	
	if os.path.exists(fits_filename):
		with fits.open(fits_filename, mode='update') as hdul:
			hdul.append(hdu_wt)
			hdul.append(hdu_ct)
			hdul.flush() 
	else:
		primary_hdu = fits.PrimaryHDU()
		hdul = fits.HDUList([primary_hdu, hdu_wt, hdu_ct])
		hdul.writeto(fits_filename)	

def extract_starlet_data_fits(fits_filename):
	with fits.open(fits_filename) as hdul:
		wt_data = {}
		ct_data = {}
		
		for hdu in hdul[1:]:  
			name = hdu.name
			data = hdu.data
			
			if name.startswith("WT_"):
				map_idx = int(name.split("_")[1])
				wt_data[map_idx] = data
			
			elif name.startswith("CT_"):
				map_idx = int(name.split("_")[1])
				ct_data[map_idx] = data
		
		wt_data_ordered = [wt_data[i] for i in sorted(wt_data.keys())]
		ct_data_ordered = [ct_data[i] for i in sorted(ct_data.keys())]
		
		X_wt = np.array(wt_data_ordered)
		cX_wt = np.array(ct_data_ordered)
		
		return X_wt, cX_wt
#-------------------------------------------------------------------------------------------------------------------        
def process_map(X_wt, cX_wt, map_index, h5_filename):
	with h5py.File(h5_filename, "a") as h5f:
		h5f.create_dataset(
			f"WT_{map_index:03d}", 
			data=X_wt, 
			compression="gzip", 
			compression_opts=9  
		)
		h5f.create_dataset(
			f"CT_{map_index:03d}", 
			data=cX_wt, 
			compression="gzip",  
			compression_opts=9  
		)

def extract_starlet_data(h5_filename):
	wt_data = []
	ct_data = []
	
	with h5py.File(h5_filename, "r") as h5f:
		map_index = sorted([int(key.split("_")[1]) for key in h5f.keys() if key.startswith("WT_")])
		
		for idx in map_index:
			wt_data.append(h5f[f"WT_{idx:03d}"][:])  
			ct_data.append(h5f[f"CT_{idx:03d}"][:])  

	X_wt = np.array(wt_data)  # Shape: (n_maps, J, npix)
	cX_wt = np.array(ct_data)  # Shape: (n_maps, J+1, npix)

	return X_wt, cX_wt
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
" end of modifications by nicolli s. p."
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#-------------------------------------------------------------------------------------------------------


# original function 
# def wavelet_transform(X,J=3): #X is a cube with [X]=nch x npix. X[:,i] = is a spectrum fo the i-pixel on the sphere. X[i,:] is a i-map on the sphere.

# 	#print('\nWavelet transforming the data . . .')
# 	print('Starting Starlet transform ...')
# 	start_w = time.time()

# 	X_wt  = np.zeros((len(X),len(X[0])*J))
# 	cX_wt = []
# 	LMAX  = 3*hp.npix2nside(len(X[0]))

# 	for r in range(X.shape[0]):                       # r - map
# 		temp = pym.wttrans(X[r],nscale=J+1,lmax=LMAX) # r - Wavelet Coeficients   #[tamp] = [col0=w0,col1=w1,..,colJ-1=wJ-1, colJ=c0]
# 		X_wt[r,:] = temp[:,0:J].reshape(1,-1)         # wavelet coefs per bin
# 		cX_wt.append(temp[:,J])  #coarse scale        # scale coefs per bin

# 	end_w = time.time()
# 	tw = end_w - start_w
# 	print("Finished wavelet transform(s) in: {:.2f} min".format(tw/60))		
# 	del tw,end_w,start_w

# 	return X_wt, np.asarray(cX_wt)	
	

def inverse_wavelet_transform():
	return None


def run_GMCA(X_wt,AInit,n_s,mints,nmax,L0,ColFixed,whitening,epsi, verbose=True):
	import numpy as np
	# First guess mixing matrix (could be set to None or not provided at all)
	if AInit is None:
		AInit = np.random.rand(len(X_wt),n_s)

	if verbose: 
		print('\nNow running GMCA . . .')

	if whitening: 

		R = np.dot(X_wt,X_wt.T) # R correlation matrix of the X_wt
		L,U = np.linalg.eig(R)  # L = diagonal matrix of the eigenvalues, U = eigenvectores matrix
		## whitening the data
		
		Q = np.dot(np.diag(1./(L+epsi*np.max(L))),U.T) # np.diag(1./(L+epsi*np.max(L))) == diagonal matrix with the same eigenvalues # if x is the data, Qx is the whitening data, where Q is L^(-1/2)*U^{adaga}
		iQ = np.dot(U,np.diag((L+epsi*np.max(L))))     # iQ = U*L^{1/2}

		if ColFixed is None:
			CL = None
		else: CL = Q*ColFixed      # select just one column from mixing matrix

		start_w = time.time()
		Results = gmca.GMCA(np.dot(Q,X_wt),n=n_s,mints=mints,nmax=nmax,L0=L0,Init=0,AInit=AInit,ColFixed=CL)
		end_w = time.time()

		Ae = iQ*Results["mixmat"]  # estimated mixing matrix    #The iQ matrix return to original space, without whitening.
		S  = iQ*Results["sources"]

	else:
		start_w = time.time()
		Results = gmca.GMCA(X_wt,n=n_s,mints=mints,nmax=nmax,L0=L0,Init=0,AInit=AInit,ColFixed=ColFixed)
		end_w = time.time()

		Ae = Results["mixmat"]
		S  = Results["sources"]

	tw = end_w - start_w
	if verbose: 
		print('. . completed in %.2f minutes\n'%(tw/60))

	return Ae, S




#########################################################
#################    radial clustering   ################
#########################################################

## HOW to use these functions:
# # # find the lines of sight over which compute the radial P(k)
# # indexes_los = np.where(mask==1.0)[0]

# ## which lines of sight
# # indexes_los = np.arange(0,hp.nside2npix(NSIDE),10)
# indexes_los = np.arange(hp.nside2npix(NSIDE))

## field_array should be nu X pixels 
## equally spaced array

def clustering_nu(field_array,indexes_los,nu_ch,verbose=False):
	
	## sanity check
	if verbose:
		print('sanity check: ')
		print('  ',(len(field_array[:,0])==len(nu_ch)),' ',(len(field_array[0,:])>=len(indexes_los)))
		print('  ',(len(nu_ch) % 2) == 0)


	## cropping the array
	T_field = field_array[:,indexes_los]
	del field_array

	## how many LOS are we considering?
	nlos = len(indexes_los)
	if verbose: print("using {} LoS".format(nlos))
	del indexes_los

	## defines cells 
	dims = len(nu_ch); dnu  = abs(nu_ch[-1]-nu_ch[-2])
	if verbose: print('each divided into {} cells of {} MHz'.format(dims,dnu))

	## remove mean from maps
	if verbose: print('removing mean from maps . .')
	mean_T_mapwise = np.mean(T_field,axis=1)
	T_field_nm =  np.array([T_field[i,:] - mean_T_mapwise[i] for i in range(dims)])
	del T_field
	if verbose: print('defining DeltaT array . .')
	deltaT = np.array([T_field_nm[:,ipix]  for ipix in range(nlos)])
	# print('i.e. deltaT --> ',deltaT.shape)
	del T_field_nm

	if verbose: print('\nFFT the overdensity temperature field along LoS')
	delta_k = scipy.fftpack.fftn(deltaT,overwrite_x=True,axes=1)
	delta_k *= dnu;  del deltaT

	delta_k_auto  = np.absolute(delta_k)**2  

	if verbose: print('done!\n')
	return dims, dnu, delta_k_auto

def doing_Pk1D(dims,dnu,delta_k_auto):

	# compute the values of k of the modes for the 1D P(k)
	modes   = np.arange(dims,dtype=np.float64);  middle = int(dims/2)
	indexes = np.where(modes>middle)[0];  modes[indexes] = modes[indexes]-dims
	k = modes*(2.0*np.pi/(dnu*dims)) # k in MHz-1
	k = np.absolute(k)               # just take the modulus
	del indexes, modes

	# define the k-bins
	k_bins = np.linspace(0,middle,middle+1)*(2.0*np.pi/(dnu*dims))

	# compute the number of modes and the average number-weighted value of k
	k_modes = np.histogram(k,bins=k_bins)[0]
	k_bin   = np.histogram(k,bins=k_bins,weights=k)[0]/k_modes

	# take all LoS and compute the average value for each mode
	delta_k2_stacked = np.mean(delta_k_auto,dtype=np.float64,axis=0)

	# compute the 1D P(k)
	Pk_mean = np.histogram(k,bins=k_bins,weights=delta_k2_stacked)[0]
	Pk_mean = Pk_mean/(dnu*dims*k_modes);  del delta_k2_stacked

	Pk_1D = np.transpose([k_bin[1:],Pk_mean[1:]])
	
	return Pk_1D


## to plot the frequency power spectrum
## returns knu and P for x and y axis
def plot_nuPk(fmap,indexes_los,nu_ch,verbose=False):

	Pk_1D = doing_Pk1D(*clustering_nu(fmap,indexes_los,nu_ch))

	if verbose:
		print("k_nu [MHz^-1] vs P [mK^2 MHz]")

	return Pk_1D[:,0],Pk_1D[:,1]

