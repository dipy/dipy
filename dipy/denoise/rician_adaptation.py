import numpy as np

def rician_adaptation(arr, sigma):

	'''
	Corrects the estimate of the Local PCA algorithm to
	adapt to the rician noise framework

	Reference
	---------
	Diffusion Weighted Image Denoising Using Overcomplete Local PCA
	Manjón JV, Coupé P, Concha L, Buades A, Collins DL

	Parameters
	----------
	arr: 3D array
		this arr is the estimate of the local PCA algorithm
	sigma : float or 3D array
		the local noise variance estimate
	Returns
	-------

	corrected_arr: 3D array 
				same size as arr with the rician correction
	'''

	# load a lookup table
	# run the loop
	# for each x/sigma compute eta(x/sigma) from the lut
	# x_est = sigma x eta(x/sigma)