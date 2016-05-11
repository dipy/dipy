import numpy as np


def localPCA_denoise(arr, sigma, patch_radius=2, tou=0, rician=True):
	'''
	Local PCA Based Denoising of Diffusion Datasets

	References
	----------
	Diffusion Weighted Image Denoising Using Overcomplete Local PCA
	Manjón JV, Coupé P, Concha L, Buades A, Collins DL

	Parameters
	----------
	arr : A 4D array which is to be denoised
	sigma : float or 3D array
        standard deviation of the noise estimated from the data
	patch_radius : The radius of the local patch to
				be taken around each voxel
	tou : float or 3D array
		threshold parameter for diagonal matrix thresholding,
		default value = (2.3 * sigma * sigma)
	rician : boolean
		If True the noise is estimated as Rician, otherwise Gaussian noise
        is assumed.

	Returns
	-------

	denoised_arr : 4D array
		this is the denoised array of the same size as that of
		arr (input data) 

	'''

	# Read the array and fix the dimensions

	if arr.ndim == 4:

		if tou == 0:
			tou = 2.3 * sigma * sigma

		if isinstance(sigma, np.ndarray) and sigma.ndim == 3:
            sigma = (np.ones(arr.shape, dtype=np.float64) *
                     sigma[..., np.newaxis])
            tou = (np.ones(arr.shape, dtype=np.float64) * 
            		 sigma[..., np.newaxis])
        else:
            sigma = np.ones(arr.shape, dtype=np.float64) * sigma
            tou = np.ones(arr.shape, dtype=np.float64) * tou	
		
		# loop around and find the 3D patch for each direction at each pixel
		
		# declare a dimensions array
		dims = []
		dims[0] = arr.shape[0]
		dims[1] = arr.shape[1]
		dims[2] = arr.shape[2]
		dims[3] = arr.shape[3]

		for k in range(0, dims[2], 2):
        	for i in range(0, dims[1], 2):
            	for j in range(0, dims[0], 2):
            		
            		X = np.zeros(patch_radius * patch_radius * patch_radius, dims[3])
            		
            		for l in range(0,dims[3],1):
            			
						# create the matrix X and normalize it
						temp = arr[...,l]
            			temp = temp.reshape(patch_radius * patch_radius * patch_radius)
            			X[:,l] = temp
            			# compute the mean and normalize
            			X[:,l] = X[:,l] - np.mean(X[:,l])

					# Compute the covariance matrix C = X_transpose X
					C = np.transpose(X).dot(X)
					# compute EVD of the covariance matrix of X get the matrices W and D, hence get matrix Y = XW
					# Threshold matrix D and then compute X_est = YW_transpose D_est
					[d,W] = np.linalg.eigh(C)
					d[d < tou[i][j][k][0]] = 0
					D_hat = np.diag(d)
					Y = X.dot(W)
					# When the block covers each pixel identify it into the label matrix theta
					X_est = Y.dot(np.transpose(W))
					X_est = X_est.dot(D_hat)

					# generate a theta matrix for patch around i,j,k and store it's theta value			
					# Also update the estimate matrix which is X_est * theta
					
					# After estimation pass it through a function ~ rician adaptation

	else:
		raise ValueError("Only 4D array are supported!", arr.shape)
	



