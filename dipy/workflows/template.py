import os
import wget
import random
import numpy as np
from dipy.io.image import load_nifti, save_nifti, save_qa_metric
from dipy.align.imaffine import AffineMap, MutualInformationMetric, AffineRegistration
from dipy.align.imwarp import DiffeomorphicMap, DiffeomorphicRegistration, SymmetricDiffeomorphicRegistration
from dipy.align.transforms import AffineTransform3D
from dipy.align.metrics import CCMetric
from dipy.workflows.workflow import Workflow

class BuildTemplateFlow(Workflow):

	@classmethod
	def get_short_name(cls):
		return 'template'

	def create_list_of_files(path_to_directory):
	
		# return a list of FA files
		files_list = []

		for (dir_path, dir_names, files) in os.walk(path_to_directory):
			files_list += [os.path.join(dir_path, file) for file in files if file.endswith('fa.nii.gz')]

		return files_list

	def prepare_data(list_of_files):

		# creates a dict: structure is 'file_name': {'im' : im, 'affine' : affine}
		dict_fa = {}

		for file in list_of_files:
			im, affine = load_nifti(file, return_img = False)
			dict_fa[file] = {'im' : im, 'affine' : affine}

		return dict_fa

	def affine_registration_pair(first_im, first_affine, second_im, second_affine):

		static_data = first_im
		static_grid2world = first_affine    
		moving_data = second_im
		moving_grid2world = second_affine

		nbins = 32
		metric = MutualInformationMetric(nbins, None)
		level_iters = [10000, 1000, 100]
		sigmas = [3.0, 1.0, 0.0]
		factors = [4, 2, 1]

		affreg = AffineRegistration(metric = metric, level_iters = level_iters, sigmas = sigmas, actors = factors)

		transform = AffineTransform3D()

		# initial parameters
		params0 = None
		starting_affine = None

		# optimize
		affine, opti_params, cost = affreg.optimize(static_data, moving_data, transform, params0, static_grid2world, moving_grid2world, starting_affine, ret_metric=True)

		# Registration cost
		print('\nCOST: MutualInformation = ', cost)

		# transform moving
		moving_transformed_ = affine.transform(moving_data)

		# inverse_transform
		static_transformed_ = affine.transform_inverse(static_data)

		# temporary middle image
		middle = (np.add(moving_transformed_, static_transformed_))/np.array([2])

		return middle, static_grid2world, cost

	def diffeomorphic_registration_pair(first_im, first_affine, second_im, second_affine):
		
		static_data = first_im
		static_grid2world = first_affine    
		moving_data = second_im
		moving_grid2world = second_affine

		metric = CCMetric(3)

		level_iters = [10, 10, 5]
		sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)

		mapping = sdr.optimize(static_data, moving_data, static_grid2world, moving_grid2world, prealign = None)

		# transform moving
		moving_transformed_ = mapping.transform(moving_data)

		# inverse_transform
		static_transformed_ = mapping.transform_inverse(static_data)

		# if return cost is True
		print('COST of registration: CCMetric = ', metric.get_energy())

		# temporary middle image
		middle = np.add(moving_transformed_, static_transformed_)/np.array([2])

		return moving_transformed_, static_grid2world, metric.get_energy()

	def iterative_registration(dictionary, algorithm = 'diffeomorphic', output_file = 'template.nii.gz'):

		print('LEN dictionary keys : =====', len(list(dictionary.keys())))

		# main condition
		if len(dictionary) == 1:

			# save result
			x = list(dictionary.keys())[0]
			save_nifti(fname = output_file, data = dictionary[x]['im'], affine = dictionary[x]['affine'])

			# return registered image
			return output_file

		else:

			# temp dict
			temp = {}

			# loop
			keys = list(dictionary.keys())

			for i in range(0, len(keys), 2):

				# check if pair is possible
				if i+1 < len(keys):

					# Print out currently registering files
					print('Currently registering: ', keys[i], '++++and++++', keys[i+1])

					# assignments
					first_im = dictionary[keys[i]]['im']
					first_affine = dictionary[keys[i]]['affine']
					second_im = dictionary[keys[i+1]]['im']
					second_affine = dictionary[keys[i+1]]['affine']

					# Register

					if algorithm == 'affine':
						temp_im, temp_affine, cost = affine_registration_pair(first_im, first_affine, second_im, second_affine)                    
					elif algorithm == 'diffeomorphic':
						temp_im, temp_affine, cost = diffeomorphic_registration_pair(first_im, first_affine, second_im, second_affine)

					# Confirmation
					print('Done registering: ', keys[i], '++++and++++', keys[i+1])
					print('\n-----------------------------------------------------\n')

					temp[keys[i] + keys[i+1]] = {'im' : temp_im, 'affine' : temp_affine}

				else:
					temp[keys[i]] = {'im' : dictionary[keys[i]]['im'], 'affine' : dictionary[keys[i]]['affine']}

			return iterative_registration(temp, algorithm, output_file)

	def register_to_standard_template(template = 'ICBM152', to_register = None, output_file = None):

		if output_file:
			assert (type(output_file) == str)
			assert (output_file.endswith('.nii') or output_file.endswith('.nii.gz'))
		else:
			output_file = 'registered_to_' + template + 'nii.gz'

		# ICBM 2009A
		if template == 'ICBM152':
			if not os.path.exists('./standard_template/ICBM152_t1.nii.gz'):
				# download template
				print('\nDownloading template')
				wget.download('https://digital.lib.washington.edu/researchworks/bitstream/handle/1773/33312/mni_icbm152_t1_tal_nlin_asym_09a.nii?sequence=4&isAllowed=y', './standard_template/ICBM152_t1.nii.gz')
				print('\nDownloaded template')
			# diffeomorphic registration
			first_im, first_aff = load_nifti('./standard_template/ICBM152_t1.nii.gz', return_img = False)
			second_im, second_aff = load_nifti(to_register, return_img = False)
			# register
			im, aff, cost = diffeomorphic_registration_pair(first_im, first_aff, second_im, second_aff)
			print('\nCost of Registering to ICBM 152: CCMetric', cost)
			save_nifti(fname = output_file, data = im, affine = aff)
			print('\nDONE: Output stored at ', output_file)

		# IITv5.0
		elif template == 'IITv5.0':
			if not os.path.exists('./standard_template/IITv5.0_mean_DTI_FA.nii.gz'):
				# download template
				print('\nDownloading template')
				wget.download('https://www.nitrc.org/frs/download.php/11271/IITmean_FA.nii.gz', './standard_template/IITv5.0_mean_DTI_FA.nii.gz')
				print('\nDownloaded template')
			# diffeomorphic registration
			first_im, first_aff = load_nifti('./standard_template/IITv5.0_mean_DTI_FA.nii.gz', return_img = False)
			second_im, second_aff = load_nifti(to_register, return_img = False)
			# register
			im, aff, cost = diffeomorphic_registration_pair(first_im, first_aff, second_im, second_aff)
			print('\nCost of Registering to IITv5.0: CCMetric', cost)
			save_nifti(fname = output_file, data = im, affine = aff)
			print('\nDONE: Output stored at ', output_file)

		# error
		else:
			raise ValueError('Invalid standard template: Please provide either ICBM152 or IITv5.0')

		return 'Registered to :'+template

	def run(self, path_to_directory, use_algorithm = 'diffeomorphic', save_template = 'template.nii.gz', register_standard = None, save_standard = None):
		"""
		Parameters
		----------
		path_to_directory : string
			Path to the directory containing FA files for building a template

		use_algorithm : string, optional
			The algorithm to use for iterative registration.
			By default it is diffeomorphic.
			Also available - affine.

		save_template : string, optional
			File name to use to store created template.
			By default it is 'template.nii.gz'.

		register_standard : string, optional
			Whether to register the created template to a standard known atlas.
			By default it is None.
			Available - 'ICBM152' and 'IITv5.0'

		save_standard : string, optional
			File name to use to store template registered to standard template specified in register_standard.
			By default it will be 'registered_to_ICBM152.nii.gz' or 'registered_to_IITv5.0.nii.gz'

		"""
		files_list = create_list_of_files(path_to_directory)
		files_dict = prepare_data(files_list)
		created_template = iterative_registration(dictionary = files_dict, algorithm = use_algorithm, output_file = save_template)

		if register_standard:
			register_to_standard_template(template = register_standard, to_register = created_template, output_file = save_standard)