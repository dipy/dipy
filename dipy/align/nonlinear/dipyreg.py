"""
This script is the main launcher of the multi-modal non-linear image
registration
"""
import sys
import os
import numpy as np
import nibabel as nib
import RegistrationCommon as rcommon
import VectorFieldUtils as vfu
from SymmetricRegistrationOptimizer import SymmetricRegistrationOptimizer
from CCMetric import CCMetric
import UpdateRule as UpdateRule
from dipy.fixes import argparse as arg

parser = arg.ArgumentParser(
    description=
        "Multi-modal, non-linear image registration. By default, it does NOT "
        "save the resulting deformation field but only the deformed target "
        "image using tri-linear interpolation under two transformations: "
        "(1)the affine transformation used to pre-register the images and (2) "
        "the resulting displacement field. If a warp directory is specified, "
        "then each image inside the directory will be warped using "
        "nearest-neighbor interpolation under the resulting deformation field."
        " The name of the warped target image under the AFFINE transformation"
        "will be 'warpedAffine_'+baseTarget+'_'+baseReference+'.nii.gz', "
        "where baseTarget is the file name of the target image (excluding all "
        "characters at and after the first '.'), baseReference is the file "
        "name of the reference image (excluding all characters at and after "
        "the first '.'). For example, if target is 'a.nii.gz' and reference is"
        " 'b.nii.gz', then the resulting deformation field will be saved as "
        "'dispDiff_a_b.npy'.\n"
        "The name of the warped target image under the NON-LINEAR "
        "transformation will be "
        "'warpedAffine_'+baseTarget+'_'+baseReference+'.nii.gz', with the "
        "convention explained above.\n"

        "Similarly, the name of each warped image inside the warp directory "
        "will be 'warpedDiff_'+baseWarp+'_'+baseReference+'.nii.gz' where "
        "baseWarp is the name of the corresponding image following the same "
        "convetion as above.\n"
        "Example:\n"
        "python dipyreg.py target.nii.gz reference.nii.gz "
        "ANTS_affine_target_reference.txt --smooth=25.0 --iter 10,50,50"
    )

parser.add_argument(
    'target', action = 'store', metavar = 'target',
    help = '''Nifti1 image or other formats supported by Nibabel''')

parser.add_argument(
    'reference', action = 'store', metavar = 'reference',
    help = '''Nifti1 image or other formats supported by Nibabel''')

parser.add_argument(
    'affine', action = 'store', metavar = 'affine',
    help = '''ANTS affine registration matrix (.txt) that registers target to
           reference''')

parser.add_argument(
    'warp_dir', action = 'store', metavar = 'warp_dir',
    help = '''Directory (relative to ./ ) containing the images to be warped
           with the obtained deformation field''')

parser.add_argument(
    '-m', '--metric', action = 'store', metavar = 'metric',
    help = '''Any of {EM[L], CC[L]} specifying the metric to be used
    SSD=sum of squared diferences (monomodal), EM=Expectation Maximization
    to fit the transfer functions (multimodal), CC=Cross Correlation (monomodal
    and some multimodal) and the comma-separated (WITH NO SPACES) parameter list L:
    EM[step_lentgh,lambda,qLevels,max_inner_iter]
        step_length: the maximum norm among all vectors of the displacement at each iteration
        lambda: the smoothing parameter (the greater the smoother)
        qLevels: number of quantization levels (hidden variables) in the EM formulation
        max_inner_iter: maximum number of iterations of each level of the multi-resolution Gauss-Seidel algorithm
        e.g.: EM[0.25,25.0,256,20] (NO SPACES)
    CC[step_length,sigma_smooth,neigh_radius]
        step_length: the maximum norm among all vectors of the displacement at each iteration
        sigma_smooth: std. dev. of the smoothing kernel to be used to smooth the gradient at each step
        neigh_radius: radius of the squared neighborhood to be used to compute the Cross Correlation at each voxel
        e.g.:CC[0.25,3.0,4] (NO SPACES)
    ''',
    default = 'CC[0.25,3.0,4]')

parser.add_argument(
    '-i', '--iter', action = 'store', metavar = 'i_0,i_1,...,i_n',
    help = '''A comma-separated (WITH NO SPACES) list of integers indicating the maximum number
           of iterations at each level of the Gaussian Pyramid (similar to
           ANTS), e.g. 10,100,100 (NO SPACES)''',
    default = '25,50,100')

parser.add_argument(
    '-inv_iter', '--inversion_iter', action = 'store', metavar = 'max_iter',
    help = '''The maximum number of iterations for the displacement field
           inversion algorithm''',
    default='20')

parser.add_argument(
    '-inv_tol', '--inversion_tolerance', action = 'store',
    metavar = 'tolerance',
    help = '''The tolerance for the displacement field inversion algorithm''',
    default = '1e-3')

parser.add_argument(
    '-ii', '--inner_iter', action = 'store', metavar = 'max_iter',
    help = '''The number of Gauss-Seidel iterations to be performed to minimize
           each linearized energy''',
    default='20')

parser.add_argument(
    '-ql', '--quantization_levels', action = 'store', metavar = 'qLevels',
    help = '''The number levels to be used for the EM quantization''',
    default = '256')

parser.add_argument(
    '-single', '--single_gradient', action = 'store_true',
    help = '''The number levels to be used for the EM quantization''')

parser.add_argument(
    '-rs', '--report_status', action = 'store_true',
    help = '''Instructs the algorithm to show the overlaid registered images
           after each pyramid level''')

parser.add_argument(
    '-sd', '--save_displacement', dest = 'output_list',
    action = 'append_const', const='displacement',
    help = r'''Specifies that the displacement field must be saved. The
           displacement field will be saved in .npy format. The filename will
           be the concatenation:
           'dispDiff_'+baseTarget+'_'+baseReference+'.npy'
           where baseTarget is the file name of the target image (excluding all
           characters at and after the first '.'), baseReference is the file
           name of the reference image (excluding all characters at and after
           the first '.'). For example, if target is 'a.nii.gz' and reference
           is 'b.nii.gz', then the resulting deformation field will be saved as
           'dispDiff_a_b.npy'.''')

parser.add_argument(
    '-si', '--save_inverse', dest = 'output_list', action = 'append_const',
    const = 'inverse',
    help = r'''Specifies that the inverse displacement field must be saved.
           The displacement field will be saved in .npy format. The filename
           will be the concatenation:
           'invDispDiff_'+baseTarget+'_'+baseReference+'.npy'
           where baseTarget is the file name of the target image (excluding all
           characters at and after the first '.'), baseReference is the file
           name of the reference image (excluding all characters at and after
           the first '.'). For example, if target is 'a.nii.gz' and reference
           is 'b.nii.gz', then the resulting deformation field will be saved as
           'invDispDiff_a_b.npy'.''')

parser.add_argument(
    '-sl', '--save_lattice', dest = 'output_list', action = 'append_const',
    const = 'lattice',
    help = r'''Specifies that the deformation lattice (i.e., the obtained
           deformation field applied to a regular lattice) must be saved. The
           displacement field will be saved in .npy format. The filename will
           be the concatenation:
           'latticeDispDiff_'+baseTarget+'_'+baseReference+'.npy'
           where baseTarget is the file name of the target image (excluding all
           characters at and after the first '.'), baseReference is the file
           name of the reference image (excluding all characters at and after
           the first '.'). For example, if target is 'a.nii.gz' and reference
           is 'b.nii.gz', then the resulting deformation field will be saved as
           'latticeDispDiff_a_b.npy'.''')

parser.add_argument(
    '-sil', '--save_inverse_lattice', dest = 'output_list',
    action = 'append_const', const = 'inv_lattice',
    help = r'''Specifies that the inverse deformation lattice (i.e., the
           obtained inverse deformation field applied to a regular lattice)
           must be saved. The displacement field will be saved in .npy format.
           The filename will be the concatenation:
           'invLatticeDispDiff_'+baseTarget+'_'+baseReference+'.npy'
           where baseTarget is the file name of the target image (excluding all
           characters at and after the first '.'), baseReference is the file
           name of the reference image (excluding all characters at and after
           the first '.'). For example, if target is 'a.nii.gz' and reference
           is 'b.nii.gz', then the resulting deformation field will be saved as
           'invLatticeDispDiff_a_b.npy'.''')

def check_arguments(params):
    r'''
    Verify all arguments were correctly parsed and interpreted
    '''
    print(params.target, params.reference, params.affine, params.warp_dir)
    print('iter:', [int(i) for i in params.iter.split(',')])
    print(params.inner_iter, params.quantization_levels,
          params.single_gradient)
    print('Inversion:', params.inversion_iter, params.inversion_tolerance)
    print('---------Output requested--------------')
    print(params.output_list)

def save_deformed_lattice_3d(displacement, oname):
    r'''
    Applies the given displacement to a regular lattice and saves the resulting
    image to a Nifti file with the given name
    '''
    min_val, max_val = vfu.get_displacement_range(displacement, None)
    shape = np.array([np.ceil(max_val[0]), np.ceil(max_val[1]),
                  np.ceil(max_val[2])], dtype = np.int32)
    lattice = np.array(rcommon.drawLattice3D(shape, 10))
    warped = np.array(vfu.warp_volume(lattice, displacement)).astype(np.int16)
    img = nib.Nifti1Image(warped, np.eye(4))
    img.to_filename(oname)

def save_registration_results(init_affine, displacement, inverse, params):
    r'''
    Warp the target image using the obtained deformation field
    '''
    fixed = nib.load(params.reference)
    fixed_affine = fixed.get_affine()
    reference_shape = np.array(fixed.shape, dtype=np.int32)
    warp_dir = params.warp_dir
    base_moving = rcommon.getBaseFileName(params.target)
    base_fixed = rcommon.getBaseFileName(params.reference)
    moving = nib.load(params.target).get_data().squeeze().astype(np.float64)
    moving = moving.copy(order='C')
    warped = np.array(vfu.warp_volume(moving, displacement)).astype(np.int16)
    img_warped = nib.Nifti1Image(warped, fixed_affine)
    img_warped.to_filename('warpedDiff_'+base_moving+'_'+base_fixed+'.nii.gz')
    #---warp the target image using the affine transformation only---
    moving = nib.load(params.target).get_data().squeeze().astype(np.float64)
    moving = moving.copy(order='C')
    warped = np.array(
        vfu.warp_volume_affine(moving, reference_shape, init_affine)
        ).astype(np.int16)
    img_warped = nib.Nifti1Image(warped, fixed_affine)
    img_warped.to_filename('warpedAffine_'+base_moving+'_'+base_fixed+'.nii.gz')
    #---warp all volumes in the warp directory using NN interpolation
    names = [os.path.join(warp_dir, name) for name in os.listdir(warp_dir)]
    for name in names:
        to_warp = nib.load(name).get_data().squeeze().astype(np.int32)
        to_warp = to_warp.copy(order='C')
        base_warp = rcommon.getBaseFileName(name)
        warped = np.array(
            vfu.warp_volume_nn(to_warp, displacement)).astype(np.int16)
        img_warped = nib.Nifti1Image(warped, fixed_affine)
        img_warped.to_filename('warpedDiff_'+base_warp+'_'+base_fixed+'.nii.gz')
    #---finally, the optional output
    if params.output_list == None:
        return
    if 'lattice' in params.output_list:
        save_deformed_lattice_3d(
            displacement,
            'latticeDispDiff_'+base_moving+'_'+base_fixed+'.nii.gz')
    if 'inv_lattice' in params.output_list:
        save_deformed_lattice_3d(
            inverse, 'invLatticeDispDiff_'+base_moving+'_'+base_fixed+'.nii.gz')
    if 'displacement' in params.output_list:
        np.save('dispDiff_'+base_moving+'_'+base_fixed+'.npy', displacement)
    if 'inverse' in params.output_list:
        np.save('invDispDiff_'+base_moving+'_'+base_fixed+'.npy', inverse)

def register_3d(params):
    r'''
    Runs the non-linear registration with the parsed parameters
    '''
    print('Registering %s to %s'%(params.target, params.reference))
    sys.stdout.flush()
    ####Initialize parameter dictionaries####
    metric_name=params.metric[0:params.metric.find('[')]
    metric_params_list=params.metric[params.metric.find('[')+1:params.metric.find(']')].split(',')
    if metric_name=='EM':
        print 'EM Metric not supported yet. Aborting.'
        return
    elif metric_name=='CC':
        metric_parameters = {
            'max_step_length':float(metric_params_list[0]),
            'sigma_diff':float(metric_params_list[1]),
            'radius':int(metric_params_list[2])}
        similarity_metric = CCMetric(3, metric_parameters)
    optimizer_parameters = {
        'max_iter':[int(i) for i in params.iter.split(',')],
        'inversion_iter':int(params.inversion_iter),
        'inversion_tolerance':float(params.inversion_tolerance),
        'report_status':True if params.report_status else False}
    moving = nib.load(params.target)
    moving_affine = moving.get_affine()
    fixed = nib.load(params.reference)
    fixed_affine = fixed.get_affine()
    print 'Affine:', params.affine
    if not params.affine:
        transform = np.eye(4)
    else:
        transform = rcommon.readAntsAffine(params.affine)
    init_affine = np.linalg.inv(moving_affine).dot(transform.dot(fixed_affine))
    #print initAffine
    moving = moving.get_data().squeeze().astype(np.float64)
    fixed = fixed.get_data().squeeze().astype(np.float64)
    moving = moving.copy(order='C')
    fixed = fixed.copy(order='C')
    moving = (moving-moving.min())/(moving.max()-moving.min())
    fixed = (fixed-fixed.min())/(fixed.max()-fixed.min())
    ###################Run registration##################

    update_rule = UpdateRule.Composition()
    registration_optimizer = SymmetricRegistrationOptimizer(
        fixed, moving, None, init_affine, similarity_metric, update_rule,
        optimizer_parameters)
    registration_optimizer.optimize()
    displacement = registration_optimizer.get_forward()
    inverse = registration_optimizer.get_backward()
    del registration_optimizer
    del similarity_metric
    del update_rule
    save_registration_results(init_affine, displacement, inverse, params)

def test_exec():
    target='target/IBSR_01_ana_strip.nii.gz'
    reference='reference/t1_icbm_normal_1mm_pn0_rf0_peeled.nii.gz'
    affine='IBSR_01_ana_strip_t1_icbm_normal_1mm_pn0_rf0_peeledAffine.txt'
    paramiter='0x30x30'
    inversion_iter='20'
    inversion_tolerance='1e-3'
    report_status=True
    print('Registering %s to %s'%(target, reference))
    sys.stdout.flush()
    ####Initialize parameter dictionaries####
    metric_parameters = {
        'max_step_length':0.25,
        'sigma_diff':3.0,
        'radius':4}
    similarity_metric = CCMetric(3, metric_parameters)
    optimizer_parameters = {
        'max_iter':[int(i) for i in paramiter.split(',')],
        'inversion_iter':int(inversion_iter),
        'inversion_tolerance':float(inversion_tolerance),
        'report_status':True if report_status else False}
    moving = nib.load(target)
    moving_affine = moving.get_affine()
    fixed = nib.load(reference)
    fixed_affine = fixed.get_affine()
    print 'Affine:', affine
    if not affine:
        transform = np.eye(4)
    else:
        transform = rcommon.readAntsAffine(affine)
    init_affine = np.linalg.inv(moving_affine).dot(transform.dot(fixed_affine))
    #print initAffine
    moving = moving.get_data().squeeze().astype(np.float64)
    fixed = fixed.get_data().squeeze().astype(np.float64)
    moving = moving.copy(order='C')
    fixed = fixed.copy(order='C')
    moving = (moving-moving.min())/(moving.max()-moving.min())
    fixed = (fixed-fixed.min())/(fixed.max()-fixed.min())
    ###################Run registration##################

    update_rule = UpdateRule.Composition()
    registration_optimizer = SymmetricRegistrationOptimizer(
        fixed, moving, None, init_affine, similarity_metric, update_rule,
        optimizer_parameters)
    registration_optimizer.optimize()
    #displacement = registration_optimizer.get_forward()
    #inverse = registration_optimizer.get_backward()
    del registration_optimizer
    del similarity_metric
    del update_rule
    #save_registration_results(init_affine, displacement, inverse, params)


if __name__ == '__main__':
    register_3d(parser.parse_args())
