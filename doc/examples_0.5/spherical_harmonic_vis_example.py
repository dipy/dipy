from numpy import abs, asarray, concatenate
from enthought.mayavi import mlab
from dipy.data import sample_hardi_data
from dipy.core.triangle_subdivide import create_unit_sphere
from dipy.reconst.shm import normalize_data, MonoExpOpdfModel

def show_blobs(blobs, v, faces):
    """Mayavi gets really slow when triangular_mesh is called too many times
    sot this function stacks blobs and calls triangular_mesh once
    """

    blobs
    xcen = blobs.shape[0]/2
    ycen = blobs.shape[1]/2
    zcen = blobs.shape[2]/2
    faces = asarray(faces, 'int')
    xx = []
    yy = []
    zz = []
    count = 0
    ff = []
    mm = []
    for ii in xrange(blobs.shape[0]):
        for jj in xrange(blobs.shape[1]):
            for kk in xrange(blobs.shape[2]):
                m = blobs[ii,jj,kk]
                m /= (2.2*abs(m).max())
                x, y, z = v.T*m
                x += ii - xcen
                y += jj - ycen
                z += kk - zcen
                ff.append(count+faces)
                count += len(x)
                xx.append(x)
                yy.append(y)
                zz.append(z)
                mm.append(m)
    ff = concatenate(ff)
    xx = concatenate(xx)
    yy = concatenate(yy)
    zz = concatenate(zz)
    mm = concatenate(mm)

    mlab.triangular_mesh(xx, yy, zz, ff, scalars=mm)

def main():
    #set some values to be used later
    sh_order = 6
    verts, edges, efaces = create_unit_sphere(4)

    #read_data from disk
    data, fa, bvec, bval, voxel_size = sample_hardi_data()
    data_slice = data[32:76, 32:76, 26:27]
    fa_slice = fa[32:76, 32:76, 26]

    #normalize data by dividing by b0, this is needed so we can take log later
    norm_data = normalize_data(data_slice, bval, min_signal=1)

    #create an instance of the model
    model_instance = MonoExpOpdfModel(sh_order, bval, bvec, .006)
    model_instance.set_sampling_points(verts, edges)

    #use the model it fit the data
    opdfs_sampled_at_verts = model_instance.evaluate(norm_data)
    opdfs_sph_harm_coef = model_instance.fit_data(norm_data)

    #display the opdf blobs using mayavi
    faces = edges[efaces, 0]
    show_blobs(opdfs_sampled_at_verts, verts, faces)
    mlab.imshow(fa_slice, colormap='gray', interpolate=False)
    mlab.show()

if __name__ == '__main__':
    main()

