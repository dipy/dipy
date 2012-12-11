import warnings
import numpy as np
from scipy.ndimage import map_coordinates
from dipy.reconst.recspeed import le_to_odf, sum_on_blocks_1d
from dipy.reconst.dsi import project_hemisph_bvecs
from scipy.ndimage.filters import laplace,gaussian_laplace
from scipy.ndimage import zoom,generic_laplace,correlate1d
from dipy.core.geometry import sphere2cart,cart2sphere,vec2vec_rotmat
from dipy.tracking.propspeed import map_coordinates_trilinear_iso
from dipy.reconst.odf import OdfModel


###############################################
# MODULE TEMPORARILY DISABLED FOR REFACTORING #
###############################################

import nose
class UnderConstruction(nose.SkipTest):
    pass

raise UnderConstruction()

###############################################



class DiffusionNablaModel(OdfModel):
    def __init__(self, bvals, gradients, odf_sphere='symmetric362',
                 half_sphere_grads=False, fast=True):
        ''' Reconstruct the signal using Diffusion Nabla Imaging

        As described in E.Garyfallidis, "Towards an accurate brain
        tractograph"tractograph, PhD thesis, 2011.

        Parameters
        -----------
        bvals : array, shape (N,)
        gradients : array, shape (N,3) also known as bvecs
        odf_sphere : str or tuple, optional
            If str, then load sphere of given name using ``get_sphere``.
            If tuple, gives (vertices, faces) for sphere.
        filter : array, shape(len(vertices),)
            default is None (using standard hanning filter for DSI)
        half_sphere_grads : boolean Default(False)
            in order to create the q-space we use the bvals and gradients.
            If the gradients are only one hemisphere then

        See also
        ----------
        dipy.reconst.eit.EquatorialInversionModel, dipy.reconst.dti.TensorModel, dipy.reconst.dsi.DiffusionSpectrumModel
        '''

        #check if bvectors are provided only on a hemisphere
        if half_sphere_grads==True:
            pass
            #bvals=np.append(bvals.copy(),bvals[1:].copy())
            #gradients=np.append(gradients.copy(),-gradients[1:].copy(),axis=0)
            #data=np.append(data.copy(),data[...,1:].copy(),axis=-1)

        #load bvals and bvecs
        self.bvals=bvals
        gradients[np.isnan(gradients)] = 0.
        self.gradients=gradients
        #save number of total diffusion volumes
        self.dn=self.gradients.shape[0] #data.shape[-1]
        odf_vertices, odf_faces = sphere_vf_from(odf_sphere)
        self.set_odf_vertices(odf_vertices,None,odf_faces)
        self.odfn=odf_vertices.shape[0]

        #odf sampling radius
        self.radius=np.arange(0,5,.2)
        #self.radiusn=len(self.radius)
        #self.create_qspace(bvals,gradients,16,8)
        #peak threshold
        #self.peak_thr=.7
        #equatorial zone
        self.zone=5.
        self.gaussian_weight=0.05
        self.fast=fast
        if fast==True:
            self.evaluate_odf=self.fast_odf
        else:
            self.evaluate_odf=self.slow_odf
        self.precompute()

    def precompute(self):

        self.radiusn=len(self.radius)
        self.create_qspace(self.bvals,self.gradients,17,8)
        if self.fast==False:
            self.radon_params()
            self.precompute_interp_coords()
        if self.fast==True:
            self.precompute_fast_coords()
            self.precompute_equator_indices(self.zone)
        self.precompute_angular(self.gaussian_weight)

    def precompute_botox(self,smooth,level):
        self.botox_smooth=.05
        self.botox_level=.3

    def precompute_angular(self,smooth):
        if smooth==None:
            self.E=None
            return
        self.W=np.dot(self.odf_vertices,self.odf_vertices.T)
        self.W=self.W.astype('f8')
        E=np.exp(self.W/smooth)
        self.E=E/np.sum(E,axis=1)[:,None]

    def create_qspace(self,bvals,gradients,size,origin):
        bv=bvals
        bmin=np.sort(bv)[1]
        bv=np.sqrt(bv/bmin)
        qtable=np.vstack((bv,bv,bv)).T*gradients
        qtable=np.floor(qtable+.5)
        self.qtable=qtable
        self.q=qtable+origin
        self.q=self.q.astype('i8')
        self.origin=origin
        self.sz=size

    def radon_params(self,ang_res=64):
        #calculate radon integration parameters
        phis=np.linspace(0,2*np.pi,ang_res)[:-1]
        planars=[]
        for phi in phis:
            planars.append(sphere2cart(1,np.pi/2,phi))
        planars=np.array(planars)
        planarsR=[]
        for v in self.odf_vertices:
            R=vec2vec_rotmat(np.array([0,0,1]),v)
            planarsR.append(np.dot(R,planars.T).T)
        self.equators=planarsR
        self.equatorn=len(phis)

    def slow_odf(self,s):
        """ Calculate the orientation distribution function
        """
        odf = np.zeros(self.odfn)
        Eq=np.zeros((self.sz,self.sz,self.sz))
        for i in range(self.dn):
            Eq[self.q[i][0],self.q[i][1],self.q[i][2]]=s[i]/np.float(s[0])
        LEq=laplace(Eq)
        self.Eq=Eq
        self.LEq=LEq
        LEs=map_coordinates(LEq,self.Xs,order=1)
        le_to_odf(odf,LEs,self.radius,self.odfn,self.radiusn,self.equatorn)
        return odf

    def odfs(self):
        return self.ODF

    def fast_odf(self,s):
        odf = np.zeros(self.odfn)
        Eq=np.zeros((self.sz,self.sz,self.sz))
        for i in xrange(self.dn):
            Eq[self.q[i][0],self.q[i][1],self.q[i][2]]+=s[i]/s[0]
        LEq=laplace(Eq)
        LEs=map_coordinates(LEq,self.Ys.T,order=1)
        LEs=LEs.reshape(self.odfn,self.radiusn)
        LEs=LEs*self.radius
        LEsum=np.sum(LEs,axis=1)
        for i in xrange(self.odfn):
            odf[i]=np.sum(LEsum[self.eqinds[i]])/self.eqinds_len[i]
        return -odf

    def precompute_equator_indices(self,thr=5):
        eq_inds=[]
        eq_inds_complete=[]
        eq_inds_len=np.zeros(self.odfn)
        for (i,v) in enumerate(self.odf_vertices):
            eq_inds.append([])
            for (j,k) in enumerate(self.odf_vertices):
                vk=np.clip(np.dot(v,k),-1,1)
                angle=np.rad2deg(np.arccos(vk))
                if  angle < 90 + thr and angle > 90 - thr:
                    eq_inds[i].append(j)
                    eq_inds_complete.append(j)
            eq_inds_len[i]=len(eq_inds[i])
        self.eqinds=eq_inds
        self.eqinds_com=np.array(eq_inds_complete)
        self.eqinds_len=np.array(eq_inds_len,dtype='i8')

    def precompute_fast_coords(self):
        Ys=[]
        for m in range(self.odfn):
            for q in self.radius:
                #print disk.shape
                xi=self.origin + q*self.odf_vertices[m,0]
                yi=self.origin + q*self.odf_vertices[m,1]
                zi=self.origin + q*self.odf_vertices[m,2]
                Ys.append(np.vstack((xi,yi,zi)).T)
        self.Ys=np.ascontiguousarray(np.concatenate(Ys))
        self.Ysn=self.Ys.shape[0]

    def precompute_interp_coords(self):
        Xs=[]
        for m in range(self.odfn):
            for q in self.radius:
                #print disk.shape
                xi=self.origin + q*self.equators[m][:,0]
                yi=self.origin + q*self.equators[m][:,1]
                zi=self.origin + q*self.equators[m][:,2]
                Xs.append(np.vstack((xi,yi,zi)).T)
        self.Xs=np.concatenate(Xs).T

class EquatorialInversionModel(DiffusionNablaModel):
    ''' Reconstruct the signal using Equatorial Inversion Transform

        As described in E.Garyfallidis, "Towards an accurate brain
        tractograph"tractograph, PhD thesis, 2011.

        Parameters
        -----------
        bvals : array, shape (N,)
        gradients : array, shape (N,3) also known as bvecs
        odf_sphere : str or tuple, optional
            If str, then load sphere of given name using ``get_sphere``.
            If tuple, gives (vertices, faces) for sphere.
        filter : array, shape(len(vertices),)
            default is None (using standard hanning filter for DSI)
        half_sphere_grads : boolean Default(False)
            in order to create the q-space we use the bvals and gradients.
            If the gradients are only one hemisphere then

        See also
        ----------
        dipy.reconst.eit.EquatorialInversionModel, dipy.reconst.dti.TensorModel, dipy.reconst.dsi.DiffusionSpectrumModel
        '''

    def set_operator(self,name):
        self.operator=name

    def set_mode(self,order=1,zoom=1,mode='constant'):
        self.order=order
        self.mode=mode
        self.zoom=zoom
        #self.Eqs=[]

    def fast_odf(self,s):
        odf = np.zeros(self.odfn)
        Eq=np.zeros((self.sz,self.sz,self.sz))
        #for i in range(self.dn):
        #    Eq[self.q[i][0],self.q[i][1],self.q[i][2]]+=s[i]/s[0]
        Eq[self.q[:,0],self.q[:,1],self.q[:,2]]=s[:]/np.float(s[0])
        #self.Eqs.append(Eq)
        if  self.operator=='laplacian':
            LEq=laplace(Eq)
            sign=-1
        if self.operator=='laplap':
            LEq=laplace(laplace(Eq))
            sign=1
        if  self.operator=='signal':
            LEq=Eq
            sign=1
        #LEs=map_coordinates(LEq,self.Ys.T,order=1)
        #"""
        LEs=np.zeros(self.Ysn)
        strides=np.array(LEq.strides,'i8')
        map_coordinates_trilinear_iso(LEq,self.Ys,
                                      strides,self.Ysn, LEs)
        #LEs=map_coordinates(LEq,self.zoom*self.Ys,order=1)
        LEs=LEs.reshape(self.odfn,self.radiusn)
        LEs=LEs*self.radius
        #LEs=LEs*self.radius*self.zoom
        LEsum=np.sum(LEs,axis=1)
        #This is what the following code is doing
        #for i in xrange(self.odfn):
        #    odf[i]=np.sum(LEsum[self.eqinds[i]])/self.eqinds_len[i]
        #odf2=odf.copy()
        LES=LEsum[self.eqinds_com]
        sum_on_blocks_1d(LES,self.eqinds_len,odf,self.odfn)
        odf=odf/self.eqinds_len
        return self.angular_weighting(sign*odf)

    def angular_weighting(self,odf):
        if self.E==None:
            return odf
        else:
            return np.dot(odf[None,:],self.E).ravel()
