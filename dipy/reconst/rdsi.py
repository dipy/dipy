import numpy as np
import numpy.matlib
import pandas as pd

import torch
import torchkbnufft as tkbn

from dipy.reconst.odf import OdfModel, OdfFit
from dipy.reconst.cache import Cache
from dipy.reconst.multi_voxel import multi_voxel_fit

from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull


class RadialDsiModel(OdfModel, Cache):

    @staticmethod
    def extract_shells(values, diff_threshold = 50):
        shells = []
        sorted_values = np.sort(values)
      
        for values_chunk in np.split(sorted_values, 1 + np.nonzero(np.diff(sorted_values) > diff_threshold)[0]):
            shells.append(np.mean(values_chunk))
        
        return np.array(shells)


    def __init__(self, gtab, sampling_length=1.2):
        self.gtab = gtab

        if self.gtab.big_delta is None:
            self.gtab.big_delta = 1
        
        if self.gtab.small_delta is None:
            self.gtab.small_delta = 0
        
        self.dir_filter = ~gtab.b0s_mask

        # Take only the first b0 image
        if (np.sum(gtab.b0s_mask)):
            self.dir_filter[np.nonzero(gtab.b0s_mask)[0][0]] = True
        
        # Multiply by 2*np.pi for backward compatibility with the Matlab version
        self.qtable = np.vstack(self.gtab.qvals[self.dir_filter]) * self.gtab.bvecs[self.dir_filter] * 2 * np.pi

        self.max_displacement = sampling_length / (2 * np.max(np.diff(np.sqrt(
            self.extract_shells(self.gtab.bvals)
        ))))
            
    
    def fit(self, data):
        return RadialDsiFit(self, data)
    
    
class RadialDsiFit(OdfFit):

    def _sinc_second_derivative(self, x):
        result = np.zeros_like(x)
        
        near_zero_filter = np.abs(x) <= 1e-3
        
        x0 = x[near_zero_filter]
        x1 = x[~near_zero_filter]
        
        result[near_zero_filter] = -1/3 + x0 * x0 / 10
        result[~near_zero_filter] = 2 * np.sin(x1) / x1 / x1 / x1 - 2 * np.cos(x1) / x1 / x1 - np.sin(x1) / x1
        
        return result


    def __init__(self, model, data):
        self._model = model
        self._data = data
    
    
    # function wi = ir_mri_dcf_voronoi0(kspace, fix_edge)
    def ir_mri_dcf_voronoi0(self, kspace, fix_edge):
    
        # M = size(kspace, 1);
        M = kspace.shape[0]
    
        # i0 = sum(abs(kspace), 2) == 0; % which points are at origin?
        i0 = np.sum(np.abs(kspace), 1) == 0
        
        # if sum(i0) > 1 % multiple DC points?
        if np.sum(i0) > 1:
        
        #     i0f = find(i0);
        #     i0f = i0f(1); % keep the first zero point only
            i0f = np.nonzero(i0)[0][0]
        
        #     i0(i0f) = false; % trick
            i0[i0f] = False
        
        #     wi = zeros(M, 1);
            wi = np.zeros((M, 1))
            
        #     wi(~i0) = ir_mri_dcf_voronoi(kspace(~i0,:), fix_edge);
            wi[~i0] = self.ir_mri_dcf_voronoi(kspace[~i0,:], fix_edge)
        
        #     i0(i0f) = true; % trick
            i0[i0f] = True
        
        #     wi(i0) = wi(i0f) / sum(i0); % distribute dcf equally
            wi[i0] = wi[i0f] / np.sum(i0)
        
        # else
        #     wi = ir_mri_dcf_voronoi(kspace, fix_edge);
        # end    
        else:
            wi = self.ir_mri_dcf_voronoi(kspace, fix_edge)
        
        return wi


    # % ir_mri_dcf_voronoi()
    # %
    # function wi = ir_mri_dcf_voronoi(kspace, fix_edge)
    def ir_mri_dcf_voronoi(self, kspace, fix_edge):
    
        # M = size(kspace, 1);
        M = kspace.shape[0]

        # wi = zeros(M,1);
        wi = np.zeros((M, 1))
        
        # [v, c] = voronoin(double(kspace));
        vor = Voronoi(kspace, qhull_options='Qbb')
        v = vor.vertices
        c = vor.regions
        
        # nbad = 0;
        nbad = 0
        
        # for mm=1:M
        for mm in range(M):
        
            #     ticker([mfilename ' (voronoi)'], mm, M)
            #     x = v(c{mm},:);
            #     if ~any(isinf(x))
            #         try
            #             [~, wi(mm)] = convhulln(x); % cell area
            #         catch
            # %            printm('bad %d', mm)
            #             nbad = nbad + 1;
            #         end
            #     end


            if -1 not in c[vor.point_region[mm]]:
                x = v[c[vor.point_region[mm]],:]
                try:
                    conv_hull = ConvexHull(x)
                    wi[mm] = conv_hull.volume
                except:
                    print('bad %d' % mm)
                    nbad = nbad + 1
            
        # end
        # if nbad
        #     printm('bad edge points %d of %d', nbad, M)
        # end
        #
        # % points at the outer edges of k-space have infinite voronoi cell area
        # % so are assigned wi=0 above.  to improve on 0, here we extrapolate
        # % based on the points near the edge.
        # switch fix_edge
        # case 2
        #     rho = sum(kspace.^2, 2); % radial frequency coordinate
        #     igood = (rho > 0.6 * max(rho)) & (wi > 0);
        #     pp = polyfit(rho(igood), wi(igood), 2);
        #     wi(wi == 0) = polyval(pp, rho(wi == 0)); % extrapolate
        #
        # % old way: look for points close to convex hull and use max of other points?
        # case 1
        #     printm('trying to fix %d zeros of %d', sum(wi==0), M)
        #     ii = logical(zeros(size(wi)));
        #     fac = 0.98;
        #     for id=1:ncol(kspace) % find cartesian edges of k-space
        #         k = kspace(:,id);
        #         ii = ii | (k > fac * max(k)) | (k < fac * min(k));
        #     end
        #     if ncol(kspace) >= 2
        #         k = sqrt(kspace(:,1).^2 + kspace(:,2).^2);
        #         ii = ii | (k > fac * max(k)); % cylindrical edge
        #     end
        #     if ncol(kspace) >= 3
        #         k = sqrt(kspace(:,1).^2 + kspace(:,2).^2 + kspace(:,3).^2);
        #         ii = ii | (k > fac * max(k)); % spherical edge
        #     end
        #
        #     pn = jf_protected_names;
        #     wmax = 2 * pn.prctile(wi(~ii), 95); % fix: this is not working well
        #     wi = min(wi, wmax);
        #     wi(wi==0) = max(wi);
        #
        # otherwise
        #     if ~isequal(fix_edge, 0), error('bad fix_edge argument'), end
        # end
        #

        return wi
    
    
    def dcf_calc(self):
        
        # normalize = true;
        normalize = True

        # dimi = size(qtable,2);
        dimi = self._model.qtable.shape[1]
        
        # qval = sqrt(sum(qtable.^2,2));
        qval = self._model.gtab.qvals * 2 * np.pi
        
        # [qshells,nmeasshell,shell] = bval2shells(qval');
        qshells = RadialDsiModel.extract_shells(qval, 5)        
        
        # steps = qshells;
        steps = qshells
        
        # % k-space traj and FoV (for normalization)
        # nz = 2;
        # dq = mean(diff(qshells*1e3)); % m-1
        # Rmax = 1/dq; % m
        # FoV = 2*Rmax*[1,1,1];
        # res = (2*length(qshells)+1)*[1,1,1];
        # kspace = qtable*FoV(1)*1/res(1)*(pi/2);
        # kspace = [kspace;-kspace];
        # kspace = kspace/max(kspace(:))*1/2*2*pi;

        nz = 2
        dq = np.mean(np.diff(qshells * 1e3))
        Rmax = 1 / dq
        FoV = 2 * Rmax * np.ones(3)
        res = (2 * len(qshells) + 1) * np.ones(3)
        kspace = (self._model.qtable * FoV[0] * np.pi) / (2 * res[0])
        kspace = np.vstack((kspace,-kspace))
        kspace = (kspace * np.pi) / np.max(kspace)
        
        # % Nufft
        # if ~exist('nufft_init','file')
        #     warning(['The IRT package seems not installed. You will find it at:' ...
        #         ' http://web.eecs.umich.edu/~fessler/irt/irt/nufft .' ...
        #         ' dcfcalc.m will use a simple calculation of the DCF.']);
        #     estmethod = 'theor';
        #     normalize = false;
        # else
        #     st = nufft_init(kspace,...
        #         res, [5,5,5], 2*res,res, 'minmax:kb');
        # end;

        # st = NUFFT()
        # st.plan(om, Nd, Kd, Jd)
        
        #
        # % calculate dcf
        # switch estmethod
        #     case 'theor'
        #         % the dcf is a multiplication of 3 dimensions
        #         % two dimensions (orthogonal to the radial lines) are each similar to the
        #         % 2D-radial dcf, hence, ~ (qval)^2
        #         % the third dimension, along the radial lines, depends on the spacing
        #         % of the samples.
        #         dcf = qval/max(qval);
        #         if (dimi == 3)
        #             dcf = dcf.*dcf;
        #         end;
        #         dd = diff(steps)';
        #         dd=dd/sum([dd;dd(end)/2])*(length(dd)+1);
        #         dcfw = 1/2*[0;dd]+1/2*[dd;dd(end)];
        #         for i = 1:length(steps)
        #             sel = (shell == i);
        #             dcf(sel) = dcf(sel)*dcfw(i);
        #         end;
        #     case 'geom'
        #         % volume in q-space, or, (4/3 pi r2^3 - 4/3 pi r1^3) / nmeas with r1
        #         % and r2 dividing the space between the neighbouring shells
        #         diffsteps = diff(steps)';
        #         dsplit = [0,(qshells(1:(end-1))+qshells(2:end))/2,qshells(end)+diffsteps(end)/2];
        #         dsplit(2) = dsplit(2)/10;
        #         dcfshells = 4/3*pi*(dsplit(2:end).^3 - dsplit(1:(end-1)).^3)./nmeasshell;
        #         dcfshells = dcfshells/dcfshells(end);
        #         dcf = dcfshells(shell)';
        #
        #     % below are for non-uniform direction distribution (general case)
        #     case 'pipe2'
        #         % add extra outside shell
        #         qvalextra = (length(qshells)*qshells(2:end)/(1:(length(qshells)-1)));
        #         dirextra = dlmread('dir1000uniform.txt');
        #         dirextra = qtable((shell == length(qshells)),:);
        #         kspacet = [kspace;dirextra*1/2*2*pi*qvalextra/qshells(end)]*qshells(end)/qvalextra;
        #         st3 = nufft_init(kspacet/nz,...
        #             nz*res, [5,5,5], 2*nz*res,res, 'minmax:kb');
        #         H.arg.st = st3;
        #         Dest = ir_mri_density_comp_v2(kspacet, estmethod,...
        #               'G',H,'arg_pipe',{'Nsize',nz*res(1),'niter',50});
        #         Dest = Dest(1:(end-size(dirextra,1)));
        #         dcf = Dest(1:(end/2));
        # %     case 'pipe2'
        # %         st2 = nufft_init(kspace/nz,...
        # %             nz*res, [5,5,5], 2*nz*res,res, 'minmax:kb');
        # %         G.arg.st = st2;
        # %         Dest = ir_mri_density_comp_v2(kspace, estmethod,...
        # %               'G',G,'arg_pipe',{'Nsize',nz*res(1),'niter',50});
        # %         dcf = Dest(1:(end/2));
        #     case 'voronoi'
        #         % add extra outside shell
        #         qvalextra = (length(qshells)*qshells(2:end)/(1:(length(qshells)-1)));
        qvalextra = max(qshells) + np.mean(np.diff(qshells[1:]))
        
        #         dirextra = dlmread('dir1000uniform.txt');
        dirextra = pd.read_csv('/home/patrykfi/matlab/odffingerprintingdev/utils/dir1000uniform.txt', sep=' ', header=None).values

        #         kspacet = [kspace;dirextra*1/2*2*pi*qvalextra/qshells(end)]*qshells(end)/qvalextra;
        kspacet = np.vstack((kspace,dirextra*1/2*2*np.pi*qvalextra/qshells[-1])) * qshells[-1]/qvalextra

        #         st3 = nufft_init(kspacet/nz,...
        #             nz*res, [5,5,5], 2*nz*res,res, 'minmax:kb');
        # st3 = NUFFT()
        # om = kspacet/nz
        # Nd = tuple(nz * res.astype(int)) 
        # Kd = tuple(2 * nz * res.astype(int))
        # Jd = (5,5,5)
        # st3.plan(om, Nd, Kd, Jd)

        #         H.arg.st = st3;
        #         Dest = ir_mri_density_comp_v2(kspacet, estmethod,...
        #           'G',H,'fix_edge',0);
        Dest = self.ir_mri_dcf_voronoi0(kspacet, 0)
        
        #         Dest = Dest(1:(end-size(dirextra,1)));
        Dest = Dest[:-dirextra.shape[0]]
        
        #         dcf = Dest(1:(end/2));
        dcf = Dest[:int(Dest.shape[0]/2)]
        
        #     case {'jackson';'pipe'}
        #         st2 = nufft_init(kspace/nz,...
        #             nz*res, [5,5,5], 2*nz*res,res, 'minmax:kb');
        #         G.arg.st = st2;
        #         Dest = ir_mri_density_comp_v2(kspace, estmethod,'G',G);
        #         dcf = Dest(1:(end/2));
        # end;
        #
        # % normalize 
        # if normalize
        #     out = (st.p'*(repmat(dcf,[2,1]).*((st.p)*ones(prod(2*res),1))));
        #     dcf = dcf/mean(abs(out(:)));
        # end;

        if normalize:
            
            om = kspace
            Nd = tuple(res.astype(int)) 
            Kd = tuple(2 * res.astype(int))
            # Jd = (5,5,5)

            spmatrix = tkbn.calc_tensor_spmatrix(torch.from_numpy(kspace.T), im_size=Nd, grid_size=Kd, numpoints=5)
            stp = spmatrix[0] + 1j*spmatrix[1]
            dcf2 = np.matlib.repmat(dcf, 2, 1)
            
            o = torch.complex(
                torch.from_numpy(np.ones((int(np.prod(2*res)), 1))), 
                torch.from_numpy(np.zeros((int(np.prod(2*res)), 1)))
            )
            
            out = torch.matmul(stp.to_dense().T, torch.from_numpy(dcf2) * torch.matmul(stp, o))
            dcf = dcf / torch.mean(torch.abs(out)).item()
        
        return dcf
    
    
    def odf(self, sphere):
        h = self.dcf_calc()

        E = np.dot(sphere.vertices, self._model.qtable.T)
        F = np.multiply(
            -self._sinc_second_derivative(2 * np.pi * E * self._model.max_displacement),
            np.matlib.repmat(h.T, E.shape[0], 1)
        )
        
        odf = np.matmul(self._data[...,self._model.dir_filter], F.T)
        
        return odf