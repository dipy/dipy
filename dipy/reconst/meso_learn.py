import numpy as np
import scipy as sc


class BayesModel(data):

    # simulation parameters
    N = 1   # number of coils (noisetype
    nt = pow(10, 4)
    ncross = [1, 2, 30]
    ten = data.tensor / 1000

    # model parameters
    lmax = 2
    # reg = pow((0 * 10), -10)
    includeb0 = True
    bayesmodel = 'poly'
    order = 3

    qspace = 'qball'
    nmax = 3
    D0 = 1

    force_retraining = 0
    noise = 'estlocal'

    # sample parameter space
    D1 = 0.2 + np.random(nt, 1) * 2.8
    D2 = 0.2 + np.random(nt, 1) * 2.8
    D3 = 0.2 + np.random(nt, 1) * 2.8
    V = np.random(nt, 1) * 1
    Vw = (1-V) * np.random(nt, 1)

    idx = np.where(abs(D1 - (D2 + 2 * D3)) < 0.5)
    D1 = np.asarray(D1[idx])
    D2 = np.asarray(D2[idx])
    D3 = np.asarray(D3[idx])
    V = np.asarray(V[idx])
    Vw = np.asarray(Vw[idx])

    meanD = (D1[:] * V[:] + (D2[:] + 2 * D3[:]) * (1-(V[:] + Vw[:])) + 9 *
             Vw[:])/3
    microAx = (D1[:] * V[:] + (D2[:]) * (1 - (V[:] + Vw[:])) + 3 * Vw[:])
    microRad = ((D3[:]) * (1 - (V[:] + Vw[:])) + 3 * Vw[:])
    microFA = np.sqrt(3/2 * (pow((microAx - (microAx + 2 * microRad) / 3), 2)
                      + 2 * pow((microRad - (microAx + 2 * microRad) / 3), 2))
                      / (pow(microAx, 2) + 2 * pow(microRad, 2)))
    microFA = abs((D2[:] - D1[:]) / D3[:] + 4) > np.sqrt(40 / 3)
    numsamples = len(D1)
    scheme, b = getDirs(ten)

    # simulate the signal
    def TrainModel(data, varagin):
        beta = []
        for j in range(len(B0)):
            S0 = B0[j] * Signal
            S = 0
            for k in N:
                S = S + pow(abs(S0 + nz * np.sqrt(N) *
                                np.random.normal(np.size(S0))
                                + li * np.random.normal(np.size(S0))), 2)
            S = np.sqrt(S / N)
            S = S / np.tile(np.mean(S[:, round(b * 10), 1], [0, size(S, 2)]))

            M, shprojmat = compPowerSpec(b, scheme, lmax, S, x, qspace, nmax,
                                         D0)
            P = compMesoInv(M, lmax, order, includeb0, bayesmodel)
            P = compRegress(P, lmax, order, includeb0, bayesmodel)

            vf_linsub = 1 / 3
            R = np.eye(size(P, 2))
            target = np.tile([D1[:], D2[:], D3[:], V[:]-vf_linsub,
                              Vw[:] - vf_linsub, 1 - (V[:] + Vw[:])
                              - vf_linsub, meanD, microAx, microRad, microFA,
                              D2[:] / D3[:]], len(ncross))

            alpha = np.linalg.pinv(P_ * P + reg * R * np.size(P, 2)) * \
                                  (P_ * target)
            pred = P * alpha
            pred[:, 3:5] = pred[:, 3:5] + vf_linsub
            target[:, 3:5] = target[:, 3:5] + vf_linsub
            ALPHA[j] = alpha
            err = np.sqrt(sum(pow(pred-target), 2)
                          / sum(pow(target - np.tile(np.mean(target),
                                    np.size(target)))))

    def est(SNR, data, ten, preprocfunc):
        scheme, b = getDirs(ten / 1000)
        S0 = SNR * nz
        M = compPowerSpec(b, scheme, lmax, data, spatpreprocfun, qspace, nmax,
                          D0)
        P = compMesoInv(M, lmax, order, includeb0, bayesmodel)
        P = compRegress(P, lmax, order, includeb0, bayesmodel)

        P(np.isnan(P[:])) = 0
        B0edge = [-inf, (B0[2:end] + B0[0:end - 1]) * 0.5, inf]
        est = np.zeros(np.size(P), np.size(ALpHA[1], 2))

        for k in range(len(B0)):
            idx = np.where((S0 > B0edge[k] and S0 <= B0edge[k+1]))
            Q = P[idx, :]
            est[idx, :] = Q * ALPHA[k]
        est[:, 3:5] = est[:, 3:5] + vf_linsub
        est = np.real(est)
        est = np.concatenate((2, est, SNR))

    def baydiff(data, varargin):
        noise = 'estglobal'
        sigma_snrestimation = 1.75
        sigma_smooth = 0

        for k in range(len(varargin)):
            eval(varargin[k])

        model = TrainModel(data, varargin)

        ten = data.tensor
        b = np.squeeze(ten[0, 0, :] + ten[1, 1, :] + ten[2, 2, :])
        sz = np.size(data.dwi)
        if sigma_smooth == 0:
            preproc = lambda x: x
        else:
            gau = makeGau(sigma_smooth, sz)
            sm2 = lambda x: np.real(sc.ifft(sc.fft(x) * gau))
            col = lambda x: x[:]
            preproc = lambda x: sm2[np.reshape(x, sz[1:3])]  # doubt
            
        b0idx = round(b / 100) == 0
        if sum(b0idx) < 6 and !noise.isnumeric():
            print("not enough b=0 images ofr noise estimation (10)")
            noise = 10
        gau = makeGau(sigma_snrestimation, sz)
        sm = lambda x: np.real(sc.ifft(sc.fft(x) * gau))
        b0 = np.mean(data.dwi[:, :, :, b0idx], 4)
        if isinstance(noise, str):
            if noise == 'estlocal':
                err = np.std(data.dwi[:, :, :, b0idx], [], 4)
                SNR = b0 / (eps + err)
            elif noise == 'estglobal':
                _, idx = np.sort(mb0[:])
                idx = idx[round(len(idx) * 0.6):end]
                err = np.std(data.dwi[:, :, :, b0idx], [], 4)
                err = np.mean(err(idx))
                SNR = b0 / (eps + err)
        else:
            if len(np.size(noise)) == 3:
                SNR = noise
            else:
                SNR = b0 / noise
        SNR[SNR > 100] = 100
        SNR = sm[SNR]
        
        S = np.reshape(data.dwi, [np.multiply(sz[0:2], sz[3])])
        