import sympy
from scipy.integrate import quad, dblquad
from scipy.optimize import fmin_powell
import numpy as np
import scipy as sc

'''
def integrand(t,n,x):
    return np.exp(-x*t) / t**n

def expint(n,x):
    return quad(integrand, 1, np.Inf, args=(n, x))[0]

vec_expint = np.vectorize(expint)

print vec_expint(3,np.arange(1.0,4.0,0.5))

'''
#array([ 0.1097,  0.0567,  0.0301,  0.0163,  0.0089,  0.0049])
'''

print sc.special.expn(3,np.arange(1.0,4.0,0.5))

'''
#array([ 0.1097,  0.0567,  0.0301,  0.0163,  0.0089,  0.0049])
'''

result = quad(lambda x: expint(3, x), 0, np.inf)

print result

'''
#(0.33333333324560266, 2.8548934485373678e-09)
'''

I3 = 1.0/3.0

print I3

#0.333333333333
'''


def bingham_kernel(k1, k2, theta, phi):
    return np.exp(((k1 * np.cos(phi) ** 2 + k2 * np.sin(phi) ** 2) * np.sin(theta) ** 2) / 4 * np.pi)


def d(k1, k2):
    #print (k1,k2)
    return dblquad(lambda theta, phi: bingham_kernel(k1, k2, theta, phi), 0, np.pi, lambda phi: 0, lambda phi: 2 * np.pi)[0]

print d(-6.999, -3.345)


# K1,K2,t1,t2,ph,th=sympy.symbols('K1,K2,t1,t2,ph,th')

N = 100


def F((k1, k2), (t1, t2, N)):
    val = -N * 4 * np.pi - N * np.log(d(k1, k2)) + k1 * t1 + k2 * t2
    print (-val, k1, k2)
    return -val


min = fmin_powell(F, (-1, -1), ((-3.345, -6.999, 1000),))

print min

#d = sympy.integrate(sympy.exp((k1*sympy.cos(phi)**2+k2*sympy.sin(phi)**2)*sympy.sin(theta)**2)/(4*sympy.pi),(phi,0,2*sympy.pi),(theta,0,sympy.pi))


'''
def I(n):
    return dblquad(lambda t, x: np.exp(-x*t)/t**n, 0, np.Inf, lambda x: 1, lambda x: np.Inf)

print I(4)

#(0.25000000000435768, 1.0518245707751597e-09)

print I(3)

#(0.33333333325010883, 2.8604069919261191e-09)

print I(2)

#(0.49999999999857514, 1.8855523253868967e-09)

k1,k2,phi,theta=sympy.symbols('k1,k2,phi,theta')

d = sympy.integrate(sympy.exp((k1*sympy.cos(phi)**2+k2*sympy.sin(phi)**2)*sympy.sin(theta)**2)/(4*sympy.pi),(phi,0,2*sympy.pi),(theta,0,sympy.pi))

from scipy.integrate import quad
from math import pi
d = sympy.integrate(sympy.exp((k1*sympy.cos(phi)**2+k2*sympy.sin(phi)**2)*sympy.sin(theta)**2)/(4*sympy.pi),(phi,0,2*sympy.pi),(theta,0,sympy.pi))
'''

'''
Table C.3: 	Maximum likelihood estimators of k1,k2 in the Bingham
distribution for given eigenvalues w1,w2. Data from Mardia and Zemroch
(1977). Upper (lower) number is k1(k2)

w1	0.02	0.04	0.06	0.08	0.10	0.12	0.14	0.16	0.18	0.20	0.22	0.24	0.26	0.28	0.30	0.32
																
w2
0.02	-25.55
	-25.55
0.04	-25.56	-13.11
	-13.09	-13.11
0.06	-25.58	-13.14	-9.043
	-8.996	-9.019	-9.043
0.08	-25.6	-13.16	-9.065	-7.035
	-6.977	-6.999	-7.020	-7.035
0.10	-25.62	-13.18	-9.080	-7.042	-5.797
	-5.760	-5.777	-5.791	-5.798	-5.797
0.12	-25.63	-13.19	-9.087	-7.041	-5.789	-4.917
	-4.923	-4.934	-4.941	-4.941	-4.933	-4.917
0.14	-25.64	-13.20	-9.087	-7.033	-5.773	-4.896	-4.231
	-4.295	-4.301	-4.301	-4.294	-4.279	-4.258	-4.231
0.16	-25.65	-13.20	-9.081	-7.019	-5.752	-4.868	-4.198	-3.659
	-3.796	-3.796	-3.790	-3.777	-3.756	-3.729	-3.697	-3.659
0.18	-25.65	-13.19	-9.068	-6.999	-5.726	-4.836	-4.160	-3.616	-3.160
	-3.381	-3.375	-3.363	-3.345	-3.319	-3.287	-3.249	-3.207	-3.160
0.20	-25.64	-13.18	-9.05	-6.974	-5.694	-4.799	-4.118	-3.570	-3.109	-2.709
	-3.025	-3.014	-2.997	-2.973	-2.942	-2.905	-2.863	-2.816	-2.765	-2.709
0.22	-25.63	-13.17	-9.027	-6.944	-5.658	-4.757	-4.071	-3.518	-3.053	-2.649	-2.289
	-2.712	-2.695	-2.673	-2.644	-2.609	-2.568	-2.521	-2.470	-2.414	-2.354	-2.289
0.24	-25.61	-23.14	-8.999	-6.910	-5.618	-4.711	-4.021	-3.463	-2.993	-2.584	-2.220	-1.888
	-2.431	-2.410	-2.382	-2.349	-2.309	-2.263	-2.212	-2.157	-2.097	-2.032	-1.963	-1.888
0.26	-25.59	-13.12	-8.966	-6.870	-5.573	-4.661	-3.965	-3.403	-2.928	-2.515	-2.146	-1.809	-1.497
	-2.175	-2.149	-2.117	-2.078	-2.034	-1.984	-1.929	-1.869	-1.805	-1.735	-1.661	-1.582	-1.497
0.28	-25.57	-13.09	-8.928	-6.827	-5.523	-4.606	-3.906	-3.338	-2.859	-2.441	-2.066	-1.724	-1.406	-1.106
	-1.939	-1.908	-1.871	-1.828	-1.779	-1.725	-1.665	-1.601	-1.532	-1.458	-1.378	-1.294	-1.203	-1.106
0.30	-25.54	-13.05	-8.886	-6.778	-5.469	-4.547	-3.842	-3.269	-2.785	-2.361	-1.981	-1.634	-1.309	-1.002	-0.708
	-1.718	-1.682	-1.641	-1.596	-1.540	-1.481	-1.417	-1.348	-1.274	-1.195	-1.110	-1.020	-0.923	-0.819	-0.708
0.32	-25.50	-13.01	-8.839	-6.725	-5.411	-4.484	-3.773	-3.195	-2.706	-2.277	-1.891	-1.537	-1.206	-0.891	-0.588	-0.292
	-1.510	-1.470	-1.423	-1.371	-1.313	-1.250	-1.181	-1.108	-1.028	-0.944	-0.853	-0.756	-0.653	-0.541	-0.421	-0.292
0.34	-25.46	-12.96	-8.788	-6.668	-5.348	-4.415	-3.699	-3.116	-2.621	-2.186	-1.794	-1.433	-1.094	-0.771	-0.459	-0.152
	-1.312	-1.267	-1.216	-1.159	-1.096	-1.028	-0.955	-0.876	-0.791	-0.701	-0.604	-0.500	-0.389	-0.269	-0.140	 0.000
0.36	-25.42	-12.91	-8.731	-6.606	-5.280	-4.342	-3.620	-3.032	-2.531	-2.089	-1.690	-1.322	-0.974	-0.642
	-1.123	-1.073	-1.017	-9.555	-0.887	-0.814	-0.736	-0.651	-0.561	-0.464	-0.360	-0.249	-0.129	 0.000
0.38	-25.37	-12.86	-8.670	-6.539	-5.207	-4.263	-3.536	-2.941	-2.434	-1.986	-1.579	-1.202
	-0.940	-0.885	-0.824	-0.757	-0.684	-0.606	-0.522	-0.432	-0.335	-0.231	-0.120	 0.000
0.40	-25.31	-12.80	-8.604	-6.466	-5.126	-4.179	-3.446	-2.845	-2.330	-1.874
	-0.762	-0.702	-0.636	-0.564	-0.486	-0.402	-0.312	-0.215	-0.111	-0.000
0.42	-25.5	-12.73	-8.532	-6.388	-5.045	-4.089	-3.349	-2.741
	-0.589	-0.523	-0.452	-0.374	-0.290	-0.200	-0.104	 0.000
0.44	-25.19	-12.66	-8.454	-6.305	-4.955	-3.992
	-0.418	-0.347	-0.270	-0.186	-0.097	 0.000
0.46	-25.12	-12.58	-8.371	-6.215
	-0.250	-0.173	-0.090	 0.000

Taken from http://magician.ucsd.edu/Essentials/WebBookse115.html#x136-237000C.2a
        
'''
