import sympy
#?sympy.symbols
k1,k2=sympy.symbols('k1,k2')
k1 
k2 
k1,k2,phi,theta=sympy.symbols('k1,k2,phi,theta')
d = sympy.integrate(sympy.exp((k1*cos(phi)^2+k2*sin(phi)^2)*sin(theta)^2)/(4*sympy.pi),(phi,0,2*sympy.pi),(theta,0,sympy.pi))
d = sympy.integrate(sympy.exp((k1*sympy.cos(phi)^2+k2*sympy.sin(phi)^2)*sympy.sin(theta)^2)/(4*sympy.pi),(phi,0,2*sympy.pi),(theta,0,sympy.pi))
d = sympy.integrate(sympy.exp((k1*sympy.cos(phi)**2+k2*sympy.sin(phi)^2)*sympy.sin(theta)**2)/(4*sympy.pi),(phi,0,2*sympy.pi),(theta,0,sympy.pi))
d = sympy.integrate(sympy.exp((k1*sympy.cos(phi)**2+k2*sympy.sin(phi)**2)*sympy.sin(theta)**2)/(4*sympy.pi),(phi,0,2*sympy.pi),(theta,0,sympy.pi))
d 
d.eval.f
d.integrate 
d.integrate()
from mpmath import *
mp.dps = 50
print mpf(2)**mpf('0.5')
print 2*pi
mpmath.runtests()
import mpmath
mpmath.runtests()
scipy
from scipy.integrate import quad
from math import pi
d = sympy.integrate(sympy.exp((k1*sympy.cos(phi)**2+k2*sympy.sin(phi)**2)*sympy.sin(theta)**2)/(4*sympy.pi),(phi,0,2*sympy.pi),(theta,0,sympy.pi))
