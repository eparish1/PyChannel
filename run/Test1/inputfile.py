import numpy as np
import scipy.fftpack
import sys

sys.path.append("../../src3D_GL")
N1 = 2**6
N2 = 65
N3 = 2**6
nthreads = 1
kc = N1/2
L1 = 2.*np.pi
L3 = 2.*np.pi
dx = L1/float( N1 )
dz = L3/float( N3 )
x = np.linspace(0,L1-dx,N1)
y = np.cos( np.pi*np.linspace(0,N2-1,N2) /(N2-1) ) 
z = np.linspace(0,L3-dz,N3)
y,x,z = np.meshgrid(y,x,z)

Re_tau = 180.
nu = 1./2745.
pbar_x = -Re_tau**2*nu**2
u_exact = pbar_x/nu*(y**2/2. - 0.5)

eps = np.mean(u_exact)*0.1
u = np.zeros((N1,N2,N3))
u[:,:,:] = u_exact*0.5 +  eps*L1/2.*np.sin(np.pi*y)*np.cos(4*np.pi*x/L1)*np.sin(2.*np.pi*z/L3)

v = np.zeros((N1,N2,N3))
v[:,:,:] = -eps*(1 + np.cos(np.pi*y))*np.sin(4*np.pi*x/L1)*np.sin(2.*np.pi*z/L3)


w = np.zeros((N1,N2,N3))
w[:,:,:] = -eps*L3/2.*np.sin(4.*np.pi*x/L1)*np.sin(np.pi*y)*np.cos(2*np.pi*z/L3)

t = 0.
et = 50000.
dt = 5.e-4
save_freq = 50
execfile('../../src3D_GL/Channel_3d.py')
