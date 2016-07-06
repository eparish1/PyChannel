from pylab import *
import numpy
import scipy
import scipy.fftpack
import time
from fftclass import *



#### Testing for using the FFT that is halved in the z direction due to conjugate symmetry

def getA1Mat(u):
  N1,N2,N3 = np.shape(u)
  A1 = np.zeros((N2,N2))
  for n in range(0,N2-1):
    for p in range(n+1,N2,2):
     A1[n,p] = 2.*p
  A1[0,:] = A1[0,:] / 2.
  return A1

def getA2Mat(u):
  N1,N2,N3 = np.shape(u)
  A2 = np.zeros((N2,N2))
  for n in range(0,N2-2):
    for p in range(n+2,N2,2):
      A2[n,p] = p*(p**2-n**2) 
  A2[0,:] = A2[0,:] /2.
  return A2


def diff_y(fhat):
  N1,N2,N3 = np.shape(fhat) 
  fhat1 = np.zeros((N1,N2,N3),dtype='complex')
  for n in range(0,N2-1):
    for p in range(n+1,N2,2): 
      fhat1[:,n,:] += fhat[:,p,:]*2.*p
  fhat1[:,0,:] = fhat1[:,0,:]/2.
  return fhat1

def diff_y2(uhat):
  N1,N2,N3 = np.shape(uhat)
  uhat2 = np.zeros((N1,N2,N3),dtype='complex')
  for n in range(0,N2-2):
    for p in range(n+2,N2,2):
      uhat2[:,n,:] += uhat[:,p,:]* p*(p**2 - n**2)
  uhat2[:,0,:] = uhat2[:,0,:]/2
  return uhat2



N1 = 2**6
N2 = 65
N3 = 2**6

myFFT = FFTclass(N1,N2,N3,1)
L1 = 2.*np.pi
L3 = 2.*np.pi

dx = L1/float( N1 )
dz = L1/float( N1 )

x = np.linspace(0,L1-dx,N1)
z = np.linspace(0,L3-dz,N3)
y = np.cos( np.pi*np.linspace(0,N2-1,N2) /(N2-1) )
y,x,z = np.meshgrid(y,x,z)

k1 = np.fft.fftshift( np.linspace(-N1/2,N1/2-1,N1) )
k2 = np.linspace(0,N2-1,N2)  #dummy 
k3 = np.linspace(0,N3/2,N3/2+1)
k2,k1,k3 = np.meshgrid(k2,k1,k3)


u = zeros((N1,N2,N3))
u[:,:,:] = L1/2.*sin(pi*y)*cos(4*pi*x/L1)*sin(2*pi*z/L3) 

u_y = pi*cos(pi*y)*cos(4*pi*x/L1)*L1/2.*sin(2*pi*z/L3)
u_yy = -pi**2*sin(pi*y)*cos(4*pi*x/L1)*L1/2.*sin(2*pi*z/L3)

u_x = -2.*pi*sin(pi*y)*sin(4.*pi*x/L1)*sin(2*pi*z/L3)
u_xx= -8*pi**2*sin(pi*y)*cos(4*pi*x/L1) / L1 *sin(2*pi*z/L3)

u_z = pi*sin(pi*y)*cos(4.*pi*x/L1)*L1*cos(2*pi*z/L3)/L3
u_zz= -2*pi**2*sin(pi*y)*cos(4*pi*x/L1)*L1*sin(2*pi*z/L3)/L3**2


print('FFT CHECK ' + str(norm(myFFT.myifft3D(myFFT.myfft3D(u) )  - u) ))

##========= CHECK FFTS ================================
t1 = time.time()
uhat = myFFT.myfft3D(u)
print(norm(1j*k1*uhat))
print('FFT Time = ' + str(time.time() - t1) )
#=======================================================

##========= Check Derivs ================================
uhat_y = diff_y(uhat)
uhat_yy = diff_y2(uhat)
uhat_x = 1j*k1*uhat
uhat_xx = -k1**2*uhat
uhat_z = 1j*k3*uhat
uhat_zz = -k3**2*uhat

u_yn = myFFT.myifft3D(uhat_y*1.)
u_yyn = myFFT.myifft3D(uhat_yy*1.)
u_xn = myFFT.myifft3D(uhat_x*1.)
print(norm(u_xn))
print(norm(u_x))
u_xxn = myFFT.myifft3D(uhat_xx*1.)
u_zn = myFFT.myifft3D(uhat_z*1.)
u_zzn = myFFT.myifft3D(uhat_zz*1.)



print('Uy  Error = ' + str(norm(u_yn - u_y) ))
print('Uyy Error = ' + str(norm(u_yyn - u_yy)) )
print('Ux  Error = ' + str(norm(u_xn - u_x) ))
print('Uxx Error = ' + str(norm(u_xxn - u_xx)) )
print('Uz  Error = ' + str(norm(u_zn - u_z) ))
print('Uzz Error = ' + str(norm(u_zzn - u_zz)) )

##======================================================
