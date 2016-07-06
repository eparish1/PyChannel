from pylab import *
import numpy
import scipy
import scipy.fftpack
import time


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


def myifft3D_fast(uhat):
  N1,N2,N3 = np.shape(uhat)
  N2 = N2 - 1
  # first do the fourier transform
  utmp = scipy.fftpack.irfft( scipy.fftpack.ifft(uhat,axis=2) , axis = 0) * N1 * N3
  umod = np.zeros((N1,2*N2,N3),dtype='complex')
  umod[:,0,:] = utmp[:,0,:]
  umod[:,1:N2+1,:] = utmp[:,1::,:]/2.
  umod[:,N2+1::,:] = np.fliplr(utmp)[:,1:-1,:]/2.
  utmp2 = scipy.fftpack.fft(umod,axis=1)
  u = np.real(utmp2[:,0:N2+1,:])
  return u


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


def myfft3D_rrfast(u):
  N1,N2,N3 = np.shape(u)
  N2 = N2 - 1
  u = scipy.fftpack.fft( scipy.fftpack.rfft(u[:,:],axis=0) , axis=2 )/( N1 * N3 )
  umod = np.zeros((N1,2*N2,N3),dtype='complex')
  umod[:,0:N2+1,:] = u[:,0:N2+1,:]
  umod[:,N2+1:2*N2,:] = np.fliplr(u)[:,1:-1,:]
  wtilde = scipy.fftpack.ifft(umod,axis=1) ## yes! actually the ifft. only god knows why
  uhat = np.zeros((N1,N2+1,N3),dtype='complex')
  uhat[:,0,:] = wtilde[:,0,:]
  uhat[:,1:-1,:] = wtilde[:,1:N2,:]*2.
  uhat[:,-1,:] = wtilde[:,N2,:]
  return uhat



N1 = 2**4
N2 = 31
N3 = 2**4
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
k3 = np.linspace(0,N3,N3+1)
k2,k1,k3 = np.meshgrid(k2,k1,k3)


u = zeros((N1,N2,N3))
u[:,:,:] = L1/2.*sin(pi*y)*cos(4*pi*x/L1)*sin(2*pi*z/L3) 

u_y = pi*cos(pi*y)*cos(4*pi*x/L1)*L1/2.*sin(2*pi*z/L3)
u_yy = -pi**2*sin(pi*y)*cos(4*pi*x/L1)*L1/2.*sin(2*pi*z/L3)

u_x = -2.*pi*sin(pi*y)*sin(4.*pi*x/L1)*sin(2*pi*z/L3)
u_xx= -8*pi**2*sin(pi*y)*cos(4*pi*x/L1) / L1 *sin(2*pi*z/L3)

u_z = pi*sin(pi*y)*cos(4.*pi*x/L1)*L1*cos(2*pi*z/L3)/L3
u_zz= -2*pi**2*sin(pi*y)*cos(4*pi*x/L1)*L1*sin(2*pi*z/L3)/L3**2


t1 = time.time()
print('FFT CHECK ' + str(norm(myifft3D_fast(myfft3D_rrfast(u) )  - u) ))

##========= CHECK FFTS ================================
uhat = myfft3D(u)
print('Direct Transform Time = ' + str(time.time() - t1) )
t2 = time.time()
uhat2 = myfft3D_fast(u)
print('FFT  Time = ' + str(time.time() - t2) )
t3 = time.time()
uhat3 = myfft3D_rfast(u)
print('RFFT  Time = ' + str(time.time() - t3) )
t4 = time.time()
uhat4 = myfft3D_rrfast(u)
print('RRFFT  Time = ' + str(time.time() - t4) )

print('Difference Between FFT methods:  Method 1 = ' + str(norm(uhat3 - uhat)) + \
                                     '  Method 2 = ' + str(norm(uhat3 - uhat)) + \
                                     '  Method 3 = ' + str(norm(uhat4 - uhat)) )
#=======================================================

##========= Check Derivs ================================
uhat_y = diff_y(uhat4)
uhat_yy = diff_y2(uhat4)
uhat_x = 1j*k1*uhat4
uhat_xx = -k1**2*uhat4
uhat_z = 1j*k3*uhat4
uhat_zz = -k3**2*uhat4

u_yn = myifft3D(uhat_y)
u_yyn = myifft3D(uhat_yy)
u_xn = myifft3D(uhat_x)
u_xxn = myifft3D(uhat_xx)
u_zn = myifft3D(uhat_z)
u_zzn = myifft3D(uhat_zz)



print('Uy  Error = ' + str(norm(u_yn - u_y) ))
print('Uyy Error = ' + str(norm(u_yyn - u_yy)) )
print('Ux  Error = ' + str(norm(u_xn - u_x) ))
print('Uxx Error = ' + str(norm(u_xxn - u_xx)) )
print('Uz  Error = ' + str(norm(u_zn - u_z) ))
print('Uzz Error = ' + str(norm(u_zzn - u_zz)) )

##======================================================
