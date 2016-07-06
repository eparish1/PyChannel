from pylab import *
import numpy
import scipy
import scipy.fftpack
import time


def getA1Mat(u):
  N1,N2 = np.shape(u)
  A1 = np.zeros((N2,N2))
  for n in range(0,N2-1):
    for p in range(n+1,N2,2):
     A1[n,p] = 2.*p
  A1[0,:] = A1[0,:] / 2.
  return A1

def getA2Mat(u):
  N1,N2 = np.shape(u)
  A2 = np.zeros((N2,N2))
  for n in range(0,N2-2):
    for p in range(n+2,N2,2):
      A2[n,p] = p*(p**2-n**2) 
  A2[0,:] = A2[0,:] /2.
  return A2

def unpad(uhat_pad,arrange):
  N1 = int( np.shape(uhat_pad)[0]*2./3. )
  N2 = int( np.shape(uhat_pad)[1])
  uhat = np.zeros((N1,N2),dtype = 'complex')
  ## Remove padding from the middle of the 3D cube
  uhat[0:N1/2   , : ] = uhat_pad[0:N1/2                 , :  ] #left     lower back (0,0,0)
  uhat[N1/2+1:: , : ] = uhat_pad[int(3./2.*N1)-N1/2+1:: , :  ] #right     lower back (1,0,0)

  return uhat


def pad(uhat,arrange):
  N1,N2 = np.shape(uhat)
  ## Add padding to the middle of the 3D cube
  uhat_pad = np.zeros((int(3./2.*N1), N2 ),dtype = 'complex')
  uhat_pad[0:N1/2                 , :  ] = uhat[0:N1/2   , : ] #left     lower back (0,0,0)
  uhat_pad[int(3./2.*N1)-N1/2+1:: , :  ] = uhat[N1/2+1:: , : ] #r    ight lower back (1,0,0)
  return uhat_pad


def diff_y(fhat):
  N1,N2 = np.shape(fhat) 
  fhat1 = np.zeros((N1,N2),dtype='complex')
  for n in range(0,N2-1):
    for p in range(n+1,N2,2): 
      fhat1[:,n] += fhat[:,p]*2.*p
  fhat1[:,0] = fhat1[:,0]/2.
  return fhat1

def diff_y2(uhat):
  N1,N2 = np.shape(uhat)
  uhat2 = np.zeros((N1,N2),dtype='complex')
  for n in range(0,N2-2):
    for p in range(n+2,N2,2):
      uhat2[:,n] += uhat[:,p]* p*(p**2 - n**2)
  uhat2[:,0] = uhat2[:,0]/2
  return uhat2


def myifft2D_fast(uhat):
  N1,N2 = np.shape(uhat)
  N2 = N2 - 1
  # first do the fourier transform
  utmp = scipy.fftpack.ifft(uhat,axis=0)*N1
  umod = zeros((N1,2*N2),dtype='complex')
  umod[:,0] = utmp[:,0]
  umod[:,1:N2+1] = utmp[:,1::]/2.
  umod[:,N2+1::] = fliplr(utmp)[:,1:-1]/2.
  utmp2 = scipy.fftpack.fft(umod,axis=1)
  u = real(utmp2[:,0:N2+1])
  return u

def myifft2D(fhat):
  N1,N2 = np.shape(fhat)
  def inverseChebTransform(fhat):
    N = np.size(fhat) - 1
    k = np.linspace(0,N,N+1)
    u = np.zeros(N+1,dtype='complex')
    for j in range(0,N+1):
      u[j] = np.sum(fhat*np.cos(np.pi*k*j/N))
    return np.real(u)

  ## First do the fourier transform
  f =  scipy.fftpack.ifft(fhat,axis=0)*N1
  ## Now do the chebyshev transform 
  for i in range(0,N1):
    f[i,:] = inverseChebTransform(f[i,:])
  return f


def myfft2D_rrfast(u):
  N1,N2 = shape(u)
  N2 = N2 - 1
  u = scipy.fftpack.fft(u[:,:],axis=0)/N1
  umod = zeros((N1,2*N2),dtype='complex')
  umod[:,0:N2+1] = u[:,0:N2+1]
  umod[:,N2+1:2*N2] = fliplr(u)[:,1:-1] 
  wtilde = scipy.fftpack.ifft(umod,axis=1)
  uhat = zeros((N1,N2+1),dtype='complex')
  uhat[:,0] = wtilde[:,0]
  uhat[:,1:-1] = wtilde[:,1:N2]*2.
  uhat[:,-1] = wtilde[:,N2]

  return uhat



def myfft2D_rfast(u):
  N1,N2 = shape(u)
  N2 = N2 - 1
  u = scipy.fftpack.fft(u[:,:],axis=0)/N1
  umod = zeros((N1,2*N2),dtype='complex')
  umod[:,0:N2+1] = u[:,0:N2+1]
  umod[:,N2+1:2*N2] = fliplr(u)[:,1:-1] 
  
  w = zeros((N1,N2),dtype='complex')
  w[:,1:N2] = umod[:,2::2] + 1j*(umod[:,3::2] - umod[:,1:-2:2] )
  w[:,0] = umod[:,0] + 1j*(umod[:,1] - umod[:,-1] )
  wtilde = scipy.fftpack.fft(w,axis=1)
  c = ones((N1,N2+1))
  c[:,0] = 2.
  c[:,-1] = 2.

  uhat = zeros((N1,N2+1),dtype='complex')

  uhat[:,0] = 1./N2*sum(1./c*u,axis=1)

  uhat[:,1:N2] =1./N2*( (0.5 + 1./(4.*sin(pi*k2[:,1:-1]/N2) ) )*fliplr(wtilde)[:,0:-1] + \
              (0.5 - 1./(4.*sin(pi*k2[:,1:-1]/N2) ) )*( wtilde[:,1::] ) )

  uhat[:,-1] = sum(1./N2*(-1.)**linspace(0,N2,N2+1)*1./c*u,axis=1)
  return uhat



def myfft2D_fast(u):
  N1,N2 = shape(u)
  N2 = N2 - 1
  u = scipy.fftpack.fft(u[:,:],axis=0)/N1
  umod = zeros((N1,2*N2),dtype='complex')
  umod[:,0:N2+1] = u[:,0:N2+1]
  for j in range(N2+1,2*N2):
    umod[:,j] = umod[:,2*N2-j]
  #umod[0:size(u)] = u[:]
  #umod[size(u)::] = flipud(u)[1:-1]
  w = zeros((N1,N2),dtype='complex')
  for j in range(0,N2):
    w[:,j] = umod[:,2*j] + 1j*(umod[:,2*j+1] - umod[:,2*j-1] )
  wtilde = scipy.fftpack.fft(w,axis=1)
  c = ones((N1,N2+1))
  c[:,0] = 2.
  c[:,-1] = 2.
#  k = linspace(0,N,N+1)
  uhat = zeros((N1,N2+1),dtype='complex')
  uhat[:,0] = 1./N2*sum(1./c*u,axis=1)
  for k in range(1,N2):
    uhat[:,k] = 1./N2*( (0.5 + 1./(4.*sin(pi*k/N2) ) )*( wtilde[:,N2-k] ) + \
              (0.5 - 1./(4.*sin(pi*k/N2) ) )*( wtilde[:,k] ))

  #uhat[:,-1] = sum(1./N2*(-1.)**linspace(0,N2,N2+1)*1./c*u,axis=1)
  for i in range(0,N1):
    uhat[i,-1] = sum(1./N2*(-1.)**linspace(0,N2,N2+1)*1./c[i,:]*u[i,:])
  return uhat


def myfft2D(f):
  def transform2Cheb(f):
    uhat = np.zeros(np.size(f),dtype='complex')
    N = np.size(f) - 1
    j = np.linspace(0,N,N+1)
    c = np.ones(N+1)
    c[0] = 2.
    c[-1] = 2.
    for i in range(0,np.size(f)):
      uhat[i] = 2./(N*c[i])*np.sum( 1./c * f * np.cos( np.pi * j * i / N ) )
    return uhat
  N1,N2 = np.shape(f)
  fhat = np.zeros((N1,N2),dtype='complex')
  ## First do the fourier transform
  fhat[:,:] =  scipy.fftpack.fft(f[:,:],axis=0)/N1
  ## Now do the chebyshev transform 
  for i in range(0,N1):
    fhat[i,:] = transform2Cheb(fhat[i,:])
  return fhat





N1 = 2**7
N2 = 61
L1 = 2.*np.pi

dx = L1/float( N1 )
x = np.linspace(0,L1-dx,N1)
y = np.cos( np.pi*np.linspace(0,N2-1,N2) /(N2-1) )
y,x = np.meshgrid(y,x)

k1 = np.fft.fftshift( np.linspace(-N1/2,N1/2-1,N1) )
k2 = np.linspace(0,N2-1,N2)  #dummy 
k2,k1 = np.meshgrid(k2,k1)

uhat = zeros((N1,N2+1),dtype='complex')

u = zeros((N1,N2))
u[:,:] = L1/2.*np.sin(np.pi*y)*np.cos(4*np.pi*x/L1)

u_y = pi*cos(pi*y)*cos(4*pi*x/L1)*L1/2.
u_yy = -pi**2*sin(pi*y)*cos(4*pi*x/L1)*L1/2.

u_x = -2.*pi*sin(pi*y)*sin(4.*pi*x/L1)
u_xx= -8*pi**2*sin(pi*y)*cos(4*pi*x/L1) / L1



t1 = time.time()
print('FFT CHECK ' + str(norm(myifft2D_fast(myfft2D_rrfast(u) )  - u) ))
##========= CHECK FFTS ================================
uhat = myfft2D(u)
print('Direct Transform Time = ' + str(time.time() - t1) )
t2 = time.time()
uhat2 = myfft2D_fast(u)
print('FFT  Time = ' + str(time.time() - t2) )
t3 = time.time()
uhat3 = myfft2D_rfast(u)
print('RFFT  Time = ' + str(time.time() - t3) )
t4 = time.time()
uhat4 = myfft2D_rrfast(u)
print('RRFFT  Time = ' + str(time.time() - t4) )

print('Difference Between FFT methods:  Method 1 = ' + str(norm(uhat3 - uhat)) + \
                                    ',  Method 2 = ' + str(norm(uhat2 - uhat)) + \
                                    ',  Method 3 = ' + str(norm(uhat4 - uhat)) )
#=======================================================

##========= Check Derivs ================================
uhat_y = diff_y(uhat4)
uhat_yy = diff_y2(uhat4)
uhat_x = 1j*k1*uhat4
uhat_xx = -k1**2*uhat4

u_yn = myifft2D(uhat_y)
u_yyn = myifft2D(uhat_yy)
u_xn = myifft2D(uhat_x)
u_xxn = myifft2D(uhat_xx)
print('Uy  Error = ' + str(norm(u_yn - u_y) ))
print('Uyy Error = ' + str(norm(u_yyn - u_yy)) )
print('Ux  Error = ' + str(norm(u_xn - u_x) ))
print('Uxx Error = ' + str(norm(u_xxn - u_xx)) )
##======================================================
