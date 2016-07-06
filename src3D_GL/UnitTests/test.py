from pylab import *
import numpy
import scipy
import scipy.fftpack
import time

def getA1Mat(u):
  N = size(u)
  A1 = zeros((N,N))
  for n in range(0,N-1):
    for p in range(n+1,N,2):
      A1[n,p] = 2.*p
  A1[0,:] = A1[0,:] / 2.
  return A1
 
def getA2Mat(u):
  N = size(u)
  A2 = zeros((N,N))
  for n in range(0,N-2):
    for p in range(n+2,N,2):
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


def transform2(u,x):
  uhat = zeros(size(x))
  j = linspace(0,N,N+1)
  c = ones(N+1)
  c[0] = 2.
  c[-1] = 2.
  for i in range(0,size(x)):
    uhat[i] = 2./(N*c[i])*sum( 1./c * u * cos( pi * j * i / N ) )
  return uhat 

def matTransform(u):
  A = zeros((size(u),size(u)))
  for k in range(0,size(u)):
    if (k == 0 or k == N ):
      ck = 2.
    else: 
      ck = 1.
    for j in range(0,size(u)):
      if (j == 0 or j == N ):
        cj = 2.
      else: 
        cj = 1.
      A[k,j] = 2./(N*cj*ck)*cos(pi*j*k/N)
  uhat = dot(A,u)
  return uhat 

def myfft1D_2(f,x,dcttype):
  N1= size(x)
  fhat = scipy.fftpack.dct( f,axis=0,type=dcttype)/(N1 )
  fhat[0] /=2
  return fhat


def myifft1D_2(uhat):
  umod = zeros(2*(size(uhat)-1),dtype='complex')
  umod[0] = uhat[0]
  umod[1:size(uhat)] = uhat[1::]/2.
  umod[size(uhat)::] = flipud(uhat)[1:-1]/2.
  utmp = scipy.fftpack.fft(umod)
  u = real(utmp[0:size(uhat)])
  return u



def myifft1D(uhat):
  k = linspace(0,N,N+1)
  u = zeros(N+1)
  for j in range(0,N+1):
    u[j] = sum(uhat*cos(pi*k*j/N) )
  return u

def diff_y(uhat):
  uhat_1 = zeros(N+1)
  c = ones(N+1)
  c[0] = 2.
  c[-1] = 2.
  p = linspace(0,N,N+1)
  for m in range(0,N):
    uhat_1[m] = 2./c[m]*sum(p[m+1::2]*uhat[m+1::2] )
  return uhat_1

def diff_y2(fhat):
  N1 = size(fhat)
  fhat2 = zeros((N1))
  for n in range(0,N1-2):
    for p in range(n+2,N1,2):
      fhat2[n] += fhat[p]* p*(p**2 - n**2)
  fhat2[0] = fhat2[0]/2
  return fhat2


def myfft1D_2(u):
  umod = zeros(2*size(u) - 2)
  umod[0:size(u)] = u[:]
  umod[size(u)::] = flipud(u)[1:-1]
  wtilde = scipy.fftpack.ifft(umod)
  uhat = zeros(size(u),dtype='complex')
  uhat[0] = wtilde[0]
  uhat[1:-1] = wtilde[1:size(u)-1]*2
  uhat[-1] = wtilde[size(u)-1] 
  return uhat


def myfft1D(u):
  umod = zeros(2*size(u) - 2)
  for j in range(0,N+1):
    umod[j] = u[j]
  for j in range(N+1,2*N):
    umod[j] = umod[2*N-j]
#  umod[0:size(u)] = u[:]
#  umod[size(u)::] = flipud(u)[1:-1]
  w = zeros(N,dtype='complex') 
  for j in range(0,N):
    w[j] = umod[2*j] + 1j*(umod[2*j+1] - umod[2*j-1] )
  wtilde = scipy.fftpack.fft(w)
  c = ones(N+1)
  c[0] = 2.
  c[-1] = 2.
#  k = linspace(0,N,N+1)
  uhat = zeros(size(u),dtype='complex')
  print(shape(wtilde))
  uhat[0] = 1./N*sum(1./c*u)
  for k in range(1,N):
    uhat[k] = 1./N*( (0.5 + 1./(4.*sin(pi*k/N) ) )*( wtilde[N-k] ) + \
              (0.5 - 1./(4.*sin(pi*k/N) ) )*( wtilde[k] ))

  uhat[-1] = (1./N*(-1.)**linspace(0,N,N+1)*1./c*u).sum()

  return real(uhat)


N = 100
x = cos( pi*linspace(0,N,N+1) /N )

u = sin(4*pi*x) + 1

A1 = getA1Mat(u)
A2 = getA2Mat(u)



t1 = time.time()

uhat = myfft1D(u)

print('FFT Time = ' + str(time.time() - t1) )
t2 = time.time()
uhat2 = transform2(u,x)
print('Direct Transform Time = ' + str(time.time() - t2) )
uhat3 = myfft1D_2(u)

u2 = myifft1D_2(uhat3)

print('FFT Error Norm = ' + str(norm(uhat3 - uhat)) )


uhat_1 = diff_y(uhat)
uhat_1m = dot(A1,uhat)
uhat_2 = diff_y2(uhat)
uhat_2m = dot(A2,uhat)

u2 = myifft1D(uhat)
uprime = myifft1D(uhat_1)
uprime2 = myifft1D(uhat_2)
