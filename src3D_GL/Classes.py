import numpy as np
import scipy
import pyfftw
from RHSfunctions import *

class variables:
  def __init__(self,grid,u,v,w,t,dt,et,nu,myFFT,Re_tau):
    self.t = t
    self.kc = np.amax(grid.k1)
    self.dt = dt
    self.et = et
    self.nu = nu
    self.Re_tau = Re_tau
    self.pbar_x = -Re_tau**2*nu**2
    self.dP = myFFT.myfft3D(self.pbar_x*np.ones(np.shape(u)))

    self.u = np.zeros((grid.N1,grid.N2,grid.N3))
    self.u[:,:,:] = u[:,:,:]
    del u
    self.v = np.zeros((grid.N1,grid.N2,grid.N3))
    self.v[:,:,:] = v[:,:,:]
    del v
    self.w = np.zeros((grid.N1,grid.N2,grid.N3))
    self.w[:,:,:] = w[:,:,:]
    del w

    self.uhat = np.zeros((grid.N1,grid.N2,grid.N3),dtype='complex')
    self.uhat[:,:,:] = myFFT.myfft3D(self.u)

    self.vhat = np.zeros((grid.N1,grid.N2,grid.N3),dtype='complex')
    self.vhat[:,:,:] = myFFT.myfft3D(self.v)

    self.what = np.zeros((grid.N1,grid.N2,grid.N3),dtype='complex')
    self.what[:,:,:] = myFFT.myfft3D(self.w)

    self.phat = np.zeros((grid.N1,grid.N2,grid.N3),dtype='complex')

    self.RHS_explicit =     np.zeros((grid.N1,3*grid.N2,grid.N3),dtype='complex')
    self.RHS_explicit_old = np.zeros((grid.N1,3*grid.N2,grid.N3),dtype='complex')
    self.RHS_implicit =     np.zeros((grid.N1,3*grid.N2,grid.N3),dtype='complex')

    self.u_exact = self.pbar_x/self.nu*(grid.y**2/2. - 0.5)

class gridclass:
  def __init__(self,N1,N2,N3,x,y,z,kc):
    self.N1 = N1
    self.N2 = N2
    self.N3 = N3
    self.x = x
    self.y = y
    self.z = z
    self.dx = x[1,0] - x[0,0]
    k1 = np.fft.fftshift( np.linspace(-N1/2,N1/2-1,N1) )
    k2 = np.fft.fftshift( np.linspace(-N2/2,N2/2-1,N2) ) #dummy 
    k3 = np.fft.fftshift( np.linspace(-N3/2,N3/2-1,N3) ) 

    self.k2,self.k1,self.k3 = np.meshgrid(k2,k1,k3)
    self.ksqr = self.k1*self.k1 + self.k3*self.k3 

    self.A1  = getA1Mat(x)
    self.A1p = getA1Mat_Truncate(x)
    self.A2  = getA2Mat(x)

class FFTclass:
  def __init__(self,N1,N2,N3,nthreads):
    self.N1,self.N2,self.N3 = N1,N2,N3
    self.nthreads = nthreads
#    self.scale = np.sqrt( (3./2.)**3*np.sqrt(N1*N2*N3) ) #scaling for FFTS
#    ## Inverse transforms of uhat,vhat,what are of the truncated padded variable. 
#    ## Input is complex truncate,output is real untruncated
    self.invalT =    pyfftw.n_byte_align_empty((int(3./2.*N1),int(N2)), 16, 'complex128')
    self.outvalT=    pyfftw.n_byte_align_empty((int(3./2.*N1),int(N2)), 16, 'complex128')
    self.ifftpad_obj = pyfftw.FFTW(self.invalT,self.outvalT,axes=(0,),\
                     direction='FFTW_BACKWARD',threads=nthreads)
    ## Fourier transforms of padded vars like u*u.
    ## Input is real full, output is imag truncated 
    self.inval =   pyfftw.n_byte_align_empty((int(3./2.*N1),int(N2)) , 16, 'complex128')
    self.outval=   pyfftw.n_byte_align_empty((int(3./2.*N1),int(N2)), 16, 'complex128')
    self.fftpad_obj = pyfftw.FFTW(self.inval,self.outval,axes=(0,),\
                    direction='FFTW_FORWARD', threads=nthreads)

    ## Inverse transforms of uhat,vhat,what are of the truncated padded variable. 
    ## Input is complex truncate,output is real untruncated
    self.invalT =    pyfftw.n_byte_align_empty((int(N1),int(N2)), 16, 'complex128')
    self.outvalT=    pyfftw.n_byte_align_empty((int(N1),int(N2)), 16, 'complex128')
    self.ifft_obj = pyfftw.FFTW(self.invalT,self.outvalT,axes=(0,),\
                     direction='FFTW_BACKWARD',threads=nthreads)
    ## Fourier transforms of padded vars like u*u.
    ## Input is real full, output is imag truncated 
    self.inval =   pyfftw.n_byte_align_empty((int(N1),int(N2) ), 16, 'complex128')
    self.outval=   pyfftw.n_byte_align_empty((int(N1),int(N2) ), 16, 'complex128')
    self.fft_obj = pyfftw.FFTW(self.inval,self.outval,axes=(0,),\
                    direction='FFTW_FORWARD', threads=nthreads)

    def myfft3D(u):
      N1,N2,N3 = np.shape(u)
      N2 = N2 - 1
      u = scipy.fftpack.fft( scipy.fftpack.fft(u[:,:],axis=0) , axis=2 )/( N1 * N3 )
      umod = np.zeros((N1,2*N2,N3),dtype='complex')
      umod[:,0:N2+1,:] = u[:,0:N2+1,:]
      umod[:,N2+1:2*N2,:] = np.fliplr(u)[:,1:-1,:]
      wtilde = scipy.fftpack.ifft(umod,axis=1) ## yes! actually the ifft. again, only god knows why
      uhat = np.zeros((N1,N2+1,N3),dtype='complex')
      uhat[:,0,:] = wtilde[:,0,:]
      uhat[:,1:-1,:] = wtilde[:,1:N2,:]*2.
      uhat[:,-1,:] = wtilde[:,N2,:]
      return uhat
    self.myfft3D = myfft3D
    self.myfft3D_pad = myfft3D


    def myifft3D(uhat):
      N1,N2,N3 = np.shape(uhat)
      N2 = N2 - 1
      # first do the fourier transform
      utmp = scipy.fftpack.ifft( scipy.fftpack.ifft(uhat,axis=0) , axis = 2) * N1 * N3
      umod = np.zeros((N1,2*N2,N3),dtype='complex')
      umod[:,0,:] = utmp[:,0,:]
      umod[:,1:N2+1,:] = utmp[:,1::,:]/2.
      umod[:,N2+1::,:] = np.fliplr(utmp)[:,1:-1,:]/2.
      utmp2 = scipy.fftpack.fft(umod,axis=1) ##yes! actually the FFT! only god knows why
      u = np.real(utmp2[:,0:N2+1,:])
      return u
    self.myifft3D = myifft3D
    self.myifft3D_pad = myifft3D


    def dealias(fhat):
      N1,N2,N3 = np.shape(fhat)
      cutoff = int( (N2+1.)/3. *2. )
      fhat[:,cutoff::,:] = 0.
      return fhat
    self.dealias = dealias
