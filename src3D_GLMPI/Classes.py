import numpy as np
import sys 
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

    self.u = np.zeros((grid.N1,grid.Npy,grid.N3))
    self.u[:,:,:] = u[:,:,:]
    del u
    self.v = np.zeros((grid.N1,grid.Npy,grid.N3))
    self.v[:,:,:] = v[:,:,:]
    del v
    self.w = np.zeros((grid.N1,grid.Npy,grid.N3))
    self.w[:,:,:] = w[:,:,:]
    del w

    self.uhat = np.zeros((grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    self.uhat[:,:,:] = myFFT.myfft3D(self.u)

    self.vhat = np.zeros((grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    self.vhat[:,:,:] = myFFT.myfft3D(self.v)

    self.what = np.zeros((grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    self.what[:,:,:] = myFFT.myfft3D(self.w)

    self.phat = np.zeros((grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')

    self.RHS_explicit =     np.zeros((grid.Npx,3*grid.N2,grid.N3/2+1),dtype='complex')
    self.RHS_explicit_old = np.zeros((grid.Npx,3*grid.N2,grid.N3/2+1),dtype='complex')
    self.RHS_implicit =     np.zeros((grid.Npx,3*grid.N2,grid.N3/2+1),dtype='complex')


class gridclass:
  def __init__(self,N1,N2,N3,x,y,z,kc,num_processes,L1,L3,mpi_rank,comm):
    self.Npx = int(float(N1 / num_processes))
    self.Npy= int(float(N2 / num_processes))
    self.N1 = N1
    self.N2 = N2
    self.N3 = N3
    self.x = x
    self.y = y
    self.z = z
    k1 = np.fft.fftfreq(N1,1./N1)*2.*np.pi/L1
    k2 = np.fft.fftfreq(N2,1./N2)
    k3 = np.fft.rfftfreq(N3,1./N3)*2.*np.pi/L3
    self.k1,k2,self.k3 = np.meshgrid(k1[mpi_rank*self.Npx:(mpi_rank+1)*self.Npx],k2,k3,indexing='ij')
    self.ksqr = self.k1*self.k1 + self.k3*self.k3 
    self.A1  = getA1Mat(N2)
    self.A1p = getA1Mat(N2-1)
    self.A2  = getA2Mat(N2)
    self.xG = allGather_physical(self.x,comm,mpi_rank,self.N1,self.N2,self.N3,num_processes,self.Npy)
    self.yG = allGather_physical(self.y,comm,mpi_rank,self.N1,self.N2,self.N3,num_processes,self.Npy)
    self.zG = allGather_physical(self.z,comm,mpi_rank,self.N1,self.N2,self.N3,num_processes,self.Npy)
   

class FFTclass:
  def __init__(self,N1,N2,N3,nthreads,fft_type,Npx,Npy,num_processes,comm,mpi_rank):
    self.N1,self.N2,self.N3 = N1,N2,N3
    self.nthreads = nthreads

    if (fft_type == 'pyfftw'):
      sys.stdout.write('Using pyfftw fft routines \n')
      sys.stdout.flush()
  #    self.scale = np.sqrt( (3./2.)**3*np.sqrt(N1*N2*N3) ) #scaling for FFTS
  #    ## Inverse transforms of uhat,vhat,what are of the truncated padded variable. 
  #    ## Input is complex truncate,output is real untruncated
      self.invalT =    pyfftw.n_byte_align_empty((int(3./2.*N1),int(N2),int(3./4.*N3+1)), 16, 'complex128')
      self.outvalT=    pyfftw.n_byte_align_empty((int(3./2.*N1),int(N2),int(3./2*N3   )), 16, 'float64')
      self.ifftpad_obj = pyfftw.FFTW(self.invalT,self.outvalT,axes=(0,2,),\
                       direction='FFTW_BACKWARD',threads=nthreads)
      ## Fourier transforms of padded vars like u*u.
      ## Input is real full, output is imag truncated 
      self.inval =   pyfftw.n_byte_align_empty((int(3./2.*N1),int(N2),int(3./2.*N3) ), 16, 'float64')
      self.outval=   pyfftw.n_byte_align_empty((int(3./2.*N1),int(N2),int(3./4*N3+1)), 16, 'complex128')
      self.fftpad_obj = pyfftw.FFTW(self.inval,self.outval,axes=(0,2,),\
                      direction='FFTW_FORWARD', threads=nthreads)
  
      ## Inverse transforms of uhat,vhat,what are of the truncated padded variable. 
      ## Input is complex truncate,output is real untruncated
      self.invalT =    pyfftw.n_byte_align_empty((int(N1),int(N2),int(N3/2)+1), 16, 'complex128')
      self.outvalT=    pyfftw.n_byte_align_empty((int(N1),int(N2),int(N3)), 16, 'float64')
      self.ifft_obj = pyfftw.FFTW(self.invalT,self.outvalT,axes=(0,2,),\
                       direction='FFTW_BACKWARD',threads=nthreads)
      ## Fourier transforms of padded vars like u*u.
      ## Input is real full, output is imag truncated 
      self.inval =   pyfftw.n_byte_align_empty((int(N1),int(N2),int(N3) ), 16, 'float64')
      self.outval=   pyfftw.n_byte_align_empty((int(N1),int(N2),int(N3/2)+1), 16, 'complex128')
      self.fft_obj = pyfftw.FFTW(self.inval,self.outval,axes=(0,2,),\
                      direction='FFTW_FORWARD', threads=nthreads)
  
  
      ## basic fourier transform in y for chebyshev variables
      ## Input is real full, output is imag truncated 
      #self.inval =   pyfftw.n_byte_align_empty((int(N1),2*int(N2-1),int(N3) ), 16, 'complex128')
      #self.outval=   pyfftw.n_byte_align_empty((int(N1),2*int(N2-1),int(N3) ), 16, 'complex128')
      #self.cfft_obj = pyfftw.FFTW(self.inval,self.outval,axes=(1,),\
      #                direction='FFTW_FORWARD', threads=nthreads)
      ### and inverse fourier transform in y for chebyshev variables
      ### Input is real full, output is imag truncated 
      #self.inval =   pyfftw.n_byte_align_empty((int(N1),2*int(N2-1),int(N3/2)+1 ), 16, 'complex128')
      #self.outval=   pyfftw.n_byte_align_empty((int(N1),2*int(N2-1),int(N3/2)+1 ), 16, 'complex128')
      #self.cifft_obj = pyfftw.FFTW(self.inval,self.outval,axes=(1,),\
      #                direction='FFTW_BACKWARD', threads=nthreads)
   #
  #    ## padded fourier transform in y for chebyshev variables
  #    ## Input is real full, output is imag truncated 
  #    self.inval =   pyfftw.n_byte_align_empty((int(3./2.*N1),2*int(N2-1),int(3./2.*N3) ), 16, 'complex128')
  #    self.outval=   pyfftw.n_byte_align_empty((int(3./2.*N1),2*int(N2-1),int(3./2.*N3) ), 16, 'complex128')
  #    self.cfftpad_obj = pyfftw.FFTW(self.inval,self.outval,axes=(1,),\
  #                    direction='FFTW_FORWARD', threads=nthreads)
  #    ## padded fourier transform in y for chebyshev variables
  #    ## Input is real full, output is imag truncated 
  #    self.inval =   pyfftw.n_byte_align_empty((int(3./2.*N1),2*int(N2-1),int(3./4.*N3+1) ), 16, 'complex128')
  #    self.outval=   pyfftw.n_byte_align_empty((int(3./2.*N1),2*int(N2-1),int(3./4.*N3+1) ), 16, 'complex128')
  #    self.cifftpad_obj = pyfftw.FFTW(self.inval,self.outval,axes=(1,),\
  #                    direction='FFTW_BACKWARD', threads=nthreads)
  
  
  
      def myfft3D_pad(u):
        N1,N2,N3 = np.shape(u) #we are bring in a physical space vector of size 3./2*N1,N2,3./2.*N3
        N2 = N2 - 1
        u = self.fftpad_obj(u[:,:,:]*1.)/( N1 * N3 ) ##first do the FFT in x and z. we now have 3./2.*N1,N2,3./4.*N3+1
        umod = np.zeros((N1,2*N2,N3/2+1),dtype='complex')
        umod[:,0:N2+1,:] = u[:,0:N2+1,:]
        umod[:,N2+1:2*N2,:] = np.fliplr(u)[:,1:-1,:]
  #      wtilde = self.cifftpad_obj(umod[:,:,:]*1.) ## yes! actually the ifft. only god knows why
        wtilde = scipy.fftpack.ifft(umod,axis=1) ## yes! actually the ifft. only god knows why
        uhat = np.zeros((N1,N2+1,N3/2+1),dtype='complex')
        uhat[:,0,:] = wtilde[:,0,:]
        uhat[:,1:-1,:] = wtilde[:,1:N2,:]*2.
        uhat[:,-1,:] = wtilde[:,N2,:]
        return uhat
  
      def myifft3D_pad(uhat):
        N1,N2,N3 = np.shape(uhat) ##Bring in a padded vector, so N1,N2,N3 are 3./2*N1, N2, 3./4*N3+1 
        N3 = (N3 - 1)*2
        N2 = N2 - 1
        # first do the invserse fourier transform
        utmp = np.empty((N1,N2+1,N3),dtype='complex') #after doing the ifft, we are of size 3./2.*N1, N2, 3./2.*N3
        utmp[:,:,:] = self.ifftpad_obj(uhat[:,:,:]*1.) * N1 * N3 
        umod = np.zeros((N1,2*N2,N3),dtype='complex') ## now do our double/mirror reflection for chebyshev fft
        umod[:,0,:] = utmp[:,0,:]
        umod[:,1:N2+1,:] = utmp[:,1::,:]/2.
        umod[:,N2+1::,:] = np.fliplr(utmp)[:,1:-1,:]/2.
        utmp2 = np.empty((N1,(N2)*2,N3),dtype='complex')
  #      utmp2[:,:,:] = self.cfftpad_obj(umod[:,:,:]*1.) #again, yes. Actually the fft. We have 3./2.*N1,2*N2,3./2.*N3 going in
        utmp2[:,:,:] = scipy.fftpack.fft(umod,axis=1)
        return np.real(utmp2[:,0:N2+1,:])
  
      def myfft3D(u):
        N1,N2,N3 = np.shape(u)
        N2 = N2 - 1
        u = self.fft_obj(u[:,:,:]*1.)/( N1 * N3 ) ##first do the FFT in x and z
        umod = np.zeros((N1,2*N2,N3/2+1),dtype='complex')
        umod[:,0:N2+1,:] = u[:,0:N2+1,:]
        umod[:,N2+1:2*N2,:] = np.fliplr(u)[:,1:-1,:]
  #      wtilde = self.cifft_obj(umod[:,:,:]*1.) ## yes! actually the ifft. only god knows why
        wtilde = scipy.fftpack.ifft(umod,axis=1) ## yes! actually the ifft. only god knows why
        uhat = np.zeros((N1,N2+1,N3/2+1),dtype='complex')
        uhat[:,0,:] = wtilde[:,0,:]
        uhat[:,1:-1,:] = wtilde[:,1:N2,:]*2.
        uhat[:,-1,:] = wtilde[:,N2,:]
        return uhat
  
      def myifft3D(uhat):
        N1,N2,N3 = np.shape(uhat)
        N3 = (N3 - 1)*2
        N2 = N2 - 1
        # first do the invserse fourier transform
        utmp = np.empty((N1,N2+1,N3),dtype='complex')
        utmp[:,:,:] = self.ifft_obj(uhat[:,:,:]*1.) * N1 * N3
        umod = np.zeros((N1,2*N2,N3),dtype='complex')
        umod[:,0,:] = utmp[:,0,:]
        umod[:,1:N2+1,:] = utmp[:,1::,:]/2.
        umod[:,N2+1::,:] = np.fliplr(utmp)[:,1:-1,:]/2.
        utmp2 = np.empty((N1,(N2)*2,N3),dtype='complex')
  #      utmp2[:,:,:] = self.cfft_obj(umod[:,:,:]*1.) #again, yes. Actually the fft
        utmp2[:,:,:] = scipy.fftpack.fft(umod,axis=1)
        return np.real(utmp2[:,0:N2+1,:])

      self.myfft3D = myfft3D
      self.myfft3D_pad = myfft3D_pad
      self.myifft3D = myifft3D
      self.myifft3D_pad = myifft3D_pad

    if (fft_type == 'scipy'):
      if (mpi_rank == 0): 
        sys.stdout.write('Using scipy fft routines \n')
        sys.stdout.flush()
      self.Uc_hat = np.zeros((Npx,N2,N3/2+1),dtype='complex')
      self.Uc_hatT = np.zeros((N1,Npy,N3/2+1) ,dtype='complex')
      self.U_mpi = np.zeros((num_processes,Npx,Npy,N3/2+1),dtype='complex')
      def myifft3D(uhat):
        uhatmod = np.zeros((Npx,2*(N2-1),N3/2+1),dtype='complex')
        uhatmod[:,0,:] = uhat[:,0,:]
        uhatmod[:,1:N2,:] = uhat[:,1::,:]/2
        uhatmod[:,N2::,:] = np.fliplr(uhat)[:,1:-1,:]/2.
        Uc_hattmp = np.fft.fft(uhatmod,axis=1) ##yes! actually the FFT! only god knows why
        self.Uc_hat[:,:,:] = Uc_hattmp[:,0:N2,:]
        self.U_mpi[:] = np.rollaxis(self.Uc_hat.reshape(Npx, num_processes, Npy, N3/2+1) ,1)
        comm.Alltoall(self.U_mpi,self.Uc_hatT)
        u = np.fft.irfft2(self.Uc_hatT,axes=(0,2) ) * N1 * N3
        return u

      def myfft3D(u):
        self.Uc_hatT[:,:,:] = np.fft.rfft2(u,axes=(0,2) ) / (N1 * N3)
        comm.Alltoall(self.Uc_hatT, self.U_mpi )
        self.Uc_hat[:,:,:] = np.rollaxis(self.U_mpi,1).reshape(self.Uc_hat.shape)
        Uc_hatmod = np.zeros((Npx,2*(N2-1),N3/2+1),dtype='complex')
        Uc_hatmod[:,0:N2,:] = self.Uc_hat[:,0:N2,:]
        Uc_hatmod[:,N2:2*(N2-1),:] = np.fliplr(self.Uc_hat)[:,1:-1,:]
        wtilde = np.fft.ifft(Uc_hatmod,axis=1) ## yes! actually the ifft. again, only god knows why
        uhat = np.zeros((Npx,N2,N3/2+1),dtype='complex')
        uhat[:,0,:] = wtilde[:,0,:]
        uhat[:,1:-1,:] = wtilde[:,1:N2-1,:]*2.
        uhat[:,-1,:] = wtilde[:,N2-1,:]
        return uhat

      self.myfft3D = myfft3D
      self.myifft3D = myifft3D


    def dealias(fhat):
#      N1,N2,N3 = np.shape(fhat)
#      cutoff = int( (N2+1.)/3. *2. )
#      fhat[:,cutoff::,:] = 0.

      N1,N2,N3 = np.shape(fhat)
      cutoff_y = int( (N2+1.)/3. *2. )
      cutoff_z = int( (N3+1.)/3. *2. )
      cutoff_x = int( (N1+1.)/3. *2. )

      fhat[:,cutoff_y::,:] = 0.
      fhat[:,:,cutoff_z::] = 0.
      fhat[cutoff_x:-cutoff_x,:,:] = 0.
      return fhat
    self.dealias = dealias




