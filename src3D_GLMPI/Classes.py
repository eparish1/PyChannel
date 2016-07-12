import numpy as np
import sys 
import scipy
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

    self.RHS_explicit =     np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    self.RHS_explicit_old = np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')
    self.RHS_implicit =     np.zeros((3,grid.Npx,grid.N2,grid.N3/2+1),dtype='complex')


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



