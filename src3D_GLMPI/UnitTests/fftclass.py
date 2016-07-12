import pyfftw
import sys
import numpy as np
import scipy
class FFTclass:
  def __init__(self,N1,N2,N3,nthreads,fft_type,Npx,Npy,num_processes,comm):
    self.N1,self.N2,self.N3 = N1,N2,N3
    self.nthreads = nthreads
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

