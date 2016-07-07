import pyfftw
import numpy as np
import scipy
class FFTclass:
  def __init__(self,N1,N2,N3,nthreads):
    self.N1,self.N2,self.N3 = N1,N2,N3
    self.nthreads = nthreads
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
#    self.inval =   pyfftw.n_byte_align_empty((int(N1),2*int(N2-1),int(N3) ), 16, 'complex128')
#    self.outval=   pyfftw.n_byte_align_empty((int(N1),2*int(N2-1),int(N3) ), 16, 'complex128')
#    self.cfft_obj = pyfftw.FFTW(self.inval1,self.outval,axes=(1,),\
#                    direction='FFTW_FORWARD', threads=nthreads)
    ## and inverse fourier transform in y for chebyshev variables
    ## Input is real full, output is imag truncated 
#    self.inval =   pyfftw.n_byte_align_empty((int(N1),2*int(N2-1),int(N3/2)+1 ), 16, 'complex128')
#    self.outval=   pyfftw.n_byte_align_empty((int(N1),2*int(N2-1),int(N3/2)+1 ), 16, 'complex128')
#    self.cifft_obj = pyfftw.FFTW(self.inval,self.outval,axes=(1,),\
#                    direction='FFTW_BACKWARD', threads=nthreads)



    def myfft3D(u):
      N1,N2,N3 = np.shape(u)
      N2 = N2 - 1
      u = self.fft_obj(u[:,:,:]*1.)/( N1 * N3 ) ##first due the FFT in x and z
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
    self.myifft3D = myifft3D
    self.myfft3D_pad = myfft3D
    self.myifft3D_pad = myifft3D
