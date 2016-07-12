import numpy as np
from fftclass import *
from mpi4py import MPI
import time


def allGather_physical(tmp_local):
  data = comm.gather(tmp_local,root = 0)
  if (mpi_rank == 0):
    tmp_global = np.empty((N1,N2,N3),dtype='complex')
    for j in range(0,num_processes):
      tmp_global[:,j*Npy:(j+1)*Npy,:] = data[j][:,:,:]
    return tmp_global

def allGather_spectral(tmp_local):
  data = comm.gather(tmp_local,root = 0)
  if (mpi_rank == 0):
    tmp_global = np.empty((N1,N2,N3),dtype='complex')
    for j in range(0,num_processes):
      tmp_global[j*Npx:(j+1)*Npx,:,:] = data[j][:,:,:]
  return tmp_global

nthreads = 1
fft_type = 'scipy'
comm = MPI.COMM_WORLD
num_processes = comm.Get_size()
mpi_rank = comm.Get_rank()
N1,N2,N3 = 32,32,32
Npx = int(float(N1 / num_processes))
Npy= int(float(N2 / num_processes))
sy = slice(mpi_rank*Npy,(mpi_rank+1)*Npy)



if (mpi_rank == 0):
  t1 = time.time()


L1 = 2.*np.pi
L2 = 2.*np.pi
L3 = 2.*np.pi
myFFT = FFTclass(N1,N2,N3,nthreads,fft_type,Npx,Npy,num_processes,comm)


dx =float( L1/N1 )
dy =float( L2/N2 )
dz =float( L3/N3 )

## Local Mesh ##
x = np.linspace(0,L1-dx,N1)
y = np.cos( np.pi*np.linspace(0,N2-1,N2) /(N2-1) )
#y = np.linspace(0,L2-dy,N2)
z = np.linspace(0,L3-dz,N3)
x,y,z = np.meshgrid(x,y[sy],z,indexing='ij')
k1 = np.fft.fftfreq(N1,1./N1)*2.*np.pi/L1
k2 = np.fft.fftfreq(N2,1./N2)*2.*np.pi/L2
k3 = np.fft.rfftfreq(N3,1./N3)*2.*np.pi/L3
k1,k2,k3 = np.meshgrid(k1[mpi_rank*Npx:(mpi_rank+1)*Npx],k2,k3,indexing='ij')



u = np.zeros(np.shape(x))
u[:,:,:] =  L1/2.*np.sin(np.pi*y)*np.cos(4*np.pi*x/L1)*np.sin(2.*np.pi*z/L3)
uhat = myFFT.myfft3D(u)
print(np.linalg.norm(u - myFFT.myifft3D(myFFT.myfft3D(u) ) ) )

#uhat_x = 1j*k1*uhat
#ifftn_mpi(uhat_x,u)
ux = myFFT.myifft3D(1j*k1*uhat)
if (mpi_rank == 0):
  t2 = time.time()
  print(t2 - t1)

uGlobal = allGather_physical(u) 

#print(np.shape(
### Check all your stuff
if (mpi_rank == 0):
  ## Create global for checking
  xG = np.linspace(0,L1-dx,N1)
  yG = np.cos( np.pi*np.linspace(0,N2-1,N2) /(N2-1) )
  zG = np.linspace(0,L3-dz,N3)
  xG,yG,zG = np.meshgrid(xG,yG,zG,indexing='ij')
  k1G = np.fft.fftfreq(N1,1./N1)*2.*np.pi/L1
  k2G = np.fft.fftfreq(N2,1./N2)*2.*np.pi/L2
  k3G = np.fft.rfftfreq(N3,1./N3)*2.*np.pi/L3
  k1G,k2G,k3G = np.meshgrid(k1G,k2G,k3G,indexing='ij')

  uG=  L1/2.*np.sin(np.pi*yG)*np.cos(4*np.pi*xG/L1)*np.sin(2.*np.pi*zG/L3)
  #uhatG = np.fft.rfftn(uG)
  uxG = -2*np.pi*np.sin(np.pi*yG)*np.sin(4.*np.pi*xG/L1)*np.sin(2.*np.pi*zG/L3)
  #print(np.shape(uGlobal))
  print(np.linalg.norm(uGlobal))

